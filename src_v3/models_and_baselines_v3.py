"""
Host Backbones (Original & Linear-Residual Skip), Callable-Objective LightGBM,
Unconfounded 2x2 Factorial DriftLens, Clean 3-Stage Survival-PU, and Within-Host
Counterfactual Suite (N05 V3).

Closes Codex Round 2 & Round 3 Findings (R3-C1, R3-C2, R3-C3, R3-C4, R3-C5, R3-C6, R3-C7, R3-M1, R3-M6):
- Strict prefix-invariant causal calibration & per-commit threshold vectors (`tuned_th_vec`, `decisions_th`)
  with zero window lookahead and zero retrospective threshold averaging (`R3-C1`).
- Immutable per-alarm boosting/optimizer budgets read strictly from `V3_CONFIG` (`lgb_init_rounds=90`,
  `lgb_alarm_rounds=75`, `Teacher_Only_Matched=150` on EVERY alarm without doubling, identical `lr` across
  `Teacher_Only_Exact`, `Teacher_Only_Matched`, and `Survival_PU`, `R3-C2`).
- Competitor ports with documented adaptations & phase-decomposed timing (`ORB_OHT_Native`,
  `BORB_Native`/`BORB_Alarm`, and `ODaSC_OHT_Native`) imported from `stream_competitors_v3.py` (`R3-C3`, `R3b-M4`, `R3b-M6`).
- Disjoint `P` vs marginal `U = X_pool` membership in `make_gbdt_fgrad_nnpu_objective` so supplemented `P`
  rows never contaminate `R_U^-`, using unit pseudo-Hessian functional gradient boosting (`R3-C4`).
- Single shared `compute_survival_pu_posterior` across bisection, reachability, student training targets,
  and residual logging (`R3-C5`).
- Strict `t0_post > t0` causal latency intervention preserving `observe_labels_at_time(..., t0)` bit-for-bit
  even at `kappa = 0.0`, and separating finite-delay `kappa = 0.0` from `post_t0_oracle_visibility=True` (`R3-C6`).
- Unconfounded `DriftMonitorV3` (`percentile=95.0`, `cooldown_windows=1`, and identical calibration pool
  across all 4 cells of `iid`/`block` x `static`/`rolling`) + per-method `executed_alarm_windows` (`R3-C7`).
- Exact `B2_2000` (`2,000` commits), `teacher_effective_pw` for `frozen_anchor_teacher`, first post-onset
  local fork selection, and separate adaptation-only vs end-to-end stream runtime telemetry (`R3-M6`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightgbm as lgb
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v3 import SECONDS_PER_DAY, V3_CONFIG
from metrics_and_stats_v3 import (
    prob_to_logit_safe,
    sigmoid_safe,
)
from stream_competitors_v3 import (
    SongDenStreamConfidenceTracker,
    _pack_timing_breakdown,
    calibrate_causal_stream_probs_strict,
    run_borb_competitor_v3,
    run_linear_competitor_v3,
    run_odasc_oht_native_v3,
    run_orb_oht_native_v3,
)
from survival_estimators_v3 import (
    build_survival_estimator_v3,
    compute_survival_pu_posterior,
    estimate_ht_shrunk_prevalence,
    solve_monotone_pu_bisection,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HostTrainingFailureError(RuntimeError):
    """Raised when an underlying optimizer/GBDT backend throws an unexpected training error (`R3c-2`)."""


@dataclass
class ComputeStats:
    opt_steps: int = 0
    examples_seen: int = 0
    trees_built: int = 0
    requested_rounds: int = 0
    actual_iterations: int = 0
    wall_ms: float = 0.0

    def add(self, other: "ComputeStats") -> None:
        self.opt_steps += other.opt_steps
        self.examples_seen += other.examples_seen
        self.trees_built += other.trees_built
        self.requested_rounds += other.requested_rounds
        self.actual_iterations += other.actual_iterations
        self.wall_ms += other.wall_ms


def count_effective_minibatches_per_epoch(
    n_samples: int,
    batch_size: int = 128,
    min_batch_size: int = 2,
) -> int:
    """
    Single shared batch-counting rule used by BOTH `train_supervised_v3` (`if len(idx) < 2: continue`)
    and `adapt_survival_pu_v3` (`Teacher_Only_Matched` step planner), closing `R3b-M2` (`N mod 128 == 1`)!
    """
    if int(n_samples) < int(min_batch_size):
        return 0
    n_full, rem = divmod(int(n_samples), int(batch_size))
    return n_full + (1 if rem >= int(min_batch_size) else 0)


# ==============================================================================
# 1. Strict Causal Label Visibility & Future-Only Latency Intervention
# ==============================================================================

def observe_labels_at_time(
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    t_now: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal real-time label visibility at timestamp `t_now`.
    Any commit with `fix_ts > t_now` (or `+inf`) has `y_obs = 0`.
    """
    c_ts = np.asarray(commit_ts, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.int32)
    f_ts = np.asarray(fix_ts, dtype=np.float64)

    valid = c_ts <= t_now
    revealed = valid & (y_t == 1) & np.isfinite(f_ts) & (f_ts <= t_now)
    return valid, revealed.astype(np.float32)


def apply_future_only_latency_intervention(
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    t0: float,
    kappa: float,
    post_t0_oracle_visibility: bool = False,
) -> np.ndarray:
    """
    Exact Causal Future-Only Latency Scaling Intervention at stream start `t0` (Codex R3-C6):
      1. Already revealed at or before `t0` (`fix_ts <= t0`): UNCHANGED (`fix_ts_k = fix_ts`).
      2. Strictly post-`t0` minimum reveal timestamp `t0_post = nextafter(t0, +inf)`:
         Guarantees NO unrevealed backlog or simultaneous `t0` commit EVER satisfies `fix_ts_k <= t0`,
         even at `kappa == 0.0`! Thus `observe_labels_at_time(..., t0)` is bit-for-bit invariant!
      3. Ordinary latency scaling (`post_t0_oracle_visibility=False`):
         - Unlinked `+inf` bugs remain `+inf` for ALL `kappa >= 0.0` (never artificially revealed!).
         - Pending finite backlog at `t0` (`commit_ts <= t0` and `fix_ts > t0`):
           `fix_ts_k = max(t0 + kappa * (fix_ts - t0), t0_post)`.
         - Post-start finite bugs (`commit_ts > t0`):
           `fix_ts_k = max(commit_ts + kappa * max(fix_ts - commit_ts, 0), t0_post)`.
      4. Explicit post-`t0` oracle visibility control (`post_t0_oracle_visibility=True`):
         - Separately named intervention for `B1b == B2` verification: reveals all `y_true == 1` not yet
           revealed at `t0` immediately after `t0` at `max(commit_ts, t0_post)`.
    """
    c_ts = np.asarray(commit_ts, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.int32)
    f_ts = np.asarray(fix_ts, dtype=np.float64).copy()
    t0_f = float(t0)
    t0_post = float(np.nextafter(t0_f, np.inf))
    k = float(max(kappa, 0.0))

    already = (y_t == 1) & np.isfinite(f_ts) & (f_ts <= t0_f)

    if post_t0_oracle_visibility:
        out = np.full_like(f_ts, np.inf)
        out[already] = f_ts[already]
        rem_pos = (y_t == 1) & (~already)
        out[rem_pos] = np.maximum(c_ts[rem_pos], t0_post)
        return out

    out = f_ts.copy()
    fin_unrev = np.isfinite(f_ts) & (f_ts > t0_f)
    pending = fin_unrev & (c_ts <= t0_f)
    out[pending] = np.maximum(t0_f + k * (f_ts[pending] - t0_f), t0_post)

    future_commits = fin_unrev & (c_ts > t0_f)
    out[future_commits] = np.maximum(
        c_ts[future_commits] + k * np.maximum(f_ts[future_commits] - c_ts[future_commits], 0.0),
        t0_post,
    )
    return out


# ==============================================================================
# 2. Unconfounded 2x2 Factorial Unsupervised Drift Monitor (Codex R3-C7)
# ==============================================================================

class DriftMonitorV3:
    """
    Unconfounded Unsupervised Drift Monitor (`R3-C7`):
      - `mode`: "frechet" (DriftLens), "mmd", "ks", "periodic"
      - `calib_sampling`: "block" (circular block bootstrap) vs "iid" (uniform i.i.d. bootstrap)
        applied to the EXACT SAME calibration pool `cal_pool`!
      - `reference_refresh`: "rolling" (post-alarm reference refresh) vs "static" (fixed reference)
      - `percentile` (`default = V3_CONFIG.drift_block_pct = 95.0`) and `cooldown_windows`
        (`default = V3_CONFIG.drift_cooldown_windows = 1`) are FIXED across ALL 4 cells of the
        2x2 (`calib_sampling` x `reference_refresh`) unless explicitly overridden!
    """

    def __init__(
        self,
        mode: str = "frechet",
        calib_sampling: str = "block",
        reference_refresh: str = "rolling",
        n_components: int = V3_CONFIG.drift_pca_dim,
        percentile: Optional[float] = None,
        periodic_interval: int = 3,
        cooldown_windows: Optional[int] = None,
        seed: int = 42,
    ):
        self.mode = mode
        self.calib_sampling = calib_sampling
        self.reference_refresh = reference_refresh
        self.n_components = int(n_components)
        self.percentile = float(percentile) if percentile is not None else float(V3_CONFIG.drift_block_pct)
        self.periodic_interval = int(periodic_interval)
        self.cooldown_windows = int(cooldown_windows) if cooldown_windows is not None else int(V3_CONFIG.drift_cooldown_windows)
        self.seed = int(seed)

        self.pca_: Optional[PCA] = None
        self.mu0_: Optional[np.ndarray] = None
        self.cov0_: Optional[np.ndarray] = None
        self.ref_sub_: Optional[np.ndarray] = None
        self.rbf_gamma_: float = 0.05
        self.threshold_: float = np.inf
        self.step_counter_: int = 0
        self.cooldown_left_: int = 0

    @staticmethod
    def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
        eps = 1e-5 * np.eye(len(mu1), dtype=np.float64)
        c1 = 0.5 * (cov1 + cov1.T) + eps
        c2 = 0.5 * (cov2 + cov2.T) + eps
        diff = mu1 - mu2
        evals1, evecs1 = np.linalg.eigh(c1)
        sqrt_c1 = (evecs1 * np.sqrt(np.maximum(evals1, 1e-10))) @ evecs1.T
        prod_sym = 0.5 * (sqrt_c1 @ c2 @ sqrt_c1 + (sqrt_c1 @ c2 @ sqrt_c1).T)
        tr_covmean = float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(prod_sym), 0.0))))
        val = float(diff @ diff + np.trace(c1) + np.trace(c2) - 2.0 * tr_covmean)
        return float(np.sqrt(max(val, 0.0)))

    def _mmd_rbf(self, X: np.ndarray, Y: np.ndarray) -> float:
        d_xx = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        d_yy = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        d_xy = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return float(max(np.mean(np.exp(-self.rbf_gamma_ * d_xx)) + np.mean(np.exp(-self.rbf_gamma_ * d_yy)) - 2.0 * np.mean(np.exp(-self.rbf_gamma_ * d_xy)), 0.0))

    @staticmethod
    def _ks_max(X: np.ndarray, Y: np.ndarray) -> float:
        k_axes = min(X.shape[1], 12)
        return float(max(scipy.stats.ks_2samp(X[:, j], Y[:, j]).statistic for j in range(k_axes)))

    def _raw_distance(self, sub_z: np.ndarray) -> float:
        if self.mode == "frechet":
            mu_w = np.mean(sub_z, axis=0)
            cov_w = np.cov(sub_z, rowvar=False)
            if cov_w.ndim == 0:
                cov_w = np.array([[float(cov_w)]])
            return self.frechet_distance(self.mu0_, self.cov0_, mu_w, cov_w)
        if self.mode == "mmd":
            step = max(len(sub_z) // 200, 1)
            return self._mmd_rbf(self.ref_sub_, sub_z[::step])
        if self.mode == "ks":
            return self._ks_max(self.ref_sub_, sub_z)
        return 0.0

    @staticmethod
    def _sample_block_bootstrap(pool_z: np.ndarray, w: int, block_len: int, rng: np.random.RandomState) -> np.ndarray:
        n_pool = len(pool_z)
        n_blocks = int(math.ceil(w / float(block_len)))
        starts = rng.randint(0, n_pool, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_len) % n_pool for st in starts])[:w]
        return pool_z[idx]

    def _project(self, emb: np.ndarray) -> np.ndarray:
        if self.pca_ is not None:
            return self.pca_.transform(emb)
        return np.asarray(emb, dtype=np.float64)

    def _set_reference_stats(self, ref_z: np.ndarray, rng: np.random.RandomState) -> None:
        self.mu0_ = np.mean(ref_z, axis=0)
        self.cov0_ = np.cov(ref_z, rowvar=False)
        if self.cov0_.ndim == 0:
            self.cov0_ = np.array([[float(self.cov0_)]])
        r_idx = rng.choice(len(ref_z), size=min(300, len(ref_z)), replace=False)
        self.ref_sub_ = ref_z[r_idx]
        med_sq = float(np.median(np.sum((self.ref_sub_[:100, None, :] - self.ref_sub_[None, :100, :]) ** 2, axis=-1)))
        self.rbf_gamma_ = 1.0 / max(med_sq, 1e-3)

    def _calibrate_null_distances(self, cal_pool: np.ndarray, w: int, n_b: int, rng: np.random.RandomState) -> List[float]:
        """
        Identical calibration-pool evaluation rule for `iid` and `block` (`R3-C7`):
        Only the bootstrap draw (`rng.choice` vs `_sample_block_bootstrap`) differs!
        """
        dists: List[float] = []
        if len(cal_pool) >= w:
            step = max(w // 3, 30)
            for s in range(0, len(cal_pool) - w + 1, step):
                dists.append(self._raw_distance(cal_pool[s : s + w]))
        block_len = max(min(w // 6, 80), 25)
        for _ in range(n_b):
            if self.calib_sampling == "iid":
                idx = rng.choice(len(cal_pool), size=w, replace=True)
                dists.append(self._raw_distance(cal_pool[idx]))
            else:
                sub_b = self._sample_block_bootstrap(cal_pool, w=w, block_len=block_len, rng=rng)
                dists.append(self._raw_distance(sub_b))
        return dists

    def fit_reference(
        self,
        ref_embeddings: np.ndarray,
        val_embeddings: np.ndarray,
        window_size: int = V3_CONFIG.window_size_default,
        n_boot: int = V3_CONFIG.drift_bootstrap_reps,
    ) -> "DriftMonitorV3":
        self.step_counter_ = 0
        self.cooldown_left_ = 0
        if self.mode == "periodic":
            self.threshold_ = 0.5
            return self

        d_eff = min(self.n_components, ref_embeddings.shape[1], max(ref_embeddings.shape[0] - 2, 2))
        if d_eff < ref_embeddings.shape[1]:
            self.pca_ = PCA(n_components=d_eff, random_state=self.seed)
            ref_z = self.pca_.fit_transform(ref_embeddings)
            val_z = self.pca_.transform(val_embeddings)
        else:
            self.pca_ = None
            ref_z = np.asarray(ref_embeddings, dtype=np.float64)
            val_z = np.asarray(val_embeddings, dtype=np.float64)

        rng = np.random.RandomState(self.seed)
        self._set_reference_stats(ref_z, rng)

        cal_pool = val_z if len(val_z) >= window_size else np.vstack([ref_z[-min(len(ref_z), window_size) :], val_z])
        w = min(window_size, len(cal_pool))
        n_b = n_boot if self.mode == "frechet" else min(n_boot, 120)
        dists = self._calibrate_null_distances(cal_pool, w=w, n_b=n_b, rng=rng)
        self.threshold_ = float(np.percentile(dists, self.percentile))
        return self

    def update_reference(self, new_ref_embeddings: np.ndarray, window_size: int = V3_CONFIG.window_size_default) -> None:
        if self.mode == "periodic" or self.reference_refresh != "rolling":
            return
        if len(new_ref_embeddings) < max(window_size, 150):
            return
        rng = np.random.RandomState(self.seed + self.step_counter_)
        new_z = self._project(new_ref_embeddings)
        self._set_reference_stats(new_z, rng)

        w = min(window_size, len(new_z))
        dists = self._calibrate_null_distances(new_z, w=w, n_b=120, rng=rng)
        new_th = float(np.percentile(dists, self.percentile))
        self.threshold_ = 0.70 * new_th + 0.30 * self.threshold_

    def score_window(self, window_embeddings: np.ndarray) -> Tuple[float, bool]:
        self.step_counter_ += 1
        if self.mode == "periodic":
            fire = (self.step_counter_ % self.periodic_interval) == 0
            return (1.0 if fire else 0.0), fire

        z = self._project(window_embeddings)
        dist = self._raw_distance(z)
        if self.cooldown_left_ > 0:
            self.cooldown_left_ -= 1
            return dist, False
        fired = bool(dist > self.threshold_)
        if fired and self.cooldown_windows > 0:
            self.cooldown_left_ = self.cooldown_windows
        return dist, fired


# ==============================================================================
# 3. Host Model Families (Dual-Embedding, Multi-Token Attention, Linear-Residual Skip,
#    and Callable-Objective LightGBM with Immutable Default Budgets)
# ==============================================================================

def fit_causal_tabular_lr_weights(
    X_tab: np.ndarray,
    y_obs: np.ndarray,
    pos_weight: float,
    c_reg: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    X_arr = np.asarray(X_tab, dtype=np.float64)
    y_bin = (np.asarray(y_obs, dtype=np.float64) > 0.5).astype(np.int32)
    d = X_arr.shape[1]
    if int(np.sum(y_bin == 1)) < 3 or int(np.sum(y_bin == 0)) < 3:
        p0 = float(np.clip((np.sum(y_bin) + 1.0) / (len(y_bin) + 2.0), 0.01, 0.99))
        return np.zeros(d, dtype=np.float32), float(np.log(p0 / (1.0 - p0)))

    sw = np.where(y_bin == 1, float(pos_weight), 1.0)
    lr = LogisticRegression(C=c_reg, solver="lbfgs", max_iter=300, random_state=seed)
    lr.fit(X_arr, y_bin, sample_weight=sw)
    return lr.coef_[0].astype(np.float32), float(lr.intercept_[0])


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.drop(F.softmax(attn, dim=-1))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, L, H)
        return self.ln(x + self.out_proj(ctx))


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, Lq, H = x_q.shape
        Lkv = x_kv.shape[1]
        q = self.q_proj(x_q).view(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).view(B, Lkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).view(B, Lkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.drop(F.softmax(attn, dim=-1))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, Lq, H)
        return self.ln(x_q + self.out_proj(ctx))


class LinearSkipMixin:
    def init_linear_skip_from_data(self, X_packed: np.ndarray, y_obs: np.ndarray, pos_weight: float, seed: int = 42) -> None:
        if getattr(self, "skip_linear", None) is None:
            return
        X_tab = self.extract_tab_np(X_packed)
        w, b = fit_causal_tabular_lr_weights(X_tab, y_obs, pos_weight=pos_weight, seed=seed)
        with torch.no_grad():
            dev = self.skip_linear.weight.device
            self.skip_linear.weight.copy_(torch.from_numpy(w[None, :]).to(dev))
            self.skip_linear.bias.copy_(torch.tensor([b], dtype=torch.float32, device=dev))
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        if getattr(self, "skip_mode", "none") == "frozen_lr":
            for p in self.skip_linear.parameters():
                p.requires_grad = False


class TabularNeuralHostV3(nn.Module, LinearSkipMixin):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.20,
        skip_mode: str = "none",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.skip_mode = skip_mode
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(embed_dim, 1)
        self.skip_linear = nn.Linear(in_dim, 1) if skip_mode != "none" else None

    def extract_tab_np(self, X_packed: np.ndarray) -> np.ndarray:
        return np.asarray(X_packed, dtype=np.float32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = h + self.res_block(h)
        z = self.embed_proj(h)
        r_logit = self.residual_head(z).squeeze(-1)
        if self.skip_linear is not None:
            return self.skip_linear(x).squeeze(-1) + r_logit, z
        return r_logit, z


class DeepJitDualEmbCNN(nn.Module, LinearSkipMixin):
    def __init__(
        self,
        vocab_msg: int,
        vocab_code: int,
        msg_len: int,
        code_len: int,
        tab_dim: int,
        tok_dim: int = 64,
        num_filters: int = 64,
        embed_dim: int = 64,
        dropout: float = 0.25,
        skip_mode: str = "none",
    ):
        super().__init__()
        self.msg_len = msg_len
        self.code_len = code_len
        self.tab_dim = tab_dim
        self.skip_mode = skip_mode

        self.msg_emb = nn.Embedding(vocab_msg, tok_dim, padding_idx=0)
        self.code_emb = nn.Embedding(vocab_code, tok_dim, padding_idx=0)
        self.msg_convs = nn.ModuleList([nn.Conv1d(tok_dim, num_filters, k, padding=k // 2) for k in (1, 2, 3)])
        self.code_convs = nn.ModuleList([nn.Conv1d(tok_dim, num_filters, k, padding=k // 2) for k in (1, 2, 3)])
        self.tab_mlp = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, 48),
            nn.GELU(),
        )
        fused_dim = 6 * num_filters + 48
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(embed_dim, 1)
        self.skip_linear = nn.Linear(tab_dim, 1) if skip_mode != "none" else None

    def extract_tab_np(self, X_packed: np.ndarray) -> np.ndarray:
        return np.asarray(X_packed[:, self.msg_len + self.code_len :], dtype=np.float32)

    def forward(self, x_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        msg_ids = x_packed[:, : self.msg_len].long().clamp(min=0)
        code_ids = x_packed[:, self.msg_len : self.msg_len + self.code_len].long().clamp(min=0)
        tab_x = x_packed[:, self.msg_len + self.code_len :]

        m_e = self.msg_emb(msg_ids).transpose(1, 2)
        c_e = self.code_emb(code_ids).transpose(1, 2)

        m_h = torch.cat([torch.max(F.relu(conv(m_e)), dim=-1)[0] for conv in self.msg_convs], dim=1)
        c_h = torch.cat([torch.max(F.relu(conv(c_e)), dim=-1)[0] for conv in self.code_convs], dim=1)
        t_h = self.tab_mlp(tab_x)

        z = self.fusion(torch.cat([m_h, c_h, t_h], dim=1))
        r_logit = self.residual_head(z).squeeze(-1)
        if self.skip_linear is not None:
            return self.skip_linear(tab_x).squeeze(-1) + r_logit, z
        return r_logit, z


class AttentionDualEmbCNN(nn.Module, LinearSkipMixin):
    def __init__(
        self,
        vocab_msg: int,
        vocab_code: int,
        msg_len: int,
        code_len: int,
        tab_dim: int,
        tok_dim: int = 64,
        num_filters: int = 48,
        num_heads: int = 6,
        n_msg_chunks: int = 8,
        n_code_hunks: int = 8,
        embed_dim: int = 64,
        dropout: float = 0.30,
        skip_mode: str = "none",
    ):
        super().__init__()
        self.msg_len = msg_len
        self.code_len = code_len
        self.tab_dim = tab_dim
        self.n_msg_chunks = n_msg_chunks
        self.n_code_hunks = n_code_hunks
        self.skip_mode = skip_mode
        assert msg_len % n_msg_chunks == 0
        assert code_len % n_code_hunks == 0
        self.msg_chunk_len = msg_len // n_msg_chunks
        self.hunk_len = code_len // n_code_hunks

        self.msg_emb = nn.Embedding(vocab_msg, tok_dim, padding_idx=0)
        self.code_emb = nn.Embedding(vocab_code, tok_dim, padding_idx=0)
        self.convs_msg = nn.ModuleList([nn.Conv1d(tok_dim, num_filters, k, padding=k // 2) for k in (1, 2, 3)])
        self.convs_code_line = nn.ModuleList([nn.Conv1d(tok_dim, num_filters, k, padding=k // 2) for k in (1, 2, 3)])

        h_cnn = 3 * num_filters
        self.msg_self_attn = MultiHeadSelfAttentionBlock(h_cnn, num_heads=num_heads, dropout=dropout)
        self.code_hunk_attn = MultiHeadSelfAttentionBlock(h_cnn, num_heads=num_heads, dropout=dropout)
        self.cross_attn = CrossAttentionBlock(h_cnn, num_heads=num_heads, dropout=dropout)
        self.convs_code_file = nn.ModuleList([nn.Conv1d(h_cnn, num_filters, k, padding=k // 2) for k in (1, 2, 3)])

        self.tab_proj = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, 48),
            nn.GELU(),
        )
        fused_in = 2 * h_cnn + 48
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_in, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.residual_head = nn.Linear(embed_dim, 1)
        self.skip_linear = nn.Linear(tab_dim, 1) if skip_mode != "none" else None

    @staticmethod
    def _conv_pool(x_emb: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        return torch.cat([torch.max(F.relu(conv(x_emb)), dim=-1)[0] for conv in convs], dim=1)

    def extract_tab_np(self, X_packed: np.ndarray) -> np.ndarray:
        return np.asarray(X_packed[:, self.msg_len + self.code_len :], dtype=np.float32)

    def forward(self, x_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_packed.shape[0]
        msg_ids = x_packed[:, : self.msg_len].long().clamp(min=0)
        code_ids = x_packed[:, self.msg_len : self.msg_len + self.code_len].long().clamp(min=0)
        tab_x = x_packed[:, self.msg_len + self.code_len :]

        m_chunks = msg_ids.view(B * self.n_msg_chunks, self.msg_chunk_len)
        m_emb = self.msg_emb(m_chunks).transpose(1, 2)
        m_chunk_feat = self._conv_pool(m_emb, self.convs_msg).view(B, self.n_msg_chunks, -1)
        m_att = self.msg_self_attn(m_chunk_feat)

        c_hunks = code_ids.view(B * self.n_code_hunks, self.hunk_len)
        c_emb = self.code_emb(c_hunks).transpose(1, 2)
        c_line_feat = self._conv_pool(c_emb, self.convs_code_line).view(B, self.n_code_hunks, -1)
        c_line_att = self.code_hunk_attn(c_line_feat)

        m_cross = torch.mean(self.cross_attn(m_att, c_line_att), dim=1)
        c_file_feat = self._conv_pool(c_line_att.transpose(1, 2), self.convs_code_file)

        t_feat = self.tab_proj(tab_x)
        z = self.fc2(self.fc1(torch.cat([m_cross, c_file_feat, t_feat], dim=1)))
        r_logit = self.residual_head(z).squeeze(-1)
        if self.skip_linear is not None:
            return self.skip_linear(tab_x).squeeze(-1) + r_logit, z
        return r_logit, z


class CodeBertBimodalAdapter(nn.Module, LinearSkipMixin):
    def __init__(
        self,
        cb_dim: int = 768,
        tab_dim: int = 13,
        proj_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.25,
        skip_mode: str = "none",
    ):
        super().__init__()
        self.cb_dim = cb_dim
        self.tab_dim = tab_dim
        self.skip_mode = skip_mode

        self.msg_adapter = nn.Sequential(
            nn.LayerNorm(cb_dim),
            nn.Linear(cb_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.code_adapter = nn.Sequential(
            nn.LayerNorm(cb_dim),
            nn.Linear(cb_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bimodal_attn = MultiHeadSelfAttentionBlock(proj_dim, num_heads=4, dropout=dropout)
        self.tab_adapter = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, 64),
            nn.GELU(),
        )
        fused_dim = 4 * proj_dim + 64
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Linear(160, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(embed_dim, 1)
        self.skip_linear = nn.Linear(tab_dim, 1) if skip_mode != "none" else None

    def extract_tab_np(self, X_packed: np.ndarray) -> np.ndarray:
        return np.asarray(X_packed[:, 2 * self.cb_dim :], dtype=np.float32)

    def forward(self, x_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m_cb = x_packed[:, : self.cb_dim]
        c_cb = x_packed[:, self.cb_dim : 2 * self.cb_dim]
        tab_x = x_packed[:, 2 * self.cb_dim :]

        hm = self.msg_adapter(m_cb)
        hc = self.code_adapter(c_cb)
        att = self.bimodal_attn(torch.stack([hm, hc], dim=1))
        am, ac = att[:, 0, :], att[:, 1, :]
        diff_abs = torch.abs(am - ac)
        prod = am * ac
        ht = self.tab_adapter(tab_x)

        z = self.fusion(torch.cat([am, ac, diff_abs, prod, ht], dim=1))
        r_logit = self.residual_head(z).squeeze(-1)
        if self.skip_linear is not None:
            return self.skip_linear(tab_x).squeeze(-1) + r_logit, z
        return r_logit, z


def make_soft_bce_lgb_objective(q_targets: np.ndarray, pos_weight: float, sample_weights: np.ndarray):
    """
    Exact Analytical Gradient & Hessian Objective for `lightgbm.train(params={"objective": ...})`
    (Proposition 3):
      Loss_i(z_i) = u_i * [ omega_+ * q_i * log(1 + exp(-z_i)) + (1 - q_i) * log(1 + exp(z_i)) ]
      grad_i = u_i * [ (1 + (omega_+ - 1) * q_i) * sigma(z_i) - omega_+ * q_i ]
      hess_i = u_i * (1 + (omega_+ - 1) * q_i) * sigma(z_i) * (1 - sigma(z_i))
    """
    q = np.clip(np.asarray(q_targets, dtype=np.float64), 0.0, 1.0)
    u = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1e-8)
    w_pos = float(max(pos_weight, 1e-6))
    eff_scale = u * (1.0 + (w_pos - 1.0) * q)
    pos_pull = u * w_pos * q

    def _objective(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        p = sigmoid_safe(preds)
        grad = eff_scale * p - pos_pull
        hess = np.maximum(eff_scale * p * (1.0 - p), 1e-8)
        return grad, hess

    return _objective


class LightGBMHostV3:
    """
    Gradient-Boosted Decision Tree Host using `lightgbm.train` with callable `objective`
    (no 2N row duplication, exact raw margins `z = raw_score + init_logit`, and immutable
    `V3_CONFIG.lgb_alarm_rounds` default so budgets NEVER mutate across alarms, `R3-C2`).
    """

    def __init__(self, in_dim: int, n_estimators_default: int = V3_CONFIG.lgb_alarm_rounds, seed: int = 42):
        self.in_dim = int(in_dim)
        self.n_estimators_default = int(n_estimators_default)
        self.seed = int(seed)
        self.booster_: Optional[lgb.Booster] = None
        self.constant_logit_: float = -1.5
        self.last_num_rounds_: int = 0
        self.degeneracy_reason_: Optional[str] = None

    def fit_weighted_soft(
        self,
        X: np.ndarray,
        y_soft: np.ndarray,
        pos_weight: float = 1.0,
        sample_weights: Optional[np.ndarray] = None,
        n_estimators: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple["LightGBMHostV3", ComputeStats]:
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float32)
        q = np.clip(np.asarray(y_soft, dtype=np.float64), 0.0, 1.0)
        u = np.asarray(sample_weights, dtype=np.float64) if sample_weights is not None else np.ones(len(X_arr), dtype=np.float64)
        n_rounds = int(n_estimators) if n_estimators is not None else int(self.n_estimators_default)

        eff_pos = float(np.sum(u * q * float(pos_weight)))
        eff_neg = float(np.sum(u * (1.0 - q)))
        p_init = float(np.clip(eff_pos / max(eff_pos + eff_neg, 1e-8), 0.005, 0.995))
        init_logit = float(np.log(p_init / (1.0 - p_init)))

        has_feat_var = len(X_arr) >= 20 and bool(np.any(np.ptp(X_arr, axis=0) > 1e-12))
        w_pos_val = float(max(pos_weight, 1e-6))
        init_grad = u * ((1.0 + (w_pos_val - 1.0) * q) * p_init - w_pos_val * q)
        has_grad_var = float(np.max(np.abs(init_grad))) > 1e-10 if len(init_grad) > 0 else False

        deg_reason: Optional[str] = None
        if not has_feat_var:
            deg_reason = "insufficient_samples_or_constant_features"
        elif float(np.sum(q > 0.01)) < 3 or float(np.sum(q < 0.99)) < 3:
            deg_reason = "single_class_soft_targets"
        elif not has_grad_var:
            deg_reason = "zero_initial_gradient_constant_soft_target"

        if deg_reason is not None:
            self.booster_ = None
            self.constant_logit_ = init_logit
            self.last_num_rounds_ = 0
            self.degeneracy_reason_ = deg_reason
            ms = (time.perf_counter() - t0) * 1000.0
            return self, ComputeStats(
                opt_steps=0,
                examples_seen=len(X_arr),
                trees_built=0,
                requested_rounds=n_rounds,
                actual_iterations=0,
                wall_ms=ms,
            )

        obj_fn = make_soft_bce_lgb_objective(q, pos_weight=pos_weight, sample_weights=u)
        dtrain = lgb.Dataset(
            X_arr,
            label=q,
            init_score=np.full(len(X_arr), init_logit, dtype=np.float64),
            free_raw_data=True,
        )
        params = {
            "objective": obj_fn,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": max(10, min(25, len(X_arr) // 25)),
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "feature_fraction": 0.85,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": int(seed),
            "num_threads": 2,
            "verbose": -1,
            "deterministic": True,
        }
        try:
            self.booster_ = lgb.train(params=params, train_set=dtrain, num_boost_round=n_rounds)
        except Exception as exc:
            raise HostTrainingFailureError(
                f"Unexpected LightGBM supervised training failure (n={len(X_arr)}, rounds={n_rounds}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        actual_iters = int(self.booster_.current_iteration())
        actual_trees = int(self.booster_.num_trees())
        self.degeneracy_reason_ = None
        self.constant_logit_ = init_logit
        self.last_num_rounds_ = actual_iters
        ms = (time.perf_counter() - t0) * 1000.0
        return self, ComputeStats(
            opt_steps=actual_iters,
            examples_seen=len(X_arr) * actual_iters,
            trees_built=actual_trees,
            requested_rounds=n_rounds,
            actual_iterations=actual_iters,
            wall_ms=ms,
        )

    def predict_logits_and_embeddings(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        if self.booster_ is None:
            logits = np.full(len(X_arr), self.constant_logit_, dtype=np.float32)
        else:
            # `predict(..., raw_score=True)` returns tree sum WITHOUT Dataset `init_score`;
            # adding `self.constant_logit_` recovers the exact margin `z_i` seen by `_objective`!
            margins = self.booster_.predict(X_arr, raw_score=True) + self.constant_logit_
            logits = np.asarray(margins, dtype=np.float32)
        z = np.hstack([X_arr[:, : min(X_arr.shape[1], 48)], logits[:, None]]).astype(np.float32)
        return logits, z


class BlendedLightGBMHostV3:
    def __init__(self, host_a: Union[LightGBMHostV3, "BlendedLightGBMHostV3"], host_b: Union[LightGBMHostV3, "BlendedLightGBMHostV3"], alpha_a: float):
        self.host_a = copy.deepcopy(host_a)
        self.host_b = copy.deepcopy(host_b)
        self.alpha_a = float(alpha_a)
        self.n_estimators_default = int(V3_CONFIG.lgb_alarm_rounds)

    def predict_logits_and_embeddings(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lg_a, z_a = self.host_a.predict_logits_and_embeddings(X)
        lg_b, _ = self.host_b.predict_logits_and_embeddings(X)
        lg = (self.alpha_a * lg_a + (1.0 - self.alpha_a) * lg_b).astype(np.float32)
        z = z_a.copy()
        z[:, -1] = lg
        return lg, z


HostModelV3 = Union[nn.Module, LightGBMHostV3, BlendedLightGBMHostV3]


@torch.no_grad()
def predict_logits_and_embeddings_v3(
    model: HostModelV3,
    X: np.ndarray,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(model, (LightGBMHostV3, BlendedLightGBMHostV3)):
        return model.predict_logits_and_embeddings(X)

    dev = device if device is not None else next(model.parameters()).device
    model.eval()
    logits_list, emb_list = [], []
    x_np = np.asarray(X, dtype=np.float32)
    for i in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(x_np[i : i + batch_size]).to(dev, non_blocking=True)
        lg, zb = model(xb)
        logits_list.append(lg.detach().cpu().numpy())
        emb_list.append(zb.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(emb_list, axis=0)


def blend_hosts_v3(model_a: HostModelV3, model_b: HostModelV3, weight_a: float) -> HostModelV3:
    if isinstance(model_a, (LightGBMHostV3, BlendedLightGBMHostV3)):
        return BlendedLightGBMHostV3(model_a, model_b, alpha_a=weight_a)
    out = copy.deepcopy(model_a)
    with torch.no_grad():
        for p_o, p_a, p_b in zip(out.parameters(), model_a.parameters(), model_b.parameters()):
            p_o.data.copy_(weight_a * p_a.data + (1.0 - weight_a) * p_b.data)
    return out


def safe_numerical_verify_update(
    prev_model: HostModelV3,
    cand_model: HostModelV3,
    X_probe: np.ndarray,
) -> Tuple[HostModelV3, bool, bool]:
    """
    Unified Numerical Failure Policy Across ALL Methods:
      - ONLY rolls back to `prev_model` if `cand_logits` contains `NaN` or `Inf`.
      - Constant / degenerate predictions (`std(prob) < 1e-6`) are NEVER rolled back or hidden;
        `is_constant_pred` is returned and logged transparently!
    """
    lg, _ = predict_logits_and_embeddings_v3(cand_model, X_probe[: min(len(X_probe), 512)])
    if not np.all(np.isfinite(lg)):
        return copy.deepcopy(prev_model), True, False
    probs = sigmoid_safe(lg)
    is_const = bool(np.std(probs) < 1e-6)
    return cand_model, False, is_const


# ==============================================================================
# 4. Supervised Soft-BCE Trainer & Kiryo nnPU (Neural Exact Alg. 1 + GBDT Functional-Gradient nnPU)
# ==============================================================================

def weighted_soft_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss_vec = float(pos_weight) * targets * F.softplus(-logits) + (1.0 - targets) * F.softplus(logits)
    if sample_weights is not None:
        return torch.sum(loss_vec * sample_weights) / torch.clamp(torch.sum(sample_weights), min=1e-6)
    return torch.mean(loss_vec)


def train_supervised_v3(
    model: HostModelV3,
    X: np.ndarray,
    y_soft: np.ndarray,
    pos_weight: float,
    epochs: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    sample_weights: Optional[np.ndarray] = None,
    max_opt_steps: Optional[int] = None,
    lgb_n_estimators: Optional[int] = None,
    seed: int = 42,
) -> Tuple[HostModelV3, ComputeStats]:
    if len(X) < 10:
        return model, ComputeStats()

    if isinstance(model, (LightGBMHostV3, BlendedLightGBMHostV3)):
        # Immutable default (`V3_CONFIG.lgb_alarm_rounds`) when `lgb_n_estimators` is omitted (`R3-C2`)!
        n_est = int(lgb_n_estimators) if lgb_n_estimators is not None else int(V3_CONFIG.lgb_alarm_rounds)
        fresh = LightGBMHostV3(X.shape[1], n_estimators_default=V3_CONFIG.lgb_alarm_rounds, seed=seed)
        return fresh.fit_weighted_soft(
            X, y_soft, pos_weight=pos_weight, sample_weights=sample_weights, n_estimators=n_est, seed=seed
        )

    t0 = time.perf_counter()
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev)
    y_t = torch.from_numpy(np.asarray(y_soft, dtype=np.float32)).to(dev)
    w_t = (
        torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)).to(dev)
        if sample_weights is not None
        else torch.ones(len(X), dtype=torch.float32, device=dev)
    )

    n = len(X)
    steps_done = 0
    ex_seen = 0
    stop_early = False
    for _ in range(epochs):
        if stop_early:
            break
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits, _ = model(x_t[idx])
            loss = weighted_soft_bce_loss(logits, y_t[idx], pos_weight=pos_weight, sample_weights=w_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()
            steps_done += 1
            ex_seen += int(len(idx))
            if max_opt_steps is not None and steps_done >= max_opt_steps:
                stop_early = True
                break
    model.eval()
    ms = (time.perf_counter() - t0) * 1000.0
    return model, ComputeStats(opt_steps=steps_done, examples_seen=ex_seen, trees_built=0, wall_ms=ms)


def make_gbdt_fgrad_nnpu_objective(
    is_p_member: np.ndarray,
    is_u_member: np.ndarray,
    prior_pi: float,
    beta: float = 0.0,
    gamma: float = 1.0,
):
    """
    First-Order Functional-Gradient GBDT Non-Negative PU Objective with Disjoint `P` & `U` Membership (`R3-C4`):
      - `u_mask` marks ONLY marginal pool commits `X_pool ~ p(x)` (`n_u = sum(u_mask)`), so any
        supplemented historical `P` rows (`is_p_member=True, is_u_member=False`) NEVER contaminate `R_U^-`!
      - `p_mask` marks positive sample `P` (`n_p = sum(p_mask)`).
      - Returns exact `n_u * d L_nnPU / d z_i` as `grad` and unit positive pseudo-Hessian `hess = 1.0`
        (Friedman 2001 functional gradient boosting, avoiding ill-posed non-convex sigmoid 2nd derivatives).
    """
    p_mask = np.asarray(is_p_member, dtype=bool)
    u_mask = np.asarray(is_u_member, dtype=bool)
    n_u = float(max(int(np.sum(u_mask)), 1))
    n_p = float(max(int(np.sum(p_mask)), 1))
    pi = float(np.clip(prior_pi, 0.02, 0.70))

    def _obj(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        s_pos = sigmoid_safe(preds)       # sigma(+z)
        deriv = s_pos * (1.0 - s_pos)     # d/dz sigma(+z)

        r_u_neg = float(np.mean(s_pos[u_mask]))
        r_p_neg = float(np.mean(s_pos[p_mask]))
        neg_risk = r_u_neg - pi * r_p_neg

        grad = np.zeros_like(preds, dtype=np.float64)
        if neg_risk >= -beta:
            # n_u * d/dz [ pi * R_P^+ + R_U^- - pi * R_P^- ]
            grad[u_mask] += deriv[u_mask]
            grad[p_mask] += (-2.0 * pi * (n_u / n_p)) * deriv[p_mask]
        else:
            # n_u * d/dz [ -gamma * (R_U^- - pi * R_P^-) ]
            grad[u_mask] += (-gamma) * deriv[u_mask]
            grad[p_mask] += (gamma * pi * (n_u / n_p)) * deriv[p_mask]

        hess = np.ones_like(grad, dtype=np.float64)
        return grad, hess

    return _obj


def train_faithful_nnpu_v3(
    model: HostModelV3,
    X_pool: np.ndarray,
    y_obs_pool: np.ndarray,
    prior_pi: float,
    X_pos_supplement: Optional[np.ndarray] = None,
    epochs: int = 6,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    beta: float = 0.0,
    gamma: float = 1.0,
    seed: int = 42,
) -> Tuple[HostModelV3, ComputeStats]:
    """
    Non-Negative PU Learner (`Pure_nnPU`):
      - Neural hosts: exact Kiryo et al. (NeurIPS 2017, Algorithm 1) stochastic mini-batch nnPU
        on marginal `U = X_pool ~ p(x)` and `P`.
      - LightGBM host: first-order functional-gradient GBDT nnPU (`make_gbdt_fgrad_nnpu_objective`)
        with explicit `u_mask` (`X_pool` only) and `p_mask` so supplemented `P` rows never alter `U` (`R3-C4`).
    """
    y_bin = np.asarray(y_obs_pool, dtype=np.float64) > 0.5
    X_u = np.asarray(X_pool, dtype=np.float32)
    X_p_in_pool = X_u[y_bin]

    X_supp = np.empty((0, X_u.shape[1]), dtype=np.float32)
    if len(X_p_in_pool) < 4 and X_pos_supplement is not None and len(X_pos_supplement) > 0:
        X_supp = np.asarray(X_pos_supplement[-32:], dtype=np.float32)

    X_p = np.vstack([X_p_in_pool, X_supp]) if (len(X_p_in_pool) > 0 and len(X_supp) > 0) else (X_p_in_pool if len(X_p_in_pool) > 0 else X_supp)
    pi = float(np.clip(prior_pi, 0.02, 0.70))
    if len(X_p) < 2 or len(X_u) < 16:
        return model, ComputeStats()

    if isinstance(model, (LightGBMHostV3, BlendedLightGBMHostV3)):
        t0 = time.perf_counter()
        if len(X_supp) == 0:
            X_fit = X_u
            is_u = np.ones(len(X_u), dtype=bool)
            is_p = y_bin.copy()
        else:
            X_fit = np.vstack([X_u, X_supp])
            is_u = np.concatenate([np.ones(len(X_u), dtype=bool), np.zeros(len(X_supp), dtype=bool)])
            is_p = np.concatenate([y_bin, np.ones(len(X_supp), dtype=bool)])

        n_est = int(V3_CONFIG.lgb_alarm_rounds)
        fresh = LightGBMHostV3(X_u.shape[1], n_estimators_default=n_est, seed=seed)
        init_lg = float(np.log(pi / (1.0 - pi)))
        obj_nn = make_gbdt_fgrad_nnpu_objective(is_p, is_u, prior_pi=pi, beta=beta, gamma=gamma)
        init_g_nn, _ = obj_nn(np.full(len(X_fit), init_lg, dtype=np.float64), None)
        if (not bool(np.any(np.ptp(X_fit, axis=0) > 1e-12))) or float(np.max(np.abs(init_g_nn))) <= 1e-10:
            fresh.booster_ = None
            fresh.constant_logit_ = init_lg
            fresh.last_num_rounds_ = 0
            fresh.degeneracy_reason_ = "constant_features_or_zero_nnpu_gradient"
            ms = (time.perf_counter() - t0) * 1000.0
            return fresh, ComputeStats(opt_steps=0, examples_seen=len(X_fit), trees_built=0, requested_rounds=n_est, actual_iterations=0, wall_ms=ms)

        dtrain = lgb.Dataset(
            X_fit,
            label=is_p.astype(np.float32),
            init_score=np.full(len(X_fit), init_lg, dtype=np.float64),
            free_raw_data=True,
        )
        params = {
            "objective": obj_nn,
            "learning_rate": 0.08,
            "num_leaves": 15,
            "min_data_in_leaf": max(10, min(25, len(X_fit) // 25)),
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "feature_fraction": 0.85,
            "lambda_l2": 1.0,
            "seed": int(seed),
            "num_threads": 2,
            "verbose": -1,
            "deterministic": True,
        }
        try:
            fresh.booster_ = lgb.train(params=params, train_set=dtrain, num_boost_round=n_est)
        except Exception as exc:
            raise HostTrainingFailureError(
                f"Unexpected LightGBM nnPU training failure (n={len(X_fit)}, rounds={n_est}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        act_it = int(fresh.booster_.current_iteration())
        act_tr = int(fresh.booster_.num_trees())
        fresh.degeneracy_reason_ = None
        fresh.constant_logit_ = init_lg
        fresh.last_num_rounds_ = act_it
        ms = (time.perf_counter() - t0) * 1000.0
        return fresh, ComputeStats(
            opt_steps=act_it,
            examples_seen=len(X_fit) * act_it,
            trees_built=act_tr,
            requested_rounds=n_est,
            actual_iterations=act_it,
            wall_ms=ms,
        )

    t0 = time.perf_counter()
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    x_p_t = torch.from_numpy(X_p).to(dev)
    x_u_t = torch.from_numpy(X_u).to(dev)

    steps_done = 0
    ex_seen = 0
    for _ in range(epochs):
        perm_u = torch.from_numpy(rng.permutation(len(x_u_t))).to(dev)
        for i in range(0, len(x_u_t), batch_size):
            u_sel = perm_u[i : i + batch_size]
            if len(u_sel) < 4:
                continue
            p_batch_sz = min(max(len(x_p_t), 16), 64)
            p_sel = torch.from_numpy(rng.choice(len(x_p_t), size=p_batch_sz, replace=len(x_p_t) < p_batch_sz)).to(dev)

            opt.zero_grad()
            lg_p, _ = model(x_p_t[p_sel])
            lg_u, _ = model(x_u_t[u_sel])

            r_p_pos = torch.mean(torch.sigmoid(-lg_p))
            r_p_neg = torch.mean(torch.sigmoid(lg_p))
            r_u_neg = torch.mean(torch.sigmoid(lg_u))
            neg_risk = r_u_neg - pi * r_p_neg

            if float(neg_risk.item()) < -beta:
                loss = -gamma * neg_risk
            else:
                loss = pi * r_p_pos + neg_risk

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()
            steps_done += 1
            ex_seen += int(len(u_sel) + len(p_sel))

    model.eval()
    ms = (time.perf_counter() - t0) * 1000.0
    return model, ComputeStats(opt_steps=steps_done, examples_seen=ex_seen, trees_built=0, wall_ms=ms)


# ==============================================================================
# 5. Clean 3-Stage Survival-PU V3 Adaptation Engine + Ablation Controls
# ==============================================================================

def compute_covariate_importance_weights_v3(
    X_source: np.ndarray,
    X_drift: np.ndarray,
    w_min: float = V3_CONFIG.imp_weight_min,
    w_max: float = V3_CONFIG.imp_weight_max,
) -> Tuple[np.ndarray, float]:
    """Label-free covariate density-ratio estimation `p_drift(x) / p_source(x)`."""
    if len(X_source) < 30 or len(X_drift) < 15:
        return np.ones(len(X_source), dtype=np.float32), 0.50

    if X_source.shape[1] > 128:
        step = max(X_source.shape[1] // 96, 1)
        Xs = X_source[:, ::step]
        Xd = X_drift[:, ::step]
    else:
        Xs, Xd = X_source, X_drift

    X_all = np.vstack([Xs, Xd])
    d_all = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xd))])
    clf = LogisticRegression(C=0.15, max_iter=200, class_weight="balanced", random_state=42)
    try:
        clf.fit(X_all, d_all)
        prob_all = clf.predict_proba(X_all)[:, 1]
        dom_auc = float(roc_auc_score(d_all, prob_all))
        prob_s = np.clip(prob_all[: len(Xs)], 0.08, 0.92)
        odds = prob_s / (1.0 - prob_s)
        odds = odds / max(float(np.mean(odds)), 1e-6)
        return np.clip(odds, w_min, w_max).astype(np.float32), dom_auc
    except Exception:
        return np.ones(len(X_source), dtype=np.float32), 0.50


@dataclass
class SurvivalPUDiagnosticsV3:
    mean_c_drift: float
    ll_median_days: float
    survival_fallback_used: bool
    pi_target: float
    pi_ht_raw: float
    n_eff_ht: float
    observed_pos_in_drift: int
    bisection_delta: float
    bisection_reachable: bool
    bisection_residual: float
    domain_shift_auc: float
    gate_state: str
    nan_rolled_back: bool
    is_constant_pred: bool
    compute: ComputeStats = field(default_factory=ComputeStats)


def adapt_survival_pu_v3(
    current_model: HostModelV3,
    anchor_model: HostModelV3,
    X_hist: np.ndarray,
    commit_ts_hist: np.ndarray,
    fix_ts_hist: np.ndarray,
    y_obs_hist: np.ndarray,
    drift_indices_in_hist: np.ndarray,
    t_alarm: float,
    pos_weight: float,
    anchor_prevalence: float,
    epochs_teacher: int = V3_CONFIG.neural_teacher_epochs,
    epochs_student: int = V3_CONFIG.neural_student_epochs,
    lr: float = V3_CONFIG.neural_adapt_lr,
    survival_estimator: str = "cohort_incidence",
    tail_cap_override: Optional[float] = None,
    ht_shrinkage_m0: float = V3_CONFIG.ht_shrinkage_m0,
    use_importance_weights: bool = True,
    use_bisection: bool = True,
    teacher_mode: str = "standard",  # "standard", "teacher_only_exact", "teacher_only_matched", "frozen_anchor_teacher"
    use_legacy_gate_ablation: bool = False,
    feature_slice_for_density: Optional[slice] = None,
    llm_drift_prior_probs: Optional[np.ndarray] = None,
    llm_blend_weight: float = V3_CONFIG.llm_blend_weight,
    seed: int = 42,
) -> Tuple[HostModelV3, SurvivalPUDiagnosticsV3]:
    """
    Clean 3-Stage Survival-PU Adaptation at `t_alarm` (Zero Heuristic Gate in Core Method):
      - Stage 1: Causal survival / cumulative-incidence maturity `c_i(t_alarm)` fit strictly on
                 `fix_ts <= t_alarm`.
      - Stage 2: Reliable-cohort covariate-shifted teacher (`c_i >= 0.45` or `y_obs_i == 1`) +
                 clipped inverse-maturity plug-in target prevalence shrunk via `n_eff_heur` (`R3-M1`).
      - Stage 3: Student distillation on `compute_survival_pu_posterior` calibrated via monotone
                 bisection on `B_cal` (`R3-C5`), using immutable `V3_CONFIG` budgets (`R3-C2`).
    """
    comp_total = ComputeStats()
    t_s1 = time.perf_counter()

    # Stage 1: Causal Survival Maturity Curve
    lat_est = build_survival_estimator_v3(survival_estimator, tail_cap_override=tail_cap_override).fit(
        commit_ts_hist, fix_ts_hist, y_obs_hist, t_alarm
    )
    ages_sec = np.maximum(t_alarm - commit_ts_hist, 0.0)
    c_all = lat_est.predict_maturity(ages_sec)
    ll_med = float(lat_est.ll_median_)
    surv_fb = bool(getattr(lat_est, "used_fallback_", False))

    drift_mask = np.zeros(len(X_hist), dtype=bool)
    drift_mask[drift_indices_in_hist] = True
    X_drift = X_hist[drift_mask]
    y_obs_drift = y_obs_hist[drift_mask]
    c_drift = c_all[drift_mask]
    obs_pos_drift = int(np.sum(y_obs_drift > 0.5))

    X_for_dom = X_hist[:, feature_slice_for_density] if feature_slice_for_density is not None else X_hist
    X_drift_dom = X_for_dom[drift_mask]

    # Anchor mature prevalence baseline from causal mature history (`c >= 0.55`)
    mat_mask = c_all >= 0.55
    if int(np.sum(mat_mask)) >= 80:
        c_mat_til = np.maximum(c_all[mat_mask], V3_CONFIG.ht_floor_c)
        pi_mature_anchor = float(np.clip(np.mean(y_obs_hist[mat_mask] / c_mat_til), V3_CONFIG.pi_min, V3_CONFIG.pi_max))
    else:
        pi_mature_anchor = float(anchor_prevalence)

    non_drift_idx = np.where(~drift_mask)[0]
    sel_anc_idx = non_drift_idx[-V3_CONFIG.max_anchor_commits :] if len(non_drift_idx) > V3_CONFIG.max_anchor_commits else non_drift_idx
    X_anc = X_hist[sel_anc_idx]
    y_obs_anc = y_obs_hist[sel_anc_idx]
    c_anc = c_all[sel_anc_idx]

    dom_auc = 0.50
    if use_importance_weights and len(X_anc) > 0:
        imp_anc, dom_auc = compute_covariate_importance_weights_v3(X_for_dom[sel_anc_idx], X_drift_dom)
        imp_anc = imp_anc * np.linspace(0.70, 1.15, len(X_anc), dtype=np.float32)
    else:
        imp_anc = np.ones(len(X_anc), dtype=np.float32)

    X_pool = np.vstack([X_anc, X_drift]) if len(X_anc) > 0 else X_drift
    y_obs_pool = np.concatenate([y_obs_anc, y_obs_drift]) if len(X_anc) > 0 else y_obs_drift
    c_pool = np.concatenate([c_anc, c_drift]) if len(X_anc) > 0 else c_drift

    fl = V3_CONFIG.maturity_weight_floor
    u_anc = imp_anc * (fl + (1.0 - fl) * c_anc)
    u_drf = np.ones(len(X_drift), dtype=np.float32) * V3_CONFIG.drift_emphasis_gamma * (fl + (1.0 - fl) * c_drift)
    u_pool = (np.concatenate([u_anc, u_drf]) if len(X_anc) > 0 else u_drf).astype(np.float32)

    # Calibration pool `B_cal` & Clipped Inverse-Maturity Shrunk Prevalence (`R3-M1`)
    n_cal = min(len(X_pool), max(len(X_drift) * 3, 900))
    cal_sl = slice(len(X_pool) - n_cal, len(X_pool))
    pi_target, pi_ht_raw, n_eff_ht = estimate_ht_shrunk_prevalence(
        y_obs=y_obs_pool[cal_sl],
        c_maturity=c_pool[cal_sl],
        sample_weights=u_pool[cal_sl],
        anchor_prevalence=pi_mature_anchor,
        m0=ht_shrinkage_m0,
    )
    comp_total.wall_ms += (time.perf_counter() - t_s1) * 1000.0

    # Stage 2: Reliable-Cohort Covariate-Shifted Teacher (Immutable Budget from `V3_CONFIG`, `R3-C2`)
    lgb_stage_rounds = int(V3_CONFIG.lgb_alarm_rounds)
    pw_upper = max(4.5, float(pos_weight))
    pos_weight_teach = float(0.5 * pos_weight + 0.5 * np.clip((1.0 - pi_target) / max(pi_target, 0.03), 1.5, pw_upper))

    if teacher_mode == "frozen_anchor_teacher":
        fresh_teacher = copy.deepcopy(anchor_model)
        teacher_effective_pw = float(pos_weight)  # Anchor was trained with `pos_weight`, NOT `pos_weight_teach` (`R3-M6`)!
    else:
        rel_mask = (y_obs_hist > 0.5) | (c_all >= V3_CONFIG.reliable_maturity_thresh)
        rel_idx = np.where(rel_mask)[0]
        if len(rel_idx) >= 100:
            sel_rel = rel_idx[-V3_CONFIG.max_reliable_commits :]
            X_rel = X_hist[sel_rel]
            y_rel = y_obs_hist[sel_rel]
            c_rel = c_all[sel_rel]
            if use_importance_weights:
                imp_rel, dom_auc_r = compute_covariate_importance_weights_v3(X_for_dom[sel_rel], X_drift_dom)
                dom_auc = max(dom_auc, dom_auc_r)
            else:
                imp_rel = np.ones(len(X_rel), dtype=np.float32)
            rel_conf = np.where((y_rel > 0.5) & (c_rel < V3_CONFIG.reliable_maturity_thresh), 0.55 + 0.45 * c_rel, 1.0).astype(np.float32)
            w_rel = (imp_rel * rel_conf * np.linspace(V3_CONFIG.recency_min, V3_CONFIG.recency_max, len(X_rel), dtype=np.float32)).astype(np.float32)
            teach_init = blend_hosts_v3(current_model, anchor_model, weight_a=0.85)
            teacher_effective_pw = float(pos_weight_teach)

            if teacher_mode == "teacher_only_matched":
                b_sz = 128
                b_s2 = count_effective_minibatches_per_epoch(len(X_rel), b_sz, 2)
                b_s3 = count_effective_minibatches_per_epoch(len(X_pool), b_sz, 2)
                steps_s2 = epochs_teacher * b_s2
                steps_s3 = epochs_student * b_s3
                target_steps = steps_s2 + steps_s3
                ep_matched = int(math.ceil(target_steps / float(max(b_s2, 1)))) if b_s2 > 0 else 0
                fresh_teacher, cs_t = train_supervised_v3(
                    teach_init,
                    X_rel,
                    y_rel,
                    pos_weight=pos_weight_teach,
                    epochs=ep_matched,
                    lr=lr,
                    batch_size=b_sz,
                    sample_weights=w_rel,
                    max_opt_steps=target_steps,
                    lgb_n_estimators=2 * lgb_stage_rounds,  # Immutable `2 * 75 = 150` on EVERY alarm (`R3-C2`)!
                    seed=seed,
                )
            else:
                fresh_teacher, cs_t = train_supervised_v3(
                    teach_init,
                    X_rel,
                    y_rel,
                    pos_weight=pos_weight_teach,
                    epochs=epochs_teacher,
                    lr=lr,  # Identical `lr` across Exact, Matched, and SPU (`R3-C2`)!
                    sample_weights=w_rel,
                    lgb_n_estimators=lgb_stage_rounds,
                    seed=seed,
                )
            comp_total.add(cs_t)
        else:
            fresh_teacher = copy.deepcopy(current_model)
            teacher_effective_pw = float(pos_weight)

    if teacher_mode in ("teacher_only_exact", "teacher_only_matched"):
        sel_m, nan_rb, is_const = safe_numerical_verify_update(current_model, fresh_teacher, X_drift)
        return sel_m, SurvivalPUDiagnosticsV3(
            mean_c_drift=float(np.mean(c_drift)),
            ll_median_days=ll_med,
            survival_fallback_used=surv_fb,
            pi_target=pi_target,
            pi_ht_raw=pi_ht_raw,
            n_eff_ht=n_eff_ht,
            observed_pos_in_drift=obs_pos_drift,
            bisection_delta=0.0,
            bisection_reachable=True,
            bisection_residual=0.0,
            domain_shift_auc=dom_auc,
            gate_state="NO_GATE_TEACHER_ONLY",
            nan_rolled_back=nan_rb,
            is_constant_pred=is_const,
            compute=comp_total,
        )

    # Stage 3: Bayesian Survival-PU Calibration & Student Distillation (`R3-C5`)
    t_s3 = time.perf_counter()
    lg_teach, _ = predict_logits_and_embeddings_v3(fresh_teacher, X_pool)
    base_eta = V3_CONFIG.teacher_sharpness * (lg_teach - float(np.log(max(teacher_effective_pw, 1e-6))))

    if llm_drift_prior_probs is not None and len(llm_drift_prior_probs) == len(X_drift):
        p_llm = np.clip(np.asarray(llm_drift_prior_probs, dtype=np.float64), 0.01, 0.99)
        eta_llm = np.log(p_llm / (1.0 - p_llm))
        base_eta[-len(X_drift) :] = (1.0 - llm_blend_weight) * base_eta[-len(X_drift) :] + llm_blend_weight * eta_llm

    if use_bisection:
        bis_res = solve_monotone_pu_bisection(
            base_eta=base_eta[cal_sl],
            y_obs=y_obs_pool[cal_sl],
            c_maturity=c_pool[cal_sl],
            sample_weights=u_pool[cal_sl],
            target_pi=pi_target,
            p_min=V3_CONFIG.posterior_p_min,
            p_max=V3_CONFIG.posterior_p_max,
        )
        delta_star = bis_res.delta_star
        bis_reach = bis_res.is_reachable
    else:
        delta_star = 0.0
        bis_reach = True

    # Single shared posterior call — identical function & bounds as `solve_monotone_pu_bisection` (`R3-C5`)!
    q_post = compute_survival_pu_posterior(
        base_eta=base_eta,
        delta=delta_star,
        y_obs=y_obs_pool,
        c_maturity=c_pool,
        p_min=V3_CONFIG.posterior_p_min,
        p_max=V3_CONFIG.posterior_p_max,
    ).astype(np.float32)

    # Measure residual directly on the final `float32` `q_post` array sent to `train_supervised_v3`!
    u_cal_f64 = np.maximum(u_pool[cal_sl].astype(np.float64), 1e-8)
    actual_cal_mean = float(np.sum(u_cal_f64 * q_post[cal_sl].astype(np.float64)) / np.sum(u_cal_f64))
    bis_resid = abs(actual_cal_mean - float(pi_target)) if use_bisection else 0.0
    comp_total.wall_ms += (time.perf_counter() - t_s3) * 1000.0

    cand_student, cs_s = train_supervised_v3(
        copy.deepcopy(fresh_teacher),
        X_pool,
        q_post,
        pos_weight=pos_weight_teach,
        epochs=epochs_student,
        lr=lr,
        sample_weights=u_pool,
        lgb_n_estimators=lgb_stage_rounds,
        seed=seed + 7,
    )
    comp_total.add(cs_s)

    gate_state = "CLEAN_3STAGE_NO_GATE"
    if use_legacy_gate_ablation:
        lg_cand_d, _ = predict_logits_and_embeddings_v3(cand_student, X_drift)
        lg_cur_d, _ = predict_logits_and_embeddings_v3(current_model, X_drift)
        pr_cand = float(np.mean(sigmoid_safe(lg_cand_d) >= 0.5))
        pr_cur = float(np.mean(sigmoid_safe(lg_cur_d) >= 0.5))
        cur_col = (pr_cur < 0.025) or (pr_cur > 0.92)
        cand_deg = (pr_cand < 0.008) or (pr_cand > 0.95)
        prev_shift = abs(pi_target - pi_mature_anchor)
        neff_d = float(np.sum(y_obs_drift + c_drift * (1.0 - y_obs_drift)))
        if cand_deg and not cur_col:
            gate_state = "LEGACY_ROLLBACK"
            raw_final = copy.deepcopy(current_model)
        elif (dom_auc < 0.68) and (prev_shift < 0.035) and (not cur_col):
            gate_state = "LEGACY_ROLLBACK"
            raw_final = copy.deepcopy(fresh_teacher)
        elif ((obs_pos_drift < 2) or (neff_d < 35.0)) and (not cur_col):
            gate_state = "LEGACY_EMA"
            raw_final = blend_hosts_v3(fresh_teacher, cand_student, weight_a=0.55)
        else:
            gate_state = "LEGACY_FULL"
            raw_final = blend_hosts_v3(fresh_teacher, cand_student, weight_a=0.20)
    else:
        raw_final = cand_student

    final_model, nan_rb, is_const = safe_numerical_verify_update(current_model, raw_final, X_drift)
    return final_model, SurvivalPUDiagnosticsV3(
        mean_c_drift=float(np.mean(c_drift)),
        ll_median_days=ll_med,
        survival_fallback_used=surv_fb,
        pi_target=pi_target,
        pi_ht_raw=pi_ht_raw,
        n_eff_ht=n_eff_ht,
        observed_pos_in_drift=obs_pos_drift,
        bisection_delta=delta_star,
        bisection_reachable=bis_reach,
        bisection_residual=bis_resid,
        domain_shift_auc=dom_auc,
        gate_state=gate_state,
        nan_rolled_back=nan_rb,
        is_constant_pred=is_const,
        compute=comp_total,
    )


# ==============================================================================
# 6. Unified Track 1 Within-Host Counterfactual Suite Runner
# ==============================================================================

def build_host_backbone_v3(
    host_type: str,
    X_sample: np.ndarray,
    meta: Optional[Dict] = None,
    skip_mode: str = "none",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> HostModelV3:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = device if device is not None else get_default_device()

    if host_type == "lightgbm":
        return LightGBMHostV3(in_dim=X_sample.shape[1], n_estimators_default=V3_CONFIG.lgb_alarm_rounds, seed=seed)
    if host_type == "mlp":
        return TabularNeuralHostV3(in_dim=X_sample.shape[1], skip_mode=skip_mode).to(dev)
    if host_type == "deepjit_cnn":
        tm = (meta or {}).get("tok_meta", {})
        return DeepJitDualEmbCNN(
            vocab_msg=int(tm.get("vocab_msg", 8000)),
            vocab_code=int(tm.get("vocab_code", 8000)),
            msg_len=int(tm.get("msg_len", 64)),
            code_len=int(tm.get("code_len", 128)),
            tab_dim=int(tm.get("tab_dim", 13)),
            skip_mode=skip_mode,
        ).to(dev)
    if host_type == "att_cnn":
        tm = (meta or {}).get("tok_meta", {})
        return AttentionDualEmbCNN(
            vocab_msg=int(tm.get("vocab_msg", 8000)),
            vocab_code=int(tm.get("vocab_code", 8000)),
            msg_len=int(tm.get("msg_len", 64)),
            code_len=int(tm.get("code_len", 128)),
            tab_dim=int(tm.get("tab_dim", 13)),
            n_msg_chunks=int(tm.get("n_msg_chunks", 8)),
            n_code_hunks=int(tm.get("n_code_hunks", 8)),
            skip_mode=skip_mode,
        ).to(dev)
    if host_type == "codebert":
        cm = (meta or {}).get("cb_meta", {})
        return CodeBertBimodalAdapter(
            cb_dim=int(cm.get("cb_dim", 768)),
            tab_dim=int(cm.get("tab_dim", 13)),
            skip_mode=skip_mode,
        ).to(dev)
    raise ValueError(f"Unknown host_type={host_type}")


def compute_shared_causal_alarm_trajectory(
    Z_init_all: np.ndarray,
    n_init: int,
    n_val: int,
    window_size: int = V3_CONFIG.window_size_default,
    monitor_mode: str = "frechet",
    calib_sampling: str = "block",
    reference_refresh: str = "rolling",
    drift_percentile: Optional[float] = None,
    drift_cooldown: Optional[float] = None,
    seed: int = 42,
    return_timing: bool = False,
):
    stream_start = n_init + n_val
    t_mf0 = time.perf_counter()
    mon = DriftMonitorV3(
        mode=monitor_mode,
        calib_sampling=calib_sampling,
        reference_refresh=reference_refresh,
        percentile=drift_percentile,
        cooldown_windows=int(drift_cooldown) if drift_cooldown is not None else None,
        seed=seed,
    ).fit_reference(Z_init_all[:n_init], Z_init_all[n_init:stream_start], window_size=window_size)
    mon_fit_ms = (time.perf_counter() - t_mf0) * 1000.0

    init_th = float(mon.threshold_)
    windows_out: List[Tuple[int, int, bool, float]] = []
    per_win_mon_ms: List[float] = []
    for st in range(stream_start, len(Z_init_all), window_size):
        ed = min(st + window_size, len(Z_init_all))
        if ed - st < max(window_size // 3, 50):
            windows_out.append((st, ed, False, 0.0))
            per_win_mon_ms.append(0.0)
            continue
        t_mw0 = time.perf_counter()
        dist, fired = mon.score_window(Z_init_all[st:ed])
        windows_out.append((st, ed, fired, float(dist)))
        if fired:
            ref_sl = slice(max(0, ed - 3 * window_size), ed)
            mon.update_reference(Z_init_all[ref_sl], window_size=window_size)
        per_win_mon_ms.append((time.perf_counter() - t_mw0) * 1000.0)
    if return_timing:
        return windows_out, init_th, float(mon_fit_ms), per_win_mon_ms
    return windows_out, init_th


def execute_within_host_counterfactual_suite_v3(
    X: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    n_init: int,
    n_val: int,
    host_type: str = "mlp",
    meta: Optional[Dict] = None,
    skip_mode: str = "none",
    methods: Optional[List[str]] = None,
    window_size: int = V3_CONFIG.window_size_default,
    monitor_mode: str = "frechet",
    calib_sampling: str = "block",
    reference_refresh: str = "rolling",
    drift_percentile: Optional[float] = None,
    drift_cooldown: Optional[int] = None,
    closed_loop_embeddings: bool = False,
    drift_onset_idx: int = -1,
    fix_ts_for_init: Optional[np.ndarray] = None,
    spu_overrides: Optional[Dict] = None,
    llm_commit_probs_stream: Optional[np.ndarray] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Execute Track 1 Within-Host Counterfactual Supervision Audit under STRICT parity:
      1. Causal initial model `m_causal_init` trained ONCE on `[0 : stream_start]` using
         `fix_ts_for_init` (defaults to `fix_ts`), freezing initial weights & monitor (`R3-C6`).
      2. Shared causal `DriftMonitorV3` alarm schedule computed ONCE; per-method `executed_alarm_windows`
         logged explicitly so closed-loop monitors report their exact executed alarm trajectory (`R3-C7`).
      3. Strict `test-then-update` + strict per-commit `calibrate_causal_stream_probs_strict` (`R3-C1`).
      4. `B2_2000` uses exact `2,000` commits (`max(0, w_ed - 2000)`), and `local_fork_summary`
         selects the first alarm with `w_ed > drift_onset_idx` on shift streams (`R3-M6`).
    """
    if methods is None:
        methods = [
            "B0_Frozen",
            "B1a_Full_Hindsight",
            "B1b_1W_Oracle",
            "B1b_4W_Oracle",
            "B2_1W",
            "B2_4W",
            "B2_4W_Bal",
            "B2_2000",
            "Wait_30",
            "Wait_60",
            "Wait_90",
            "Pure_nnPU",
            "FixedCurve_PU",
            "ODaSC_Host",
            "Teacher_Only_Exact",
            "Teacher_Only_Matched",
            "Survival_PU",
            "Survival_PU_WithGate",
        ]

    spu_kw = dict(spu_overrides or {})
    stream_start = n_init + n_val
    n_total = len(X)
    n_eval = n_total - stream_start
    t_start = float(commit_ts[stream_start - 1])
    fts_init_src = fix_ts_for_init if fix_ts_for_init is not None else fix_ts

    dens_slice: Optional[slice] = None
    if host_type in ("deepjit_cnn", "att_cnn"):
        tm = (meta or {}).get("tok_meta", {})
        dens_slice = slice(int(tm.get("msg_len", 64)) + int(tm.get("code_len", 128)), X.shape[1])

    # 1. Causal Initial Checkpoint `m_causal_init` (with phase-decomposed init timing, `R3b-M6`)
    t_ic0 = time.perf_counter()
    _, y_obs_start = observe_labels_at_time(
        commit_ts[:stream_start], y_true[:stream_start], fts_init_src[:stream_start], t_start
    )
    ages_start_d = (t_start - commit_ts[:stream_start]) / SECONDS_PER_DAY
    mat_init = ages_start_d >= 90.0
    if int(np.sum(mat_init)) >= 100:
        anchor_prev = float(np.clip(np.mean(y_obs_start[mat_init]) * 1.15, 0.03, 0.65))
    else:
        anchor_prev = float(np.clip(np.mean(y_obs_start) * 1.25, 0.03, 0.65))
    pos_weight_init = float(np.clip((1.0 - anchor_prev) / max(anchor_prev, 0.03), 1.2, 8.0))

    m_causal_init = build_host_backbone_v3(host_type, X[:16], meta=meta, skip_mode=skip_mode, seed=seed, device=device)
    if hasattr(m_causal_init, "init_linear_skip_from_data") and skip_mode != "none":
        m_causal_init.init_linear_skip_from_data(X[:stream_start], y_obs_start, pos_weight=pos_weight_init, seed=seed)

    init_epochs = 12 if host_type != "lightgbm" else 1
    m_causal_init, init_comp = train_supervised_v3(
        m_causal_init,
        X[:stream_start],
        y_obs_start,
        pos_weight=pos_weight_init,
        epochs=init_epochs,
        lr=8e-4 if host_type in ("mlp", "codebert") else 7e-4,
        lgb_n_estimators=int(V3_CONFIG.lgb_init_rounds),
        seed=seed,
    )
    init_causal_ms = (time.perf_counter() - t_ic0) * 1000.0

    # Optional Hindsight Reference Checkpoint ONLY for `B1a_Full_Hindsight`
    m_hindsight_init: Optional[HostModelV3] = None
    logits_hind_init_past: Optional[np.ndarray] = None
    pos_weight_hind = pos_weight_init
    init_hind_ms = 0.0
    if "B1a_Full_Hindsight" in methods:
        t_ih0 = time.perf_counter()
        prev_true_init = float(np.clip(np.mean(y_true[:stream_start]), 0.03, 0.65))
        pos_weight_hind = float(np.clip((1.0 - prev_true_init) / max(prev_true_init, 0.03), 1.2, 8.0))
        m_hindsight_init = build_host_backbone_v3(host_type, X[:16], meta=meta, skip_mode=skip_mode, seed=seed, device=device)
        if hasattr(m_hindsight_init, "init_linear_skip_from_data") and skip_mode != "none":
            m_hindsight_init.init_linear_skip_from_data(X[:stream_start], y_true[:stream_start].astype(np.float32), pos_weight=pos_weight_hind, seed=seed)
        m_hindsight_init, _ = train_supervised_v3(
            m_hindsight_init,
            X[:stream_start],
            y_true[:stream_start].astype(np.float32),
            pos_weight=pos_weight_hind,
            epochs=init_epochs,
            lr=8e-4 if host_type in ("mlp", "codebert") else 7e-4,
            lgb_n_estimators=int(V3_CONFIG.lgb_init_rounds),
            seed=seed,
        )
        logits_hind_init_past, _ = predict_logits_and_embeddings_v3(m_hindsight_init, X[:stream_start], device=device)
        init_hind_ms = (time.perf_counter() - t_ih0) * 1000.0

    # 2. Shared Causal Alarm Schedule & B0 Predictions (explicitly split init vs eval inference timing, `R3b-M6`)
    t_p_past0 = time.perf_counter()
    logits_b0_past, Z_init_past = predict_logits_and_embeddings_v3(m_causal_init, X[:stream_start], device=device)
    b0_init_pred_ms = (time.perf_counter() - t_p_past0) * 1000.0

    t_p_eval0 = time.perf_counter()
    logits_b0_eval, Z_init_eval = predict_logits_and_embeddings_v3(m_causal_init, X[stream_start:], device=device)
    b0_eval_pred_ms = (time.perf_counter() - t_p_eval0) * 1000.0

    logits_b0_all = np.concatenate([logits_b0_past, logits_b0_eval], axis=0)
    Z_init_all = np.concatenate([Z_init_past, Z_init_eval], axis=0)
    z_init_causal_past = logits_b0_past.astype(np.float64)

    shared_windows, init_drift_th, shared_mon_fit_ms, per_win_shared_mon_ms = compute_shared_causal_alarm_trajectory(
        Z_init_all,
        n_init=n_init,
        n_val=n_val,
        window_size=window_size,
        monitor_mode=monitor_mode,
        calib_sampling=calib_sampling,
        reference_refresh=reference_refresh,
        drift_percentile=drift_percentile,
        drift_cooldown=drift_cooldown,
        seed=seed,
        return_timing=True,
    )
    shared_monitor_ms = float(sum(per_win_shared_mon_ms))

    # 3. Local 1-Alarm Counterfactual Fork (`R3-M6`: strictly first post-onset alarm when `drift_onset_idx >= stream_start`!)
    local_fork_summary: Optional[Dict] = None
    if drift_onset_idx >= stream_start:
        cand_fw = [idx_w for idx_w, (_, we, f, _) in enumerate(shared_windows) if f and we > drift_onset_idx]
        fork_kind = "first_post_onset_alarm"
    else:
        cand_fw = [idx_w for idx_w, (_, _, f, _) in enumerate(shared_windows) if f]
        fork_kind = "first_stream_alarm"

    if cand_fw and (cand_fw[0] + 1) < len(shared_windows):
        fw_idx = cand_fw[0]
        f_st, f_ed, _, _ = shared_windows[fw_idx]
        nxt_st = f_ed
        nxt_ed = min(f_ed + 2 * window_size, n_total)
        if nxt_ed - nxt_st >= 50 and len(np.unique(y_true[nxt_st:nxt_ed])) > 1:
            t_fork = float(commit_ts[f_ed - 1])
            _, y_obs_fork = observe_labels_at_time(commit_ts[:f_ed], y_true[:f_ed], fix_ts[:f_ed], t_fork)
            sl_4w = slice(max(0, f_ed - 4 * window_size), f_ed)

            m_fk_ora, _ = train_supervised_v3(
                copy.deepcopy(m_causal_init), X[sl_4w], y_true[sl_4w].astype(np.float32),
                pos_weight=pos_weight_init, epochs=5, lr=V3_CONFIG.neural_adapt_lr,
                lgb_n_estimators=int(V3_CONFIG.lgb_alarm_rounds), seed=seed + 101,
            )
            m_fk_cen, _ = train_supervised_v3(
                copy.deepcopy(m_causal_init), X[sl_4w], y_obs_fork[sl_4w],
                pos_weight=pos_weight_init, epochs=5, lr=V3_CONFIG.neural_adapt_lr,
                lgb_n_estimators=int(V3_CONFIG.lgb_alarm_rounds), seed=seed + 101,
            )
            m_fk_spu, _ = adapt_survival_pu_v3(
                current_model=copy.deepcopy(m_causal_init),
                anchor_model=m_causal_init,
                X_hist=X[:f_ed],
                commit_ts_hist=commit_ts[:f_ed],
                fix_ts_hist=fix_ts[:f_ed],
                y_obs_hist=y_obs_fork,
                drift_indices_in_hist=np.arange(f_st, f_ed),
                t_alarm=t_fork,
                pos_weight=pos_weight_init,
                anchor_prevalence=anchor_prev,
                feature_slice_for_density=dens_slice,
                seed=seed + 101,
                **spu_kw,
            )
            p_b0_nx = sigmoid_safe(logits_b0_all[nxt_st:nxt_ed])
            p_ora_nx = sigmoid_safe(predict_logits_and_embeddings_v3(m_fk_ora, X[nxt_st:nxt_ed], device=device)[0])
            p_cen_nx = sigmoid_safe(predict_logits_and_embeddings_v3(m_fk_cen, X[nxt_st:nxt_ed], device=device)[0])
            p_spu_nx = sigmoid_safe(predict_logits_and_embeddings_v3(m_fk_spu, X[nxt_st:nxt_ed], device=device)[0])
            local_fork_summary = {
                "fork_kind": fork_kind,
                "is_post_onset_fork": bool(drift_onset_idx >= stream_start and f_ed > drift_onset_idx),
                "fork_alarm_end_idx": int(f_ed),
                "eval_slice": [int(nxt_st), int(nxt_ed)],
                "censoring_rate_in_4W": float(1.0 - np.sum(y_obs_fork[sl_4w]) / max(np.sum(y_true[sl_4w]), 1)),
                "ap_B0": float(average_precision_score(y_true[nxt_st:nxt_ed], p_b0_nx)),
                "ap_B1b_Local": float(average_precision_score(y_true[nxt_st:nxt_ed], p_ora_nx)),
                "ap_B2_Local": float(average_precision_score(y_true[nxt_st:nxt_ed], p_cen_nx)),
                "ap_SPU_Local": float(average_precision_score(y_true[nxt_st:nxt_ed], p_spu_nx)),
            }

    # 4. Run Every Requested Method Along the Stream (Complete Standalone Dependency Chain + Incremental Timing, `R3c-3`)
    def _calibrate_with_window_timing(
        past_z: np.ndarray,
        stream_z: np.ndarray,
        n_wins: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
        tc0 = time.perf_counter()
        try:
            ret = calibrate_causal_stream_probs_strict(
                past_z, stream_z, commit_ts, y_true, fix_ts, stream_start,
                window_size=window_size, return_window_ms=True, clock_fn=time.perf_counter,
            )
        except TypeError:
            ret = calibrate_causal_stream_probs_strict(
                past_z, stream_z, commit_ts, y_true, fix_ts, stream_start, window_size=window_size,
            )
        tc1 = time.perf_counter()
        outer_ms = float((tc1 - tc0) * 1000.0)
        if isinstance(ret, tuple) and len(ret) == 3:
            cp, tv, w_ms = ret
            w_list = [float(v) for v in w_ms]
        else:
            cp, tv = ret
            w_list = [0.0] * max(n_wins, 1)
        if len(w_list) < n_wins:
            w_list.extend([0.0] * (n_wins - len(w_list)))
        sum_w = float(sum(w_list))
        tot_ms = max(outer_ms, sum_w)
        if tot_ms > sum_w + 1e-9 and len(w_list) > 0:
            rem_share = (tot_ms - sum_w) / float(len(w_list))
            w_list = [v + rem_share for v in w_list]
        return cp, tv, tot_ms, w_list

    method_outputs: Dict[str, Dict] = {}
    shared_anchor_init_total_ms = float(init_causal_ms + b0_init_pred_ms)

    for m_name in methods:
        if m_name == "B0_Frozen":
            t_p_b0 = time.perf_counter()
            z_b0 = logits_b0_eval.astype(np.float64)
            p_b0 = sigmoid_safe(z_b0)
            b0_marginal_post_ms = (time.perf_counter() - t_p_b0) * 1000.0

            cal_b0, th_vec_b0, cal_b0_inner_ms, _ = _calibrate_with_window_timing(
                z_init_causal_past, z_b0, len(shared_windows)
            )
            t_dec1 = time.perf_counter()
            dec_b0 = (p_b0 >= th_vec_b0).astype(np.int32)
            cal_b0_ms = cal_b0_inner_ms + max((time.perf_counter() - t_dec1) * 1000.0, 0.0)

            # Classify precomputed shared anchor init & eval inference under `shared_anchor_*` (`R3d-2`):
            # included in standalone totals (`init_fit_ms`, `stream_predict_and_embed_ms`, `consistent_*`),
            # excluded from marginal `incremental_suite_*` totals!
            tb_b0 = _pack_timing_breakdown(
                init_fit_ms=0.0,
                stream_predict_and_embed_ms=b0_marginal_post_ms,
                drift_monitor_ms=0.0,
                adaptation_total_ms=0.0,
                calibration_ms=cal_b0_ms,
                shared_anchor_init_ms=shared_anchor_init_total_ms,
                shared_monitor_init_ms=0.0,
                shared_anchor_eval_embed_ms=float(b0_eval_pred_ms),
                incremental_monitor_ms=0.0,
            )
            method_outputs[m_name] = {
                "raw_logits": z_b0,
                "raw_probs": p_b0,
                "cal_probs": cal_b0,
                "tuned_th_vec": th_vec_b0,
                "decisions_th": dec_b0,
                "n_alarms": 0,
                "alarm_rate": 0.0,
                "nan_rollbacks": 0,
                "constant_output_alarms": 0,
                "survival_fallbacks": 0,
                "bisection_unreachable": 0,
                "per_alarm_wall_ms": [],
                "per_alarm_step_total_ms": [],
                "end_to_end_eval_ms": tb_b0["consistent_stream_eval_ms"],
                "timing_breakdown_ms": tb_b0,
                "executed_alarm_windows": [
                    {"start": int(ws), "end": int(we), "fired": False, "distance": float(di)}
                    for (ws, we, _, di) in shared_windows
                ],
                "compute": ComputeStats(),
            }
            continue

        cur_model = copy.deepcopy(m_hindsight_init if m_name == "B1a_Full_Hindsight" else m_causal_init)
        anc_model = m_hindsight_init if m_name == "B1a_Full_Hindsight" else m_causal_init
        pw_base = pos_weight_hind if m_name == "B1a_Full_Hindsight" else pos_weight_init
        past_init_z = (
            logits_hind_init_past.astype(np.float64)
            if (m_name == "B1a_Full_Hindsight" and logits_hind_init_past is not None)
            else z_init_causal_past
        )

        cl_mon: Optional[DriftMonitorV3] = None
        cl_mon_fit_ms = 0.0
        if closed_loop_embeddings:
            t_cl0 = time.perf_counter()
            cl_mon = DriftMonitorV3(
                mode=monitor_mode,
                calib_sampling=calib_sampling,
                reference_refresh=reference_refresh,
                percentile=drift_percentile,
                cooldown_windows=drift_cooldown,
                seed=seed,
            ).fit_reference(Z_init_all[:n_init], Z_init_all[n_init:stream_start], window_size=window_size)
            cl_mon_fit_ms = (time.perf_counter() - t_cl0) * 1000.0

        odasc_host_tracker: Optional[SongDenStreamConfidenceTracker] = None
        odasc_init_ms = 0.0
        if m_name == "ODaSC_Host":
            t_od0 = time.perf_counter()
            Xd_init = (X[:stream_start, dens_slice] if dens_slice is not None else X[:stream_start]).astype(np.float64)
            sub_i = Xd_init[: min(len(Xd_init), 250)]
            sq_i = np.sum(sub_i ** 2, axis=1)
            dp_i = np.sqrt(np.maximum(sq_i[:, None] + sq_i[None, :] - 2.0 * (sub_i @ sub_i.T), 0.0))
            np.fill_diagonal(dp_i, 0.0)
            eps_h = float(0.5 * np.percentile(dp_i[dp_i > 0], 15)) if np.any(dp_i > 0) else 0.75
            odasc_host_tracker = SongDenStreamConfidenceTracker(eps_radius=eps_h)
            odasc_host_tracker.calibrate_and_compute_alpha(
                Xd_init[-min(len(Xd_init), 600) :], (y_obs_start[-min(len(Xd_init), 600) :] > 0.5).astype(np.int32), t_sec=t_start
            )
            odasc_init_ms = (time.perf_counter() - t_od0) * 1000.0

        z_stream = np.zeros(n_eval, dtype=np.float64)
        comp_m = ComputeStats()
        pred_ms_total = 0.0
        mon_ms_total = 0.0
        adapt_ms_total = 0.0
        per_alarm_adapt_ms: List[float] = []
        fired_window_step_records: List[Tuple[int, int, int, float, float, float]] = []
        executed_alarm_windows: List[Dict] = []
        n_alarms_fired = 0
        nan_rollbacks = 0
        const_alarms = 0
        surv_fallbacks = 0
        bis_unreach = 0
        consec_alarms = 0
        lgb_rounds = int(V3_CONFIG.lgb_alarm_rounds)
        adapt_lr = float(V3_CONFIG.neural_adapt_lr)

        for w_idx, (w_st, w_ed, shared_fired, shared_dist) in enumerate(shared_windows):
            # STRICT TEST-BEFORE-UPDATE!
            t_wp0 = time.perf_counter()
            lg_w, z_emb_w = predict_logits_and_embeddings_v3(cur_model, X[w_st:w_ed], device=device)
            z_stream[w_st - stream_start : w_ed - stream_start] = lg_w
            w_pred_ms = (time.perf_counter() - t_wp0) * 1000.0
            pred_ms_total += w_pred_ms

            if closed_loop_embeddings and cl_mon is not None:
                t_wm0 = time.perf_counter()
                if (w_ed - w_st) < max(window_size // 3, 50):
                    dist_used, fired = 0.0, False
                else:
                    dist_used, fired = cl_mon.score_window(z_emb_w)
                w_mon_ms = (time.perf_counter() - t_wm0) * 1000.0
            else:
                dist_used, fired = shared_dist, shared_fired
                w_mon_ms = float(per_win_shared_mon_ms[w_idx])

            executed_alarm_windows.append({
                "start": int(w_st),
                "end": int(w_ed),
                "fired": bool(fired),
                "distance": float(dist_used),
            })

            if not fired:
                mon_ms_total += w_mon_ms
                consec_alarms = 0
                continue

            n_alarms_fired += 1
            consec_alarms += 1
            t_al = float(commit_ts[w_ed - 1])
            t_up0 = time.perf_counter()
            _, y_obs_al = observe_labels_at_time(commit_ts[:w_ed], y_true[:w_ed], fix_ts[:w_ed], t_al)
            step_seed = seed + 31 * (w_idx + 1)

            sl_1w = slice(w_st, w_ed)
            sl_4w = slice(max(0, w_ed - 4 * window_size), w_ed)
            sl_2000 = slice(max(0, w_ed - int(V3_CONFIG.max_anchor_commits)), w_ed)  # Exact 2,000 commits (`R3-M6`)!

            if m_name == "B1a_Full_Hindsight":
                sl_use = sl_4w if consec_alarms >= 2 else sl_1w
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_use], y_true[sl_use].astype(np.float32), pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B1b_1W_Oracle":
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_1w], y_true[sl_1w].astype(np.float32), pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B1b_4W_Oracle":
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_4w], y_true[sl_4w].astype(np.float32), pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B2_1W":
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_1w], y_obs_al[sl_1w], pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B2_4W":
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_4w], y_obs_al[sl_4w], pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B2_4W_Bal":
                y_sub = y_obs_al[sl_4w]
                np_obs = float(np.sum(y_sub > 0.5))
                nn_obs = float(len(y_sub) - np_obs)
                pw_dyn = float(np.clip(nn_obs / max(np_obs, 1.0), 1.2, 25.0))
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_4w], y_sub, pos_weight=pw_dyn, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "B2_2000":
                cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sl_2000], y_obs_al[sl_2000], pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name in ("Wait_30", "Wait_60", "Wait_90"):
                wait_d = float(int(m_name.split("_")[1]))
                ages_d = (t_al - commit_ts[:w_ed]) / SECONDS_PER_DAY
                elig = np.where((ages_d >= wait_d) | (y_obs_al > 0.5))[0]
                if len(elig) >= 80:
                    sel_w = elig[-4 * window_size :]
                    cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sel_w], y_obs_al[sel_w], pos_weight=pw_base, epochs=5, lr=adapt_lr, lgb_n_estimators=lgb_rounds, seed=step_seed)
                    cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                    comp_m.add(cs)
                    nan_rollbacks += int(nr)
                    const_alarms += int(ic)

            elif m_name == "Pure_nnPU":
                lat_tmp = build_survival_estimator_v3("cohort_incidence").fit(commit_ts[:w_ed], fix_ts[:w_ed], y_obs_al, t_al)
                c_tmp = lat_tmp.predict_maturity(np.maximum(t_al - commit_ts[sl_4w], 0.0))
                pi_nn, _, _ = estimate_ht_shrunk_prevalence(
                    y_obs=y_obs_al[sl_4w],
                    c_maturity=c_tmp,
                    sample_weights=np.ones(w_ed - sl_4w.start, dtype=np.float32),
                    anchor_prevalence=anchor_prev,
                )
                pos_hist = X[:w_ed][y_obs_al > 0.5]
                cand, cs = train_faithful_nnpu_v3(
                    copy.deepcopy(cur_model),
                    X_pool=X[sl_4w],
                    y_obs_pool=y_obs_al[sl_4w],
                    prior_pi=pi_nn,
                    X_pos_supplement=pos_hist,
                    epochs=6,
                    lr=adapt_lr,
                    seed=step_seed,
                )
                cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                comp_m.add(cs)
                nan_rollbacks += int(nr)
                const_alarms += int(ic)

            elif m_name == "ODaSC_Host" and odasc_host_tracker is not None:
                Xd_all = (X[:w_ed, dens_slice] if dens_slice is not None else X[:w_ed]).astype(np.float64)
                ages_d = (t_al - commit_ts[:w_ed]) / SECONDS_PER_DAY
                elig_o = np.where((y_obs_al > 0.5) | (ages_d >= 15.0))[0]
                if len(elig_o) >= 80:
                    sel_o = elig_o[-1600:]
                    y_bin_o = (y_obs_al[sel_o] > 0.5).astype(np.int32)
                    # Update DenStream micro-clusters on new trailing arrivals over ALL `D` features (`R3b-M4`)
                    n_new = min(len(sel_o), window_size)
                    odasc_host_tracker.calibrate_and_compute_alpha(Xd_all[sel_o[-n_new:]], y_bin_o[-n_new:], t_sec=t_al)
                    a1_pool = odasc_host_tracker.estimate_batch_alpha1_raw(Xd_all[sel_o], y_bin_o)
                    conf_o = np.clip(np.where(y_bin_o == 1, 1.0, 1.0 - a1_pool), 0.05, 1.0).astype(np.float32)
                    cand, cs = train_supervised_v3(copy.deepcopy(cur_model), X[sel_o], y_obs_al[sel_o], pos_weight=pw_base, epochs=5, lr=adapt_lr, sample_weights=conf_o, lgb_n_estimators=lgb_rounds, seed=step_seed)
                    cur_model, nr, ic = safe_numerical_verify_update(cur_model, cand, X[w_st:w_ed])
                    comp_m.add(cs)
                    nan_rollbacks += int(nr)
                    const_alarms += int(ic)

            elif m_name in ("FixedCurve_PU", "Teacher_Only_Exact", "Teacher_Only_Matched", "Survival_PU", "Survival_PU_WithGate"):
                call_kw = dict(spu_kw)
                if m_name == "FixedCurve_PU":
                    call_kw["survival_estimator"] = "fixed_90d"
                elif m_name == "Teacher_Only_Exact":
                    call_kw["teacher_mode"] = "teacher_only_exact"
                elif m_name == "Teacher_Only_Matched":
                    call_kw["teacher_mode"] = "teacher_only_matched"
                elif m_name == "Survival_PU_WithGate":
                    call_kw["use_legacy_gate_ablation"] = True

                llm_sl = None
                if llm_commit_probs_stream is not None:
                    llm_sl = llm_commit_probs_stream[w_st - stream_start : w_ed - stream_start]

                cur_model, diag = adapt_survival_pu_v3(
                    current_model=cur_model,
                    anchor_model=anc_model,
                    X_hist=X[:w_ed],
                    commit_ts_hist=commit_ts[:w_ed],
                    fix_ts_hist=fix_ts[:w_ed],
                    y_obs_hist=y_obs_al,
                    drift_indices_in_hist=np.arange(w_st, w_ed),
                    t_alarm=t_al,
                    pos_weight=pw_base,
                    anchor_prevalence=anchor_prev,
                    feature_slice_for_density=dens_slice,
                    llm_drift_prior_probs=llm_sl,
                    seed=step_seed,
                    **call_kw,
                )
                comp_m.add(diag.compute)
                nan_rollbacks += int(diag.nan_rolled_back)
                const_alarms += int(diag.is_constant_pred)
                surv_fallbacks += int(diag.survival_fallback_used)
                bis_unreach += int(not diag.bisection_reachable)

            w_adapt_ms = (time.perf_counter() - t_up0) * 1000.0
            adapt_ms_total += w_adapt_ms
            per_alarm_adapt_ms.append(w_adapt_ms)

            if closed_loop_embeddings and cl_mon is not None:
                t_wmr0 = time.perf_counter()
                ref_sl = slice(max(0, w_ed - 3 * window_size), w_ed)
                _, z_ref_new = predict_logits_and_embeddings_v3(cur_model, X[ref_sl], device=device)
                cl_mon.update_reference(z_ref_new, window_size=window_size)
                w_mon_ms += (time.perf_counter() - t_wmr0) * 1000.0

            mon_ms_total += w_mon_ms
            fired_window_step_records.append((w_idx, w_st, w_ed, w_pred_ms, w_mon_ms, w_adapt_ms))

        t_ps0 = time.perf_counter()
        p_stream = sigmoid_safe(z_stream)
        pred_ms_total += (time.perf_counter() - t_ps0) * 1000.0

        cal_p, th_vec, cal_inner_ms, per_win_cal_ms = _calibrate_with_window_timing(
            past_init_z, z_stream, len(shared_windows)
        )
        t_dc0 = time.perf_counter()
        dec_m = (p_stream >= th_vec).astype(np.int32)
        dec_ms = (time.perf_counter() - t_dc0) * 1000.0
        cal_ms = cal_inner_ms + dec_ms
        dec_share_per_win = dec_ms / max(len(shared_windows), 1)

        # Standalone vs Incremental dependency classification (`R3c-3`, `R3d-2`):
        # - `own_init_fit_ms` holds ONLY initialization performed specifically for `m_name`
        #   (`init_hind_ms` for `B1a`, `odasc_init_ms` for `ODaSC_Host`, and `cl_mon_fit_ms` when closed-loop).
        # - Reused precomputed `m_causal_init` (`shared_anchor_init_total_ms`) and open-loop `shared_mon_fit_ms`
        #   are classified under `shared_anchor_init_ms` / `shared_monitor_init_ms`, so they are included in
        #   standalone totals (`init_fit_ms`, `consistent_end_to_end_ms`) and excluded from `incremental_suite_*`!
        sh_anc_eval_ms_m = 0.0 if closed_loop_embeddings else float(b0_eval_pred_ms)
        cl_init_own = float(cl_mon_fit_ms) if closed_loop_embeddings else 0.0
        own_init_ms_m = (
            float(init_hind_ms + odasc_init_ms + cl_init_own)
            if m_name == "B1a_Full_Hindsight"
            else float(odasc_init_ms + cl_init_own)
        )
        sh_anc_init_ms_m = shared_anchor_init_total_ms
        sh_mon_init_ms_m = 0.0 if closed_loop_embeddings else float(shared_mon_fit_ms)

        per_alarm_step_total_ms = [
            (sh_anc_eval_ms_m * (we - ws) / max(n_eval, 1))
            + wp
            + wm
            + wa
            + (per_win_cal_ms[wi] if wi < len(per_win_cal_ms) else 0.0)
            + dec_share_per_win
            for (wi, ws, we, wp, wm, wa) in fired_window_step_records
        ]

        tb_m = _pack_timing_breakdown(
            init_fit_ms=own_init_ms_m,
            stream_predict_and_embed_ms=pred_ms_total,
            drift_monitor_ms=mon_ms_total,
            adaptation_total_ms=adapt_ms_total,
            calibration_ms=cal_ms,
            shared_anchor_init_ms=sh_anc_init_ms_m,
            shared_monitor_init_ms=sh_mon_init_ms_m,
            shared_anchor_eval_embed_ms=sh_anc_eval_ms_m,
            incremental_monitor_ms=mon_ms_total if closed_loop_embeddings else 0.0,
        )
        n_scored_wins = max(sum(1 for (ws, we, _, _) in shared_windows if (we - ws) >= max(window_size // 3, 50)), 1)
        method_outputs[m_name] = {
            "raw_logits": z_stream,
            "raw_probs": p_stream,
            "cal_probs": cal_p,
            "tuned_th_vec": th_vec,
            "decisions_th": dec_m,
            "n_alarms": n_alarms_fired,
            "alarm_rate": float(n_alarms_fired / n_scored_wins),
            "nan_rollbacks": nan_rollbacks,
            "constant_output_alarms": const_alarms,
            "survival_fallbacks": surv_fallbacks,
            "bisection_unreachable": bis_unreach,
            "per_alarm_wall_ms": per_alarm_adapt_ms,
            "per_alarm_step_total_ms": per_alarm_step_total_ms,
            "end_to_end_eval_ms": tb_m["consistent_stream_eval_ms"],
            "timing_breakdown_ms": tb_m,
            "executed_alarm_windows": executed_alarm_windows,
            "compute": comp_m,
        }

    return {
        "stream_start": stream_start,
        "init_compute": init_comp,
        "init_drift_threshold": init_drift_th,
        "shared_anchor_init_ms": shared_anchor_init_total_ms,
        "shared_anchor_eval_embed_ms": float(b0_eval_pred_ms),
        "shared_monitor_ms": shared_monitor_ms,
        "shared_monitor_fit_ms": float(shared_mon_fit_ms),
        "shared_alarm_windows": [
            {"start": int(ws), "end": int(we), "fired": bool(fi), "distance": float(di)}
            for (ws, we, fi, di) in shared_windows
        ],
        "local_fork_summary": local_fork_summary,
        "methods": method_outputs,
    }
