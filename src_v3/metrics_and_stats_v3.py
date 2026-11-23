"""
Endpoint Evaluation, Causal Per-Commit Thresholding, Platt Calibration, LOC-Masking Diagnostics,
and Exact Subset-Sum Signed-Rank Donor-Cluster Inference (N05 V3).

Closes Codex Round 3 R3-C1, R3-C8, and R3-M5:
1. `evaluate_horizon_v3` consumes causal per-commit `per_commit_thresholds` (`shape [N]`, NEVER a
   stream-averaged scalar!), guaranteeing strict prefix-invariance of `f1_th`, `gmean_th`, and
   binary classification decisions.
2. `windowed_batch_recall_at_20pct_effort` accurately names end-of-window batch effort allocation;
   `spearman_with_inv_loc` returns `NaN` when density has zero variance (never imputing `1.0`),
   while `exact_tie_fraction` uses exact IEEE-754 equality alongside `constant_density_flag`.
3. Donor-cluster aggregation & paired inference enforce 100% completeness (`ValueError` on any
   missing stream/cluster/method), orient lower-is-better calibration metrics (`brier_cal`, `ece_cal`)
   accurately via `higher_is_better=False`, and compute 100% EXACT two-sided signed-rank p-values
   for BOTH `pratt` and `wilcox` tie conventions via exact integer `2 * midrank` subset-sum DP
   (eliminating normal approximations at K=4, K=5, K=24, and K=25!).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def sigmoid_safe(z: np.ndarray) -> np.ndarray:
    z_c = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_c))


def prob_to_logit_safe(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_c = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def fit_causal_platt_calibrator(
    cal_logits: np.ndarray,
    cal_y_obs: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Consistent 1D Causal Platt Scaling `sigma(a * z + b)` fit strictly on a causal mature slice
    (`age >= wait_days_mature` with real-time `y_obs(t)`).
    """
    z = np.asarray(cal_logits, dtype=np.float64).reshape(-1, 1)
    y = (np.asarray(cal_y_obs, dtype=np.float64) > 0.5).astype(np.int32)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if len(y) < 30 or n_pos < 3 or n_neg < 3 or float(np.std(z)) < 1e-6:
        p_emp = float(np.clip((n_pos + 1.0) / (len(y) + 2.0), 0.01, 0.99))
        mean_z = float(np.mean(z)) if len(z) > 0 else 0.0
        b_shift = float(np.log(p_emp / (1.0 - p_emp))) - mean_z
        return 1.0, b_shift

    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        lr.fit(z, y, sample_weight=sample_weights)
        a = float(np.clip(lr.coef_[0, 0], 0.05, 5.0))
        b = float(np.clip(lr.intercept_[0], -10.0, 10.0))
        return a, b
    except Exception:
        return 1.0, 0.0


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid_safe(float(a) * np.asarray(logits, dtype=np.float64) + float(b))


def tune_f1_threshold_on_mature_slice(
    probs: np.ndarray,
    y_obs: np.ndarray,
    default_th: float = 0.50,
) -> float:
    """Tune binary decision threshold strictly on a disjoint causal mature validation slice."""
    p = np.asarray(probs, dtype=np.float64)
    y = (np.asarray(y_obs, dtype=np.float64) > 0.5).astype(np.int32)
    if len(y) < 40 or int(np.sum(y == 1)) < 3 or int(np.sum(y == 0)) < 3:
        return float(default_th)

    grid = np.unique(np.concatenate([np.linspace(0.10, 0.90, 41), np.quantile(p, np.linspace(0.05, 0.95, 19))]))
    grid = np.clip(grid, 0.05, 0.95)
    best_th = float(default_th)
    best_f1 = -1.0
    for th in grid:
        pred = (p >= th).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1 + 1e-9:
            best_f1 = f1
            best_th = float(th)
    return best_th


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y))
    for k in range(n_bins):
        if k < n_bins - 1:
            m = (p >= bins[k]) & (p < bins[k + 1])
        else:
            m = (p >= bins[k]) & (p <= bins[k + 1])
        cnt = int(np.sum(m))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return float(ece)


def compute_effort_and_loc_masking_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    raw_logits: np.ndarray,
    effort: np.ndarray,
    budget_frac: float = 0.20,
    batch_window_size: int = 400,
) -> Dict[str, float]:
    """
    Compute retrospective full-horizon & windowed-batch 20%-effort recall plus exact diagnostics
    for when risk-density `y_prob / effort` collapses toward trivial `1 / effort` (`LOC_Only`).
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    z = np.asarray(raw_logits, dtype=np.float64)
    eff = np.maximum(np.asarray(effort, dtype=np.float64), 1.0)

    total_bugs = float(np.sum(y))
    total_eff = float(np.sum(eff))
    if len(y) == 0 or total_eff <= 0:
        return {
            "recall_at_20pct_effort": float("nan"),
            "windowed_batch_recall_at_20pct_effort": float("nan"),
            "popt": float("nan"),
            "ifa": float("nan"),
            "spearman_with_inv_loc": float("nan"),
            "top20_jaccard_with_loc_only": float("nan"),
            "logit_iqr": float("nan"),
            "prob_std": float("nan"),
            "constant_density_flag": float("nan"),
            "exact_tie_fraction": float("nan"),
        }

    density = p / eff
    inv_loc = 1.0 / eff

    # Deterministic tie-breaking: higher score first, smaller effort first
    order = np.lexsort((eff, -density))
    order_loc = np.lexsort((eff, -inv_loc))

    def _fractional_bugs(ord_idx: np.ndarray, y_vec: np.ndarray, e_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        sy = y_vec[ord_idx]
        se = e_vec[ord_idx]
        ce = np.cumsum(se)
        cb = np.cumsum(sy)
        bgt = budget_frac * float(np.sum(e_vec))
        within = ce <= bgt
        nw = int(np.sum(within))
        bf = float(cb[nw - 1]) if nw > 0 else 0.0
        if nw < len(se):
            rem = bgt - (ce[nw - 1] if nw > 0 else 0.0)
            if rem > 0 and se[nw] > 0:
                bf += float(sy[nw]) * (rem / float(se[nw]))
        sel = ord_idx[: max(nw, 1)]
        return bf, sel

    bugs_found, sel_model = _fractional_bugs(order, y, eff)
    _, sel_loc = _fractional_bugs(order_loc, y, eff)
    rec_20 = (bugs_found / total_bugs) if total_bugs > 0 else float("nan")

    # End-of-window batch 20%-effort budget allocation (no lookahead beyond each batch window `W`!)
    batch_bugs = 0.0
    for st in range(0, len(y), max(batch_window_size, 50)):
        ed = min(st + max(batch_window_size, 50), len(y))
        yw, pw, ew = y[st:ed], p[st:ed], eff[st:ed]
        ord_w = np.lexsort((ew, -(pw / ew)))
        bf_w, _ = _fractional_bugs(ord_w, yw, ew)
        batch_bugs += bf_w
    batch_rec_20 = (batch_bugs / total_bugs) if total_bugs > 0 else float("nan")

    def _area(ord_idx: np.ndarray) -> float:
        en = np.concatenate(([0.0], np.cumsum(eff[ord_idx]) / total_eff))
        bn = np.concatenate(([0.0], np.cumsum(y[ord_idx]) / total_bugs)) if total_bugs > 0 else np.zeros_like(en)
        return float(np.trapezoid(bn, en)) if hasattr(np, "trapezoid") else float(np.trapz(bn, en))

    if total_bugs > 0:
        opt_o = np.lexsort((eff, -(y / eff)))
        wst_o = np.lexsort((-eff, (y / eff)))
        a_m, a_opt, a_wst = _area(order), _area(opt_o), _area(wst_o)
        popt = float(np.clip((a_m - a_wst) / max(a_opt - a_wst, 1e-12), 0.0, 1.0))
        pos_idx = np.where(y[order] > 0.5)[0]
        ifa = float(pos_idx[0]) if len(pos_idx) > 0 else float(len(y))
    else:
        popt = float("nan")
        ifa = float("nan")

    # Codex R3-M5: Constant density has undefined Spearman (`NaN`); report `constant_density_flag` & exact float ties separately!
    is_const_dens = bool(np.std(density) == 0.0)
    if is_const_dens or np.std(inv_loc) == 0.0:
        sp_corr = float("nan")
    else:
        sp_val = scipy.stats.spearmanr(density, inv_loc).statistic
        sp_corr = float(sp_val) if np.isfinite(sp_val) else float("nan")

    set_m = set(int(i) for i in sel_model)
    set_l = set(int(i) for i in sel_loc)
    jacc = float(len(set_m & set_l) / max(len(set_m | set_l), 1))

    fin_z = z[np.isfinite(z)]
    logit_iqr = float(np.percentile(fin_z, 75) - np.percentile(fin_z, 25)) if len(fin_z) > 0 else 0.0
    prob_std = float(np.std(p))
    sorted_dens = density[order]
    exact_tie_frac = float(np.mean(sorted_dens[:-1] == sorted_dens[1:])) if len(sorted_dens) > 1 else 0.0

    return {
        "recall_at_20pct_effort": float(np.clip(rec_20, 0.0, 1.0)) if np.isfinite(rec_20) else float("nan"),
        "windowed_batch_recall_at_20pct_effort": float(np.clip(batch_rec_20, 0.0, 1.0)) if np.isfinite(batch_rec_20) else float("nan"),
        "popt": popt,
        "ifa": ifa,
        "spearman_with_inv_loc": sp_corr,
        "top20_jaccard_with_loc_only": jacc,
        "logit_iqr": logit_iqr,
        "prob_std": prob_std,
        "constant_density_flag": float(is_const_dens),
        "exact_tie_fraction": exact_tie_frac,
    }


def _binary_cls_stats_vec(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Union[float, np.ndarray],
    decisions_override: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if decisions_override is not None:
        y_pred = np.asarray(decisions_override, dtype=np.int32)
    else:
        th_arr = np.asarray(thresholds, dtype=np.float64)
        y_pred = (y_prob >= th_arr).astype(np.int32)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    gmean = float(np.sqrt(max(tpr * tnr, 0.0)))
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "gmean": gmean,
        "pred_pos_rate": float(np.mean(y_pred)) if len(y_pred) > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_horizon_v3(
    y_true: np.ndarray,
    raw_logits: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: Optional[np.ndarray],
    effort: np.ndarray,
    per_commit_thresholds: Optional[Union[float, np.ndarray]] = None,
    per_commit_decisions: Optional[np.ndarray] = None,
    batch_window_size: int = 400,
) -> Dict[str, float]:
    """
    Comprehensive V3 horizon evaluator with Average Precision (`ap`) as pre-declared primary endpoint.
    Strictly evaluates `f1_th` using causal per-commit `per_commit_thresholds` (`shape [N]`),
    NEVER a retrospective stream-averaged threshold (Codex R3-C1)!
    """
    y = np.asarray(y_true, dtype=np.int32)
    z = np.asarray(raw_logits, dtype=np.float64)
    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 0.0, 1.0)
    p_cal = np.clip(np.asarray(cal_probs if cal_probs is not None else p, dtype=np.float64), 0.0, 1.0)
    eff = np.maximum(np.asarray(effort, dtype=np.float64), 1.0)

    if len(y) == 0:
        return {"n_commits": 0, "n_defective": 0, "n_clean": 0, "ap": float("nan"), "roc_auc": float("nan"), "f1": float("nan")}

    if per_commit_thresholds is None:
        th_vec = np.full(len(y), 0.50, dtype=np.float64)
    else:
        th_in = np.asarray(per_commit_thresholds, dtype=np.float64)
        if th_in.ndim == 0:
            th_vec = np.full(len(y), float(th_in), dtype=np.float64)
        else:
            assert len(th_in) == len(y), f"Threshold vector length {len(th_in)} != y length {len(y)}!"
            th_vec = th_in

    if per_commit_decisions is not None:
        assert len(per_commit_decisions) == len(y), "per_commit_decisions length mismatch!"

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    has_both = (n_pos > 0) and (n_neg > 0)

    ap = float(average_precision_score(y, p)) if has_both else float("nan")
    roc_auc = float(roc_auc_score(y, p)) if has_both else float("nan")

    cls_05 = _binary_cls_stats_vec(y, p, thresholds=0.50)
    cls_th = _binary_cls_stats_vec(y, p, thresholds=th_vec, decisions_override=per_commit_decisions)
    eff_d = compute_effort_and_loc_masking_metrics(y, p, z, eff, budget_frac=0.20, batch_window_size=batch_window_size)

    brier_raw = float(brier_score_loss(y, p))
    brier_cal = float(brier_score_loss(y, p_cal))
    ece_cal = compute_ece(y, p_cal, n_bins=10)

    return {
        "n_commits": int(len(y)),
        "n_defective": n_pos,
        "n_clean": n_neg,
        "prevalence": float(np.mean(y)),
        "ap": ap,
        "pr_auc": ap,
        "roc_auc": roc_auc,
        "f1": cls_05["f1"],
        "precision": cls_05["precision"],
        "recall": cls_05["recall"],
        "gmean": cls_05["gmean"],
        "pred_pos_rate": cls_05["pred_pos_rate"],
        "f1_th": cls_th["f1"],
        "gmean_th": cls_th["gmean"],
        "pred_pos_rate_th": cls_th["pred_pos_rate"],
        "applied_threshold_mean_diagnostic": float(np.mean(th_vec)),
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "ece_cal": ece_cal,
        **eff_d,
    }


REQUIRED_TWO_CLASS_HORIZON_METRICS: Tuple[str, ...] = (
    "ap",
    "roc_auc",
    "f1",
    "f1_th",
    "gmean",
    "brier_raw",
    "brier_cal",
    "ece_cal",
    "recall_at_20pct_effort",
    "windowed_batch_recall_at_20pct_effort",
    "popt",
    "ifa",
)


def aggregate_seed_horizon_metrics(
    dlist: List[Dict],
    stream_name: str,
    horizon_name: str = "full_stream",
) -> Dict[str, Optional[float]]:
    """
    Strict Multi-Seed Horizon Aggregator (`R3b-M3`):
      1. Verifies `len(dlist) >= 1` and identical `(n_commits, n_defective)` across all seeds;
      2. Reads canonical `n_defective` (NOT non-existent `n_pos`!) to determine `has_two_classes`;
      3. Whenever `has_two_classes` is `True`, raises `ValueError` if ANY seed has missing/`None`/`NaN`
         on any metric in `REQUIRED_TWO_CLASS_HORIZON_METRICS` (preventing `[0.5, NaN]` from silently
         becoming `mean=0.5, std=0.0`!).
    """
    if not dlist:
        raise ValueError(f"Empty seed metric list on {stream_name}:{horizon_name}!")

    n_com_set = {int(d.get("n_commits", -1)) for d in dlist}
    n_def_set = {int(d.get("n_defective", -1)) for d in dlist}
    if len(n_com_set) != 1 or (-1 in n_com_set) or len(n_def_set) != 1 or (-1 in n_def_set):
        raise ValueError(
            f"Inconsistent or missing (n_commits, n_defective) across seeds on {stream_name}:{horizon_name}: "
            f"n_commits={n_com_set}, n_defective={n_def_set}"
        )

    n_com = next(iter(n_com_set))
    n_def = next(iter(n_def_set))
    n_cln = n_com - n_def
    has_two_classes = (n_def > 0) and (n_cln > 0)

    keys = list(dlist[0].keys())
    if has_two_classes:
        for req_k in REQUIRED_TWO_CLASS_HORIZON_METRICS:
            if req_k not in keys:
                raise KeyError(f"Missing required metric '{req_k}' on {stream_name}:{horizon_name}!")

    out: Dict[str, Optional[float]] = {}
    for k in keys:
        raw_vals = [d.get(k) for d in dlist]
        if has_two_classes and k in REQUIRED_TWO_CLASS_HORIZON_METRICS:
            if any(v is None or (not np.isfinite(float(v))) for v in raw_vals):
                raise ValueError(
                    f"Missing or NaN required two-class metric '{k}' across seeds on "
                    f"{stream_name}:{horizon_name}: {raw_vals}"
                )
        fin_vals = [float(v) for v in raw_vals if v is not None and np.isfinite(float(v))]
        out[k] = float(np.mean(fin_vals)) if fin_vals else None
        if k in ("ap", "roc_auc", "f1", "f1_th", "brier_cal", "recall_at_20pct_effort"):
            out[f"{k}_std"] = float(np.std(fin_vals)) if len(fin_vals) > 1 else 0.0
    return out


def evaluate_window_trajectories_v3(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    effort: np.ndarray,
    window_size: int = 400,
) -> Dict:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(raw_probs, dtype=np.float64)

    win_records = []
    n_single_class = 0
    for w_idx, st in enumerate(range(0, len(y), window_size)):
        ed = min(st + window_size, len(y))
        if ed - st < 25:
            continue
        yw, pw = y[st:ed], p[st:ed]
        n_pos = int(np.sum(yw == 1))
        n_neg = int(np.sum(yw == 0))
        if n_pos > 0 and n_neg > 0:
            ap_w = float(average_precision_score(yw, pw))
            auc_w = float(roc_auc_score(yw, pw))
        else:
            ap_w = None
            auc_w = None
            n_single_class += 1
        cls_w = _binary_cls_stats_vec(yw, pw, thresholds=0.50)
        win_records.append({
            "window_idx": w_idx,
            "start_offset": st,
            "end_offset": ed,
            "n_pos": n_pos,
            "ap": ap_w,
            "roc_auc": auc_w,
            "f1": cls_w["f1"],
            "recall": cls_w["recall"],
        })

    defined_aps = [r["ap"] for r in win_records if r["ap"] is not None]
    return {
        "n_windows": len(win_records),
        "n_single_class_windows": n_single_class,
        "mean_defined_window_ap": float(np.mean(defined_aps)) if defined_aps else float("nan"),
        "windows": win_records,
    }


# ==============================================================================
# Exact Subset-Sum Signed-Rank Enumeration & Strict Donor-Cluster Inference
# ==============================================================================

def exact_signed_rank_two_sided_pvalue(diff: np.ndarray, zero_method: str = "pratt") -> float:
    """
    100% Exact Two-Sided Signed-Rank Permutation p-value via Integer `2 * midrank` Subset-Sum DP
    (Codex R3-C8):
      - Because the average of any `m` consecutive integer ranks `s .. s + m - 1` is `s + (m - 1)/2`,
        `2 * midrank` is ALGEBRAICALLY AN EXACT INTEGER for every possible tie configuration!
      - Under `zero_method="pratt"`, zero differences (`diff == 0`) participate in midrank assignment
        and remain fixed at sign `0` across all `2^{K_nz}` sign flips of non-zero pairs.
      - Under `zero_method="wilcox"`, zero differences are excluded before midrank assignment.
      - Computes the exact proportion of the `2^{K_nz}` equiprobable sign assignments whose centered
        signed-rank magnitude `|2 * W_+ - W_total|` is `>= |2 * W_obs_+ - W_total|`.
    """
    d = np.asarray(diff, dtype=np.float64)
    nz_mask = d != 0.0
    n_nz = int(np.sum(nz_mask))
    if n_nz == 0:
        return 1.0

    if zero_method == "pratt":
        ranks_all = scipy.stats.rankdata(np.abs(d), method="average")
        two_r = np.rint(2.0 * ranks_all[nz_mask]).astype(np.int64)
        signs_pos = d[nz_mask] > 0.0
    elif zero_method == "wilcox":
        d_nz = d[nz_mask]
        ranks_nz = scipy.stats.rankdata(np.abs(d_nz), method="average")
        two_r = np.rint(2.0 * ranks_nz).astype(np.int64)
        signs_pos = d_nz > 0.0
    else:
        raise ValueError(f"Unknown zero_method={zero_method}")

    w_total = int(np.sum(two_r))
    w_obs_pos = int(np.sum(two_r[signs_pos]))
    obs_dev = abs(2 * w_obs_pos - w_total)

    # Exact subset-sum count over `w in [0 .. w_total]` in float64 (exact up to 2^53 ~ 9e15, i.e., K_nz <= 52!)
    if n_nz <= 50:
        dp = np.zeros(w_total + 1, dtype=np.float64)
        dp[0] = 1.0
        for w_i in two_r:
            wi = int(w_i)
            dp[wi:] += dp[: w_total + 1 - wi]
        all_sums = np.arange(w_total + 1, dtype=np.int64)
        extreme = np.abs(2 * all_sums - w_total) >= obs_dev
        p_val = float(np.sum(dp[extreme]) / (2.0 ** n_nz))
        return float(np.clip(p_val, 0.0, 1.0))

    # Fallback only if K_nz > 50
    res = scipy.stats.wilcoxon(d, zero_method=zero_method, alternative="two-sided", mode="approx")
    return float(res.pvalue)


def aggregate_streams_to_donor_clusters(
    per_stream_method_scores: Dict[str, Dict[str, float]],
    stream_to_cluster: Dict[str, str],
    exclude_clusters: Tuple[str, ...] = (),
    expected_methods: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Strict Two-Stage Donor-Cluster Aggregation (Codex R3-M5):
      - Verifies every expected stream in `stream_to_cluster` (outside `exclude_clusters`) is present;
      - Verifies every expected method is present and finite on every stream (raising `ValueError`
        instead of silently skipping missing streams/clusters/methods!).
    """
    active_stream_map = {s: c for s, c in stream_to_cluster.items() if c not in exclude_clusters}
    missing_streams = sorted(s for s in active_stream_map if s not in per_stream_method_scores)
    if missing_streams:
        raise ValueError(f"Missing streams in donor-cluster aggregation: {missing_streams}")

    clusters = sorted(set(active_stream_map.values()))
    if expected_methods is not None:
        all_methods = list(expected_methods)
    else:
        first_s = next(iter(active_stream_map))
        all_methods = sorted(per_stream_method_scores[first_s].keys())

    cluster_arrays: Dict[str, List[float]] = {m: [] for m in all_methods}
    for cl in clusters:
        member_streams = sorted(s for s, dc in active_stream_map.items() if dc == cl)
        for m in all_methods:
            vals = []
            for s in member_streams:
                if m not in per_stream_method_scores[s]:
                    raise KeyError(f"Missing method '{m}' in stream '{s}'!")
                v = float(per_stream_method_scores[s][m])
                if not np.isfinite(v):
                    raise ValueError(f"Non-finite endpoint value {v} for method '{m}' on stream '{s}'!")
                vals.append(v)
            cluster_arrays[m].append(float(np.mean(vals)))

    return clusters, {m: np.array(v, dtype=np.float64) for m, v in cluster_arrays.items()}


def paired_cluster_inference_v3(
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    higher_is_better: bool = True,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Strict Paired Cluster-Level Inference (`raw_diff = A - B`, `benefit_diff = +(A - B)` if
    `higher_is_better` else `-(A - B)` for Brier/ECE; Codex R3-C8 & R3-M5):
      - Rejects any `NaN`/`Inf` or length mismatch (`ValueError`);
      - Computes 100% EXACT two-sided signed-rank p-values for BOTH `pratt` and `wilcox`!
    """
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError(f"Invalid paired cluster array lengths: {len(a)} vs {len(b)}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Non-finite values encountered in paired_cluster_inference_v3!")

    k = int(len(a))
    raw_diff = a - b
    benefit_diff = raw_diff if higher_is_better else (-raw_diff)

    ties = benefit_diff == 0.0
    n_ties = int(np.sum(ties))
    wins = int(np.sum(benefit_diff > 0.0))
    losses = int(np.sum(benefit_diff < 0.0))

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_raw_diff = float(np.mean(raw_diff))
    median_raw_diff = float(np.median(raw_diff))

    p_pratt = exact_signed_rank_two_sided_pvalue(benefit_diff, zero_method="pratt")
    p_wilcox = exact_signed_rank_two_sided_pvalue(benefit_diff, zero_method="wilcox")

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, k, size=(n_boot, k))
    boot_means = np.mean(raw_diff[idx], axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    sd_b = float(np.std(benefit_diff, ddof=1)) if k > 1 else 0.0
    mean_b_diff = float(np.mean(benefit_diff))
    if sd_b > 1e-12:
        cohen_dz: Optional[float] = float(mean_b_diff / sd_b)
        cohen_dz_undef = False
    elif abs(mean_b_diff) <= 1e-12:
        cohen_dz = 0.0
        cohen_dz_undef = False
    else:
        cohen_dz = None
        cohen_dz_undef = True

    nz = benefit_diff[~ties]
    n_nz = int(len(nz))
    if n_nz > 0:
        ranks = scipy.stats.rankdata(np.abs(nz), method="average")
        r_pos = float(np.sum(ranks[nz > 0.0]))
        r_neg = float(np.sum(ranks[nz < 0.0]))
        rank_biserial = (r_pos - r_neg) / max(r_pos + r_neg, 1.0)
    else:
        rank_biserial = 0.0

    min_p_exact = float(2.0 / (2.0 ** n_nz)) if (0 < n_nz <= 30) else (1.0 if n_nz == 0 else 0.0)

    return {
        "K": k,
        "K_nonzero": n_nz,
        "higher_is_better": bool(higher_is_better),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_raw_diff,
        "median_diff": median_raw_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "wins": wins,
        "ties": n_ties,
        "losses": losses,
        "min_attainable_exact_two_sided_p": min_p_exact,
        "p_wilcoxon_pratt": p_pratt,
        "p_wilcoxon_wilcox": p_wilcox,
        "cohen_dz": cohen_dz,
        "cohen_dz_undefined_zero_sd": cohen_dz_undef,
        "rank_biserial": float(rank_biserial),
    }


def run_holm_bonferroni_family(
    cluster_scores: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    family_name: str,
    higher_is_better: bool = True,
    primary_zero_method: str = "pratt",
) -> List[Dict]:
    """
    Strict Step-Down Holm-Bonferroni Correction Across a Pre-Declared Family (Codex R3-M5):
    Raises `KeyError` if ANY method in `pairs` is missing from `cluster_scores`!
    """
    raw_rows = []
    p_key = "p_wilcoxon_pratt" if primary_zero_method == "pratt" else "p_wilcoxon_wilcox"
    for m_a, m_b in pairs:
        if m_a not in cluster_scores:
            raise KeyError(f"Missing method '{m_a}' for Holm family '{family_name}'!")
        if m_b not in cluster_scores:
            raise KeyError(f"Missing method '{m_b}' for Holm family '{family_name}'!")
        st = paired_cluster_inference_v3(cluster_scores[m_a], cluster_scores[m_b], higher_is_better=higher_is_better)
        st["family"] = family_name
        st["method_a"] = m_a
        st["method_b"] = m_b
        raw_rows.append(st)

    m_tot = len(raw_rows)
    if m_tot == 0:
        return []

    order = np.argsort([r[p_key] for r in raw_rows])
    running_max = 0.0
    adjusted = [1.0] * m_tot
    for rank_idx, orig_i in enumerate(order):
        mult = m_tot - rank_idx
        p_adj = min(1.0, raw_rows[orig_i][p_key] * mult)
        running_max = max(running_max, p_adj)
        adjusted[orig_i] = running_max

    for i, r in enumerate(raw_rows):
        r["p_holm"] = float(adjusted[i])
        r["significant_005_holm"] = bool(adjusted[i] < 0.05)
    return raw_rows
