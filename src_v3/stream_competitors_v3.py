"""
Stream & Simple Baseline Competitors with Documented Adaptations & Phase-Decomposed Timing (N05 V3 Track 2).

Closes Codex Round 3 & Round 3b Findings (`R3-C1`, `R3-C3`, `R3b-C1`, `R3b-C2`, `R3b-M4`, `R3b-M6`):
1. `calibrate_causal_stream_probs_strict` (`R3-C1`):
   - Cold-start uses ONLY `past_init_logits` from `[0 : stream_start]` (zero lookahead into `z_slice`).
   - Returns causal per-commit `tuned_th_vec` (`shape [N_eval]`).
2. `ORB_OHT_Native` (`R3b-C2`, `R3b-M4`):
   - Implements Cabral et al. (ICSE 2019, Alg. 1 + `jitsdp/orb.py` & `baseline.py`) on `river 0.27`
     `HoeffdingTreeClassifier`s (`M = 10`, independent `Poisson(lambda) * obf` per tree as in Alg. 1 line 12;
     note `jitsdp/baseline.py` alternatively wraps `OzaBagging(Poisson(1))` around a single `ORB` draw).
   - Consumes ONLY events with `timestamp_event < commit_ts[i]` (`right=False`, matching `baseline.py:121`),
     guaranteeing zero-latency / simultaneous-timestamp future commits NEVER leak into commit `i` while
     deferring simultaneous events cleanly without dropping them (`R3b-C2`).
   - Uses exact lossless float feature tuples `tuple(row.tolist())` (zero decimal rounding!), runs the
     duplicate-feature noise filter (`clean_count > noise_o`) BEFORE `p1` updates (`baseline.py:191-197`),
     and draws `rng.poisson(lam)` directly without artificial `min(lam, 50)` clipping (`R3b-M4`).
3. `BORB_Native` & `BORB_Alarm` (`R3b-C1`, `R3b-M4`):
   - Implements Cabral et al. (EMSE 2023, Alg. 2-3 + `jitsdp/pipeline.py` 20-iteration inner `ma`/`OBF`/
     `skewedSample` loop + `RateFixed._tune_threshold` causal rolling quantile threshold), explicitly
     adapted to standardized multi-cohort tabular streams with `ElasticNet SGDClassifier(loss='log_loss')`
     (whereas EMSE 2023 used `MLPMask`).
   - In `update_mode == "alarm"`, strictly predicts `[c_st : c_ed]` FIRST, and retrains on `[:c_ed]`
     AFTER the window only when `fired` is `True` (`R3b-C1`: zero within-window alarm lookahead!).
4. `ODaSC_OHT_Native` (`R3b-C2`, `R3b-M4`):
   - Specification-level implementation of Song et al. (IJCNN 2022, Eqs. 1-12; no official author code
     was published), operating over ALL `D` features (zero 24-feature truncation!), consuming strictly
     `timestamp_event < commit_ts[i]`, and enforcing `new_r <= eps_radius` strictly on every DenStream
     micro-cluster merge (evicting the lowest-`CT` cluster when `max_mc = 250` is reached rather than
     violating `eps_radius`!).
5. Unified Phase-Decomposed Timing (`R3b-M6`):
   - Every competitor returns `"timing_breakdown_ms"` (`init_fit_ms`, `stream_predict_and_embed_ms`,
     `drift_monitor_ms`, `adaptation_total_ms`, `calibration_ms`, `consistent_stream_eval_ms`,
     `consistent_end_to_end_ms`) aligned with Track 1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from river import tree as river_tree
from sklearn.linear_model import LogisticRegression, SGDClassifier

from config_v3 import SECONDS_PER_DAY
from metrics_and_stats_v3 import (
    apply_platt,
    fit_causal_platt_calibrator,
    prob_to_logit_safe,
    sigmoid_safe,
    tune_f1_threshold_on_mature_slice,
)


@dataclass
class StreamComputeStats:
    opt_steps: int = 0
    examples_seen: int = 0
    trees_built: int = 0
    requested_rounds: int = 0
    wall_ms: float = 0.0

    def add(self, other: "StreamComputeStats") -> None:
        self.opt_steps += other.opt_steps
        self.examples_seen += other.examples_seen
        self.trees_built += other.trees_built
        self.requested_rounds += other.requested_rounds
        self.wall_ms += other.wall_ms


def _pack_timing_breakdown(
    init_fit_ms: float,
    stream_predict_and_embed_ms: float,
    drift_monitor_ms: float,
    adaptation_total_ms: float,
    calibration_ms: float,
    shared_anchor_init_ms: float = 0.0,
    shared_monitor_init_ms: float = 0.0,
    shared_anchor_eval_embed_ms: float = 0.0,
    incremental_monitor_ms: float = 0.0,
) -> Dict[str, float]:
    """
    Complete Standalone vs Shared-Suite Timing Accounting (`R3b-M6`, `R3c-3`):
      - Standalone deployment totals account for EVERY upstream dependency required to deploy
        the method in isolation (`shared_anchor_init_ms`, `shared_monitor_init_ms`,
        `shared_anchor_eval_embed_ms`, `drift_monitor_ms`, own init/predict/adapt/calibration).
      - Incremental suite totals (`incremental_suite_*`) report the marginal execution cost inside
        a multi-method counterfactual suite after shared anchor/monitor precomputation.
    """
    own_init = float(init_fit_ms)
    sh_anc_init = float(shared_anchor_init_ms)
    sh_mon_init = float(shared_monitor_init_ms)
    standalone_init = float(own_init + sh_anc_init + sh_mon_init)

    own_pred = float(stream_predict_and_embed_ms)
    sh_anc_eval = float(shared_anchor_eval_embed_ms)
    total_pred_and_embed = float(own_pred + sh_anc_eval)

    mon = float(drift_monitor_ms)
    adapt = float(adaptation_total_ms)
    cal = float(calibration_ms)

    standalone_stream_eval = float(total_pred_and_embed + mon + adapt + cal)
    standalone_e2e = float(standalone_init + standalone_stream_eval)

    inc_eval = float(own_pred + float(incremental_monitor_ms) + adapt + cal)
    inc_e2e = float(own_init + inc_eval)

    return {
        "own_init_fit_ms": own_init,
        "shared_anchor_init_ms": sh_anc_init,
        "shared_monitor_init_ms": sh_mon_init,
        "init_fit_ms": standalone_init,
        "own_stream_predict_and_embed_ms": own_pred,
        "shared_anchor_eval_embed_ms": sh_anc_eval,
        "stream_predict_and_embed_ms": total_pred_and_embed,
        "drift_monitor_ms": mon,
        "adaptation_total_ms": adapt,
        "calibration_ms": cal,
        "consistent_stream_eval_ms": standalone_stream_eval,
        "consistent_end_to_end_ms": standalone_e2e,
        "incremental_suite_stream_eval_ms": inc_eval,
        "incremental_suite_end_to_end_ms": inc_e2e,
    }


def observe_labels_causal(
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    t_now: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c_ts = np.asarray(commit_ts, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.int32)
    f_ts = np.asarray(fix_ts, dtype=np.float64)
    valid = c_ts <= t_now
    revealed = valid & (y_t == 1) & np.isfinite(f_ts) & (f_ts <= t_now)
    return valid, revealed.astype(np.float32)


def calibrate_causal_stream_probs_strict(
    past_init_logits: np.ndarray,
    raw_logits_stream: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    stream_start: int,
    window_size: int = 400,
    wait_days_mature: float = 90.0,
    return_window_ms: bool = False,
    clock_fn=None,
):
    clk = clock_fn if clock_fn is not None else time.perf_counter
    n_eval = len(commit_ts) - stream_start
    cal_probs = np.zeros(n_eval, dtype=np.float64)
    tuned_th_vec = np.full(n_eval, 0.50, dtype=np.float64)
    per_win_cal_ms: List[float] = []

    t_setup0 = clk()
    init_z = np.asarray(past_init_logits, dtype=np.float64)
    init_med = float(np.median(init_z)) if len(init_z) > 0 else 0.0
    setup_ms = (clk() - t_setup0) * 1000.0

    win_starts = list(range(stream_start, len(commit_ts), window_size))
    share_setup = setup_ms / max(len(win_starts), 1)

    for st in win_starts:
        tw0 = clk()
        ed = min(st + window_size, len(commit_ts))
        t_now = float(commit_ts[st - 1])
        _, y_obs_now = observe_labels_causal(commit_ts[:st], y_true[:st], fix_ts[:st], t_now)
        ages_d = (t_now - commit_ts[:st]) / SECONDS_PER_DAY
        z_target = raw_logits_stream[st - stream_start : ed - stream_start]

        used_stream_mat = False
        if st > stream_start:
            past_eval_ages = ages_d[stream_start:st]
            mat_m = past_eval_ages >= wait_days_mature
            if int(np.sum(mat_m)) >= 80:
                z_cal = raw_logits_stream[: st - stream_start][mat_m][-1200:]
                y_cal = y_obs_now[stream_start:st][mat_m][-1200:]
                a_p, b_p = fit_causal_platt_calibrator(z_cal, y_cal)
                th_w = tune_f1_threshold_on_mature_slice(sigmoid_safe(z_cal), y_cal)
                cal_probs[st - stream_start : ed - stream_start] = apply_platt(z_target, a_p, b_p)
                tuned_th_vec[st - stream_start : ed - stream_start] = th_w
                used_stream_mat = True

        if not used_stream_mat:
            mat_init = ages_d[:stream_start] >= wait_days_mature
            if int(np.sum(mat_init)) >= 80 and len(init_z) == stream_start:
                z_ic = init_z[mat_init][-1200:]
                y_ic = y_obs_now[:stream_start][mat_init][-1200:]
                a_p0, b_p0 = fit_causal_platt_calibrator(z_ic, y_ic)
                th_0 = tune_f1_threshold_on_mature_slice(sigmoid_safe(z_ic), y_ic)
                cal_probs[st - stream_start : ed - stream_start] = apply_platt(z_target, a_p0, b_p0)
                tuned_th_vec[st - stream_start : ed - stream_start] = th_0
            else:
                y_init_slice = y_obs_now[:stream_start]
                base_pi = float(np.clip(np.mean(y_init_slice[mat_init]) if int(np.sum(mat_init)) >= 30 else np.mean(y_init_slice), 0.02, 0.65))
                shift = float(np.log(base_pi / (1.0 - base_pi))) - init_med
                cal_probs[st - stream_start : ed - stream_start] = sigmoid_safe(z_target + shift)
                tuned_th_vec[st - stream_start : ed - stream_start] = 0.50

        per_win_cal_ms.append(share_setup + (clk() - tw0) * 1000.0)

    if return_window_ms:
        return cal_probs, tuned_th_vec, per_win_cal_ms
    return cal_probs, tuned_th_vec


# ==============================================================================
# 1. Simple Linear & Effort Competitors (`LOC_Only`, `LApredict`, `Churn_LR`, `Full_LR`)
# ==============================================================================

def run_linear_competitor_v3(
    X_feat: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    effort: np.ndarray,
    stream_start: int,
    alarm_windows: List[Tuple[int, int, bool]],
    mode: str = "frozen",  # "loc_only", "frozen", "adapt_4w"
    c_reg: float = 0.5,
    window_size: int = 400,
    shared_anchor_init_ms: float = 0.0,
    shared_monitor_init_ms: float = 0.0,
    shared_anchor_eval_embed_ms: float = 0.0,
    shared_monitor_ms: float = 0.0,
    seed: int = 42,
) -> Dict:
    n_eval = len(commit_ts) - stream_start

    if mode == "loc_only":
        tp0 = time.perf_counter()
        eff_ev = np.maximum(np.asarray(effort[stream_start:], dtype=np.float64), 1.0)
        p_raw = np.clip(1.0 / (1.0 + 0.05 * eff_ev), 0.001, 0.999)
        z_raw = prob_to_logit_safe(p_raw)
        th_vec = np.full(n_eval, 0.50, dtype=np.float64)
        dec_vec = (p_raw >= th_vec).astype(np.int32)
        pred_ms = (time.perf_counter() - tp0) * 1000.0
        tb = _pack_timing_breakdown(0.0, pred_ms, 0.0, 0.0, 0.0)
        return {
            "raw_logits": z_raw,
            "raw_probs": p_raw,
            "cal_probs": p_raw,
            "tuned_th_vec": th_vec,
            "decisions_th": dec_vec,
            "n_updates": 0,
            "per_alarm_wall_ms": [],
            "per_alarm_step_total_ms": [],
            "end_to_end_eval_ms": tb["consistent_stream_eval_ms"],
            "timing_breakdown_ms": tb,
            "compute": StreamComputeStats(wall_ms=tb["consistent_end_to_end_ms"]),
        }

    t_init0 = time.perf_counter()
    X_arr = np.asarray(X_feat, dtype=np.float64)
    t_init = float(commit_ts[stream_start - 1])
    _, y_obs_init = observe_labels_causal(commit_ts[:stream_start], y_true[:stream_start], fix_ts[:stream_start], t_init)

    def _fit_lr(X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, float]:
        yb = (y_tr > 0.5).astype(np.int32)
        if int(np.sum(yb == 1)) >= 3 and int(np.sum(yb == 0)) >= 3 and np.any(np.ptp(X_tr, axis=0) > 1e-12):
            clf = LogisticRegression(C=c_reg, class_weight="balanced", solver="lbfgs", max_iter=300, random_state=seed)
            clf.fit(X_tr, yb)
            return clf.coef_[0].copy(), float(clf.intercept_[0])
        p0 = float(np.clip((np.sum(yb) + 1.0) / (len(yb) + 2.0), 0.01, 0.99))
        return np.zeros(X_tr.shape[1], dtype=np.float64), float(np.log(p0 / (1.0 - p0)))

    w_cur, b_cur = _fit_lr(X_arr[:stream_start], y_obs_init)
    z_init_past = (X_arr[:stream_start] @ w_cur) + b_cur
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    z_out = np.zeros(n_eval, dtype=np.float64)
    n_upd = 0
    opt_steps = 1
    ex_seen = stream_start
    pred_ms = 0.0
    adapt_ms = 0.0
    alarm_ms_list: List[float] = []
    fired_win_records: List[Tuple[int, int, int, float, float]] = []
    per_win_mon_ms = (float(shared_monitor_ms) / max(len(alarm_windows), 1)) if mode != "frozen" else 0.0
    use_shared_det = (mode != "frozen")

    for w_idx, (w_st, w_ed, fired) in enumerate(alarm_windows):
        tp0 = time.perf_counter()
        z_out[w_st - stream_start : w_ed - stream_start] = (X_arr[w_st:w_ed] @ w_cur) + b_cur
        dt_p = (time.perf_counter() - tp0) * 1000.0
        pred_ms += dt_p

        if mode != "frozen" and fired:
            ta0 = time.perf_counter()
            t_al = float(commit_ts[w_ed - 1])
            _, y_obs_al = observe_labels_causal(commit_ts[:w_ed], y_true[:w_ed], fix_ts[:w_ed], t_al)
            win_sz = max(w_ed - w_st, 200)
            sl = slice(max(0, w_ed - 4 * win_sz), w_ed)
            yb = (y_obs_al[sl] > 0.5).astype(np.int32)
            if int(np.sum(yb == 1)) >= 3 and int(np.sum(yb == 0)) >= 3:
                w_cand, b_cand = _fit_lr(X_arr[sl], yb)
                if np.all(np.isfinite(w_cand)) and np.isfinite(b_cand):
                    w_cur, b_cur = w_cand, b_cand
                    n_upd += 1
                    opt_steps += 1
                    ex_seen += int(w_ed - sl.start)
            dt_a = (time.perf_counter() - ta0) * 1000.0
            adapt_ms += dt_a
            alarm_ms_list.append(dt_a)
            fired_win_records.append((w_idx, w_st, w_ed, dt_p, dt_a))

    tp1 = time.perf_counter()
    p_out = sigmoid_safe(z_out)
    pred_ms += (time.perf_counter() - tp1) * 1000.0

    tc0 = time.perf_counter()
    cal_p, th_vec, win_cal_ms = calibrate_causal_stream_probs_strict(
        z_init_past, z_out, commit_ts, y_true, fix_ts, stream_start, window_size=window_size, return_window_ms=True
    )
    dec_vec = (p_out >= th_vec).astype(np.int32)
    cal_ms = (time.perf_counter() - tc0) * 1000.0
    extra_cal_per_win = max(cal_ms - sum(win_cal_ms), 0.0) / max(len(win_cal_ms), 1)

    sh_anc_ev = float(shared_anchor_eval_embed_ms) if use_shared_det else 0.0
    alarm_step_total_list = [
        dt_p
        + (sh_anc_ev * (we - ws) / max(n_eval, 1))
        + per_win_mon_ms
        + dt_a
        + (win_cal_ms[wi] if wi < len(win_cal_ms) else 0.0)
        + extra_cal_per_win
        for (wi, ws, we, dt_p, dt_a) in fired_win_records
    ]

    mon_ms = float(shared_monitor_ms) if use_shared_det else 0.0
    tb = _pack_timing_breakdown(
        init_fit_ms=init_ms,
        stream_predict_and_embed_ms=pred_ms,
        drift_monitor_ms=mon_ms,
        adaptation_total_ms=adapt_ms,
        calibration_ms=cal_ms,
        shared_anchor_init_ms=float(shared_anchor_init_ms) if use_shared_det else 0.0,
        shared_monitor_init_ms=float(shared_monitor_init_ms) if use_shared_det else 0.0,
        shared_anchor_eval_embed_ms=sh_anc_ev,
    )
    return {
        "raw_logits": z_out,
        "raw_probs": p_out,
        "cal_probs": cal_p,
        "tuned_th_vec": th_vec,
        "decisions_th": dec_vec,
        "n_updates": n_upd,
        "per_alarm_wall_ms": alarm_ms_list,
        "per_alarm_step_total_ms": alarm_step_total_list,
        "end_to_end_eval_ms": tb["consistent_stream_eval_ms"],
        "timing_breakdown_ms": tb,
        "compute": StreamComputeStats(opt_steps=opt_steps, examples_seen=ex_seen, trees_built=0, wall_ms=adapt_ms),
    }


# ==============================================================================
# 2. Verification-Latency Event Stream Generator (`jitsdp/baseline.py:extract_events`)
# ==============================================================================

def extract_verification_latency_events(
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    waiting_days: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches `jitsdp/baseline.py:extract_events` (Cabral ICSE 2019 Sec. IV-A & Song IJCNN 2022 Sec. II):
      - Clean commits (`y == 0`): emit `(commit_ts + wait_sec, commit_idx, target=0)`.
      - Fast-fixed buggy commits (`y == 1`, `fix_ts` finite, `fix_ts - commit_ts < wait_sec`):
        emit `(fix_ts, commit_idx, target=1)`.
      - Delayed/unrevealed buggy commits (`y == 1`, `~isfinite(fix_ts)` or `fix_ts - commit_ts >= wait_sec`):
        emit `(commit_ts + wait_sec, commit_idx, target=0)` at end of waiting period, AND if
        `fix_ts` is finite, emit a SECOND relabeling event `(fix_ts, commit_idx, target=1)` at `fix_ts`.
      - Sorted chronologically by `(ev_ts, ev_idx, ev_lbl)`.
      - Strict simultaneous-timestamp rule (`R3b-C2`, matching `jitsdp/baseline.py:121` `right=False`):
        Before predicting commit `i` at `t_cur = commit_ts[i]`, ONLY events with `ev_ts < t_cur` are
        consumed. Because `commit_ts` is non-decreasing and `ev_ts >= commit_ts[ev_idx]`, `ev_ts < t_cur`
        algebraically guarantees `ev_idx < i` and defers all same-timestamp (`ev_ts == t_cur`) events
        until `commit_ts > t_cur` without ever dropping an event!
    """
    c_ts = np.asarray(commit_ts, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.int32)
    f_ts = np.asarray(fix_ts, dtype=np.float64)
    wait_sec = float(waiting_days) * SECONDS_PER_DAY
    idx_all = np.arange(len(c_ts), dtype=np.int32)

    clean_m = y_t == 0
    bug_m = y_t == 1
    # Quarantine any retroactive `f_ts < c_ts` to unrevealed so `ev_ts >= c_ts[ev_idx]` holds universally!
    fin_fix = bug_m & np.isfinite(f_ts) & (f_ts >= c_ts)
    fast_bug = fin_fix & ((f_ts - c_ts) < wait_sec)
    late_or_unrev_bug = bug_m & (~fast_bug)
    late_fin_bug = fin_fix & ((f_ts - c_ts) >= wait_sec)

    ev_ts = np.concatenate([
        c_ts[clean_m] + wait_sec,
        f_ts[fast_bug],
        c_ts[late_or_unrev_bug] + wait_sec,
        f_ts[late_fin_bug],
    ])
    ev_idx = np.concatenate([
        idx_all[clean_m],
        idx_all[fast_bug],
        idx_all[late_or_unrev_bug],
        idx_all[late_fin_bug],
    ])
    ev_lbl = np.concatenate([
        np.zeros(int(np.sum(clean_m)), dtype=np.int32),
        np.ones(int(np.sum(fast_bug)), dtype=np.int32),
        np.zeros(int(np.sum(late_or_unrev_bug)), dtype=np.int32),
        np.ones(int(np.sum(late_fin_bug)), dtype=np.int32),
    ])

    order = np.lexsort((ev_lbl, ev_idx, ev_ts))
    return ev_ts[order], ev_idx[order], ev_lbl[order]


# ==============================================================================
# 3. `ORB_OHT_Native` (Cabral et al., ICSE 2019, Alg. 1 + `jitsdp/orb.py` on `river` OHT)
# ==============================================================================

def compute_cabral_obf(target: int, ma: float, th: float, l0: float, l1: float, m: float) -> float:
    """Matches `jitsdp/orb.py:63-68` & `pipeline.py:300-308` Oversampling Boost Factor."""
    if target == 0 and ma > th:
        return float(((math.pow(m, ma) - math.pow(m, th)) * l0) / max(math.pow(m, 1.0) - math.pow(m, th), 1e-12) + 1.0)
    if target == 1 and ma < th:
        return float(((math.pow(m, th - ma) - 1.0) * l1) / max(math.pow(m, th) - 1.0, 1e-12) + 1.0)
    return 1.0


def run_orb_oht_native_v3(
    X_tab: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    stream_start: int,
    waiting_days: float = 90.0,
    n_ensembles: int = 10,
    ma_window_size: int = 100,
    th: float = 0.40,
    l0: float = 10.0,
    l1: float = 12.0,
    m: float = 1.5,
    decay_factor: float = 0.99,
    noise_o: int = 5,
    window_size_cal: int = 400,
    seed: int = 42,
    return_trace: bool = False,
) -> Dict:
    """
    Cabral et al. (ICSE 2019, Algorithm 1 + `jitsdp/orb.py` & `baseline.py`) implemented on `river 0.27`:
      - Strict `ev_ts < commit_ts[i]` event consumption (`baseline.py:121` `right=False`, `R3b-C2`).
      - Exact lossless float tuple `tuple(row.tolist())` duplicate-feature noise filter executed BEFORE
        `p1` updates (`baseline.py:191-197`), unclipped `rng.poisson(lam)` (`orb.py:77`), and cold-start
        `0.0` clean prediction until both classes `{0, 1}` are observed (`R3b-M4`).
      - Documented adaptation vs `jitsdp/baseline.py`: draws independent `Poisson(lam) * obf` per tree
        across `n_ensembles = 10` `river.tree.HoeffdingTreeClassifier`s (following ICSE 2019 Alg. 1 line 12).
    """
    rng = np.random.RandomState(seed)
    X_arr = np.asarray(X_tab, dtype=np.float64)
    X_dicts = [{j: float(v) for j, v in enumerate(row)} for row in X_arr]
    feat_keys = [tuple(row.tolist()) for row in X_arr]

    trees = [
        river_tree.HoeffdingTreeClassifier(grace_period=200, delta=1e-7, tau=0.05, leaf_prediction="nba")
        for _ in range(n_ensembles)
    ]

    ev_ts, ev_idx, ev_lbl = extract_verification_latency_events(commit_ts, y_true, fix_ts, waiting_days=waiting_days)
    n_events = len(ev_ts)
    ev_ptr = 0

    observed_classes = set()
    clean_key_counts: Dict[Tuple[float, ...], int] = {}
    p1 = 0.50
    ma = float(th)
    ma_buf: List[int] = []

    n_total = len(commit_ts)
    z_all = np.zeros(n_total, dtype=np.float64)
    p_all = np.zeros(n_total, dtype=np.float64)

    n_learned_events = 0
    n_noise_skipped = 0
    ex_seen = 0
    init_ms = 0.0
    pred_ms = 0.0
    adapt_ms = 0.0
    trace_log: List[Dict] = []

    for i in range(n_total):
        t_cur = float(commit_ts[i])

        # 1. Process ALL training events strictly prior to `t_cur` (`ev_ts < t_cur` => `ev_idx < i`)!
        t_tr0 = time.perf_counter()
        while ev_ptr < n_events and ev_ts[ev_ptr] < t_cur:
            c_idx = int(ev_idx[ev_ptr])
            target = int(ev_lbl[ev_ptr])
            ev_time = float(ev_ts[ev_ptr])
            ev_ptr += 1
            assert c_idx < i, f"Internal event ordering invariant violated: c_idx={c_idx} >= i={i}"

            # Noise filter BEFORE `ORB.update_state` (`jitsdp/baseline.py:116, 191-197`)!
            fkey = feat_keys[c_idx]
            if target == 1 and clean_key_counts.get(fkey, 0) > noise_o:
                n_noise_skipped += 1
                if return_trace:
                    trace_log.append({"event_ts": ev_time, "commit_idx": c_idx, "target": target, "skipped_noise": True, "p1": p1, "ma": ma})
                continue

            if target == 0:
                clean_key_counts[fkey] = clean_key_counts.get(fkey, 0) + 1

            is_trained = len(observed_classes) == 2

            # Exact `jitsdp/orb.py:46-68` `update_lambda` + `update_obf`
            p1 = decay_factor * p1 + (1.0 - decay_factor) * float(target)
            p0 = 1.0 - p1
            lam = 1.0
            if is_trained:
                if target == 1 and p1 < p0:
                    lam = p0 / p1
                elif target == 0 and p0 < p1:
                    lam = p1 / p0

            ma = float(np.mean(ma_buf)) if len(ma_buf) > 0 else float(th)
            obf = compute_cabral_obf(target=target, ma=ma, th=th, l0=l0, l1=l1, m=m) if is_trained else 1.0

            any_pos_weight = False
            x_d = X_dicts[c_idx]
            ks_drawn = []
            for ht in trees:
                k_pois = int(rng.poisson(lam))
                k_w = float(k_pois) * obf
                ks_drawn.append(k_w)
                if k_w > 0.0:
                    ht.learn_one(x_d, target, w=k_w)
                    any_pos_weight = True
                    ex_seen += 1
            if any_pos_weight:
                observed_classes.add(target)
            n_learned_events += 1
            if return_trace:
                trace_log.append({
                    "event_ts": ev_time,
                    "commit_idx": c_idx,
                    "target": target,
                    "skipped_noise": False,
                    "p1": p1,
                    "lambda": lam,
                    "ma": ma,
                    "obf": obf,
                    "k_mean": float(np.mean(ks_drawn)),
                })
        dt_tr = (time.perf_counter() - t_tr0) * 1000.0
        if i < stream_start:
            init_ms += dt_tr
        else:
            adapt_ms += dt_tr

        # 2. Prequential Prediction on commit `i` (`jitsdp/orb.py:80-97`)
        t_pr0 = time.perf_counter()
        if len(observed_classes) == 2:
            x_d_i = X_dicts[i]
            prob_i = float(np.mean([ht.predict_proba_one(x_d_i).get(1, 0.0) for ht in trees]))
            pred_i = int(prob_i >= 0.5)
            ma_buf.append(pred_i)
            if len(ma_buf) > ma_window_size:
                ma_buf.pop(0)
        else:
            prob_i = 0.0

        p_all[i] = np.clip(prob_i, 1e-6, 1.0 - 1e-6)
        z_all[i] = prob_to_logit_safe(p_all[i])
        dt_pr = (time.perf_counter() - t_pr0) * 1000.0
        if i < stream_start:
            init_ms += dt_pr
        else:
            pred_ms += dt_pr

    z_out = z_all[stream_start:]
    p_out = p_all[stream_start:]
    tc0 = time.perf_counter()
    cal_p, th_vec = calibrate_causal_stream_probs_strict(
        z_all[:stream_start], z_out, commit_ts, y_true, fix_ts, stream_start, window_size=window_size_cal
    )
    dec_vec = (p_out >= th_vec).astype(np.int32)
    cal_ms = (time.perf_counter() - tc0) * 1000.0
    tb = _pack_timing_breakdown(init_ms, pred_ms, 0.0, adapt_ms, cal_ms)

    res = {
        "raw_logits": z_out,
        "raw_probs": p_out,
        "cal_probs": cal_p,
        "tuned_th_vec": th_vec,
        "decisions_th": dec_vec,
        "n_updates": n_learned_events,
        "n_noise_skipped": n_noise_skipped,
        "per_alarm_wall_ms": [],
        "per_alarm_step_total_ms": [],
        "end_to_end_eval_ms": tb["consistent_stream_eval_ms"],
        "timing_breakdown_ms": tb,
        "compute": StreamComputeStats(opt_steps=n_learned_events, examples_seen=ex_seen, trees_built=n_ensembles, wall_ms=adapt_ms),
    }
    if return_trace:
        res["trace"] = trace_log
    return res


# ==============================================================================
# 4. `BORB_Native` & `BORB_Alarm` (Cabral et al., EMSE 2023, Alg. 2-3 + `jitsdp/pipeline.py`
#    adapted to standardized tabular streams with `ElasticNet SGDClassifier`)
# ==============================================================================

def _borb_skewed_sample_alg3(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    obf0: float,
    obf1: float,
    n_samples: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    r_draws = rng.uniform(0.0, obf0 + obf1, size=n_samples)
    choose_neg = r_draws < obf0
    n_neg = int(np.sum(choose_neg))
    n_pos = n_samples - n_neg
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=True) if n_neg > 0 else np.empty(0, dtype=np.int64)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=True) if n_pos > 0 else np.empty(0, dtype=np.int64)
    sel = np.concatenate([sel_neg, sel_pos])
    y_sel = np.concatenate([np.zeros(n_neg, dtype=np.int32), np.ones(n_pos, dtype=np.int32)])
    perm = rng.permutation(len(sel))
    return sel[perm], y_sel[perm]


def run_borb_competitor_v3(
    X_tab: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    stream_start: int,
    alarm_windows: Optional[List[Tuple[int, int, bool]]] = None,
    update_mode: str = "native",  # "native" (`rp = 200` at fold start) or "alarm" (strict test-before-update at `c_ed`!)
    waiting_days: float = 90.0,
    pull_request_size: int = 200,
    ma_window_size: int = 100,
    max_sample_size: int = 2000,
    n_iterations: int = 20,
    th: float = 0.40,
    l0: float = 10.0,
    l1: float = 12.0,
    m: float = 1.5,
    window_size_cal: int = 400,
    shared_anchor_init_ms: float = 0.0,
    shared_monitor_init_ms: float = 0.0,
    shared_anchor_eval_embed_ms: float = 0.0,
    shared_monitor_ms: float = 0.0,
    seed: int = 42,
    return_trace: bool = False,
) -> Dict:
    """
    Cabral et al. (EMSE 2023, Algorithms 2 & 3 + `jitsdp/pipeline.py` & `evaluation.py`) with
    `ElasticNet SGDClassifier(loss='log_loss')` base learner:
      - `update_mode == "native"`: deterministic fold size `pull_request_size = 200`; retrains on `[:c_st]`
        at each fold boundary `c_st > stream_start`, then scores `[c_st : c_ed]`.
      - `update_mode == "alarm"` (`R3b-C1`): STRICT TEST-BEFORE-UPDATE! Scores `[c_st : c_ed]` FIRST
        with `clf_cur`, and ONLY AFTER window end `c_ed` retrains `clf_cur` on `[:c_ed]` when `fired` is
        `True` — zero within-window future-alarm lookahead!
    """
    X_arr = np.asarray(X_tab, dtype=np.float64)
    n_total = len(commit_ts)
    n_eval = n_total - stream_start
    wait_sec = float(waiting_days) * SECONDS_PER_DAY
    classes = np.array([0, 1], dtype=np.int32)

    def _prepare_tau(upto_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        t_tr = float(commit_ts[upto_idx - 1])
        c_sub = commit_ts[:upto_idx]
        y_sub = y_true[:upto_idx]
        f_sub = fix_ts[:upto_idx]
        ind_1 = (y_sub == 1) & np.isfinite(f_sub) & (f_sub <= t_tr)
        ind_0 = (~ind_1) & (c_sub <= t_tr - wait_sec)
        pos_idx = np.where(ind_1)[0]
        neg_idx = np.where(ind_0)[0]
        return pos_idx, neg_idx

    def _fit_borb_step(upto_idx: int, step_seed: int) -> Tuple[Optional[SGDClassifier], List[Dict]]:
        pos_idx, neg_idx = _prepare_tau(upto_idx)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return None, []

        tail_st = max(0, upto_idx - ma_window_size)
        X_ma = X_arr[tail_st:upto_idx]
        n_tau = len(pos_idx) + len(neg_idx)
        num_samp = min(n_tau, max_sample_size)

        clf = SGDClassifier(
            loss="log_loss", penalty="elasticnet", alpha=0.02, l1_ratio=0.15,
            learning_rate="optimal", random_state=step_seed,
        )
        loc_rng = np.random.RandomState(step_seed)
        ma_cur = float(th)
        it_trace: List[Dict] = []

        for it_i in range(n_iterations):
            obf0 = compute_cabral_obf(0, ma_cur, th, l0, l1, m)
            obf1 = compute_cabral_obf(1, ma_cur, th, l0, l1, m)
            sel_idx, sel_y = _borb_skewed_sample_alg3(pos_idx, neg_idx, obf0, obf1, num_samp, loc_rng)
            if len(np.unique(sel_y)) < 2:
                sel_idx = np.concatenate([sel_idx, pos_idx[:1], neg_idx[:1]])
                sel_y = np.concatenate([sel_y, np.array([1, 0], dtype=np.int32)])
            clf.partial_fit(X_arr[sel_idx], sel_y, classes=classes)
            logits_ma = (X_ma @ clf.coef_[0]) + float(clf.intercept_[0])
            preds_ma = (sigmoid_safe(logits_ma) >= 0.5).astype(np.float64)
            ma_cur = float(np.mean(preds_ma))
            if return_trace:
                it_trace.append({"iter": it_i, "obf0": obf0, "obf1": obf1, "ma_after": ma_cur})

        return clf, it_trace

    def _predict_fold(clf_in: Optional[SGDClassifier], c_st: int, c_ed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if clf_in is not None:
            val_st = max(0, c_st - ma_window_size)
            val_probs = sigmoid_safe((X_arr[val_st:c_st] @ clf_in.coef_[0]) + float(clf_in.intercept_[0]))
            z_fold = (X_arr[c_st:c_ed] @ clf_in.coef_[0]) + float(clf_in.intercept_[0])
            p_fold = sigmoid_safe(z_fold)
            w_q = len(val_probs)
            if w_q > 0:
                cat_p = np.concatenate([val_probs, p_fold])
                q_target = 1.0 - th
                th_fold = np.array(
                    [float(np.quantile(cat_p[j + 1 : j + 1 + w_q], q_target)) for j in range(len(p_fold))],
                    dtype=np.float64,
                )
            else:
                th_fold = np.full(len(p_fold), 0.50, dtype=np.float64)
        else:
            p_fold = np.zeros(c_ed - c_st, dtype=np.float64)
            z_fold = prob_to_logit_safe(p_fold)
            th_fold = np.full(c_ed - c_st, 0.50, dtype=np.float64)
        return z_fold, p_fold, th_fold

    t_i0 = time.perf_counter()
    clf_cur, init_trace = _fit_borb_step(stream_start, step_seed=seed)
    if clf_cur is not None:
        z_init_past = (X_arr[:stream_start] @ clf_cur.coef_[0]) + float(clf_cur.intercept_[0])
    else:
        z_init_past = np.full(stream_start, -2.0, dtype=np.float64)
    init_ms = (time.perf_counter() - t_i0) * 1000.0

    z_out = np.zeros(n_eval, dtype=np.float64)
    p_out = np.zeros(n_eval, dtype=np.float64)
    borb_th_vec = np.full(n_eval, 0.50, dtype=np.float64)

    n_retrains = 0
    opt_steps = n_iterations if clf_cur is not None else 0
    ex_seen = max_sample_size * n_iterations if clf_cur is not None else 0
    pred_ms = 0.0
    adapt_ms = 0.0
    alarm_ms_list: List[float] = []
    fired_borb_records: List[Tuple[int, int, int, float, float]] = []
    all_traces: List[Dict] = [{"step": stream_start, "iterations": init_trace}] if return_trace else []
    use_shared_det_b = (update_mode == "alarm" and alarm_windows is not None)

    if use_shared_det_b and alarm_windows is not None:
        per_win_mon_ms = float(shared_monitor_ms) / max(len(alarm_windows), 1)
        # STRICT TEST-BEFORE-UPDATE (`R3b-C1`): predict `[c_st : c_ed]` BEFORE observing `fired` at `c_ed`!
        for f_idx, (c_st, c_ed, fired) in enumerate(alarm_windows):
            tp0 = time.perf_counter()
            zf, pf, thf = _predict_fold(clf_cur, c_st, c_ed)
            z_out[c_st - stream_start : c_ed - stream_start] = zf
            p_out[c_st - stream_start : c_ed - stream_start] = pf
            borb_th_vec[c_st - stream_start : c_ed - stream_start] = thf
            dt_p = (time.perf_counter() - tp0) * 1000.0
            pred_ms += dt_p

            if fired:
                ta0 = time.perf_counter()
                cand_clf, tr_f = _fit_borb_step(c_ed, step_seed=seed + 37 * (f_idx + 1))
                if cand_clf is not None and np.all(np.isfinite(cand_clf.coef_)):
                    clf_cur = cand_clf
                    n_retrains += 1
                    opt_steps += n_iterations
                    ex_seen += max_sample_size * n_iterations
                    if return_trace and len(all_traces) < 5:
                        all_traces.append({"step": c_ed, "iterations": tr_f})
                dt_a = (time.perf_counter() - ta0) * 1000.0
                adapt_ms += dt_a
                alarm_ms_list.append(dt_a)
                fired_borb_records.append((f_idx, c_st, c_ed, dt_p, dt_a))
    else:
        per_win_mon_ms = 0.0
        folds = [(st, min(st + pull_request_size, n_total)) for st in range(stream_start, n_total, pull_request_size)]
        for f_idx, (c_st, c_ed) in enumerate(folds):
            dt_a = 0.0
            if f_idx > 0:
                ta0 = time.perf_counter()
                cand_clf, tr_f = _fit_borb_step(c_st, step_seed=seed + 37 * f_idx)
                if cand_clf is not None and np.all(np.isfinite(cand_clf.coef_)):
                    clf_cur = cand_clf
                    n_retrains += 1
                    opt_steps += n_iterations
                    ex_seen += max_sample_size * n_iterations
                    if return_trace and len(all_traces) < 5:
                        all_traces.append({"step": c_st, "iterations": tr_f})
                dt_a = (time.perf_counter() - ta0) * 1000.0
                adapt_ms += dt_a
                alarm_ms_list.append(dt_a)

            tp0 = time.perf_counter()
            zf, pf, thf = _predict_fold(clf_cur, c_st, c_ed)
            z_out[c_st - stream_start : c_ed - stream_start] = zf
            p_out[c_st - stream_start : c_ed - stream_start] = pf
            borb_th_vec[c_st - stream_start : c_ed - stream_start] = thf
            dt_p = (time.perf_counter() - tp0) * 1000.0
            pred_ms += dt_p
            if f_idx > 0:
                fired_borb_records.append((f_idx, c_st, c_ed, dt_p, dt_a))

    tc0 = time.perf_counter()
    cal_p, _, win_cal_ms_b = calibrate_causal_stream_probs_strict(
        z_init_past, z_out, commit_ts, y_true, fix_ts, stream_start, window_size=window_size_cal, return_window_ms=True
    )
    dec_vec = (p_out >= borb_th_vec).astype(np.int32)
    cal_ms = (time.perf_counter() - tc0) * 1000.0
    sh_anc_ev_b = float(shared_anchor_eval_embed_ms) if use_shared_det_b else 0.0
    cal_per_commit = cal_ms / max(n_eval, 1)
    alarm_step_total_list = [
        dt_p
        + (sh_anc_ev_b * (ce - cs) / max(n_eval, 1))
        + per_win_mon_ms
        + dt_a
        + (win_cal_ms_b[fi] if (use_shared_det_b and fi < len(win_cal_ms_b)) else cal_per_commit * (ce - cs))
        for (fi, cs, ce, dt_p, dt_a) in fired_borb_records
    ]

    mon_ms = float(shared_monitor_ms) if use_shared_det_b else 0.0
    tb = _pack_timing_breakdown(
        init_fit_ms=init_ms,
        stream_predict_and_embed_ms=pred_ms,
        drift_monitor_ms=mon_ms,
        adaptation_total_ms=adapt_ms,
        calibration_ms=cal_ms,
        shared_anchor_init_ms=float(shared_anchor_init_ms) if use_shared_det_b else 0.0,
        shared_monitor_init_ms=float(shared_monitor_init_ms) if use_shared_det_b else 0.0,
        shared_anchor_eval_embed_ms=sh_anc_ev_b,
    )

    res = {
        "raw_logits": z_out,
        "raw_probs": p_out,
        "cal_probs": cal_p,
        "tuned_th_vec": borb_th_vec,
        "decisions_th": dec_vec,
        "n_updates": n_retrains,
        "per_alarm_wall_ms": alarm_ms_list,
        "per_alarm_step_total_ms": alarm_step_total_list,
        "end_to_end_eval_ms": tb["consistent_stream_eval_ms"],
        "timing_breakdown_ms": tb,
        "compute": StreamComputeStats(opt_steps=opt_steps, examples_seen=ex_seen, trees_built=0, wall_ms=adapt_ms),
    }
    if return_trace:
        res["trace"] = all_traces
    return res


# ==============================================================================
# 5. `ODaSC_OHT_Native` (Specification-Level Implementation of Song et al., IJCNN 2022,
#    Eqs. 1-12 with Strict Radius-Bounded DenStream Micro-Clusters on All `D` Features)
# ==============================================================================

class SongDenStreamConfidenceTracker:
    """
    Specification-Level Implementation of Song et al. (IJCNN 2022, Sections IV-V, Eqs. 1-12):
      - Variable-count damped DenStream micro-clusters `{CF1, CF2, CW, CT, C0, C1}` over ALL `D` features.
      - STRICT RADIUS INVARIANT (`R3b-M4`): A point merges into nearest cluster `j_best` ONLY when
        `new_r <= self.eps_radius`. It NEVER merges when `new_r > self.eps_radius`! If `len(self.cw_) >= max_mc`
        when creating a new cluster, the most obsolete cluster (`argmin(CT)`) is evicted first so every
        cluster ALWAYS satisfies `radius <= self.eps_radius` strictly!
    """

    def __init__(self, eps_radius: float, lam_fade: float = 0.1, temp_t: float = 10.0, delta_thresh: float = 0.85, max_mc: int = 250):
        self.eps_radius = max(float(eps_radius), 1e-4)
        self.lam_fade = lam_fade
        self.temp_t = temp_t
        self.delta_thresh = delta_thresh
        self.max_mc = int(max_mc)
        self.cf1_: List[np.ndarray] = []
        self.cf2_: List[np.ndarray] = []
        self.cw_: List[float] = []
        self.ct_: List[float] = []
        self.c0_: List[float] = []
        self.c1_: List[float] = []
        self.last_t_: float = 0.0
        self.initialized_t_: bool = False

    def _apply_decay(self, t_sec: float) -> None:
        if not self.initialized_t_:
            self.last_t_ = float(t_sec)
            self.initialized_t_ = True
            return
        dt_units = max((float(t_sec) - self.last_t_) / (30.0 * SECONDS_PER_DAY), 0.0)
        if dt_units > 0.0 and len(self.cw_) > 0:
            fade = math.pow(2.0, -self.lam_fade * dt_units)
            for j in range(len(self.cw_)):
                self.cf1_[j] *= fade
                self.cf2_[j] *= fade
                self.cw_[j] *= fade
                self.ct_[j] *= fade
            keep = [j for j in range(len(self.cw_)) if self.ct_[j] >= 1e-3]
            if len(keep) < len(self.cw_):
                self.cf1_ = [self.cf1_[j] for j in keep]
                self.cf2_ = [self.cf2_[j] for j in keep]
                self.cw_ = [self.cw_[j] for j in keep]
                self.ct_ = [self.ct_[j] for j in keep]
                self.c0_ = [self.c0_[j] for j in keep]
                self.c1_ = [self.c1_[j] for j in keep]
        self.last_t_ = float(t_sec)

    def _centroids_and_radii(self) -> Tuple[np.ndarray, np.ndarray]:
        centers = np.array([cf1 / max(cw, 1e-8) for cf1, cw in zip(self.cf1_, self.cw_)], dtype=np.float64)
        radii = []
        for cf1, cf2, cw in zip(self.cf1_, self.cf2_, self.cw_):
            w = max(cw, 1e-8)
            var_sum = float(np.sum(np.maximum(cf2 / w - (cf1 / w) ** 2, 0.0)))
            radii.append(math.sqrt(var_sum))
        return centers, np.array(radii, dtype=np.float64)

    def max_cluster_radius(self) -> float:
        if len(self.cw_) == 0:
            return 0.0
        _, radii = self._centroids_and_radii()
        return float(np.max(radii))

    def estimate_batch_alpha1_raw(self, X_batch: np.ndarray, y_obs_batch: np.ndarray) -> np.ndarray:
        n = len(X_batch)
        if len(self.cw_) == 0:
            return np.where(y_obs_batch == 1, 1.0, 0.20).astype(np.float64)

        centers, radii = self._centroids_and_radii()
        c0_arr = np.asarray(self.c0_, dtype=np.float64)
        c1_arr = np.asarray(self.c1_, dtype=np.float64)
        ct_arr = np.asarray(self.ct_, dtype=np.float64)

        tot_0 = float(np.sum(c0_arr))
        tot_1 = float(np.sum(c1_arr))
        denom_d = max(tot_0 + tot_1, 1e-8)
        p0_d = tot_0 / denom_d
        p1_d = tot_1 / denom_d

        mc_tot = np.maximum(c0_arr + c1_arr, 1e-8)
        p0_c = c0_arr / mc_tot
        p1_c = c1_arr / mc_tot

        d0 = 0.5 * (p0_c - p0_d) / np.maximum(p0_c + p0_d, 1e-8)
        d1 = 0.5 * (p1_c - p1_d) / np.maximum(p1_c + p1_d, 1e-8)
        rep0 = p0_c * (1.0 + d0)
        rep1 = p1_c * (1.0 + d1)
        obs = np.tanh(ct_arr / self.temp_t)

        alpha1 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            dists = np.linalg.norm(centers - X_batch[i][None, :], axis=1)
            exc_d = np.maximum(0.0, dists - radii)
            cor = np.exp(-self.temp_t * np.minimum(exc_d, 5.0))
            w_mc = cor * obs
            if float(np.sum(w_mc)) < 1e-5:
                alpha1[i] = 1.0 if y_obs_batch[i] == 1 else p1_d
            else:
                rho_x = float(np.mean(w_mc) - np.std(w_mc))
                f_mask = w_mc >= rho_x
                if not np.any(f_mask):
                    f_mask = np.ones_like(w_mc, dtype=bool)
                num1 = float(np.sum(rep1[f_mask] * w_mc[f_mask]))
                num0 = float(np.sum(rep0[f_mask] * w_mc[f_mask]))
                alpha1[i] = num1 / max(num0 + num1, 1e-8)
        return np.clip(alpha1, 0.0, 1.0)

    def calibrate_and_compute_alpha(self, X_batch: np.ndarray, y_obs_batch: np.ndarray, t_sec: float) -> np.ndarray:
        self._apply_decay(t_sec)
        alpha1 = self.estimate_batch_alpha1_raw(X_batch, y_obs_batch)

        pos_m = y_obs_batch == 1
        d_pos = int(np.sum(pos_m))
        if d_pos >= 1:
            a1_pos = alpha1[pos_m]
            a1_min = max(float(np.min(a1_pos)), 1e-4)
            if d_pos == 1:
                bench = a1_min
            else:
                mu_p = float(np.mean(a1_pos))
                sd_p = float(np.std(a1_pos))
                k_min = 3
                for k_cand in (1, 2, 3):
                    if mu_p - k_cand * sd_p <= a1_min + 1e-12:
                        k_min = k_cand
                        break
                bench = max(mu_p - k_min * sd_p, 0.0)
            cal_mask = alpha1 >= bench
            alpha1[cal_mask] = np.minimum(1.0, alpha1[cal_mask] / a1_min)

        alpha_final = np.where(pos_m, alpha1, 1.0 - alpha1)
        alpha_final = np.clip(alpha_final, 0.01, 1.0)

        for i in range(len(X_batch)):
            xi = X_batch[i]
            yi = int(y_obs_batch[i])
            ai = float(alpha_final[i])

            merged = False
            if len(self.cw_) > 0:
                centers, _ = self._centroids_and_radii()
                j_best = int(np.argmin(np.linalg.norm(centers - xi[None, :], axis=1)))
                new_cw = self.cw_[j_best] + 1.0
                new_cf1 = self.cf1_[j_best] + xi
                new_cf2 = self.cf2_[j_best] + (xi ** 2)
                new_r = math.sqrt(float(np.sum(np.maximum(new_cf2 / new_cw - (new_cf1 / new_cw) ** 2, 0.0))))
                # STRICT RADIUS INVARIANT: never violate `new_r <= eps_radius`!
                if new_r <= self.eps_radius:
                    self.cf1_[j_best] = new_cf1
                    self.cf2_[j_best] = new_cf2
                    self.cw_[j_best] = new_cw
                    self.ct_[j_best] += 1.0
                    if yi == 1:
                        self.c1_[j_best] += 1.0
                    else:
                        self.c0_[j_best] += 1.0
                        if ai < (1.0 - self.delta_thresh):
                            self.c1_[j_best] += 1.0
                    merged = True

            if not merged:
                if len(self.cw_) >= self.max_mc:
                    # Evict the most obsolete micro-cluster (`argmin(CT)`) before spawning new radius-0 cluster!
                    evict_j = int(np.argmin(self.ct_))
                    self.cf1_.pop(evict_j)
                    self.cf2_.pop(evict_j)
                    self.cw_.pop(evict_j)
                    self.ct_.pop(evict_j)
                    self.c0_.pop(evict_j)
                    self.c1_.pop(evict_j)

                self.cf1_.append(xi.copy())
                self.cf2_.append(xi ** 2)
                self.cw_.append(1.0)
                self.ct_.append(1.0)
                c0_init = 1.0 if yi == 0 else 0.0
                c1_init = 1.0 if (yi == 1 or (yi == 0 and ai < (1.0 - self.delta_thresh))) else 0.0
                self.c0_.append(c0_init)
                self.c1_.append(c1_init)

        return alpha_final


def run_odasc_oht_native_v3(
    X_tab: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    stream_start: int,
    waiting_days: float = 15.0,
    n_ensembles: int = 10,
    decay_factor: float = 0.99,
    delta_thresh: float = 0.85,
    window_size_cal: int = 400,
    seed: int = 42,
    return_trace: bool = False,
) -> Dict:
    """
    Specification-Level Implementation of `ODaSC` (`Song et al., IJCNN 2022`, Eqs. 1-12):
      - Uses ALL `D` features of `X_tab` (no 24-feature truncation!).
      - Consumes strictly `ev_ts < commit_ts[i]` (`R3b-C2`).
      - Enforces `new_r <= eps_radius` strictly on every DenStream merge (`R3b-M4`).
    """
    t_i0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    X_full = np.asarray(X_tab, dtype=np.float64)
    sub_init = X_full[: min(stream_start, 250)]
    sq_i = np.sum(sub_init ** 2, axis=1)
    d_pair = np.sqrt(np.maximum(sq_i[:, None] + sq_i[None, :] - 2.0 * (sub_init @ sub_init.T), 0.0))
    np.fill_diagonal(d_pair, 0.0)
    eps_r = float(0.5 * np.percentile(d_pair[d_pair > 0], 15)) if np.any(d_pair > 0) else 0.75

    tracker = SongDenStreamConfidenceTracker(eps_radius=eps_r, delta_thresh=delta_thresh)
    X_dicts = [{j: float(v) for j, v in enumerate(row)} for row in X_full]
    trees = [
        river_tree.HoeffdingTreeClassifier(grace_period=200, delta=1e-7, tau=0.05, leaf_prediction="nba")
        for _ in range(n_ensembles)
    ]

    ev_ts, ev_idx, ev_lbl = extract_verification_latency_events(commit_ts, y_true, fix_ts, waiting_days=waiting_days)
    n_events = len(ev_ts)
    ev_ptr = 0
    init_ms = (time.perf_counter() - t_i0) * 1000.0

    observed_classes = set()
    p1 = 0.50
    n_total = len(commit_ts)
    z_all = np.zeros(n_total, dtype=np.float64)
    p_all = np.zeros(n_total, dtype=np.float64)
    n_learned = 0
    ex_seen = 0
    pred_ms = 0.0
    adapt_ms = 0.0
    trace_log: List[Dict] = []

    for i in range(n_total):
        t_cur = float(commit_ts[i])

        t_tr0 = time.perf_counter()
        st_ptr = ev_ptr
        while ev_ptr < n_events and ev_ts[ev_ptr] < t_cur:
            assert int(ev_idx[ev_ptr]) < i, "Internal event ordering invariant violated!"
            ev_ptr += 1

        if ev_ptr > st_ptr:
            batch_cidx = ev_idx[st_ptr:ev_ptr]
            batch_lbl = ev_lbl[st_ptr:ev_ptr]
            alphas = tracker.calibrate_and_compute_alpha(X_full[batch_cidx], batch_lbl, t_sec=t_cur)

            for b_k in range(len(batch_cidx)):
                c_id = int(batch_cidx[b_k])
                y_st = int(batch_lbl[b_k])
                alp = float(alphas[b_k])

                p1 = decay_factor * p1 + (1.0 - decay_factor) * float(y_st)
                p0 = 1.0 - p1
                is_tr = len(observed_classes) == 2

                lam_oob = 1.0
                is_minority = False
                if is_tr:
                    if y_st == 1 and p1 < p0:
                        lam_oob = p0 / p1
                        is_minority = True
                    elif y_st == 0 and p0 < p1:
                        lam_oob = p1 / p0
                        is_minority = True

                if is_minority or y_st == 0:
                    lam_odasc = lam_oob * math.exp(alp - delta_thresh)
                else:
                    lam_odasc = lam_oob

                x_d = X_dicts[c_id]
                any_k = False
                for ht in trees:
                    k_draw = int(rng.poisson(max(lam_odasc, 0.0)))
                    if k_draw > 0:
                        ht.learn_one(x_d, y_st, w=float(k_draw))
                        any_k = True
                        ex_seen += k_draw
                if any_k:
                    observed_classes.add(y_st)
                n_learned += 1
                if return_trace and len(trace_log) < 80:
                    trace_log.append({
                        "commit_idx": c_id,
                        "y_star": y_st,
                        "alpha": alp,
                        "lambda_oob": lam_oob,
                        "lambda_odasc": lam_odasc,
                        "n_microclusters": len(tracker.cw_),
                    })
        dt_tr = (time.perf_counter() - t_tr0) * 1000.0
        if i < stream_start:
            init_ms += dt_tr
        else:
            adapt_ms += dt_tr

        t_pr0 = time.perf_counter()
        if len(observed_classes) == 2:
            x_di = X_dicts[i]
            prob_i = float(np.mean([ht.predict_proba_one(x_di).get(1, 0.0) for ht in trees]))
        else:
            prob_i = 0.0

        p_all[i] = np.clip(prob_i, 1e-6, 1.0 - 1e-6)
        z_all[i] = prob_to_logit_safe(p_all[i])
        dt_pr = (time.perf_counter() - t_pr0) * 1000.0
        if i < stream_start:
            init_ms += dt_pr
        else:
            pred_ms += dt_pr

    z_out = z_all[stream_start:]
    p_out = p_all[stream_start:]
    tc0 = time.perf_counter()
    cal_p, th_vec = calibrate_causal_stream_probs_strict(
        z_all[:stream_start], z_out, commit_ts, y_true, fix_ts, stream_start, window_size=window_size_cal
    )
    dec_vec = (p_out >= th_vec).astype(np.int32)
    cal_ms = (time.perf_counter() - tc0) * 1000.0
    tb = _pack_timing_breakdown(init_ms, pred_ms, 0.0, adapt_ms, cal_ms)

    res = {
        "raw_logits": z_out,
        "raw_probs": p_out,
        "cal_probs": cal_p,
        "tuned_th_vec": th_vec,
        "decisions_th": dec_vec,
        "n_updates": n_learned,
        "final_microclusters": len(tracker.cw_),
        "max_microcluster_radius": tracker.max_cluster_radius(),
        "eps_radius": eps_r,
        "per_alarm_wall_ms": [],
        "per_alarm_step_total_ms": [],
        "end_to_end_eval_ms": tb["consistent_stream_eval_ms"],
        "timing_breakdown_ms": tb,
        "compute": StreamComputeStats(opt_steps=n_learned, examples_seen=ex_seen, trees_built=n_ensembles, wall_ms=adapt_ms),
    }
    if return_trace:
        res["trace"] = trace_log
    return res
