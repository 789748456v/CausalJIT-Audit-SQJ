r"""
Companion Forensic & Mechanism Audit Runner (`scripts_v3/run_reviewer_mechanism_audits_v3.py`).

Read-only with respect to immutable `src_v3/` and all 14 archived `results_v3/*_v3.json` files.
Computes provenance-stamped empirical diagnostics across seven companion audits:

  - Audit A: Complete ApacheJIT confirmatory filtering funnel (`apachejit_total.csv`
    -> `<=2019-01-01` -> `init/val/eval`), upstream `apache/hadoop` 11,964 clean-commit disclosure
    (`7,452 + 4,512 = 11,964`), and separated Standalone-3 vs Monorepo-2 Confirmatory AP summaries.
  - Audit B: End-of-archive follow-up proxy distributions across all 34 natural
    repository streams (`2 + 14 + 8 + 5 + 5 = 34`) plus complete Benchmark A follow-up sensitivity
    table (`All Eval`, `rem >= 180d`, `rem >= 365d`) with N, prevalence, and AP/F1 for all core methods
    (including `Full_LR_B0`).
  - Audit C: Alarm-level Stage 1-3 intermediate estimation audit
    comparing `Survival-PU` (CohortIncidence) vs `FixedCurve-PU` (90d half-life) on the exact same
    62 DriftLens alarms (10 streams, seed 42): unique commit counts, alarm-level & stream-level
    mean +/- SD, strictly unrevealed (`y_obs == 0`) Brier/bias by age stratum, fair writeback control
    (`De-Weighted Teacher + Writeback [y_obs==1 -> 1.0]`), and unreachable-bisection failure rates
    with exact total-alarm denominators across all archived JSONs.
  - Audit D: 90-day calibration-pool retrospective censoring rate across
    cohorts & RQ4a kappas + symmetrical Raw/No-Platt vs Shared 90d-Platt evaluation under IDENTICAL
    90d causal threshold search across ALL FIVE methods (`B2_4W`, alarm-synchronized
    `B2_4W_PostHocLogit`, Proposition-1-inspired plug-in `B2_4W_PostHocRatio` with clipping/tie-rate
    diagnostics, `FixedCurve_PU`, `Survival_PU`) on kappas `{0.0, 1.0, 3.0}` (10 streams x 5 seeds).
  - Audit E: Per-stream breakdown of all 4 Benchmark B controlled abrupt shifts
    and exact splice timeline statistics (`1h` gap + fraction of pre-onset fixes revealed post-onset).
  - Audit F: Multi-seed (5 seeds `42..46`) Whole-Stream AP vs Unweighted &
    Positive-Weighted Per-Window Mean AP across all 27 Benchmark B natural streams (K=25 clusters)
    for all 7 core methods (`B0`, `B1b*`, `B2-4W`, `Wait-90`, `FixedCurve-PU`, `Survival-PU`,
    `Full-LR Adapt-4W`), including zero-positive window exclusion statistics, K=25 paired Wilcoxon
    inference with 10,000-bootstrap 95% CIs, and controlled within-window scale-removal ablations.
  - Audit G: Paired `TP_rescued`, `TP_harmed`, `Net_TP`, `Net_FP`, and exact
    consumed LOC % under BOTH whole-stream retrospective budgets (`Fixed 0.50`, `Top-10%`,
    `Top-20%`, `20%-LOC`) and per-window operational 400-commit budgets (`Win Top-10%`, `Win Top-20%`,
    `Win 20%-LOC`) on Benchmark A seed 42, plus 5-seed multi-seed `R@20%Effort` & `Windowed Batch
    R@20%Effort` corroboration from archived `benchmark_a_multihost_v3.json`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from config_v3 import NATURAL_STREAM_TO_DONOR_CLUSTER, SECONDS_PER_DAY, V3_CONFIG
from metrics_and_stats_v3 import (
    compute_ece,
    evaluate_horizon_v3,
    paired_cluster_inference_v3,
    prob_to_logit_safe,
    sigmoid_safe,
    tune_f1_threshold_on_mature_slice,
)
from models_and_baselines_v3 import (
    blend_hosts_v3,
    build_host_backbone_v3,
    compute_covariate_importance_weights_v3,
    compute_shared_causal_alarm_trajectory,
    execute_within_host_counterfactual_suite_v3,
    predict_logits_and_embeddings_v3,
    train_supervised_v3,
)
from run_v3_experiments import (
    REPRESENTATIVE_10_STREAMS,
    apply_future_only_latency_intervention,
    build_provenance_header,
    save_json_atomic,
)
from stream_competitors_v3 import (
    calibrate_causal_stream_probs_strict,
    observe_labels_causal,
    run_linear_competitor_v3,
)
from survival_estimators_v3 import (
    CohortIncidenceEstimator,
    FixedHalfLifeMaturityEstimator,
    compute_survival_pu_posterior,
    estimate_ht_shrunk_prevalence,
    solve_monotone_pu_bisection,
)


def _calibration_slope(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    if len(y) < 20 or np.sum(y == 1) < 3 or np.sum(y == 0) < 3:
        return float("nan")
    z = prob_to_logit_safe(probs).reshape(-1, 1)
    if float(np.std(z)) < 1e-6:
        return 0.0
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(z, y)
        return float(lr.coef_[0, 0])
    except Exception:
        return float("nan")


def _window_ap_stats(
    y_true: np.ndarray,
    raw_logits: np.ndarray,
    raw_probs: np.ndarray,
    window_size: int = 400,
) -> Dict[str, float]:
    """Compute unweighted & pos-weighted per-window AP plus controlled scale-removed whole-stream AP."""
    y = np.asarray(y_true, dtype=np.int32)
    z = np.asarray(raw_logits, dtype=np.float64)
    p = np.asarray(raw_probs, dtype=np.float64)
    n = len(y)

    win_aps: List[float] = []
    win_pos_w: List[float] = []
    n_total_w = 0
    n_excluded_w = 0

    z_std_all = np.zeros(n, dtype=np.float64)
    r_norm_all = np.zeros(n, dtype=np.float64)

    for st in range(0, n, window_size):
        ed = min(st + window_size, n)
        n_total_w += 1
        yw = y[st:ed]
        zw = z[st:ed]
        pw = p[st:ed]
        np_w = int(np.sum(yw == 1))
        nn_w = int(np.sum(yw == 0))
        if np_w > 0 and nn_w > 0:
            win_aps.append(float(average_precision_score(yw, pw)))
            win_pos_w.append(float(np_w))
        else:
            n_excluded_w += 1

        # Controlled scale-removal transforms (strictly monotone within each window!)
        sd_z = float(np.std(zw))
        z_std_all[st:ed] = (zw - float(np.mean(zw))) / max(sd_z, 1e-6)
        if len(pw) > 1:
            r_norm_all[st:ed] = (rankdata(pw, method="average") - 1.0) / float(len(pw) - 1)
        else:
            r_norm_all[st:ed] = 0.5

    has_both = (int(np.sum(y == 1)) > 0) and (int(np.sum(y == 0)) > 0)
    whole_ap = float(average_precision_score(y, p)) if has_both else float("nan")
    zstd_whole_ap = float(average_precision_score(y, z_std_all)) if has_both else float("nan")
    rank_whole_ap = float(average_precision_score(y, r_norm_all)) if has_both else float("nan")

    unw = float(np.mean(win_aps)) if win_aps else float("nan")
    pw_ap = float(np.average(win_aps, weights=win_pos_w)) if win_aps and sum(win_pos_w) > 0 else float("nan")
    return {
        "whole_stream_ap": whole_ap,
        "win_ap_unweighted": unw,
        "win_ap_pos_weighted": pw_ap,
        "scale_norm_zscore_whole_ap": zstd_whole_ap,
        "scale_norm_rank_whole_ap": rank_whole_ap,
        "n_windows_total": float(n_total_w),
        "n_windows_excluded": float(n_excluded_w),
    }


def _tune_causal_threshold_on_platted_probs(
    past_init_logits: np.ndarray,
    raw_logits_stream: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    stream_start: int,
    window_size: int = 400,
    wait_days_mature: float = 90.0,
) -> np.ndarray:
    """Tune binary threshold on 90d Platted probabilities using identical mature pool & grid."""
    from metrics_and_stats_v3 import apply_platt, fit_causal_platt_calibrator

    n_eval = len(commit_ts) - stream_start
    th_platt = np.full(n_eval, 0.50, dtype=np.float64)
    init_z = np.asarray(past_init_logits, dtype=np.float64)
    for st in range(stream_start, len(commit_ts), window_size):
        ed = min(st + window_size, len(commit_ts))
        t_now = float(commit_ts[st - 1])
        _, y_obs_now = observe_labels_causal(commit_ts[:st], y_true[:st], fix_ts[:st], t_now)
        ages_d = (t_now - commit_ts[:st]) / SECONDS_PER_DAY
        used = False
        if st > stream_start:
            mat_m = ages_d[stream_start:st] >= wait_days_mature
            if int(np.sum(mat_m)) >= 80:
                z_cal = raw_logits_stream[: st - stream_start][mat_m][-1200:]
                y_cal = y_obs_now[stream_start:st][mat_m][-1200:]
                a_p, b_p = fit_causal_platt_calibrator(z_cal, y_cal)
                th_platt[st - stream_start : ed - stream_start] = tune_f1_threshold_on_mature_slice(
                    apply_platt(z_cal, a_p, b_p), y_cal
                )
                used = True
        if not used:
            mat_init = ages_d[:stream_start] >= wait_days_mature
            if int(np.sum(mat_init)) >= 80 and len(init_z) == stream_start:
                z_ic = init_z[mat_init][-1200:]
                y_ic = y_obs_now[:stream_start][mat_init][-1200:]
                a_p0, b_p0 = fit_causal_platt_calibrator(z_ic, y_ic)
                th_platt[st - stream_start : ed - stream_start] = tune_f1_threshold_on_mature_slice(
                    apply_platt(z_ic, a_p0, b_p0), y_ic
                )
    return th_platt


# ==============================================================================
# Audit A (Point #2 / R2 #6): ApacheJIT Filtering Funnel & Standalone vs Monorepo
# ==============================================================================
def run_audit_a_apachejit_funnel() -> Dict:
    raw_csv = os.path.join(BASE_DIR, "data_sync", "confirmatory_apachejit_raw", "apachejit_total.csv")
    df = pd.read_csv(raw_csv)
    cutoff = pd.Timestamp("2019-01-01").timestamp()

    proj_order = [
        ("apache/hadoop", "HADOOP_MONOREPO_CLEAN", None),
        ("apache/hadoop-hdfs", "HDFS", "conf_apache_hdfs"),
        ("apache/hadoop-mapreduce", "MAPREDUCE", "conf_apache_mapreduce"),
        ("apache/spark", "SPARK", "conf_apache_spark"),
        ("apache/zeppelin", "ZEPPELIN", "conf_apache_zeppelin"),
        ("apache/zookeeper", "ZOOKEEPER", "conf_apache_zookeeper"),
    ]
    funnel_rows = []
    for proj_full, short_code, sname in proj_order:
        sub = df[df["project"] == proj_full]
        sub_c = sub[sub["author_date"] <= cutoff].sort_values("author_date", kind="mergesort")
        row = {
            "project": proj_full,
            "short_code": short_code,
            "raw_N": int(len(sub)),
            "raw_pos": int(sub["buggy"].sum()),
            "raw_neg": int((~sub["buggy"]).sum()),
            "raw_prev": float(sub["buggy"].mean()) if len(sub) else 0.0,
            "raw_date_min": pd.to_datetime(sub["author_date"].min(), unit="s").strftime("%Y-%m-%d"),
            "raw_date_max": pd.to_datetime(sub["author_date"].max(), unit="s").strftime("%Y-%m-%d"),
            "post_cutoff_N": int(len(sub_c)),
            "post_cutoff_pos": int(sub_c["buggy"].sum()),
            "post_cutoff_neg": int((~sub_c["buggy"]).sum()),
            "post_cutoff_prev": float(sub_c["buggy"].mean()) if len(sub_c) else 0.0,
            "post_cutoff_date_max": pd.to_datetime(sub_c["author_date"].max(), unit="s").strftime("%Y-%m-%d"),
        }
        if sname is not None:
            b = np.load(os.path.join(BASE_DIR, "cache", "confirmatory_v3", f"{sname}.npz"))
            ni, nv = int(b["n_init"]), int(b["n_val"])
            y = b["y_true"].astype(int)
            row.update({
                "n_init": ni,
                "n_val": nv,
                "n_eval": int(len(y) - ni - nv),
                "init_val_pos": int(np.sum(y[: ni + nv] == 1)),
                "init_val_prev": float(np.mean(y[: ni + nv])),
                "eval_pos": int(np.sum(y[ni + nv :] == 1)),
                "eval_neg": int(np.sum(y[ni + nv :] == 0)),
                "eval_prev": float(np.mean(y[ni + nv :])),
            })
        funnel_rows.append(row)

    conf_j = json.load(open(os.path.join(BASE_DIR, "results_v3", "confirmatory_cohort_v3.json"), "r", encoding="utf-8"))
    standalone = ["conf_apache_spark", "conf_apache_zeppelin", "conf_apache_zookeeper"]
    monorepo = ["conf_apache_hdfs", "conf_apache_mapreduce"]
    mlp_methods = [
        "ODaSC_OHT_Native", "BORB_Native", "ORB_OHT_Native", "Full_LR_B0",
        "B1a_Full_Hindsight", "B0_Frozen", "B1b_4W_Oracle", "Teacher_Only_Matched",
        "B2_4W", "Survival_PU", "FixedCurve_PU", "Wait_90",
    ]
    lgb_methods = ["B0_Frozen", "B1b_4W_Oracle", "Teacher_Only_Matched", "B2_4W", "Survival_PU", "FixedCurve_PU", "Wait_90"]

    split_summary = {"standalone_3_mlp": {}, "monorepo_2_mlp": {}, "standalone_3_lgb": {}, "monorepo_2_lgb": {}}
    for m in mlp_methods:
        split_summary["standalone_3_mlp"][m] = float(np.mean([conf_j["streams_mlp"][s]["methods"][m]["full_stream"]["ap"] for s in standalone]))
        split_summary["monorepo_2_mlp"][m] = float(np.mean([conf_j["streams_mlp"][s]["methods"][m]["full_stream"]["ap"] for s in monorepo]))
    for m in lgb_methods:
        split_summary["standalone_3_lgb"][m] = float(np.mean([conf_j["streams_lightgbm"][s]["methods"][m]["full_stream"]["ap"] for s in standalone]))
        split_summary["monorepo_2_lgb"][m] = float(np.mean([conf_j["streams_lightgbm"][s]["methods"][m]["full_stream"]["ap"] for s in monorepo]))

    return {
        "total_apachejit_csv_rows": int(len(df)),
        "total_apachejit_csv_buggy": int(df["buggy"].sum()),
        "hadoop_clean_reconciliation": {
            "paper_hdfs_total": 10359,
            "csv_hdfs_raw": int(funnel_rows[1]["raw_N"]),
            "hdfs_diff": 10359 - int(funnel_rows[1]["raw_N"]),
            "paper_mapreduce_total": 5833,
            "csv_mapreduce_raw": int(funnel_rows[2]["raw_N"]),
            "mapreduce_diff": 5833 - int(funnel_rows[2]["raw_N"]),
            "sum_of_diffs": (10359 - int(funnel_rows[1]["raw_N"])) + (5833 - int(funnel_rows[2]["raw_N"])),
            "csv_hadoop_clean_only": int(funnel_rows[0]["raw_N"]),
        },
        "funnel_rows": funnel_rows,
        "separated_confirmatory_ap": split_summary,
    }


# ==============================================================================
# Audit B (Point #3 / R2 #9): End-of-Archive Follow-Up Proxy & Bench A Sensitivity
# ==============================================================================
def run_audit_b_followup_cutoffs() -> Dict:
    mb = json.load(open(os.path.join(BASE_DIR, "cache", "benchmark_b_v3", "manifest_b_v3.json"), "r", encoding="utf-8"))
    groups = {
        "Benchmark_A_Natural": [
            ("qt_natural_25k", os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "qt_natural_25k_tab.npz")),
            ("openstack_natural_13k", os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "openstack_natural_13k_tab.npz")),
        ],
        "Benchmark_B_Cabral19": [
            (e["name"], os.path.join(BASE_DIR, "cache", "benchmark_b_v3", f"{e['name']}.npz"))
            for e in mb["natural_streams"] if e["cohort"] == "Cabral19"
        ],
        "Benchmark_B_ApacheJIT": [
            (e["name"], os.path.join(BASE_DIR, "cache", "benchmark_b_v3", f"{e['name']}.npz"))
            for e in mb["natural_streams"] if e["cohort"] == "ApacheJIT"
        ],
        "Benchmark_B_PRSZZ": [
            (e["name"], os.path.join(BASE_DIR, "cache", "benchmark_b_v3", f"{e['name']}.npz"))
            for e in mb["natural_streams"] if e["cohort"] == "N60_PRSZZ"
        ],
        "Confirmatory_ApacheJIT": [
            (os.path.basename(p).replace(".npz", ""), p)
            for p in sorted(glob.glob(os.path.join(BASE_DIR, "cache", "confirmatory_v3", "*.npz")))
        ],
    }

    cohort_summary = {}
    per_stream_cutoffs = {}
    total_natural_streams = 0
    for gname, items in groups.items():
        total_natural_streams += len(items)
        mins, meds, f90, f180, f365 = [], [], [], [], []
        for sname, path in items:
            b = np.load(path)
            ts = b["commit_ts"].astype(np.float64)
            fts = b["fix_ts"].astype(np.float64)
            ni, nv = int(b["n_init"]), int(b["n_val"])
            fin = np.isfinite(fts)
            max_c = float(np.max(ts))
            max_f = float(np.max(fts[fin])) if np.any(fin) else max_c
            end_ts = max(max_c, max_f)
            rem = (end_ts - ts[ni + nv :]) / SECONDS_PER_DAY
            rec = {
                "n_eval": int(len(rem)),
                "max_commit_date": pd.to_datetime(max_c, unit="s").strftime("%Y-%m-%d"),
                "max_finite_fix_date": pd.to_datetime(max_f, unit="s").strftime("%Y-%m-%d"),
                "rem_min_days": float(np.min(rem)),
                "rem_p10_days": float(np.percentile(rem, 10)),
                "rem_median_days": float(np.median(rem)),
                "frac_lt_90d": float(np.mean(rem < 90.0)),
                "frac_lt_180d": float(np.mean(rem < 180.0)),
                "frac_lt_365d": float(np.mean(rem < 365.0)),
            }
            per_stream_cutoffs[sname] = rec
            mins.append(rec["rem_min_days"])
            meds.append(rec["rem_median_days"])
            f90.append(rec["frac_lt_90d"])
            f180.append(rec["frac_lt_180d"])
            f365.append(rec["frac_lt_365d"])
        cohort_summary[gname] = {
            "n_streams": len(items),
            "min_of_rem_min_days": float(min(mins)),
            "median_of_rem_median_days": float(np.median(meds)),
            "max_frac_lt_90d": float(max(f90)),
            "max_frac_lt_180d": float(max(f180)),
            "max_frac_lt_365d": float(max(f365)),
            "mean_frac_lt_365d": float(np.mean(f365)),
        }

    # Complete Benchmark A follow-up sensitivity table across ALL 4 hosts + Full-LR Frozen (seed 42)
    pred_dir = os.path.join(BASE_DIR, "results_v3", "commit_predictions", "benchmark_a")
    bench_a_sens = {}
    hosts_spec = [
        ("att_cnn", "att_cnn__frozen_lr"),
        ("codebert", "codebert__frozen_lr"),
        ("deepjit_cnn", "deepjit_cnn__frozen_lr"),
        ("lightgbm", "lightgbm__none"),
    ]
    methods_a = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"]
    for eco, cache_fn in [("qt_natural_25k", "qt_natural_25k_tab.npz"), ("openstack_natural_13k", "openstack_natural_13k_tab.npz")]:
        raw_b = np.load(os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", cache_fn))
        X_tab = raw_b["X"].astype(np.float64)
        ts = raw_b["commit_ts"].astype(np.float64)
        y_all = raw_b["y_true"].astype(np.int32)
        fts = raw_b["fix_ts"].astype(np.float64)
        eff = raw_b["effort"].astype(np.float64)
        ni, nv = int(raw_b["n_init"]), int(raw_b["n_val"])
        s_start = ni + nv
        archive_end = max(float(np.max(ts)), float(np.max(fts[np.isfinite(fts)])))

        # Compute exact Full_LR_B0 predictions on this stream (seed 42)
        win_dummy = [(st, min(st + 400, len(ts)), False) for st in range(s_start, len(ts), 400)]
        flr = run_linear_competitor_v3(X_tab, ts, y_all, fts, eff, s_start, win_dummy, mode="frozen", seed=42)
        p_flr_all = flr["raw_probs"]
        th_flr_all = flr["tuned_th_vec"]
        pcal_flr_all = flr["cal_probs"]

        eco_res = {}
        for min_rem in (0, 180, 365):
            per_host = {}
            n_kept_eco = 0
            prev_kept_eco = 0.0
            for h_short, h_tag in hosts_spec:
                pf = os.path.join(pred_dir, f"{eco}__{h_tag}__seed42.npz")
                z = np.load(pf)
                cidx = z["commit_idx"].astype(int)
                mask = ((archive_end - ts[cidx]) / SECONDS_PER_DAY) >= float(min_rem)
                n_kept_eco = int(np.sum(mask))
                yt = z["y_true"][mask].astype(int)
                prev_kept_eco = float(np.mean(yt))
                per_host[h_short] = {}
                for m in methods_a:
                    pr = z[f"{m}__raw_prob"][mask]
                    pcal = z[f"{m}__cal_prob"][mask]
                    th = z[f"{m}__tuned_th"][mask]
                    per_host[h_short][m] = {
                        "ap": float(average_precision_score(yt, pr)),
                        "f1": float(precision_recall_fscore_support(yt, (pr >= 0.5).astype(int), average="binary", zero_division=0)[2]),
                        "f1_th": float(precision_recall_fscore_support(yt, (pr >= th).astype(int), average="binary", zero_division=0)[2]),
                        "brier_cal": float(brier_score_loss(yt, pcal)),
                    }
            # Full_LR_B0 on exact same mask
            mask_flr = ((archive_end - ts[s_start:]) / SECONDS_PER_DAY) >= float(min_rem)
            yt_flr = y_all[s_start:][mask_flr]
            flr_m = {
                "ap": float(average_precision_score(yt_flr, p_flr_all[mask_flr])),
                "f1": float(precision_recall_fscore_support(yt_flr, (p_flr_all[mask_flr] >= 0.5).astype(int), average="binary", zero_division=0)[2]),
                "f1_th": float(precision_recall_fscore_support(yt_flr, (p_flr_all[mask_flr] >= th_flr_all[mask_flr]).astype(int), average="binary", zero_division=0)[2]),
                "brier_cal": float(brier_score_loss(yt_flr, pcal_flr_all[mask_flr])),
            }
            h4_mean = {
                m: {
                    met: float(np.mean([per_host[h][m][met] for h, _ in hosts_spec]))
                    for met in ("ap", "f1", "f1_th", "brier_cal")
                }
                for m in methods_a
            }
            h4_mean["Full_LR_B0"] = flr_m
            eco_res[f"rem_ge_{min_rem}d"] = {
                "n_eval_kept": n_kept_eco,
                "eval_prevalence": prev_kept_eco,
                "per_host": per_host,
                "mean_4host": h4_mean,
            }
        bench_a_sens[eco] = eco_res

    return {
        "total_natural_streams": total_natural_streams,
        "cohort_followup_summary": cohort_summary,
        "per_stream_cutoffs": per_stream_cutoffs,
        "benchmark_a_followup_sensitivity_seed42": bench_a_sens,
    }


# ==============================================================================
# Audit C (Points #5, #6, #7 / R2 #4, #8): Unrevealed-Only Stage 1-3 Audit +
# FixedCurve Head-to-Head + Writeback Control + Unreachable Denominators
# ==============================================================================
def run_audit_c_stage123_mechanisms() -> Dict:
    import torch
    torch.set_num_threads(1)

    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    alarm_records = []
    unique_commits = set()
    total_pool_evals = 0

    def _new_bucket():
        return {"y": [], "y_obs": [], "p_teach_raw": [], "p_teach_dewt": [], "p_teach_wb": [], "q_spu": [], "q_fc": []}

    strata_all = {s: _new_bucket() for s in ("age_lt_30d", "age_30_90d", "age_ge_90d", "all_ages")}
    strata_unrev = {s: _new_bucket() for s in ("age_lt_30d", "age_30_90d", "age_ge_90d", "all_unrevealed")}

    fc_est = FixedHalfLifeMaturityEstimator(half_life_days=90.0)

    for sname in REPRESENTATIVE_10_STREAMS:
        b = np.load(os.path.join(b_dir, f"{sname}.npz"))
        X, ts, y, fts = b["X"].astype(np.float32), b["commit_ts"].astype(np.float64), b["y_true"].astype(np.int32), b["fix_ts"].astype(np.float64)
        ni, nv = int(b["n_init"]), int(b["n_val"])
        s_start = ni + nv
        t0 = float(ts[s_start - 1])

        pos_mask_all = y == 1
        delays_all = np.where(pos_mask_all & np.isfinite(fts), np.maximum((fts - ts) / SECONDS_PER_DAY, 0.0), np.inf)
        pos_delays = delays_all[pos_mask_all]

        def true_maturity(ages_d: np.ndarray) -> np.ndarray:
            if len(pos_delays) == 0:
                return np.zeros_like(ages_d, dtype=np.float64)
            return np.mean(pos_delays[:, None] <= ages_d[None, :], axis=0)

        _, y_obs_init = observe_labels_causal(ts[:s_start], y[:s_start], fts[:s_start], t0)
        ages_init_d = (t0 - ts[:s_start]) / SECONDS_PER_DAY
        mat_i = ages_init_d >= 90.0
        anc_prev = float(np.clip(np.mean(y_obs_init[mat_i]) * 1.15, 0.03, 0.65)) if int(np.sum(mat_i)) >= 100 else float(np.clip(np.mean(y_obs_init) * 1.25, 0.03, 0.65))
        pos_w = float(np.clip((1.0 - anc_prev) / max(anc_prev, 0.03), 1.2, 8.0))

        anchor = build_host_backbone_v3("mlp", X[:16], seed=42)
        anchor, _ = train_supervised_v3(anchor, X[:s_start], y_obs_init, pos_weight=pos_w, epochs=12, lr=8e-4, seed=42)
        _, emb_all = predict_logits_and_embeddings_v3(anchor, X)

        W = V3_CONFIG.window_size_default
        shared_windows, _ = compute_shared_causal_alarm_trajectory(
            emb_all,
            n_init=ni,
            n_val=nv,
            window_size=W,
            monitor_mode="frechet",
            calib_sampling="block",
            reference_refresh="rolling",
            drift_percentile=V3_CONFIG.drift_block_pct,
            drift_cooldown=V3_CONFIG.drift_cooldown_windows,
            seed=42,
            return_timing=False,
        )

        cur_spu = build_host_backbone_v3("mlp", X[:16], seed=42)
        cur_spu.load_state_dict(anchor.state_dict())
        cur_fc = build_host_backbone_v3("mlp", X[:16], seed=42)
        cur_fc.load_state_dict(anchor.state_dict())

        for w_idx, (st, ed, fired, _) in enumerate(shared_windows):
            if not fired:
                continue

            t_alarm = float(ts[ed - 1])
            step_seed = 42 + 31 * (w_idx + 1)
            _, y_obs_hist = observe_labels_causal(ts[:ed], y[:ed], fts[:ed], t_alarm)
            est = CohortIncidenceEstimator().fit(ts[:ed], fts[:ed], y_obs_hist, t_alarm)
            ages_sec = np.maximum(t_alarm - ts[:ed], 0.0)
            ages_d = ages_sec / SECONDS_PER_DAY
            c_all_spu = est.predict_maturity(ages_sec).astype(np.float64)
            c_all_fc = fc_est.predict_maturity(ages_sec).astype(np.float64)

            drift_idx = np.arange(st, ed)
            drift_mask = np.zeros(ed, dtype=bool)
            drift_mask[drift_idx] = True

            def _run_stage23(c_vec: np.ndarray, cur_m):
                mat_m = c_vec >= 0.55
                pi_anc_v = float(np.clip(np.mean(y_obs_hist[mat_m] / np.maximum(c_vec[mat_m], V3_CONFIG.ht_floor_c)), V3_CONFIG.pi_min, V3_CONFIG.pi_max)) if int(np.sum(mat_m)) >= 80 else anc_prev
                non_d = np.where(~drift_mask)[0]
                sel_a = non_d[-V3_CONFIG.max_anchor_commits :] if len(non_d) > V3_CONFIG.max_anchor_commits else non_d
                imp_a, _ = compute_covariate_importance_weights_v3(X[sel_a], X[drift_idx])
                imp_a = imp_a * np.linspace(0.70, 1.15, len(sel_a), dtype=np.float32)
                p_idx = np.concatenate([sel_a, drift_idx])
                y_o_p = y_obs_hist[p_idx]
                c_p = c_vec[p_idx]
                fl = V3_CONFIG.maturity_weight_floor
                u_a = imp_a * (fl + (1.0 - fl) * c_vec[sel_a])
                u_d = np.ones(len(drift_idx), dtype=np.float32) * V3_CONFIG.drift_emphasis_gamma * (fl + (1.0 - fl) * c_vec[drift_idx])
                u_p = np.concatenate([u_a, u_d]).astype(np.float64)
                n_c = min(len(p_idx), max(len(drift_idx) * 3, 900))
                c_sl = slice(len(p_idx) - n_c, len(p_idx))
                pi_t, pi_ht_r, neff = estimate_ht_shrunk_prevalence(
                    y_obs=y_o_p[c_sl], c_maturity=c_p[c_sl], sample_weights=u_p[c_sl], anchor_prevalence=pi_anc_v
                )
                pw_up = max(4.5, float(pos_w))
                pw_t = float(0.5 * pos_w + 0.5 * np.clip((1.0 - pi_t) / max(pi_t, 0.03), 1.5, pw_up))
                rel_i = np.where((y_obs_hist > 0.5) | (c_vec >= V3_CONFIG.reliable_maturity_thresh))[0]
                if len(rel_i) >= 100:
                    sel_r = rel_i[-V3_CONFIG.max_reliable_commits :]
                    imp_r, _ = compute_covariate_importance_weights_v3(X[sel_r], X[drift_idx])
                    r_conf = np.where((y_obs_hist[sel_r] > 0.5) & (c_vec[sel_r] < V3_CONFIG.reliable_maturity_thresh), 0.55 + 0.45 * c_vec[sel_r], 1.0).astype(np.float32)
                    w_r = (imp_r * r_conf * np.linspace(V3_CONFIG.recency_min, V3_CONFIG.recency_max, len(sel_r), dtype=np.float32)).astype(np.float32)
                    t_init = blend_hosts_v3(cur_m, anchor, weight_a=0.85)
                    teach, _ = train_supervised_v3(t_init, X[sel_r], y_obs_hist[sel_r], pos_weight=pw_t, epochs=4, lr=V3_CONFIG.neural_adapt_lr, sample_weights=w_r, seed=step_seed)
                    eff_pw = pw_t
                else:
                    teach = build_host_backbone_v3("mlp", X[:16], seed=step_seed)
                    teach.load_state_dict(cur_m.state_dict())
                    eff_pw = pos_w
                lg_t, _ = predict_logits_and_embeddings_v3(teach, X[p_idx])
                p_raw = sigmoid_safe(lg_t)
                b_eta = V3_CONFIG.teacher_sharpness * (lg_t - float(np.log(max(eff_pw, 1e-6))))
                p_dw = sigmoid_safe(b_eta)
                bis = solve_monotone_pu_bisection(b_eta[c_sl], y_o_p[c_sl], c_p[c_sl], u_p[c_sl], pi_t)
                q_p = compute_survival_pu_posterior(b_eta, bis.delta_star, y_o_p, c_p)
                stud, _ = train_supervised_v3(teach, X[p_idx], q_p.astype(np.float32), pos_weight=pw_t, epochs=4, lr=V3_CONFIG.neural_adapt_lr, sample_weights=u_p.astype(np.float32), seed=step_seed + 7)
                return p_idx, c_sl, u_p, pi_t, pi_ht_r, neff, p_raw, p_dw, q_p, bis, stud

            p_idx, cal_sl, u_spu, pi_spu, pi_ht_raw, n_eff_ht, p_t_raw, p_t_dw, q_spu, bis_spu, cur_spu = _run_stage23(c_all_spu, cur_spu)
            _, _, u_fc, pi_fc, _, _, _, _, q_fc, bis_fc, cur_fc = _run_stage23(c_all_fc, cur_fc)

            cal_idx = p_idx[cal_sl]
            for cid in cal_idx:
                unique_commits.add((sname, int(cid)))
            total_pool_evals += int(len(cal_idx))

            age_c = ages_d[cal_idx]
            yt_c = y[cal_idx]
            yo_c = (y_obs_hist[cal_idx] > 0.5).astype(np.int32)
            c_star_c = true_maturity(age_c)

            c_mae_spu = float(np.mean(np.abs(c_all_spu[cal_idx] - c_star_c)))
            c_mae_fc = float(np.mean(np.abs(c_all_fc[cal_idx] - c_star_c)))
            u_c_spu = u_spu[cal_sl]
            u_c_fc = u_fc[cal_sl]
            pi_true_spu = float(np.sum(u_c_spu * yt_c) / np.sum(u_c_spu))
            pi_true_fc = float(np.sum(u_c_fc * yt_c) / np.sum(u_c_fc))
            anchor_w = float(V3_CONFIG.ht_shrinkage_m0 / (n_eff_ht + V3_CONFIG.ht_shrinkage_m0))

            p_raw_c = p_t_raw[cal_sl]
            p_dw_c = p_t_dw[cal_sl]
            p_wb_c = np.where(yo_c == 1, 1.0, p_dw_c)
            q_spu_c = q_spu[cal_sl]
            q_fc_c = q_fc[cal_sl]

            unrev_m = yo_c == 0
            brier_unrev_dw = float(brier_score_loss(yt_c[unrev_m], p_dw_c[unrev_m])) if np.any(unrev_m) else float("nan")
            brier_unrev_spu = float(brier_score_loss(yt_c[unrev_m], q_spu_c[unrev_m])) if np.any(unrev_m) else float("nan")
            brier_unrev_fc = float(brier_score_loss(yt_c[unrev_m], q_fc_c[unrev_m])) if np.any(unrev_m) else float("nan")

            for s_lbl, m_s in [
                ("age_lt_30d", age_c < 30.0),
                ("age_30_90d", (age_c >= 30.0) & (age_c < 90.0)),
                ("age_ge_90d", age_c >= 90.0),
                ("all_ages", np.ones(len(age_c), dtype=bool)),
            ]:
                if np.any(m_s):
                    strata_all[s_lbl]["y"].append(yt_c[m_s])
                    strata_all[s_lbl]["y_obs"].append(yo_c[m_s])
                    strata_all[s_lbl]["p_teach_raw"].append(p_raw_c[m_s])
                    strata_all[s_lbl]["p_teach_dewt"].append(p_dw_c[m_s])
                    strata_all[s_lbl]["p_teach_wb"].append(p_wb_c[m_s])
                    strata_all[s_lbl]["q_spu"].append(q_spu_c[m_s])
                    strata_all[s_lbl]["q_fc"].append(q_fc_c[m_s])

            for s_lbl, m_s in [
                ("age_lt_30d", unrev_m & (age_c < 30.0)),
                ("age_30_90d", unrev_m & (age_c >= 30.0) & (age_c < 90.0)),
                ("age_ge_90d", unrev_m & (age_c >= 90.0)),
                ("all_unrevealed", unrev_m),
            ]:
                if np.any(m_s):
                    strata_unrev[s_lbl]["y"].append(yt_c[m_s])
                    strata_unrev[s_lbl]["y_obs"].append(yo_c[m_s])
                    strata_unrev[s_lbl]["p_teach_raw"].append(p_raw_c[m_s])
                    strata_unrev[s_lbl]["p_teach_dewt"].append(p_dw_c[m_s])
                    strata_unrev[s_lbl]["p_teach_wb"].append(p_wb_c[m_s])
                    strata_unrev[s_lbl]["q_spu"].append(q_spu_c[m_s])
                    strata_unrev[s_lbl]["q_fc"].append(q_fc_c[m_s])

            alarm_records.append({
                "stream": sname,
                "is_shift": sname.startswith("abrupt_"),
                "c_mae_spu": c_mae_spu,
                "c_mae_fc": c_mae_fc,
                "pi_abs_err_spu": abs(float(pi_spu) - pi_true_spu),
                "pi_abs_err_fc": abs(float(pi_fc) - pi_true_fc),
                "pi_signed_bias_spu": float(pi_spu) - pi_true_spu,
                "pi_ht_abs_err": abs(float(pi_ht_raw) - pi_true_spu),
                "n_eff_ht": float(n_eff_ht),
                "anchor_shrinkage_weight": anchor_w,
                "unrev_frac": float(np.mean(unrev_m)),
                "brier_unrev_dw": brier_unrev_dw,
                "brier_unrev_spu": brier_unrev_spu,
                "brier_unrev_fc": brier_unrev_fc,
                "delta_brier_unrev_spu_vs_dw": brier_unrev_spu - brier_unrev_dw,
                "delta_brier_unrev_spu_vs_fc": brier_unrev_spu - brier_unrev_fc,
            })

    df_al = pd.DataFrame(alarm_records)
    df_st = df_al.groupby("stream", as_index=False).mean(numeric_only=True)

    def _summarize_buckets(buckets: Dict) -> Dict:
        out = {}
        for s_lbl, d in buckets.items():
            yy = np.concatenate(d["y"]) if d["y"] else np.array([])
            if len(yy) == 0:
                continue
            yo = np.concatenate(d["y_obs"])
            p_raw = np.concatenate(d["p_teach_raw"])
            p_dw = np.concatenate(d["p_teach_dewt"])
            p_wb = np.concatenate(d["p_teach_wb"])
            qs = np.concatenate(d["q_spu"])
            qf = np.concatenate(d["q_fc"])
            out[s_lbl] = {
                "n_samples": int(len(yy)),
                "obs_pos_frac": float(np.mean(yo)),
                "true_prevalence": float(np.mean(yy)),
                "teacher_raw_brier": float(brier_score_loss(yy, p_raw)),
                "teacher_raw_bias": float(np.mean(p_raw - yy)),
                "teacher_dewt_brier": float(brier_score_loss(yy, p_dw)),
                "teacher_dewt_bias": float(np.mean(p_dw - yy)),
                "teacher_wb_brier": float(brier_score_loss(yy, p_wb)),
                "teacher_wb_bias": float(np.mean(p_wb - yy)),
                "q_fc_brier": float(brier_score_loss(yy, qf)),
                "q_fc_bias": float(np.mean(qf - yy)),
                "q_post_brier": float(brier_score_loss(yy, qs)),
                "q_post_bias": float(np.mean(qs - yy)),
            }
        return out

    # Unreachable bisection registry WITH exact total-alarm denominators (Point #8)
    unreachable_with_denoms = []
    bb = json.load(open(os.path.join(BASE_DIR, "results_v3", "benchmark_b_main_v3.json"), "r", encoding="utf-8"))
    conf = json.load(open(os.path.join(BASE_DIR, "results_v3", "confirmatory_cohort_v3.json"), "r", encoding="utf-8"))
    rq3 = json.load(open(os.path.join(BASE_DIR, "results_v3", "rq3_ablations_v3.json"), "r", encoding="utf-8"))
    bb_tot_al = sum(v["methods"]["Survival_PU"]["telemetry"]["n_alarms"] for v in bb["natural_streams_mlp"].values())
    bb_tot_un = sum(v["methods"]["Survival_PU"]["telemetry"]["bisection_unreachable"] for v in bb["natural_streams_mlp"].values())

    for sname, sdata in bb["natural_streams_mlp"].items():
        for m in ("Survival_PU", "FixedCurve_PU"):
            tel = sdata["methods"][m]["telemetry"]
            if tel.get("bisection_unreachable", 0) > 0:
                unreachable_with_denoms.append({
                    "cohort": "Benchmark_B_MLP",
                    "stream": sname,
                    "method": m,
                    "n_alarms_mean": float(tel["n_alarms"]),
                    "unreachable_mean": float(tel["bisection_unreachable"]),
                    "failure_rate": float(tel["bisection_unreachable"] / max(tel["n_alarms"], 1e-9)),
                    "survival_fallbacks_mean": float(tel.get("survival_fallbacks", 0.0)),
                })
    for host_lbl, root_k in [("Confirmatory_MLP", "streams_mlp"), ("Confirmatory_LightGBM", "streams_lightgbm")]:
        for sname, sdata in conf[root_k].items():
            for m in ("Survival_PU", "FixedCurve_PU"):
                if m in sdata["methods"]:
                    tel = sdata["methods"][m]["telemetry"]
                    if tel.get("bisection_unreachable", 0) > 0:
                        unreachable_with_denoms.append({
                            "cohort": host_lbl,
                            "stream": sname,
                            "method": m,
                            "n_alarms_mean": float(tel["n_alarms"]),
                            "unreachable_mean": float(tel["bisection_unreachable"]),
                            "failure_rate": float(tel["bisection_unreachable"] / max(tel["n_alarms"], 1e-9)),
                            "survival_fallbacks_mean": float(tel.get("survival_fallbacks", 0.0)),
                        })
    for sname in ("apache_flink", "n60_airflow"):
        un_v = float(rq3["per_stream"][sname]["SPU_FrozenAnchorTeacher"]["bisection_unreachable"])
        na_bb = float(bb["natural_streams_mlp"][sname]["methods"]["Survival_PU"]["telemetry"]["n_alarms"])
        unreachable_with_denoms.append({
            "cohort": "RQ3_FrozenAnchorTeacher_5seed_sum",
            "stream": sname,
            "method": "SPU_FrozenAnchorTeacher",
            "unreachable_5seed_sum": un_v,
            "ref_bb_n_alarms_mean": na_bb,
        })

    return {
        "n_alarms_audited": int(len(df_al)),
        "n_streams_audited": int(len(df_st)),
        "total_pool_evaluations": int(total_pool_evals),
        "unique_commits_audited": int(len(unique_commits)),
        "c_hat_mae_spu_alarm_mean": float(df_al["c_mae_spu"].mean()),
        "c_hat_mae_spu_alarm_sd": float(df_al["c_mae_spu"].std(ddof=1)),
        "c_hat_mae_spu_stream_mean": float(df_st["c_mae_spu"].mean()),
        "c_hat_mae_spu_stream_sd": float(df_st["c_mae_spu"].std(ddof=1)),
        "c_hat_mae_spu_natural": float(df_al[~df_al["is_shift"]]["c_mae_spu"].mean()),
        "c_hat_mae_spu_shift": float(df_al[df_al["is_shift"]]["c_mae_spu"].mean()),
        "c_hat_mae_fc_alarm_mean": float(df_al["c_mae_fc"].mean()),
        "c_hat_mae_fc_alarm_sd": float(df_al["c_mae_fc"].std(ddof=1)),
        "c_hat_mae_fc_stream_mean": float(df_st["c_mae_fc"].mean()),
        "c_hat_mae_fc_stream_sd": float(df_st["c_mae_fc"].std(ddof=1)),
        "c_hat_mae_fc_natural": float(df_al[~df_al["is_shift"]]["c_mae_fc"].mean()),
        "c_hat_mae_fc_shift": float(df_al[df_al["is_shift"]]["c_mae_fc"].mean()),
        "pi_target_mae_spu_alarm_mean": float(df_al["pi_abs_err_spu"].mean()),
        "pi_target_mae_spu_alarm_sd": float(df_al["pi_abs_err_spu"].std(ddof=1)),
        "pi_target_mae_fc_alarm_mean": float(df_al["pi_abs_err_fc"].mean()),
        "pi_target_mae_fc_alarm_sd": float(df_al["pi_abs_err_fc"].std(ddof=1)),
        "pi_ht_unshrunk_mae_alarm_mean": float(df_al["pi_ht_abs_err"].mean()),
        "pi_target_signed_bias_mean": float(df_al["pi_signed_bias_spu"].mean()),
        "n_eff_median": float(df_al["n_eff_ht"].median()),
        "n_eff_p10": float(df_al["n_eff_ht"].quantile(0.10)),
        "n_eff_p90": float(df_al["n_eff_ht"].quantile(0.90)),
        "anchor_shrinkage_weight_mean": float(df_al["anchor_shrinkage_weight"].mean()),
        "anchor_shrinkage_weight_median": float(df_al["anchor_shrinkage_weight"].median()),
        "alarm_level_unrev_brier_delta_spu_vs_dw_mean": float(df_al["delta_brier_unrev_spu_vs_dw"].mean()),
        "alarm_level_unrev_brier_delta_spu_vs_dw_sd": float(df_al["delta_brier_unrev_spu_vs_dw"].std(ddof=1)),
        "stream_level_unrev_brier_delta_spu_vs_dw_mean": float(df_st["delta_brier_unrev_spu_vs_dw"].mean()),
        "stream_level_unrev_brier_delta_spu_vs_dw_sd": float(df_st["delta_brier_unrev_spu_vs_dw"].std(ddof=1)),
        "alarm_level_unrev_brier_delta_spu_vs_fc_mean": float(df_al["delta_brier_unrev_spu_vs_fc"].mean()),
        "alarm_level_unrev_brier_delta_spu_vs_fc_sd": float(df_al["delta_brier_unrev_spu_vs_fc"].std(ddof=1)),
        "age_stratified_soft_target_audit": _summarize_buckets(strata_all),
        "age_stratified_unrevealed_only_audit": _summarize_buckets(strata_unrev),
        "benchmark_b_27natural_total_alarms_mean": float(bb_tot_al),
        "benchmark_b_27natural_unreachable_mean": float(bb_tot_un),
        "benchmark_b_27natural_unreachable_rate": float(bb_tot_un / max(bb_tot_al, 1e-9)),
        "unreachable_bisection_with_denominators": unreachable_with_denoms,
    }


# ==============================================================================
# Audit D (Point #8 / R2 #1, #2, #3): Symmetrical Raw vs 90d-Platt Calibration +
# Alarm-Synchronized PostHoc Logit & Proposition-1 PostHoc Ratio Controls
# ==============================================================================
def _measure_cal_pool_censoring(paths: List[str], kappa: Optional[float] = None) -> float:
    rates = []
    W = V3_CONFIG.window_size_default
    for p in paths:
        b = np.load(p)
        ts = b["commit_ts"].astype(np.float64)
        y = b["y_true"].astype(np.int32)
        fts = b["fix_ts"].astype(np.float64)
        ni, nv = int(b["n_init"]), int(b["n_val"])
        s_start = ni + nv
        if kappa is not None:
            t0 = float(ts[s_start - 1])
            fts = apply_future_only_latency_intervention(ts, y, fts, t0=t0, kappa=kappa, post_t0_oracle_visibility=False)
        for st in range(s_start, len(ts), W):
            t_now = float(ts[st - 1])
            ages_d = (t_now - ts[:st]) / SECONDS_PER_DAY
            if st > s_start and int(np.sum(ages_d[s_start:st] >= 90.0)) >= 80:
                idx = np.where(ages_d[s_start:st] >= 90.0)[0][-1200:] + s_start
            else:
                mat_i = ages_d[:s_start] >= 90.0
                if int(np.sum(mat_i)) >= 80:
                    idx = np.where(mat_i)[0][-1200:]
                else:
                    idx = np.arange(s_start)
            yt_p = y[idx]
            n_pos = int(np.sum(yt_p == 1))
            if n_pos > 0:
                _, y_obs_p = observe_labels_causal(ts[idx], yt_p, fts[idx], t_now)
                rates.append(float(1.0 - np.sum(y_obs_p > 0.5) / float(n_pos)))
    return float(np.mean(rates)) if rates else float("nan")


def _run_posthoc_b2_one_job(task: Tuple[str, str, float, List[int]]) -> Tuple[str, str, Dict]:
    k_key, sname, kap, seeds = task
    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    b = np.load(os.path.join(b_dir, f"{sname}.npz"))
    X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
    ni, nv = int(b["n_init"]), int(b["n_val"])
    s_start = ni + nv
    t0 = float(ts[s_start - 1])
    fts_k = apply_future_only_latency_intervention(ts, y, fts, t0=t0, kappa=kap, post_t0_oracle_visibility=False)

    W = V3_CONFIG.window_size_default
    methods_run = ["B2_4W", "FixedCurve_PU", "Survival_PU"]
    all_methods = ["B2_4W", "B2_4W_PostHocLogit", "B2_4W_PostHocRatio", "FixedCurve_PU", "Survival_PU"]
    acc = {m: [] for m in all_methods}

    # Exact pre-stream anchor parameters matching `models_and_baselines_v3.py` L1746-1755
    _, y_obs_start = observe_labels_causal(ts[:s_start], y[:s_start], fts[:s_start], t0)
    ages_start_d = (t0 - ts[:s_start]) / SECONDS_PER_DAY
    mat_init = ages_start_d >= 90.0
    if int(np.sum(mat_init)) >= 100:
        anchor_prev = float(np.clip(np.mean(y_obs_start[mat_init]) * 1.15, 0.03, 0.65))
    else:
        anchor_prev = float(np.clip(np.mean(y_obs_start) * 1.25, 0.03, 0.65))
    pw_base = float(np.clip((1.0 - anchor_prev) / max(anchor_prev, 0.03), 1.2, 8.0))
    log_pw_base = float(np.log(pw_base))

    # Pre-stream anchor maturity estimate at t0 for consistent pre-first-alarm scale
    est_init = CohortIncidenceEstimator().fit(ts[:s_start], fts[:s_start], y_obs_start, t0)
    c_init_mean = float(np.clip(np.mean(est_init.predict_maturity(np.maximum(t0 - ts[:s_start], 0.0))), V3_CONFIG.ht_floor_c, 1.0))

    def _eval_complete(z_vec: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, th_raw: np.ndarray, th_platt: np.ndarray, dec_override=None) -> Dict[str, float]:
        ev = evaluate_horizon_v3(
            y[s_start:], z_vec, p_raw, p_cal, eff[s_start:],
            per_commit_thresholds=th_raw, per_commit_decisions=dec_override,
        )
        ws = _window_ap_stats(y[s_start:], z_vec, p_raw, window_size=W)
        ev["win_ap"] = ws["win_ap_unweighted"]
        ev["ece_raw"] = float(compute_ece(y[s_start:], p_raw, n_bins=10))
        ev["slope_raw"] = _calibration_slope(y[s_start:], p_raw)
        ev["cal_slope"] = _calibration_slope(y[s_start:], p_cal)
        ev["f1_th_on_platt"] = float(
            precision_recall_fscore_support(y[s_start:], (p_cal >= th_platt).astype(int), average="binary", zero_division=0)[2]
        )
        return ev

    for sd in seeds:
        res = execute_within_host_counterfactual_suite_v3(
            X, ts, y, fts_k, ni, nv, host_type="mlp", methods=methods_run, fix_ts_for_init=fts, seed=sd
        )

        # Compute exact anchor init logits matching `execute_within_host_counterfactual_suite_v3`
        m_anc = build_host_backbone_v3("mlp", X[:16], seed=sd)
        m_anc, _ = train_supervised_v3(m_anc, X[:s_start], y_obs_start, pos_weight=pw_base, epochs=12, lr=8e-4, seed=sd)
        z_init_exact, _ = predict_logits_and_embeddings_v3(m_anc, X[:s_start])
        z_init_exact = np.asarray(z_init_exact, dtype=np.float64)

        for m in methods_run:
            mo = res["methods"][m]
            th_pl = _tune_causal_threshold_on_platted_probs(z_init_exact, mo["raw_logits"], ts, y, fts_k, s_start, window_size=W)
            acc[m].append(_eval_complete(mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], mo["tuned_th_vec"], th_pl, dec_override=mo["decisions_th"]))

        # Alarm-synchronized Post-Hoc Controls on B2_4W's EXACT unchanged weights/logits!
        b2_mo = res["methods"]["B2_4W"]
        b2_raw_z = np.asarray(b2_mo["raw_logits"], dtype=np.float64)
        exec_wins = b2_mo["executed_alarm_windows"]

        # 1) B2_4W_PostHocLogit: update shift delta ONLY at fired alarms (zero non-alarm jitter!)
        # Initialize pre-first-alarm shift delta_0 so mean(sigmoid(z_init + delta_0)) == anchor_prev
        lo0, hi0 = -8.0, 8.0
        for _ in range(28):
            mid0 = 0.5 * (lo0 + hi0)
            if float(np.mean(sigmoid_safe(z_init_exact + mid0))) < anchor_prev:
                lo0 = mid0
            else:
                hi0 = mid0
        cur_delta = 0.5 * (lo0 + hi0)
        z_init_logit = z_init_exact + cur_delta

        # 2) B2_4W_PostHocRatio: exact Proposition-1 ideal eta(x) = clip(sigmoid(z_B2 - log(pw_base)) / c_bar, 1e-4, 1-1e-4)
        cur_cbar = c_init_mean
        p_init_ratio = np.clip(sigmoid_safe(z_init_exact - log_pw_base) / cur_cbar, 1e-4, 1.0 - 1e-4)
        z_init_ratio = prob_to_logit_safe(p_init_ratio)

        z_logit_stream = np.zeros_like(b2_raw_z)
        z_ratio_stream = np.zeros_like(b2_raw_z)

        for w_rec in exec_wins:
            st, ed, fired = int(w_rec["start"]), int(w_rec["end"]), bool(w_rec["fired"])
            sl_e = slice(st - s_start, ed - s_start)
            z_w = b2_raw_z[sl_e]

            # Test-before-update using current active post-hoc parameters
            z_logit_stream[sl_e] = z_w + cur_delta
            p_r_w = np.clip(sigmoid_safe(z_w - log_pw_base) / cur_cbar, 1e-4, 1.0 - 1e-4)
            z_ratio_stream[sl_e] = prob_to_logit_safe(p_r_w)

            if fired:
                t_al = float(ts[ed - 1])
                _, y_obs_h = observe_labels_causal(ts[:ed], y[:ed], fts_k[:ed], t_al)
                est_al = CohortIncidenceEstimator().fit(ts[:ed], fts_k[:ed], y_obs_h, t_al)
                ages_al = np.maximum(t_al - ts[:ed], 0.0)
                c_al = est_al.predict_maturity(ages_al).astype(np.float64)

                w4_st = max(0, ed - 4 * W)
                c_4w = c_al[w4_st:ed]
                mat_m = c_al >= 0.55
                pi_anc = float(np.clip(np.mean(y_obs_h[mat_m] / np.maximum(c_al[mat_m], V3_CONFIG.ht_floor_c)), V3_CONFIG.pi_min, V3_CONFIG.pi_max)) if int(np.sum(mat_m)) >= 80 else anchor_prev
                pi_tgt, _, _ = estimate_ht_shrunk_prevalence(
                    y_obs=y_obs_h[w4_st:ed], c_maturity=c_4w,
                    sample_weights=np.ones(ed - w4_st, dtype=np.float64), anchor_prevalence=pi_anc
                )

                # Update scalar logit shift on trailing evaluated window's B2 logits (strictly causal <= ed)
                past_ref_z = b2_raw_z[max(0, ed - s_start - W) : ed - s_start]
                if len(past_ref_z) >= 20:
                    lo_s, hi_s = -8.0, 8.0
                    for _ in range(28):
                        mid_s = 0.5 * (lo_s + hi_s)
                        if float(np.mean(sigmoid_safe(past_ref_z + mid_s))) < pi_tgt:
                            lo_s = mid_s
                        else:
                            hi_s = mid_s
                    cur_delta = 0.5 * (lo_s + hi_s)

                # Update Proposition-1 uniform-maturity denominator c_bar_4W on trailing 4W window
                cur_cbar = float(np.clip(np.mean(c_4w), V3_CONFIG.ht_floor_c, 1.0))

        # Evaluate BOTH PostHoc controls under IDENTICAL 90d causal Platt + IDENTICAL 90d causal threshold search
        for m_ph, z_init_ph, z_str_ph in [
            ("B2_4W_PostHocLogit", z_init_logit, z_logit_stream),
            ("B2_4W_PostHocRatio", z_init_ratio, z_ratio_stream),
        ]:
            p_raw_ph = sigmoid_safe(z_str_ph)
            p_cal_ph, th_raw_ph = calibrate_causal_stream_probs_strict(z_init_ph, z_str_ph, ts, y, fts_k, s_start, window_size=W)
            th_pl_ph = _tune_causal_threshold_on_platted_probs(z_init_ph, z_str_ph, ts, y, fts_k, s_start, window_size=W)
            acc[m_ph].append(_eval_complete(z_str_ph, p_raw_ph, p_cal_ph, th_raw_ph, th_pl_ph))

    mets = (
        "ap", "win_ap", "f1", "f1_th", "f1_th_on_platt",
        "brier_raw", "ece_raw", "slope_raw",
        "brier_cal", "ece_cal", "cal_slope", "recall_at_20pct_effort",
    )
    out_s = {m: {met: float(np.nanmean([d[met] for d in acc[m]])) for met in mets} for m in all_methods}
    return k_key, sname, out_s


def run_audit_d_calibration_and_posthoc() -> Dict:
    mb = json.load(open(os.path.join(BASE_DIR, "cache", "benchmark_b_v3", "manifest_b_v3.json"), "r", encoding="utf-8"))
    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    cal_censoring = {
        "Benchmark_A_Natural": _measure_cal_pool_censoring([
            os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "qt_natural_25k_tab.npz"),
            os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "openstack_natural_13k_tab.npz"),
        ]),
        "Benchmark_B_Natural_27": _measure_cal_pool_censoring([
            os.path.join(b_dir, f"{e['name']}.npz") for e in mb["natural_streams"]
        ]),
        "Confirmatory_5": _measure_cal_pool_censoring(sorted(glob.glob(os.path.join(BASE_DIR, "cache", "confirmatory_v3", "*.npz")))),
    }
    rep10_paths = [os.path.join(b_dir, f"{s}.npz") for s in REPRESENTATIVE_10_STREAMS]
    for kap in (0.0, 0.5, 1.0, 2.0, 3.0):
        cal_censoring[f"RQ4a_10streams_kappa_{kap:.2f}"] = _measure_cal_pool_censoring(rep10_paths, kappa=kap)

    kappas = [0.0, 1.0, 3.0]
    seeds = [42, 43, 44, 45, 46]
    tasks = [(f"{kap:.2f}", sname, kap, seeds) for kap in kappas for sname in REPRESENTATIVE_10_STREAMS]
    print(f"  [Audit D] Running Symmetrical Raw/Platt + PostHoc Logit/Ratio controls ({len(tasks)} jobs x 5 seeds)...", flush=True)
    raw_by_k = {f"{kap:.2f}": {} for kap in kappas}
    t0w = time.time()
    for idx_t, task in enumerate(tasks, start=1):
        k_key, sname, out_s = _run_posthoc_b2_one_job(task)
        raw_by_k[k_key][sname] = out_s
        if idx_t % 5 == 0 or idx_t == len(tasks):
            print(f"    [Audit D] Progress: {idx_t}/{len(tasks)} ({time.time() - t0w:.1f}s)", flush=True)
    print(f"  [Audit D] Completed in {time.time() - t0w:.1f}s.", flush=True)

    orig_r4a = json.load(open(os.path.join(BASE_DIR, "results_v3", "rq4a_latency_intervention_v3.json"), "r", encoding="utf-8"))
    orig_fc = json.load(open(os.path.join(BASE_DIR, "results_v3", "rq4a_fixedcurve_control_v3.json"), "r", encoding="utf-8"))
    mets = (
        "ap", "win_ap", "f1", "f1_th", "f1_th_on_platt",
        "brier_raw", "ece_raw", "slope_raw",
        "brier_cal", "ece_cal", "cal_slope", "recall_at_20pct_effort",
    )
    all_methods = ("B2_4W", "B2_4W_PostHocLogit", "B2_4W_PostHocRatio", "FixedCurve_PU", "Survival_PU")
    by_k_summary = {}
    max_parity_err = 0.0
    for kap in kappas:
        kk = f"{kap:.2f}"
        means_k = {
            m: {met: float(np.nanmean([raw_by_k[kk][s][m][met] for s in REPRESENTATIVE_10_STREAMS])) for met in mets}
            for m in all_methods
        }
        for m_chk in ("B2_4W", "Survival_PU"):
            for met_chk in ("ap", "f1", "f1_th"):
                err = abs(means_k[m_chk][met_chk] - float(orig_r4a["results_by_kappa"][kk]["means"][m_chk][met_chk]))
                max_parity_err = max(max_parity_err, err)
                assert err < 1e-6, f"Parity mismatch on {kk}.{m_chk}.{met_chk}: err={err}"
        for met_chk in ("ap", "f1", "f1_th", "brier_cal"):
            err = abs(means_k["FixedCurve_PU"][met_chk] - float(orig_fc["results_by_kappa"][kk]["means"]["FixedCurve_PU"][met_chk]))
            max_parity_err = max(max_parity_err, err)
            assert err < 1e-6, f"Parity mismatch on FixedCurve {kk}.{met_chk}: err={err}"
        by_k_summary[kk] = {"means": means_k}

    return {
        "calibration_pool_90d_censoring_rates": cal_censoring,
        "max_parity_diff_vs_rq4a": float(max_parity_err),
        "posthoc_b2_by_kappa": by_k_summary,
        "posthoc_ratio_clipping_diagnostics": run_posthoc_ratio_clipping_diagnostics(),
    }


def run_posthoc_ratio_clipping_diagnostics(
    existing_by_kappa: Optional[Dict] = None,
) -> Dict:
    """Measure exact upper/lower clipping rates, within-window ceiling-tie rates, and
    unclipped-ratio AP/WinAP for `B2_4W_PostHocRatio` across kappas {0.0, 1.0, 3.0}
    (10 representative streams x 5 seeds 42..46) to isolate the exact source of the
    PostHoc Ratio within-window AP drop without altering `B2_4W` or archived JSONs.
    """
    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    W = V3_CONFIG.window_size_default
    kappas = [0.0, 1.0, 3.0]
    seeds = [42, 43, 44, 45, 46]

    out_by_k: Dict[str, Dict] = {}
    max_parity_err = 0.0
    t0c = time.time()
    print(f"  [Audit D-Clip] Running PostHoc Ratio clipping & tie-rate diagnostic ({len(kappas) * len(REPRESENTATIVE_10_STREAMS)} stream-kappa jobs x {len(seeds)} seeds)...", flush=True)

    for kap in kappas:
        kk = f"{kap:.2f}"
        stream_recs = []
        for sname in REPRESENTATIVE_10_STREAMS:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            s_start = ni + nv
            t0 = float(ts[s_start - 1])
            fts_k = apply_future_only_latency_intervention(ts, y, fts, t0=t0, kappa=kap, post_t0_oracle_visibility=False)
            y_ev = y[s_start:].astype(np.int32)

            _, y_obs_start = observe_labels_causal(ts[:s_start], y[:s_start], fts[:s_start], t0)
            ages_start_d = (t0 - ts[:s_start]) / SECONDS_PER_DAY
            mat_init = ages_start_d >= 90.0
            if int(np.sum(mat_init)) >= 100:
                anchor_prev = float(np.clip(np.mean(y_obs_start[mat_init]) * 1.15, 0.03, 0.65))
            else:
                anchor_prev = float(np.clip(np.mean(y_obs_start) * 1.25, 0.03, 0.65))
            pw_base = float(np.clip((1.0 - anchor_prev) / max(anchor_prev, 0.03), 1.2, 8.0))
            log_pw_base = float(np.log(pw_base))

            est_init = CohortIncidenceEstimator().fit(ts[:s_start], fts[:s_start], y_obs_start, t0)
            c_init_mean = float(np.clip(np.mean(est_init.predict_maturity(np.maximum(t0 - ts[:s_start], 0.0))), V3_CONFIG.ht_floor_c, 1.0))

            sd_recs = []
            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts_k, ni, nv, host_type="mlp", methods=["B2_4W"], fix_ts_for_init=fts, seed=sd
                )
                b2_mo = res["methods"]["B2_4W"]
                b2_raw_z = np.asarray(b2_mo["raw_logits"], dtype=np.float64)
                b2_raw_p = np.asarray(b2_mo["raw_probs"], dtype=np.float64)
                exec_wins = b2_mo["executed_alarm_windows"]

                cur_cbar = c_init_mean
                unclipped_stream = np.zeros_like(b2_raw_z)
                clipped_stream = np.zeros_like(b2_raw_z)
                cbar_stream = np.zeros_like(b2_raw_z)

                for w_rec in exec_wins:
                    st, ed, fired = int(w_rec["start"]), int(w_rec["end"]), bool(w_rec["fired"])
                    sl_e = slice(st - s_start, ed - s_start)
                    z_w = b2_raw_z[sl_e]
                    raw_r_w = sigmoid_safe(z_w - log_pw_base) / cur_cbar
                    p_r_w = np.clip(raw_r_w, 1e-4, 1.0 - 1e-4)
                    unclipped_stream[sl_e] = raw_r_w
                    clipped_stream[sl_e] = p_r_w
                    cbar_stream[sl_e] = cur_cbar

                    if fired:
                        t_al = float(ts[ed - 1])
                        _, y_obs_h = observe_labels_causal(ts[:ed], y[:ed], fts_k[:ed], t_al)
                        est_al = CohortIncidenceEstimator().fit(ts[:ed], fts_k[:ed], y_obs_h, t_al)
                        ages_al = np.maximum(t_al - ts[:ed], 0.0)
                        c_al = est_al.predict_maturity(ages_al).astype(np.float64)
                        w4_st = max(0, ed - 4 * W)
                        c_4w = c_al[w4_st:ed]
                        cur_cbar = float(np.clip(np.mean(c_4w), V3_CONFIG.ht_floor_c, 1.0))

                upper_mask = unclipped_stream >= (1.0 - 1e-4)
                lower_mask = unclipped_stream <= 1e-4

                # Measure exact within-window tie & upper-ceiling-tie rates + unclipped WinAP
                n_wins = 0
                n_wins_upper_tie = 0
                n_commits_upper_tied = 0
                n_commits_any_tie_ratio = 0
                n_commits_any_tie_b2 = 0
                unclip_win_aps = []
                clip_win_aps = []
                b2_win_aps = []

                for st_e in range(0, len(y_ev), W):
                    ed_e = min(st_e + W, len(y_ev))
                    n_wins += 1
                    u_w = upper_mask[st_e:ed_e]
                    if int(np.sum(u_w)) >= 2:
                        n_wins_upper_tie += 1
                        n_commits_upper_tied += int(np.sum(u_w))
                    _, cnt_r = np.unique(clipped_stream[st_e:ed_e], return_counts=True)
                    n_commits_any_tie_ratio += int(np.sum(cnt_r[cnt_r >= 2]))
                    _, cnt_b = np.unique(b2_raw_p[st_e:ed_e], return_counts=True)
                    n_commits_any_tie_b2 += int(np.sum(cnt_b[cnt_b >= 2]))

                    yw = y_ev[st_e:ed_e]
                    if int(np.sum(yw == 1)) > 0 and int(np.sum(yw == 0)) > 0:
                        b2_win_aps.append(float(average_precision_score(yw, b2_raw_p[st_e:ed_e])))
                        unclip_win_aps.append(float(average_precision_score(yw, unclipped_stream[st_e:ed_e])))
                        clip_win_aps.append(float(average_precision_score(yw, clipped_stream[st_e:ed_e])))

                sd_recs.append({
                    "b2_ap": float(average_precision_score(y_ev, b2_raw_p)),
                    "ratio_ap": float(average_precision_score(y_ev, clipped_stream)),
                    "ratio_unclipped_ap": float(average_precision_score(y_ev, unclipped_stream)),
                    "b2_win_ap": float(np.mean(b2_win_aps)),
                    "ratio_win_ap": float(np.mean(clip_win_aps)),
                    "ratio_unclipped_win_ap": float(np.mean(unclip_win_aps)),
                    "ratio_f1": float(precision_recall_fscore_support(y_ev, (clipped_stream >= 0.5).astype(int), average="binary", zero_division=0)[2]),
                    "ratio_unclipped_f1": float(precision_recall_fscore_support(y_ev, (unclipped_stream >= 0.5).astype(int), average="binary", zero_division=0)[2]),
                    "upper_clip_frac": float(np.mean(upper_mask)),
                    "lower_clip_frac": float(np.mean(lower_mask)),
                    "window_upper_tie_frac": float(n_wins_upper_tie / max(n_wins, 1)),
                    "commits_in_upper_tie_frac": float(n_commits_upper_tied / max(len(y_ev), 1)),
                    "ratio_exact_tie_commit_frac": float(n_commits_any_tie_ratio / max(len(y_ev), 1)),
                    "b2_exact_tie_commit_frac": float(n_commits_any_tie_b2 / max(len(y_ev), 1)),
                    "mean_cbar": float(np.mean(cbar_stream)),
                    "min_cbar": float(np.min(cbar_stream)),
                })
            keys = list(sd_recs[0].keys())
            stream_recs.append({k: float(np.mean([r[k] for r in sd_recs])) for k in keys})

        keys = list(stream_recs[0].keys())
        means_k = {k: float(np.mean([r[k] for r in stream_recs])) for k in keys}
        means_k["upper_clip_frac_sd_across_10streams"] = float(np.std([r["upper_clip_frac"] for r in stream_recs], ddof=1))
        means_k["ratio_exact_tie_commit_frac_sd_across_10streams"] = float(np.std([r["ratio_exact_tie_commit_frac"] for r in stream_recs], ddof=1))

        if existing_by_kappa is not None and kk in existing_by_kappa:
            ex = existing_by_kappa[kk]["means"]
            for a_k, m_k, met_k in [
                ("b2_ap", "B2_4W", "ap"),
                ("b2_win_ap", "B2_4W", "win_ap"),
                ("ratio_ap", "B2_4W_PostHocRatio", "ap"),
                ("ratio_win_ap", "B2_4W_PostHocRatio", "win_ap"),
                ("ratio_f1", "B2_4W_PostHocRatio", "f1"),
            ]:
                err = abs(means_k[a_k] - float(ex[m_k][met_k]))
                max_parity_err = max(max_parity_err, err)
                assert err < 1e-6, f"Clipping diagnostic parity mismatch on {kk}.{a_k}: err={err}"
        out_by_k[kk] = means_k

    print(f"  [Audit D-Clip] Completed in {time.time() - t0c:.1f}s (max parity err={max_parity_err:.2e}).", flush=True)
    return {
        "max_parity_err_vs_audit_d": float(max_parity_err),
        "by_kappa": out_by_k,
    }


# ==============================================================================
# Audit E (Points #11, #12): All 4 Controlled Shifts + Splice Timeline Stats
# ==============================================================================
def run_audit_e_controlled_shifts() -> Dict:
    bm = json.load(open(os.path.join(BASE_DIR, "results_v3", "benchmark_b_main_v3.json"), "r", encoding="utf-8"))
    cs = bm["controlled_shift_streams_mlp"]
    methods_e = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "B2_4W_Bal", "Survival_PU", "FixedCurve_PU", "Wait_30", "Wait_90", "ODaSC_Host", "BORB_Alarm", "BORB_Native", "Pure_nnPU", "ORB_OHT_Native"]
    per_stream = {}
    for st in ["abrupt_nova_zulip", "abrupt_ignite_cassandra", "abrupt_airflow_pulsar", "abrupt_deepjit_nova_neutron_pure13"]:
        b = np.load(os.path.join(BASE_DIR, "cache", "benchmark_b_v3", f"{st}.npz"))
        ts, fts, y = b["commit_ts"], b["fix_ts"], b["y_true"]
        onset = int(b["drift_onset_idx"])
        pre_pos_fin = int(np.sum((y[:onset] == 1) & np.isfinite(fts[:onset])))
        pre_cross = int(np.sum((y[:onset] == 1) & np.isfinite(fts[:onset]) & (fts[:onset] >= ts[onset])))
        per_stream[st] = {
            "N": int(len(ts)),
            "onset": onset,
            "gap_hours": float((ts[onset] - ts[onset - 1]) / 3600.0),
            "pre_revealed_pos": pre_pos_fin,
            "pre_fixes_revealed_post_onset": pre_cross,
            "frac_pre_fixes_crossing_onset": float(pre_cross / max(pre_pos_fin, 1)),
            "methods": {
                m: {
                    "post_onset_ap": float(cs[st]["methods"][m]["post_onset"]["ap"]),
                    "transient_1200_ap": float(cs[st]["methods"][m]["transient_1200"]["ap"]),
                }
                for m in methods_e if m in cs[st]["methods"]
            },
        }
    return per_stream


# ==============================================================================
# Audit F (Point #16 / R2 #5): Multi-Seed (5 Seeds) Whole-Stream vs Per-Window AP
# Across All 7 Core Methods + Controlled Scale-Removal Test + K=25 Paired Tests
# ==============================================================================
def run_audit_f_window_vs_whole_ap() -> Dict:
    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    W = V3_CONFIG.window_size_default
    seeds = [42, 43, 44, 45, 46]
    mlp_methods = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "FixedCurve_PU", "Survival_PU"]
    all_methods = mlp_methods + ["Full_LR_Adapt"]
    streams = sorted(NATURAL_STREAM_TO_DONOR_CLUSTER.keys())

    print(f"  [Audit F] Running 5-seed Per-Window AP & Scale-Removal Ablation ({len(streams)} streams x {len(seeds)} seeds)...", flush=True)
    t0f = time.time()
    per_stream_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    total_wins_sum = 0
    excluded_wins_sum = 0

    for idx_s, sname in enumerate(streams, start=1):
        b = np.load(os.path.join(b_dir, f"{sname}.npz"))
        X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
        ni, nv = int(b["n_init"]), int(b["n_val"])
        s_start = ni + nv
        y_ev = y[s_start:]

        seed_acc = {m: [] for m in all_methods}
        for sd in seeds:
            res = execute_within_host_counterfactual_suite_v3(
                X, ts, y, fts, ni, nv, host_type="mlp", methods=mlp_methods, seed=sd
            )
            for m in mlp_methods:
                mo = res["methods"][m]
                ws = _window_ap_stats(y_ev, mo["raw_logits"], mo["raw_probs"], window_size=W)
                seed_acc[m].append(ws)
            # Run Full_LR_Adapt on the exact same shared alarm trajectory!
            al_tuples = [(int(w["start"]), int(w["end"]), bool(w["fired"])) for w in res["methods"]["B2_4W"]["executed_alarm_windows"]]
            flr = run_linear_competitor_v3(X, ts, y, fts, eff, s_start, al_tuples, mode="adapt_4w", window_size=W, seed=sd)
            seed_acc["Full_LR_Adapt"].append(_window_ap_stats(y_ev, flr["raw_logits"], flr["raw_probs"], window_size=W))

        total_wins_sum += int(seed_acc["Survival_PU"][0]["n_windows_total"])
        excluded_wins_sum += int(seed_acc["Survival_PU"][0]["n_windows_excluded"])
        per_stream_metrics[sname] = {
            m: {
                k: float(np.nanmean([d[k] for d in seed_acc[m]]))
                for k in ("whole_stream_ap", "win_ap_unweighted", "win_ap_pos_weighted", "scale_norm_zscore_whole_ap", "scale_norm_rank_whole_ap")
            }
            for m in all_methods
        }
        if idx_s % 9 == 0 or idx_s == len(streams):
            print(f"    [Audit F] Progress: {idx_s}/{len(streams)} ({time.time() - t0f:.1f}s)", flush=True)

    clusters = sorted(set(NATURAL_STREAM_TO_DONOR_CLUSTER.values()))
    metric_keys = ("whole_stream_ap", "win_ap_unweighted", "win_ap_pos_weighted", "scale_norm_zscore_whole_ap", "scale_norm_rank_whole_ap")
    k25_means = {mk: {} for mk in metric_keys}
    s27_means = {mk: {} for mk in metric_keys}
    k25_vectors = {mk: {} for mk in metric_keys}

    for mk in metric_keys:
        for m in all_methods:
            s27_means[mk][m] = float(np.mean([per_stream_metrics[s][m][mk] for s in streams]))
            cl_v = []
            for cl in clusters:
                mems = [s for s, dc in NATURAL_STREAM_TO_DONOR_CLUSTER.items() if dc == cl]
                cl_v.append(float(np.mean([per_stream_metrics[s][m][mk] for s in mems])))
            k25_vectors[mk][m] = np.asarray(cl_v, dtype=np.float64)
            k25_means[mk][m] = float(np.mean(cl_v))

    # K=25 cluster-level paired Wilcoxon + 10,000-bootstrap 95% CI tests
    paired_tests = {}
    for mk in ("whole_stream_ap", "win_ap_unweighted", "win_ap_pos_weighted", "scale_norm_zscore_whole_ap", "scale_norm_rank_whole_ap"):
        paired_tests[mk] = {}
        for comp in ("B2_4W", "Wait_90", "FixedCurve_PU", "Full_LR_Adapt"):
            paired_tests[mk][f"Survival_PU_vs_{comp}"] = paired_cluster_inference_v3(
                k25_vectors[mk]["Survival_PU"], k25_vectors[mk][comp], higher_is_better=True, n_boot=10000, seed=42
            )

    # Keep original seed0 Benchmark B window AP summary + Benchmark A saved `.npz` whole vs window AP for full backward traceability
    bm = json.load(open(os.path.join(BASE_DIR, "results_v3", "benchmark_b_main_v3.json"), "r", encoding="utf-8"))
    ms_wt = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"]
    k25_s0_win = {}
    for m in ms_wt:
        cl_win = [
            float(np.mean([bm["natural_streams_mlp"][s]["seed0_window_trajectories"][m]["mean_defined_window_ap"] for s, dc in NATURAL_STREAM_TO_DONOR_CLUSTER.items() if dc == cl]))
            for cl in clusters
        ]
        k25_s0_win[m] = float(np.mean(cl_win))

    pred_dir = os.path.join(BASE_DIR, "results_v3", "commit_predictions", "benchmark_a")
    bench_a_cmp = {}
    for eco in ("qt_natural_25k", "openstack_natural_13k"):
        bench_a_cmp[eco] = {}
        for h_short, h_tag in [("att_cnn", "att_cnn__frozen_lr"), ("codebert", "codebert__frozen_lr"), ("deepjit_cnn", "deepjit_cnn__frozen_lr"), ("lightgbm", "lightgbm__none")]:
            z = np.load(os.path.join(pred_dir, f"{eco}__{h_tag}__seed42.npz"))
            yt = z["y_true"].astype(int)
            bench_a_cmp[eco][h_short] = {}
            for m in ms_wt:
                pr = z[f"{m}__raw_prob"]
                ws = _window_ap_stats(yt, prob_to_logit_safe(pr), pr, window_size=W)
                bench_a_cmp[eco][h_short][m] = {
                    "whole_stream_ap": ws["whole_stream_ap"],
                    "mean_window_ap": ws["win_ap_unweighted"],
                    "scale_norm_zscore_whole_ap": ws["scale_norm_zscore_whole_ap"],
                }

    return {
        "seeds_used": seeds,
        "window_size": W,
        "total_eval_windows_across_27streams": int(total_wins_sum),
        "excluded_single_class_windows": int(excluded_wins_sum),
        "excluded_window_fraction": float(excluded_wins_sum / max(total_wins_sum, 1)),
        "K25_5seed_means": k25_means,
        "S27_5seed_means": s27_means,
        "K25_5seed_paired_wilcoxon": paired_tests,
        "benchmark_b_K25_seed0_window_ap": k25_s0_win,
        "benchmark_a_seed42_whole_vs_window_ap": bench_a_cmp,
    }


# ==============================================================================
# Audit G (Point #19 / R2 #10): Whole-Stream & Per-Window Operational Matched
# Budgets + Exact Consumed LOC % + 5-Seed Multi-Seed Corroboration
# ==============================================================================
def run_audit_g_paired_net_gain() -> Dict:
    pred_dir = os.path.join(BASE_DIR, "results_v3", "commit_predictions", "benchmark_a")
    ba_j = json.load(open(os.path.join(BASE_DIR, "results_v3", "benchmark_a_multihost_v3.json"), "r", encoding="utf-8"))
    W = V3_CONFIG.window_size_default

    # Index 5-seed Benchmark A effort metrics
    ba_5seed = {}
    for v in ba_j["cells"].values():
        if v["stream_name"] in ("qt_natural_25k", "openstack_natural_13k"):
            if (v["host_type"] == "lightgbm" and v.get("skip_mode", "none") == "none") or (v["host_type"] != "lightgbm" and v.get("skip_mode") == "frozen_lr"):
                ba_5seed[(v["stream_name"], v["host_type"])] = {
                    "b2_r20_5seed": float(v["methods"]["B2_4W"]["full_stream"]["recall_at_20pct_effort"]),
                    "spu_r20_5seed": float(v["methods"]["Survival_PU"]["full_stream"]["recall_at_20pct_effort"]),
                    "delta_r20_5seed": float(v["methods"]["Survival_PU"]["full_stream"]["recall_at_20pct_effort"] - v["methods"]["B2_4W"]["full_stream"]["recall_at_20pct_effort"]),
                    "b2_win_r20_5seed": float(v["methods"]["B2_4W"]["full_stream"]["windowed_batch_recall_at_20pct_effort"]),
                    "spu_win_r20_5seed": float(v["methods"]["Survival_PU"]["full_stream"]["windowed_batch_recall_at_20pct_effort"]),
                    "delta_win_r20_5seed": float(v["methods"]["Survival_PU"]["full_stream"]["windowed_batch_recall_at_20pct_effort"] - v["methods"]["B2_4W"]["full_stream"]["windowed_batch_recall_at_20pct_effort"]),
                }

    out = {}
    for eco in ("qt_natural_25k", "openstack_natural_13k"):
        out[eco] = {}
        for h_short, h_tag in [("att_cnn", "att_cnn__frozen_lr"), ("codebert", "codebert__frozen_lr"), ("deepjit_cnn", "deepjit_cnn__frozen_lr"), ("lightgbm", "lightgbm__none")]:
            z = np.load(os.path.join(pred_dir, f"{eco}__{h_tag}__seed42.npz"))
            yt = z["y_true"].astype(int)
            eff = np.maximum(z["effort"].astype(np.float64), 1.0)
            p_b2 = z["B2_4W__raw_prob"].astype(np.float64)
            p_spu = z["Survival_PU__raw_prob"].astype(np.float64)
            th_b2 = z["B2_4W__tuned_th"].astype(np.float64)
            th_spu = z["Survival_PU__tuned_th"].astype(np.float64)
            tot_eff = float(np.sum(eff))

            def _pair_stats(s_b2: np.ndarray, s_spu: np.ndarray) -> Dict:
                tp_r = int(np.sum((yt == 1) & (s_b2 == 0) & (s_spu == 1)))
                tp_h = int(np.sum((yt == 1) & (s_b2 == 1) & (s_spu == 0)))
                fp_a = int(np.sum((yt == 0) & (s_b2 == 0) & (s_spu == 1)))
                fp_r = int(np.sum((yt == 0) & (s_b2 == 1) & (s_spu == 0)))
                return {
                    "b2_tp": int(np.sum((yt == 1) & (s_b2 == 1))),
                    "spu_tp": int(np.sum((yt == 1) & (s_spu == 1))),
                    "tp_rescued": tp_r,
                    "tp_harmed": tp_h,
                    "net_tp": tp_r - tp_h,
                    "fp_added": fp_a,
                    "fp_removed": fp_r,
                    "net_fp": fp_a - fp_r,
                    "b2_loc_pct": float(np.sum(eff[s_b2 == 1]) / tot_eff),
                    "spu_loc_pct": float(np.sum(eff[s_spu == 1]) / tot_eff),
                }

            def _eff_sel_slice(prob_v: np.ndarray, eff_v: np.ndarray, frac: float = 0.20) -> np.ndarray:
                # Strict greedy non-splitting stop: sort by descending risk density p/LOC (ties: smaller LOC),
                # include commits while cumulative LOC <= frac * slice_LOC (first commit exceeding budget stops selection).
                bgt_s = frac * float(np.sum(eff_v))
                ord_i = np.lexsort((eff_v, -(prob_v / eff_v)))
                cs = np.cumsum(eff_v[ord_i])
                sel = np.zeros(len(prob_v), dtype=int)
                sel[ord_i[cs <= bgt_s]] = 1
                return sel

            rec = {
                "n_eval": int(len(yt)),
                "n_defective": int(np.sum(yt == 1)),
                "fixed_th_050": _pair_stats((p_b2 >= 0.50).astype(int), (p_spu >= 0.50).astype(int)),
                "causal_tuned_th": _pair_stats((p_b2 >= th_b2).astype(int), (p_spu >= th_spu).astype(int)),
            }
            # 1) Whole-stream retrospective budgets
            for frac, lbl in [(0.10, "matched_top10pct_commits"), (0.20, "matched_top20pct_commits")]:
                k = int(round(frac * len(yt)))
                sb2 = np.zeros(len(yt), dtype=int); sb2[np.argsort(-p_b2, kind="mergesort")[:k]] = 1
                sspu = np.zeros(len(yt), dtype=int); sspu[np.argsort(-p_spu, kind="mergesort")[:k]] = 1
                rec[lbl] = _pair_stats(sb2, sspu)
            rec["matched_20pct_effort_loc"] = _pair_stats(_eff_sel_slice(p_b2, eff, 0.20), _eff_sel_slice(p_spu, eff, 0.20))

            # 2) Per-window operational 400-commit budgets (direct CI sprint review capacity!)
            for frac, lbl in [(0.10, "win_top10pct_commits"), (0.20, "win_top20pct_commits")]:
                sb2_w = np.zeros(len(yt), dtype=int)
                sspu_w = np.zeros(len(yt), dtype=int)
                for st in range(0, len(yt), W):
                    ed = min(st + W, len(yt))
                    kw = max(1, int(round(frac * (ed - st))))
                    sb2_w[st + np.argsort(-p_b2[st:ed], kind="mergesort")[:kw]] = 1
                    sspu_w[st + np.argsort(-p_spu[st:ed], kind="mergesort")[:kw]] = 1
                rec[lbl] = _pair_stats(sb2_w, sspu_w)

            sb2_we = np.zeros(len(yt), dtype=int)
            sspu_we = np.zeros(len(yt), dtype=int)
            for st in range(0, len(yt), W):
                ed = min(st + W, len(yt))
                sb2_we[st:ed] = _eff_sel_slice(p_b2[st:ed], eff[st:ed], 0.20)
                sspu_we[st:ed] = _eff_sel_slice(p_spu[st:ed], eff[st:ed], 0.20)
            rec["win_20pct_effort_loc"] = _pair_stats(sb2_we, sspu_we)
            rec["multiseed_5seed_effort"] = ba_5seed[(eco, h_short)]
            out[eco][h_short] = rec
    return out


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Run companion mechanism & forensic audits.")
    parser.add_argument(
        "--update-ratio-clipping-only",
        action="store_true",
        help="Compute only the PostHoc Ratio clipping/tie-rate diagnostic (B2_4W only, ~3 min) and merge into reviewer_mechanism_audits_v3.json.",
    )
    args = parser.parse_args()
    out_p = os.path.join(BASE_DIR, "results_v3", "reviewer_mechanism_audits_v3.json")

    if args.update_ratio_clipping_only:
        t0 = time.time()
        payload = json.load(open(out_p, "r", encoding="utf-8"))
        existing_by_k = payload["audit_d_calibration_and_posthoc"]["posthoc_b2_by_kappa"]
        clip_diag = run_posthoc_ratio_clipping_diagnostics(existing_by_kappa=existing_by_k)
        payload["audit_d_calibration_and_posthoc"]["posthoc_ratio_clipping_diagnostics"] = clip_diag
        payload["provenance"] = build_provenance_header(BASE_DIR)
        save_json_atomic(out_p, payload)
        print("\n=== PostHoc Ratio Clipping & Tie-Rate Diagnostic Summary ===", flush=True)
        for kk, d in clip_diag["by_kappa"].items():
            print(
                f"  kappa={kk}: upper_clip={d['upper_clip_frac']:.4f} (+/-{d['upper_clip_frac_sd_across_10streams']:.4f}) | "
                f"win_with_upper_tie={d['window_upper_tie_frac']:.4f} | ratio_tie_commits={d['ratio_exact_tie_commit_frac']:.4f} (B2={d['b2_exact_tie_commit_frac']:.4f}) | "
                f"mean_cbar={d['mean_cbar']:.4f} | WinAP: B2={d['b2_win_ap']:.4f}, UnclipRatio={d['ratio_unclipped_win_ap']:.4f}, ClipRatio={d['ratio_win_ap']:.4f} | "
                f"WholeAP: B2={d['b2_ap']:.4f}, UnclipRatio={d['ratio_unclipped_ap']:.4f}, ClipRatio={d['ratio_ap']:.4f} | "
                f"F1(0.5): Unclip={d['ratio_unclipped_f1']:.4f}, Clip={d['ratio_f1']:.4f}",
                flush=True,
            )
        print(f"Updated {out_p} in {time.time() - t0:.1f}s.", flush=True)
        return

    t0 = time.time()
    print("=== Running Round-2 Comprehensive Reviewer Mechanism & Forensic Audits ===", flush=True)
    audit_a = run_audit_a_apachejit_funnel()
    print("  [Audit A] ApacheJIT Funnel & Standalone-3 vs Monorepo-2 DONE.", flush=True)
    audit_b = run_audit_b_followup_cutoffs()
    print(f"  [Audit B] 34 Natural Streams Follow-Up Proxy & Full Bench A Sensitivity DONE.", flush=True)
    audit_c = run_audit_c_stage123_mechanisms()
    print(f"  [Audit C] Unrevealed-Only Stage 1-3 & FixedCurve Head-to-Head ({audit_c['n_alarms_audited']} alarms, {audit_c['unique_commits_audited']} unique commits) DONE.", flush=True)
    audit_e = run_audit_e_controlled_shifts()
    print("  [Audit E] 4 Controlled Shifts & Splice Timeline Stats DONE.", flush=True)
    audit_g = run_audit_g_paired_net_gain()
    print("  [Audit G] Whole-Stream & Per-Window Operational Matched Budgets DONE.", flush=True)
    audit_d = run_audit_d_calibration_and_posthoc()
    print(f"  [Audit D] Symmetrical Raw/Platt & PostHoc Logit/Ratio Controls (parity err={audit_d['max_parity_diff_vs_rq4a']:.2e}) DONE.", flush=True)
    audit_f = run_audit_f_window_vs_whole_ap()
    print("  [Audit F] 5-Seed K=25 Per-Window AP & Controlled Scale-Removal Ablations DONE.", flush=True)

    payload = {
        "provenance": build_provenance_header(BASE_DIR),
        "note": "Read-only Round-2 companion mechanism, data-funnel, follow-up sensitivity, unrevealed-only Stage 1-3 audit, symmetrical Raw/Platt & PostHoc Logit/Ratio controls, 5-seed K=25 window-AP & scale-removal test, and whole-stream/per-window matched-budget audits generated without modifying src_v3/ or archived results_v3 JSONs.",
        "audit_a_apachejit_funnel": audit_a,
        "audit_b_followup_cutoffs": audit_b,
        "audit_c_stage123_mechanisms": audit_c,
        "audit_d_calibration_and_posthoc": audit_d,
        "audit_e_controlled_shifts": audit_e,
        "audit_f_window_vs_whole_ap": audit_f,
        "audit_g_paired_net_gain": audit_g,
    }
    out_p = os.path.join(BASE_DIR, "results_v3", "reviewer_mechanism_audits_v3.json")
    save_json_atomic(out_p, payload)

    print("\n================ ROUND-2 KEY FINDINGS SUMMARY ================", flush=True)
    print("Audit C (Stage 1 c_hat MAE): SPU =", round(audit_c["c_hat_mae_spu_alarm_mean"], 4), "+/-", round(audit_c["c_hat_mae_spu_alarm_sd"], 4),
          "| FixedCurve =", round(audit_c["c_hat_mae_fc_alarm_mean"], 4), "+/-", round(audit_c["c_hat_mae_fc_alarm_sd"], 4))
    print("Audit C (Unrevealed-Only Brier):", json.dumps(audit_c["age_stratified_unrevealed_only_audit"], indent=2))
    print("Audit C (All-Pool Writeback Control):", json.dumps(audit_c["age_stratified_soft_target_audit"]["all_ages"], indent=2))
    for kk in ("0.00", "1.00", "3.00"):
        print(f"Audit D Symmetrical kappa={kk}:")
        for m, md in audit_d["posthoc_b2_by_kappa"][kk]["means"].items():
            print(f"  {m:22s}: AP={md['ap']:.4f} WinAP={md['win_ap']:.4f} F1={md['f1']:.4f} F1th={md['f1_th']:.4f} F1thP={md['f1_th_on_platt']:.4f} | Raw(Br={md['brier_raw']:.4f},ECE={md['ece_raw']:.4f},Sl={md['slope_raw']:.3f}) | 90dPlatt(Br={md['brier_cal']:.4f},ECE={md['ece_cal']:.4f},Sl={md['cal_slope']:.3f})")
    print("Audit F 5-Seed K=25 Means:", json.dumps(audit_f["K25_5seed_means"], indent=2))
    print("Audit F 5-Seed K=25 Paired Wilcoxon (SPU vs B2_4W):", {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v["Survival_PU_vs_B2_4W"].items() if kk in ("mean_diff", "ci_low", "ci_high", "wins", "losses", "p_wilcoxon")} for k, v in audit_f["K25_5seed_paired_wilcoxon"].items()})
    for eco in ("qt_natural_25k", "openstack_natural_13k"):
        print(f"Audit G ({eco}):")
        for h, hd in audit_g[eco].items():
            print(f"  {h:12s}: Whole10={hd['matched_top10pct_commits']['net_tp']:+d} Win10={hd['win_top10pct_commits']['net_tp']:+d} | WholeEff={hd['matched_20pct_effort_loc']['net_tp']:+d} (LOC {hd['matched_20pct_effort_loc']['b2_loc_pct']:.1%}/{hd['matched_20pct_effort_loc']['spu_loc_pct']:.1%}) WinEff={hd['win_20pct_effort_loc']['net_tp']:+d} | 5sWinR20={hd['multiseed_5seed_effort']['delta_win_r20_5seed']:+.4f}")
    print(f"Saved {out_p} in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
