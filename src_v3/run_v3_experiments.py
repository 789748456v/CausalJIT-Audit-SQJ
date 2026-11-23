"""
Master Experimental Execution Suite for N05 V3 (Software Quality Journal).

Closes Codex Round 2, Round 3, & Round 3b Findings (R3-C1..C9, R3-M2, R3-M5, R3-M6, R3b-C1, R3b-M2..M4, R3b-M6):
- Consumes per-commit causal `tuned_th_vec` and `decisions_th` across ALL evaluation horizons and saves
  them in commit-level `.npz` artifacts (`R3-C1`).
- Evaluates Track 2 stream & simple baseline competitors (`ORB_OHT_Native`, `BORB_Native`/`BORB_Alarm`,
  `ODaSC_OHT_Native`, and linear baselines) via a single canonical `run_single_seed_stream_all_methods`
  helper shared with Check 2 so 100% of runner methods are prefix-invariance verified (`R3-C3`, `R3b-C1`, `R3b-M4`).
- Uses `aggregate_seed_horizon_metrics` (`n_defective` schema) across all evaluation horizons so missing/NaN
  metrics on two-class evaluation slices raise `ValueError` immediately (`R3b-M3`).
- Logs full phase-decomposed timing (`init_fit_ms`, `stream_predict_and_embed_ms`, `drift_monitor_ms`,
  `adapt_only_wall_ms`, `calibration_ms`, `end_to_end_eval_ms`, `alarm_step_total_ms_p50/p95`) and exact
  `requested_rounds` / `actual_iterations` / `trees_built` (`R3b-M2`, `R3b-M6`).
- Freezes `FAMILY_3` component/control Holm-Bonferroni comparisons (`Survival_PU` vs `FixedCurve_PU` and
  `Teacher_Only_Matched`) and `PRE_REGISTERED_INTERPRETATION_BOUNDARIES` prior to unlocking confirmatory data.
- Separates future-only finite latency scaling (`kappa >= 0.0` with `fix_ts_for_init=fts`) from
  `oracle_post_t0` visibility control (`R3-C6`).
- Runs unconfounded 2x2 DriftLens factorial (`pct=95, cooldown=1` across all 4 cells) + `N46_Ref_98_CD0`,
  measures stationary FAR on the exact `m_causal_init` embeddings (including rolling reference refresh),
  reads `executed_alarm_windows` per method, and separates `undetected_shift_rate` from conditional delay (`R3-C7`).
- Runs `Family 1`, `Family 2`, and `Family 3` on `umbrella-4` (`K=4`) and `repo-5` (`K=5`) in Phase C under the
  pre-declared directional & effect-size replication plan with 100% exact signed-rank enumeration (`R3-C8`).
- Attaches an immutable SHA-256 `run_fingerprint` to every resumable per-stream checkpoint and rejects/
  recomputes stale checkpoints (`R3-C9`).
- Evaluates Benchmark A sensitivity streams on both native horizons and common evaluation commit IDs (`R3-M2`).
- Orients `brier_cal` / `ece_cal` with `higher_is_better=False` (`R3-M5`).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_v3 import (
    FAMILY_1_PARADOX_PAIRS,
    FAMILY_2_COMPETITOR_PAIRS,
    FAMILY_3_COMPONENT_CONTROL_PAIRS,
    FAMILY_3_TRACK1_CONTROL_PAIRS,
    NATURAL_STREAM_TO_DONOR_CLUSTER,
    PRE_REGISTERED_INTERPRETATION_BOUNDARIES,
    SECONDS_PER_DAY,
    V3_CONFIG,
)
from data_audit_v3 import sha256_file
from metrics_and_stats_v3 import (
    aggregate_seed_horizon_metrics,
    aggregate_streams_to_donor_clusters,
    evaluate_horizon_v3,
    evaluate_window_trajectories_v3,
    paired_cluster_inference_v3,
    run_holm_bonferroni_family,
)
from models_and_baselines_v3 import (
    DriftMonitorV3,
    apply_future_only_latency_intervention,
    build_host_backbone_v3,
    execute_within_host_counterfactual_suite_v3,
    observe_labels_at_time,
    predict_logits_and_embeddings_v3,
    run_borb_competitor_v3,
    run_linear_competitor_v3,
    run_odasc_oht_native_v3,
    run_orb_oht_native_v3,
    sigmoid_safe,
    train_supervised_v3,
)


TRACK1_MAIN_METHODS: List[str] = [
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

ARTIFACT_SAVE_METHODS: Tuple[str, ...] = (
    "B0_Frozen",
    "B1b_4W_Oracle",
    "B2_4W",
    "Wait_90",
    "Pure_nnPU",
    "Survival_PU",
    "Full_LR_Adapt",
    "BORB_Native",
    "ORB_OHT_Native",
    "ODaSC_OHT_Native",
)

CORE_EXECUTION_MODULES: Tuple[str, ...] = (
    "config_v3.py",
    "data_audit_v3.py",
    "metrics_and_stats_v3.py",
    "survival_estimators_v3.py",
    "stream_competitors_v3.py",
    "models_and_baselines_v3.py",
    "run_v3_experiments.py",
)

LOWER_IS_BETTER_ENDPOINTS: Tuple[str, ...] = (
    "brier_cal",
    "ece_cal",
    "brier_raw",
    "ece_raw",
    "ifa",
)


def build_provenance_header(base_dir: str, extra_manifest_paths: Optional[List[str]] = None) -> Dict:
    src_v3_dir = os.path.join(base_dir, "src_v3")
    code_hashes = {}
    for py_p in sorted(glob.glob(os.path.join(src_v3_dir, "*.py"))):
        code_hashes[os.path.basename(py_p)] = sha256_file(py_p)
    data_hashes = {}
    for mp in (extra_manifest_paths or []):
        if os.path.exists(mp):
            data_hashes[os.path.relpath(mp, base_dir).replace("\\", "/")] = sha256_file(mp)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v3_config": json.loads(json.dumps(V3_CONFIG.to_dict())),
        "code_sha256": code_hashes,
        "manifest_sha256": data_hashes,
    }


def compute_task_run_fingerprint(
    base_dir: str,
    bundle_path: str,
    host_type: str,
    skip_mode: str,
    seeds: List[int],
    include_track2: bool,
    methods_track1: List[str],
    window_size: int = V3_CONFIG.window_size_default,
) -> str:
    """
    Immutable SHA-256 fingerprint binding code, V3_CONFIG, data bundle, seed list, and method list (`R3-C9`).
    """
    src_v3_dir = os.path.join(base_dir, "src_v3")
    core_hashes = {
        fn: sha256_file(os.path.join(src_v3_dir, fn))
        for fn in CORE_EXECUTION_MODULES
        if os.path.exists(os.path.join(src_v3_dir, fn))
    }
    spec_obj = {
        "core_code_sha256": core_hashes,
        "bundle_sha256": sha256_file(bundle_path),
        "v3_config": json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True)),
        "host_type": str(host_type),
        "skip_mode": str(skip_mode),
        "seeds": [int(s) for s in seeds],
        "include_track2": bool(include_track2),
        "methods_track1": list(methods_track1),
        "window_size": int(window_size),
    }
    canon = json.dumps(spec_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_verified_checkpoint_or_none(ckpt_path: str, expected_fingerprint: str) -> Optional[Dict]:
    """
    Loads a resumable per-stream checkpoint ONLY if its `run_fingerprint` matches `expected_fingerprint`
    bit-for-bit (`R3-C9`). Stale or unverified checkpoints return `None` and trigger fresh recomputation.
    """
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj.get("run_fingerprint") == expected_fingerprint:
            return obj
    except Exception:
        return None
    return None


def save_json_atomic(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_p = f"{path}.tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_p, path)


def _compute_generation_ids(n_eval: int, stream_start: int, shared_windows: List[Dict], is_frozen: bool = False) -> np.ndarray:
    gen = np.zeros(n_eval, dtype=np.int32)
    if is_frozen:
        return gen
    cur_g = 0
    for w in shared_windows:
        ws, we = int(w["start"]) - stream_start, int(w["end"]) - stream_start
        gen[ws:we] = cur_g
        if w.get("fired", False):
            cur_g += 1
    return gen


def run_single_seed_stream_all_methods(
    X: np.ndarray,
    X_tab: np.ndarray,
    X_la: np.ndarray,
    X_ch: np.ndarray,
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
    effort: np.ndarray,
    n_init: int,
    n_val: int,
    drift_onset_idx: int = -1,
    host_type: str = "mlp",
    meta: Optional[Dict] = None,
    skip_mode: str = "none",
    include_track2: bool = True,
    methods_track1: Optional[List[str]] = None,
    window_size: int = V3_CONFIG.window_size_default,
    seed: int = 42,
    device=None,
) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Single Canonical Single-Seed Execution Entrypoint (`R3b-C1`, `R3b-M6`):
      - Executes ALL active Track 1 methods AND (when `include_track2=True`) ALL 11 Track 2 competitors
        (`LOC_Only`, `LApredict_B0`, `LApredict_Adapt`, `Churn_LR_B0`, `Churn_LR_Adapt`, `Full_LR_B0`,
        `Full_LR_Adapt`, `BORB_Native`, `BORB_Alarm`, `ORB_OHT_Native`, `ODaSC_OHT_Native`), attributing
        `shared_monitor_ms` consistently to every alarm-triggered Track 2 method.
      - Imported directly by `verify_mechanisms_and_protocol_v3.py` Check 2 so EVERY method executed by
        the master runner is automatically verified for strict prefix invariance without hardcoded lists!
    """
    active_m_t1 = list(methods_track1 or TRACK1_MAIN_METHODS)
    stream_start = int(n_init) + int(n_val)
    suite = execute_within_host_counterfactual_suite_v3(
        X=X,
        commit_ts=commit_ts,
        y_true=y_true,
        fix_ts=fix_ts,
        n_init=n_init,
        n_val=n_val,
        host_type=host_type,
        meta=meta,
        skip_mode=skip_mode,
        methods=active_m_t1,
        window_size=window_size,
        drift_onset_idx=drift_onset_idx,
        seed=seed,
        device=device,
    )
    combined_outputs: Dict[str, Dict] = dict(suite["methods"])

    if include_track2:
        aw = [(w["start"], w["end"], w["fired"]) for w in suite["shared_alarm_windows"]]
        sh_anc_init = float(suite.get("shared_anchor_init_ms", 0.0))
        sh_mon_init = float(suite.get("shared_monitor_fit_ms", 0.0))
        sh_anc_eval = float(suite.get("shared_anchor_eval_embed_ms", 0.0))
        sh_mon_ms = float(suite.get("shared_monitor_ms", 0.0))
        combined_outputs["LOC_Only"] = run_linear_competitor_v3(
            X_la, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="loc_only", window_size=window_size, seed=seed
        )
        combined_outputs["LApredict_B0"] = run_linear_competitor_v3(
            X_la, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="frozen", window_size=window_size, seed=seed
        )
        combined_outputs["LApredict_Adapt"] = run_linear_competitor_v3(
            X_la, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="adapt_4w",
            window_size=window_size,
            shared_anchor_init_ms=sh_anc_init,
            shared_monitor_init_ms=sh_mon_init,
            shared_anchor_eval_embed_ms=sh_anc_eval,
            shared_monitor_ms=sh_mon_ms,
            seed=seed,
        )
        combined_outputs["Churn_LR_B0"] = run_linear_competitor_v3(
            X_ch, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="frozen", window_size=window_size, seed=seed
        )
        combined_outputs["Churn_LR_Adapt"] = run_linear_competitor_v3(
            X_ch, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="adapt_4w",
            window_size=window_size,
            shared_anchor_init_ms=sh_anc_init,
            shared_monitor_init_ms=sh_mon_init,
            shared_anchor_eval_embed_ms=sh_anc_eval,
            shared_monitor_ms=sh_mon_ms,
            seed=seed,
        )
        combined_outputs["Full_LR_B0"] = run_linear_competitor_v3(
            X_tab, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="frozen", window_size=window_size, seed=seed
        )
        combined_outputs["Full_LR_Adapt"] = run_linear_competitor_v3(
            X_tab, commit_ts, y_true, fix_ts, effort, stream_start, aw, mode="adapt_4w",
            window_size=window_size,
            shared_anchor_init_ms=sh_anc_init,
            shared_monitor_init_ms=sh_mon_init,
            shared_anchor_eval_embed_ms=sh_anc_eval,
            shared_monitor_ms=sh_mon_ms,
            seed=seed,
        )
        combined_outputs["BORB_Native"] = run_borb_competitor_v3(
            X_tab, commit_ts, y_true, fix_ts, stream_start, update_mode="native", window_size_cal=window_size, seed=seed
        )
        combined_outputs["BORB_Alarm"] = run_borb_competitor_v3(
            X_tab, commit_ts, y_true, fix_ts, stream_start, alarm_windows=aw, update_mode="alarm",
            window_size_cal=window_size,
            shared_anchor_init_ms=sh_anc_init,
            shared_monitor_init_ms=sh_mon_init,
            shared_anchor_eval_embed_ms=sh_anc_eval,
            shared_monitor_ms=sh_mon_ms,
            seed=seed,
        )
        combined_outputs["ORB_OHT_Native"] = run_orb_oht_native_v3(
            X_tab, commit_ts, y_true, fix_ts, stream_start, window_size_cal=window_size, seed=seed
        )
        combined_outputs["ODaSC_OHT_Native"] = run_odasc_oht_native_v3(
            X_tab, commit_ts, y_true, fix_ts, stream_start, window_size_cal=window_size, seed=seed
        )

    return suite, combined_outputs


def _agg_numeric_dict_list(dlist: List[Dict]) -> Dict[str, Optional[float]]:
    if not dlist:
        return {}
    keys = list(dlist[0].keys())
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        fin_vals = [float(d[k]) for d in dlist if d.get(k) is not None and np.isfinite(float(d[k]))]
        out[k] = float(np.mean(fin_vals)) if fin_vals else None
    return out


def evaluate_and_aggregate_stream_runs(
    stream_name: str,
    bundle_path: str,
    host_type: str,
    seeds: List[int],
    meta: Optional[Dict] = None,
    skip_mode: str = "none",
    include_track2: bool = True,
    methods_track1: Optional[List[str]] = None,
    window_size: int = V3_CONFIG.window_size_default,
    save_commit_artifact_dir: Optional[str] = None,
    common_eval_mask: Optional[np.ndarray] = None,
    run_fingerprint: Optional[str] = None,
) -> Dict:
    """
    Run `seeds` on a single stream bundle, evaluate full-horizon & post-onset/transient-1200 horizons
    using strict per-commit `tuned_th_vec` & `decisions_th` (`R3-C1`), enforce `aggregate_seed_horizon_metrics`
    (`R3b-M3`), save seed-0 commit-level `.npz` artifact, and return aggregated stats.
    """
    b = np.load(bundle_path)
    X = b["X"].astype(np.float32)
    X_tab = b["X_tab"].astype(np.float32) if "X_tab" in b else X
    X_la = b["X_la"].astype(np.float32)
    X_ch = b["X_churn"].astype(np.float32)
    ts = b["commit_ts"].astype(np.float64)
    y = b["y_true"].astype(np.int32)
    fts = b["fix_ts"].astype(np.float64)
    eff = b["effort"].astype(np.float32)
    n_init = int(b["n_init"])
    n_val = int(b["n_val"])
    onset = int(b["drift_onset_idx"])
    stream_start = n_init + n_val

    y_eval = y[stream_start:]
    eff_eval = eff[stream_start:]
    active_m_t1 = list(methods_track1 or TRACK1_MAIN_METHODS)

    has_onset = onset >= stream_start and (len(y) - onset) >= 200
    onset_rel = (onset - stream_start) if has_onset else 0
    trans_end_rel = min(onset_rel + V3_CONFIG.fixed_transient_commits, len(y_eval)) if has_onset else min(V3_CONFIG.fixed_transient_commits, len(y_eval))

    per_method_seed_metrics: Dict[str, List[Dict]] = {}
    per_method_post_onset: Dict[str, List[Dict]] = {}
    per_method_transient1200: Dict[str, List[Dict]] = {}
    per_method_common_subset: Dict[str, List[Dict]] = {}
    per_method_compute: Dict[str, List[Dict]] = {}
    local_forks: List[Dict] = []
    seed0_traj: Dict[str, Dict] = {}

    for s_idx, sd in enumerate(seeds):
        suite, combined_outputs = run_single_seed_stream_all_methods(
            X=X,
            X_tab=X_tab,
            X_la=X_la,
            X_ch=X_ch,
            commit_ts=ts,
            y_true=y,
            fix_ts=fts,
            effort=eff,
            n_init=n_init,
            n_val=n_val,
            drift_onset_idx=onset,
            host_type=host_type,
            meta=meta,
            skip_mode=skip_mode,
            include_track2=include_track2,
            methods_track1=active_m_t1,
            window_size=window_size,
            seed=sd,
        )
        if suite["local_fork_summary"] is not None:
            lf_clean = {k: v for k, v in suite["local_fork_summary"].items() if isinstance(v, (int, float, bool))}
            local_forks.append(lf_clean)

        if s_idx == 0 and save_commit_artifact_dir is not None:
            os.makedirs(save_commit_artifact_dir, exist_ok=True)
            art_dict = {
                "commit_idx": np.arange(stream_start, len(y), dtype=np.int32),
                "prediction_timestamp": ts[stream_start:].astype(np.float64),
                "y_true": y_eval.astype(np.int32),
                "effort": eff_eval.astype(np.float32),
                "shared_model_gen_id": _compute_generation_ids(len(y_eval), stream_start, suite["shared_alarm_windows"]),
            }
            for mk in ARTIFACT_SAVE_METHODS:
                if mk in combined_outputs:
                    art_dict[f"{mk}__raw_prob"] = combined_outputs[mk]["raw_probs"].astype(np.float32)
                    art_dict[f"{mk}__cal_prob"] = combined_outputs[mk]["cal_probs"].astype(np.float32)
                    art_dict[f"{mk}__tuned_th"] = combined_outputs[mk]["tuned_th_vec"].astype(np.float32)
                    art_dict[f"{mk}__decision_th"] = combined_outputs[mk]["decisions_th"].astype(np.int8)
            art_file = os.path.join(save_commit_artifact_dir, f"{stream_name}__{host_type}__{skip_mode}__seed{sd}.npz")
            np.savez_compressed(art_file, **art_dict)

            for mk in ("B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"):
                if mk in combined_outputs:
                    seed0_traj[mk] = evaluate_window_trajectories_v3(y_eval, combined_outputs[mk]["raw_probs"], eff_eval, window_size=window_size)

        for m_name, m_out in combined_outputs.items():
            z_m = m_out["raw_logits"]
            p_m = m_out["raw_probs"]
            pc_m = m_out["cal_probs"]
            th_vec = m_out["tuned_th_vec"]
            dec_vec = m_out["decisions_th"]

            ev_full = evaluate_horizon_v3(
                y_eval, z_m, p_m, pc_m, eff_eval,
                per_commit_thresholds=th_vec, per_commit_decisions=dec_vec, batch_window_size=window_size,
            )
            per_method_seed_metrics.setdefault(m_name, []).append(ev_full)

            ev_post = evaluate_horizon_v3(
                y_eval[onset_rel:], z_m[onset_rel:], p_m[onset_rel:], pc_m[onset_rel:], eff_eval[onset_rel:],
                per_commit_thresholds=th_vec[onset_rel:], per_commit_decisions=dec_vec[onset_rel:], batch_window_size=window_size,
            )
            per_method_post_onset.setdefault(m_name, []).append(ev_post)

            ev_tr = evaluate_horizon_v3(
                y_eval[onset_rel:trans_end_rel], z_m[onset_rel:trans_end_rel], p_m[onset_rel:trans_end_rel],
                pc_m[onset_rel:trans_end_rel], eff_eval[onset_rel:trans_end_rel],
                per_commit_thresholds=th_vec[onset_rel:trans_end_rel],
                per_commit_decisions=dec_vec[onset_rel:trans_end_rel],
                batch_window_size=window_size,
            )
            per_method_transient1200.setdefault(m_name, []).append(ev_tr)

            if common_eval_mask is not None:
                ev_cm = evaluate_horizon_v3(
                    y_eval[common_eval_mask], z_m[common_eval_mask], p_m[common_eval_mask],
                    pc_m[common_eval_mask], eff_eval[common_eval_mask],
                    per_commit_thresholds=th_vec[common_eval_mask],
                    per_commit_decisions=dec_vec[common_eval_mask],
                    batch_window_size=window_size,
                )
                per_method_common_subset.setdefault(m_name, []).append(ev_cm)

            cs = m_out["compute"]
            ams = m_out.get("per_alarm_wall_ms", [])
            astep = m_out.get("per_alarm_step_total_ms", ams)
            tb = m_out.get("timing_breakdown_ms", {})
            e2e_ms = float(tb.get("consistent_stream_eval_ms", m_out.get("end_to_end_eval_ms", cs.wall_ms)))
            per_method_compute.setdefault(m_name, []).append({
                "opt_steps": int(cs.opt_steps),
                "examples_seen": int(cs.examples_seen),
                "trees_built": int(cs.trees_built),
                "requested_rounds": int(getattr(cs, "requested_rounds", 0)),
                "actual_iterations": int(getattr(cs, "actual_iterations", 0)),
                "own_init_fit_ms": float(tb.get("own_init_fit_ms", tb.get("init_fit_ms", 0.0))),
                "shared_anchor_init_ms": float(tb.get("shared_anchor_init_ms", 0.0)),
                "shared_monitor_init_ms": float(tb.get("shared_monitor_init_ms", 0.0)),
                "init_fit_ms": float(tb.get("init_fit_ms", 0.0)),
                "own_stream_predict_and_embed_ms": float(tb.get("own_stream_predict_and_embed_ms", tb.get("stream_predict_and_embed_ms", 0.0))),
                "shared_anchor_eval_embed_ms": float(tb.get("shared_anchor_eval_embed_ms", 0.0)),
                "stream_predict_and_embed_ms": float(tb.get("stream_predict_and_embed_ms", 0.0)),
                "drift_monitor_ms": float(tb.get("drift_monitor_ms", 0.0)),
                "adapt_only_wall_ms": float(tb.get("adaptation_total_ms", cs.wall_ms)),
                "calibration_ms": float(tb.get("calibration_ms", 0.0)),
                "end_to_end_eval_ms": e2e_ms,
                "consistent_end_to_end_ms": float(tb.get("consistent_end_to_end_ms", e2e_ms)),
                "incremental_suite_stream_eval_ms": float(tb.get("incremental_suite_stream_eval_ms", e2e_ms)),
                "incremental_suite_end_to_end_ms": float(tb.get("incremental_suite_end_to_end_ms", e2e_ms)),
                "alarm_adapt_ms_p50": float(np.percentile(ams, 50)) if ams else 0.0,
                "alarm_adapt_ms_p95": float(np.percentile(ams, 95)) if ams else 0.0,
                "alarm_adapt_ms_max": float(np.max(ams)) if ams else 0.0,
                "alarm_step_total_ms_p50": float(np.percentile(astep, 50)) if astep else 0.0,
                "alarm_step_total_ms_p95": float(np.percentile(astep, 95)) if astep else 0.0,
                "alarm_step_total_ms_max": float(np.max(astep)) if astep else 0.0,
                "n_alarms": int(m_out.get("n_alarms", m_out.get("n_updates", 0))),
                "alarm_rate": float(m_out.get("alarm_rate", 0.0)),
                "nan_rollbacks": int(m_out.get("nan_rollbacks", 0)),
                "constant_output_alarms": int(m_out.get("constant_output_alarms", 0)),
                "survival_fallbacks": int(m_out.get("survival_fallbacks", 0)),
                "bisection_unreachable": int(m_out.get("bisection_unreachable", 0)),
            })

    agg_methods = {}
    for m_name in per_method_seed_metrics:
        m_entry = {
            "full_stream": aggregate_seed_horizon_metrics(
                per_method_seed_metrics[m_name], stream_name=stream_name, horizon_name=f"{m_name}:full_stream"
            ),
            "post_onset": aggregate_seed_horizon_metrics(
                per_method_post_onset[m_name], stream_name=stream_name, horizon_name=f"{m_name}:post_onset"
            ),
            "transient_1200": aggregate_seed_horizon_metrics(
                per_method_transient1200[m_name], stream_name=stream_name, horizon_name=f"{m_name}:transient_1200"
            ),
            "telemetry": _agg_numeric_dict_list(per_method_compute[m_name]),
            "seed_ap_values": [float(d["ap"]) for d in per_method_seed_metrics[m_name]],
        }
        if m_name in per_method_common_subset:
            m_entry["common_eval_subset"] = aggregate_seed_horizon_metrics(
                per_method_common_subset[m_name], stream_name=stream_name, horizon_name=f"{m_name}:common_eval_subset"
            )
        agg_methods[m_name] = m_entry

    local_fork_agg = _agg_numeric_dict_list(local_forks) if local_forks else None
    out_obj = {
        "stream_name": stream_name,
        "bundle_sha256": sha256_file(bundle_path),
        "host_type": host_type,
        "skip_mode": skip_mode,
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "N": int(len(y)),
        "stream_eval_N": int(len(y_eval)),
        "drift_onset_idx": onset,
        "local_fork_counterfactual": local_fork_agg,
        "seed0_window_trajectories": seed0_traj,
        "methods": agg_methods,
    }
    if run_fingerprint is not None:
        out_obj["run_fingerprint"] = run_fingerprint
    return out_obj


# ==============================================================================
# Phase A: Benchmark A Multi-Host, Linear-Residual Skip, & Provenance Sensitivity
# ==============================================================================

def select_skip_mode_strictly_on_dev_slice(base_dir: str) -> Dict:
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        meta_map = {m["name"]: m for m in json.load(f)}

    scores = {"frozen_lr": [], "trainable_lr": []}
    details = {}
    from sklearn.metrics import average_precision_score as _ap
    for sname in ("qt_natural_25k", "openstack_natural_13k"):
        meta = meta_map[sname]
        for h_type, rep in [("att_cnn", "tok"), ("codebert", "cb"), ("deepjit_cnn", "tok")]:
            b = np.load(os.path.join(mm_dir, f"{sname}_{rep}.npz"))
            X, ts, y, fts = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            t_ni = float(ts[ni - 1])
            t_nv = float(ts[ni + nv - 1])
            _, y_obs_tr = observe_labels_at_time(ts[:ni], y[:ni], fts[:ni], t_ni)
            _, y_obs_val = observe_labels_at_time(ts[: ni + nv], y[: ni + nv], fts[: ni + nv], t_nv)
            yv = (y_obs_val[ni : ni + nv] > 0.5).astype(np.int32)
            pw = float(np.clip((1.0 - np.mean(y_obs_tr)) / max(np.mean(y_obs_tr), 0.03), 1.5, 8.0))

            for sm in ("frozen_lr", "trainable_lr"):
                m = build_host_backbone_v3(h_type, X[:16], meta=meta, skip_mode=sm, seed=42)
                m.init_linear_skip_from_data(X[:ni], y_obs_tr, pos_weight=pw, seed=42)
                m, _ = train_supervised_v3(m, X[:ni], y_obs_tr, pos_weight=pw, epochs=8, lr=7e-4, seed=42)
                pv = sigmoid_safe(predict_logits_and_embeddings_v3(m, X[ni : ni + nv])[0])
                ap_v = float(_ap(yv, pv)) if len(np.unique(yv)) > 1 else 0.0
                scores[sm].append(ap_v)
                details[f"{sname}:{h_type}:{sm}"] = ap_v

    mean_fr = float(np.mean(scores["frozen_lr"]))
    mean_tr = float(np.mean(scores["trainable_lr"]))
    chosen = "frozen_lr" if mean_fr >= mean_tr else "trainable_lr"
    return {
        "chosen_skip_mode": chosen,
        "mean_dev_ap_frozen_lr": mean_fr,
        "mean_dev_ap_trainable_lr": mean_tr,
        "cell_details": details,
    }


def run_phase_a_benchmark_a(base_dir: str, seeds: List[int]) -> Dict:
    out_path = os.path.join(base_dir, "results_v3", "benchmark_a_multihost_v3.json")
    per_cell_dir = os.path.join(base_dir, "results_v3", "per_cell_a")
    art_dir = os.path.join(base_dir, "results_v3", "commit_predictions", "benchmark_a")
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    sens_dir = os.path.join(base_dir, "cache", "multimodal_v3_sens")
    os.makedirs(per_cell_dir, exist_ok=True)

    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        mm_manifest = json.load(f)

    print("\n=== [Phase A.1] Selecting Linear-Residual Skip Mode on Causal Dev Slice ===", flush=True)
    skip_sel = select_skip_mode_strictly_on_dev_slice(base_dir)
    chosen_skip = skip_sel["chosen_skip_mode"]
    print(f"  [Dev Selection] chosen={chosen_skip} (frozen_lr={skip_sel['mean_dev_ap_frozen_lr']:.4f}, trainable_lr={skip_sel['mean_dev_ap_trainable_lr']:.4f})", flush=True)

    cells_results: Dict[str, Dict] = {}
    host_specs = [
        ("att_cnn", "tok", ["none", chosen_skip]),
        ("codebert", "cb", ["none", chosen_skip]),
        ("deepjit_cnn", "tok", ["none", chosen_skip]),
        ("lightgbm", "svd", ["none"]),
    ]

    for m_info in mm_manifest:
        sname = m_info["name"]
        for h_type, rep_key, skip_modes in host_specs:
            b_path = os.path.join(mm_dir, f"{sname}_{rep_key}.npz")
            for sm in skip_modes:
                cell_key = f"{sname}__{h_type}__{sm}"
                inc_t2 = (h_type == "lightgbm")
                fp_cell = compute_task_run_fingerprint(
                    base_dir, b_path, h_type, sm, seeds, inc_t2, TRACK1_MAIN_METHODS
                )
                ckpt_p = os.path.join(per_cell_dir, f"{cell_key}.json")
                art_f = os.path.join(art_dir, f"{sname}__{h_type}__{sm}__seed{seeds[0]}.npz")
                need_art = sname in ("qt_natural_25k", "openstack_natural_13k")
                res_cell = load_verified_checkpoint_or_none(ckpt_p, fp_cell)
                if res_cell is None or (need_art and not os.path.exists(art_f)):
                    t_c0 = time.perf_counter()
                    res_cell = evaluate_and_aggregate_stream_runs(
                        stream_name=sname,
                        bundle_path=b_path,
                        host_type=h_type,
                        seeds=seeds,
                        meta=m_info,
                        skip_mode=sm,
                        include_track2=inc_t2,
                        save_commit_artifact_dir=art_dir if need_art else None,
                        run_fingerprint=fp_cell,
                    )
                    save_json_atomic(ckpt_p, res_cell)
                    dt_s = time.perf_counter() - t_c0
                else:
                    dt_s = 0.0
                cells_results[cell_key] = res_cell
                ap_b0 = res_cell["methods"]["B0_Frozen"]["full_stream"]["ap"]
                ap_b2 = res_cell["methods"]["B2_4W"]["full_stream"]["ap"]
                ap_spu = res_cell["methods"]["Survival_PU"]["full_stream"]["ap"]
                print(
                    f"  [Phase A] {cell_key:46s} | AP: B0={ap_b0:.4f}, B2_4W={ap_b2:.4f}, SPU={ap_spu:.4f} ({dt_s:.1f}s)",
                    flush=True,
                )

    print("\n=== [Phase A.2] Running Benchmark A Provenance Sensitivity Suite (Aligned Temporal Cutoffs + Common Eval IDs) ===", flush=True)
    with open(os.path.join(sens_dir, "manifest_sens_v3.json"), "r", encoding="utf-8") as f:
        sens_manifest = json.load(f)

    # Build exact common evaluation commit ID sets per ecosystem (`R3-M2`: `drop_unlinked` eval IDs subset)
    common_eval_id_sets: Dict[str, set] = {}
    for eco_k in ("qt", "openstack"):
        b_dm = np.load(os.path.join(sens_dir, f"{eco_k}_drop_unlinked_meta.npz"), allow_pickle=True)
        b_dc = np.load(os.path.join(sens_dir, f"{eco_k}_drop_unlinked_cb.npz"))
        st_d = int(b_dc["n_init"]) + int(b_dc["n_val"])
        common_eval_id_sets[eco_k] = set(str(x) for x in b_dm["commit_ids"][st_d:])

    sens_results: Dict[str, Dict] = {}
    sens_methods = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Pure_nnPU", "Survival_PU"]
    for sm_info in sens_manifest:
        sn = sm_info["name"]
        eco_k = "qt" if sn.startswith("qt_") else "openstack"
        b_sm_meta = np.load(os.path.join(sens_dir, f"{sn}_meta.npz"), allow_pickle=True)
        st_s = int(sm_info["n_init"]) + int(sm_info["n_val"])
        eval_cids_s = [str(x) for x in b_sm_meta["commit_ids"][st_s:]]
        cm_mask = np.array([cid in common_eval_id_sets[eco_k] for cid in eval_cids_s], dtype=bool)

        for h_t, rk, skm in [("codebert", "cb", chosen_skip), ("lightgbm", "svd", "none")]:
            bp = os.path.join(sens_dir, f"{sn}_{rk}.npz")
            rc = evaluate_and_aggregate_stream_runs(
                stream_name=sn,
                bundle_path=bp,
                host_type=h_t,
                seeds=seeds[:3],
                meta=sm_info,
                skip_mode=skm,
                include_track2=(h_t == "lightgbm"),
                methods_track1=sens_methods,
                common_eval_mask=cm_mask,
            )
            rc["common_eval_commits_N"] = int(np.sum(cm_mask))
            sens_results[f"{sn}__{h_t}"] = rc
            print(
                f"  [Sens-A] {sn}__{h_t:10s} | Native AP: SPU={rc['methods']['Survival_PU']['full_stream']['ap']:.4f} | "
                f"CommonIDs(N={int(np.sum(cm_mask))}) SPU={rc['methods']['Survival_PU']['common_eval_subset']['ap']:.4f}",
                flush=True,
            )

    payload = {
        "provenance": build_provenance_header(
            base_dir,
            [os.path.join(mm_dir, "manifest_mm_v3.json"), os.path.join(sens_dir, "manifest_sens_v3.json")],
        ),
        "scope_note": (
            "Benchmark A streams represent author-date chronological historical replay over merged repository "
            "commits using strictly 13 commit-authoring tabular metrics (`PURE_COMMIT_COLS_QT_OS`); post-authoring "
            "Gerrit review fields are excluded in the primary suite and isolated in `premerge28` sensitivity."
        ),
        "dev_skip_selection": skip_sel,
        "cells": cells_results,
        "provenance_sensitivity": sens_results,
    }
    save_json_atomic(out_path, payload)
    print(f"==> Saved Phase A Benchmark A Results -> {out_path}", flush=True)
    return payload


# ==============================================================================
# Phase B: Benchmark B Cross-Project Natural Suite (K=25 Clusters) + Shift Suite
# ==============================================================================

def compute_benchmark_b_cluster_statistics(
    per_stream_results: Dict[str, Dict],
    endpoint: str = "ap",
    include_competitor_family: bool = True,
) -> Dict:
    higher_is_better = endpoint not in LOWER_IS_BETTER_ENDPOINTS
    per_stream_scores: Dict[str, Dict[str, float]] = {}
    for sname, sres in per_stream_results.items():
        if sname not in NATURAL_STREAM_TO_DONOR_CLUSTER:
            continue
        per_stream_scores[sname] = {
            m: float(m_dict["full_stream"][endpoint])
            for m, m_dict in sres["methods"].items()
            if endpoint in m_dict["full_stream"] and m_dict["full_stream"][endpoint] is not None
        }

    clusters_25, scores_25 = aggregate_streams_to_donor_clusters(per_stream_scores, NATURAL_STREAM_TO_DONOR_CLUSTER)
    clusters_24, scores_24 = aggregate_streams_to_donor_clusters(
        per_stream_scores, NATURAL_STREAM_TO_DONOR_CLUSTER, exclude_clusters=("openstack",)
    )

    fam1_25 = run_holm_bonferroni_family(scores_25, FAMILY_1_PARADOX_PAIRS, "Family1_Paradox_K25", higher_is_better=higher_is_better)
    fam1_24 = run_holm_bonferroni_family(scores_24, FAMILY_1_PARADOX_PAIRS, "Family1_Paradox_K24_DisjointA", higher_is_better=higher_is_better)

    fam2_25 = (
        run_holm_bonferroni_family(scores_25, FAMILY_2_COMPETITOR_PAIRS, "Family2_Competitors_K25", higher_is_better=higher_is_better)
        if include_competitor_family
        else []
    )
    fam2_24 = (
        run_holm_bonferroni_family(scores_24, FAMILY_2_COMPETITOR_PAIRS, "Family2_Competitors_K24_DisjointA", higher_is_better=higher_is_better)
        if include_competitor_family
        else []
    )
    active_f3_pairs = [(ma, mb) for (ma, mb) in FAMILY_3_TRACK1_CONTROL_PAIRS if ma in scores_25 and mb in scores_25]
    fam3_25 = (
        run_holm_bonferroni_family(scores_25, active_f3_pairs, "Family3_Controls_K25", higher_is_better=higher_is_better)
        if active_f3_pairs
        else []
    )
    fam3_24 = (
        run_holm_bonferroni_family(scores_24, active_f3_pairs, "Family3_Controls_K24_DisjointA", higher_is_better=higher_is_better)
        if active_f3_pairs
        else []
    )

    cluster_means_25 = {m: float(np.mean(arr)) for m, arr in scores_25.items()}
    cluster_medians_25 = {m: float(np.median(arr)) for m, arr in scores_25.items()}

    return {
        "endpoint": endpoint,
        "higher_is_better": higher_is_better,
        "pre_registered_interpretation_boundaries": PRE_REGISTERED_INTERPRETATION_BOUNDARIES,
        "K25_clusters": clusters_25,
        "K25_cluster_means": cluster_means_25,
        "K25_cluster_medians": cluster_medians_25,
        "K25_per_cluster_vectors": {m: [float(x) for x in arr] for m, arr in scores_25.items()},
        "family1_paradox_K25": fam1_25,
        "family2_competitors_K25": fam2_25,
        "family3_controls_K25": fam3_25,
        "sensitivity_K24_disjoint_from_benchmark_a": {
            "K24_clusters": clusters_24,
            "K24_cluster_means": {m: float(np.mean(arr)) for m, arr in scores_24.items()},
            "family1_paradox_K24": fam1_24,
            "family2_competitors_K24": fam2_24,
            "family3_controls_K24": fam3_24,
        },
    }


def run_phase_b_benchmark_b(base_dir: str, seeds_mlp: List[int], seeds_lgb: List[int]) -> Dict:
    out_path = os.path.join(base_dir, "results_v3", "benchmark_b_main_v3.json")
    per_stream_dir = os.path.join(base_dir, "results_v3", "per_stream_b")
    art_dir = os.path.join(base_dir, "results_v3", "commit_predictions", "benchmark_b")
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    os.makedirs(per_stream_dir, exist_ok=True)

    with open(os.path.join(b_dir, "manifest_b_v3.json"), "r", encoding="utf-8") as f:
        man_b = json.load(f)

    nat_mlp: Dict[str, Dict] = {}
    nat_lgb: Dict[str, Dict] = {}
    shift_mlp: Dict[str, Dict] = {}

    print(f"\n=== [Phase B.1] Running 27 Natural Benchmark B Streams (MLP {len(seeds_mlp)} seeds + Track 2 Competitors) ===", flush=True)
    for item in man_b["natural_streams"]:
        sname = item["name"]
        b_path = os.path.join(b_dir, f"{sname}.npz")
        fp_s = compute_task_run_fingerprint(base_dir, b_path, "mlp", "none", seeds_mlp, True, TRACK1_MAIN_METHODS)
        ckpt_p = os.path.join(per_stream_dir, f"{sname}__mlp.json")
        art_f = os.path.join(art_dir, f"{sname}__mlp__none__seed{seeds_mlp[0]}.npz")
        res_s = load_verified_checkpoint_or_none(ckpt_p, fp_s)
        if res_s is None or (not os.path.exists(art_f)):
            t0 = time.perf_counter()
            res_s = evaluate_and_aggregate_stream_runs(
                stream_name=sname,
                bundle_path=b_path,
                host_type="mlp",
                seeds=seeds_mlp,
                include_track2=True,
                save_commit_artifact_dir=art_dir,
                run_fingerprint=fp_s,
            )
            save_json_atomic(ckpt_p, res_s)
            print(
                f"  [Nat-MLP] {sname:20s} | AP: B0={res_s['methods']['B0_Frozen']['full_stream']['ap']:.4f}, "
                f"B2_4W={res_s['methods']['B2_4W']['full_stream']['ap']:.4f}, "
                f"BORB={res_s['methods']['BORB_Native']['full_stream']['ap']:.4f}, "
                f"ORB={res_s['methods']['ORB_OHT_Native']['full_stream']['ap']:.4f}, "
                f"SPU={res_s['methods']['Survival_PU']['full_stream']['ap']:.4f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
        nat_mlp[sname] = res_s

    print(f"\n=== [Phase B.2] Running 27 Natural Benchmark B Streams on Tree Host (LightGBM {len(seeds_lgb)} seeds) ===", flush=True)
    lgb_methods = [
        "B0_Frozen", "B1b_4W_Oracle", "B2_1W", "B2_4W", "Wait_90",
        "Pure_nnPU", "FixedCurve_PU", "Teacher_Only_Exact", "Teacher_Only_Matched", "Survival_PU",
    ]
    for item in man_b["natural_streams"]:
        sname = item["name"]
        b_path = os.path.join(b_dir, f"{sname}.npz")
        fp_l = compute_task_run_fingerprint(base_dir, b_path, "lightgbm", "none", seeds_lgb, False, lgb_methods)
        ckpt_p = os.path.join(per_stream_dir, f"{sname}__lightgbm.json")
        res_l = load_verified_checkpoint_or_none(ckpt_p, fp_l)
        if res_l is None:
            t0 = time.perf_counter()
            res_l = evaluate_and_aggregate_stream_runs(
                stream_name=sname,
                bundle_path=b_path,
                host_type="lightgbm",
                seeds=seeds_lgb,
                include_track2=False,
                methods_track1=lgb_methods,
                run_fingerprint=fp_l,
            )
            save_json_atomic(ckpt_p, res_l)
            print(
                f"  [Nat-LGB] {sname:20s} | AP: B0={res_l['methods']['B0_Frozen']['full_stream']['ap']:.4f}, "
                f"B2_4W={res_l['methods']['B2_4W']['full_stream']['ap']:.4f}, "
                f"TeachMatch={res_l['methods']['Teacher_Only_Matched']['full_stream']['ap']:.4f}, "
                f"SPU={res_l['methods']['Survival_PU']['full_stream']['ap']:.4f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
        nat_lgb[sname] = res_l

    print(f"\n=== [Phase B.3] Running 4 Controlled Tabular Shift Streams ({len(seeds_mlp)} seeds, Strictly Separated) ===", flush=True)
    for item in man_b["controlled_shift_streams"]:
        sname = item["name"]
        b_path = os.path.join(b_dir, f"{sname}.npz")
        fp_sh = compute_task_run_fingerprint(base_dir, b_path, "mlp", "none", seeds_mlp, True, TRACK1_MAIN_METHODS)
        ckpt_p = os.path.join(per_stream_dir, f"{sname}__mlp.json")
        art_f = os.path.join(art_dir, f"{sname}__mlp__none__seed{seeds_mlp[0]}.npz")
        res_sh = load_verified_checkpoint_or_none(ckpt_p, fp_sh)
        if res_sh is None or (not os.path.exists(art_f)):
            t0 = time.perf_counter()
            res_sh = evaluate_and_aggregate_stream_runs(
                stream_name=sname,
                bundle_path=b_path,
                host_type="mlp",
                seeds=seeds_mlp,
                include_track2=True,
                save_commit_artifact_dir=art_dir,
                run_fingerprint=fp_sh,
            )
            save_json_atomic(ckpt_p, res_sh)
            print(
                f"  [Shift-MLP] {sname:34s} | Post-Onset AP: B0={res_sh['methods']['B0_Frozen']['post_onset']['ap']:.4f}, "
                f"B2_4W={res_sh['methods']['B2_4W']['post_onset']['ap']:.4f}, SPU={res_sh['methods']['Survival_PU']['post_onset']['ap']:.4f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
        shift_mlp[sname] = res_sh

    stats_mlp_ap = compute_benchmark_b_cluster_statistics(nat_mlp, endpoint="ap", include_competitor_family=True)
    stats_mlp_f1 = compute_benchmark_b_cluster_statistics(nat_mlp, endpoint="f1", include_competitor_family=True)
    stats_mlp_f1th = compute_benchmark_b_cluster_statistics(nat_mlp, endpoint="f1_th", include_competitor_family=True)
    stats_mlp_auc = compute_benchmark_b_cluster_statistics(nat_mlp, endpoint="roc_auc", include_competitor_family=True)
    stats_mlp_brier = compute_benchmark_b_cluster_statistics(nat_mlp, endpoint="brier_cal", include_competitor_family=True)
    stats_lgb_ap = compute_benchmark_b_cluster_statistics(nat_lgb, endpoint="ap", include_competitor_family=False)

    payload = {
        "provenance": build_provenance_header(base_dir, [os.path.join(b_dir, "manifest_b_v3.json")]),
        "cluster_inference_mlp": {
            "primary_ap": stats_mlp_ap,
            "secondary_f1": stats_mlp_f1,
            "secondary_f1_th": stats_mlp_f1th,
            "secondary_roc_auc": stats_mlp_auc,
            "secondary_brier_cal": stats_mlp_brier,
        },
        "cluster_inference_lightgbm_ap": stats_lgb_ap,
        "natural_streams_mlp": nat_mlp,
        "natural_streams_lightgbm": nat_lgb,
        "controlled_shift_streams_mlp": shift_mlp,
    }
    save_json_atomic(out_path, payload)
    print(f"==> Saved Phase B Benchmark B Master Results -> {out_path}", flush=True)
    return payload


# ==============================================================================
# Phase C: Untouched Confirmatory Cohort (Strictly Post-Sign-Off, `R3-C8`)
# ==============================================================================

def run_phase_c_confirmatory(base_dir: str, seeds: List[int]) -> Dict:
    conf_dir = os.path.join(base_dir, "cache", "confirmatory_v3")
    man_p = os.path.join(conf_dir, "manifest_confirmatory_v3.json")
    assert os.path.exists(man_p), "Confirmatory cache not unlocked yet! Run `data_audit_v3.py --unlock-confirmatory` after sign-off."

    with open(man_p, "r", encoding="utf-8") as f:
        man_c = json.load(f)

    out_path = os.path.join(base_dir, "results_v3", "confirmatory_cohort_v3.json")
    art_dir = os.path.join(base_dir, "results_v3", "commit_predictions", "confirmatory")

    streams_mlp: Dict[str, Dict] = {}
    streams_lgb: Dict[str, Dict] = {}
    repo_to_umbrella = {}

    print(f"\n=== [Phase C] Running Untouched Confirmatory ApacheJIT Cohort ({len(man_c['confirmatory_streams'])} repos, {len(seeds)} seeds) ===", flush=True)
    lgb_conf_methods = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Pure_nnPU", "FixedCurve_PU", "Teacher_Only_Matched", "Survival_PU"]
    for item in man_c["confirmatory_streams"]:
        sname = item["name"]
        repo_to_umbrella[sname] = item["umbrella_ecosystem_cluster"]
        bp = os.path.join(conf_dir, f"{sname}.npz")

        r_m = evaluate_and_aggregate_stream_runs(
            stream_name=sname, bundle_path=bp, host_type="mlp", seeds=seeds, include_track2=True, save_commit_artifact_dir=art_dir
        )
        r_l = evaluate_and_aggregate_stream_runs(
            stream_name=sname, bundle_path=bp, host_type="lightgbm", seeds=seeds[:5], include_track2=False,
            methods_track1=lgb_conf_methods,
        )
        streams_mlp[sname] = r_m
        streams_lgb[sname] = r_l
        print(
            f"  [Confirmatory] {sname:22s} | MLP AP: B0={r_m['methods']['B0_Frozen']['full_stream']['ap']:.4f}, "
            f"B1b={r_m['methods']['B1b_4W_Oracle']['full_stream']['ap']:.4f}, "
            f"B2_4W={r_m['methods']['B2_4W']['full_stream']['ap']:.4f}, "
            f"BORB={r_m['methods']['BORB_Native']['full_stream']['ap']:.4f}, "
            f"ORB={r_m['methods']['ORB_OHT_Native']['full_stream']['ap']:.4f}, "
            f"SPU={r_m['methods']['Survival_PU']['full_stream']['ap']:.4f}",
            flush=True,
        )

    repo_map_5 = {s: s for s in streams_mlp}
    clusters_repo5, sc_repo5 = aggregate_streams_to_donor_clusters(
        {s: {m: float(d["full_stream"]["ap"]) for m, d in r["methods"].items()} for s, r in streams_mlp.items()},
        repo_map_5,
    )
    clusters_umb4, sc_umb4 = aggregate_streams_to_donor_clusters(
        {s: {m: float(d["full_stream"]["ap"]) for m, d in r["methods"].items()} for s, r in streams_mlp.items()},
        repo_to_umbrella,
    )
    _, sc_umb4_lgb = aggregate_streams_to_donor_clusters(
        {s: {m: float(d["full_stream"]["ap"]) for m, d in r["methods"].items()} for s, r in streams_lgb.items()},
        repo_to_umbrella,
    )

    payload = {
        "provenance": build_provenance_header(base_dir, [man_p]),
        "pre_registered_statistical_plan": {
            "primary_unit": "umbrella4 (K=4 independent Apache ecosystems: Hadoop, Spark, Zeppelin, ZooKeeper)",
            "secondary_unit": "repo5 (K=5 repositories)",
            "min_attainable_two_sided_exact_p_umbrella4": 0.125,
            "min_attainable_two_sided_exact_p_repo5": 0.0625,
            "interpretation_rule": (
                "Pre-registered out-of-sample directional consistency (W/T/L) and effect-size (Delta-AP) "
                "replication check under frozen V3_CONFIG; alpha=0.05 hypothesis testing is powered on K=25/K=24 natural clusters."
            ),
        },
        "umbrella4_clusters": clusters_umb4,
        "repo5_clusters": clusters_repo5,
        "umbrella4_means_ap": {m: float(np.mean(v)) for m, v in sc_umb4.items()},
        "umbrella4_lightgbm_means_ap": {m: float(np.mean(v)) for m, v in sc_umb4_lgb.items()},
        "repo5_means_ap": {m: float(np.mean(v)) for m, v in sc_repo5.items()},
        "umbrella4_family1_paradox": run_holm_bonferroni_family(sc_umb4, FAMILY_1_PARADOX_PAIRS, "Conf_Umbrella4_Paradox"),
        "umbrella4_family2_competitors": run_holm_bonferroni_family(sc_umb4, FAMILY_2_COMPETITOR_PAIRS, "Conf_Umbrella4_Competitors"),
        "umbrella4_family3_controls": run_holm_bonferroni_family(sc_umb4, FAMILY_3_TRACK1_CONTROL_PAIRS, "Conf_Umbrella4_Controls"),
        "umbrella4_lightgbm_family1_paradox": run_holm_bonferroni_family(sc_umb4_lgb, FAMILY_1_PARADOX_PAIRS, "Conf_Umbrella4_LGB_Paradox"),
        "repo5_family1_paradox": run_holm_bonferroni_family(sc_repo5, FAMILY_1_PARADOX_PAIRS, "Conf_Repo5_Paradox"),
        "repo5_family2_competitors": run_holm_bonferroni_family(sc_repo5, FAMILY_2_COMPETITOR_PAIRS, "Conf_Repo5_Competitors"),
        "repo5_family3_controls": run_holm_bonferroni_family(sc_repo5, FAMILY_3_TRACK1_CONTROL_PAIRS, "Conf_Repo5_Controls"),
        "streams_mlp": streams_mlp,
        "streams_lightgbm": streams_lgb,
    }
    save_json_atomic(out_path, payload)
    print(f"==> Saved Untouched Confirmatory Results -> {out_path}", flush=True)
    return payload


# ==============================================================================
# Phase D: Mechanistic Ablations, Causal Interventions, Factorial DriftLens,
#          Semi-Synthetic SZZ Noise, & Fixed-Horizon Sensitivity (RQ3, RQ4a-e)
# ==============================================================================

REPRESENTATIVE_10_STREAMS: List[str] = [
    "cabral_nova",
    "cabral_tomcat",
    "apache_ignite",
    "apache_flink",
    "apache_cassandra",
    "n60_airflow",
    "n60_angular",
    "abrupt_nova_zulip",
    "abrupt_ignite_cassandra",
    "abrupt_airflow_pulsar",
]


def run_rq3_ablations_v3(base_dir: str, seeds: List[int]) -> Dict:
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    out_path = os.path.join(base_dir, "results_v3", "rq3_ablations_v3.json")

    variants = {
        "B0_Frozen": ("B0_Frozen", {}),
        "B1b_4W_Oracle": ("B1b_4W_Oracle", {}),
        "B2_4W": ("B2_4W", {}),
        "Survival_PU_Default": ("Survival_PU", {}),
        "SPU_KaplanMeier": ("Survival_PU", {"survival_estimator": "kaplan_meier"}),
        "SPU_WeibullCure": ("Survival_PU", {"survival_estimator": "weibull_cure"}),
        "SPU_Fixed90d": ("Survival_PU", {"survival_estimator": "fixed_90d"}),
        "SPU_NoShrinkage_m0_0": ("Survival_PU", {"ht_shrinkage_m0": 0.0}),
        "SPU_NoBisection": ("Survival_PU", {"use_bisection": False}),
        "SPU_NoCovariateWeight": ("Survival_PU", {"use_importance_weights": False}),
        "SPU_FrozenAnchorTeacher": ("Survival_PU", {"teacher_mode": "frozen_anchor_teacher"}),
        "Teacher_Only_Exact": ("Teacher_Only_Exact", {}),
        "Teacher_Only_Matched": ("Teacher_Only_Matched", {}),
        "SPU_WithLegacyGate": ("Survival_PU_WithGate", {}),
    }

    print(f"\n=== [RQ3] Running Pure Component & Compute-Matched Ablations ({len(REPRESENTATIVE_10_STREAMS)} streams x {len(seeds)} seeds) ===", flush=True)
    per_stream_var: Dict[str, Dict[str, Dict]] = {s: {} for s in REPRESENTATIVE_10_STREAMS}

    for sname in REPRESENTATIVE_10_STREAMS:
        b = np.load(os.path.join(b_dir, f"{sname}.npz"))
        X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
        ni, nv = int(b["n_init"]), int(b["n_val"])
        s_start = ni + nv

        for v_label, (m_internal, spu_ov) in variants.items():
            evs, comps = [], []
            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts, ni, nv, host_type="mlp", methods=[m_internal], spu_overrides=spu_ov, seed=sd
                )
                mo = res["methods"][m_internal]
                evs.append(
                    evaluate_horizon_v3(
                        y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                        per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                    )
                )
                ams = mo["per_alarm_wall_ms"]
                comps.append({
                    "opt_steps": mo["compute"].opt_steps,
                    "examples_seen": mo["compute"].examples_seen,
                    "end_to_end_eval_ms": mo["end_to_end_eval_ms"],
                    "alarm_adapt_ms_p50": float(np.percentile(ams, 50)) if ams else 0.0,
                    "alarm_adapt_ms_p95": float(np.percentile(ams, 95)) if ams else 0.0,
                    "alarm_adapt_ms_max": float(np.max(ams)) if ams else 0.0,
                    "survival_fallbacks": mo["survival_fallbacks"],
                    "bisection_unreachable": mo["bisection_unreachable"],
                    "constant_output_alarms": mo["constant_output_alarms"],
                })
            per_stream_var[sname][v_label] = {
                "ap": float(np.mean([e["ap"] for e in evs])),
                "f1": float(np.mean([e["f1"] for e in evs])),
                "f1_th": float(np.mean([e["f1_th"] for e in evs])),
                "roc_auc": float(np.mean([e["roc_auc"] for e in evs])),
                "brier_cal": float(np.mean([e["brier_cal"] for e in evs])),
                "recall_at_20pct_effort": float(np.mean([e["recall_at_20pct_effort"] for e in evs])),
                "opt_steps": float(np.mean([c["opt_steps"] for c in comps])),
                "examples_seen": float(np.mean([c["examples_seen"] for c in comps])),
                "end_to_end_eval_ms": float(np.mean([c["end_to_end_eval_ms"] for c in comps])),
                "alarm_adapt_ms_p50": float(np.mean([c["alarm_adapt_ms_p50"] for c in comps])),
                "alarm_adapt_ms_p95": float(np.mean([c["alarm_adapt_ms_p95"] for c in comps])),
                "alarm_adapt_ms_max": float(np.max([c["alarm_adapt_ms_max"] for c in comps])),
                "survival_fallbacks": float(np.sum([c["survival_fallbacks"] for c in comps])),
                "bisection_unreachable": float(np.sum([c["bisection_unreachable"] for c in comps])),
                "constant_output_alarms": float(np.sum([c["constant_output_alarms"] for c in comps])),
            }
        print(
            f"  [RQ3] {sname:24s} | AP: Default={per_stream_var[sname]['Survival_PU_Default']['ap']:.4f}, "
            f"Fixed90={per_stream_var[sname]['SPU_Fixed90d']['ap']:.4f}, "
            f"TeachExact={per_stream_var[sname]['Teacher_Only_Exact']['ap']:.4f}, "
            f"TeachMatch={per_stream_var[sname]['Teacher_Only_Matched']['ap']:.4f}, "
            f"WithGate={per_stream_var[sname]['SPU_WithLegacyGate']['ap']:.4f}",
            flush=True,
        )

    summary_keys = (
        "ap", "f1", "f1_th", "roc_auc", "brier_cal", "recall_at_20pct_effort",
        "opt_steps", "examples_seen", "end_to_end_eval_ms",
        "alarm_adapt_ms_p50", "alarm_adapt_ms_p95", "alarm_adapt_ms_max",
        "survival_fallbacks", "bisection_unreachable", "constant_output_alarms",
    )
    summary_means = {
        v_label: {
            k: float(np.mean([per_stream_var[s][v_label][k] for s in REPRESENTATIVE_10_STREAMS]))
            for k in summary_keys
        }
        for v_label in variants
    }

    # Descriptive Component Contrasts on the 7 mutually independent natural streams (`R3c-1`:
    # Because `2 / 2^7 = 0.015625 > 0.05 / 4 = 0.0125`, a 4-pair Holm family on K=7 cannot reject ANY
    # hypothesis at alpha=0.05; formal alpha=0.05 Holm inference is scoped strictly to powered K=25 / K=24,
    # while K=7 reports descriptive effect sizes, 95% CIs, W/T/L, and exact unadjusted p within resolution!)
    indep_nat_7 = [s for s in REPRESENTATIVE_10_STREAMS if s in NATURAL_STREAM_TO_DONOR_CLUSTER]
    scores_nat7 = {
        v_label: np.array([per_stream_var[s][v_label]["ap"] for s in indep_nat_7], dtype=np.float64)
        for v_label in variants
    }
    desc_contrasts_nat7 = [
        {
            "method_a": ma,
            "method_b": mb,
            "powered_for_alpha_005_4pair_holm": False,
            "statistical_role": "descriptive_effect_size_and_directional_summary_only_formal_holm_powered_on_K25_K24",
            **paired_cluster_inference_v3(scores_nat7[ma], scores_nat7[mb], higher_is_better=True),
        }
        for (ma, mb) in FAMILY_3_COMPONENT_CONTROL_PAIRS
    ]

    payload = {
        "provenance": build_provenance_header(base_dir),
        "pre_registered_interpretation_boundaries": PRE_REGISTERED_INTERPRETATION_BOUNDARIES,
        "streams": REPRESENTATIVE_10_STREAMS,
        "independent_natural_streams_K7": indep_nat_7,
        "descriptive_component_contrasts_natural_K7": desc_contrasts_nat7,
        "summary_means": summary_means,
        "per_stream": per_stream_var,
    }
    save_json_atomic(out_path, payload)
    return payload


def run_rq4a_future_only_latency_v3(base_dir: str, seeds: List[int]) -> Dict:
    """
    Future-Only Causal Latency Intervention (`R3-C6`):
      - Freezes initial `t0` supervision (`fix_ts_for_init=fts`) AND guarantees `t0_post > t0` so
        `B0_Frozen` and initial checkpoint/monitor are bit-for-bit identical across ALL `kappa`!
      - Separates pure finite-delay scaling (`kappa in [0.0 .. 3.0]`, unlinked `+inf` kept as `+inf`)
        from `oracle_post_t0` (`post_t0_oracle_visibility=True`, where `B1b == B2` identically).
    """
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    out_path = os.path.join(base_dir, "results_v3", "rq4a_latency_intervention_v3.json")
    kappas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    regimes_spec = [("oracle_post_t0", 0.0, True)] + [(f"{kap:.2f}", kap, False) for kap in kappas]
    methods_k = ["B0_Frozen", "B1b_4W_Oracle", "B2_1W", "B2_4W", "Wait_90", "Pure_nnPU", "Survival_PU"]

    print(f"\n=== [RQ4a] Running Future-Only Causal Latency Intervention (Frozen Pre-t0 State, {len(regimes_spec)} regimes) ===", flush=True)
    by_kappa: Dict[str, Dict] = {}

    for k_key, kap, is_oracle_vis in regimes_spec:
        stream_rows = {}
        for sname in REPRESENTATIVE_10_STREAMS:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            s_start = ni + nv
            t0 = float(ts[s_start - 1])
            fts_k = apply_future_only_latency_intervention(
                ts, y, fts, t0=t0, kappa=kap, post_t0_oracle_visibility=is_oracle_vis
            )
            m_acc: Dict[str, List[Dict]] = {m: [] for m in methods_k}
            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts_k, ni, nv, host_type="mlp", methods=methods_k, fix_ts_for_init=fts, seed=sd
                )
                for m in methods_k:
                    mo = res["methods"][m]
                    m_acc[m].append(
                        evaluate_horizon_v3(
                            y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                            per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                        )
                    )
            stream_rows[sname] = {
                m: {
                    "ap": float(np.mean([d["ap"] for d in m_acc[m]])),
                    "f1": float(np.mean([d["f1"] for d in m_acc[m]])),
                    "f1_th": float(np.mean([d["f1_th"] for d in m_acc[m]])),
                    "recall_at_20pct_effort": float(np.mean([d["recall_at_20pct_effort"] for d in m_acc[m]])),
                    "top20_jaccard_with_loc_only": float(np.mean([d["top20_jaccard_with_loc_only"] for d in m_acc[m]])),
                }
                for m in methods_k
            }
        means_k = {
            m: {
                metric: float(np.mean([stream_rows[s][m][metric] for s in REPRESENTATIVE_10_STREAMS]))
                for metric in ("ap", "f1", "f1_th", "recall_at_20pct_effort", "top20_jaccard_with_loc_only")
            }
            for m in methods_k
        }
        by_kappa[k_key] = {
            "kappa": kap,
            "post_t0_oracle_visibility": is_oracle_vis,
            "means": means_k,
            "per_stream": stream_rows,
        }
        print(
            f"  [RQ4a {k_key:14s}] AP: B0={means_k['B0_Frozen']['ap']:.4f}, B1b={means_k['B1b_4W_Oracle']['ap']:.4f}, "
            f"B2_4W={means_k['B2_4W']['ap']:.4f}, Wait90={means_k['Wait_90']['ap']:.4f}, SPU={means_k['Survival_PU']['ap']:.4f}",
            flush=True,
        )

    payload = {"provenance": build_provenance_header(base_dir), "kappas": kappas, "results_by_kappa": by_kappa}
    save_json_atomic(out_path, payload)
    return payload


def run_rq4b_factorial_drift_monitors_v3(base_dir: str, seeds: List[int]) -> Dict:
    """
    Unconfounded 2x2 Factorial (`iid` vs `block` x `static` vs `rolling` at fixed `pct=95, cooldown=1`)
    + Separate `N46_Ref_98_CD0` + Closed-Loop Embeddings + MMD/KS/Periodic (`R3-C7`):
      1. Stationary False-Alarm Rate (`stationary_far`) evaluated on the EXACT `m_causal_init`
         embeddings `Z0`, executing `update_reference` on rolling monitors!
      2. Shift detection delay & `undetected_shift_rate` read from each method's ACTUAL
         `executed_alarm_windows` across all 4 controlled shift streams (never imputing `N - onset`!).
    """
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    out_path = os.path.join(base_dir, "results_v3", "rq4b_factorial_monitors_v3.json")

    configs = [
        # Unconfounded 2x2 (`percentile=95.0, cooldown=1` identical across all 4 cells!)
        ("DriftLens_IID_Static_95_CD1", {"monitor_mode": "frechet", "calib_sampling": "iid", "reference_refresh": "static", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        ("DriftLens_IID_Rolling_95_CD1", {"monitor_mode": "frechet", "calib_sampling": "iid", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        ("DriftLens_Block_Static_95_CD1", {"monitor_mode": "frechet", "calib_sampling": "block", "reference_refresh": "static", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        ("DriftLens_Block_Rolling_95_CD1_V3", {"monitor_mode": "frechet", "calib_sampling": "block", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        # Separate Pote et al. (2026) reference configuration (`pct=98.0, cooldown=0`)
        ("DriftLens_N46_Ref_IID_Static_98_CD0", {"monitor_mode": "frechet", "calib_sampling": "iid", "reference_refresh": "static", "drift_percentile": 98.0, "drift_cooldown": 0, "closed_loop_embeddings": False}),
        # Closed-loop & alternative statistical detectors (`pct=95.0, cooldown=1`)
        ("DriftLens_Block_Rolling_ClosedLoop", {"monitor_mode": "frechet", "calib_sampling": "block", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": True}),
        ("RBF_MMD_Block_Rolling", {"monitor_mode": "mmd", "calib_sampling": "block", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        ("PCA_KS_Block_Rolling", {"monitor_mode": "ks", "calib_sampling": "block", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 1, "closed_loop_embeddings": False}),
        ("Periodic_Every3W", {"monitor_mode": "periodic", "calib_sampling": "block", "reference_refresh": "rolling", "drift_percentile": 95.0, "drift_cooldown": 0, "closed_loop_embeddings": False}),
    ]
    methods_m = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"]
    rq4b_streams = list(REPRESENTATIVE_10_STREAMS) + ["abrupt_deepjit_nova_neutron_pure13"]
    shift_streams = set(["abrupt_nova_zulip", "abrupt_ignite_cassandra", "abrupt_airflow_pulsar", "abrupt_deepjit_nova_neutron_pure13"])

    print("\n=== [RQ4b] Precomputing Exact Causal Initial Embeddings `Z0` for Stationary Null FAR ===", flush=True)
    cached_z0: Dict[Tuple[str, int], Tuple[np.ndarray, int, int]] = {}
    for sname in rq4b_streams:
        b = np.load(os.path.join(b_dir, f"{sname}.npz"))
        X, ts, y, fts = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"]
        ni, nv = int(b["n_init"]), int(b["n_val"])
        s_start = ni + nv
        t_start = float(ts[s_start - 1])
        _, y_obs_s = observe_labels_at_time(ts[:s_start], y[:s_start], fts[:s_start], t_start)
        ages_d = (t_start - ts[:s_start]) / SECONDS_PER_DAY
        mat_i = ages_d >= 90.0
        anc_p = float(np.clip(np.mean(y_obs_s[mat_i]) * 1.15 if int(np.sum(mat_i)) >= 100 else np.mean(y_obs_s) * 1.25, 0.03, 0.65))
        pw_i = float(np.clip((1.0 - anc_p) / max(anc_p, 0.03), 1.2, 8.0))
        for sd in seeds:
            m_init = build_host_backbone_v3("mlp", X[:16], seed=sd)
            m_init, _ = train_supervised_v3(m_init, X[:s_start], y_obs_s, pos_weight=pw_i, epochs=12, lr=8e-4, seed=sd)
            _, Z0 = predict_logits_and_embeddings_v3(m_init, X[:s_start])
            cached_z0[(sname, sd)] = (Z0, ni, s_start)

    print("\n=== [RQ4b] Running Unconfounded 2x2 Factorial DriftLens + Stationary FAR + Shift Detection Delay ===", flush=True)
    out_configs: Dict[str, Dict] = {}

    for c_name, c_kw in configs:
        stream_acc: Dict[str, Dict] = {}
        stat_fars: List[float] = []
        shift_detected_flags: List[float] = []
        shift_delays_commits_cond: List[float] = []
        shift_delays_wins_cond: List[float] = []

        for sname in rq4b_streams:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            onset = int(b["drift_onset_idx"])
            s_start = ni + nv

            m_runs: Dict[str, List[Dict]] = {m: [] for m in methods_m}
            al_rates: List[float] = []

            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts, ni, nv, host_type="mlp", methods=methods_m, drift_onset_idx=onset, seed=sd, **c_kw
                )
                al_rates.append(float(res["methods"]["Survival_PU"]["alarm_rate"]))
                for m in methods_m:
                    mo = res["methods"][m]
                    m_runs[m].append(
                        evaluate_horizon_v3(
                            y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                            per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                        )
                    )

                # 1. Stationary FAR on `Z0` from the EXACT `m_causal_init`, including rolling reference refresh!
                if sname in REPRESENTATIVE_10_STREAMS:
                    if c_kw["monitor_mode"] != "periodic":
                        Z0, ni0, s0 = cached_z0[(sname, sd)]
                        rng_null = np.random.RandomState(sd + 900)
                        mon_null = DriftMonitorV3(
                            mode=c_kw["monitor_mode"],
                            calib_sampling=c_kw["calib_sampling"],
                            reference_refresh=c_kw["reference_refresh"],
                            percentile=c_kw["drift_percentile"],
                            cooldown_windows=c_kw["drift_cooldown"],
                            seed=sd,
                        ).fit_reference(Z0[:ni0], Z0[ni0:s0], window_size=400)
                        null_pool = Z0[ni0:s0] if len(Z0[ni0:s0]) >= 400 else Z0
                        null_stream_blocks = [
                            DriftMonitorV3._sample_block_bootstrap(null_pool, w=400, block_len=50, rng=rng_null)
                            for _ in range(20)
                        ]
                        fires_null = 0
                        rolling_buf = [Z0[max(0, s0 - 800) : s0]]
                        for sub_null in null_stream_blocks:
                            _, f_n = mon_null.score_window(sub_null)
                            rolling_buf.append(sub_null)
                            if f_n:
                                fires_null += 1
                                ref_cat = np.vstack(rolling_buf)[-1200:]
                                mon_null.update_reference(ref_cat, window_size=400)
                        stat_fars.append(fires_null / 20.0)
                    else:
                        stat_fars.append(1.0 / 3.0)

                # 2. Shift detection delay & undetected rate from ACTUAL `executed_alarm_windows` (`R3-C7`)!
                if sname in shift_streams and onset >= s_start:
                    exec_wins = res["methods"]["Survival_PU"]["executed_alarm_windows"]
                    post_wins = [w for w in exec_wins if int(w["end"]) > onset]
                    fired_post = [idx_p for idx_p, w in enumerate(post_wins) if w["fired"]]
                    if fired_post:
                        w_hit = post_wins[fired_post[0]]
                        shift_detected_flags.append(1.0)
                        shift_delays_wins_cond.append(float(fired_post[0] + 1))
                        shift_delays_commits_cond.append(float(max(int(w_hit["end"]) - onset, 0)))
                    else:
                        shift_detected_flags.append(0.0)

            stream_acc[sname] = {
                "alarm_rate": float(np.mean(al_rates)),
                "methods": {
                    m: {
                        "ap": float(np.mean([d["ap"] for d in m_runs[m]])),
                        "f1": float(np.mean([d["f1"] for d in m_runs[m]])),
                        "f1_th": float(np.mean([d["f1_th"] for d in m_runs[m]])),
                    }
                    for m in methods_m
                },
            }

        means_c = {
            m: {
                "ap": float(np.mean([stream_acc[s]["methods"][m]["ap"] for s in REPRESENTATIVE_10_STREAMS])),
                "f1": float(np.mean([stream_acc[s]["methods"][m]["f1"] for s in REPRESENTATIVE_10_STREAMS])),
                "f1_th": float(np.mean([stream_acc[s]["methods"][m]["f1_th"] for s in REPRESENTATIVE_10_STREAMS])),
            }
            for m in methods_m
        }
        undet_rate = float(1.0 - np.mean(shift_detected_flags)) if shift_detected_flags else 0.0
        cond_del_w = float(np.mean(shift_delays_wins_cond)) if shift_delays_wins_cond else None
        cond_del_c = float(np.mean(shift_delays_commits_cond)) if shift_delays_commits_cond else None
        out_configs[c_name] = {
            "params": c_kw,
            "mean_alarm_rate_10streams": float(np.mean([stream_acc[s]["alarm_rate"] for s in REPRESENTATIVE_10_STREAMS])),
            "stationary_false_alarm_rate": float(np.mean(stat_fars)) if stat_fars else 0.0,
            "undetected_shift_rate": undet_rate,
            "conditional_shift_delay_windows_given_detected": cond_del_w,
            "conditional_shift_delay_commits_given_detected": cond_del_c,
            "means_10streams": means_c,
            "per_stream": stream_acc,
        }
        del_str = f"{cond_del_c:5.0f}c" if cond_del_c is not None else "  N/A"
        print(
            f"  [RQ4b] {c_name:38s} | Alarm={out_configs[c_name]['mean_alarm_rate_10streams']:.1%}, "
            f"StatFAR={out_configs[c_name]['stationary_false_alarm_rate']:.1%}, "
            f"Undet={undet_rate:.1%}, CondDelay={del_str} | "
            f"AP: B0={means_c['B0_Frozen']['ap']:.4f}, B2_4W={means_c['B2_4W']['ap']:.4f}, SPU={means_c['Survival_PU']['ap']:.4f}",
            flush=True,
        )

    payload = {"provenance": build_provenance_header(base_dir), "configs": out_configs}
    save_json_atomic(out_path, payload)
    return payload


def _perturb_szz_labels_and_event_times(
    commit_ts: np.ndarray,
    y_ref: np.ndarray,
    fix_ts_ref: np.ndarray,
    t0: float,
    alpha_fp: float,
    beta_fn: float,
    fn_mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    y_out = np.asarray(y_ref, dtype=np.int32).copy()
    f_out = np.asarray(fix_ts_ref, dtype=np.float64).copy()

    pre_rev = (commit_ts < t0) & (y_ref == 1) & np.isfinite(fix_ts_ref) & (fix_ts_ref <= t0)
    if int(np.sum(pre_rev)) >= 10:
        emp_delays = np.maximum(fix_ts_ref[pre_rev] - commit_ts[pre_rev], 3600.0)
    else:
        emp_delays = np.array([30.0, 60.0, 120.0, 240.0]) * SECONDS_PER_DAY

    clean_idx = np.where(y_ref == 0)[0]
    pos_idx = np.where(y_ref == 1)[0]

    fp_hit = clean_idx[rng.rand(len(clean_idx)) < float(alpha_fp)]
    if len(fp_hit) > 0:
        sampled_del = rng.choice(emp_delays, size=len(fp_hit), replace=True)
        y_out[fp_hit] = 1
        f_out[fp_hit] = commit_ts[fp_hit] + sampled_del

    fn_hit = pos_idx[rng.rand(len(pos_idx)) < float(beta_fn)]
    if len(fn_hit) > 0:
        if fn_mode == "permanent_inf":
            y_out[fn_hit] = 0
            f_out[fn_hit] = np.inf
        elif fn_mode == "delayed_2x":
            fin_h = fn_hit[np.isfinite(f_out[fn_hit])]
            f_out[fin_h] = commit_ts[fin_h] + 2.0 * np.maximum(f_out[fin_h] - commit_ts[fin_h], 3600.0)
        else:
            raise ValueError(f"Unknown fn_mode={fn_mode}")

    return y_out, f_out


def run_rq4d_semisynthetic_szz_perturbation_v3(base_dir: str, seeds: List[int]) -> Dict:
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    out_path = os.path.join(base_dir, "results_v3", "rq4d_szz_perturbation_v3.json")
    prszz_streams = ["n60_airflow", "n60_angular", "n60_jenkins", "n60_kafka", "n60_pulsar"]
    methods_szz = ["B0_Frozen", "B2_4W", "Wait_90", "Pure_nnPU", "BORB_Native", "ORB_OHT_Native", "Survival_PU"]

    regimes = [
        ("Pop_N60_Shared_PermFN", 0.0842, 0.4715, "permanent_inf", True),
        ("Pop_N60_Indep_PermFN", 0.0842, 0.4715, "permanent_inf", False),
        ("Pop_N60_Shared_Delayed2xFN", 0.0842, 0.4715, "delayed_2x", True),
        ("Mild_Half_Indep_PermFN", 0.0421, 0.2358, "permanent_inf", False),
    ]

    print("\n=== [RQ4d] Running 2x2 Semi-Synthetic SZZ Perturbation (Frozen-Prediction Evaluation) ===", flush=True)
    out_regimes: Dict[str, Dict] = {}

    for r_name, a_fp, b_fn, fn_mode, shared_err in regimes:
        cells_acc = {
            "TrainRef__EvalRef": {m: [] for m in methods_szz},
            "TrainRef__EvalPert": {m: [] for m in methods_szz},
            "TrainPert__EvalRef": {m: [] for m in methods_szz},
            "TrainPert__EvalPert": {m: [] for m in methods_szz},
        }
        for sname in prszz_streams:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y_ref, f_ref, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            s_start = ni + nv
            t0 = float(ts[s_start - 1])

            for sd in seeds:
                y_tr_p, f_tr_p = _perturb_szz_labels_and_event_times(ts, y_ref, f_ref, t0, a_fp, b_fn, fn_mode, seed=sd + 500)
                if shared_err:
                    y_ev_p = y_tr_p
                else:
                    y_ev_p, _ = _perturb_szz_labels_and_event_times(ts, y_ref, f_ref, t0, a_fp, b_fn, "permanent_inf", seed=sd + 9500)

                t1_ref = execute_within_host_counterfactual_suite_v3(
                    X, ts, y_ref, f_ref, ni, nv, host_type="mlp",
                    methods=["B0_Frozen", "B2_4W", "Wait_90", "Pure_nnPU", "Survival_PU"], seed=sd,
                )
                borb_ref = run_borb_competitor_v3(X, ts, y_ref, f_ref, s_start, update_mode="native", seed=sd)
                orb_ref = run_orb_oht_native_v3(X, ts, y_ref, f_ref, s_start, seed=sd)
                comb_ref = {**t1_ref["methods"], "BORB_Native": borb_ref, "ORB_OHT_Native": orb_ref}

                t1_prt = execute_within_host_counterfactual_suite_v3(
                    X, ts, y_tr_p, f_tr_p, ni, nv, host_type="mlp",
                    methods=["B0_Frozen", "B2_4W", "Wait_90", "Pure_nnPU", "Survival_PU"], seed=sd,
                )
                borb_prt = run_borb_competitor_v3(X, ts, y_tr_p, f_tr_p, s_start, update_mode="native", seed=sd)
                orb_prt = run_orb_oht_native_v3(X, ts, y_tr_p, f_tr_p, s_start, seed=sd)
                comb_prt = {**t1_prt["methods"], "BORB_Native": borb_prt, "ORB_OHT_Native": orb_prt}

                for m in methods_szz:
                    mr = comb_ref[m]
                    mp = comb_prt[m]
                    cells_acc["TrainRef__EvalRef"][m].append(
                        evaluate_horizon_v3(y_ref[s_start:], mr["raw_logits"], mr["raw_probs"], mr["cal_probs"], eff[s_start:], per_commit_thresholds=mr["tuned_th_vec"], per_commit_decisions=mr["decisions_th"])["ap"]
                    )
                    cells_acc["TrainRef__EvalPert"][m].append(
                        evaluate_horizon_v3(y_ev_p[s_start:], mr["raw_logits"], mr["raw_probs"], mr["cal_probs"], eff[s_start:], per_commit_thresholds=mr["tuned_th_vec"], per_commit_decisions=mr["decisions_th"])["ap"]
                    )
                    cells_acc["TrainPert__EvalRef"][m].append(
                        evaluate_horizon_v3(y_ref[s_start:], mp["raw_logits"], mp["raw_probs"], mp["cal_probs"], eff[s_start:], per_commit_thresholds=mp["tuned_th_vec"], per_commit_decisions=mp["decisions_th"])["ap"]
                    )
                    cells_acc["TrainPert__EvalPert"][m].append(
                        evaluate_horizon_v3(y_ev_p[s_start:], mp["raw_logits"], mp["raw_probs"], mp["cal_probs"], eff[s_start:], per_commit_thresholds=mp["tuned_th_vec"], per_commit_decisions=mp["decisions_th"])["ap"]
                    )

        out_regimes[r_name] = {
            "alpha_fp": a_fp,
            "beta_fn": b_fn,
            "fn_mode": fn_mode,
            "shared_train_eval_noise": shared_err,
            "cells_mean_ap": {
                ck: {m: float(np.mean(vals)) for m, vals in m_dict.items()}
                for ck, m_dict in cells_acc.items()
            },
        }
        c_pr = out_regimes[r_name]["cells_mean_ap"]["TrainPert__EvalRef"]
        print(
            f"  [RQ4d {r_name:28s}] TrainPert->EvalRef AP: B0={c_pr['B0_Frozen']:.4f}, "
            f"B2_4W={c_pr['B2_4W']:.4f}, BORB={c_pr['BORB_Native']:.4f}, ORB={c_pr['ORB_OHT_Native']:.4f}, SPU={c_pr['Survival_PU']:.4f}",
            flush=True,
        )

    payload = {
        "provenance": build_provenance_header(base_dir),
        "note": "Semi-synthetic perturbation over automated PR-SZZ reference labels parameterized by Bludau & Pretschner (SQJ 2026, Tables 4-5); evaluation perturbation applied strictly to frozen predictions.",
        "regimes": out_regimes,
    }
    save_json_atomic(out_path, payload)
    return payload


def run_rq4e_window_and_tail_sensitivity_v3(base_dir: str, seeds: List[int]) -> Dict:
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    out_path = os.path.join(base_dir, "results_v3", "rq4e_sensitivity_sweeps_v3.json")
    methods_w = ["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"]

    print("\n=== [RQ4e] Running Fixed-1200 Window Sweep & Tail-Cap / Shrinkage Sensitivity ===", flush=True)
    win_results = {}
    for w_sz in (200, 400, 600, 800):
        full_acc = {m: [] for m in methods_w}
        tr1200_acc = {m: [] for m in methods_w}
        for sname in REPRESENTATIVE_10_STREAMS:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv, onset = int(b["n_init"]), int(b["n_val"]), int(b["drift_onset_idx"])
            s_start = ni + nv
            on_rel = (onset - s_start) if onset >= s_start else 0
            tr_end = min(on_rel + 1200, len(y) - s_start)

            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts, ni, nv, host_type="mlp", methods=methods_w, window_size=w_sz, drift_onset_idx=onset, seed=sd
                )
                for m in methods_w:
                    mo = res["methods"][m]
                    full_acc[m].append(
                        evaluate_horizon_v3(
                            y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                            per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                            batch_window_size=w_sz,
                        )["ap"]
                    )
                    if onset >= s_start:
                        tr1200_acc[m].append(
                            evaluate_horizon_v3(
                                y[s_start + on_rel : s_start + tr_end],
                                mo["raw_logits"][on_rel:tr_end],
                                mo["raw_probs"][on_rel:tr_end],
                                mo["cal_probs"][on_rel:tr_end],
                                eff[s_start + on_rel : s_start + tr_end],
                                per_commit_thresholds=mo["tuned_th_vec"][on_rel:tr_end],
                                per_commit_decisions=mo["decisions_th"][on_rel:tr_end],
                                batch_window_size=w_sz,
                            )["ap"]
                        )
        win_results[str(w_sz)] = {
            "full_stream_ap": {m: float(np.mean(full_acc[m])) for m in methods_w},
            "fixed_1200_post_onset_ap": {m: float(np.mean(tr1200_acc[m])) for m in methods_w},
        }
        print(
            f"  [RQ4e W={w_sz}] Full AP: B0={win_results[str(w_sz)]['full_stream_ap']['B0_Frozen']:.4f}, "
            f"B2_4W={win_results[str(w_sz)]['full_stream_ap']['B2_4W']:.4f}, SPU={win_results[str(w_sz)]['full_stream_ap']['Survival_PU']:.4f} | "
            f"Fixed-1200 SPU={win_results[str(w_sz)]['fixed_1200_post_onset_ap']['Survival_PU']:.4f}",
            flush=True,
        )

    tail_results = {}
    for tc in (0.65, 0.75, 0.85, 0.94):
        aps, briers = [], []
        for sname in REPRESENTATIVE_10_STREAMS:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            s_start = ni + nv
            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts, ni, nv, host_type="mlp", methods=["Survival_PU"], spu_overrides={"tail_cap_override": tc}, seed=sd
                )
                mo = res["methods"]["Survival_PU"]
                ev = evaluate_horizon_v3(
                    y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                    per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                )
                aps.append(ev["ap"])
                briers.append(ev["brier_cal"])
        tail_results[f"{tc:.2f}"] = {"ap": float(np.mean(aps)), "brier_cal": float(np.mean(briers))}
        print(f"  [RQ4e tail_cap={tc:.2f}] SPU AP={tail_results[f'{tc:.2f}']['ap']:.4f}, Brier_cal={tail_results[f'{tc:.2f}']['brier_cal']:.4f}", flush=True)

    m0_results = {}
    for m0_val in (0.0, 25.0, 50.0, 100.0):
        aps = []
        for sname in REPRESENTATIVE_10_STREAMS:
            b = np.load(os.path.join(b_dir, f"{sname}.npz"))
            X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
            ni, nv = int(b["n_init"]), int(b["n_val"])
            s_start = ni + nv
            for sd in seeds:
                res = execute_within_host_counterfactual_suite_v3(
                    X, ts, y, fts, ni, nv, host_type="mlp", methods=["Survival_PU"], spu_overrides={"ht_shrinkage_m0": m0_val}, seed=sd
                )
                mo = res["methods"]["Survival_PU"]
                aps.append(
                    evaluate_horizon_v3(
                        y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                        per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                    )["ap"]
                )
        m0_results[str(int(m0_val))] = {"ap": float(np.mean(aps))}

    payload = {
        "provenance": build_provenance_header(base_dir),
        "window_size_sweep": win_results,
        "tail_cap_sweep": tail_results,
        "ht_shrinkage_m0_sweep": m0_results,
    }
    save_json_atomic(out_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run N05 V3 Experimental Suite")
    parser.add_argument("--base-dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all_dev", "phase_a", "phase_b", "mechanisms", "confirmatory"],
        default="all_dev",
    )
    parser.add_argument("--seeds-a", type=int, default=5)
    parser.add_argument("--seeds-b", type=int, default=10)
    parser.add_argument("--seeds-mech", type=int, default=5)
    args = parser.parse_args()

    seeds_a = [42 + i for i in range(args.seeds_a)]
    seeds_b = [42 + i for i in range(args.seeds_b)]
    seeds_lgb = [42 + i for i in range(min(args.seeds_b, 5))]
    seeds_mech = [42 + i for i in range(args.seeds_mech)]

    if args.phase in ("all_dev", "phase_a"):
        run_phase_a_benchmark_a(args.base_dir, seeds=seeds_a)
    if args.phase in ("all_dev", "phase_b"):
        run_phase_b_benchmark_b(args.base_dir, seeds_mlp=seeds_b, seeds_lgb=seeds_lgb)
    if args.phase in ("all_dev", "mechanisms"):
        run_rq3_ablations_v3(args.base_dir, seeds=seeds_mech)
        run_rq4a_future_only_latency_v3(args.base_dir, seeds=seeds_mech)
        run_rq4b_factorial_drift_monitors_v3(args.base_dir, seeds=seeds_mech)
        run_rq4d_semisynthetic_szz_perturbation_v3(args.base_dir, seeds=seeds_mech)
        run_rq4e_window_and_tail_sensitivity_v3(args.base_dir, seeds=seeds_mech)
    if args.phase == "confirmatory":
        run_phase_c_confirmatory(args.base_dir, seeds=seeds_b)


if __name__ == "__main__":
    main()
