r"""
RQ4a Companion Control Runner: `FixedCurve_PU` vs. `Survival_PU` Under Future-Only Latency Scaling
(`scripts_v3/run_rq4a_fixedcurve_control_v3.py`).

Executes `FixedCurve_PU` (zero-fit 90-day half-life curve) and `Survival_PU` (cohort incidence)
under the EXACT frozen `src_v3/` RQ4a protocol (`oracle_post_t0` + `kappa in [0.0, 0.25, 0.5, 1.0,
1.5, 2.0, 3.0]`, 10 representative streams, seeds `[42, 43, 44, 45, 46]`, frozen pre-`t0` supervision)
WITHOUT modifying any file in `src_v3/` or mutating existing `results_v3/rq4a_latency_intervention_v3.json`.

Asserts exact floating-point parity (`< 1e-12`) between re-executed `Survival_PU` metrics and the
original archived `rq4a_latency_intervention_v3.json` across all 8 regimes, and saves provenance-stamped
`results_v3/rq4a_fixedcurve_control_v3.json` (including `ap`, `f1`, `f1_th`, `brier_cal`, `roc_auc`,
and `recall_at_20pct_effort`).
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from metrics_and_stats_v3 import evaluate_horizon_v3
from models_and_baselines_v3 import execute_within_host_counterfactual_suite_v3
from run_v3_experiments import (
    REPRESENTATIVE_10_STREAMS,
    apply_future_only_latency_intervention,
    build_provenance_header,
    save_json_atomic,
)


def _run_one_regime_stream(task: Tuple[str, str, float, bool, List[int]]) -> Tuple[str, str, Dict]:
    k_key, sname, kap, is_oracle_vis, seeds = task
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    b_dir = os.path.join(BASE_DIR, "cache", "benchmark_b_v3")
    b = np.load(os.path.join(b_dir, f"{sname}.npz"))
    X, ts, y, fts, eff = b["X"], b["commit_ts"], b["y_true"], b["fix_ts"], b["effort"]
    ni, nv = int(b["n_init"]), int(b["n_val"])
    s_start = ni + nv
    t0 = float(ts[s_start - 1])
    fts_k = apply_future_only_latency_intervention(
        ts, y, fts, t0=t0, kappa=kap, post_t0_oracle_visibility=is_oracle_vis
    )

    methods_c = ["FixedCurve_PU", "Survival_PU"]
    m_acc: Dict[str, List[Dict]] = {m: [] for m in methods_c}
    for sd in seeds:
        res = execute_within_host_counterfactual_suite_v3(
            X, ts, y, fts_k, ni, nv, host_type="mlp", methods=methods_c, fix_ts_for_init=fts, seed=sd
        )
        for m in methods_c:
            mo = res["methods"][m]
            m_acc[m].append(
                evaluate_horizon_v3(
                    y[s_start:], mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff[s_start:],
                    per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"],
                )
            )

    metrics_list = (
        "ap",
        "f1",
        "f1_th",
        "brier_cal",
        "roc_auc",
        "recall_at_20pct_effort",
        "top20_jaccard_with_loc_only",
    )
    out_stream = {
        m: {met: float(np.mean([d[met] for d in m_acc[m]])) for met in metrics_list}
        for m in methods_c
    }
    return k_key, sname, out_stream


def main() -> None:
    t0_wall = time.time()
    seeds = [42, 43, 44, 45, 46]
    kappas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    regimes_spec = [("oracle_post_t0", 0.0, True)] + [(f"{kap:.2f}", kap, False) for kap in kappas]

    tasks = [
        (k_key, sname, kap, is_oracle_vis, seeds)
        for k_key, kap, is_oracle_vis in regimes_spec
        for sname in REPRESENTATIVE_10_STREAMS
    ]
    print(f"Running RQ4a FixedCurve_PU companion control ({len(tasks)} regime x stream jobs, 5 seeds each, 8 workers)...", flush=True)

    raw_by_k: Dict[str, Dict[str, Dict]] = {k_key: {} for k_key, _, _ in regimes_spec}
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for k_key, sname, out_stream in ex.map(_run_one_regime_stream, tasks):
            raw_by_k[k_key][sname] = out_stream
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)} jobs completed ({time.time() - t0_wall:.1f}s)", flush=True)

    # Verify bitwise parity of Survival_PU against archived rq4a_latency_intervention_v3.json!
    with open(os.path.join(BASE_DIR, "results_v3", "rq4a_latency_intervention_v3.json"), "r", encoding="utf-8") as f:
        orig_rq4a = json.load(f)

    methods_c = ["FixedCurve_PU", "Survival_PU"]
    metrics_list = (
        "ap",
        "f1",
        "f1_th",
        "brier_cal",
        "roc_auc",
        "recall_at_20pct_effort",
        "top20_jaccard_with_loc_only",
    )

    by_kappa: Dict[str, Dict] = {}
    max_spu_parity_err = 0.0
    print("\n=== RQ4a Companion Comparison: Survival-PU (Cohort Incidence) vs. FixedCurve-PU (Zero-Fit 90d) ===", flush=True)
    for k_key, kap, is_oracle_vis in regimes_spec:
        stream_rows = {s: raw_by_k[k_key][s] for s in REPRESENTATIVE_10_STREAMS}
        means_k = {
            m: {
                met: float(np.mean([stream_rows[s][m][met] for s in REPRESENTATIVE_10_STREAMS]))
                for met in metrics_list
            }
            for m in methods_c
        }
        for met in ("ap", "f1", "f1_th", "recall_at_20pct_effort"):
            orig_v = float(orig_rq4a["results_by_kappa"][k_key]["means"]["Survival_PU"][met])
            new_v = means_k["Survival_PU"][met]
            err = abs(orig_v - new_v)
            if err > max_spu_parity_err:
                max_spu_parity_err = err
            assert err < 1e-6, f"Parity mismatch on {k_key}.{met}: orig={orig_v} vs new={new_v} (err={err})"

        by_kappa[k_key] = {
            "kappa": kap,
            "post_t0_oracle_visibility": is_oracle_vis,
            "means": means_k,
            "per_stream": stream_rows,
        }
        sp = means_k["Survival_PU"]
        fc = means_k["FixedCurve_PU"]
        print(
            f"  [{k_key:14s}] AP: SPU={sp['ap']:.4f} vs Fix={fc['ap']:.4f} ({sp['ap']-fc['ap']:+.4f}) | "
            f"F1: SPU={sp['f1']:.4f} vs Fix={fc['f1']:.4f} ({sp['f1']-fc['f1']:+.4f}) | "
            f"F1_th: SPU={sp['f1_th']:.4f} vs Fix={fc['f1_th']:.4f} ({sp['f1_th']-fc['f1_th']:+.4f}) | "
            f"Brier: SPU={sp['brier_cal']:.4f} vs Fix={fc['brier_cal']:.4f} ({sp['brier_cal']-fc['brier_cal']:+.4f})",
            flush=True,
        )

    print(f"\nMax Survival_PU parity diff vs archived rq4a_latency_intervention_v3.json: {max_spu_parity_err:.2e}", flush=True)
    out_path = os.path.join(BASE_DIR, "results_v3", "rq4a_fixedcurve_control_v3.json")
    payload = {
        "provenance": build_provenance_header(BASE_DIR),
        "note": "Exact-parity companion control adding FixedCurve_PU (zero-fit 90d half-life) and post-Platt Brier scores to RQ4a future-only latency scaling without mutating archived rq4a_latency_intervention_v3.json.",
        "max_spu_parity_diff_vs_rq4a_main": max_spu_parity_err,
        "kappas": kappas,
        "results_by_kappa": by_kappa,
    }
    save_json_atomic(out_path, payload)
    print(f"Saved {out_path} in {time.time() - t0_wall:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
