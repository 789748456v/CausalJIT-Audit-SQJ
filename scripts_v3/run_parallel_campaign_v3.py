"""
Multi-Worker Resumable Orchestrator for N05 V3 (`scripts_v3/run_parallel_campaign_v3.py`).

Strictly preserves 100% immutability of `src_v3/*.py` (`code_sha256` unchanged!):
  1. Populates SHA-256 fingerprint-verified checkpoints in `results_v3/per_cell_a/` and
     `results_v3/per_stream_b/` in parallel across `N_WORKERS` processes (each pinned to 2 CPU threads
     to eliminate OpenMP contention on the 24-core + RTX 4080 workstation).
  2. Executes the 5 independent mechanism suites (`rq3`, `rq4a`, `rq4b`, `rq4d`, `rq4e`) in parallel
     using their exact unmodified functions from `src_v3/run_v3_experiments.py`.
  3. Calls unmodified `run_phase_a_benchmark_a` and `run_phase_b_benchmark_b` from
     `src_v3/run_v3_experiments.py` to verify all checkpoints, run Benchmark A sensitivity, and write
     `results_v3/benchmark_a_multihost_v3.json` and `results_v3/benchmark_b_main_v3.json`.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from typing import Any, Dict, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

import torch
torch.set_num_threads(2)

from run_v3_experiments import (
    TRACK1_MAIN_METHODS,
    compute_task_run_fingerprint,
    evaluate_and_aggregate_stream_runs,
    load_verified_checkpoint_or_none,
    run_phase_a_benchmark_a,
    run_phase_b_benchmark_b,
    run_rq3_ablations_v3,
    run_rq4a_future_only_latency_v3,
    run_rq4b_factorial_drift_monitors_v3,
    run_rq4d_semisynthetic_szz_perturbation_v3,
    run_rq4e_window_and_tail_sensitivity_v3,
    save_json_atomic,
    select_skip_mode_strictly_on_dev_slice,
)


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch as _th
    _th.set_num_threads(2)


def _execute_unit_job(job: Dict[str, Any]) -> Tuple[str, float]:
    t0 = time.perf_counter()
    kind = job["kind"]
    base_dir = job["base_dir"]

    if kind == "stream_ckpt":
        ckpt_p = job["ckpt_p"]
        fp = job["fp"]
        art_f = job.get("art_f")
        existing = load_verified_checkpoint_or_none(ckpt_p, fp)
        if existing is not None and (art_f is None or os.path.exists(art_f)):
            return (f"[SKIP-VERIFIED] {job['label']}", 0.0)

        res = evaluate_and_aggregate_stream_runs(
            stream_name=job["stream_name"],
            bundle_path=job["bundle_path"],
            host_type=job["host_type"],
            seeds=job["seeds"],
            meta=job.get("meta"),
            skip_mode=job["skip_mode"],
            include_track2=job["include_track2"],
            methods_track1=job["methods_track1"],
            save_commit_artifact_dir=job.get("save_commit_artifact_dir"),
            run_fingerprint=fp,
        )
        save_json_atomic(ckpt_p, res)
        ap_spu = res["methods"]["Survival_PU"]["full_stream"]["ap"]
        ap_b2 = res["methods"]["B2_4W"]["full_stream"]["ap"]
        return (f"[DONE-CKPT] {job['label']} (B2={ap_b2:.4f}, SPU={ap_spu:.4f})", time.perf_counter() - t0)

    elif kind == "mech":
        m_name = job["mech_name"]
        out_p = os.path.join(base_dir, "results_v3", job["out_json"])
        if os.path.exists(out_p):
            return (f"[SKIP-EXISTS] {m_name}", 0.0)
        seeds = job["seeds"]
        if m_name == "rq3":
            run_rq3_ablations_v3(base_dir, seeds=seeds)
        elif m_name == "rq4a":
            run_rq4a_future_only_latency_v3(base_dir, seeds=seeds)
        elif m_name == "rq4b":
            run_rq4b_factorial_drift_monitors_v3(base_dir, seeds=seeds)
        elif m_name == "rq4d":
            run_rq4d_semisynthetic_szz_perturbation_v3(base_dir, seeds=seeds)
        elif m_name == "rq4e":
            run_rq4e_window_and_tail_sensitivity_v3(base_dir, seeds=seeds)
        else:
            raise ValueError(f"Unknown mech_name={m_name}")
        return (f"[DONE-MECH] {m_name} -> {job['out_json']}", time.perf_counter() - t0)

    raise ValueError(f"Unknown job kind={kind}")


def build_all_jobs(base_dir: str) -> List[Dict[str, Any]]:
    seeds_a = [42, 43, 44, 45, 46]
    seeds_b = [42 + i for i in range(10)]
    seeds_lgb = [42, 43, 44, 45, 46]
    seeds_mech = [42, 43, 44, 45, 46]

    jobs: List[Dict[str, Any]] = []

    # 1. Put the 5 Mechanism jobs FIRST in queue so long-pole jobs start immediately alongside stream jobs!
    for m_name, out_j in [
        ("rq4b", "rq4b_factorial_monitors_v3.json"),
        ("rq4a", "rq4a_latency_intervention_v3.json"),
        ("rq3", "rq3_ablations_v3.json"),
        ("rq4d", "rq4d_szz_perturbation_v3.json"),
        ("rq4e", "rq4e_sensitivity_sweeps_v3.json"),
    ]:
        jobs.append({
            "kind": "mech",
            "base_dir": base_dir,
            "mech_name": m_name,
            "out_json": out_j,
            "seeds": seeds_mech,
            "label": f"MECH:{m_name}",
        })

    # 2. Phase A cells
    skip_sel = select_skip_mode_strictly_on_dev_slice(base_dir)
    chosen_skip = skip_sel["chosen_skip_mode"]
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    per_cell_dir = os.path.join(base_dir, "results_v3", "per_cell_a")
    art_a = os.path.join(base_dir, "results_v3", "commit_predictions", "benchmark_a")
    os.makedirs(per_cell_dir, exist_ok=True)
    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        mm_manifest = json.load(f)

    host_specs = [
        ("att_cnn", "tok", ["none", chosen_skip]),
        ("codebert", "cb", ["none", chosen_skip]),
        ("deepjit_cnn", "tok", ["none", chosen_skip]),
        ("lightgbm", "svd", ["none"]),
    ]
    for m_info in mm_manifest:
        sname = m_info["name"]
        need_art = sname in ("qt_natural_25k", "openstack_natural_13k")
        for h_type, rep_key, skip_modes in host_specs:
            b_path = os.path.join(mm_dir, f"{sname}_{rep_key}.npz")
            inc_t2 = (h_type == "lightgbm")
            for sm in skip_modes:
                ck_key = f"{sname}__{h_type}__{sm}"
                fp = compute_task_run_fingerprint(base_dir, b_path, h_type, sm, seeds_a, inc_t2, TRACK1_MAIN_METHODS)
                jobs.append({
                    "kind": "stream_ckpt",
                    "base_dir": base_dir,
                    "label": f"PhaseA:{ck_key}",
                    "stream_name": sname,
                    "bundle_path": b_path,
                    "host_type": h_type,
                    "skip_mode": sm,
                    "seeds": seeds_a,
                    "meta": m_info,
                    "include_track2": inc_t2,
                    "methods_track1": TRACK1_MAIN_METHODS,
                    "save_commit_artifact_dir": art_a if need_art else None,
                    "art_f": os.path.join(art_a, f"{sname}__{h_type}__{sm}__seed{seeds_a[0]}.npz") if need_art else None,
                    "ckpt_p": os.path.join(per_cell_dir, f"{ck_key}.json"),
                    "fp": fp,
                })

    # 3. Phase B streams (sort larger streams earlier for optimal load balancing!)
    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    per_stream_dir = os.path.join(base_dir, "results_v3", "per_stream_b")
    art_b = os.path.join(base_dir, "results_v3", "commit_predictions", "benchmark_b")
    os.makedirs(per_stream_dir, exist_ok=True)
    with open(os.path.join(b_dir, "manifest_b_v3.json"), "r", encoding="utf-8") as f:
        man_b = json.load(f)

    nat_sorted = sorted(man_b["natural_streams"], key=lambda d: int(d["N"]), reverse=True)
    shf_sorted = sorted(man_b["controlled_shift_streams"], key=lambda d: int(d["N"]), reverse=True)

    for item in nat_sorted + shf_sorted:
        sname = item["name"]
        b_path = os.path.join(b_dir, f"{sname}.npz")
        fp_s = compute_task_run_fingerprint(base_dir, b_path, "mlp", "none", seeds_b, True, TRACK1_MAIN_METHODS)
        jobs.append({
            "kind": "stream_ckpt",
            "base_dir": base_dir,
            "label": f"PhaseB-MLP:{sname}",
            "stream_name": sname,
            "bundle_path": b_path,
            "host_type": "mlp",
            "skip_mode": "none",
            "seeds": seeds_b,
            "meta": None,
            "include_track2": True,
            "methods_track1": TRACK1_MAIN_METHODS,
            "save_commit_artifact_dir": art_b,
            "art_f": os.path.join(art_b, f"{sname}__mlp__none__seed{seeds_b[0]}.npz"),
            "ckpt_p": os.path.join(per_stream_dir, f"{sname}__mlp.json"),
            "fp": fp_s,
        })

    lgb_methods = [
        "B0_Frozen", "B1b_4W_Oracle", "B2_1W", "B2_4W", "Wait_90",
        "Pure_nnPU", "FixedCurve_PU", "Teacher_Only_Exact", "Teacher_Only_Matched", "Survival_PU",
    ]
    for item in nat_sorted:
        sname = item["name"]
        b_path = os.path.join(b_dir, f"{sname}.npz")
        fp_l = compute_task_run_fingerprint(base_dir, b_path, "lightgbm", "none", seeds_lgb, False, lgb_methods)
        jobs.append({
            "kind": "stream_ckpt",
            "base_dir": base_dir,
            "label": f"PhaseB-LGB:{sname}",
            "stream_name": sname,
            "bundle_path": b_path,
            "host_type": "lightgbm",
            "skip_mode": "none",
            "seeds": seeds_lgb,
            "meta": None,
            "include_track2": False,
            "methods_track1": lgb_methods,
            "save_commit_artifact_dir": None,
            "art_f": None,
            "ckpt_p": os.path.join(per_stream_dir, f"{sname}__lightgbm.json"),
            "fp": fp_l,
        })

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t_all = time.perf_counter()
    jobs = build_all_jobs(args.base_dir)
    total = len(jobs)
    print(f"=== Launching Parallel V3 Campaign: {total} total jobs across {args.workers} workers ===", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futs = {pool.submit(_execute_unit_job, jb): jb["label"] for jb in jobs}
        for fut in as_completed(futs):
            lbl = futs[fut]
            msg, dt = fut.result()
            done += 1
            elapsed_m = (time.perf_counter() - t_all) / 60.0
            print(f"[{done:3d}/{total:3d} | +{elapsed_m:5.1f}m] {msg} ({dt:.1f}s)", flush=True)

    print("\n=== Finalizing Master Phase A & Phase B JSONs from Verified Checkpoints ===", flush=True)
    if not os.path.exists(os.path.join(args.base_dir, "results_v3", "benchmark_a_multihost_v3.json")):
        run_phase_a_benchmark_a(args.base_dir, seeds=[42, 43, 44, 45, 46])
    if not os.path.exists(os.path.join(args.base_dir, "results_v3", "benchmark_b_main_v3.json")):
        run_phase_b_benchmark_b(args.base_dir, seeds_mlp=[42 + i for i in range(10)], seeds_lgb=[42, 43, 44, 45, 46])

    print(f"\n==> ALL V3 EXPERIMENTAL SUITES COMPLETED in {(time.perf_counter() - t_all)/60.0:.2f} minutes!", flush=True)


if __name__ == "__main__":
    main()
