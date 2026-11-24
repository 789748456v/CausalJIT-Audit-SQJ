"""
Master V3 Empirical Digest, Publication-Grade Plotting, LaTeX Table Generator,
& Exact Cell-Level Registry Builder (`scripts_v3/analyze_and_plot_v3.py`).

Strictly read-only with respect to `src_v3/*.py` so `code_sha256` provenance never changes.
Outputs:
  1. `figures/fig1_multimodal_audit_v3.{pdf,png}` .. `figures/fig8_llm_and_runtime_forensics_v3.{pdf,png}`
  2. `results_v3/manuscript_derived_aggregates_v3.json` (provenance-stamped deterministic cross-stream summaries)
  3. `results_v3/manuscript_cell_registry_v3.json` (100% semantic `(table, row, col, role)` cell bindings)
  4. `<scratch>/v3_summary_digest.md` & `<scratch>/v3_tables_generated.tex`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from config_v3 import SECONDS_PER_DAY, V3_CONFIG
from models_and_baselines_v3 import observe_labels_at_time
from run_v3_experiments import build_provenance_header, save_json_atomic


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.4,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Publication-clean display names (ZERO internal codes N46/N60/N33/cabral_*/n60_*!)
METHOD_DISPLAY: Dict[str, str] = {
    "B1a_Full_Hindsight": "B1a* (Full Hindsight)",
    "B1b_4W_Oracle": "B1b* (4W Oracle)",
    "B1b_1W_Oracle": "B1b* (1W Oracle)",
    "B0_Frozen": "B0 (Frozen)",
    "B2_1W": "B2-1W (Censored)",
    "B2_4W": "B2-4W (Censored)",
    "B2_4W_Bal": "B2-4W-Bal",
    "B2_2000": "B2-2000",
    "Wait_30": "Wait-30",
    "Wait_60": "Wait-60",
    "Wait_90": "Wait-90",
    "Pure_nnPU": "Pure-nnPU",
    "FixedCurve_PU": "FixedCurve-PU (90d)",
    "ODaSC_Host": "ODaSC (Host)",
    "Teacher_Only_Exact": "Teacher-Only (Exact)",
    "Teacher_Only_Matched": "Teacher-Only (Matched)",
    "Survival_PU": "Survival-PU",
    "Survival_PU_WithGate": "Survival-PU (+Legacy Gate)",
    "LOC_Only": "LOC-Only",
    "LApredict_B0": "LApredict (Frozen)",
    "LApredict_Adapt": "LApredict (Adapt-4W)",
    "Churn_LR_B0": "Churn-LR (Frozen)",
    "Churn_LR_Adapt": "Churn-LR (Adapt-4W)",
    "Full_LR_B0": "Full-LR (Frozen)",
    "Full_LR_Adapt": "Full-LR (Adapt-4W)",
    "BORB_Native": "BORB (ElasticNet)",
    "BORB_Alarm": "BORB-Alarm (ElasticNet)",
    "ORB_OHT_Native": "ORB (Hoeffding)",
    "ODaSC_OHT_Native": "ODaSC (Hoeffding)",
}

STREAM_DISPLAY: Dict[str, str] = {
    "qt_natural_25k": "Qt Natural (25.7k)",
    "openstack_natural_13k": "OpenStack Natural (13.3k)",
    "qt_abrupt_qtbase_qtdecl": "Qt Shift (qtbase→qtdecl)",
    "qt_abrupt_core_ux": "Qt Shift (Core→UX)",
    "abrupt_nova_zulip": "Shift: Nova→Zulip",
    "abrupt_ignite_cassandra": "Shift: Ignite→Cassandra",
    "abrupt_airflow_pulsar": "Shift: Airflow→Pulsar",
    "abrupt_deepjit_nova_neutron_pure13": "Shift: Nova→Neutron",
    "conf_apache_hdfs": "Apache HDFS",
    "conf_apache_mapreduce": "Apache MapReduce",
    "conf_apache_spark": "Apache Spark",
    "conf_apache_zeppelin": "Apache Zeppelin",
    "conf_apache_zookeeper": "Apache ZooKeeper",
}

PALETTE = {
    "oracle": "#D97706",
    "b0": "#475569",
    "b2": "#DC2626",
    "wait": "#2563EB",
    "nnpu": "#7C3AED",
    "fixed": "#0891B2",
    "teacher": "#0D9488",
    "spu": "#059669",
    "track2": "#EA580C",
    "gate": "#64748B",
}


def _load_json(res_dir: str, fname: str) -> Dict:
    p = os.path.join(res_dir, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required V3 result file missing: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_empirical_window_censoring_stats(base_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Computes exact causal right-censoring rates inside 1W (`W=400`) and 4W (`1600`) windows:
      1. Window-averaged proportions across all scheduled 400-commit stream boundaries (`cens_1w`, `cens_4w`),
         plus pooled defective-commit denominators (`n_windows`, `pooled_bugs_1w`, `pooled_cens_1w`).
      2. Actual fired-alarm-conditioned rates (`alarm_cens_1w`, `alarm_cens_4w`, `n_fired_alarms`) extracted
         from saved seed-42 `shared_model_gen_id` transitions in `results_v3/commit_predictions/`.
    """
    import glob as _glob
    out: Dict[str, Dict[str, float]] = {}
    pred_root = os.path.join(base_dir, "results_v3", "commit_predictions")

    def _censoring_for_npz(npz_path: str, pred_npz_glob: Optional[str] = None, w_size: int = 400) -> Dict[str, Any]:
        b = np.load(npz_path)
        ts = b["commit_ts"].astype(np.float64)
        y = b["y_true"].astype(np.int32)
        fts = b["fix_ts"].astype(np.float64)
        s0 = int(b["n_init"]) + int(b["n_val"])
        c1_rates, c4_rates = [], []
        tb1_sum, rb1_sum, tb4_sum, rb4_sum = 0, 0, 0, 0
        for end in range(s0 + w_size, len(y) + 1, w_size):
            t_al = float(ts[end - 1])
            _, y_obs = observe_labels_at_time(ts[:end], y[:end], fts[:end], t_al)
            tb1 = int(np.sum(y[end - w_size : end] == 1))
            if tb1 > 0:
                rev1 = int(np.sum((y[end - w_size : end] == 1) & (y_obs[end - w_size : end] > 0.5)))
                c1_rates.append(1.0 - rev1 / tb1)
                tb1_sum += tb1
                rb1_sum += rev1
            s4 = max(0, end - 4 * w_size)
            tb4 = int(np.sum(y[s4:end] == 1))
            if tb4 > 0:
                rev4 = int(np.sum((y[s4:end] == 1) & (y_obs[s4:end] > 0.5)))
                c4_rates.append(1.0 - rev4 / tb4)
                tb4_sum += tb4
                rb4_sum += rev4
        fin = (y == 1) & np.isfinite(fts) & (fts >= ts)
        med_lat = float(np.median((fts[fin] - ts[fin]) / SECONDS_PER_DAY)) if np.any(fin) else 0.0

        n_fired = 0
        al_c1, al_c4 = [], []
        if pred_npz_glob:
            matches = sorted(_glob.glob(pred_npz_glob))
            if matches:
                p_arr = np.load(matches[0])
                gen = p_arr["shared_model_gen_id"]
                fired_ends = [s0 + k for k in range(w_size, len(gen), w_size) if gen[k] > gen[k - 1]]
                n_fired = len(fired_ends)
                for end in fired_ends:
                    t_al = float(ts[end - 1])
                    _, yo = observe_labels_at_time(ts[:end], y[:end], fts[:end], t_al)
                    tb1 = int(np.sum(y[end - w_size : end] == 1))
                    if tb1 > 0:
                        rb1 = int(np.sum((y[end - w_size : end] == 1) & (yo[end - w_size : end] > 0.5)))
                        al_c1.append(1.0 - rb1 / tb1)
                    s4 = max(0, end - 4 * w_size)
                    tb4 = int(np.sum(y[s4:end] == 1))
                    if tb4 > 0:
                        rb4 = int(np.sum((y[s4:end] == 1) & (yo[s4:end] > 0.5)))
                        al_c4.append(1.0 - rb4 / tb4)

        return {
            "N": int(len(y)),
            "prevalence": float(np.mean(y)),
            "n_scheduled_windows": int(len(c1_rates)),
            "cens_1w": float(np.mean(c1_rates)) if c1_rates else 0.0,
            "cens_4w": float(np.mean(c4_rates)) if c4_rates else 0.0,
            "pooled_cens_1w": float(1.0 - rb1_sum / max(tb1_sum, 1)),
            "pooled_cens_4w": float(1.0 - rb4_sum / max(tb4_sum, 1)),
            "pooled_bugs_1w": int(tb1_sum),
            "median_latency_days": med_lat,
            "seed0_fired_alarms": int(n_fired),
            "alarm_cens_1w": float(np.mean(al_c1)) if al_c1 else None,
            "alarm_cens_4w": float(np.mean(al_c4)) if al_c4 else None,
        }

    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        man_mm = json.load(f)
    for st_mm in man_mm:
        sn = st_mm["name"]
        p_glob = os.path.join(pred_root, "benchmark_a", f"{sn}__lightgbm__none__seed*.npz")
        out[sn] = _censoring_for_npz(os.path.join(mm_dir, f"{sn}_tab.npz"), p_glob)

    b_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    with open(os.path.join(b_dir, "manifest_b_v3.json"), "r", encoding="utf-8") as f:
        man_b = json.load(f)
    by_coh: Dict[str, List[Dict[str, Any]]] = {}
    for st in man_b["natural_streams"]:
        sn = st["name"]
        p_glob = os.path.join(pred_root, "benchmark_b", f"{sn}__mlp*.npz")
        by_coh.setdefault(st["cohort"], []).append(_censoring_for_npz(os.path.join(b_dir, f"{sn}.npz"), p_glob))
    for coh, rows in by_coh.items():
        al1 = [r["alarm_cens_1w"] for r in rows if r["alarm_cens_1w"] is not None]
        al4 = [r["alarm_cens_4w"] for r in rows if r["alarm_cens_4w"] is not None]
        out[f"cohort_{coh}"] = {
            "total_commits": int(sum(r["N"] for r in rows)),
            "mean_prevalence": float(np.mean([r["prevalence"] for r in rows])),
            "n_scheduled_windows_total": int(sum(r["n_scheduled_windows"] for r in rows)),
            "cens_1w": float(np.mean([r["cens_1w"] for r in rows])),
            "cens_4w": float(np.mean([r["cens_4w"] for r in rows])),
            "median_latency_days": float(np.median([r["median_latency_days"] for r in rows])),
            "seed0_fired_alarms_total": int(sum(r["seed0_fired_alarms"] for r in rows)),
            "alarm_cens_1w": float(np.mean(al1)) if al1 else 0.0,
            "alarm_cens_4w": float(np.mean(al4)) if al4 else 0.0,
        }

    c_dir = os.path.join(base_dir, "cache", "confirmatory_v3")
    with open(os.path.join(c_dir, "manifest_confirmatory_v3.json"), "r", encoding="utf-8") as f:
        man_c = json.load(f)
    c_rows = [
        _censoring_for_npz(
            os.path.join(c_dir, f"{st['name']}.npz"),
            os.path.join(pred_root, "confirmatory", f"{st['name']}__mlp*.npz"),
        )
        for st in man_c["confirmatory_streams"]
    ]
    al1_c = [r["alarm_cens_1w"] for r in c_rows if r["alarm_cens_1w"] is not None]
    al4_c = [r["alarm_cens_4w"] for r in c_rows if r["alarm_cens_4w"] is not None]
    out["cohort_Confirmatory_ApacheJIT"] = {
        "total_commits": int(sum(r["N"] for r in c_rows)),
        "mean_prevalence": float(np.mean([r["prevalence"] for r in c_rows])),
        "n_scheduled_windows_total": int(sum(r["n_scheduled_windows"] for r in c_rows)),
        "cens_1w": float(np.mean([r["cens_1w"] for r in c_rows])),
        "cens_4w": float(np.mean([r["cens_4w"] for r in c_rows])),
        "median_latency_days": float(np.median([r["median_latency_days"] for r in c_rows])),
        "seed0_fired_alarms_total": int(sum(r["seed0_fired_alarms"] for r in c_rows)),
        "alarm_cens_1w": float(np.mean(al1_c)) if al1_c else 0.0,
        "alarm_cens_4w": float(np.mean(al4_c)) if al4_c else 0.0,
    }
    standalone_names = {"conf_apache_spark", "conf_apache_zeppelin", "conf_apache_zookeeper"}
    s3_rows = [
        r for st, r in zip(man_c["confirmatory_streams"], c_rows)
        if st["name"] in standalone_names
    ]
    al1_s3 = [r["alarm_cens_1w"] for r in s3_rows if r["alarm_cens_1w"] is not None]
    al4_s3 = [r["alarm_cens_4w"] for r in s3_rows if r["alarm_cens_4w"] is not None]
    out["cohort_Confirmatory_Standalone3"] = {
        "total_commits": int(sum(r["N"] for r in s3_rows)),
        "mean_prevalence": float(np.mean([r["prevalence"] for r in s3_rows])),
        "n_scheduled_windows_total": int(sum(r["n_scheduled_windows"] for r in s3_rows)),
        "cens_1w": float(np.mean([r["cens_1w"] for r in s3_rows])),
        "cens_4w": float(np.mean([r["cens_4w"] for r in s3_rows])),
        "median_latency_days": float(np.median([r["median_latency_days"] for r in s3_rows])),
        "seed0_fired_alarms_total": int(sum(r["seed0_fired_alarms"] for r in s3_rows)),
        "alarm_cens_1w": float(np.mean(al1_s3)) if al1_s3 else 0.0,
        "alarm_cens_4w": float(np.mean(al4_s3)) if al4_s3 else 0.0,
    }
    return out


def build_derived_aggregates(
    base_dir: str,
    audit_j: Dict,
    ba_j: Dict,
    bb_j: Dict,
    conf_j: Dict,
) -> Dict:
    """
    Computes exact, deterministic cross-stream/cross-host means for Benchmark A & runtime summaries
    and stamps them with `build_provenance_header` so every aggregate number in manuscript tables
    is 100% reproducible and verified by `verify_manuscript_v3.py`.
    """
    cens = compute_empirical_window_censoring_stats(base_dir)
    chosen_skip = ba_j["dev_skip_selection"]["chosen_skip_mode"]
    with open(os.path.join(base_dir, "cache", "multimodal_v3_pure13", "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        man_mm = json.load(f)
    ba_streams = [x["name"] for x in man_mm]
    ba_nat = [x["name"] for x in man_mm if x["stream_kind"] == "natural_ecosystem"]
    ba_shf = [x["name"] for x in man_mm if x["stream_kind"] != "natural_ecosystem"]

    host_configs = [
        ("att_cnn_skip", "att_cnn", chosen_skip),
        ("codebert_skip", "codebert", chosen_skip),
        ("deepjit_cnn_skip", "deepjit_cnn", chosen_skip),
        ("lightgbm_none", "lightgbm", "none"),
        ("att_cnn_none", "att_cnn", "none"),
        ("codebert_none", "codebert", "none"),
        ("deepjit_cnn_none", "deepjit_cnn", "none"),
    ]
    methods_ba = [
        "B0_Frozen", "B1a_Full_Hindsight", "B1b_1W_Oracle", "B1b_4W_Oracle",
        "B2_1W", "B2_4W", "Wait_90", "Pure_nnPU", "FixedCurve_PU",
        "Teacher_Only_Exact", "Teacher_Only_Matched", "Survival_PU", "Survival_PU_WithGate",
    ]
    metrics_ba = ["ap", "roc_auc", "f1", "f1_th", "brier_cal", "recall_at_20pct_effort"]

    by_host_6stream: Dict[str, Dict] = {}
    by_host_nat2: Dict[str, Dict] = {}
    by_host_shift4_post: Dict[str, Dict] = {}

    for h_key, h_type, skm in host_configs:
        by_host_6stream[h_key] = {}
        by_host_nat2[h_key] = {}
        by_host_shift4_post[h_key] = {}
        for m in methods_ba:
            by_host_6stream[h_key][m] = {
                met: float(np.mean([ba_j["cells"][f"{s}__{h_type}__{skm}"]["methods"][m]["full_stream"][met] for s in ba_streams]))
                for met in metrics_ba
            }
            by_host_nat2[h_key][m] = {
                met: float(np.mean([ba_j["cells"][f"{s}__{h_type}__{skm}"]["methods"][m]["full_stream"][met] for s in ba_nat]))
                for met in metrics_ba
            }
            by_host_shift4_post[h_key][m] = {
                met: float(np.mean([ba_j["cells"][f"{s}__{h_type}__{skm}"]["methods"][m]["post_onset"][met] for s in ba_shf]))
                for met in metrics_ba
            }

    # 4-Backbone primary mean across all 6 Benchmark A streams (`att_cnn_skip`, `codebert_skip`, `deepjit_cnn_skip`, `lightgbm_none`)
    primary_4hosts = ["att_cnn_skip", "codebert_skip", "deepjit_cnn_skip", "lightgbm_none"]
    mean_4hosts_6streams = {
        m: {
            met: float(np.mean([by_host_6stream[hk][m][met] for hk in primary_4hosts]))
            for met in metrics_ba
        }
        for m in methods_ba
    }

    # Track 2 simple & online stream baselines across all 6 Benchmark A streams (stored in `lightgbm__none` cells)
    track2_ba_methods = [
        "LOC_Only", "LApredict_B0", "LApredict_Adapt", "Churn_LR_B0", "Churn_LR_Adapt",
        "Full_LR_B0", "Full_LR_Adapt", "BORB_Native", "BORB_Alarm", "ORB_OHT_Native", "ODaSC_OHT_Native",
    ]
    track2_ba_means = {
        m: {
            met: float(np.mean([ba_j["cells"][f"{s}__lightgbm__none"]["methods"][m]["full_stream"][met] for s in ba_streams]))
            for met in metrics_ba
        }
        for m in track2_ba_methods
    }

    #Controlled shift 4-stream mean on Benchmark B
    shf_b_names = list(bb_j["controlled_shift_streams_mlp"].keys())
    shf_b_methods = list(bb_j["controlled_shift_streams_mlp"][shf_b_names[0]]["methods"].keys())
    shift_b_means = {
        m: {
            "full_ap": float(np.mean([bb_j["controlled_shift_streams_mlp"][s]["methods"][m]["full_stream"]["ap"] for s in shf_b_names])),
            "post_onset_ap": float(np.mean([bb_j["controlled_shift_streams_mlp"][s]["methods"][m]["post_onset"]["ap"] for s in shf_b_names])),
            "transient_1200_ap": float(np.mean([bb_j["controlled_shift_streams_mlp"][s]["methods"][m]["transient_1200"]["ap"] for s in shf_b_names])),
            "post_onset_r20": float(np.mean([bb_j["controlled_shift_streams_mlp"][s]["methods"][m]["post_onset"]["recall_at_20pct_effort"] for s in shf_b_names])),
        }
        for m in shf_b_methods
    }

    # Phase-decomposed runtime telemetry across Benchmark B natural 27 streams (`mlp`) and Qt (`codebert` + `lightgbm`)
    nat_b_names = list(bb_j["natural_streams_mlp"].keys())
    tel_keys = [
        "own_init_fit_ms", "shared_anchor_init_ms", "shared_monitor_init_ms",
        "stream_predict_and_embed_ms", "drift_monitor_ms", "adapt_only_wall_ms",
        "calibration_ms", "consistent_end_to_end_ms", "incremental_suite_end_to_end_ms",
        "alarm_step_total_ms_p50", "alarm_step_total_ms_p95", "opt_steps", "n_alarms", "alarm_rate",
    ]
    runtime_b_mlp = {}
    for m in bb_j["natural_streams_mlp"][nat_b_names[0]]["methods"]:
        runtime_b_mlp[m] = {
            tk: float(np.mean([bb_j["natural_streams_mlp"][s]["methods"][m]["telemetry"][tk] for s in nat_b_names]))
            for tk in tel_keys
        }

    runtime_qt_codebert = {
        m: {
            tk: float(ba_j["cells"][f"qt_natural_25k__codebert__{chosen_skip}"]["methods"][m]["telemetry"][tk])
            for tk in tel_keys
        }
        for m in ("B0_Frozen", "B2_4W", "Wait_90", "Pure_nnPU", "Teacher_Only_Matched", "Survival_PU")
    }
    runtime_qt_lgb = {
        m: {
            tk: float(ba_j["cells"]["qt_natural_25k__lightgbm__none"]["methods"][m]["telemetry"][tk])
            for tk in tel_keys
        }
        for m in ("B0_Frozen", "B2_4W", "Wait_90", "Pure_nnPU", "Teacher_Only_Matched", "Survival_PU", "BORB_Native", "ORB_OHT_Native", "ODaSC_OHT_Native")
    }

    # Dataset totals
    n_commits_ba_nat = int(audit_j["benchmark_a"]["qt_audit"]["total_commits"]) + int(audit_j["benchmark_a"]["openstack_audit"]["total_commits"])
    with open(os.path.join(base_dir, "cache", "benchmark_b_v3", "manifest_b_v3.json"), "r", encoding="utf-8") as f:
        mb = json.load(f)
    with open(os.path.join(base_dir, "cache", "confirmatory_v3", "manifest_confirmatory_v3.json"), "r", encoding="utf-8") as f:
        mc = json.load(f)
    n_commits_bb_nat = int(sum(x["N"] for x in mb["natural_streams"]))
    n_commits_conf = int(sum(x["N"] for x in mc["confirmatory_streams"]))

    derived = {
        "provenance": build_provenance_header(
            base_dir,
            [
                os.path.join(base_dir, "results_v3", "benchmark_a_multihost_v3.json"),
                os.path.join(base_dir, "results_v3", "benchmark_b_main_v3.json"),
                os.path.join(base_dir, "results_v3", "confirmatory_cohort_v3.json"),
            ],
        ),
        "corpus_totals": {
            "benchmark_a_natural_commits": n_commits_ba_nat,
            "benchmark_b_natural_27streams_commits": n_commits_bb_nat,
            "confirmatory_5repos_commits": n_commits_conf,
            "total_natural_repository_commits": n_commits_ba_nat + n_commits_bb_nat + n_commits_conf,
        },
        "window_censoring_audit": cens,
        "benchmark_a_by_host_6streams": by_host_6stream,
        "benchmark_a_by_host_natural2": by_host_nat2,
        "benchmark_a_by_host_shift4_post_onset": by_host_shift4_post,
        "benchmark_a_4host_6stream_mean": mean_4hosts_6streams,
        "benchmark_a_track2_6stream_mean": track2_ba_means,
        "benchmark_b_controlled_shift_4stream_mean": shift_b_means,
        "runtime_summary": {
            "benchmark_b_mlp_27streams_mean": runtime_b_mlp,
            "qt_codebert_skip": runtime_qt_codebert,
            "qt_lightgbm": runtime_qt_lgb,
        },
    }
    save_json_atomic(os.path.join(base_dir, "results_v3", "manuscript_derived_aggregates_v3.json"), derived)
    return derived


# ==============================================================================
# Publication-Grade Figures (Fig 1 .. Fig 8)
# ==============================================================================

def _save_fig(fig, fig_dir: str, stem: str, pad_inches: float = 0.16) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"{stem}.pdf"), bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(os.path.join(fig_dir, f"{stem}.png"), dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def plot_fig1_multimodal_audit(fig_dir: str, ba_j: Dict, derived: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.14, 1.32]})
    cens = derived["window_censoring_audit"]

    # Panel (a): Right-censoring rates inside 1W vs 4W across cohorts
    cohorts = [
        ("qt_natural_25k", "Qt\n(25.7k)"),
        ("openstack_natural_13k", "OpenStack\n(13.3k)"),
        ("cohort_Cabral19", "Cabral-14\n(ICSE'19)"),
        ("cohort_ApacheJIT", "Apache-8\n(Dev)"),
        ("cohort_N60_PRSZZ", "PR-SZZ-5\n(SQJ'26)"),
        ("cohort_Confirmatory_ApacheJIT", "Conf.-5*\n(All)"),
        ("cohort_Confirmatory_Standalone3", "Stand.-3\n(Unsplit)"),
    ]
    x = np.arange(len(cohorts))
    w = 0.36
    v1 = [100.0 * cens[k]["cens_1w"] for k, _ in cohorts]
    v4 = [100.0 * cens[k]["cens_4w"] for k, _ in cohorts]
    b1 = ax1.bar(x - w / 2, v1, width=w, color="#DC2626", alpha=0.88, label="Latest Drift Window (1W = 400)")
    b4 = ax1.bar(x + w / 2, v4, width=w, color="#F59E0B", alpha=0.88, label="Escalated Replay (4W = 1,600)")
    for bars in (b1, b4):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.4, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in cohorts], fontsize=7.1)
    ax1.set_ylabel("Window-Averaged Unrevealed Defects (%)")
    ax1.set_ylim(0, 106)
    ax1.set_title("(a) Window-Averaged Censoring at Scheduled 400c Boundaries", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): Benchmark A 6-stream mean AP across all 4 backbones + Full-LR Frozen reference
    hosts = [
        ("att_cnn_skip", "Att-CNN\n(+LinSkip)"),
        ("codebert_skip", "CodeBERT\n(+LinSkip)"),
        ("deepjit_cnn_skip", "DeepJIT-CNN\n(+LinSkip)"),
        ("lightgbm_none", "LightGBM\n(SVD+Tab13)"),
    ]
    m_list = [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"]),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
        ("Wait_90", "Wait-90", PALETTE["wait"]),
        ("Pure_nnPU", "Pure-nnPU", PALETTE["nnpu"]),
        ("Teacher_Only_Matched", "Teacher-Matched", PALETTE["teacher"]),
        ("Survival_PU", "Survival-PU", PALETTE["spu"]),
    ]
    xh = np.arange(len(hosts))
    bw = 0.112
    offs = (np.arange(len(m_list)) - (len(m_list) - 1) / 2.0) * bw
    ymax = 0.0
    for idx_m, (mk, mlbl, col) in enumerate(m_list):
        vals = [derived["benchmark_a_by_host_6streams"][hk][mk]["ap"] for hk, _ in hosts]
        ymax = max(ymax, max(vals))
        ax2.bar(xh + offs[idx_m], vals, width=bw * 0.94, color=col, label=mlbl, edgecolor="white", linewidth=0.4)
    flr_ap = derived["benchmark_a_track2_6stream_mean"]["Full_LR_B0"]["ap"]
    ymax = max(ymax, flr_ap)
    ax2.axhline(flr_ap, color="#0F172A", linestyle="--", lw=1.35, label=f"Full-LR Frozen (Tab-13: {flr_ap:.4f})")
    ax2.set_xticks(xh)
    ax2.set_xticklabels([hl for _, hl in hosts], fontsize=8.4)
    ax2.set_ylabel("Primary Endpoint: Average Precision (AP)")
    ax2.set_ylim(0, ymax * 1.36)
    ax2.set_title("(b) Benchmark A Multi-Host & Linear Baseline AP (6-Stream Mean)", fontweight="bold")
    ax2.legend(ncol=2, loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.3)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig1_multimodal_audit_v3")


def plot_fig2_trajectories_and_skip(fig_dir: str, ba_j: Dict) -> None:
    chosen_skip = ba_j["dev_skip_selection"]["chosen_skip_mode"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Panel (a): Prequential AP trajectory on Qt Natural (CodeBERT + skip)
    qt_cell = ba_j["cells"][f"qt_natural_25k__codebert__{chosen_skip}"]
    traj = qt_cell["seed0_window_trajectories"]
    for mk, lbl, col, ls, lw in [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"], "--", 1.5),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"], "-.", 1.5),
        ("B2_4W", "Causal-4W", PALETTE["b2"], "-", 1.8),
        ("Wait_90", "Wait-90", PALETTE["wait"], ":", 1.6),
        ("Survival_PU", "Survival-PU", PALETTE["spu"], "-", 2.1),
    ]:
        rows = traj.get(mk, {}).get("windows", [])
        if not rows:
            continue
        # Smooth with 3-window rolling mean over valid two-class windows for clean legibility
        valid = [(int(r["window_idx"]), float(r["ap"])) for r in rows if r.get("ap") is not None and np.isfinite(r["ap"])]
        if not valid:
            continue
        wx = np.array([v[0] for v in valid])
        wy = np.array([v[1] for v in valid])
        if len(wy) >= 5:
            kernel = np.ones(3) / 3.0
            wy_plot = np.convolve(wy, kernel, mode="same")
            wy_plot[0], wy_plot[-1] = wy[0], wy[-1]
        else:
            wy_plot = wy
        ax1.plot(wx, wy_plot, label=lbl, color=col, linestyle=ls, linewidth=lw)

    ax1.set_xlabel("Streaming Evaluation Window Index ($W = 400$ Commits)")
    ax1.set_ylabel("3-Window Rolling Average Precision (AP)")
    ax1.set_title("(a) Qt Longitudinal Trajectory (CodeBERT + Frozen Linear Skip)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.9)
    ax1.grid(linestyle=":", alpha=0.5)
    ymin_cur, ymax_cur = ax1.get_ylim()
    ax1.set_ylim(max(0.0, ymin_cur - 0.02), min(1.0, ymax_cur * 1.18))

    # Panel (b): Architectural Linear-Residual Skip Effect (skip_mode='none' vs chosen_skip) on Natural Qt + OpenStack
    backbones = [("att_cnn", "Att-CNN"), ("codebert", "CodeBERT"), ("deepjit_cnn", "DeepJIT-CNN")]
    xb = np.arange(len(backbones))
    bw = 0.19
    series = [
        ("B0_Frozen", "none", "Frozen (No Skip)", "#94A3B8"),
        ("B0_Frozen", chosen_skip, "Frozen (+LinSkip)", PALETTE["b0"]),
        ("Survival_PU", "none", "Survival-PU (No Skip)", "#6EE7B7"),
        ("Survival_PU", chosen_skip, "Survival-PU (+LinSkip)", PALETTE["spu"]),
    ]
    ymax2 = 0.0
    for idx_s, (mk, sm, slbl, col) in enumerate(series):
        vals = [
            0.5 * (
                ba_j["cells"][f"qt_natural_25k__{ht}__{sm}"]["methods"][mk]["full_stream"]["ap"]
                + ba_j["cells"][f"openstack_natural_13k__{ht}__{sm}"]["methods"][mk]["full_stream"]["ap"]
            )
            for ht, _ in backbones
        ]
        ymax2 = max(ymax2, max(vals))
        offs = (idx_s - 1.5) * bw
        bars = ax2.bar(xb + offs, vals, width=bw * 0.92, color=col, label=slbl, edgecolor="#1E293B", linewidth=0.4)
        for b in bars:
            hv = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, hv + 0.008, f"{hv:.2f}", ha="center", va="bottom", fontsize=7.2)

    ax2.set_xticks(xb)
    ax2.set_xticklabels([bl for _, bl in backbones])
    ax2.set_ylabel("Mean AP on Natural Qt & OpenStack")
    ax2.set_ylim(0, ymax2 * 1.24)
    ax2.set_title("(b) Effect of Causal Tabular Linear-Residual Skip", fontweight="bold")
    ax2.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=7.7)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig2_trajectories_and_skip_v3")


def plot_fig3_benchmark_b_clusters(fig_dir: str, bb_j: Dict, derived: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # Panel (a): K=25 natural donor clusters mean AP (MLP Track 1 + Track 2 competitors)
    means25 = bb_j["cluster_inference_mlp"]["primary_ap"]["K25_cluster_means"]
    vecs25 = bb_j["cluster_inference_mlp"]["primary_ap"]["K25_per_cluster_vectors"]
    order_m = [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"]),
        ("Survival_PU", "Survival-PU", PALETTE["spu"]),
        ("Teacher_Only_Matched", "Teacher-Matched", PALETTE["teacher"]),
        ("FixedCurve_PU", "FixedCurve-PU", PALETTE["fixed"]),
        ("BORB_Native", "BORB (ElasticNet)", PALETTE["track2"]),
        ("ODaSC_OHT_Native", "ODaSC (Hoeffding)", "#D97706"),
        ("ORB_OHT_Native", "ORB (Hoeffding)", "#EA580C"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
        ("Full_LR_Adapt", "Full-LR (Adapt-4W)", "#64748B"),
        ("Wait_90", "Wait-90", PALETTE["wait"]),
        ("Pure_nnPU", "Pure-nnPU", PALETTE["nnpu"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
        ("LOC_Only", "LOC-Only", "#94A3B8"),
    ]
    # Sort descending by K=25 mean AP (keeping Oracle at top if highest)
    order_sorted = sorted(order_m, key=lambda t: means25[t[0]], reverse=True)
    y_pos = np.arange(len(order_sorted))
    vals = [means25[k] for k, _, _ in order_sorted]
    sems = [float(np.std(vecs25[k], ddof=1) / np.sqrt(len(vecs25[k]))) for k, _, _ in order_sorted]
    cols = [c for _, _, c in order_sorted]
    lbls = [l for _, l, _ in order_sorted]

    bars = ax1.barh(y_pos, vals, xerr=sems, height=0.68, color=cols, alpha=0.90, capsize=2.5, error_kw={"lw": 1.0, "ecolor": "#1E293B"})
    ax1.invert_yaxis()
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(lbls, fontsize=8.2)
    for b, v in zip(bars, vals):
        ax1.text(v + max(sems) + 0.008, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", ha="left", fontsize=7.8)
    ax1.set_xlim(0, max(vals) * 1.22)
    ax1.set_xlabel("Mean Average Precision Across $K=25$ Independent Donor Clusters ($\\pm 1$ SEM)")
    ax1.set_title("(a) Powered Cross-Project Benchmark B ($K=25$ Natural Clusters)", fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel (b): Strictly Separated Controlled Shift Suite (4 Streams Post-Onset & First-1200 AP)
    sh_means = derived["benchmark_b_controlled_shift_4stream_mean"]
    sh_methods = [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"]),
        ("Survival_PU", "Survival-PU", PALETTE["spu"]),
        ("ODaSC_Host", "ODaSC-Host", "#10B981"),
        ("FixedCurve_PU", "FixedCurve-PU", PALETTE["fixed"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
        ("Wait_90", "Wait-90", PALETTE["wait"]),
        ("BORB_Alarm", "BORB-Alarm", "#C2410C"),
        ("BORB_Native", "BORB-Native", PALETTE["track2"]),
        ("ODaSC_OHT_Native", "ODaSC", "#F59E0B"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
    ]
    xs = np.arange(len(sh_methods))
    v_post = [sh_means[mk]["post_onset_ap"] for mk, _, _ in sh_methods]
    v_tr12 = [sh_means[mk]["transient_1200_ap"] for mk, _, _ in sh_methods]
    bw2 = 0.38
    ax2.bar(xs - bw2 / 2, v_post, width=bw2, color=[c for _, _, c in sh_methods], alpha=0.92, label="Full Post-Onset Horizon")
    ax2.bar(xs + bw2 / 2, v_tr12, width=bw2, color=[c for _, _, c in sh_methods], alpha=0.48, edgecolor="#1E293B", hatch="//", label="First 1,200 Post-Onset Commits")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([l for _, l, _ in sh_methods], rotation=32, ha="right", fontsize=8.0)
    ax2.set_ylabel("Post-Shift Average Precision (AP)")
    ax2.set_ylim(0, max(max(v_post), max(v_tr12)) * 1.22)
    ax2.set_title("(b) Controlled Abrupt Shifts (4 Streams, Separated)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig3_benchmark_b_clusters_v3")


def plot_fig4_confirmatory_boundary(fig_dir: str, bb_j: Dict, conf_j: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Panel (a): Per-repository AP on the 5 untouched confirmatory ApacheJIT repositories
    repos = [
        ("conf_apache_hdfs", "HDFS*\n(N=2,850, 77%)"),
        ("conf_apache_mapreduce", "MapReduce*\n(N=1,318, 64%)"),
        ("conf_apache_spark", "Spark\n(N=1,273, 47%)"),
        ("conf_apache_zeppelin", "Zeppelin\n(N=1,350, 45%)"),
        ("conf_apache_zookeeper", "ZooKeeper\n(N=686, 49%)"),
    ]
    m_conf = [
        ("ODaSC_OHT_Native", "ODaSC (Hoeffding)", "#D97706"),
        ("BORB_Native", "BORB (ElasticNet)", PALETTE["track2"]),
        ("ORB_OHT_Native", "ORB (Hoeffding)", "#FB923C"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"]),
        ("Survival_PU", "Survival-PU", PALETTE["spu"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
    ]
    xr = np.arange(len(repos))
    bw = 0.112
    offs = (np.arange(len(m_conf)) - (len(m_conf) - 1) / 2.0) * bw
    for idx_m, (mk, mlbl, col) in enumerate(m_conf):
        vals = [conf_j["streams_mlp"][rk]["methods"][mk]["full_stream"]["ap"] for rk, _ in repos]
        ax1.bar(xr + offs[idx_m], vals, width=bw * 0.93, color=col, label=mlbl)
    ax1.set_xticks(xr)
    ax1.set_xticklabels([rl for _, rl in repos], fontsize=8.0)
    ax1.set_ylabel("Average Precision (AP)")
    ax1.set_ylim(0.0, 1.04)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_title("(a) Confirmatory Cohort (*Hadoop Monorepo Clean Commits in apache/hadoop)", fontweight="bold", fontsize=9.3, pad=38)
    ax1.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=True, facecolor="white", framealpha=0.95, fontsize=7.2, columnspacing=0.9, handlelength=1.2)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): Primary Hindsight Optimism Gap (B1b* - B2-4W) vs Evaluation Horizon N_eval
    nat_x, nat_y = [], []
    for s, d in bb_j["natural_streams_mlp"].items():
        nat_x.append(int(d["stream_eval_N"]))
        nat_y.append(float(d["methods"]["B1b_4W_Oracle"]["full_stream"]["ap"] - d["methods"]["B2_4W"]["full_stream"]["ap"]))
    conf_st_x, conf_st_y = [], []
    conf_mo_x, conf_mo_y = [], []
    for s, d in conf_j["streams_mlp"].items():
        xv = int(d["stream_eval_N"])
        yv = float(d["methods"]["B1b_4W_Oracle"]["full_stream"]["ap"] - d["methods"]["B2_4W"]["full_stream"]["ap"])
        if s in ("conf_apache_hdfs", "conf_apache_mapreduce"):
            conf_mo_x.append(xv)
            conf_mo_y.append(yv)
        else:
            conf_st_x.append(xv)
            conf_st_y.append(yv)

    ax2.axhline(0.0, color="#334155", linestyle="--", lw=1.1)
    ax2.scatter(nat_x, nat_y, s=46, color="#2563EB", alpha=0.84, label="Bench. B Natural (27 Streams; 22/27 > 0)")
    ax2.scatter(conf_st_x, conf_st_y, s=72, marker="D", color="#DC2626", alpha=0.94, label="Confirmatory Standalone (3 Repos; 0/3 > 0)")
    ax2.scatter(conf_mo_x, conf_mo_y, s=76, marker="s", facecolors="none", edgecolors="#DC2626", linewidths=1.8, label="Confirmatory Hadoop Subproj.* (2; 0/2 > 0)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Streaming Evaluation Horizon $N_{\\mathrm{eval}}$ (Commits, Log Scale)")
    ax2.set_ylabel("Hindsight Optimism Gap $\\Delta\\mathrm{AP}$ (Oracle $-$ Causal-4W)")
    ax2.set_title("(b) Horizon vs. Matched Oracle–Causal Gap (MLP)", fontweight="bold", fontsize=9.5, pad=10)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)
    ax2.grid(linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig4_confirmatory_boundary_v3")


def plot_fig5_ablations_and_controls(fig_dir: str, rq3_j: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.88), gridspec_kw={"width_ratios": [1.10, 1.08]})
    sm = rq3_j["summary_means"]

    # Panel (a): Survival estimator sensitivity & internal component ablations (AP)
    ab_rows = [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"]),
        ("Survival_PU_Default", "Survival-PU (Cohort Incidence)", PALETTE["spu"]),
        ("SPU_KaplanMeier", "Survival-PU (Kaplan–Meier)", "#10B981"),
        ("SPU_WeibullCure", "Survival-PU (Weibull Cure)", "#34D399"),
        ("SPU_Fixed90d", "FixedCurve-PU (Zero-Fit 90d)", PALETTE["fixed"]),
        ("SPU_NoCovariateWeight", "w/o Domain Density Ratio", "#60A5FA"),
        ("SPU_NoShrinkage_m0_0", "w/o Anchor Shrinkage (m0=0)", "#818CF8"),
        ("SPU_FrozenAnchorTeacher", "Frozen Anchor Teacher", "#A78BFA"),
        ("SPU_NoBisection", "w/o Prevalence Bisection", "#F472B6"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
    ]
    yp = np.arange(len(ab_rows))
    vals = [sm[k]["ap"] for k, _, _ in ab_rows]
    bars = ax1.barh(yp, vals, height=0.68, color=[c for _, _, c in ab_rows], alpha=0.90)
    ax1.invert_yaxis()
    ax1.set_yticks(yp)
    ax1.set_yticklabels([l for _, l, _ in ab_rows], fontsize=9.8)
    for b, v in zip(bars, vals):
        ax1.text(v + 0.007, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", ha="left", fontsize=9.4)
    ax1.set_xlim(0, max(vals) * 1.24)
    ax1.tick_params(axis="x", labelsize=9.6)
    ax1.set_xlabel("10-Stream Mean Average Precision (AP)", fontsize=10.6)
    ax1.set_title("(a) Maturity-Estimator Sensitivity & Internal Ablations", fontweight="bold", fontsize=11.0, pad=12)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel (b): Dual-metric comparison (AP AND Recall@20%Effort) on Compute-Matched Teacher & Gate Controls
    ctrl_rows = [
        ("Survival_PU_Default", "Survival-PU\n(3-Stage)"),
        ("Teacher_Only_Matched", "Teacher-\nMatched"),
        ("Teacher_Only_Exact", "Teacher-\nExact"),
        ("SPU_WithLegacyGate", "SPU\n(+Gate)"),
        ("SPU_Fixed90d", "FixedCurve\n(90d)"),
    ]
    xc = np.arange(len(ctrl_rows))
    bw = 0.39
    v_ap = [sm[k]["ap"] for k, _ in ctrl_rows]
    v_r20 = [sm[k]["recall_at_20pct_effort"] for k, _ in ctrl_rows]
    b_ap = ax2.bar(xc - bw / 2, v_ap, width=bw, color=PALETTE["spu"], alpha=0.90, label="Average Precision (AP)")
    b_r2 = ax2.bar(xc + bw / 2, v_r20, width=bw, color="#2563EB", alpha=0.85, label="Recall@20%Effort")
    for bars_g in (b_ap, b_r2):
        for b in bars_g:
            hv = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, hv + 0.012, f"{hv:.3f}", ha="center", va="bottom", fontsize=8.8)
    ax2.set_xticks(xc)
    ax2.set_xticklabels([l for _, l in ctrl_rows], fontsize=9.4)
    ax2.tick_params(axis="y", labelsize=9.6)
    ax2.set_ylim(0, max(max(v_ap), max(v_r20)) * 1.28)
    ax2.set_ylabel("Score (Higher is Better)", fontsize=10.6)
    ax2.set_title("(b) Distillation, Compute-Matched & Gate Controls", fontweight="bold", fontsize=11.0, pad=12)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=9.4)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=1.8, rect=[0.0, 0.0, 1.0, 0.93])
    _save_fig(fig, fig_dir, "fig5_ablations_and_controls_v3", pad_inches=0.28)


def plot_fig6_latency_and_drift_factorial(fig_dir: str, rq4a_j: Dict, rq4b_j: Dict, rq4a_fc_j: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.08, 1.15]})

    # Panel (a): RQ4a Future-Only Causal Latency Intervention (Frozen Pre-t0 State)
    kappas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    k_keys = [f"{k:.2f}" for k in kappas]
    by_k = rq4a_j["results_by_kappa"]
    by_k_fc = rq4a_fc_j["results_by_kappa"]

    for mk, lbl, col, ls, mk_sym, src_dict in [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"], "--", "s", by_k),
        ("B2_4W", "Causal-4W", PALETTE["b2"], "-", "v", by_k),
        ("FixedCurve_PU", "FixedCurve-PU (Zero-Fit 90d)", PALETTE["fixed"], "-.", "P", by_k_fc),
        ("Survival_PU", "Survival-PU (Cohort Inc.)", PALETTE["spu"], "-", "o", by_k),
        ("Wait_90", "Wait-90", PALETTE["wait"], ":", "^", by_k),
        ("Pure_nnPU", "Pure-nnPU", PALETTE["nnpu"], ":", "d", by_k),
        ("B0_Frozen", "Frozen Host (Pre-t0 Invariant)", PALETTE["b0"], "--", None, by_k),
    ]:
        ys = [src_dict[kk]["means"][mk]["ap"] for kk in k_keys]
        ax1.plot(kappas, ys, label=lbl, color=col, linestyle=ls, marker=mk_sym, markersize=5, lw=1.8)

    ax1.set_xlabel("Post-$t_0$ Finite Fix-Delay Multiplier $\\kappa$ (Unlinked $+\\infty$ Preserved)")
    ax1.set_ylabel("10-Stream Mean Average Precision (AP)")
    ax1.set_title("(a) Future-Only Latency Scaling ($\\kappa$, Frozen Initial State)", fontweight="bold")
    ax1.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.2)
    ax1.grid(linestyle=":", alpha=0.5)
    ymin1, ymax1 = ax1.get_ylim()
    ax1.set_ylim(max(0.0, ymin1 - 0.02), ymax1 * 1.15)

    # Panel (b): RQ4b Unconfounded 2x2 DriftLens Factorial + Pote et al. Ref
    cfgs = rq4b_j["configs"]
    mon_rows = [
        ("DriftLens_IID_Static_95_CD1", "IID+Static\n(p95, cd=1)"),
        ("DriftLens_IID_Rolling_95_CD1", "IID+RollRefTh\n(p95, cd=1)"),
        ("DriftLens_Block_Static_95_CD1", "Block+Static\n(p95, cd=1)"),
        ("DriftLens_Block_Rolling_95_CD1_V3", "Block+RollRefTh\n(Default)"),
        ("DriftLens_N46_Ref_IID_Static_98_CD0", "Pote Ref\n(p98, cd=0)"),
        ("RBF_MMD_Block_Rolling", "RBF-MMD\n(Block+Roll)"),
    ]
    xm = np.arange(len(mon_rows))
    bw = 0.27
    al_r = [100.0 * cfgs[k]["mean_alarm_rate_10streams"] for k, _ in mon_rows]
    far_r = [100.0 * cfgs[k]["stationary_false_alarm_rate"] for k, _ in mon_rows]
    spu_ap = [100.0 * cfgs[k]["means_10streams"]["Survival_PU"]["ap"] for k, _ in mon_rows]

    ax2.bar(xm - bw, al_r, width=bw, color="#F59E0B", alpha=0.88, label="Stream Alarm Rate (%)")
    ax2.bar(xm, far_r, width=bw, color="#DC2626", alpha=0.84, label="Resampled-Null Policy Alarm (%)")
    ax2.bar(xm + bw, spu_ap, width=bw, color=PALETTE["spu"], alpha=0.90, label="Survival-PU AP (\u00d7100)")
    ax2.set_xticks(xm)
    ax2.set_xticklabels([l for _, l in mon_rows], fontsize=7.4)
    ax2.set_ylabel("Percentage (%) / Scaled AP (\u00d7100)")
    ax2.set_ylim(0, max(max(al_r), max(far_r), max(spu_ap)) * 1.22)
    ax2.set_title("(b) 2\u00d72 DriftLens Calibration & Resampled-Null Policy Alarm", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig6_latency_and_drift_factorial_v3")


def plot_fig7_szz_window_sensitivity(fig_dir: str, rq4d_j: Dict, rq4e_j: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel (a): RQ4d Semi-Synthetic PR-SZZ Perturbation (Pop_N60_Indep_PermFN 2x2 Train x Eval)
    reg = rq4d_j["regimes"]["Pop_N60_Indep_PermFN"]["cells_mean_ap"]
    cells = [
        ("TrainRef__EvalRef", "Train: Ref\nEval: Ref"),
        ("TrainPert__EvalRef", "Train: Noisy SZZ\nEval: Unpert. Ref"),
        ("TrainRef__EvalPert", "Train: Ref\nEval: Noisy SZZ"),
        ("TrainPert__EvalPert", "Train: Noisy SZZ\nEval: Noisy SZZ"),
    ]
    m_szz = [
        ("Survival_PU", "Survival-PU", PALETTE["spu"]),
        ("BORB_Native", "BORB (ElasticNet)", PALETTE["track2"]),
        ("ORB_OHT_Native", "ORB (Hoeffding)", "#FB923C"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"]),
        ("Wait_90", "Wait-90", PALETTE["wait"]),
        ("B2_4W", "Causal-4W", PALETTE["b2"]),
    ]
    xs = np.arange(len(cells))
    bw = 0.13
    offs = (np.arange(len(m_szz)) - (len(m_szz) - 1) / 2.0) * bw
    ymax_s = 0.0
    for idx_m, (mk, mlbl, col) in enumerate(m_szz):
        vals = [reg[ck][mk] for ck, _ in cells]
        ymax_s = max(ymax_s, max(vals))
        ax1.bar(xs + offs[idx_m], vals, width=bw * 0.93, color=col, label=mlbl)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([cl for _, cl in cells], fontsize=8.2)
    ax1.set_ylabel("Mean AP Across 5 PR-SZZ Streams")
    ax1.set_ylim(0, ymax_s * 1.25)
    ax1.set_title("(a) 2\u00d72 Semi-Synthetic SZZ Noise Audit ($\\alpha_{\\mathrm{FP}}=8.4\\%,\\;\\beta_{\\mathrm{FN}}=47.2\\%$)", fontweight="bold")
    ax1.legend(ncol=2, loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): RQ4e Window Size Sweep — Strictly Matched Fixed-1200 Post-Onset Horizon Across All 5 Methods!
    ws = [200, 400, 600, 800]
    w_sw = rq4e_j["window_size_sweep"]
    for mk, lbl, col, ls, m_sym in [
        ("B1b_4W_Oracle", "Window Oracle", PALETTE["oracle"], "--", "s"),
        ("Survival_PU", "Survival-PU", PALETTE["spu"], "-", "o"),
        ("Wait_90", "Wait-90", PALETTE["wait"], "-.", "^"),
        ("B2_4W", "Causal-4W", PALETTE["b2"], "-", "v"),
        ("B0_Frozen", "Frozen Host", PALETTE["b0"], ":", "d"),
    ]:
        yv = [w_sw[str(w)]["fixed_1200_post_onset_ap"][mk] for w in ws]
        ax2.plot(ws, yv, marker=m_sym, markersize=5, lw=1.8, linestyle=ls, color=col, label=lbl)

    ax2.set_xticks(ws)
    ax2.set_xlabel("Streaming Window Size $W$ (Commits)")
    ax2.set_ylabel("Fixed-1,200 Post-Onset AP (3 Rep. Shifts)")
    ax2.set_title("(b) Matched-Horizon Sweep (3 Rep. Controlled Shifts)", fontweight="bold")
    ax2.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)
    ax2.grid(linestyle=":", alpha=0.5)
    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(max(0.0, ymin2 - 0.02), ymax2 * 1.14)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig7_szz_window_sensitivity_v3")


def plot_fig8_llm_and_runtime_forensics(fig_dir: str, llm_j: Dict, for_j: Dict, derived: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.12, 1.08]})

    # Panel (a): RQ4f Live Audited Code-LLM (`deepseek-flash`, 6,400 calls) on Qt & OpenStack (320 commits each)
    qt_r = llm_j["ecosystems"]["qt"]["regimes"]
    os_r = llm_j["ecosystems"]["openstack"]["regimes"]
    llm_items = [
        ("LLM_ZeroShot_Causal", "ap", "Zero-Shot\n(LLM)"),
        ("LLM_FewShot_Censored_1W", "ap_mean", "6-Shot\n(Cens. 1W)"),
        ("LLM_FewShot_Oracle_Paired_1W", "ap_mean", "6-Shot\n(Oracle)"),
        ("LLM_FewShot_Mature_Causal", "ap_mean", "6-Shot\n(Mature)"),
        ("CodeBERT_B0_Frozen", "ap", "CodeBERT\n(Init, 0 Al.)"),
        ("LGB_B0_Frozen", "ap", "LightGBM\n(Init, 0 Al.)"),
        ("Stream_BORB_Native", "ap", "BORB\n(ElasticNet)"),
    ]
    xl = np.arange(len(llm_items))
    bw = 0.36
    v_qt = [float(qt_r[k][fk]) for k, fk, _ in llm_items]
    v_os = [float(os_r[k][fk]) for k, fk, _ in llm_items]
    ax1.bar(xl - bw / 2, v_qt, width=bw, color="#2563EB", alpha=0.88, label="Qt Slice (320 Commits, AP)")
    ax1.bar(xl + bw / 2, v_os, width=bw, color="#059669", alpha=0.88, label="OpenStack Slice (320 Commits, AP)")
    ax1.set_xticks(xl)
    ax1.set_xticklabels([lbl for _, _, lbl in llm_items], fontsize=7.6)
    ax1.set_ylabel("Average Precision (AP)")
    ax1.set_ylim(0, max(max(v_qt), max(v_os)) * 1.24)
    ax1.set_title("(a) Audited Code-LLM (deepseek-flash, 6,400 Calls) vs. JIT Models", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.0)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): RQ5 Phase-Decomposed End-to-End Stream Runtime (Benchmark B 27-Stream Mean, Seconds)
    rt = derived["runtime_summary"]["benchmark_b_mlp_27streams_mean"]
    rt_methods = [
        ("B0_Frozen", "Frozen\n(MLP)"),
        ("B2_4W", "Causal-4W\n(MLP)"),
        ("Wait_90", "Wait-90\n(MLP)"),
        ("Teacher_Only_Matched", "Teach-M.\n(MLP)"),
        ("Survival_PU", "SPU\n(MLP)"),
        ("BORB_Native", "BORB\n(Elast.)"),
        ("ORB_OHT_Native", "ORB\n(Hoeff.)"),
        ("ODaSC_OHT_Native", "ODaSC\n(Hoeff.)"),
    ]
    xr = np.arange(len(rt_methods))
    init_s = [(rt[k]["own_init_fit_ms"] + rt[k]["shared_anchor_init_ms"] + rt[k]["shared_monitor_init_ms"]) / 1000.0 for k, _ in rt_methods]
    pred_mon_s = [(rt[k]["stream_predict_and_embed_ms"] + rt[k]["drift_monitor_ms"]) / 1000.0 for k, _ in rt_methods]
    adapt_cal_s = [(rt[k]["adapt_only_wall_ms"] + rt[k]["calibration_ms"]) / 1000.0 for k, _ in rt_methods]

    ax2.bar(xr, init_s, width=0.56, color="#64748B", label="Init & Reference Fit (s)")
    ax2.bar(xr, pred_mon_s, bottom=init_s, width=0.56, color="#3B82F6", label="Stream Predict & Monitor (s)")
    bot2 = [a + b for a, b in zip(init_s, pred_mon_s)]
    ax2.bar(xr, adapt_cal_s, bottom=bot2, width=0.56, color=PALETTE["spu"], label="Online / Alarm Adapt + Calib. (s)")
    totals = [a + b + c for a, b, c in zip(init_s, pred_mon_s, adapt_cal_s)]
    for idx_i, tot in enumerate(totals):
        ax2.text(idx_i, tot + max(totals) * 0.02, f"{tot:.2f}s", ha="center", va="bottom", fontsize=7.4)
    ax2.set_xticks(xr)
    ax2.set_xticklabels([l for _, l in rt_methods], fontsize=7.6)
    ax2.set_ylabel("Standalone End-to-End Wall Time per Stream (s)")
    ax2.set_ylim(0, max(totals) * 1.20)
    ax2.set_title("(b) Standalone End-to-End Runtime Breakdown (27-Stream Mean)", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, fig_dir, "fig8_llm_and_runtime_forensics_v3")


# ==============================================================================
# Programmatic LaTeX Tables + 100% Cell-Level Registry Builder
# ==============================================================================

class CellRegistryBuilder:
    def __init__(self, loaded_jsons: Dict[str, Dict]):
        self.loaded = loaded_jsons
        self.entries: List[Dict[str, Any]] = []

    def _resolve(self, jf: str, jpath: Tuple[str, ...]) -> float:
        cur: Any = self.loaded[jf]
        for k in jpath:
            cur = cur[int(k)] if isinstance(cur, list) else cur[k]
        return float(cur)

    def num(
        self,
        table: str,
        row: str,
        col: int,
        jf: str,
        jpath: Tuple[str, ...],
        fmt: str = ".4f",
        bold: bool = False,
        plus: bool = False,
    ) -> str:
        val = self._resolve(jf, jpath)
        self.entries.append({
            "table": table,
            "row": row,
            "col": int(col),
            "json_file": jf,
            "json_path": list(jpath),
            "fmt": fmt,
            "role": "primary",
        })
        txt = format(val, fmt)
        if plus and val > 0 and not txt.startswith("+"):
            txt = "+" + txt
        if txt.endswith("%"):
            txt = txt[:-1] + r"\%"
        return f"\\textbf{{{txt}}}" if bold else txt

    def ci_cell(
        self,
        table: str,
        row: str,
        col: int,
        jf: str,
        mean_path: Tuple[str, ...],
        lo_path: Tuple[str, ...],
        hi_path: Tuple[str, ...],
        fmt: str = ".4f",
        bold: bool = False,
    ) -> str:
        vm = self._resolve(jf, mean_path)
        vlo = self._resolve(jf, lo_path)
        vhi = self._resolve(jf, hi_path)
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(mean_path), "fmt": fmt, "role": "primary"})
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(lo_path), "fmt": fmt, "role": "ci_lo"})
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(hi_path), "fmt": fmt, "role": "ci_hi"})
        sm = format(vm, fmt)
        if vm > 0 and not sm.startswith("+"):
            sm = "+" + sm
        if bold:
            sm = f"\\textbf{{{sm}}}"
        return f"{sm} [{format(vlo, fmt)}, {format(vhi, fmt)}]"

    def wtl_cell(
        self,
        table: str,
        row: str,
        col: int,
        jf: str,
        w_path: Tuple[str, ...],
        t_path: Tuple[str, ...],
        l_path: Tuple[str, ...],
    ) -> str:
        vw = int(round(self._resolve(jf, w_path)))
        vt = int(round(self._resolve(jf, t_path)))
        vl = int(round(self._resolve(jf, l_path)))
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(w_path), "fmt": ".0f", "role": "slash_0"})
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(t_path), "fmt": ".0f", "role": "slash_1"})
        self.entries.append({"table": table, "row": row, "col": int(col), "json_file": jf, "json_path": list(l_path), "fmt": ".0f", "role": "slash_2"})
        return f"{vw}/{vt}/{vl}"


def generate_digest_and_tables(base_dir: str, scratch_dir: str, skip_plots: bool = False) -> Dict:
    res_dir = os.path.join(base_dir, "results_v3")
    fig_dir = os.path.join(base_dir, "figures")

    audit_j = _load_json(res_dir, "data_provenance_audit_v3.json")
    ba_j = _load_json(res_dir, "benchmark_a_multihost_v3.json")
    bb_j = _load_json(res_dir, "benchmark_b_main_v3.json")
    conf_j = _load_json(res_dir, "confirmatory_cohort_v3.json")
    rq3_j = _load_json(res_dir, "rq3_ablations_v3.json")
    rq4a_j = _load_json(res_dir, "rq4a_latency_intervention_v3.json")
    rq4a_fc_j = _load_json(res_dir, "rq4a_fixedcurve_control_v3.json")
    rq4b_j = _load_json(res_dir, "rq4b_factorial_monitors_v3.json")
    rq4d_j = _load_json(res_dir, "rq4d_szz_perturbation_v3.json")
    rq4e_j = _load_json(res_dir, "rq4e_sensitivity_sweeps_v3.json")
    llm_j = _load_json(res_dir, "rq4f_llm_audit_v3.json")
    for_j = _load_json(res_dir, "rq5_natural_forensics_v3.json")
    rev_j = _load_json(res_dir, "reviewer_mechanism_audits_v3.json")

    derived = build_derived_aggregates(base_dir, audit_j, ba_j, bb_j, conf_j)

    # Render all 8 empirical figures unless skipped for fast table regeneration
    if not skip_plots:
        plot_fig1_multimodal_audit(fig_dir, ba_j, derived)
        plot_fig2_trajectories_and_skip(fig_dir, ba_j)
        plot_fig3_benchmark_b_clusters(fig_dir, bb_j, derived)
        plot_fig4_confirmatory_boundary(fig_dir, bb_j, conf_j)
        plot_fig5_ablations_and_controls(fig_dir, rq3_j)
        plot_fig6_latency_and_drift_factorial(fig_dir, rq4a_j, rq4b_j, rq4a_fc_j)
        plot_fig7_szz_window_sensitivity(fig_dir, rq4d_j, rq4e_j)
        plot_fig8_llm_and_runtime_forensics(fig_dir, llm_j, for_j, derived)

    loaded_map = {
        "data_provenance_audit_v3.json": audit_j,
        "benchmark_a_multihost_v3.json": ba_j,
        "benchmark_b_main_v3.json": bb_j,
        "confirmatory_cohort_v3.json": conf_j,
        "rq3_ablations_v3.json": rq3_j,
        "rq4a_latency_intervention_v3.json": rq4a_j,
        "rq4a_fixedcurve_control_v3.json": rq4a_fc_j,
        "rq4b_factorial_monitors_v3.json": rq4b_j,
        "rq4d_szz_perturbation_v3.json": rq4d_j,
        "rq4e_sensitivity_sweeps_v3.json": rq4e_j,
        "rq4f_llm_audit_v3.json": llm_j,
        "rq5_natural_forensics_v3.json": for_j,
        "reviewer_mechanism_audits_v3.json": rev_j,
        "manuscript_derived_aggregates_v3.json": derived,
    }
    reg = CellRegistryBuilder(loaded_map)
    reg_supp = CellRegistryBuilder(loaded_map)
    tex_tables: List[str] = []
    supp_tables: List[str] = []

    def _is_best_fmt(val: float, candidates: List[float], fmt: str, lower_better: bool = False) -> bool:
        if not candidates:
            return False
        best_v = min(candidates) if lower_better else max(candidates)
        return format(val, fmt) == format(best_v, fmt)

    # 1. Table `tab:audit_censoring` (Corpus Provenance, Scheduled-Boundary & Alarm-Conditioned Censoring Audit)
    t_lbl = "tab:audit_censoring"
    rows_t1 = []
    for r_lbl, c_key, n_jf, n_jp, prev_jf, prev_jp in [
        ("Qt Natural (13-Metric)", "qt_natural_25k", "data_provenance_audit_v3.json", ("benchmark_a", "qt_audit", "total_commits"), "data_provenance_audit_v3.json", ("benchmark_a", "qt_audit", "prevalence")),
        ("OpenStack Natural (13-Metric)", "openstack_natural_13k", "data_provenance_audit_v3.json", ("benchmark_a", "openstack_audit", "total_commits"), "data_provenance_audit_v3.json", ("benchmark_a", "openstack_audit", "prevalence")),
        ("Qt Shift (qtbase $\\to$ qtdecl)", "qt_abrupt_qtbase_qtdecl", "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "2", "N"), "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "2", "prevalence")),
        ("Qt Shift (Core $\\to$ UX)", "qt_abrupt_core_ux", "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "3", "N"), "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "3", "prevalence")),
        ("OS Shift (nova $\\to$ neutron)", "openstack_abrupt_nova_neut", "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "4", "N"), "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "4", "prevalence")),
        ("OS Shift (Comp $\\to$ Storage)", "openstack_abrupt_comp_stor", "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "5", "N"), "data_provenance_audit_v3.json", ("benchmark_a", "primary_streams", "5", "prevalence")),
    ]:
        c_n = reg.num(t_lbl, r_lbl, 1, n_jf, n_jp, ".0f")
        c_pr = reg.num(t_lbl, r_lbl, 2, prev_jf, prev_jp, ".2%")
        c_ml = reg.num(t_lbl, r_lbl, 3, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "median_latency_days"), ".1f")
        c_1w = reg.num(t_lbl, r_lbl, 4, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "cens_1w"), ".1%")
        c_4w = reg.num(t_lbl, r_lbl, 5, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "cens_4w"), ".1%")
        al_v = derived["window_censoring_audit"][c_key].get("alarm_cens_1w")
        c_al = reg.num(t_lbl, r_lbl, 6, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "alarm_cens_1w"), ".1%") if al_v is not None else "--"
        rows_t1.append(f"{r_lbl} & {c_n} & {c_pr} & {c_ml} & {c_1w} & {c_4w} & {c_al} \\\\")
    rows_t1.append("\\midrule")
    for r_lbl, c_key in [
        ("Cabral ICSE'19 (14 Streams)", "cohort_Cabral19"),
        ("ApacheJIT MSR'22 Dev (8 Streams)", "cohort_ApacheJIT"),
        ("PR-SZZ SQJ'26 (5 Streams)", "cohort_N60_PRSZZ"),
        ("Confirmatory ApacheJIT (5 Repos*)", "cohort_Confirmatory_ApacheJIT"),
        ("Confirmatory (3 Standalone)", "cohort_Confirmatory_Standalone3"),
    ]:
        c_n = reg.num(t_lbl, r_lbl, 1, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "total_commits"), ".0f")
        c_pr = reg.num(t_lbl, r_lbl, 2, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "mean_prevalence"), ".2%")
        c_ml = reg.num(t_lbl, r_lbl, 3, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "median_latency_days"), ".1f")
        c_1w = reg.num(t_lbl, r_lbl, 4, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "cens_1w"), ".1%")
        c_4w = reg.num(t_lbl, r_lbl, 5, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "cens_4w"), ".1%")
        c_al = reg.num(t_lbl, r_lbl, 6, "manuscript_derived_aggregates_v3.json", ("window_censoring_audit", c_key, "alarm_cens_1w"), ".1%")
        rows_t1.append(f"{r_lbl} & {c_n} & {c_pr} & {c_ml} & {c_1w} & {c_4w} & {c_al} \\\\")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Verification latency and right-censoring audit across cohorts at $W=400$. Asterisks mark Hadoop monorepo slices (Table~S-7).}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Stream / Cohort} & \\textbf{$N$} & \\textbf{Prev.} & \\textbf{Lag} & \\textbf{1W Sch.} & \\textbf{4W Sch.} & \\textbf{1W Alm.} \\\\\n\\midrule\n"
        + "\n".join(rows_t1)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: Prev.\\ = defect prevalence; Lag = median SZZ fix delay (days); 1W/4W Sch.\\ = unrevealed defect share at scheduled 1-window ($400$c) / 4-window ($1{,}600$c) boundaries; 1W Alm.\\ = unrevealed share at fired DriftLens alarms.\\par}\n"
        "\\end{table}\n"
    )

    # 2. Table `tab:bench_a_main` (Main 7.5pt 8-col summary) + `tab:or_bench_a_full` (Online Resource 1 full 10-col per-host breakdown)
    t_lbl = "tab:bench_a_main"
    t_lbl_full = "tab:or_bench_a_full"
    rows_ba = []
    rows_ba_full = []
    ba_methods_show = [
        ("B1b_4W_Oracle", "Window Oracle (4W)", True),
        ("B0_Frozen", "Frozen Host", True),
        ("B2_1W", "Causal-1W", True),
        ("B2_4W", "Causal-4W", True),
        ("Wait_90", "Wait-90", True),
        ("Pure_nnPU", "Pure-nnPU", True),
        ("FixedCurve_PU", "FixedCurve-PU", True),
        ("Teacher_Only_Matched", "Teacher-Matched", True),
        ("Survival_PU_WithGate", "SPU (+Gate)", True),
        ("Survival_PU", "Survival-PU", True),
        ("Full_LR_B0", "Full-LR Frozen", False),
        ("LApredict_B0", "LApredict Frozen", False),
        ("Churn_LR_B0", "Churn-LR Frozen", False),
        ("BORB_Native", "BORB (ElasticNet)", False),
        ("ORB_OHT_Native", "ORB (Hoeffding)", False),
        ("ODaSC_OHT_Native", "ODaSC (Hoeffding)", False),
        ("LOC_Only", "LOC-Only", False),
    ]
    dep_t1_ba = [mk for mk, _, is_t1 in ba_methods_show if is_t1 and mk != "B1b_4W_Oracle"]
    cands_ba_h = {
        hk: [derived["benchmark_a_by_host_6streams"][hk][mk]["ap"] for mk in dep_t1_ba]
        for hk in ("att_cnn_skip", "codebert_skip", "deepjit_cnn_skip", "lightgbm_none")
    }
    cands_ba_all = {}
    for met in ("ap", "f1", "f1_th", "recall_at_20pct_effort", "brier_cal"):
        cands_ba_all[met] = [
            (derived["benchmark_a_4host_6stream_mean"][mk][met] if is_t1 else derived["benchmark_a_track2_6stream_mean"][mk][met])
            for mk, _, is_t1 in ba_methods_show if mk != "B1b_4W_Oracle"
        ]

    for mk, mlbl, is_t1 in ba_methods_show:
        is_ora = (mk == "B1b_4W_Oracle")
        if is_t1:
            v1 = derived["benchmark_a_by_host_6streams"]["att_cnn_skip"][mk]["ap"]
            v2 = derived["benchmark_a_by_host_6streams"]["codebert_skip"][mk]["ap"]
            v3 = derived["benchmark_a_by_host_6streams"]["deepjit_cnn_skip"][mk]["ap"]
            v4 = derived["benchmark_a_by_host_6streams"]["lightgbm_none"][mk]["ap"]
            # Main table: DJIT (col 1), LGBM (col 2)
            m1 = reg.num(t_lbl, mlbl, 1, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "deepjit_cnn_skip", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v3, cands_ba_h["deepjit_cnn_skip"], ".4f")))
            m2 = reg.num(t_lbl, mlbl, 2, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "lightgbm_none", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v4, cands_ba_h["lightgbm_none"], ".4f")))
            # Online Resource full table: all 4 hosts
            f1 = reg_supp.num(t_lbl_full, mlbl, 1, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "att_cnn_skip", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v1, cands_ba_h["att_cnn_skip"], ".4f")))
            f2 = reg_supp.num(t_lbl_full, mlbl, 2, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "codebert_skip", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v2, cands_ba_h["codebert_skip"], ".4f")))
            f3 = reg_supp.num(t_lbl_full, mlbl, 3, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "deepjit_cnn_skip", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v3, cands_ba_h["deepjit_cnn_skip"], ".4f")))
            f4 = reg_supp.num(t_lbl_full, mlbl, 4, "manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "lightgbm_none", mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(v4, cands_ba_h["lightgbm_none"], ".4f")))
            src_key = "benchmark_a_4host_6stream_mean"
        else:
            m1 = m2 = "--"
            f1 = f2 = f3 = f4 = "--"
            src_key = "benchmark_a_track2_6stream_mean"

        va = derived[src_key][mk]["ap"]
        vf = derived[src_key][mk]["f1"]
        vft = derived[src_key][mk]["f1_th"]
        vr = derived[src_key][mk]["recall_at_20pct_effort"]
        vb = derived[src_key][mk]["brier_cal"]
        m3 = reg.num(t_lbl, mlbl, 3, "manuscript_derived_aggregates_v3.json", (src_key, mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(va, cands_ba_all["ap"], ".4f")))
        m4 = reg.num(t_lbl, mlbl, 4, "manuscript_derived_aggregates_v3.json", (src_key, mk, "f1"), ".4f", bold=(not is_ora and _is_best_fmt(vf, cands_ba_all["f1"], ".4f")))
        m5 = reg.num(t_lbl, mlbl, 5, "manuscript_derived_aggregates_v3.json", (src_key, mk, "f1_th"), ".4f", bold=(not is_ora and _is_best_fmt(vft, cands_ba_all["f1_th"], ".4f")))
        m6 = reg.num(t_lbl, mlbl, 6, "manuscript_derived_aggregates_v3.json", (src_key, mk, "recall_at_20pct_effort"), ".4f", bold=(not is_ora and _is_best_fmt(vr, cands_ba_all["recall_at_20pct_effort"], ".4f")))
        m7 = reg.num(t_lbl, mlbl, 7, "manuscript_derived_aggregates_v3.json", (src_key, mk, "brier_cal"), ".4f", bold=(not is_ora and _is_best_fmt(vb, cands_ba_all["brier_cal"], ".4f", lower_better=True)))
        rows_ba.append(f"{mlbl} & {m1} & {m2} & {m3} & {m4} & {m5} & {m6} & {m7} \\\\")

        f5 = reg_supp.num(t_lbl_full, mlbl, 5, "manuscript_derived_aggregates_v3.json", (src_key, mk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(va, cands_ba_all["ap"], ".4f")))
        f6 = reg_supp.num(t_lbl_full, mlbl, 6, "manuscript_derived_aggregates_v3.json", (src_key, mk, "f1"), ".4f", bold=(not is_ora and _is_best_fmt(vf, cands_ba_all["f1"], ".4f")))
        f7 = reg_supp.num(t_lbl_full, mlbl, 7, "manuscript_derived_aggregates_v3.json", (src_key, mk, "f1_th"), ".4f", bold=(not is_ora and _is_best_fmt(vft, cands_ba_all["f1_th"], ".4f")))
        f8 = reg_supp.num(t_lbl_full, mlbl, 8, "manuscript_derived_aggregates_v3.json", (src_key, mk, "recall_at_20pct_effort"), ".4f", bold=(not is_ora and _is_best_fmt(vr, cands_ba_all["recall_at_20pct_effort"], ".4f")))
        f9 = reg_supp.num(t_lbl_full, mlbl, 9, "manuscript_derived_aggregates_v3.json", (src_key, mk, "brier_cal"), ".4f", bold=(not is_ora and _is_best_fmt(vb, cands_ba_all["brier_cal"], ".4f", lower_better=True)))
        rows_ba_full.append(f"{mlbl} & {f1} & {f2} & {f3} & {f4} & {f5} & {f6} & {f7} & {f8} & {f9} \\\\")
        if mk in ("B1b_4W_Oracle", "Survival_PU"):
            rows_ba.append("\\midrule")
            rows_ba_full.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Benchmark A six-stream mean across four Track~1 hosts and Track~2 baselines over five seeds (full per-host breakdown in Table~S-3). Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.8pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{DJIT} & \\textbf{LGBM} & \\textbf{AP} & \\textbf{$F_1$} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{R@20} & \\textbf{Br.$\\downarrow$} \\\\\n\\midrule\n"
        + "\n".join(rows_ba)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: DJIT = DeepJIT-CNN AP; LGBM = LightGBM AP; AP = 4-host (Track~1) or model (Track~2) mean AP; $F_{1\\mathrm{t}}$ = causally tuned $F_{1,\\mathrm{th}}$; R@20 = Recall@20\\%Effort; Br.$\\downarrow$ = post-Platt Brier loss.\\par}\n"
        "\\end{table}\n"
    )
    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Complete 10-column Benchmark A causal supervision audit across all four Track~1 hosts (Attention-CNN, CodeBERT, DeepJIT-CNN, LightGBM) and Track~2 baselines over six Qt and OpenStack streams (5 seeds; 13 commit-authoring metrics). Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl_full}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{Att.} & \\textbf{CodeB.} & \\textbf{DJIT} & \\textbf{LGBM} & \\textbf{AP} & \\textbf{$F_1$} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{R@20} & \\textbf{Brier$\\downarrow$} \\\\\n\\midrule\n"
        + "\n".join(rows_ba_full)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 3. Table `tab:bench_a_sens` -> Online Resource 1 (Table S4)
    t_lbl = "tab:bench_a_sens"
    rows_sens = []
    for s_key, r_lbl in [
        ("qt_premerge28__codebert", "Qt Pre-Merge 28 (CodeBERT)"),
        ("qt_tier1_only__codebert", "Qt High-Conf. SZZ Only (CodeBERT)"),
        ("qt_drop_unlinked__codebert", "Qt Drop Unlinked (CodeBERT)"),
        ("openstack_premerge28__codebert", "OpenStack Pre-Merge 28 (CodeBERT)"),
        ("openstack_tier1_only__codebert", "OpenStack High-Conf. SZZ Only (CodeBERT)"),
        ("openstack_drop_unlinked__codebert", "OpenStack Drop Unlinked (CodeBERT)"),
    ]:
        dep_vals = [ba_j["provenance_sensitivity"][s_key]["methods"][m]["common_eval_subset"]["ap"] for m in ("B0_Frozen", "B2_4W", "Wait_90", "Survival_PU")]
        v_b0 = ba_j["provenance_sensitivity"][s_key]["methods"]["B0_Frozen"]["common_eval_subset"]["ap"]
        v_b2 = ba_j["provenance_sensitivity"][s_key]["methods"]["B2_4W"]["common_eval_subset"]["ap"]
        v_w9 = ba_j["provenance_sensitivity"][s_key]["methods"]["Wait_90"]["common_eval_subset"]["ap"]
        v_sp = ba_j["provenance_sensitivity"][s_key]["methods"]["Survival_PU"]["common_eval_subset"]["ap"]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "benchmark_a_multihost_v3.json", ("provenance_sensitivity", s_key, "methods", "B0_Frozen", "common_eval_subset", "ap"), ".4f", bold=_is_best_fmt(v_b0, dep_vals, ".4f"))
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "benchmark_a_multihost_v3.json", ("provenance_sensitivity", s_key, "methods", "B1b_4W_Oracle", "common_eval_subset", "ap"), ".4f")
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "benchmark_a_multihost_v3.json", ("provenance_sensitivity", s_key, "methods", "B2_4W", "common_eval_subset", "ap"), ".4f", bold=_is_best_fmt(v_b2, dep_vals, ".4f"))
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "benchmark_a_multihost_v3.json", ("provenance_sensitivity", s_key, "methods", "Wait_90", "common_eval_subset", "ap"), ".4f", bold=_is_best_fmt(v_w9, dep_vals, ".4f"))
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "benchmark_a_multihost_v3.json", ("provenance_sensitivity", s_key, "methods", "Survival_PU", "common_eval_subset", "ap"), ".4f", bold=_is_best_fmt(v_sp, dep_vals, ".4f"))
        rows_sens.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} \\\\")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Benchmark A provenance and feature-scope sensitivity evaluated strictly on identical common evaluation commit IDs per ecosystem (CodeBERT with frozen linear skip, AP). Bold marks the highest deployable AP in each row.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.4pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Sensitivity Variant} & \\textbf{Frozen} & \\textbf{Oracle} & \\textbf{Causal-4W} & \\textbf{Wait-90} & \\textbf{Survival-PU} \\\\\n\\midrule\n"
        + "\n".join(rows_sens)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 4. Table `tab:bench_b_clusters` (Main 7.5pt 7-col) + `tab:or_bench_b_full` (Online Resource 8-col with ROC-AUC)
    t_lbl = "tab:bench_b_clusters"
    t_lbl_bb_full = "tab:or_bench_b_full"
    rows_bb = []
    rows_bb_full = []
    bb_methods_show = [
        ("B1b_4W_Oracle", "Window Oracle (4W)", True),
        ("B0_Frozen", "Frozen Host", True),
        ("B2_1W", "Causal-1W", True),
        ("B2_4W", "Causal-4W", True),
        ("Wait_90", "Wait-90", True),
        ("Pure_nnPU", "Pure-nnPU", True),
        ("ODaSC_Host", "ODaSC-Host", False),
        ("FixedCurve_PU", "FixedCurve-PU", True),
        ("Teacher_Only_Exact", "Teacher-Exact", True),
        ("Teacher_Only_Matched", "Teacher-Matched", True),
        ("Survival_PU_WithGate", "SPU (+Gate)", False),
        ("Survival_PU", "Survival-PU", True),
        ("LOC_Only", "LOC-Only", False),
        ("LApredict_B0", "LApredict Frozen", False),
        ("Full_LR_Adapt", "Full-LR Adapt-4W", False),
        ("BORB_Native", "BORB (ElasticNet)", False),
        ("BORB_Alarm", "BORB-Alarm", False),
        ("ORB_OHT_Native", "ORB (Hoeffding)", False),
        ("ODaSC_OHT_Native", "ODaSC (Hoeffding)", False),
    ]
    dep_bb = [mk for mk, _, _ in bb_methods_show if mk != "B1b_4W_Oracle"]
    dep_bb_lgb = [mk for mk, _, hl in bb_methods_show if mk != "B1b_4W_Oracle" and hl]
    c_k25 = [bb_j["cluster_inference_mlp"]["primary_ap"]["K25_cluster_means"][m] for m in dep_bb]
    c_k24 = [bb_j["cluster_inference_mlp"]["primary_ap"]["sensitivity_K24_disjoint_from_benchmark_a"]["K24_cluster_means"][m] for m in dep_bb]
    c_f1t = [bb_j["cluster_inference_mlp"]["secondary_f1_th"]["K25_cluster_means"][m] for m in dep_bb]
    c_auc = [bb_j["cluster_inference_mlp"]["secondary_roc_auc"]["K25_cluster_means"][m] for m in dep_bb]
    c_bri = [bb_j["cluster_inference_mlp"]["secondary_brier_cal"]["K25_cluster_means"][m] for m in dep_bb]
    c_lgb = [bb_j["cluster_inference_lightgbm_ap"]["K25_cluster_means"][m] for m in dep_bb_lgb]
    c_shf = [derived["benchmark_b_controlled_shift_4stream_mean"][m]["post_onset_ap"] for m in dep_bb]

    for mk, mlbl, has_lgb in bb_methods_show:
        is_ora = (mk == "B1b_4W_Oracle")
        v1 = bb_j["cluster_inference_mlp"]["primary_ap"]["K25_cluster_means"][mk]
        v2 = bb_j["cluster_inference_mlp"]["primary_ap"]["sensitivity_K24_disjoint_from_benchmark_a"]["K24_cluster_means"][mk]
        v3 = bb_j["cluster_inference_mlp"]["secondary_f1_th"]["K25_cluster_means"][mk]
        v4 = bb_j["cluster_inference_mlp"]["secondary_roc_auc"]["K25_cluster_means"][mk]
        v5 = bb_j["cluster_inference_mlp"]["secondary_brier_cal"]["K25_cluster_means"][mk]
        v7 = derived["benchmark_b_controlled_shift_4stream_mean"][mk]["post_onset_ap"]
        # Main 7-col table (AP K25, AP K24, F1t, Brier, LGB K25, Shift AP)
        m1 = reg.num(t_lbl, mlbl, 1, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v1, c_k25, ".4f")))
        m2 = reg.num(t_lbl, mlbl, 2, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", "sensitivity_K24_disjoint_from_benchmark_a", "K24_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v2, c_k24, ".4f")))
        m3 = reg.num(t_lbl, mlbl, 3, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "secondary_f1_th", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v3, c_f1t, ".4f")))
        m4 = reg.num(t_lbl, mlbl, 4, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "secondary_brier_cal", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v5, c_bri, ".4f", lower_better=True)))
        if has_lgb:
            v6 = bb_j["cluster_inference_lightgbm_ap"]["K25_cluster_means"][mk]
            m5 = reg.num(t_lbl, mlbl, 5, "benchmark_b_main_v3.json", ("cluster_inference_lightgbm_ap", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v6, c_lgb, ".4f")))
        else:
            m5 = "--"
        m6 = reg.num(t_lbl, mlbl, 6, "manuscript_derived_aggregates_v3.json", ("benchmark_b_controlled_shift_4stream_mean", mk, "post_onset_ap"), ".4f", bold=(not is_ora and _is_best_fmt(v7, c_shf, ".4f")))
        rows_bb.append(f"{mlbl} & {m1} & {m2} & {m3} & {m4} & {m5} & {m6} \\\\")

        # Online Resource full 8-col mirror (includes ROC-AUC)
        f1 = reg_supp.num(t_lbl_bb_full, mlbl, 1, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v1, c_k25, ".4f")))
        f2 = reg_supp.num(t_lbl_bb_full, mlbl, 2, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", "sensitivity_K24_disjoint_from_benchmark_a", "K24_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v2, c_k24, ".4f")))
        f3 = reg_supp.num(t_lbl_bb_full, mlbl, 3, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "secondary_f1_th", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v3, c_f1t, ".4f")))
        f4 = reg_supp.num(t_lbl_bb_full, mlbl, 4, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "secondary_roc_auc", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v4, c_auc, ".4f")))
        f5 = reg_supp.num(t_lbl_bb_full, mlbl, 5, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "secondary_brier_cal", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v5, c_bri, ".4f", lower_better=True)))
        f6 = reg_supp.num(t_lbl_bb_full, mlbl, 6, "benchmark_b_main_v3.json", ("cluster_inference_lightgbm_ap", "K25_cluster_means", mk), ".4f", bold=(not is_ora and _is_best_fmt(v6, c_lgb, ".4f"))) if has_lgb else "--"
        f7 = reg_supp.num(t_lbl_bb_full, mlbl, 7, "manuscript_derived_aggregates_v3.json", ("benchmark_b_controlled_shift_4stream_mean", mk, "post_onset_ap"), ".4f", bold=(not is_ora and _is_best_fmt(v7, c_shf, ".4f")))
        rows_bb_full.append(f"{mlbl} & {f1} & {f2} & {f3} & {f4} & {f5} & {f6} & {f7} \\\\")
        if mk in ("B1b_4W_Oracle", "Survival_PU"):
            rows_bb.append("\\midrule")
            rows_bb_full.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Benchmark B evaluation across 25 natural donor clusters, the 24-cluster disjoint check, LightGBM, and four controlled shifts (full metrics in Table~S-5). Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{$K{=}25$} & \\textbf{$K{=}24$} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{Br.$\\downarrow$} & \\textbf{LGB} & \\textbf{Shift} \\\\\n\\midrule\n"
        + "\n".join(rows_bb)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: $K{=}25$/$K{=}24$ = MLP AP across all 25 / 24 Benchmark-A-disjoint clusters; $F_{1\\mathrm{t}}$ = tuned $F_{1,\\mathrm{th}}$; Br.$\\downarrow$ = post-Platt Brier; LGB = LightGBM $K{=}25$ AP; Shift = 4-stream post-onset AP.\\par}\n"
        "\\end{table}\n"
    )
    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Complete 8-column Benchmark B evaluation including ROC-AUC across $K=25$ natural donor clusters ($K=24$ sensitivity), LightGBM ($K=25$), and four controlled shift streams. Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl_bb_full}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{$K{=}25$} & \\textbf{$K{=}24$} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{AUC} & \\textbf{Br.$\\downarrow$} & \\textbf{LGB} & \\textbf{Shift} \\\\\n\\midrule\n"
        + "\n".join(rows_bb_full)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 5. Table `tab:bench_b_holm` (Pre-Declared Holm-Bonferroni Hypothesis Families 1, 2, 3 on Powered K=25)
    t_lbl = "tab:bench_b_holm"
    rows_hm = []
    short_hm = {
        "B1b_4W_Oracle": "Oracle-4W",
        "B0_Frozen": "Frozen",
        "B2_4W": "Causal-4W",
        "Wait_90": "Wait-90",
        "Pure_nnPU": "Pure-nnPU",
        "FixedCurve_PU": "FixedCurve",
        "Teacher_Only_Exact": "Teach-Exact",
        "Teacher_Only_Matched": "Teach-Match",
        "Survival_PU": "SPU",
        "Survival_PU_WithGate": "SPU+Gate",
        "Full_LR_Adapt": "Full-LR",
        "BORB_Native": "BORB",
        "ORB_OHT_Native": "ORB",
        "ODaSC_OHT_Native": "ODaSC",
    }
    ap_inf = bb_j["cluster_inference_mlp"]["primary_ap"]
    for fam_key, fam_tag in [
        ("family1_paradox_K25", "F1"),
        ("family2_competitors_K25", "F2"),
        ("family3_controls_K25", "F3"),
    ]:
        for idx_i, item in enumerate(ap_inf[fam_key]):
            ma_s = short_hm.get(item["method_a"], item["method_a"])
            mb_s = short_hm.get(item["method_b"], item["method_b"])
            r_lbl = f"({fam_tag}) {ma_s} vs. {mb_s}"
            sig = bool(item["significant_005_holm"])
            c_ci = reg.ci_cell(
                t_lbl, r_lbl, 1, "benchmark_b_main_v3.json",
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "mean_diff"),
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "ci95_lo"),
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "ci95_hi"),
                fmt=".4f", bold=sig,
            )
            c_wtl = reg.wtl_cell(
                t_lbl, r_lbl, 2, "benchmark_b_main_v3.json",
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "wins"),
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "ties"),
                ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "losses"),
            )
            c_dz = reg.num(t_lbl, r_lbl, 3, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "cohen_dz"), ".3f")
            p_val = float(item["p_holm"])
            p_fmt = ".2e" if p_val < 1e-3 else ".4f"
            c_ph = reg.num(t_lbl, r_lbl, 4, "benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", fam_key, str(idx_i), "p_holm"), p_fmt, bold=sig)
            rows_hm.append(f"{r_lbl} & {c_ci} & {c_wtl} & {c_dz} & {c_ph} \\\\")
        if fam_tag != "F3":
            rows_hm.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Paired cluster-level Wilcoxon signed-rank inference with Holm--Bonferroni adjustment across 25 natural donor clusters for whole-stream AP. Bold marks $p_{\\mathrm{Holm}} < 0.05$.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccc@{{}}}}\n\\toprule\n"
        "\\textbf{Contrast ($A$ vs.\\ $B$)} & \\textbf{$\\Delta\\mathrm{AP}$ [95\\% CI]} & \\textbf{W/T/L} & \\textbf{$d_z$} & \\textbf{$p_{\\mathrm{Holm}}$} \\\\\n\\midrule\n"
        + "\n".join(rows_hm)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 6. Table `tab:confirmatory` (Untouched Out-of-Sample Confirmatory Cohort)
    t_lbl = "tab:confirmatory"
    rows_cf = []
    conf_methods = [
        ("ODaSC_OHT_Native", "ODaSC (Hoeffding)", False),
        ("BORB_Native", "BORB (ElasticNet)", False),
        ("ORB_OHT_Native", "ORB (Hoeffding)", False),
        ("Full_LR_B0", "Full-LR Frozen", False),
        ("B1a_Full_Hindsight", "Full Hindsight", False),
        ("B0_Frozen", "Frozen Host", True),
        ("B1b_4W_Oracle", "Window Oracle", True),
        ("Teacher_Only_Matched", "Teacher-Matched", True),
        ("B2_4W", "Causal-4W", True),
        ("Survival_PU", "Survival-PU", True),
        ("FixedCurve_PU", "FixedCurve-PU", True),
        ("Wait_90", "Wait-90", True),
    ]
    dep_cf = [mk for mk, _, _ in conf_methods if mk not in ("B1a_Full_Hindsight", "B1b_4W_Oracle")]
    dep_cf_lgb = [mk for mk, _, hl in conf_methods if mk not in ("B1a_Full_Hindsight", "B1b_4W_Oracle") and hl]
    cf_repos = ["conf_apache_hdfs", "conf_apache_mapreduce", "conf_apache_spark", "conf_apache_zeppelin", "conf_apache_zookeeper"]
    cands_cf_r = {rk: [conf_j["streams_mlp"][rk]["methods"][m]["full_stream"]["ap"] for m in dep_cf] for rk in cf_repos}
    cands_cf_u4 = [conf_j["umbrella4_means_ap"][m] for m in dep_cf]
    cands_cf_lgb = [conf_j["umbrella4_lightgbm_means_ap"][m] for m in dep_cf_lgb]

    for mk, mlbl, has_lgb in conf_methods:
        is_ora = mk in ("B1a_Full_Hindsight", "B1b_4W_Oracle")
        r_cells = []
        for idx_r, rk in enumerate(cf_repos, start=1):
            vr = conf_j["streams_mlp"][rk]["methods"][mk]["full_stream"]["ap"]
            r_cells.append(reg.num(t_lbl, mlbl, idx_r, "confirmatory_cohort_v3.json", ("streams_mlp", rk, "methods", mk, "full_stream", "ap"), ".4f", bold=(not is_ora and _is_best_fmt(vr, cands_cf_r[rk], ".4f"))))
        vu = conf_j["umbrella4_means_ap"][mk]
        c6 = reg.num(t_lbl, mlbl, 6, "confirmatory_cohort_v3.json", ("umbrella4_means_ap", mk), ".4f", bold=(not is_ora and _is_best_fmt(vu, cands_cf_u4, ".4f")))
        if has_lgb:
            vl = conf_j["umbrella4_lightgbm_means_ap"][mk]
            c7 = reg.num(t_lbl, mlbl, 7, "confirmatory_cohort_v3.json", ("umbrella4_lightgbm_means_ap", mk), ".4f", bold=(not is_ora and _is_best_fmt(vl, cands_cf_lgb, ".4f")))
        else:
            c7 = "--"
        rows_cf.append(f"{mlbl} & {' & '.join(r_cells)} & {c6} & {c7} \\\\")
        if mk == "Full_LR_B0":
            rows_cf.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Out-of-sample AP on the five confirmatory ApacheJIT repositories and four umbrella ecosystems. Asterisks mark Hadoop subprojects whose clean commits reside under the umbrella key (Table~S-7); bold marks the best deployable AP per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.5pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{HDFS*} & \\textbf{MR*} & \\textbf{Spark} & \\textbf{Zepp.} & \\textbf{ZooK.} & \\textbf{U4 MLP} & \\textbf{U4 LGB} \\\\\n\\midrule\n"
        + "\n".join(rows_cf)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: MR* = MapReduce*; Zepp.\\ = Zeppelin; ZooK.\\ = ZooKeeper; U4 MLP / U4 LGB = four-umbrella mean AP merging HDFS* and MR* under Hadoop on MLP / LightGBM.\\par}\n"
        "\\end{table}\n"
    )

    # 7. Table `tab:rq3_ablations` (Main 7.5pt 6-col) + `tab:or_rq3_full` (Online Resource 7-col with ROC-AUC)
    t_lbl = "tab:rq3_ablations"
    t_lbl_r3_full = "tab:or_rq3_full"
    rows_r3 = []
    rows_r3_full = []
    r3_list = [
        ("B1b_4W_Oracle", "Window Oracle (4W)"),
        ("Survival_PU_Default", "Survival-PU (Default Cohort)"),
        ("SPU_KaplanMeier", "Survival-PU (Kaplan--Meier Cure)"),
        ("SPU_WeibullCure", "Survival-PU (Weibull Mixture)"),
        ("SPU_Fixed90d", "FixedCurve-PU (Zero-Fit 90d)"),
        ("SPU_NoShrinkage_m0_0", "w/o Anchor Shrinkage ($m_0=0$)"),
        ("SPU_NoBisection", "w/o Monotone Bisection"),
        ("SPU_NoCovariateWeight", "w/o Covariate Density Ratio"),
        ("SPU_FrozenAnchorTeacher", "Frozen Pre-Stream Teacher"),
        ("Teacher_Only_Exact", "Teacher-Exact ($1\\times$)"),
        ("Teacher_Only_Matched", "Teacher-Matched ($2\\times$)"),
        ("SPU_WithLegacyGate", "Survival-PU (+Gate)"),
        ("B0_Frozen", "Frozen Host"),
        ("B2_4W", "Causal-4W (Censored)"),
    ]
    dep_r3 = [vk for vk, _ in r3_list if vk != "B1b_4W_Oracle"]
    cands_r3 = {met: [rq3_j["summary_means"][vk][met] for vk in dep_r3] for met in ("ap", "f1_th", "roc_auc", "recall_at_20pct_effort", "brier_cal")}
    for vk, r_lbl in r3_list:
        is_ora = (vk == "B1b_4W_Oracle")
        sm_v = rq3_j["summary_means"][vk]
        # Main 6-col table (AP, F1t, R@20, Br., Steps)
        m1 = reg.num(t_lbl, r_lbl, 1, "rq3_ablations_v3.json", ("summary_means", vk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["ap"], cands_r3["ap"], ".4f")))
        m2 = reg.num(t_lbl, r_lbl, 2, "rq3_ablations_v3.json", ("summary_means", vk, "f1_th"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["f1_th"], cands_r3["f1_th"], ".4f")))
        m3 = reg.num(t_lbl, r_lbl, 3, "rq3_ablations_v3.json", ("summary_means", vk, "recall_at_20pct_effort"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["recall_at_20pct_effort"], cands_r3["recall_at_20pct_effort"], ".4f")))
        m4 = reg.num(t_lbl, r_lbl, 4, "rq3_ablations_v3.json", ("summary_means", vk, "brier_cal"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["brier_cal"], cands_r3["brier_cal"], ".4f", lower_better=True)))
        m5 = reg.num(t_lbl, r_lbl, 5, "rq3_ablations_v3.json", ("summary_means", vk, "opt_steps"), ".1f")
        rows_r3.append(f"{r_lbl} & {m1} & {m2} & {m3} & {m4} & {m5} \\\\")
        # Online Resource full 7-col mirror (includes ROC-AUC)
        f1 = reg_supp.num(t_lbl_r3_full, r_lbl, 1, "rq3_ablations_v3.json", ("summary_means", vk, "ap"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["ap"], cands_r3["ap"], ".4f")))
        f2 = reg_supp.num(t_lbl_r3_full, r_lbl, 2, "rq3_ablations_v3.json", ("summary_means", vk, "f1_th"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["f1_th"], cands_r3["f1_th"], ".4f")))
        f3 = reg_supp.num(t_lbl_r3_full, r_lbl, 3, "rq3_ablations_v3.json", ("summary_means", vk, "roc_auc"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["roc_auc"], cands_r3["roc_auc"], ".4f")))
        f4 = reg_supp.num(t_lbl_r3_full, r_lbl, 4, "rq3_ablations_v3.json", ("summary_means", vk, "recall_at_20pct_effort"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["recall_at_20pct_effort"], cands_r3["recall_at_20pct_effort"], ".4f")))
        f5 = reg_supp.num(t_lbl_r3_full, r_lbl, 5, "rq3_ablations_v3.json", ("summary_means", vk, "brier_cal"), ".4f", bold=(not is_ora and _is_best_fmt(sm_v["brier_cal"], cands_r3["brier_cal"], ".4f", lower_better=True)))
        f6 = reg_supp.num(t_lbl_r3_full, r_lbl, 6, "rq3_ablations_v3.json", ("summary_means", vk, "opt_steps"), ".1f")
        rows_r3_full.append(f"{r_lbl} & {f1} & {f2} & {f3} & {f4} & {f5} & {f6} \\\\")
        if vk in ("B1b_4W_Oracle", "SPU_Fixed90d", "SPU_FrozenAnchorTeacher", "SPU_WithLegacyGate"):
            rows_r3.append("\\midrule")
            rows_r3_full.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{RQ3 maturity-estimator sensitivity, internal ablations, teacher controls, and confidence-gating ablation across ten representative streams (full metrics in Table~S-10). Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Variant / Control} & \\textbf{AP} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{R@20} & \\textbf{Br.$\\downarrow$} & \\textbf{Steps} \\\\\n\\midrule\n"
        + "\n".join(rows_r3)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: $F_{1\\mathrm{t}}$ = causally tuned $F_{1,\\mathrm{th}}$; R@20 = Recall@20\\%Effort; Br.$\\downarrow$ = post-Platt Brier loss; Steps = mean optimizer gradient steps per stream.\\par}\n"
        "\\end{table}\n"
    )
    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Complete 7-column RQ3 maturity-estimator sensitivity, component ablations, step-matched teacher controls, and legacy safety-gate ablation including ROC-AUC across 10 representative streams (5 seeds). Bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl_r3_full}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.4pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Variant / Control} & \\textbf{AP} & \\textbf{$F_{1,\\mathrm{th}}$} & \\textbf{ROC-AUC} & \\textbf{R@20\\%} & \\textbf{Brier$\\downarrow$} & \\textbf{Opt.\\ Steps} \\\\\n\\midrule\n"
        + "\n".join(rows_r3_full)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 8. Table `tab:rq4b_monitors` (Main 7.5pt 7-col)
    t_lbl = "tab:rq4b_monitors"
    rows_m4 = []
    m4_list = [
        ("DriftLens_IID_Static_95_CD1", "DriftLens: IID + Static ($p_{95}$, cd=1)"),
        ("DriftLens_IID_Rolling_95_CD1", "DriftLens: IID + Rolling ($p_{95}$, cd=1)"),
        ("DriftLens_Block_Static_95_CD1", "DriftLens: Block + Static ($p_{95}$, cd=1)"),
        ("DriftLens_Block_Rolling_95_CD1_V3", "DriftLens: Block + Rolling (Def., cd=1)"),
        ("DriftLens_N46_Ref_IID_Static_98_CD0", "Pote-Ref: IID + Static ($p_{98}$, cd=0)"),
        ("DriftLens_Block_Rolling_ClosedLoop", "DriftLens: Block + Rolling (Closed-Loop)"),
        ("RBF_MMD_Block_Rolling", "RBF-MMD: Block + Rolling ($p_{95}$, cd=1)"),
        ("PCA_KS_Block_Rolling", "PCA-KS: Block + Rolling ($p_{95}$, cd=1)"),
        ("Periodic_Every3W", "Periodic Fixed Schedule (Every $3W$)"),
    ]
    cands_far = [rq4b_j["configs"][ck]["stationary_false_alarm_rate"] for ck, _ in m4_list]
    cands_b2m = [rq4b_j["configs"][ck]["means_10streams"]["B2_4W"]["ap"] for ck, _ in m4_list]
    cands_spm = [rq4b_j["configs"][ck]["means_10streams"]["Survival_PU"]["ap"] for ck, _ in m4_list]
    for ck, r_lbl in m4_list:
        cfg_v = rq4b_j["configs"][ck]
        c1 = reg.num(t_lbl, r_lbl, 1, "rq4b_factorial_monitors_v3.json", ("configs", ck, "mean_alarm_rate_10streams"), ".1%")
        c2 = reg.num(t_lbl, r_lbl, 2, "rq4b_factorial_monitors_v3.json", ("configs", ck, "stationary_false_alarm_rate"), ".1%", bold=_is_best_fmt(cfg_v["stationary_false_alarm_rate"], cands_far, ".1%", lower_better=True))
        c3 = reg.num(t_lbl, r_lbl, 3, "rq4b_factorial_monitors_v3.json", ("configs", ck, "undetected_shift_rate"), ".1%")
        cond_d = cfg_v["conditional_shift_delay_commits_given_detected"]
        if cond_d is not None:
            c4 = reg.num(t_lbl, r_lbl, 4, "rq4b_factorial_monitors_v3.json", ("configs", ck, "conditional_shift_delay_commits_given_detected"), ".0f")
        else:
            c4 = "--"
        vb2 = cfg_v["means_10streams"]["B2_4W"]["ap"]
        vsp = cfg_v["means_10streams"]["Survival_PU"]["ap"]
        c5 = reg.num(t_lbl, r_lbl, 5, "rq4b_factorial_monitors_v3.json", ("configs", ck, "means_10streams", "B2_4W", "ap"), ".4f", bold=_is_best_fmt(vb2, cands_b2m, ".4f"))
        c6 = reg.num(t_lbl, r_lbl, 6, "rq4b_factorial_monitors_v3.json", ("configs", ck, "means_10streams", "Survival_PU", "ap"), ".4f", bold=_is_best_fmt(vsp, cands_spm, ".4f"))
        rows_m4.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} \\\\")
        if ck == "DriftLens_Block_Rolling_95_CD1_V3":
            rows_m4.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{RQ4b $2\\times 2$ DriftLens bootstrap and refresh factorial, Pote et al.\\ reference, alternative block monitors, and periodic retraining across ten representative streams. Bold marks the lowest null rate and highest AP per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.5pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Monitor Configuration} & \\textbf{Alm.} & \\textbf{Null} & \\textbf{Miss} & \\textbf{Dly} & \\textbf{C4W} & \\textbf{SPU} \\\\\n\\midrule\n"
        + "\n".join(rows_m4)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: Alm.\\ = mean triggered-window alarm rate; Null = validation-pool false-alarm rate; Miss = undetected shift rate (20 shift runs); Dly = mean conditional detection delay (commits); C4W / SPU = 10-stream mean AP for Causal-4W / Survival-PU.\\par}\n"
        "\\end{table}\n"
    )

    # 9. Table `tab:rq4f_llm` -> Online Resource 1 (Table S16)
    t_lbl = "tab:rq4f_llm"
    rows_llm = []
    llm_rows_spec = [
        ("LLM_ZeroShot_Causal", "DeepSeek Zero-Shot (Causal)", "ap", "f1", "roc_auc"),
        ("LLM_FewShot_Censored_1W", "DeepSeek 6-Shot (Censored 1W)", "ap_mean", "f1_mean", "roc_auc_mean"),
        ("LLM_FewShot_Oracle_Paired_1W", "DeepSeek 6-Shot (Paired Oracle 1W)", "ap_mean", "f1_mean", "roc_auc_mean"),
        ("LLM_FewShot_Mature_Causal", "DeepSeek 6-Shot (Causal Mature $\\ge 90$d)", "ap_mean", "f1_mean", "roc_auc_mean"),
        ("CodeBERT_B0_Frozen", "CodeBERT + LinSkip (Init Prefix, 0 Alarms)", "ap", "f1", "roc_auc"),
        ("LGB_B0_Frozen", "LightGBM SVD+Tab13 (Init Prefix, 0 Alarms)", "ap", "f1", "roc_auc"),
        ("Stream_Full_LR_Adapt", "Full-LR Adapt-4W (Tabular-13)", "ap", "f1", "roc_auc"),
        ("Stream_BORB_Native", "BORB ElasticNet (Tabular-13)", "ap", "f1", "roc_auc"),
    ]
    dep_llm = [t for t in llm_rows_spec if t[0] != "LLM_FewShot_Oracle_Paired_1W"]
    cands_llm = {
        (eco, idx_m): [llm_j["ecosystems"][eco]["regimes"][rk][mk_f] for rk, _, *mfs in dep_llm for idx_k, mk_f in enumerate(mfs) if idx_k == idx_m]
        for eco in ("qt", "openstack") for idx_m in range(3)
    }
    for rk, r_lbl, ap_k, f1_k, auc_k in llm_rows_spec:
        is_ora = (rk == "LLM_FewShot_Oracle_Paired_1W")
        vq = [llm_j["ecosystems"]["qt"]["regimes"][rk][k] for k in (ap_k, f1_k, auc_k)]
        vo = [llm_j["ecosystems"]["openstack"]["regimes"][rk][k] for k in (ap_k, f1_k, auc_k)]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "rq4f_llm_audit_v3.json", ("ecosystems", "qt", "regimes", rk, ap_k), ".4f", bold=(not is_ora and _is_best_fmt(vq[0], cands_llm[("qt", 0)], ".4f")))
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "rq4f_llm_audit_v3.json", ("ecosystems", "qt", "regimes", rk, f1_k), ".4f", bold=(not is_ora and _is_best_fmt(vq[1], cands_llm[("qt", 1)], ".4f")))
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "rq4f_llm_audit_v3.json", ("ecosystems", "qt", "regimes", rk, auc_k), ".4f", bold=(not is_ora and _is_best_fmt(vq[2], cands_llm[("qt", 2)], ".4f")))
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "rq4f_llm_audit_v3.json", ("ecosystems", "openstack", "regimes", rk, ap_k), ".4f", bold=(not is_ora and _is_best_fmt(vo[0], cands_llm[("openstack", 0)], ".4f")))
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "rq4f_llm_audit_v3.json", ("ecosystems", "openstack", "regimes", rk, f1_k), ".4f", bold=(not is_ora and _is_best_fmt(vo[1], cands_llm[("openstack", 1)], ".4f")))
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "rq4f_llm_audit_v3.json", ("ecosystems", "openstack", "regimes", rk, auc_k), ".4f", bold=(not is_ora and _is_best_fmt(vo[2], cands_llm[("openstack", 2)], ".4f")))
        rows_llm.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} \\\\")
        if rk == "LLM_FewShot_Mature_Causal":
            rows_llm.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{RQ4e audited Code-LLM evaluation (deepseek-flash, 6,400 logical evaluations across 6,819 HTTP attempts on 320-commit Qt and OpenStack slices) versus domain-trained JIT predictors (initial frozen checkpoints for alarm-triggered hosts, as 0 alarms fired before window 2). Bold marks the best causal value per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Model / Prompt Regime} & \\textbf{Qt AP} & \\textbf{Qt $F_1$} & \\textbf{Qt AUC} & \\textbf{OS AP} & \\textbf{OS $F_1$} & \\textbf{OS AUC} \\\\\n\\midrule\n"
        + "\n".join(rows_llm)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 10. Table `tab:rq5_runtime` -> Online Resource 1 (Table S17)
    t_lbl = "tab:rq5_runtime"
    rows_rt = []
    rt_list = [
        ("B0_Frozen", "Frozen MLP"),
        ("B2_4W", "Causal-4W (MLP)"),
        ("Wait_90", "Wait-90 (MLP)"),
        ("Pure_nnPU", "Pure-nnPU (MLP)"),
        ("Teacher_Only_Matched", "Teacher-Matched (MLP)"),
        ("Survival_PU", "Survival-PU (MLP)"),
        ("BORB_Native", "BORB ElasticNet (Track 2)"),
        ("ORB_OHT_Native", "ORB Hoeffding (Track 2)"),
        ("ODaSC_OHT_Native", "ODaSC Hoeffding (Track 2)"),
    ]
    cands_ste2e = [derived["runtime_summary"]["benchmark_b_mlp_27streams_mean"][mk]["consistent_end_to_end_ms"] for mk, _ in rt_list]
    for mk, r_lbl in rt_list:
        p_base = ("runtime_summary", "benchmark_b_mlp_27streams_mean", mk)
        ve2e = derived["runtime_summary"]["benchmark_b_mlp_27streams_mean"][mk]["consistent_end_to_end_ms"]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "manuscript_derived_aggregates_v3.json", p_base + ("stream_predict_and_embed_ms",), ".1f")
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "manuscript_derived_aggregates_v3.json", p_base + ("drift_monitor_ms",), ".1f")
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "manuscript_derived_aggregates_v3.json", p_base + ("adapt_only_wall_ms",), ".1f")
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "manuscript_derived_aggregates_v3.json", p_base + ("calibration_ms",), ".1f")
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "manuscript_derived_aggregates_v3.json", p_base + ("consistent_end_to_end_ms",), ".1f", bold=_is_best_fmt(ve2e, cands_ste2e, ".1f", lower_better=True))
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "manuscript_derived_aggregates_v3.json", p_base + ("incremental_suite_end_to_end_ms",), ".1f")
        rows_rt.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} \\\\")
        if mk == "Survival_PU":
            rows_rt.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{RQ5 phase-decomposed wall-clock runtime telemetry on pre-extracted tabular features (milliseconds per stream, mean across 27 natural Benchmark B streams), separating standalone execution from incremental shared-backbone suite cost. Bold marks the lowest standalone runtime.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.4pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{Pred.} & \\textbf{Mon.} & \\textbf{Adapt} & \\textbf{Calib.} & \\textbf{Stand.\\ E2E} & \\textbf{Incr.\\ E2E} \\\\\n\\midrule\n"
        + "\n".join(rows_rt)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 11. Table `tab:rq4a_kappa` (Main 7.5pt 8-col) + `tab:or_rq4a_kappa_full` (Online Resource 11-col w/ Wait-90)
    t_lbl = "tab:rq4a_kappa"
    t_lbl_ka_full = "tab:or_rq4a_kappa_full"
    rows_ka = []
    rows_ka_full = []
    for kk, r_lbl_short, r_lbl_full in [
        ("oracle_post_t0", "Oracle Post-$t_0$", "Oracle Post-$t_0$ ($B_{1b}^* = B_2$)"),
        ("0.00", "$\\kappa=0.00\\times$ (Zero)", "$\\kappa = 0.00\\times$ (Zero Delay)"),
        ("0.25", "$\\kappa=0.25\\times$ (Fast)", "$\\kappa = 0.25\\times$ (Fast Fixes)"),
        ("0.50", "$\\kappa=0.50\\times$", "$\\kappa = 0.50\\times$"),
        ("1.00", "$\\kappa=1.00\\times$ (Emp.)", "$\\kappa = 1.00\\times$ (Emp.\\ SZZ)"),
        ("1.50", "$\\kappa=1.50\\times$", "$\\kappa = 1.50\\times$"),
        ("2.00", "$\\kappa=2.00\\times$", "$\\kappa = 2.00\\times$"),
        ("3.00", "$\\kappa=3.00\\times$ (Heavy)", "$\\kappa = 3.00\\times$ (Heavy Lat.)"),
    ]:
        p_b = ("results_by_kappa", kk, "means")
        km = rq4a_j["results_by_kappa"][kk]["means"]
        ap_cands_main = [km[m]["ap"] for m in ("B2_4W", "Survival_PU")]
        f1_cands_main = [km[m]["f1"] for m in ("B2_4W", "Survival_PU")]
        f1th_cands_main = [km[m]["f1_th"] for m in ("B2_4W", "Survival_PU")]
        ap_cands_all = [km[m]["ap"] for m in ("B0_Frozen", "B2_4W", "Wait_90", "Survival_PU")]
        f1_cands_all = [km[m]["f1"] for m in ("B2_4W", "Wait_90", "Survival_PU")]
        f1th_cands_all = [km[m]["f1_th"] for m in ("B2_4W", "Wait_90", "Survival_PU")]
        is_ora_row = (kk == "oracle_post_t0")
        # Main 8-col table: Ora., C4W AP, SPU AP, C4W F1, SPU F1, C4W F1t, SPU F1t
        m1 = reg.num(t_lbl, r_lbl_short, 1, "rq4a_latency_intervention_v3.json", p_b + ("B1b_4W_Oracle", "ap"), ".4f")
        m2 = reg.num(t_lbl, r_lbl_short, 2, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "ap"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["ap"], ap_cands_main, ".4f")))
        m3 = reg.num(t_lbl, r_lbl_short, 3, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "ap"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["ap"], ap_cands_main, ".4f")))
        m4 = reg.num(t_lbl, r_lbl_short, 4, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "f1"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["f1"], f1_cands_main, ".4f")))
        m5 = reg.num(t_lbl, r_lbl_short, 5, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "f1"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["f1"], f1_cands_main, ".4f")))
        m6 = reg.num(t_lbl, r_lbl_short, 6, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "f1_th"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["f1_th"], f1th_cands_main, ".4f")))
        m7 = reg.num(t_lbl, r_lbl_short, 7, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "f1_th"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["f1_th"], f1th_cands_main, ".4f")))
        rows_ka.append(f"{r_lbl_short} & {m1} & {m2} & {m3} & {m4} & {m5} & {m6} & {m7} \\\\")
        # Online Resource full 11-col mirror (includes Wait-90)
        f1 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 1, "rq4a_latency_intervention_v3.json", p_b + ("B1b_4W_Oracle", "ap"), ".4f")
        f2 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 2, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "ap"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["ap"], ap_cands_all, ".4f")))
        f3 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 3, "rq4a_latency_intervention_v3.json", p_b + ("Wait_90", "ap"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Wait_90"]["ap"], ap_cands_all, ".4f")))
        f4 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 4, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "ap"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["ap"], ap_cands_all, ".4f")))
        f5 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 5, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "f1"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["f1"], f1_cands_all, ".4f")))
        f6 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 6, "rq4a_latency_intervention_v3.json", p_b + ("Wait_90", "f1"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Wait_90"]["f1"], f1_cands_all, ".4f")))
        f7 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 7, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "f1"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["f1"], f1_cands_all, ".4f")))
        f8 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 8, "rq4a_latency_intervention_v3.json", p_b + ("B2_4W", "f1_th"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["B2_4W"]["f1_th"], f1th_cands_all, ".4f")))
        f9 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 9, "rq4a_latency_intervention_v3.json", p_b + ("Wait_90", "f1_th"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Wait_90"]["f1_th"], f1th_cands_all, ".4f")))
        f10 = reg_supp.num(t_lbl_ka_full, r_lbl_full, 10, "rq4a_latency_intervention_v3.json", p_b + ("Survival_PU", "f1_th"), ".4f", bold=(not is_ora_row and _is_best_fmt(km["Survival_PU"]["f1_th"], f1th_cands_all, ".4f")))
        rows_ka_full.append(f"{r_lbl_full} & {f1} & {f2} & {f3} & {f4} & {f5} & {f6} & {f7} & {f8} & {f9} & {f10} \\\\")
        if kk == "oracle_post_t0":
            rows_ka.append("\\midrule")
            rows_ka_full.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{RQ4a future-only causal latency scaling across ten representative streams with frozen pre-$t_0$ supervision (Wait-90 and FixedCurve-PU mirrors in Tables~S-12--S-13). Bold marks the higher value between Causal-4W and Survival-PU per metric pair.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Latency Regime} & \\textbf{Ora.} & \\textbf{C4W AP} & \\textbf{SPU AP} & \\textbf{C4W $F_1$} & \\textbf{SPU $F_1$} & \\textbf{C4W $F_{1\\mathrm{t}}$} & \\textbf{SPU $F_{1\\mathrm{t}}$} \\\\\n\\midrule\n"
        + "\n".join(rows_ka)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: Ora.\\ = Window Oracle AP; C4W = Causal-4W; SPU = Survival-PU; $F_{1\\mathrm{t}}$ = causally tuned $F_{1,\\mathrm{th}}$.\\par}\n"
        "\\end{table}\n"
    )
    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Complete 11-column RQ4a future-only causal latency scaling including Wait-90 across 10 representative streams (5 seeds; Frozen Host AP invariant at $0.4872$). Bold marks the highest deployable value per metric block.}\n"
        f"\\label{{{t_lbl_ka_full}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.8pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Latency Regime} & \\textbf{Ora.} & \\textbf{C4W AP} & \\textbf{W90 AP} & \\textbf{SPU AP} & \\textbf{C4W $F_1$} & \\textbf{W90 $F_1$} & \\textbf{SPU $F_1$} & \\textbf{C4W $F_{1\\mathrm{t}}$} & \\textbf{W90 $F_{1\\mathrm{t}}$} & \\textbf{SPU $F_{1\\mathrm{t}}$} \\\\\n\\midrule\n"
        + "\n".join(rows_ka_full)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 11b. Table `tab:rq4a_fixedcurve` -> Online Resource 1 (Table S13)
    t_lbl = "tab:rq4a_fixedcurve"
    rows_fc = []
    for kk, r_lbl in [
        ("oracle_post_t0", "Oracle Post-$t_0$"),
        ("0.00", "$\\kappa = 0.00\\times$ (Zero Delay)"),
        ("0.25", "$\\kappa = 0.25\\times$ (Fast Fixes)"),
        ("0.50", "$\\kappa = 0.50\\times$"),
        ("1.00", "$\\kappa = 1.00\\times$ (Emp.\\ SZZ)"),
        ("1.50", "$\\kappa = 1.50\\times$"),
        ("2.00", "$\\kappa = 2.00\\times$"),
        ("3.00", "$\\kappa = 3.00\\times$ (Heavy Lat.)"),
    ]:
        p_b = ("results_by_kappa", kk, "means")
        km = rq4a_fc_j["results_by_kappa"][kk]["means"]
        fc_d = km["FixedCurve_PU"]
        sp_d = km["Survival_PU"]
        ap_pair = [fc_d["ap"], sp_d["ap"]]
        f1_pair = [fc_d["f1"], sp_d["f1"]]
        f1t_pair = [fc_d["f1_th"], sp_d["f1_th"]]
        br_pair = [fc_d["brier_cal"], sp_d["brier_cal"]]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "rq4a_fixedcurve_control_v3.json", p_b + ("FixedCurve_PU", "ap"), ".4f", bold=_is_best_fmt(fc_d["ap"], ap_pair, ".4f"))
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "rq4a_fixedcurve_control_v3.json", p_b + ("Survival_PU", "ap"), ".4f", bold=_is_best_fmt(sp_d["ap"], ap_pair, ".4f"))
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "rq4a_fixedcurve_control_v3.json", p_b + ("FixedCurve_PU", "f1"), ".4f", bold=_is_best_fmt(fc_d["f1"], f1_pair, ".4f"))
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "rq4a_fixedcurve_control_v3.json", p_b + ("Survival_PU", "f1"), ".4f", bold=_is_best_fmt(sp_d["f1"], f1_pair, ".4f"))
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "rq4a_fixedcurve_control_v3.json", p_b + ("FixedCurve_PU", "f1_th"), ".4f", bold=_is_best_fmt(fc_d["f1_th"], f1t_pair, ".4f"))
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "rq4a_fixedcurve_control_v3.json", p_b + ("Survival_PU", "f1_th"), ".4f", bold=_is_best_fmt(sp_d["f1_th"], f1t_pair, ".4f"))
        c7 = reg_supp.num(t_lbl, r_lbl, 7, "rq4a_fixedcurve_control_v3.json", p_b + ("FixedCurve_PU", "brier_cal"), ".4f", bold=_is_best_fmt(fc_d["brier_cal"], br_pair, ".4f", lower_better=True))
        c8 = reg_supp.num(t_lbl, r_lbl, 8, "rq4a_fixedcurve_control_v3.json", p_b + ("Survival_PU", "brier_cal"), ".4f", bold=_is_best_fmt(sp_d["brier_cal"], br_pair, ".4f", lower_better=True))
        rows_fc.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} & {c8} \\\\")
        if kk == "oracle_post_t0":
            rows_fc.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{RQ4a Stage-1 maturity-misspecification control: zero-fit 90-day FixedCurve-PU (Fix) versus adaptive cohort-incidence Survival-PU (SPU) across identical future-only latency regimes (10 streams, 5 seeds; exact parity with Table~S-12 on SPU). Bold marks the better estimator per metric (lower is better for post-Platt Brier loss).}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Latency Regime} & \\textbf{Fix AP} & \\textbf{SPU AP} & \\textbf{Fix $F_1$} & \\textbf{SPU $F_1$} & \\textbf{Fix $F_{1\\mathrm{t}}$} & \\textbf{SPU $F_{1\\mathrm{t}}$} & \\textbf{Fix Br.$\\downarrow$} & \\textbf{SPU Br.$\\downarrow$} \\\\\n\\midrule\n"
        + "\n".join(rows_fc)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 12. Table `tab:rq4d_szz` -> Online Resource 1 (Table S15)
    t_lbl = "tab:rq4d_szz"
    rows_szz = []
    szz_ms = ["B0_Frozen", "B2_4W", "Wait_90", "BORB_Native", "ORB_OHT_Native", "Survival_PU"]
    for ck, r_lbl in [
        ("TrainRef__EvalRef", "Train: PR-SZZ Ref / Eval: PR-SZZ Ref"),
        ("TrainPert__EvalRef", "Train: Noisy SZZ / Eval: PR-SZZ Ref"),
        ("TrainRef__EvalPert", "Train: PR-SZZ Ref / Eval: Noisy SZZ"),
        ("TrainPert__EvalPert", "Train: Noisy SZZ / Eval: Noisy SZZ"),
    ]:
        p_s = ("regimes", "Pop_N60_Indep_PermFN", "cells_mean_ap", ck)
        row_d = rq4d_j["regimes"]["Pop_N60_Indep_PermFN"]["cells_mean_ap"][ck]
        row_cands = [row_d[m] for m in szz_ms]
        cells_s = [
            reg_supp.num(t_lbl, r_lbl, idx_m, "rq4d_szz_perturbation_v3.json", p_s + (m,), ".4f", bold=_is_best_fmt(row_d[m], row_cands, ".4f"))
            for idx_m, m in enumerate(szz_ms, start=1)
        ]
        rows_szz.append(f"{r_lbl} & {' & '.join(cells_s)} \\\\")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{RQ4c $2\\times 2$ frozen-prediction semi-synthetic SZZ perturbation across five Bludau \\& Pretschner PR-SZZ streams ($\\alpha_{\\mathrm{FP}}=8.42\\%,\\;\\beta_{\\mathrm{FN}}=47.15\\%$, independent permanent FN noise; AP). Bold marks the highest deployable AP in each row.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.4pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Train / Evaluation Cell} & \\textbf{Frozen} & \\textbf{Causal-4W} & \\textbf{Wait-90} & \\textbf{BORB} & \\textbf{ORB} & \\textbf{Survival-PU} \\\\\n\\midrule\n"
        + "\n".join(rows_szz)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 13. Table `tab:rq5_forensics` -> Online Resource 1 (Table S18)
    t_lbl = "tab:rq5_forensics"
    rows_for = []
    for idx_f, fcase in enumerate(for_j["ecosystems"]["qt"]["rescues"]):
        sha8 = fcase["inducing_sha"][:8]
        fsha8 = fcase["fix_sha"][:8]
        r_lbl = f"\\texttt{{{fcase['repo']}:{sha8}}} $\\to$ \\texttt{{{fsha8}}}"
        p_f = ("ecosystems", "qt", "rescues", str(idx_f))
        sc_d = fcase["scores"]
        dep_sc = [sc_d[m] for m in ("B0_Frozen", "B2_4W", "Wait_90", "BORB_Native", "Survival_PU")]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "rq5_natural_forensics_v3.json", p_f + ("latency_days",), ".1f")
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "rq5_natural_forensics_v3.json", p_f + ("scores", "B0_Frozen"), ".4f", bold=_is_best_fmt(sc_d["B0_Frozen"], dep_sc, ".4f"))
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "rq5_natural_forensics_v3.json", p_f + ("scores", "B1b_4W_Oracle"), ".4f")
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "rq5_natural_forensics_v3.json", p_f + ("scores", "B2_4W"), ".4f", bold=_is_best_fmt(sc_d["B2_4W"], dep_sc, ".4f"))
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "rq5_natural_forensics_v3.json", p_f + ("scores", "Wait_90"), ".4f", bold=_is_best_fmt(sc_d["Wait_90"], dep_sc, ".4f"))
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "rq5_natural_forensics_v3.json", p_f + ("scores", "BORB_Native"), ".4f", bold=_is_best_fmt(sc_d["BORB_Native"], dep_sc, ".4f"))
        c7 = reg_supp.num(t_lbl, r_lbl, 7, "rq5_natural_forensics_v3.json", p_f + ("scores", "Survival_PU"), ".4f", bold=_is_best_fmt(sc_d["Survival_PU"], dep_sc, ".4f"))
        rows_for.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} \\\\")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{RQ5 pre-declared SZZ-linked commit illustrations on natural Qt (CodeBERT with frozen linear skip; selected under pre-declared forensic rules requiring high-confidence McIntosh--Kamei SZZ links; Alarm \\#1 at commit 9,900 had 88.19\\% 4W censoring). Bold marks the highest deployable predicted risk per commit.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Repo:Inducing $\\to$ Fix} & \\textbf{Lag (d)} & \\textbf{Frozen} & \\textbf{Oracle} & \\textbf{Causal-4W} & \\textbf{Wait-90} & \\textbf{BORB} & \\textbf{Survival-PU} \\\\\n\\midrule\n"
        + "\n".join(rows_for)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 14. Table `tab:rq4a_posthoc_b2` (Main 7.5pt 8-col) + `tab:or_rq4a_posthoc_full` (Online Resource 10-col w/ ECE_raw & Slope)
    t_lbl = "tab:rq4a_posthoc_b2"
    t_lbl_ph_full = "tab:or_rq4a_posthoc_full"
    rows_ph = []
    rows_ph_full = []
    ph_specs = [
        ("0.00", "0\\times"),
        ("1.00", "1\\times"),
        ("3.00", "3\\times"),
    ]
    ph_methods = [
        ("B2_4W", "Causal-4W", "Causal-4W"),
        ("B2_4W_PostHocLogit", "+PostHoc Logit", "Causal-4W + PostHoc Logit"),
        ("B2_4W_PostHocRatio", "+PostHoc Ratio", "Causal-4W + PostHoc Ratio"),
        ("FixedCurve_PU", "FixedCurve-PU", "FixedCurve-PU"),
        ("Survival_PU", "Survival-PU", "Survival-PU"),
    ]
    for kk, k_tex in ph_specs:
        kdict = rev_j["audit_d_calibration_and_posthoc"]["posthoc_b2_by_kappa"][kk]["means"]
        cands_ap = [kdict[m]["ap"] for m, _, _ in ph_methods]
        cands_wap = [kdict[m]["win_ap"] for m, _, _ in ph_methods]
        cands_f1 = [kdict[m]["f1"] for m, _, _ in ph_methods]
        cands_f1t = [kdict[m]["f1_th"] for m, _, _ in ph_methods]
        cands_brr = [kdict[m]["brier_raw"] for m, _, _ in ph_methods]
        cands_ecer = [kdict[m]["ece_raw"] for m, _, _ in ph_methods]
        cands_brc = [kdict[m]["brier_cal"] for m, _, _ in ph_methods]
        cands_ecec = [kdict[m]["ece_cal"] for m, _, _ in ph_methods]
        for mk, mlbl_s, mlbl_f in ph_methods:
            r_lbl_s = f"${k_tex}$: {mlbl_s}"
            r_lbl_f = f"$\\kappa={k_tex}$: {mlbl_f}"
            pb = ("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", kk, "means", mk)
            md = kdict[mk]
            # Main 8-col: AP, WinAP, F1, F1t, Br_raw, Br_90d, ECE_90d
            m1 = reg.num(t_lbl, r_lbl_s, 1, "reviewer_mechanism_audits_v3.json", pb + ("ap",), ".4f", bold=_is_best_fmt(md["ap"], cands_ap, ".4f"))
            m2 = reg.num(t_lbl, r_lbl_s, 2, "reviewer_mechanism_audits_v3.json", pb + ("win_ap",), ".4f", bold=_is_best_fmt(md["win_ap"], cands_wap, ".4f"))
            m3 = reg.num(t_lbl, r_lbl_s, 3, "reviewer_mechanism_audits_v3.json", pb + ("f1",), ".4f", bold=_is_best_fmt(md["f1"], cands_f1, ".4f"))
            m4 = reg.num(t_lbl, r_lbl_s, 4, "reviewer_mechanism_audits_v3.json", pb + ("f1_th",), ".4f", bold=_is_best_fmt(md["f1_th"], cands_f1t, ".4f"))
            m5 = reg.num(t_lbl, r_lbl_s, 5, "reviewer_mechanism_audits_v3.json", pb + ("brier_raw",), ".4f", bold=_is_best_fmt(md["brier_raw"], cands_brr, ".4f", lower_better=True))
            m6 = reg.num(t_lbl, r_lbl_s, 6, "reviewer_mechanism_audits_v3.json", pb + ("brier_cal",), ".4f", bold=_is_best_fmt(md["brier_cal"], cands_brc, ".4f", lower_better=True))
            m7 = reg.num(t_lbl, r_lbl_s, 7, "reviewer_mechanism_audits_v3.json", pb + ("ece_cal",), ".4f", bold=_is_best_fmt(md["ece_cal"], cands_ecec, ".4f", lower_better=True))
            rows_ph.append(f"{r_lbl_s} & {m1} & {m2} & {m3} & {m4} & {m5} & {m6} & {m7} \\\\")
            # Online Resource full 10-col (adds ECE_raw & Calib Slope)
            f1 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 1, "reviewer_mechanism_audits_v3.json", pb + ("ap",), ".4f", bold=_is_best_fmt(md["ap"], cands_ap, ".4f"))
            f2 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 2, "reviewer_mechanism_audits_v3.json", pb + ("win_ap",), ".4f", bold=_is_best_fmt(md["win_ap"], cands_wap, ".4f"))
            f3 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 3, "reviewer_mechanism_audits_v3.json", pb + ("f1",), ".4f", bold=_is_best_fmt(md["f1"], cands_f1, ".4f"))
            f4 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 4, "reviewer_mechanism_audits_v3.json", pb + ("f1_th",), ".4f", bold=_is_best_fmt(md["f1_th"], cands_f1t, ".4f"))
            f5 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 5, "reviewer_mechanism_audits_v3.json", pb + ("brier_raw",), ".4f", bold=_is_best_fmt(md["brier_raw"], cands_brr, ".4f", lower_better=True))
            f6 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 6, "reviewer_mechanism_audits_v3.json", pb + ("ece_raw",), ".4f", bold=_is_best_fmt(md["ece_raw"], cands_ecer, ".4f", lower_better=True))
            f7 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 7, "reviewer_mechanism_audits_v3.json", pb + ("brier_cal",), ".4f", bold=_is_best_fmt(md["brier_cal"], cands_brc, ".4f", lower_better=True))
            f8 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 8, "reviewer_mechanism_audits_v3.json", pb + ("ece_cal",), ".4f", bold=_is_best_fmt(md["ece_cal"], cands_ecec, ".4f", lower_better=True))
            f9 = reg_supp.num(t_lbl_ph_full, r_lbl_f, 9, "reviewer_mechanism_audits_v3.json", pb + ("cal_slope",), ".3f")
            rows_ph_full.append(f"{r_lbl_f} & {f1} & {f2} & {f3} & {f4} & {f5} & {f6} & {f7} & {f8} & {f9} \\\\")
        if kk != "3.00":
            rows_ph.append("\\midrule")
            rows_ph_full.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{RQ4a raw versus shared 90-day Platt calibration and post-hoc Causal-4W controls across ten representative streams (raw ECE and slope in Table~S-14). Bold marks the best value per $\\kappa$ block.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Regime \\& Method} & \\textbf{AP} & \\textbf{WinAP} & \\textbf{$F_1$} & \\textbf{$F_{1\\mathrm{t}}$} & \\textbf{Br$_{\\mathrm{raw}}\\downarrow$} & \\textbf{Br$_{90}\\downarrow$} & \\textbf{ECE$_{90}\\downarrow$} \\\\\n\\midrule\n"
        + "\n".join(rows_ph)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: WinAP = unweighted 400-commit per-window AP; $F_{1\\mathrm{t}}$ = tuned $F_{1,\\mathrm{th}}$; Br$_{\\mathrm{raw}}$ = raw Brier loss; Br$_{90}$ / ECE$_{90}$ = 90-day Platt Brier / ECE.\\par}\n"
        "\\end{table}\n"
    )
    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Complete 10-column RQ4a symmetrical raw versus 90-day Platt calibration and post-hoc Causal-4W controls including raw ECE and 90-day calibration slope (10 streams, 5 seeds). Bold marks the best value per $\\kappa$ block.}\n"
        f"\\label{{{t_lbl_ph_full}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.8pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Regime \\& Method} & \\textbf{AP} & \\textbf{WinAP} & \\textbf{$F_1$} & \\textbf{$F_{1,\\mathrm{th}}$} & \\textbf{Br$_{\\mathrm{raw}}\\downarrow$} & \\textbf{ECE$_{\\mathrm{raw}}\\downarrow$} & \\textbf{Br$_{90\\mathrm{d}}\\downarrow$} & \\textbf{ECE$_{90\\mathrm{d}}\\downarrow$} & \\textbf{Sl$_{90\\mathrm{d}}$} \\\\\n\\midrule\n"
        + "\n".join(rows_ph_full)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 15. Table `tab:rq5_paired_budget` (Main 7.5pt 8-col)
    t_lbl = "tab:rq5_paired_budget"
    rows_pb = []
    pb_rows_spec = [
        ("qt_natural_25k", "att_cnn", "Qt: Att.-CNN"),
        ("qt_natural_25k", "codebert", "Qt: CodeBERT"),
        ("qt_natural_25k", "deepjit_cnn", "Qt: DeepJIT"),
        ("qt_natural_25k", "lightgbm", "Qt: LightGBM"),
        ("openstack_natural_13k", "att_cnn", "OS: Att.-CNN"),
        ("openstack_natural_13k", "codebert", "OS: CodeB. (0)"),
        ("openstack_natural_13k", "deepjit_cnn", "OS: DeepJIT"),
        ("openstack_natural_13k", "lightgbm", "OS: LightGBM"),
    ]
    for eco, h_short, r_lbl in pb_rows_spec:
        base_p = ("audit_g_paired_net_gain", eco, h_short)
        c1 = reg.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", base_p + ("fixed_th_050", "net_tp"), ".0f", plus=True)
        c2 = reg.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", base_p + ("fixed_th_050", "net_fp"), ".0f", plus=True)
        c3 = reg.num(t_lbl, r_lbl, 3, "reviewer_mechanism_audits_v3.json", base_p + ("matched_top10pct_commits", "net_tp"), ".0f", plus=True)
        c4 = reg.num(t_lbl, r_lbl, 4, "reviewer_mechanism_audits_v3.json", base_p + ("win_top10pct_commits", "net_tp"), ".0f", plus=True)
        c5 = reg.num(t_lbl, r_lbl, 5, "reviewer_mechanism_audits_v3.json", base_p + ("matched_20pct_effort_loc", "net_tp"), ".0f", plus=True)
        c6 = reg.num(t_lbl, r_lbl, 6, "reviewer_mechanism_audits_v3.json", base_p + ("win_20pct_effort_loc", "net_tp"), ".0f", plus=True)
        c7 = reg.num(t_lbl, r_lbl, 7, "reviewer_mechanism_audits_v3.json", base_p + ("multiseed_5seed_effort", "delta_win_r20_5seed"), ".4f", plus=True)
        rows_pb.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} \\\\")
        if eco == "qt_natural_25k" and h_short == "lightgbm":
            rows_pb.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{RQ5 paired defect trade-off of Survival-PU minus Causal-4W on a single representative seed under fixed-0.5 thresholding and equal-cap budgets, plus five-seed windowed effort recall delta.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Host} & \\textbf{TP$_{0.5}$} & \\textbf{FP$_{0.5}$} & \\textbf{Wh.10} & \\textbf{Wn.10} & \\textbf{Wh.20L} & \\textbf{Wn.20L} & \\textbf{$\\Delta$WnR$_{20}$} \\\\\n\\midrule\n"
        + "\n".join(rows_pb)
        + "\n\\bottomrule\n\\end{tabular*}\n"
        "\\vspace{2pt}\\par\\noindent{\\scriptsize\\raggedright Abbreviations: TP$_{0.5}$ / FP$_{0.5}$ = net true / false positives at fixed threshold 0.50; Wh.10 / Wn.10 = whole-stream / 400-commit per-window Top-10\\% commit-cap net TP; Wh.20L / Wn.20L = whole-stream / per-window 20\\%-LOC cap net TP; $\\Delta$WnR$_{20}$ = five-seed per-window 20\\%-LOC recall delta; OS: CodeB.\\ (0) = OpenStack CodeBERT (zero drift alarms fired on the representative seed).\\par}\n"
        "\\end{table}\n"
    )

    # 16. Table `tab:confirmatory_funnel` -> Online Resource 1 (Table S7)
    t_lbl = "tab:confirmatory_funnel"
    rows_fn = []
    fn_labels = [
        ("0", "\\texttt{apache/hadoop} (Monorepo Clean Only)"),
        ("1", "\\texttt{apache/hadoop-hdfs} (HDFS*)"),
        ("2", "\\texttt{apache/hadoop-mapreduce} (MapRed.*)"),
        ("3", "\\texttt{apache/spark} (Standalone)"),
        ("4", "\\texttt{apache/zeppelin} (Standalone)"),
        ("5", "\\texttt{apache/zookeeper} (Standalone)"),
    ]
    for idx_s, r_lbl in fn_labels:
        bp = ("audit_a_apachejit_funnel", "funnel_rows", idx_s)
        fr = rev_j["audit_a_apachejit_funnel"]["funnel_rows"][int(idx_s)]
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", bp + ("raw_N",), ".0f")
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", bp + ("raw_pos",), ".0f")
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "reviewer_mechanism_audits_v3.json", bp + ("raw_prev",), ".1%")
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "reviewer_mechanism_audits_v3.json", bp + ("post_cutoff_N",), ".0f")
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "reviewer_mechanism_audits_v3.json", bp + ("post_cutoff_prev",), ".1%")
        if "n_eval" in fr:
            c6 = reg_supp.num(t_lbl, r_lbl, 6, "reviewer_mechanism_audits_v3.json", bp + ("n_eval",), ".0f")
            c7 = reg_supp.num(t_lbl, r_lbl, 7, "reviewer_mechanism_audits_v3.json", bp + ("eval_prev",), ".1%")
        else:
            c6, c7 = "--", "--"
        rows_fn.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} \\\\")
        if idx_s in ("0", "2"):
            rows_fn.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Confirmatory ApacheJIT data-construction funnel from upstream \\texttt{apachejit\\_total.csv} (106,674 rows) through our chronological cutoff ($\\text{author\\_date} \\le \\text{2019-01-01}$, Cut) to evaluation splits. Upstream assigns 11,964 clean Hadoop monorepo commits to \\texttt{apache/hadoop}; the three standalone repositories have no monorepo split.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Upstream Project Key} & \\textbf{Raw $N$} & \\textbf{Raw Pos} & \\textbf{Raw Prev.} & \\textbf{Cut $N$} & \\textbf{Cut Prev.} & \\textbf{Eval $N$} & \\textbf{Eval Prev.} \\\\\n\\midrule\n"
        + "\n".join(rows_fn)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 17. Table `tab:followup_audit` -> Online Resource 1 (Table S8)
    t_lbl = "tab:followup_audit"
    rows_fu = []
    fu_specs = [
        ("Benchmark_A_Natural", "Bench A Natural (Qt, OS)"),
        ("Benchmark_B_Cabral19", "Bench B: Cabral ICSE'19 (14)"),
        ("Benchmark_B_ApacheJIT", "Bench B: ApacheJIT Dev (8)"),
        ("Benchmark_B_PRSZZ", "Bench B: PR-SZZ SQJ'26 (5)"),
        ("Confirmatory_ApacheJIT", "Phase C: Confirmatory (5)"),
    ]
    for gkey, r_lbl in fu_specs:
        bp = ("audit_b_followup_cutoffs", "cohort_followup_summary", gkey)
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", bp + ("n_streams",), ".0f")
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", bp + ("min_of_rem_min_days",), ".1f")
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "reviewer_mechanism_audits_v3.json", bp + ("median_of_rem_median_days",), ".1f")
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "reviewer_mechanism_audits_v3.json", bp + ("max_frac_lt_90d",), ".1%")
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "reviewer_mechanism_audits_v3.json", bp + ("max_frac_lt_180d",), ".1%")
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "reviewer_mechanism_audits_v3.json", bp + ("max_frac_lt_365d",), ".1%")
        rows_fu.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} \\\\")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Observable archive-horizon follow-up proxy audit across all 34 natural repository streams: minimum and median remaining window between evaluation commit timestamp and observable archive-horizon proxy $T_{\\mathrm{end}} = \\max(\\max_i t_i, \\max_{i:\\, t_i^{\\mathrm{fix}}<\\infty} t_i^{\\mathrm{fix}})$, plus maximum stream fraction with $<90$, $<180$, or $<365$ days of follow-up.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.2pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Stream Cohort} & \\textbf{Streams} & \\textbf{Min Rem (d)} & \\textbf{Med Rem (d)} & \\textbf{Max $<$90d} & \\textbf{Max $<$180d} & \\textbf{Max $<$365d} \\\\\n\\midrule\n"
        + "\n".join(rows_fu)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 17b. Table `tab:followup_sensitivity` -> Online Resource 1 (Table S9)
    t_lbl = "tab:followup_sensitivity"
    rows_fs = []
    fs_methods = ["Full_LR_B0", "B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Survival_PU"]
    for met_k, met_lbl in [("ap", "AP"), ("f1", "$F_1$")]:
        for eco, eco_lbl in [("qt_natural_25k", "Qt"), ("openstack_natural_13k", "OS")]:
            for rk, r_short in [("rem_ge_0d", "All"), ("rem_ge_180d", "$\\ge 180$d"), ("rem_ge_365d", "$\\ge 365$d")]:
                r_lbl = f"{met_lbl} --- {eco_lbl} ({r_short})"
                bp = ("audit_b_followup_cutoffs", "benchmark_a_followup_sensitivity_seed42", eco, rk)
                c_n = reg_supp.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", bp + ("n_eval_kept",), ".0f")
                c_pr = reg_supp.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", bp + ("eval_prevalence",), ".1%")
                cands = [rev_j["audit_b_followup_cutoffs"]["benchmark_a_followup_sensitivity_seed42"][eco][rk]["mean_4host"][m][met_k] for m in fs_methods if m != "B1b_4W_Oracle"]
                cells_m = [
                    reg_supp.num(
                        t_lbl, r_lbl, idx_m, "reviewer_mechanism_audits_v3.json", bp + ("mean_4host", mk, met_k), ".4f",
                        bold=(mk != "B1b_4W_Oracle" and _is_best_fmt(rev_j["audit_b_followup_cutoffs"]["benchmark_a_followup_sensitivity_seed42"][eco][rk]["mean_4host"][mk][met_k], cands, ".4f")),
                    )
                    for idx_m, mk in enumerate(fs_methods, start=3)
                ]
                rows_fs.append(f"{r_lbl} & {c_n} & {c_pr} & {' & '.join(cells_m)} \\\\")
        if met_k == "ap":
            rows_fs.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Benchmark A archive follow-up sensitivity (seed 42; 4-host mean AP and $F_1$ plus Full-LR Frozen) restricting evaluation to commits with $\\ge 180$ or $\\ge 365$ days before $T_{\\mathrm{end}}$. Bold marks the best deployable value per row.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrcccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Metric --- Slice} & \\textbf{$N$} & \\textbf{Prev.} & \\textbf{Full-LR} & \\textbf{Frozen} & \\textbf{Oracle} & \\textbf{Causal-4W} & \\textbf{Wait-90} & \\textbf{SPU} \\\\\n\\midrule\n"
        + "\n".join(rows_fs)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 18. Table `tab:mechanism_audit` -> Online Resource 1 (Table S11)
    t_lbl = "tab:mechanism_audit"
    rows_ma = []
    for skey, r_lbl in [
        ("age_lt_30d", "Unrev.\\ Young ($<30$d)"),
        ("age_30_90d", "Unrev.\\ Mid ($30$--$90$d)"),
        ("age_ge_90d", "Unrev.\\ Aged ($\\ge 90$d)"),
        ("all_unrevealed", "All Unrevealed ($y_{\\mathrm{obs}}=0$)"),
    ]:
        bp = ("audit_c_stage123_mechanisms", "age_stratified_unrevealed_only_audit", skey)
        c1 = reg_supp.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", bp + ("n_samples",), ".0f")
        c2 = reg_supp.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", bp + ("true_prevalence",), ".3f")
        c3 = reg_supp.num(t_lbl, r_lbl, 3, "reviewer_mechanism_audits_v3.json", bp + ("teacher_raw_brier",), ".4f")
        c4 = reg_supp.num(t_lbl, r_lbl, 4, "reviewer_mechanism_audits_v3.json", bp + ("teacher_dewt_brier",), ".4f")
        c5 = reg_supp.num(t_lbl, r_lbl, 5, "reviewer_mechanism_audits_v3.json", bp + ("q_fc_brier",), ".4f")
        c6 = reg_supp.num(t_lbl, r_lbl, 6, "reviewer_mechanism_audits_v3.json", bp + ("q_post_brier",), ".4f")
        c7 = reg_supp.num(t_lbl, r_lbl, 7, "reviewer_mechanism_audits_v3.json", bp + ("q_post_bias",), ".4f", plus=True)
        rows_ma.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} \\\\")
    rows_ma.append("\\midrule")
    r_wb = "All Pool (+Writeback Ctrl)"
    bp_all = ("audit_c_stage123_mechanisms", "age_stratified_soft_target_audit", "all_ages")
    c1 = reg_supp.num(t_lbl, r_wb, 1, "reviewer_mechanism_audits_v3.json", bp_all + ("n_samples",), ".0f")
    c2 = reg_supp.num(t_lbl, r_wb, 2, "reviewer_mechanism_audits_v3.json", bp_all + ("true_prevalence",), ".3f")
    c3 = reg_supp.num(t_lbl, r_wb, 3, "reviewer_mechanism_audits_v3.json", bp_all + ("teacher_raw_brier",), ".4f")
    c4 = reg_supp.num(t_lbl, r_wb, 4, "reviewer_mechanism_audits_v3.json", bp_all + ("teacher_wb_brier",), ".4f")
    c5 = reg_supp.num(t_lbl, r_wb, 5, "reviewer_mechanism_audits_v3.json", bp_all + ("q_fc_brier",), ".4f")
    c6 = reg_supp.num(t_lbl, r_wb, 6, "reviewer_mechanism_audits_v3.json", bp_all + ("q_post_brier",), ".4f")
    c7 = reg_supp.num(t_lbl, r_wb, 7, "reviewer_mechanism_audits_v3.json", bp_all + ("q_post_bias",), ".4f", plus=True)
    rows_ma.append(f"{r_wb} & {c1} & {c2} & {c3} & {c4} & {c5} & {c6} & {c7} \\\\")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Stage 1--3 intermediate audit across 62 fired DriftLens alarms on 10 representative streams (seed 42; 74,400 pool evaluations across 59,600 unique commits). Upper rows restrict to unrevealed commits ($y_{\\mathrm{obs}}=0$); the bottom row adds fair positive writeback ($y_{\\mathrm{obs}}=1 \\to 1.0$) to the de-weighted teacher.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{2.0pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lrrrrrrr@{{}}}}\n\\toprule\n"
        "\\textbf{Pool Stratum ($\\mathcal{B}_{\\mathrm{cal}}$)} & \\textbf{Evals} & \\textbf{True $\\pi^*$} & \\textbf{Raw Br.} & \\textbf{DeWt/WB Br.} & \\textbf{Fix $q_i$ Br.} & \\textbf{SPU $q_i$ Br.} & \\textbf{SPU Bias} \\\\\n\\midrule\n"
        + "\n".join(rows_ma)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 18b. Table `tab:window_ap_scale` (Main 7.5pt 6-col)
    t_lbl = "tab:window_ap_scale"
    rows_wa = []
    wa_methods = [
        ("B1b_4W_Oracle", "Window Oracle (4W)"),
        ("Full_LR_Adapt", "Full-LR Adapt-4W"),
        ("B2_4W", "Causal-4W (Censored)"),
        ("Wait_90", "Wait-90"),
        ("FixedCurve_PU", "FixedCurve-PU (90d)"),
        ("Survival_PU", "Survival-PU"),
        ("B0_Frozen", "Frozen Host"),
    ]
    wa_means = rev_j["audit_f_window_vs_whole_ap"]["K25_5seed_means"]
    dep_wa = [m for m, _ in wa_methods if m != "B1b_4W_Oracle"]
    for mk, r_lbl in wa_methods:
        c1 = reg.num(t_lbl, r_lbl, 1, "reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "whole_stream_ap", mk), ".4f", bold=(mk in dep_wa and _is_best_fmt(wa_means["whole_stream_ap"][mk], [wa_means["whole_stream_ap"][x] for x in dep_wa], ".4f")))
        c2 = reg.num(t_lbl, r_lbl, 2, "reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "win_ap_unweighted", mk), ".4f", bold=(mk in dep_wa and _is_best_fmt(wa_means["win_ap_unweighted"][mk], [wa_means["win_ap_unweighted"][x] for x in dep_wa], ".4f")))
        c3 = reg.num(t_lbl, r_lbl, 3, "reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "win_ap_pos_weighted", mk), ".4f", bold=(mk in dep_wa and _is_best_fmt(wa_means["win_ap_pos_weighted"][mk], [wa_means["win_ap_pos_weighted"][x] for x in dep_wa], ".4f")))
        c4 = reg.num(t_lbl, r_lbl, 4, "reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "scale_norm_zscore_whole_ap", mk), ".4f", bold=(mk in dep_wa and _is_best_fmt(wa_means["scale_norm_zscore_whole_ap"][mk], [wa_means["scale_norm_zscore_whole_ap"][x] for x in dep_wa], ".4f")))
        c5 = reg.num(t_lbl, r_lbl, 5, "reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "scale_norm_rank_whole_ap", mk), ".4f", bold=(mk in dep_wa and _is_best_fmt(wa_means["scale_norm_rank_whole_ap"][mk], [wa_means["scale_norm_rank_whole_ap"][x] for x in dep_wa], ".4f")))
        rows_wa.append(f"{r_lbl} & {c1} & {c2} & {c3} & {c4} & {c5} \\\\")
        if mk == "B1b_4W_Oracle":
            rows_wa.append("\\midrule")

    tex_tables.append(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Evaluation-granularity and cross-window score-comparability diagnostic across 25 natural Benchmark~B clusters over five seeds and 531 operational windows. Exploratory comparisons here are separate from Table~\\ref{tab:bench_b_holm}'s Holm--Bonferroni tests; bold marks the best deployable value per column.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.8pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method ($K{=}25$, 5 Seeds)} & \\textbf{Whole} & \\textbf{Win-Unw.} & \\textbf{Win-Pos} & \\textbf{Z-Norm} & \\textbf{Rank} \\\\\n\\midrule\n"
        + "\n".join(rows_wa)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # 19. Table `tab:shift_4streams` -> Online Resource 1 (Table S6)
    t_lbl = "tab:shift_4streams"
    rows_sh = []
    sh_ms = [
        ("B0_Frozen", "Frozen Host"),
        ("B1b_4W_Oracle", "Window Oracle (4W)"),
        ("B2_4W", "Causal-4W (Censored)"),
        ("Wait_90", "Wait-90"),
        ("FixedCurve_PU", "FixedCurve-PU"),
        ("Survival_PU", "Survival-PU"),
        ("ODaSC_Host", "ODaSC-Host"),
        ("BORB_Alarm", "BORB-Alarm"),
    ]
    sh_streams = [
        "abrupt_nova_zulip",
        "abrupt_ignite_cassandra",
        "abrupt_airflow_pulsar",
        "abrupt_deepjit_nova_neutron_pure13",
    ]
    for mk, r_lbl in sh_ms:
        cells = []
        col_i = 1
        for st in sh_streams:
            bp = ("audit_e_controlled_shifts", st, "methods", mk)
            cells.append(reg_supp.num(t_lbl, r_lbl, col_i, "reviewer_mechanism_audits_v3.json", bp + ("post_onset_ap",), ".4f"))
            col_i += 1
            cells.append(reg_supp.num(t_lbl, r_lbl, col_i, "reviewer_mechanism_audits_v3.json", bp + ("transient_1200_ap",), ".4f"))
            col_i += 1
        rows_sh.append(f"{r_lbl} & {' & '.join(cells)} \\\\")
        if mk == "B1b_4W_Oracle":
            rows_sh.append("\\midrule")

    supp_tables.append(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Per-stream breakdown across all four Benchmark~B controlled concatenated distribution-shift streams (MLP, 10 seeds): full post-onset AP (Post) and first 1,200 post-onset commits AP (1.2k). At the 1-hour splice boundary, 33.2\\% (Nova$\\to$Zulip), 60.0\\% (Ignite$\\to$Cassandra), 23.7\\% (Airflow$\\to$Pulsar), and 35.0\\% (OS Nova$\\to$Neutron) of revealed pre-onset fixes occur after the onset timestamp.}\n"
        f"\\label{{{t_lbl}}}\n\\fontsize{{7.5}}{{9.0}}\\selectfont\n\\setlength{{\\tabcolsep}}{{1.8pt}}\n\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}lcccccccc@{{}}}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{N$\\to$Z Post} & \\textbf{N$\\to$Z 1.2k} & \\textbf{I$\\to$C Post} & \\textbf{I$\\to$C 1.2k} & \\textbf{A$\\to$P Post} & \\textbf{A$\\to$P 1.2k} & \\textbf{OS Post} & \\textbf{OS 1.2k} \\\\\n\\midrule\n"
        + "\n".join(rows_sh)
        + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    )

    # Save exact cell registry JSONs (Main + Online Resource 1)
    reg_out = os.path.join(res_dir, "manuscript_cell_registry_v3.json")
    save_json_atomic(reg_out, reg.entries)
    reg_supp_out = os.path.join(res_dir, "online_resource_cell_registry.json")
    save_json_atomic(reg_supp_out, reg_supp.entries)

    os.makedirs(scratch_dir, exist_ok=True)
    tex_out = os.path.join(scratch_dir, "v3_tables_generated.tex")
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tex_tables))
    supp_tex_out = os.path.join(scratch_dir, "v3_online_resource_tables.tex")
    with open(supp_tex_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(supp_tables))

    # Build comprehensive digest markdown checking every pre-registered interpretation boundary!
    f3_k25 = {f"{x['method_a']}__vs__{x['method_b']}": x for x in ap_inf["family3_controls_K25"]}
    f3_k24 = {f"{x['method_a']}__vs__{x['method_b']}": x for x in ap_inf["sensitivity_K24_disjoint_from_benchmark_a"]["family3_controls_K24"]}

    spu_vs_fixed_k25 = f3_k25.get("Survival_PU__vs__FixedCurve_PU", {})
    spu_vs_teach_k25 = f3_k25.get("Survival_PU__vs__Teacher_Only_Matched", {})
    spu_vs_fixed_k24 = f3_k24.get("Survival_PU__vs__FixedCurve_PU", {})
    spu_vs_teach_k24 = f3_k24.get("Survival_PU__vs__Teacher_Only_Matched", {})

    digest_lines = [
        "# V3 Complete Empirical & Statistical Digest",
        "",
        f"- Main registered & verified LaTeX table cells: **{len(reg.entries)}** across **{len(tex_tables)}** empirical tables.",
        f"- Online Resource 1 registered & verified LaTeX table cells: **{len(reg_supp.entries)}** across **{len(supp_tables)}** supplementary tables.",
        f"- Corpus totals (natural repositories): Benchmark A = `{derived['corpus_totals']['benchmark_a_natural_commits']}`, "
        f"Benchmark B (27 natural) = `{derived['corpus_totals']['benchmark_b_natural_27streams_commits']}`, "
        f"Confirmatory (5 untouched) = `{derived['corpus_totals']['confirmatory_5repos_commits']}`, "
        f"Total = `{derived['corpus_totals']['total_natural_repository_commits']}`.",
        f"- Benchmark A Dev Skip Selection: chosen = `{ba_j['dev_skip_selection']['chosen_skip_mode']}` "
        f"(`frozen_lr={ba_j['dev_skip_selection']['mean_dev_ap_frozen_lr']:.4f}` vs `trainable_lr={ba_j['dev_skip_selection']['mean_dev_ap_trainable_lr']:.4f}`).",
        "",
        "## Pre-Registered Interpretation Boundary Checks (MUST ENFORCE VERBATIM!)",
        f"1. **Survival_PU vs. FixedCurve_PU**:",
        f"   - K=25: Delta-AP = `{spu_vs_fixed_k25.get('mean_diff', 0):.4f}`, W/T/L = `{spu_vs_fixed_k25.get('wins')}/{spu_vs_fixed_k25.get('ties')}/{spu_vs_fixed_k25.get('losses')}`, p_Holm = `{spu_vs_fixed_k25.get('p_holm', 1):.4e}`, sig_005 = `{spu_vs_fixed_k25.get('significant_005_holm')}`",
        f"   - K=24: Delta-AP = `{spu_vs_fixed_k24.get('mean_diff', 0):.4f}`, p_Holm = `{spu_vs_fixed_k24.get('p_holm', 1):.4e}`, sig_005 = `{spu_vs_fixed_k24.get('significant_005_holm')}`",
        f"2. **Survival_PU vs. Teacher_Only_Matched**:",
        f"   - K=25: Delta-AP = `{spu_vs_teach_k25.get('mean_diff', 0):.4f}`, W/T/L = `{spu_vs_teach_k25.get('wins')}/{spu_vs_teach_k25.get('ties')}/{spu_vs_teach_k25.get('losses')}`, p_Holm = `{spu_vs_teach_k25.get('p_holm', 1):.4e}`, sig_005 = `{spu_vs_teach_k25.get('significant_005_holm')}`",
        f"   - K=24: Delta-AP = `{spu_vs_teach_k24.get('mean_diff', 0):.4f}`, p_Holm = `{spu_vs_teach_k24.get('p_holm', 1):.4e}`, sig_005 = `{spu_vs_teach_k24.get('significant_005_holm')}`",
        "",
        "## Benchmark A 4-Host x 6-Stream Summary (AP)",
        json.dumps(derived["benchmark_a_4host_6stream_mean"], indent=2),
        "",
        "## Benchmark B K=25 Cluster Means (MLP AP)",
        json.dumps(ap_inf["K25_cluster_means"], indent=2),
        "",
        "## Benchmark B Family 1, 2, 3 Holm Results (K=25 MLP AP)",
        json.dumps({
            "family1": ap_inf["family1_paradox_K25"],
            "family2": ap_inf["family2_competitors_K25"],
            "family3": ap_inf["family3_controls_K25"],
        }, indent=2),
        "",
        "## Benchmark B Secondary Endpoints K=25 (F1_th, ROC-AUC, Brier_cal) & LightGBM K=25 AP",
        json.dumps({
            "f1_th_K25": bb_j["cluster_inference_mlp"]["secondary_f1_th"]["K25_cluster_means"],
            "brier_cal_K25": bb_j["cluster_inference_mlp"]["secondary_brier_cal"]["K25_cluster_means"],
            "lgb_ap_K25": bb_j["cluster_inference_lightgbm_ap"]["K25_cluster_means"],
            "lgb_family1_K25": bb_j["cluster_inference_lightgbm_ap"]["family1_paradox_K25"],
            "lgb_family3_K25": bb_j["cluster_inference_lightgbm_ap"]["family3_controls_K25"],
        }, indent=2),
        "",
        "## RQ3 Summary Means & K=7 Descriptive Contrasts",
        json.dumps({
            "summary_means": rq3_j["summary_means"],
            "descriptive_K7": rq3_j["descriptive_component_contrasts_natural_K7"],
        }, indent=2),
        "",
        "## RQ4a Latency Intervention Means",
        json.dumps({k: v["means"] for k, v in rq4a_j["results_by_kappa"].items()}, indent=2),
        "",
        "## RQ4b Monitor Configs Summary",
        json.dumps({
            k: {
                "alarm_rate": v["mean_alarm_rate_10streams"],
                "null_far": v["stationary_false_alarm_rate"],
                "undetected_shift": v["undetected_shift_rate"],
                "cond_delay_commits": v["conditional_shift_delay_commits_given_detected"],
                "means": v["means_10streams"],
            }
            for k, v in rq4b_j["configs"].items()
        }, indent=2),
        "",
        "## RQ4d Semi-Synthetic SZZ Regimes",
        json.dumps(rq4d_j["regimes"], indent=2),
        "",
        "## RQ4e Sensitivity Sweeps",
        json.dumps({
            "window_size_sweep": rq4e_j["window_size_sweep"],
            "tail_cap_sweep": rq4e_j["tail_cap_sweep"],
            "ht_shrinkage_m0_sweep": rq4e_j["ht_shrinkage_m0_sweep"],
        }, indent=2),
    ]
    digest_out = os.path.join(scratch_dir, "v3_summary_digest.md")
    with open(digest_out, "w", encoding="utf-8") as f:
        f.write("\n".join(digest_lines))

    return {
        "n_figures": 8,
        "n_main_tables": len(tex_tables),
        "n_supp_tables": len(supp_tables),
        "n_main_registered_cells": len(reg.entries),
        "n_supp_registered_cells": len(reg_supp.entries),
        "digest_path": digest_out,
        "tables_tex_path": tex_out,
        "supp_tables_tex_path": supp_tex_out,
        "registry_json_path": reg_out,
        "supp_registry_json_path": reg_supp_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.environ.get("CAUSALJIT_BASE_DIR", BASE_DIR))
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=os.path.join(os.environ.get("CAUSALJIT_BASE_DIR", BASE_DIR), "build_artifacts"),
    )
    parser.add_argument("--tables-only", action="store_true", help="Skip re-plotting figures and only regenerate tables + cell registries.")
    args = parser.parse_args()
    rep = generate_digest_and_tables(args.base_dir, args.scratch_dir, skip_plots=args.tables_only)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
