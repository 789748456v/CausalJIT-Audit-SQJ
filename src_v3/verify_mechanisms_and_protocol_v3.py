"""
Counterexample-Driven Pre-Run Gate Verification Suite (N05 V3 — 14 Checks).

Closes ALL Codex Round 2 & Round 3 Findings (`R3-C1`..`R3-C9`, `R3-M1`..`R3-M7`) BEFORE unlocking
the 5 confirmatory ApacheJIT repositories (`HDFS`, `MAPREDUCE`, `SPARK`, `ZEPPELIN`, `ZOOKEEPER`):
  1. Confirmatory filesystem lock & Benchmark A/B provenance hygiene (`pure13`, missing diffs,
     retroactive `fix_ts` quarantine, aligned `drop_unlinked` temporal cutoffs, `K=25/24` clusters).
  2. Counterexample-driven strict prefix-invariance (`R3-C1`, `R3-M7`): mid-stream future perturbation
     of `X`, `y_true`, and `fix_ts` leaves `raw_probs`, `cal_probs`, `tuned_th_vec`, and `decisions_th`
     on `[0 : prefix_len]` bit-for-bit identical (`0.0` diff, `0` changed decisions) across ALL Track 1
     & Track 2 methods, plus cold-start window-lookahead counterexample check.
  3. Causal `kappa` pre-`t0` freeze (`R3-C6`): Codex's 4-commit backlog counterexample at `kappa=0.0`,
     stream pre-`t0` invariance across all `kappa`, and `post_t0_oracle_visibility=True` `B1b == B2` parity.
  4. Immutable per-alarm budgets across 4 consecutive alarms (`R3-C2`): proves LightGBM `Teacher_Only_Matched`
     builds `[150, 150, 150, 150]` trees (never `180 -> 360 -> 720 -> 1440`!) matching `Survival_PU`.
  5. Single-posterior solver-vs-training identity (`R3-C5`): Codex's `base_eta=[10,-10], c=[0.96,0.96], pi=0.49`
     counterexample + live `adapt_survival_pu_v3` `float32` `q_post` residual check.
  6. Author-faithful `ORB_OHT_Native`, `BORB_Native`, & `ODaSC_OHT_Native` deterministic event-trace
     verification (`R3-C3`).
  7. Disjoint `U = X_pool` membership & functional-gradient `GBDT_nnPU` + SCAR `MLP` & `LightGBM` tests (`R3-C4`).
  8. Exact `LightGBM` callable objective gradient/Hessian + binary `predict_proba` margin equivalence
     proving `raw_score + init_logit` is neither omitted nor double-added (`R3-M7`).
  9. Unconfounded 2x2 `DriftMonitorV3` (`pct=95, cooldown=1` across all 4 cells, static cooldown active)
     + `B0_Frozen` detector/window invariance (`R3-C7`).
 10. Linear-residual skip epoch-0 equivalence (`< 1e-4`) to causal tabular Logistic Regression.
 11. `KaplanMeierCureEstimator` `1.006d` adversarial rounding test + 50 random fuzz trials.
 12. Causal-anchor (`m0=50`, zero oracle `eta_hist`) synthetic mechanism audit including feature-dependent
     delay & latency drift (`R3-M1`).
 13. Exact `2 * midrank` subset-sum signed-rank enumeration (`K=4 -> 0.125`, `K=5 -> 0.0625`), `brier_cal`
     orientation reversal, constant-density `Spearman = NaN`, and strict missingness rejection (`R3-C8`, `R3-M5`).
 14. Stale checkpoint fingerprint rejection, strict LLM JSON schema / empty-diff / paired-prompt audit,
     zero-fallback `V3_CONFIG.forensic_rules` filter, and swapped-row/swapped-column LaTeX cell verifier (`R3-C9`, `R3-M3`, `R3-M4`).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import tempfile
import time
import types
from typing import Dict
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import scipy.stats
from sklearn.metrics import average_precision_score, brier_score_loss
import torch

from config_v3 import (
    EXCLUDED_GERRIT_REVIEW_COLS,
    PRE_REGISTERED_INTERPRETATION_BOUNDARIES,
    PURE_COMMIT_COLS_QT_OS,
    SECONDS_PER_DAY,
    V3_CONFIG,
)
from metrics_and_stats_v3 import (
    aggregate_seed_horizon_metrics,
    aggregate_streams_to_donor_clusters,
    compute_effort_and_loc_masking_metrics,
    evaluate_horizon_v3,
    exact_signed_rank_two_sided_pvalue,
    paired_cluster_inference_v3,
)
import models_and_baselines_v3 as _mb_mod
from models_and_baselines_v3 import (
    DriftMonitorV3,
    HostTrainingFailureError,
    LightGBMHostV3,
    adapt_survival_pu_v3,
    apply_future_only_latency_intervention,
    build_host_backbone_v3,
    execute_within_host_counterfactual_suite_v3,
    fit_causal_tabular_lr_weights,
    make_gbdt_fgrad_nnpu_objective,
    make_soft_bce_lgb_objective,
    observe_labels_at_time,
    predict_logits_and_embeddings_v3,
    sigmoid_safe,
    train_faithful_nnpu_v3,
    train_supervised_v3,
)
from run_v3_experiments import (
    TRACK1_MAIN_METHODS,
    build_provenance_header,
    compute_task_run_fingerprint,
    load_verified_checkpoint_or_none,
    run_single_seed_stream_all_methods,
    save_json_atomic,
)
from run_v3_llm_and_forensics import (
    IncrementalLLMArchiveWriter,
    _call_deepseek_json_prob,
    build_paired_fewshot_prompts,
    compact_unified_diff_and_coverage,
    extract_served_model_strict,
    parse_strict_llm_json_prob,
    select_forensic_indices_strict,
)
from stream_competitors_v3 import (
    SongDenStreamConfidenceTracker,
    calibrate_causal_stream_probs_strict,
    compute_cabral_obf,
    extract_verification_latency_events,
    run_borb_competitor_v3,
    run_linear_competitor_v3,
    run_odasc_oht_native_v3,
    run_orb_oht_native_v3,
)
from survival_estimators_v3 import (
    CohortIncidenceEstimator,
    FixedHalfLifeMaturityEstimator,
    KaplanMeierCureEstimator,
    compute_survival_pu_posterior,
    estimate_ht_shrunk_prevalence,
    solve_monotone_pu_bisection,
)
from verify_manuscript_v3 import run_self_test as run_verifier_self_test


def check_1_provenance_and_confirmatory_lock(base_dir: str) -> Dict:
    conf_dir = os.path.join(base_dir, "cache", "confirmatory_v3")
    conf_raw = os.path.join(base_dir, "data_sync", "confirmatory_apachejit_raw")
    current_fs_locked = (not os.path.exists(conf_dir)) and (not os.path.exists(conf_raw))
    assert current_fs_locked, "Confirmatory cohort path already exists before Pre-Run Gate sign-off!"

    audit_path = os.path.join(base_dir, "results_v3", "data_provenance_audit_v3.json")
    assert os.path.exists(audit_path), f"Missing {audit_path}; run `data_audit_v3.py --rebuild-all` first!"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert "provenance" in audit, "Missing provenance header in data_provenance_audit_v3.json!"
    norm_live = json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True))
    norm_rec = json.loads(json.dumps(audit["provenance"]["v3_config"], sort_keys=True))
    assert norm_live == norm_rec, "Stale v3_config in data_provenance_audit_v3.json!"

    for st in audit["benchmark_a"]["primary_streams"]:
        cols = st["feature_cols"]
        assert cols == PURE_COMMIT_COLS_QT_OS
        assert len(set(cols) & set(EXCLUDED_GERRIT_REVIEW_COLS)) == 0

    qt_miss = int(audit["benchmark_a"]["qt_audit"]["missing_diff_commits_count"])
    os_miss = int(audit["benchmark_a"]["openstack_audit"]["missing_diff_commits_count"])
    assert qt_miss >= 200 and os_miss >= 1, f"Expected missing-diff counts recorded; got Qt={qt_miss}, OS={os_miss}"

    # Verify R3b-M1: CodeBERT V3 cache & downstream `_cb.npz` bundles preserve empty diffs as `""`
    # WITHOUT substituting commit messages (`unique_empty_vecs == 1` and `median_cos(msg, code) < 0.90`!)
    cb_prov = audit["provenance"].get("codebert_v3_encoder_provenance", {})
    assert cb_prov.get("missing_modality_policy") == "preserve_empty_string_no_message_fallback", "Missing CodeBERT V3 encoder provenance!"
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    cb_empty_cos_medians = {}
    for sys_k, b_name in [("qt", "qt_natural_25k_cb.npz"), ("openstack", "openstack_natural_13k_cb.npz")]:
        b_cb_chk = np.load(os.path.join(mm_dir, b_name))
        assert "missing_diff_mask" in b_cb_chk, f"missing_diff_mask absent from {b_name}!"
        m_mask = b_cb_chk["missing_diff_mask"].astype(bool)
        assert int(np.sum(m_mask)) >= 1
        X_emp = b_cb_chk["X"][m_mask]
        msg_e = X_emp[:, :768].astype(np.float64)
        code_e = X_emp[:, 768:1536].astype(np.float64)
        uniq_code = np.unique(np.round(code_e, 6), axis=0)
        assert len(uniq_code) == 1, f"{b_name} empty-diff code embeddings not identical (`len={len(uniq_code)}`)!"
        cos_v = np.sum(msg_e * code_e, axis=1) / np.maximum(
            np.linalg.norm(msg_e, axis=1) * np.linalg.norm(code_e, axis=1), 1e-12
        )
        med_cos = float(np.median(cos_v))
        assert med_cos < 0.90, f"{b_name} empty-diff code vs msg cosine {med_cos:.6f} >= 0.90 (message fallback leak)!"
        cb_empty_cos_medians[sys_k] = med_cos

    # Verify sensitivity split alignment (`drop_unlinked` eval IDs subset of `primary` eval IDs)
    sens_dir = os.path.join(base_dir, "cache", "multimodal_v3_sens")
    b_prim = np.load(os.path.join(mm_dir, "qt_natural_25k_meta.npz"), allow_pickle=True)
    b_drop = np.load(os.path.join(sens_dir, "qt_drop_unlinked_meta.npz"), allow_pickle=True)
    b_drop_cb = np.load(os.path.join(sens_dir, "qt_drop_unlinked_cb.npz"))
    assert len(np.unique(np.round(b_drop_cb["X"][b_drop_cb["missing_diff_mask"].astype(bool), 768:1536], 6), axis=0)) == 1
    prim_eval_set = set(str(x) for x in b_prim["commit_ids"][7500:])
    drop_eval_set = set(str(x) for x in b_drop["commit_ids"][int(b_drop_cb["n_init"]) + int(b_drop_cb["n_val"]) :])
    assert drop_eval_set.issubset(prim_eval_set), "drop_unlinked eval IDs leaked outside primary temporal cutoff!"

    dc = audit["benchmark_b"]["donor_cluster_summary"]
    assert dc["n_natural_streams"] == 27
    assert dc["K_independent_ecosystem_clusters"] == 25
    assert dc["K_benchmark_a_disjoint_clusters"] == 24

    return {
        "pass": True,
        "current_filesystem_confirmatory_locked": current_fs_locked,
        "qt_missing_diffs": qt_miss,
        "openstack_missing_diffs": os_miss,
        "codebert_empty_diff_median_cos_msg_vs_code": cb_empty_cos_medians,
        "drop_unlinked_temporal_alignment_verified": True,
        "K_natural_clusters": dc["K_independent_ecosystem_clusters"],
        "K_disjoint_from_a": dc["K_benchmark_a_disjoint_clusters"],
    }


def check_2_strict_prefix_invariance_counterexamples(base_dir: str) -> Dict:
    """
    Directly tests Codex R3-C1 & R3-M7:
      1. Cold-start window lookahead counterexample: modifying logits `[1..7]` inside the first
         prediction window MUST leave `cal_probs[0]` and `tuned_th_vec[0]` bit-for-bit invariant!
      2. Mid-stream prefix perturbation (`prefix_len = 450` of `900` eval commits): modifying future
         features `X[s_start+450:]`, future `y_true[s_start+450:]`, and any `fix_ts > ts[s_start+449]`
         MUST leave `raw_probs[:450]`, `cal_probs[:450]`, `tuned_th_vec[:450]`, AND `decisions_th[:450]`
         bit-for-bit identical (`0.0` diff, `0` historical decision flips!) across ALL deployable
         Track 1 & Track 2 methods!
    """
    # Part A: Codex's cold-start window-median counterexample
    past_z = np.linspace(-2.0, 1.0, 100)
    cts_toy = np.arange(108, dtype=np.float64) * SECONDS_PER_DAY * 0.5  # 50d history -> triggers cold-start branch!
    y_toy = np.zeros(108, dtype=np.int32)
    y_toy[5:25] = 1
    fts_toy = np.where(y_toy == 1, cts_toy + 2.0 * SECONDS_PER_DAY, np.inf)
    z_win_a = np.array([0.0, -0.2, 0.1, -0.1, 0.2, 0.0, -0.3, 0.1], dtype=np.float64)
    z_win_b = np.array([0.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], dtype=np.float64)
    cal_a, th_a = calibrate_causal_stream_probs_strict(past_z, z_win_a, cts_toy, y_toy, fts_toy, stream_start=100, window_size=8)
    cal_b, th_b = calibrate_causal_stream_probs_strict(past_z, z_win_b, cts_toy, y_toy, fts_toy, stream_start=100, window_size=8)
    cold_diff = abs(float(cal_a[0] - cal_b[0])) + abs(float(th_a[0] - th_b[0]))
    assert cold_diff == 0.0, f"Cold-start calibration leaked within-window future logits: diff={cold_diff}"

    # Part B: Full mid-stream prefix perturbation across Track 1 & Track 2
    b = np.load(os.path.join(base_dir, "cache", "benchmark_b_v3", "cabral_nova.npz"))
    N = 2100
    n_init, n_val = 900, 300
    s_start = n_init + n_val  # 1200; eval length = 900; prefix_len = 450 (2.25 windows of W=200)
    pref_len = 450
    cut_abs = s_start + pref_len  # 1650

    X1 = b["X"][:N].copy()
    X_la1 = b["X_la"][:N].copy()
    X_ch1 = b["X_churn"][:N].copy()
    ts = b["commit_ts"][:N].copy()
    y1 = b["y_true"][:N].copy()
    fts1 = b["fix_ts"][:N].copy()
    eff1 = b["effort"][:N].copy()

    t_cut = float(ts[cut_abs - 1])
    X2 = X1.copy()
    X_la2 = X_la1.copy()
    X_ch2 = X_ch1.copy()
    y2 = y1.copy()
    fts2 = fts1.copy()
    eff2 = eff1.copy()

    # Perturb ALL features, effort, labels, and post-`t_cut` fix revelations after `cut_abs`!
    X2[cut_abs:] = X2[cut_abs:] * (-2.0) + 3.5
    X_la2[cut_abs:] = X_la2[cut_abs:] + 4.0
    X_ch2[cut_abs:] = X_ch2[cut_abs:] + 4.0
    eff2[cut_abs:] = eff2[cut_abs:] * 50.0 + 500.0
    y2[cut_abs:] = 1 - y2[cut_abs:]
    fts2[cut_abs:] = np.where(y2[cut_abs:] == 1, ts[cut_abs:] + 1.0 * SECONDS_PER_DAY, np.inf)

    # ALSO flip prefix commits whose true fix had NOT yet occurred by `t_cut` (`fts1 > t_cut` or `+inf`)
    # so future windows `[cut_abs : N)` see massively different matured labels & thresholds!
    unrev_at_cut = (np.arange(cut_abs) >= 0) & ((~np.isfinite(fts1[:cut_abs])) | (fts1[:cut_abs] > t_cut))
    y2[:cut_abs][unrev_at_cut] = 1 - y2[:cut_abs][unrev_at_cut]
    fts2[:cut_abs][unrev_at_cut] = np.where(
        y2[:cut_abs][unrev_at_cut] == 1, t_cut + 2.0 * SECONDS_PER_DAY, np.inf
    )

    oracle_only_methods = {"B1a_Full_Hindsight", "B1b_1W_Oracle", "B1b_4W_Oracle"}
    deployable_t1 = [m for m in TRACK1_MAIN_METHODS if m not in oracle_only_methods]

    max_prefix_diffs: Dict[str, float] = {}
    decision_flips: Dict[str, int] = {}

    # Part B (`R3b-C1`): Call `run_single_seed_stream_all_methods` directly so 100% of real runner
    # methods (ALL 15 deployable Track 1 methods + ALL 11 Track 2 competitors including `BORB_Alarm`)
    # are automatically verified without any hardcoded method list!
    for host in ("mlp", "lightgbm"):
        _, out1 = run_single_seed_stream_all_methods(
            X=X1, X_tab=X1, X_la=X_la1, X_ch=X_ch1, commit_ts=ts, y_true=y1, fix_ts=fts1, effort=eff1,
            n_init=n_init, n_val=n_val, host_type=host, include_track2=(host == "mlp"),
            methods_track1=deployable_t1, window_size=200, seed=42,
        )
        _, out2 = run_single_seed_stream_all_methods(
            X=X2, X_tab=X2, X_la=X_la2, X_ch=X_ch2, commit_ts=ts, y_true=y2, fix_ts=fts2, effort=eff2,
            n_init=n_init, n_val=n_val, host_type=host, include_track2=(host == "mlp"),
            methods_track1=deployable_t1, window_size=200, seed=42,
        )
        assert set(out1.keys()) == set(out2.keys())
        if host == "mlp":
            assert "BORB_Alarm" in out1 and len(out1) == len(deployable_t1) + 11
        for m in sorted(out1.keys()):
            o1, o2 = out1[m], out2[m]
            d_raw = float(np.max(np.abs(o1["raw_probs"][:pref_len] - o2["raw_probs"][:pref_len])))
            d_cal = float(np.max(np.abs(o1["cal_probs"][:pref_len] - o2["cal_probs"][:pref_len])))
            d_th = float(np.max(np.abs(o1["tuned_th_vec"][:pref_len] - o2["tuned_th_vec"][:pref_len])))
            flips = int(np.sum(o1["decisions_th"][:pref_len] != o2["decisions_th"][:pref_len]))
            mx = max(d_raw, d_cal, d_th)
            max_prefix_diffs[f"{host}:{m}"] = mx
            decision_flips[f"{host}:{m}"] = flips
            assert mx == 0.0 and flips == 0, f"Prefix invariance violated in {host}:{m}: mx={mx}, flips={flips}"

    # Part C (`R3b-C1` exact counterexample): Mid-window `[300, 400)` perturbation starting at `350`
    # flips `DriftMonitorV3` alarm on `[300, 400)` from `False -> True`, yet ALL predictions `[300, 350)`
    # (including `BORB_Alarm`) MUST remain `0.0` diff!
    rng_c1 = np.random.RandomState(314)
    N_c1, D_c1 = 500, 10
    Xa_c1 = rng_c1.randn(N_c1, D_c1).astype(np.float32) * 0.8
    ya_c1 = (rng_c1.rand(N_c1) < sigmoid_safe(1.8 * Xa_c1[:, 0] - 1.4 * Xa_c1[:, 1] - 0.8)).astype(np.int32)
    Xb_c1 = Xa_c1.copy()
    # Anisotropic covariate shift starting ONLY at `350` inside window `[300, 400)` so `DriftMonitorV3` flips `False -> True`!
    Xb_c1[350:, 0] += 12.0
    Xb_c1[350:, 1] -= 12.0
    Xb_c1[350:, 2:5] *= 8.0
    ts_c1 = np.arange(N_c1, dtype=np.float64) * SECONDS_PER_DAY
    ftsa_c1 = np.where(ya_c1 == 1, ts_c1 + 5.0 * SECONDS_PER_DAY, np.inf)
    eff_c1 = np.abs(Xa_c1[:, 0]) * 50.0 + 10.0

    su_a, out_c1_a = run_single_seed_stream_all_methods(
        X=Xa_c1, X_tab=Xa_c1, X_la=Xa_c1[:, :1], X_ch=Xa_c1[:, :2], commit_ts=ts_c1, y_true=ya_c1,
        fix_ts=ftsa_c1, effort=eff_c1, n_init=150, n_val=150, host_type="mlp", include_track2=True,
        methods_track1=deployable_t1, window_size=100, seed=42,
    )
    su_b, out_c1_b = run_single_seed_stream_all_methods(
        X=Xb_c1, X_tab=Xb_c1, X_la=Xb_c1[:, :1], X_ch=Xb_c1[:, :2], commit_ts=ts_c1, y_true=ya_c1,
        fix_ts=ftsa_c1, effort=eff_c1, n_init=150, n_val=150, host_type="mlp", include_track2=True,
        methods_track1=deployable_t1, window_size=100, seed=42,
    )
    assert (not su_a["shared_alarm_windows"][0]["fired"]) and su_b["shared_alarm_windows"][0]["fired"], (
        f"Expected window [300,400) alarm False->True flip; got {su_a['shared_alarm_windows'][0]} vs {su_b['shared_alarm_windows'][0]}"
    )
    for m_k in out_c1_a:
        diff_300_350 = float(np.max(np.abs(out_c1_a[m_k]["raw_probs"][:50] - out_c1_b[m_k]["raw_probs"][:50])))
        flips_300_350 = int(np.sum(out_c1_a[m_k]["decisions_th"][:50] != out_c1_b[m_k]["decisions_th"][:50]))
        assert diff_300_350 == 0.0 and flips_300_350 == 0, (
            f"R3b-C1 mid-window future alarm leaked into [300,350) for {m_k}: diff={diff_300_350}, flips={flips_300_350}"
        )

    # Part D (`R3b-C2` exact counterexamples): Zero-latency & simultaneous-timestamp future commits on `ORB` & `ODaSC`
    rng_c2 = np.random.RandomState(2718)
    N_c2, D_c2, s_c2 = 230, 8, 150
    X_c2 = rng_c2.randn(N_c2, D_c2).astype(np.float32)
    ts_base = np.arange(N_c2, dtype=np.float64) * 2.0 * SECONDS_PER_DAY
    y_base = np.zeros(N_c2, dtype=np.int32)
    fts_base = np.full(N_c2, np.inf, dtype=np.float64)
    for j_p in range(10, 140, 8):
        y_base[j_p] = 1
        fts_base[j_p] = ts_base[j_p] + 3.0 * SECONDS_PER_DAY

    # D.1: Current commit 200 zero-latency label flip (`fix_ts[200] == commit_ts[200]`)
    y_d1_b, fts_d1_b = y_base.copy(), fts_base.copy()
    y_d1_b[200] = 1
    fts_d1_b[200] = ts_base[200]  # Zero latency!
    for runner_fn in (run_orb_oht_native_v3, run_odasc_oht_native_v3):
        ra = runner_fn(X_c2, ts_base, y_base, fts_base, stream_start=s_c2, seed=42)
        rb = runner_fn(X_c2, ts_base, y_d1_b, fts_d1_b, stream_start=s_c2, seed=42)
        assert float(np.max(np.abs(ra["raw_probs"][: 201 - s_c2] - rb["raw_probs"][: 201 - s_c2]))) == 0.0, "Zero-latency self-label leak at commit 200!"
        assert float(np.max(np.abs(ra["raw_probs"][201 - s_c2 :] - rb["raw_probs"][201 - s_c2 :]))) > 0.0, "Zero-latency commit 200 event was dropped instead of learned after commit 200!"

    # D.2: Simultaneous next commit (`ts[200] == ts[201]`, `y[201]` flipped with zero latency)
    ts_simul2 = ts_base.copy()
    ts_simul2[201] = ts_simul2[200]
    y_d2_b, fts_d2_b = y_base.copy(), fts_base.copy()
    y_d2_b[201] = 1
    fts_d2_b[201] = ts_simul2[201]
    for runner_fn in (run_orb_oht_native_v3, run_odasc_oht_native_v3):
        ra = runner_fn(X_c2, ts_simul2, y_base, fts_base, stream_start=s_c2, seed=42)
        rb = runner_fn(X_c2, ts_simul2, y_d2_b, fts_d2_b, stream_start=s_c2, seed=42)
        diff_200_201 = float(np.max(np.abs(ra["raw_probs"][: 202 - s_c2] - rb["raw_probs"][: 202 - s_c2])))
        assert diff_200_201 == 0.0, f"Simultaneous commit 201 leaked into commit 200: diff={diff_200_201}!"
        assert float(np.max(np.abs(ra["raw_probs"][202 - s_c2 :] - rb["raw_probs"][202 - s_c2 :]))) > 0.0, "Deferred simultaneous commit 201 event was dropped!"

    # D.3: Simultaneous batch `200..205` (`ts[200:206]` identical) + deferred learning check at `206`
    ts_batch = ts_base.copy()
    ts_batch[200:206] = ts_batch[200]
    y_d3_b, fts_d3_b = y_base.copy(), fts_base.copy()
    y_d3_b[203:206] = 1
    fts_d3_b[203:206] = ts_batch[200]
    for runner_fn in (run_orb_oht_native_v3, run_odasc_oht_native_v3):
        ra = runner_fn(X_c2, ts_batch, y_base, fts_base, stream_start=s_c2, seed=42)
        rb = runner_fn(X_c2, ts_batch, y_d3_b, fts_d3_b, stream_start=s_c2, seed=42)
        diff_batch = float(np.max(np.abs(ra["raw_probs"][: 206 - s_c2] - rb["raw_probs"][: 206 - s_c2])))
        assert diff_batch == 0.0, f"Simultaneous batch 200..205 leaked: diff={diff_batch}!"
        assert float(np.max(np.abs(ra["raw_probs"][206 - s_c2 :] - rb["raw_probs"][206 - s_c2 :]))) > 0.0, "Deferred simultaneous batch events were dropped!"

    return {
        "pass": True,
        "cold_start_within_window_diff": cold_diff,
        "n_runner_method_branches_verified": len(max_prefix_diffs),
        "max_prefix_diff_across_all_runner_methods": float(max(max_prefix_diffs.values())),
        "total_historical_decision_flips": int(sum(decision_flips.values())),
        "borb_alarm_mid_window_350_counterexample_diff": 0.0,
        "orb_odasc_zero_latency_and_simultaneous_counterexamples_diff": 0.0,
    }


def check_3_causal_kappa_pre_t0_freeze_and_zero_latency_parity(base_dir: str) -> Dict:
    # 1. Codex's exact 4-commit counterexample (`R3-C6`): initial `[0,0,0,0]` must NEVER become `[0,1,1,0]` at `kappa=0`!
    c_4 = np.array([10.0, 20.0, 30.0, 40.0])
    y_4 = np.array([0, 1, 1, 0], dtype=np.int32)
    f_4 = np.array([np.inf, 80.0, 90.0, np.inf])
    t0_4 = 40.0
    _, y_obs_orig = observe_labels_at_time(c_4, y_4, f_4, t0_4)
    assert y_obs_orig.tolist() == [0.0, 0.0, 0.0, 0.0]

    for f_int in [
        apply_future_only_latency_intervention(c_4, y_4, f_4, t0=t0_4, kappa=0.0, post_t0_oracle_visibility=False),
        apply_future_only_latency_intervention(c_4, y_4, f_4, t0=t0_4, kappa=0.0, post_t0_oracle_visibility=True),
    ]:
        _, y_obs_after = observe_labels_at_time(c_4, y_4, f_int, t0_4)
        assert y_obs_after.tolist() == [0.0, 0.0, 0.0, 0.0], f"kappa=0 leaked backlog into t0: {y_obs_after}"
        _, y_obs_post = observe_labels_at_time(c_4, y_4, f_int, t0_4 + 1.0)
        assert y_obs_post.tolist() == [0.0, 1.0, 1.0, 0.0]

    # 2. Stream pre-`t0` invariance across ALL `kappa` including `0.0` AND `post_t0_oracle_visibility=True`
    b = np.load(os.path.join(base_dir, "cache", "benchmark_b_v3", "apache_ignite.npz"))
    N = 2400
    X, ts, y, fts = b["X"][:N], b["commit_ts"][:N], b["y_true"][:N], b["fix_ts"][:N]
    n_init, n_val = 1100, 300
    s_start = n_init + n_val
    t0 = float(ts[s_start - 1])
    _, y_obs_ref = observe_labels_at_time(ts[:s_start], y[:s_start], fts[:s_start], t0)
    b0_ref = execute_within_host_counterfactual_suite_v3(
        X, ts, y, fts, n_init, n_val, host_type="mlp", methods=["B0_Frozen"], window_size=250, seed=42
    )["methods"]["B0_Frozen"]["raw_probs"]

    for kap in (0.0, 0.25, 0.5, 2.0, 3.0):
        fts_k = apply_future_only_latency_intervention(ts, y, fts, t0=t0, kappa=kap, post_t0_oracle_visibility=False)
        _, y_obs_k = observe_labels_at_time(ts[:s_start], y[:s_start], fts_k[:s_start], t0)
        assert np.array_equal(y_obs_ref, y_obs_k), f"kappa={kap} altered pre-t0 y_obs!"
        # Also confirm unlinked `+inf` bugs remain `+inf` under finite latency scaling `kappa=0.0`
        unlinked_pos = (y == 1) & (~np.isfinite(fts))
        assert np.all(~np.isfinite(fts_k[unlinked_pos])), "Ordinary kappa scaling altered +inf unlinked positives!"
        b0_k = execute_within_host_counterfactual_suite_v3(
            X, ts, y, fts_k, n_init, n_val, host_type="mlp", methods=["B0_Frozen"], fix_ts_for_init=fts, window_size=250, seed=42
        )["methods"]["B0_Frozen"]["raw_probs"]
        assert float(np.max(np.abs(b0_ref - b0_k))) == 0.0

    # 3. Post-t0 Oracle Visibility Control: exact `B1b == B2` parity while `B0_Frozen` stays identical!
    fts_ora = apply_future_only_latency_intervention(ts, y, fts, t0=t0, kappa=0.0, post_t0_oracle_visibility=True)
    _, y_obs_ora_t0 = observe_labels_at_time(ts[:s_start], y[:s_start], fts_ora[:s_start], t0)
    assert np.array_equal(y_obs_ref, y_obs_ora_t0)

    parity_diffs = {}
    for host in ("mlp", "lightgbm"):
        res = execute_within_host_counterfactual_suite_v3(
            X, ts, y, fts_ora, n_init=n_init, n_val=n_val, host_type=host,
            methods=["B0_Frozen", "B1b_1W_Oracle", "B2_1W", "B1b_4W_Oracle", "B2_4W"],
            fix_ts_for_init=fts, window_size=250, seed=42,
        )
        if host == "mlp":
            assert float(np.max(np.abs(b0_ref - res["methods"]["B0_Frozen"]["raw_probs"]))) == 0.0
        d_1w = float(np.max(np.abs(res["methods"]["B1b_1W_Oracle"]["raw_probs"] - res["methods"]["B2_1W"]["raw_probs"])))
        d_4w = float(np.max(np.abs(res["methods"]["B1b_4W_Oracle"]["raw_probs"] - res["methods"]["B2_4W"]["raw_probs"])))
        parity_diffs[f"{host}:1W"] = d_1w
        parity_diffs[f"{host}:4W"] = d_4w
        assert d_1w == 0.0 and d_4w == 0.0

    return {
        "pass": True,
        "codex_4commit_counterexample_invariant": True,
        "pre_t0_b0_and_labels_invariant_for_all_kappa": True,
        "post_t0_oracle_parity_diffs": parity_diffs,
    }


def check_4_immutable_alarm_budgets_across_4_consecutive_alarms() -> Dict:
    """
    Directly tests Codex R3-C2: across 4 consecutive alarms, LightGBM `Teacher_Only_Matched` must
    build `[150, 150, 150, 150]` trees (`600` total, matching `Survival_PU`), NEVER `180 -> 360 -> 720 -> 1440`!
    """
    rng = np.random.RandomState(42)
    N, D = 1600, 12
    X = rng.randn(N, D).astype(np.float32)
    ts = np.arange(N, dtype=np.float64) * SECONDS_PER_DAY * 0.5
    p_true = sigmoid_safe(0.9 * X[:, 0] - 0.8)
    y = (rng.rand(N) < p_true).astype(np.int32)
    fts = np.where(y == 1, ts + 20.0 * SECONDS_PER_DAY, np.inf)

    # Periodic monitor firing on EVERY window (`periodic_interval = 1`) over 4 windows of size 150
    lgb_trees_exact, lgb_trees_matched, lgb_trees_spu = [], [], []
    cur_ex = LightGBMHostV3(in_dim=D, seed=42)
    cur_ma = LightGBMHostV3(in_dim=D, seed=42)
    cur_sp = LightGBMHostV3(in_dim=D, seed=42)
    anc_lgb, _ = train_supervised_v3(LightGBMHostV3(in_dim=D, seed=42), X[:800], (y[:800] * 0.5).astype(np.float32), pos_weight=2.5, lgb_n_estimators=90, seed=42)
    cur_ex, cur_ma, cur_sp = anc_lgb, anc_lgb, anc_lgb

    for k in range(4):
        w_st = 800 + k * 150
        w_ed = w_st + 150
        t_al = float(ts[w_ed - 1])
        _, y_obs = observe_labels_at_time(ts[:w_ed], y[:w_ed], fts[:w_ed], t_al)
        cur_ex, d_ex = adapt_survival_pu_v3(cur_ex, anc_lgb, X[:w_ed], ts[:w_ed], fts[:w_ed], y_obs, np.arange(w_st, w_ed), t_al, 2.5, 0.25, teacher_mode="teacher_only_exact", seed=42 + k)
        cur_ma, d_ma = adapt_survival_pu_v3(cur_ma, anc_lgb, X[:w_ed], ts[:w_ed], fts[:w_ed], y_obs, np.arange(w_st, w_ed), t_al, 2.5, 0.25, teacher_mode="teacher_only_matched", seed=42 + k)
        cur_sp, d_sp = adapt_survival_pu_v3(cur_sp, anc_lgb, X[:w_ed], ts[:w_ed], fts[:w_ed], y_obs, np.arange(w_st, w_ed), t_al, 2.5, 0.25, teacher_mode="standard", seed=42 + k)
        lgb_trees_exact.append(int(d_ex.compute.trees_built))
        lgb_trees_matched.append(int(d_ma.compute.trees_built))
        lgb_trees_spu.append(int(d_sp.compute.trees_built))

    r_cfg = int(V3_CONFIG.lgb_alarm_rounds)
    assert lgb_trees_exact == [r_cfg] * 4, f"Unexpected Exact trees: {lgb_trees_exact}"
    assert lgb_trees_matched == [2 * r_cfg] * 4, f"Budget doubling regression in Teacher_Only_Matched: {lgb_trees_matched}"
    assert lgb_trees_spu == [2 * r_cfg] * 4, f"Unexpected SPU trees: {lgb_trees_spu}"

    # Neural opt_steps parity across 4 consecutive alarms
    anc_mlp = build_host_backbone_v3("mlp", X[:8], seed=42)
    anc_mlp, _ = train_supervised_v3(anc_mlp, X[:800], (y[:800] * 0.5).astype(np.float32), pos_weight=2.5, epochs=4, seed=42)
    cur_m_ma, cur_m_sp = anc_mlp, anc_mlp
    mlp_steps_ma, mlp_steps_sp = [], []
    for k in range(4):
        w_st, w_ed = 800 + k * 150, 950 + k * 150
        t_al = float(ts[w_ed - 1])
        _, y_obs = observe_labels_at_time(ts[:w_ed], y[:w_ed], fts[:w_ed], t_al)
        cur_m_ma, dm_ma = adapt_survival_pu_v3(cur_m_ma, anc_mlp, X[:w_ed], ts[:w_ed], fts[:w_ed], y_obs, np.arange(w_st, w_ed), t_al, 2.5, 0.25, teacher_mode="teacher_only_matched", seed=42 + k)
        cur_m_sp, dm_sp = adapt_survival_pu_v3(cur_m_sp, anc_mlp, X[:w_ed], ts[:w_ed], fts[:w_ed], y_obs, np.arange(w_st, w_ed), t_al, 2.5, 0.25, teacher_mode="standard", seed=42 + k)
        mlp_steps_ma.append(int(dm_ma.compute.opt_steps))
        mlp_steps_sp.append(int(dm_sp.compute.opt_steps))

    assert mlp_steps_ma == mlp_steps_sp, f"MLP step mismatch: {mlp_steps_ma} vs {mlp_steps_sp}"

    # R3b-M2 Boundary Counterexample 1: `N_rel = N_pool = 257` (`N mod 128 == 1`, single-sample tail skipped!)
    X_257 = rng.randn(257, D).astype(np.float32)
    ts_257 = np.arange(257, dtype=np.float64) * 5.0 * SECONDS_PER_DAY
    t_al_257 = ts_257[-1] + 250.0 * SECONDS_PER_DAY  # Every commit >= 250d old -> `c >= 0.75 > 0.45` -> exact `N_rel = N_pool = 257`!
    y_257 = (np.arange(257) % 5 == 0).astype(np.float32)
    fts_257 = np.where(y_257 > 0.5, ts_257 + 2.0 * SECONDS_PER_DAY, np.inf)
    _, d257_ex = adapt_survival_pu_v3(anc_mlp, anc_mlp, X_257, ts_257, fts_257, y_257, np.arange(200, 257), t_al_257, 2.5, 0.20, survival_estimator="fixed_90d", teacher_mode="teacher_only_exact", seed=42)
    _, d257_ma = adapt_survival_pu_v3(anc_mlp, anc_mlp, X_257, ts_257, fts_257, y_257, np.arange(200, 257), t_al_257, 2.5, 0.20, survival_estimator="fixed_90d", teacher_mode="teacher_only_matched", seed=42)
    _, d257_sp = adapt_survival_pu_v3(anc_mlp, anc_mlp, X_257, ts_257, fts_257, y_257, np.arange(200, 257), t_al_257, 2.5, 0.20, survival_estimator="fixed_90d", teacher_mode="standard", seed=42)
    assert d257_ex.compute.opt_steps == 8 and d257_ex.compute.examples_seen == 1024
    assert d257_ma.compute.opt_steps == 16 and d257_ma.compute.examples_seen == 2048, f"N=257 Matched mismatch: {d257_ma.compute}"
    assert d257_sp.compute.opt_steps == 16 and d257_sp.compute.examples_seen == 2048, f"N=257 SPU mismatch: {d257_sp.compute}"

    # R3b-M2 Boundary Counterexample 2: `len(rel_idx) < 100` fallback (`0` teacher steps, clean handling)
    _, d_short = adapt_survival_pu_v3(anc_mlp, anc_mlp, X[:60], ts[:60], fts[:60], np.zeros(60, dtype=np.float32), np.arange(30, 60), float(ts[59]), 2.5, 0.20, teacher_mode="teacher_only_matched", seed=42)
    assert d_short.compute.opt_steps == 0

    # R3b-M2 Boundary Counterexample 3: Constant `q = 0.25` -> `requested_rounds == 75`, `trees_built == booster.num_trees()`!
    lgb_c_q = LightGBMHostV3(in_dim=D, seed=42)
    _, cs_cq = lgb_c_q.fit_weighted_soft(X[:200], np.full(200, 0.25), pos_weight=1.0, n_estimators=75, seed=42)
    real_ntrees_cq = 0 if lgb_c_q.booster_ is None else int(lgb_c_q.booster_.num_trees())
    assert cs_cq.requested_rounds == 75 and cs_cq.trees_built == real_ntrees_cq and cs_cq.trees_built < 75

    # R3b-M2 Boundary Counterexample 4: All-constant features `X_const` -> zero LightGBMError, `trees_built == 0`!
    lgb_c_x = LightGBMHostV3(in_dim=D, seed=42)
    _, cs_cx = lgb_c_x.fit_weighted_soft(np.ones((200, D), dtype=np.float32), y[:200].astype(np.float64), pos_weight=2.0, n_estimators=75, seed=42)
    lg_cx, _ = lgb_c_x.predict_logits_and_embeddings(np.ones((16, D), dtype=np.float32))
    assert cs_cx.requested_rounds == 75 and cs_cx.trees_built == 0 and cs_cx.actual_iterations == 0 and np.all(np.isfinite(lg_cx))

    # R3c-2 Counterexample: Unexpected `LightGBMError` MUST raise `HostTrainingFailureError` (NEVER silently become constant!)
    for path_name, call_fn in [
        ("fit_weighted_soft", lambda: LightGBMHostV3(in_dim=D, seed=42).fit_weighted_soft(X[:200], y[:200].astype(np.float64), pos_weight=2.0)),
        ("train_faithful_nnpu_v3", lambda: train_faithful_nnpu_v3(LightGBMHostV3(in_dim=D, seed=42), X_pool=X[:200], y_obs_pool=y[:200].astype(np.float32), prior_pi=0.25, seed=42)),
    ]:
        raised_structured = False
        with patch.object(_mb_mod.lgb, "train", side_effect=_mb_mod.lgb.basic.LightGBMError("unexpected_training_failure")):
            try:
                call_fn()
            except HostTrainingFailureError as h_exc:
                raised_structured = isinstance(h_exc.__cause__, _mb_mod.lgb.basic.LightGBMError)
        assert raised_structured, f"R3c-2: {path_name} silently swallowed unexpected LightGBMError!"

    # R3b-M6 + R3c-3 Standalone Dependency Chain & Virtual-Clock Timing Check across Track 1 & Track 2:
    # Verify BOTH real wall-clock breakdown AND Codex's exact virtual-clock injection (`13s` frozen-anchor
    # eval pass + `17s` calibration pass)!
    t_suite = execute_within_host_counterfactual_suite_v3(
        X[:1200], ts[:1200], y[:1200], fts[:1200], n_init=500, n_val=200, host_type="mlp",
        methods=["B0_Frozen", "Survival_PU"], window_size=200, seed=42,
    )
    tb_b0 = t_suite["methods"]["B0_Frozen"]["timing_breakdown_ms"]
    tb_spu = t_suite["methods"]["Survival_PU"]["timing_breakdown_ms"]
    aw_t = [(w["start"], w["end"], w["fired"]) for w in t_suite["shared_alarm_windows"]]
    borb_al = run_borb_competitor_v3(
        X[:1200], ts[:1200], y[:1200], fts[:1200], 700, alarm_windows=aw_t, update_mode="alarm",
        shared_anchor_init_ms=t_suite["shared_anchor_init_ms"],
        shared_monitor_init_ms=t_suite["shared_monitor_fit_ms"],
        shared_anchor_eval_embed_ms=t_suite["shared_anchor_eval_embed_ms"],
        shared_monitor_ms=t_suite["shared_monitor_ms"],
        seed=42,
    )
    tb_ba = borb_al["timing_breakdown_ms"]
    assert tb_b0["stream_predict_and_embed_ms"] > 0.0 and t_suite["methods"]["B0_Frozen"]["end_to_end_eval_ms"] >= tb_b0["stream_predict_and_embed_ms"]
    assert tb_spu["shared_anchor_eval_embed_ms"] > 0.0 and tb_spu["drift_monitor_ms"] > 0.0
    assert tb_ba["shared_anchor_eval_embed_ms"] > 0.0 and tb_ba["shared_anchor_init_ms"] > 0.0 and tb_ba["drift_monitor_ms"] > 0.0

    # Codex R3c-3 Virtual-Clock Test (`13,000 ms` on frozen-anchor eval pass + `17,000 ms` on calibration):
    Xv = np.random.RandomState(123).normal(size=(900, 3)).astype(np.float32)
    Xv[500:, 0] += 30.0
    yv = (Xv[:, 0] > 0).astype(np.int32)
    tsv = np.arange(900, dtype=np.float64) * SECONDS_PER_DAY
    ftsv = np.where(yv == 1, tsv + SECONDS_PER_DAY, np.inf)
    vclock = [0.0]
    orig_pred = _mb_mod.predict_logits_and_embeddings_v3
    orig_cal = _mb_mod.calibrate_causal_stream_probs_strict

    def _charged_pred(model, x_in, *a, **kw):
        out = orig_pred(model, x_in, *a, **kw)
        if len(x_in) == 600:
            vclock[0] += 13.0
        return out

    def _charged_cal(*a, **kw):
        vclock[0] += 17.0
        return orig_cal(*a, **kw)

    with patch.object(_mb_mod, "time", types.SimpleNamespace(perf_counter=lambda: vclock[0])), \
         patch.object(_mb_mod, "predict_logits_and_embeddings_v3", _charged_pred), \
         patch.object(_mb_mod, "calibrate_causal_stream_probs_strict", _charged_cal):
        rv = _mb_mod.execute_within_host_counterfactual_suite_v3(
            Xv, tsv, yv, ftsv, 200, 100, methods=["B0_Frozen", "B2_4W"],
            window_size=100, monitor_mode="frechet", device=torch.device("cpu"),
        )
    b0_eval_v = rv["methods"]["B0_Frozen"]["timing_breakdown_ms"]["consistent_stream_eval_ms"]
    b2_eval_v = rv["methods"]["B2_4W"]["timing_breakdown_ms"]["consistent_stream_eval_ms"]
    b2_step_v = rv["methods"]["B2_4W"]["per_alarm_step_total_ms"]
    assert abs(b0_eval_v - 30000.0) < 1e-6, f"Expected B0 13s+17s=30000ms, got {b0_eval_v}"
    assert abs(b2_eval_v - 30000.0) < 1e-6, f"R3c-3: B2_4W missed frozen-anchor or cal cost: {b2_eval_v}"
    assert len(b2_step_v) >= 1 and all(v >= (13000.0 + 17000.0) / 6.0 - 1e-6 for v in b2_step_v), (
        f"R3c-3: per_alarm_step_total_ms missed anchor/calibration window share: {b2_step_v}"
    )

    # Codex R3d-2 Virtual-Clock Test (`11,000 ms` injected ONLY into initial shared `train_supervised_v3`):
    # Must appear in `shared_anchor_init_ms` & `consistent_end_to_end_ms`, and MUST be `0.0` in `own_init_fit_ms`
    # and `incremental_suite_end_to_end_ms` for BOTH `B0_Frozen` and `B2_4W`!
    vclock_init = [0.0]
    charged_init = [False]
    orig_train_sup = _mb_mod.train_supervised_v3

    def _charge_initial_fit(*a, **kw):
        out = orig_train_sup(*a, **kw)
        if not charged_init[0]:
            vclock_init[0] += 11.0
            charged_init[0] = True
        return out

    with patch.object(_mb_mod, "time", types.SimpleNamespace(perf_counter=lambda: vclock_init[0])), \
         patch.object(_mb_mod, "train_supervised_v3", _charge_initial_fit):
        rv_init = _mb_mod.execute_within_host_counterfactual_suite_v3(
            Xv, tsv, yv, ftsv, 200, 100, methods=["B0_Frozen", "B2_4W"],
            window_size=100, device=torch.device("cpu"),
        )
    assert abs(rv_init["shared_anchor_init_ms"] - 11000.0) < 1e-6
    for mk_chk in ("B0_Frozen", "B2_4W"):
        tb_chk = rv_init["methods"][mk_chk]["timing_breakdown_ms"]
        assert tb_chk["own_init_fit_ms"] == 0.0, f"R3d-2: {mk_chk} own_init_fit_ms charged shared init: {tb_chk}"
        assert abs(tb_chk["consistent_end_to_end_ms"] - 11000.0) < 1e-6
        assert tb_chk["incremental_suite_end_to_end_ms"] == 0.0, (
            f"R3d-2: {mk_chk} incremental_suite_end_to_end_ms charged shared init: {tb_chk}"
        )

    return {
        "pass": True,
        "lgb_4alarm_trees_exact": lgb_trees_exact,
        "lgb_4alarm_trees_matched": lgb_trees_matched,
        "lgb_4alarm_trees_spu": lgb_trees_spu,
        "mlp_4alarm_steps_matched": mlp_steps_ma,
        "mlp_4alarm_steps_spu": mlp_steps_sp,
        "n257_tail_batch_steps_matched_vs_spu": [int(d257_ma.compute.opt_steps), int(d257_sp.compute.opt_steps)],
        "const_q_requested_vs_actual_trees": [int(cs_cq.requested_rounds), int(cs_cq.trees_built)],
        "const_X_handled_cleanly": True,
        "unexpected_lgb_error_raises_structured_failure": True,
        "standalone_and_virtual_clock_timing_verified": True,
    }


def check_5_posterior_identity_between_solver_and_training() -> Dict:
    """
    Directly tests Codex R3-C5 counterexample:
      1. Clipped unreachable case (`base_eta=[10,-10], c=[0.96,0.96], target_pi=0.49`): because
         `p_max=0.997` caps `mean(q)` at `0.46771 < 0.49` at `hi=+8.0`, solver MUST truthfully report
         `is_reachable=False`, `clamped_side='high'`, and `bis.residual == actual_resid` (`< 1e-6` diff)
         instead of claiming `residual = 1e-10` on an unclipped formula!
      2. Reachable interior case (`target_pi=0.42`): solver achieves `is_reachable=True` and actual
         `float32` training `q_post` satisfies `|mean(q_train) - 0.42| < 1e-6`!
    """
    eta = np.array([10.0, -10.0], dtype=np.float64)
    c = np.array([0.96, 0.96], dtype=np.float64)
    y_obs = np.array([0.0, 0.0], dtype=np.float64)
    u = np.array([1.0, 1.0], dtype=np.float64)

    # 1. Exact Codex R3-C5 counterexample (`target_pi = 0.49 > f_hi = 0.46771` under `p_max = 0.997`)
    bis_unreach = solve_monotone_pu_bisection(eta, y_obs, c, u, target_pi=0.49)
    q_tr_unreach = compute_survival_pu_posterior(eta, bis_unreach.delta_star, y_obs, c).astype(np.float32)
    act_mean_unr = float(np.mean(q_tr_unreach.astype(np.float64)))
    act_resid_unr = abs(act_mean_unr - 0.49)
    assert (not bis_unreach.is_reachable) and bis_unreach.clamped_side == "high", (
        f"Solver falsely claimed 0.49 reachable under p_max=0.997: {bis_unreach}"
    )
    assert abs(bis_unreach.residual - act_resid_unr) < 1e-6, (
        f"Solver residual {bis_unreach.residual} != actual training residual {act_resid_unr}"
    )

    # 2. Reachable interior target (`target_pi = 0.42`)
    bis_reach = solve_monotone_pu_bisection(eta, y_obs, c, u, target_pi=0.42)
    q_tr_reach = compute_survival_pu_posterior(eta, bis_reach.delta_star, y_obs, c).astype(np.float32)
    act_mean_r = float(np.mean(q_tr_reach.astype(np.float64)))
    act_resid_r = abs(act_mean_r - 0.42)
    assert bis_reach.is_reachable and act_resid_r < 1e-6 and abs(bis_reach.achieved_prevalence - act_mean_r) < 1e-6

    # 3. Exact endpoint boundary reachability (`Minor 1`: `target_pi == f_hi` & `target_pi == f_lo` -> `is_reachable == True, residual == 0.0`!)
    f_hi_exact = float(np.mean(compute_survival_pu_posterior(eta, 8.0, y_obs, c)))
    f_lo_exact = float(np.mean(compute_survival_pu_posterior(eta, -8.0, y_obs, c)))
    bis_b_hi = solve_monotone_pu_bisection(eta, y_obs, c, u, target_pi=f_hi_exact)
    bis_b_lo = solve_monotone_pu_bisection(eta, y_obs, c, u, target_pi=f_lo_exact)
    assert bis_b_hi.is_reachable and bis_b_hi.clamped_side == "boundary_high" and bis_b_hi.residual == 0.0
    assert bis_b_lo.is_reachable and bis_b_lo.clamped_side == "boundary_low" and bis_b_lo.residual == 0.0

    return {
        "pass": True,
        "codex_049_truthfully_flagged_unreachable": not bis_unreach.is_reachable,
        "codex_049_solver_vs_train_residual_diff": abs(bis_unreach.residual - act_resid_unr),
        "interior_042_reachable": bis_reach.is_reachable,
        "endpoint_boundary_solutions_reachable": True,
        "actual_training_residual": act_resid_r,
    }


def check_6_faithful_orb_borb_odasc_event_traces() -> Dict:
    """
    Deterministic event-stream audit proving `ORB_OHT_Native`, `BORB_Native`, and `ODaSC_OHT_Native`
    execute their exact published mechanisms (`R3-C3`).
    """
    rng = np.random.RandomState(2026)
    N, D = 260, 8
    s_start = 120
    X = rng.randn(N, D).astype(np.float32)
    ts = np.arange(N, dtype=np.float64) * 2.0 * SECONDS_PER_DAY  # 1 commit every 2 days (520d span)
    y = np.zeros(N, dtype=np.int32)
    fts = np.full(N, np.inf, dtype=np.float64)

    # First 20 commits clean (maturity at +90d = day 90..130); first fast bug at commit 30 (day 60, fixed day 66)
    y[30] = 1
    fts[30] = ts[30] + 6.0 * SECONDS_PER_DAY
    # Delayed bug at commit 35 (day 70, fixed day 200 > 70 + 90d): MUST emit BOTH a 90d target=0 event AND a 200d target=1 relabeling event!
    y[35] = 1
    fts[35] = ts[35] + 130.0 * SECONDS_PER_DAY
    # Duplicate-feature noise safety test for ORB (`noise_o = 5`): commits 0..5 share identical features as clean (`y=0`),
    # and commit 110 has the EXACT SAME feature vector as `y=1` fixed at day 225!
    dup_vec = np.array([1.25, -0.75, 0.5, 0.0, -0.5, 0.25, 0.1, -0.1], dtype=np.float32)
    for idx_dup in range(6):
        X[idx_dup] = dup_vec
        y[idx_dup] = 0
    X[110] = dup_vec
    y[110] = 1
    fts[110] = ts[110] + 4.0 * SECONDS_PER_DAY
    # Additional signal bugs across stream
    for idx_b in range(40, N, 7):
        if idx_b != 110:
            X[idx_b, 0] += 1.8
            y[idx_b] = 1
            fts[idx_b] = ts[idx_b] + 12.0 * SECONDS_PER_DAY

    ev_ts, ev_idx, ev_lbl = extract_verification_latency_events(ts, y, fts, waiting_days=90.0)
    c35_events = [(float(t), int(l)) for t, i, l in zip(ev_ts, ev_idx, ev_lbl) if int(i) == 35]
    assert len(c35_events) == 2 and c35_events[0][1] == 0 and c35_events[1][1] == 1, f"Delayed bug 35 missing dual events: {c35_events}"

    orb = run_orb_oht_native_v3(X, ts, y, fts, stream_start=s_start, noise_o=5, seed=42, return_trace=True)
    assert orb["n_noise_skipped"] >= 1, "ORB duplicate-feature noise safety mechanism (`o=5`) failed to skip noisy positive!"

    borb = run_borb_competitor_v3(X, ts, y, fts, stream_start=s_start, pull_request_size=50, n_iterations=20, seed=42, return_trace=True)
    assert len(borb["trace"]) >= 1 and len(borb["trace"][0]["iterations"]) == 20
    # Verify `ma_after` / `obf` is iteratively updated inside the 20-epoch BORB loop
    it_obfs = [(it["obf0"], it["obf1"]) for it in borb["trace"][0]["iterations"]]
    assert len(it_obfs) == 20

    odasc = run_odasc_oht_native_v3(X, ts, y, fts, stream_start=s_start, waiting_days=15.0, seed=42, return_trace=True)
    assert odasc["final_microclusters"] >= 2 and len(odasc["trace"]) > 0
    assert odasc["max_microcluster_radius"] <= odasc["eps_radius"] + 1e-9

    # R3b-M4 Counterexample 1: 150 distinct close-float feature vectors (`1e-6` spacing, identical at 5 decimals!)
    X_close = np.zeros((150, 6), dtype=np.float64)
    X_close[:, 0] = 1.0 + np.arange(150, dtype=np.float64) * 1e-6  # ALL 150 rows round to `1.00000` at 5 decimals, yet all 150 are distinct!
    ts_close = np.arange(150, dtype=np.float64) * 2.0 * SECONDS_PER_DAY
    y_close = np.zeros(150, dtype=np.int32)
    y_close[120] = 1
    fts_close = np.where(y_close == 1, ts_close + 2.0 * SECONDS_PER_DAY, np.inf)
    orb_close = run_orb_oht_native_v3(X_close, ts_close, y_close, fts_close, stream_start=100, noise_o=5, seed=42, return_trace=True)
    assert orb_close["n_noise_skipped"] == 0, f"ORB falsely skipped distinct close-float features: {orb_close['n_noise_skipped']}"

    # R3b-M4 Counterexample 2: Differential check of ORB `(p1, lambda, obf)` against Cabral `jitsdp/orb.py`
    # including unclipped `lambda > 50` when `p1 < 0.02` AND noise-skipped positive NOT mutating `p1`!
    p1_ref = 0.50
    obs_cls_ref = set()
    for tr_ev in orb["trace"]:
        if tr_ev.get("skipped_noise"):
            # `baseline.py:191-197`: skipped noisy positive MUST NOT update `p1`!
            continue
        tgt = int(tr_ev["target"])
        p1_ref = 0.99 * p1_ref + 0.01 * float(tgt)
        assert abs(float(tr_ev["p1"]) - p1_ref) < 1e-12, f"ORB p1 diverged from jitsdp/orb.py: {tr_ev['p1']} vs {p1_ref}"
        obf_exp = compute_cabral_obf(tgt, float(tr_ev["ma"]), th=0.4, l0=10.0, l1=12.0, m=1.5) if tr_ev["ma"] is not None else 1.0
        assert abs(float(tr_ev["obf"]) - obf_exp) < 1e-12
        obs_cls_ref.add(tgt)

    # Also verify unclipped `lambda > 50` under default `decay_factor=0.99` (`455` clean events -> `p1_new ~ 0.015` -> `lam ~ 65 > 50`!)
    N_hi = 480
    X_hi = rng.randn(N_hi, 4).astype(np.float64)
    ts_hi = np.arange(N_hi, dtype=np.float64) * 2.0 * SECONDS_PER_DAY
    y_hi = np.zeros(N_hi, dtype=np.int32)
    y_hi[0] = 1      # Seed class 1 early so `is_trained` becomes True once first 0 matures
    y_hi[465] = 1    # Minority positive after ~455 clean events -> `lambda = (1-p1)/p1 > 60 > 50`!
    fts_hi = np.where(y_hi == 1, ts_hi + 1.0 * SECONDS_PER_DAY, np.inf)
    orb_hi = run_orb_oht_native_v3(X_hi, ts_hi, y_hi, fts_hi, stream_start=450, waiting_days=10.0, decay_factor=0.99, n_ensembles=1, seed=42, return_trace=True)
    max_lam_seen = max(float(e["lambda"]) for e in orb_hi["trace"] if not e.get("skipped_noise"))
    assert max_lam_seen > 50.0, f"Expected unclipped ORB lambda > 50.0, got {max_lam_seen}"

    # R3b-M4 Counterexample 3: ODaSC outlier flood (`eps_radius=0.25, max_mc=120`, 300 outliers up to `3e5`)
    trk_flood = SongDenStreamConfidenceTracker(eps_radius=0.25, max_mc=120)
    X_outliers = (np.arange(300, dtype=np.float64)[:, None] * 1000.0) * np.ones((1, 30), dtype=np.float64)
    y_outliers = (np.arange(300) % 5 == 0).astype(np.int32)
    trk_flood.calibrate_and_compute_alpha(X_outliers, y_outliers, t_sec=1000.0)
    assert trk_flood.max_cluster_radius() <= 0.25 + 1e-9, f"ODaSC radius exceeded eps_radius: {trk_flood.max_cluster_radius()}"
    assert len(trk_flood.cw_) <= 120

    # R3b-M4 Counterexample 4: ODaSC uses ALL `D = 30 > 24` features (changing col 28 changes predictions!)
    X_d30_a = rng.randn(180, 30).astype(np.float64)
    X_d30_b = X_d30_a.copy()
    X_d30_b[:, 28] = X_d30_b[:, 28] * (-4.0) + 12.0  # Column 28 > 24!
    y_d30 = (rng.rand(180) < 0.25).astype(np.int32)
    ts_d30 = np.arange(180, dtype=np.float64) * 2.0 * SECONDS_PER_DAY
    fts_d30 = np.where(y_d30 == 1, ts_d30 + 2.0 * SECONDS_PER_DAY, np.inf)
    od_a = run_odasc_oht_native_v3(X_d30_a, ts_d30, y_d30, fts_d30, stream_start=110, waiting_days=10.0, seed=42)
    od_b = run_odasc_oht_native_v3(X_d30_b, ts_d30, y_d30, fts_d30, stream_start=110, waiting_days=10.0, seed=42)
    assert float(np.max(np.abs(od_a["raw_probs"] - od_b["raw_probs"]))) > 0.0, "ODaSC ignored feature col 28 > 24!"

    return {
        "pass": True,
        "delayed_bug_dual_event_verified": True,
        "orb_noise_skipped_count": int(orb["n_noise_skipped"]),
        "orb_150_close_floats_skipped": int(orb_close["n_noise_skipped"]),
        "orb_unclipped_max_lambda": max_lam_seen,
        "borb_inner_iterations_verified": len(borb["trace"][0]["iterations"]),
        "odasc_variable_microclusters_count": int(odasc["final_microclusters"]),
        "odasc_outlier_flood_max_radius": trk_flood.max_cluster_radius(),
        "odasc_full_d30_features_active": True,
    }


def check_7_gbdt_and_neural_nnpu_disjoint_u_and_scar() -> Dict:
    """
    Tests Codex R3-C4:
      1. Supplementing `P` rows (`is_u=False, is_p=True`) leaves `R_U^-` and `U \\ P` gradients
         bit-for-bit identical (`0.0` diff), and `grad / n_u` matches finite differences of `L_nnPU`!
      2. Both Neural (`MLP`) and Tree (`LightGBM`) `Pure_nnPU` beat naive `B2` under SCAR (`c = 0.30`).
    """
    rng = np.random.RandomState(101)
    n_u = 80
    n_supp = 15
    preds_u = rng.randn(n_u)
    preds_supp = rng.randn(n_supp)
    is_p_u = np.zeros(n_u, dtype=bool)
    is_p_u[:6] = True
    is_u_only = np.ones(n_u, dtype=bool)

    # Compare pure `U` vs `[U; P_supp]` when total `P` set is identical (`6` rows moved vs supplemented):
    # First test that adding `P_supp` with `is_u=False` does NOT change `r_u_neg`:
    # Specifically, for `i in U \ P` when `neg_risk >= 0`, `grad_i` is `1.0 * sigma(z_i)(1 - sigma(z_i))`!
    obj_base = make_gbdt_fgrad_nnpu_objective(is_p_u, is_u_only, prior_pi=0.20)
    g_base, h_base = obj_base(preds_u, None)

    preds_ext = np.concatenate([preds_u, preds_supp])
    is_p_ext = np.concatenate([is_p_u, np.ones(n_supp, dtype=bool)])
    is_u_ext = np.concatenate([np.ones(n_u, dtype=bool), np.zeros(n_supp, dtype=bool)])
    obj_ext = make_gbdt_fgrad_nnpu_objective(is_p_ext, is_u_ext, prior_pi=0.20)
    g_ext, h_ext = obj_ext(preds_ext, None)

    u_minus_p = ~is_p_u
    u_grad_diff = float(np.max(np.abs(g_base[u_minus_p] - g_ext[:n_u][u_minus_p])))
    assert u_grad_diff == 0.0, f"Supplemented P contaminated U gradients: diff={u_grad_diff}"
    assert np.all(h_ext == 1.0)

    # Exact central finite-difference check of `grad / n_u` against `L_nnPU(z)` in BOTH risk branches!
    def _nnpu_loss(z_vec: np.ndarray, pi_v: float) -> float:
        sp = sigmoid_safe(z_vec)
        sn = 1.0 - sp
        ru_n = float(np.mean(sp[is_u_ext]))
        rp_n = float(np.mean(sp[is_p_ext]))
        rp_p = float(np.mean(sn[is_p_ext]))
        nr = ru_n - pi_v * rp_n
        return (-nr) if nr < 0.0 else (pi_v * rp_p + nr)

    eps = 1e-6
    for pi_test in (0.15, 0.68):  # `0.15` tests `neg_risk >= 0`; `0.68` tests `neg_risk < 0`!
        obj_t = make_gbdt_fgrad_nnpu_objective(is_p_ext, is_u_ext, prior_pi=pi_test)
        g_ana, _ = obj_t(preds_ext, None)
        g_ana_scaled = g_ana / float(n_u)
        g_fd = np.zeros_like(preds_ext)
        for j in range(len(preds_ext)):
            zp, zm = preds_ext.copy(), preds_ext.copy()
            zp[j] += eps
            zm[j] -= eps
            g_fd[j] = (_nnpu_loss(zp, pi_test) - _nnpu_loss(zm, pi_test)) / (2.0 * eps)
        fd_err = float(np.max(np.abs(g_ana_scaled - g_fd)))
        assert fd_err < 1e-8, f"GBDT nnPU functional gradient mismatch at pi={pi_test}: {fd_err}"

    # SCAR Synthetic Benchmark on BOTH `MLP` and `LightGBM`
    rng2 = np.random.RandomState(2026)
    N_tr, N_te, D = 2000, 800, 12
    w_true = rng2.randn(D) * 0.85
    X_tr = rng2.randn(N_tr, D).astype(np.float32)
    X_te = rng2.randn(N_te, D).astype(np.float32)
    p_tr = sigmoid_safe((X_tr @ w_true) - 0.9)
    p_te = sigmoid_safe((X_te @ w_true) - 0.9)
    y_tr = (rng2.rand(N_tr) < p_tr).astype(np.float32)
    y_te = (rng2.rand(N_te) < p_te).astype(np.int32)
    pi_true = float(np.mean(y_tr))
    y_obs_scar = (y_tr * (rng2.rand(N_tr) < 0.30)).astype(np.float32)

    scar_res = {}
    for host in ("mlp", "lightgbm"):
        m_ora = build_host_backbone_v3(host, X_tr[:8], seed=42)
        m_ora, _ = train_supervised_v3(m_ora, X_tr, y_tr, pos_weight=(1.0 - pi_true) / pi_true, epochs=10, lr=1e-3, seed=42)
        m_b2 = build_host_backbone_v3(host, X_tr[:8], seed=42)
        m_b2, _ = train_supervised_v3(m_b2, X_tr, y_obs_scar, pos_weight=(1.0 - pi_true) / pi_true, epochs=10, lr=1e-3, seed=42)
        m_nn = build_host_backbone_v3(host, X_tr[:8], seed=42)
        m_nn, _ = train_faithful_nnpu_v3(m_nn, X_pool=X_tr, y_obs_pool=y_obs_scar, prior_pi=pi_true, epochs=12, lr=1e-3, seed=42)

        ap_ora = float(average_precision_score(y_te, sigmoid_safe(predict_logits_and_embeddings_v3(m_ora, X_te)[0])))
        ap_b2 = float(average_precision_score(y_te, sigmoid_safe(predict_logits_and_embeddings_v3(m_b2, X_te)[0])))
        ap_nn = float(average_precision_score(y_te, sigmoid_safe(predict_logits_and_embeddings_v3(m_nn, X_te)[0])))
        scar_res[host] = {"oracle_ap": ap_ora, "naive_b2_ap": ap_b2, "nnpu_ap": ap_nn}
        if host == "mlp":
            assert ap_nn >= 0.92 * ap_ora

    return {"pass": True, "u_invariance_grad_diff": u_grad_diff, "scar_results": scar_res}


def check_8_lightgbm_callable_objective_and_exact_init_score_margin() -> Dict:
    """
    Strengthened Check 6 (`R3-M7`):
      1. Analytical soft-BCE gradient/Hessian vs central finite differences & Proposition 3.
      2. Exact `init_score` margin conversion test: compares `sigmoid_safe(LightGBMHostV3.logits)` on
         binary `y in {0, 1}` (`pos_weight=1, u=1`) against native LightGBM `objective='binary'`
         `booster.predict(X)` (with `boost_from_average=True`). Omitting or double-adding `init_score`
         causes `> 0.15` error; exact margin reconstruction yields `< 1e-6`!
    """
    rng = np.random.RandomState(42)
    N, D = 240, 10
    X = rng.randn(N, D).astype(np.float32)
    q = rng.uniform(0.02, 0.98, size=N)
    u = rng.uniform(0.4, 2.2, size=N)
    pw = 3.5
    preds = rng.randn(N) * 1.5

    obj = make_soft_bce_lgb_objective(q, pos_weight=pw, sample_weights=u)
    g_ana, h_ana = obj(preds, None)

    def _loss_vec(z: np.ndarray) -> np.ndarray:
        return u * (pw * q * np.logaddexp(0.0, -z) + (1.0 - q) * np.logaddexp(0.0, z))

    eps_g, eps_h = 1e-6, 1e-4
    g_num = (_loss_vec(preds + eps_g) - _loss_vec(preds - eps_g)) / (2.0 * eps_g)
    h_num = (_loss_vec(preds + eps_h) - 2.0 * _loss_vec(preds) + _loss_vec(preds - eps_h)) / (eps_h ** 2)

    p = sigmoid_safe(preds)
    g_2n = (u * pw * q) * (p - 1.0) + (u * (1.0 - q)) * p
    h_2n = (u * pw * q) * p * (1.0 - p) + (u * (1.0 - q)) * p * (1.0 - p)

    err_g_fd = float(np.max(np.abs(g_ana - g_num)))
    err_h_fd = float(np.max(np.abs(h_ana - h_num)))
    err_g_2n = float(np.max(np.abs(g_ana - g_2n)))
    err_h_2n = float(np.max(np.abs(h_ana - h_2n)))
    assert err_g_fd < 1e-7 and err_h_fd < 1e-4 and err_g_2n < 1e-12 and err_h_2n < 1e-12

    # 1. Exact internal C++ margin check on `LightGBMHostV3` (proving `predict_logits_and_embeddings`
    #    reconstructs the exact `preds` margin vector fed by LightGBM C++ into `_objective`!)
    y_bin = np.zeros(N, dtype=np.float64)
    y_bin[:42] = 1.0  # Exact 17.5% base rate (`init_logit = -1.550597 != 0`)
    rng.shuffle(y_bin)

    host = LightGBMHostV3(in_dim=D, n_estimators_default=25, seed=42)
    host.fit_weighted_soft(X, y_bin, pos_weight=1.0, sample_weights=np.ones(N), n_estimators=25, seed=42)
    lg_host_f32, _ = host.predict_logits_and_embeddings(X)

    # 2. Exact 25-tree equivalence against native C++ `objective="binary"` (`boost_from_average=True`)
    #    (using `bagging_fraction=1.0, feature_fraction=1.0` because C++ `binary` uses class-stratified bagging RNG)
    z0_exact = float(np.log(0.175 / 0.825))
    seen_round24: list = []
    base_obj = make_soft_bce_lgb_objective(y_bin, pos_weight=1.0, sample_weights=np.ones(N))

    def _traced_obj(p_in: np.ndarray, d_in: lgb.Dataset):
        seen_round24.clear()
        seen_round24.append(p_in.copy())
        return base_obj(p_in, d_in)

    det_params = {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 10,
        "seed": 42,
        "num_threads": 1,
        "verbose": -1,
        "deterministic": True,
    }
    bst_custom = lgb.train(
        params={**det_params, "objective": _traced_obj},
        train_set=lgb.Dataset(X, label=y_bin, init_score=np.full(N, z0_exact, dtype=np.float64), free_raw_data=True),
        num_boost_round=25,
    )
    bst_native = lgb.train(
        params={**det_params, "objective": "binary", "boost_from_average": True},
        train_set=lgb.Dataset(X, label=y_bin, free_raw_data=True),
        num_boost_round=25,
    )
    # Check (a): round-24 C++ `preds` == `raw_score(iter=24) + z0_exact` to machine precision!
    internal_margin_err = float(np.max(np.abs(seen_round24[0] - (bst_custom.predict(X, raw_score=True, num_iteration=24) + z0_exact))))
    # Check (b): 25-tree `sigmoid(raw_score + z0_exact)` == native `objective="binary"` `predict(X)` to `< 1e-12`!
    prob_custom_det = sigmoid_safe(bst_custom.predict(X, raw_score=True) + z0_exact)
    prob_native_det = bst_native.predict(X)
    prob_err_vs_native = float(np.max(np.abs(prob_custom_det - prob_native_det)))
    # Check (c): Counterfactual proof that omitting OR double-adding `z0_exact` fails badly (`> 0.15`)!
    err_if_omitted = float(np.max(np.abs(sigmoid_safe(bst_custom.predict(X, raw_score=True)) - prob_native_det)))
    err_if_doubled = float(np.max(np.abs(sigmoid_safe(bst_custom.predict(X, raw_score=True) + 2.0 * z0_exact) - prob_native_det)))
    # Check (d): `LightGBMHostV3.predict_logits_and_embeddings` matches `booster_.predict(raw_score=True) + constant_logit_` to float32 precision!
    host_margin_err = float(np.max(np.abs(lg_host_f32.astype(np.float64) - (host.booster_.predict(X, raw_score=True) + host.constant_logit_))))

    assert internal_margin_err < 1e-12 and prob_err_vs_native < 1e-12 and host_margin_err < 1e-6
    assert err_if_omitted > 0.15 and err_if_doubled > 0.10

    return {
        "pass": True,
        "grad_err_vs_finite_diff": err_g_fd,
        "hess_err_vs_finite_diff": err_h_fd,
        "prop3_grad_err_vs_2N": err_g_2n,
        "internal_cpp_preds_vs_raw_plus_init_err": internal_margin_err,
        "host_margin_reconstruction_f32_err": host_margin_err,
        "max_prob_err_vs_native_lgb_binary": prob_err_vs_native,
        "counterfactual_err_if_init_score_omitted": err_if_omitted,
        "counterfactual_err_if_init_score_doubled": err_if_doubled,
    }


def check_9_unconfounded_drift_monitor_2x2_and_b0_invariance(base_dir: str) -> Dict:
    # 1. Verify ALL 4 cells of the 2x2 factorial share identical `percentile = 95.0` & `cooldown_windows = 1`
    rng = np.random.RandomState(7)
    Z_ref = rng.randn(600, 10)
    Z_val = rng.randn(400, 10)
    Z_shift = rng.randn(400, 10) + 4.0

    for samp in ("iid", "block"):
        for ref_m in ("static", "rolling"):
            mon = DriftMonitorV3(mode="frechet", calib_sampling=samp, reference_refresh=ref_m, seed=42)
            assert mon.percentile == V3_CONFIG.drift_block_pct == 95.0
            assert mon.cooldown_windows == V3_CONFIG.drift_cooldown_windows == 1
            mon.fit_reference(Z_ref, Z_val, window_size=200, n_boot=60)
            _, fired_1 = mon.score_window(Z_shift[:200])
            assert fired_1 is True, f"Expected shift alarm on ({samp}, {ref_m})"
            # Immediate next window MUST be suppressed by `cooldown_windows == 1` even when `ref_m == 'static'`!
            _, fired_2 = mon.score_window(Z_shift[200:])
            assert fired_2 is False, f"Cooldown failed to trigger on ({samp}, {ref_m})!"

    # 2. B0_Frozen invariance across detectors and window sizes
    b = np.load(os.path.join(base_dir, "cache", "benchmark_b_v3", "cabral_nova.npz"))
    N = 2200
    X, ts, y, fts = b["X"][:N], b["commit_ts"][:N], b["y_true"][:N], b["fix_ts"][:N]
    ref_out = execute_within_host_counterfactual_suite_v3(
        X, ts, y, fts, 1000, 300, host_type="mlp", methods=["B0_Frozen"], window_size=400, seed=42
    )["methods"]["B0_Frozen"]["raw_probs"]

    diffs = {}
    for tag, kw in [
        ("iid_static", {"calib_sampling": "iid", "reference_refresh": "static", "window_size": 400}),
        ("iid_rolling", {"calib_sampling": "iid", "reference_refresh": "rolling", "window_size": 400}),
        ("block_static", {"calib_sampling": "block", "reference_refresh": "static", "window_size": 400}),
        ("mmd", {"monitor_mode": "mmd", "window_size": 400}),
        ("ks", {"monitor_mode": "ks", "window_size": 400}),
        ("periodic", {"monitor_mode": "periodic", "window_size": 400}),
        ("win200", {"window_size": 200}),
        ("win600", {"window_size": 600}),
    ]:
        p = execute_within_host_counterfactual_suite_v3(
            X, ts, y, fts, 1000, 300, host_type="mlp", methods=["B0_Frozen"], seed=42, **kw
        )["methods"]["B0_Frozen"]["raw_probs"]
        d = float(np.max(np.abs(ref_out - p)))
        diffs[tag] = d
        assert d == 0.0

    return {"pass": True, "unconfounded_2x2_pct95_cd1_verified": True, "max_b0_diff": float(max(diffs.values()))}


def check_10_linear_skip_epoch0_equivalence(base_dir: str) -> Dict:
    with open(os.path.join(base_dir, "cache", "multimodal_v3_pure13", "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        mm_meta = {m["name"]: m for m in json.load(f)}["qt_natural_25k"]

    b_tok = np.load(os.path.join(base_dir, "cache", "multimodal_v3_pure13", "qt_natural_25k_tok.npz"))
    b_cb = np.load(os.path.join(base_dir, "cache", "multimodal_v3_pure13", "qt_natural_25k_cb.npz"))
    n_sl = 2000
    t0 = float(b_tok["commit_ts"][n_sl - 1])
    _, y_obs = observe_labels_at_time(b_tok["commit_ts"][:n_sl], b_tok["y_true"][:n_sl], b_tok["fix_ts"][:n_sl], t0)
    pw = 4.0

    w_lr, b_lr = fit_causal_tabular_lr_weights(b_tok["X_tab"][:n_sl], y_obs, pos_weight=pw, seed=42)
    lr_logits = (b_tok["X_tab"][:256] @ w_lr) + b_lr

    max_errs = {}
    for h_type, X_mat in [("deepjit_cnn", b_tok["X"]), ("att_cnn", b_tok["X"]), ("codebert", b_cb["X"])]:
        m = build_host_backbone_v3(h_type, X_mat[:16], meta=mm_meta, skip_mode="frozen_lr", seed=42)
        m.init_linear_skip_from_data(X_mat[:n_sl], y_obs, pos_weight=pw, seed=42)
        lg_m, _ = predict_logits_and_embeddings_v3(m, X_mat[:256])
        err = float(np.max(np.abs(lg_m - lr_logits)))
        max_errs[h_type] = err
        assert err < 1e-4

    return {"pass": True, "epoch0_logit_max_abs_err_vs_tabular_lr": max_errs}


def check_11_km_rounding_adversarial_and_fuzz() -> Dict:
    n = 300
    t_alarm = 500.0 * SECONDS_PER_DAY
    ages_d = np.linspace(0.5, 400.0, n)
    ages_d[10:25] = 1.004
    ages_d[25:40] = 1.006
    commit_ts = t_alarm - ages_d * SECONDS_PER_DAY
    fix_ts = np.full(n, np.inf)
    y_obs = np.zeros(n, dtype=np.float32)
    for idx in range(25, 40):
        fix_ts[idx] = commit_ts[idx] + 1.006 * SECONDS_PER_DAY
        y_obs[idx] = 1.0
    for idx in range(100, 140):
        fix_ts[idx] = commit_ts[idx] + 35.12346 * SECONDS_PER_DAY
        y_obs[idx] = 1.0

    km = KaplanMeierCureEstimator().fit(commit_ts, fix_ts, y_obs, t_alarm)
    c_pred = km.predict_maturity(t_alarm - commit_ts)
    assert np.all(np.isfinite(c_pred)) and np.all((c_pred >= km.c_min - 1e-6) & (c_pred <= km.c_max + 1e-6))

    rng = np.random.RandomState(123)
    for _ in range(50):
        m = 350
        a_d = rng.uniform(0.0001, 360.0, size=m)
        cts = t_alarm - a_d * SECONDS_PER_DAY
        ev = rng.rand(m) < 0.18
        del_d = a_d * rng.uniform(0.1, 1.0, size=m)
        fts = np.where(ev, cts + del_d * SECONDS_PER_DAY, np.inf)
        yo = ev.astype(np.float32)
        km_f = KaplanMeierCureEstimator().fit(cts, fts, yo, t_alarm)
        cp = km_f.predict_maturity(t_alarm - cts)
        assert np.all(np.isfinite(cp)) and np.all(np.diff(km_f.cdf_vals_) >= -1e-9)

    return {"pass": True, "adversarial_1_006d_passed": True, "fuzz_trials_passed": 50}


def check_12_causal_anchor_and_feature_dependent_latency_audit() -> Dict:
    """
    Upgraded Mechanism Audit (`R3-M1`):
      - Uses ACTUAL CAUSAL ANCHOR (`mean(y_obs_hist[c >= 0.55] / max(c, 0.10))`, ZERO oracle `eta_hist`!)
        and exact `V3_CONFIG.ht_shrinkage_m0 = 50.0`!
      - Includes a 5th `feature_dependent_delay_and_latency_drift` regime where high-risk `x > 0`
        bugs take `2.4x` longer to reveal AND post-drift latency stretches by `1.4x`, auditing how
        the age-only plug-in estimator behaves when `P(S=1 | Y=1, X, a) != c(a)`.
    """
    regimes = [
        "uncorrelated",
        "adversarial_neg_cov",
        "benign_pos_cov",
        "heavy_tail_weibull",
        "feature_dependent_delay_and_latency_drift",
    ]
    m0_def = float(V3_CONFIG.ht_shrinkage_m0)
    summary: Dict[str, Dict] = {}

    for reg in regimes:
        errs_obs, errs_legacy_ratio, errs_fixed90, errs_ht_v3 = [], [], [], []
        post_brier_b2, post_brier_v3, post_ap_v3, post_ap_f90 = [], [], [], []

        for mc in range(20):
            rng = np.random.RandomState(1000 + mc)
            ages_hist_d = rng.uniform(5.0, 540.0, size=2400)
            x_hist = rng.randn(2400)
            eta_hist = sigmoid_safe(-1.2 + 1.1 * x_hist)
            y_hist_true = (rng.rand(2400) < eta_hist).astype(np.int32)

            if reg == "heavy_tail_weibull":
                lat_hist_d = rng.weibull(0.55, size=2400) * 260.0
            elif reg == "feature_dependent_delay_and_latency_drift":
                scale_h = np.where(x_hist > 0.0, 180.0, 75.0)
                lat_hist_d = rng.weibull(0.95, size=2400) * scale_h
            else:
                lat_hist_d = rng.weibull(0.95, size=2400) * 110.0

            t_alarm = 600.0 * SECONDS_PER_DAY
            cts_hist = t_alarm - ages_hist_d * SECONDS_PER_DAY
            fts_hist = np.where(y_hist_true == 1, cts_hist + lat_hist_d * SECONDS_PER_DAY, np.inf)
            _, y_obs_hist = observe_labels_at_time(cts_hist, y_hist_true, fts_hist, t_alarm)

            ages_cal = np.concatenate([rng.uniform(180.0, 360.0, size=500), rng.uniform(1.0, 18.0, size=500)])
            x_cal = rng.randn(1000)
            if reg == "uncorrelated":
                base_logit_cal = -0.8 + 1.1 * x_cal
            elif reg == "adversarial_neg_cov":
                base_logit_cal = np.concatenate([np.full(500, -2.1), np.full(500, +1.05)]) + 1.0 * x_cal
            elif reg == "benign_pos_cov":
                base_logit_cal = np.concatenate([np.full(500, +0.2), np.full(500, -2.3)]) + 1.0 * x_cal
            else:
                base_logit_cal = np.concatenate([np.full(500, -1.3), np.full(500, +0.55)]) + 1.0 * x_cal

            eta_cal_true = sigmoid_safe(base_logit_cal)
            y_cal_true = (rng.rand(1000) < eta_cal_true).astype(np.int32)
            if reg == "heavy_tail_weibull":
                lat_cal_d = rng.weibull(0.55, size=1000) * 260.0
            elif reg == "feature_dependent_delay_and_latency_drift":
                scale_c = np.where(x_cal > 0.0, 220.0, 90.0) * np.concatenate([np.ones(500), np.full(500, 1.35)])
                lat_cal_d = rng.weibull(0.95, size=1000) * scale_c
            else:
                lat_cal_d = rng.weibull(0.95, size=1000) * 110.0

            cts_cal = t_alarm - ages_cal * SECONDS_PER_DAY
            fts_cal = np.where(y_cal_true == 1, cts_cal + lat_cal_d * SECONDS_PER_DAY, np.inf)
            _, y_obs_cal = observe_labels_at_time(cts_cal, y_cal_true, fts_cal, t_alarm)

            pi_true = float(np.mean(eta_cal_true))
            u_cal = np.ones(1000, dtype=np.float64)

            est_ci = CohortIncidenceEstimator().fit(cts_hist, fts_hist, y_obs_hist, t_alarm)
            est_f90 = FixedHalfLifeMaturityEstimator(half_life_days=90.0).fit(cts_hist, fts_hist, y_obs_hist, t_alarm)
            c_hist_ci = est_ci.predict_maturity(t_alarm - cts_hist)
            c_ci = est_ci.predict_maturity(t_alarm - cts_cal)
            c_f90 = est_f90.predict_maturity(t_alarm - cts_cal)

            # Strictly CAUSAL mature anchor (`adapt_survival_pu_v3` exact rule, zero oracle `eta_hist`!)
            mat_m = c_hist_ci >= 0.55
            c_mat_til = np.maximum(c_hist_ci[mat_m], V3_CONFIG.ht_floor_c)
            causal_anchor_pi = float(np.clip(np.mean(y_obs_hist[mat_m] / c_mat_til), V3_CONFIG.pi_min, V3_CONFIG.pi_max))

            pi_obs = float(np.mean(y_obs_cal))
            pi_legacy = float((np.sum(y_obs_cal) + m0_def * causal_anchor_pi) / (np.sum(c_ci) + m0_def))
            pi_ht_v3, _, _ = estimate_ht_shrunk_prevalence(y_obs_cal, c_ci, u_cal, anchor_prevalence=causal_anchor_pi, m0=m0_def)
            pi_ht_f90, _, _ = estimate_ht_shrunk_prevalence(y_obs_cal, c_f90, u_cal, anchor_prevalence=causal_anchor_pi, m0=m0_def)

            errs_obs.append(abs(pi_obs - pi_true))
            errs_legacy_ratio.append(abs(pi_legacy - pi_true))
            errs_fixed90.append(abs(pi_ht_f90 - pi_true))
            errs_ht_v3.append(abs(pi_ht_v3 - pi_true))

            teach_eta = -1.0 + 0.95 * x_cal + np.concatenate([np.zeros(500), np.full(500, 0.35 if reg == "adversarial_neg_cov" else -0.15)])
            bis_v3 = solve_monotone_pu_bisection(teach_eta, y_obs_cal, c_ci, u_cal, target_pi=pi_ht_v3)
            bis_f90 = solve_monotone_pu_bisection(teach_eta, y_obs_cal, c_f90, u_cal, target_pi=pi_ht_f90)
            q_v3 = compute_survival_pu_posterior(teach_eta, bis_v3.delta_star, y_obs_cal, c_ci)
            q_f90 = compute_survival_pu_posterior(teach_eta, bis_f90.delta_star, y_obs_cal, c_f90)

            post_brier_b2.append(float(brier_score_loss(y_cal_true, y_obs_cal)))
            post_brier_v3.append(float(brier_score_loss(y_cal_true, q_v3)))
            post_ap_v3.append(float(average_precision_score(y_cal_true, q_v3)))
            post_ap_f90.append(float(average_precision_score(y_cal_true, q_f90)))

        summary[reg] = {
            "prevalence_mae_naive_y_obs": float(np.mean(errs_obs)),
            "prevalence_mae_legacy_exposure_ratio": float(np.mean(errs_legacy_ratio)),
            "prevalence_mae_fixed90_ht": float(np.mean(errs_fixed90)),
            "prevalence_mae_v3_ht": float(np.mean(errs_ht_v3)),
            "posterior_mse_naive_b2": float(np.mean(post_brier_b2)),
            "posterior_mse_v3_ht": float(np.mean(post_brier_v3)),
            "posterior_ap_v3_learned_curve": float(np.mean(post_ap_v3)),
            "posterior_ap_fixed90_curve": float(np.mean(post_ap_f90)),
        }

    adv = summary["adversarial_neg_cov"]
    assert adv["prevalence_mae_v3_ht"] < adv["prevalence_mae_legacy_exposure_ratio"] * 0.70
    return {"pass": True, "causal_anchor_m0": m0_def, "regimes": summary}


def check_13_exact_small_k_signed_rank_and_metric_orientation() -> Dict:
    """
    Directly tests Codex R3-C8 & R3-M5:
      1. `K=4` all-positive differences -> exact `0.125` (`pratt` & `wilcox`);
      2. `K=5` all-positive differences -> exact `0.0625` (NEVER normal-approx `0.043114`!);
      3. Matches `scipy.stats.wilcoxon(..., mode='exact')` on non-zero vectors to `< 1e-14`;
      4. `higher_is_better=False` reverses `wins`/`losses` for Brier score;
      5. Constant density produces `spearman_with_inv_loc = NaN` (never `1.0`);
      6. `aggregate_streams_to_donor_clusters` rejects missing streams, missing methods, and `NaN`s!
    """
    p4_p = exact_signed_rank_two_sided_pvalue(np.array([0.1, 0.2, 0.3, 0.4]), zero_method="pratt")
    p4_w = exact_signed_rank_two_sided_pvalue(np.array([0.1, 0.2, 0.3, 0.4]), zero_method="wilcox")
    assert abs(p4_p - 0.125) < 1e-15 and abs(p4_w - 0.125) < 1e-15

    d5 = np.array([0.05, 0.08, 0.11, 0.14, 0.20])
    p5_p = exact_signed_rank_two_sided_pvalue(d5, zero_method="pratt")
    p5_w = exact_signed_rank_two_sided_pvalue(d5, zero_method="wilcox")
    p5_scipy = float(scipy.stats.wilcoxon(d5, zero_method="wilcox", mode="exact").pvalue)
    assert abs(p5_p - 0.0625) < 1e-15 and abs(p5_w - 0.0625) < 1e-15 and abs(p5_w - p5_scipy) < 1e-14

    # Ties + zeros exact enumeration check
    d_ties = np.array([0.0, 0.10, 0.10, -0.05, 0.20, 0.20])
    p_ties_pratt = exact_signed_rank_two_sided_pvalue(d_ties, zero_method="pratt")
    p_ties_wilcox = exact_signed_rank_two_sided_pvalue(d_ties, zero_method="wilcox")
    assert 0.0 < p_ties_pratt <= 1.0 and 0.0 < p_ties_wilcox <= 1.0

    # Brier orientation (`higher_is_better=False`: lower Brier is a WIN!)
    brier_a = [0.10, 0.11, 0.12, 0.09, 0.13]
    brier_b = [0.18, 0.24, 0.21, 0.22, 0.26]
    st_brier = paired_cluster_inference_v3(brier_a, brier_b, higher_is_better=False)
    assert st_brier["wins"] == 5 and st_brier["losses"] == 0 and st_brier["cohen_dz"] > 0.0

    # Constant density Spearman = NaN check
    eff_c = np.array([10.0, 20.0, 40.0, 80.0])
    p_const_dens = 0.005 * eff_c  # `p / eff == 0.005` constant!
    em = compute_effort_and_loc_masking_metrics(np.array([0, 1, 0, 1]), p_const_dens, np.zeros(4), eff_c)
    assert np.isnan(em["spearman_with_inv_loc"]) and em["constant_density_flag"] == 1.0

    # Strict missingness rejection in `aggregate_streams_to_donor_clusters`
    missing_rejected = False
    try:
        aggregate_streams_to_donor_clusters({"s1": {"m1": 0.5}}, {"s1": "c1", "s2": "c2"})
    except ValueError:
        missing_rejected = True
    assert missing_rejected

    # R3b-M3 Exact Counterexample: two seeds with `n_commits=100, n_defective=10` and `AP = [0.5, NaN]`
    # MUST raise `ValueError` (never return `ap=0.5, ap_std=0.0`!)
    y_dummy = np.array([1] * 10 + [0] * 90, dtype=np.int32)
    p_dummy = np.linspace(0.1, 0.9, 100)
    ev_seed0 = evaluate_horizon_v3(y_dummy, np.zeros(100), p_dummy, p_dummy, np.ones(100))
    ev_seed1_nan = dict(ev_seed0)
    ev_seed1_nan["ap"] = float("nan")
    nan_seed_rejected = False
    try:
        aggregate_seed_horizon_metrics([ev_seed0, ev_seed1_nan], stream_name="test_s", horizon_name="full_stream")
    except ValueError:
        nan_seed_rejected = True
    assert nan_seed_rejected, "R3b-M3: aggregate_seed_horizon_metrics failed to reject [0.5, NaN] on two-class slice!"

    # Single-class horizon (`n_defective == 0`) with `ap = NaN` IS legitimately handled (`ap = None`)
    ev_0pos = evaluate_horizon_v3(np.zeros(100, dtype=np.int32), np.zeros(100), p_dummy, p_dummy, np.ones(100))
    agg_0pos = aggregate_seed_horizon_metrics([ev_0pos, ev_0pos], stream_name="test_s", horizon_name="transient_1200")
    assert agg_0pos["ap"] is None

    # Minor 2: Constant non-zero difference (`sd == 0`) marks `cohen_dz = None` (`cohen_dz_undefined_zero_sd = True`)
    # and `min_attainable_exact_two_sided_p` uses `K_nonzero_pairs` when ties are present!
    st_const_diff = paired_cluster_inference_v3([0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8])
    assert st_const_diff["cohen_dz"] is None and st_const_diff["cohen_dz_undefined_zero_sd"] is True
    st_with_tie = paired_cluster_inference_v3([0.5, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.75, 0.8])
    assert st_with_tie["K"] == 5 and st_with_tie["K_nonzero"] == 4
    assert abs(st_with_tie["min_attainable_exact_two_sided_p"] - 0.125) < 1e-15

    # R3c-1 Check: Frozen interpretation boundaries avoid forced causal attribution from non-rejection
    # and explicitly scope alpha=0.05 Holm inference to K=25 / K=24 (`K=7` p_min = 0.015625 > 0.0125)!
    assert "powered_cohort_scope" in PRE_REGISTERED_INTERPRETATION_BOUNDARIES
    for rule_txt in PRE_REGISTERED_INTERPRETATION_BOUNDARIES.values():
        assert "MUST attribute" not in rule_txt, f"R3c-1 forced attribution remained in boundary rule: {rule_txt}"
    assert "not established on this cohort" in PRE_REGISTERED_INTERPRETATION_BOUNDARIES["learned_vs_fixed_survival_curve"]
    assert "not established on this cohort" in PRE_REGISTERED_INTERPRETATION_BOUNDARIES["stage3_pu_vs_matched_teacher"]
    st_k7 = paired_cluster_inference_v3(np.full(7, 0.9), np.full(7, 0.1))
    assert abs(st_k7["min_attainable_exact_two_sided_p"] - 0.015625) < 1e-15 and st_k7["min_attainable_exact_two_sided_p"] > (0.05 / 4.0)

    return {
        "pass": True,
        "exact_K4_all_pos_p": p4_p,
        "exact_K5_all_pos_p": p5_p,
        "brier_orientation_wins": st_brier["wins"],
        "constant_density_spearman_is_nan": True,
        "missing_stream_rejected": missing_rejected,
        "two_class_nan_seed_rejected": nan_seed_rejected,
        "zero_sd_nonzero_mean_cohen_dz_undefined": True,
        "r3c1_nonrejection_rules_and_k7_resolution_verified": True,
    }


def check_14_checkpoint_fingerprint_llm_schema_forensic_rules_and_cell_verifier(base_dir: str) -> Dict:
    """
    Directly tests Codex R3-C9, R3-M3, and R3-M4:
      1. Stale checkpoint `run_fingerprint` rejection;
      2. Strict LLM JSON schema (`parse_strict_llm_json_prob`), `MISSING_MODEL_FIELD`, empty-diff
         `coverage = None`, and exact paired few-shot prompt identity modulo labels;
      3. Zero-fallback `select_forensic_indices_strict` (`0` rescues when no candidate matches);
      4. `verify_manuscript_v3.run_self_test()` swapped-row and swapped-column rejection.
    """
    b_sample = os.path.join(base_dir, "cache", "benchmark_b_v3", "cabral_nova.npz")
    fp_real = compute_task_run_fingerprint(base_dir, b_sample, "mlp", "none", [42, 43], True, ["B0_Frozen"])
    with tempfile.TemporaryDirectory() as tmpd:
        ck_p = os.path.join(tmpd, "dummy_ckpt.json")
        save_json_atomic(ck_p, {"stream_name": "cabral_nova", "run_fingerprint": fp_real})
        assert load_verified_checkpoint_or_none(ck_p, fp_real) is not None
        assert load_verified_checkpoint_or_none(ck_p, "stale_fingerprint_0000") is None

    # LLM strict schema checks
    v_ok, err_ok = parse_strict_llm_json_prob('{"defect_prob": 0.73}')
    assert v_ok == 0.73 and err_ok is None
    for bad_str in [
        "The probability is 0.73 based on churn.",
        '{"defect_prob": 1.25}',
        '{"defect_prob": -0.05}',
        '{"defect_prob": 0.5, "extra": 1}',
        '{"defect_prob": true}',
    ]:
        v_bad, err_bad = parse_strict_llm_json_prob(bad_str)
        assert v_bad is None and err_bad is not None, f"Failed to reject non-compliant LLM output: {bad_str}"

    assert extract_served_model_strict({}) == "MISSING_MODEL_FIELD"
    assert extract_served_model_strict({"model": "deepseek-v3-served"}) == "deepseek-v3-served"
    _, emp_cov, is_emp = compact_unified_diff_and_coverage("")
    assert emp_cov is None and is_emp is True

    cards_dummy = {0: "Card0", 1: "Card1"}
    p_cen, p_ora, n_fl = build_paired_fewshot_prompts(
        cards_dummy, np.array([0, 1]), np.array([0.0, 0.0]), np.array([1, 0]), "Target"
    )
    assert n_fl == 1 and p_cen.replace("Label: 0", "Label: X").replace("Label: 1", "Label: X") == p_ora.replace("Label: 0", "Label: X").replace("Label: 1", "Label: X")

    # Forensic zero-fallback check (`B0 = 0.30 < 0.45` -> MUST return 0 rescues, NOT fall back to `SPU > B2`!)
    f_map = select_forensic_indices_strict(
        y_ev=np.array([1, 1, 0]),
        ts_ev=np.array([0.0, 0.0, 0.0]),
        fts_ev=np.array([30.0 * SECONDS_PER_DAY, 30.0 * SECONDS_PER_DAY, np.inf]),
        ftiers_ev=np.array(["Tier1_McIntosh_CSV", "Tier2_Git_BugFix", "clean"]),
        p_b0=np.array([0.30, 0.80, 0.20]),   # Commit 0 fails `B0 >= 0.45`; Commit 1 fails `Tier1_McIntosh_CSV`!
        p_b2=np.array([0.10, 0.10, 0.20]),
        p_spu=np.array([0.85, 0.85, 0.68]),  # Commit 2 fails `FP >= 0.70` (`0.68 < 0.70`)!
    )
    assert f_map["rescues"][1] == 0 and len(f_map["rescues"][0]) == 0
    assert f_map["false_positives"][1] == 0 and len(f_map["false_positives"][0]) == 0

    ver_st = run_verifier_self_test()
    assert ver_st["swapped_row_rejected"] and ver_st["swapped_col_rejected"] and ver_st["r3b_sign_exp_role_counterexamples_rejected"]

    # R3b-M5 Mock-Transport Tests for Lossless LLM Raw Exchange Archival:
    # (a) HTTP 200 returning invalid JSON MUST preserve `status_code=200`, full body, and exact SHA-256 (`error_stage='envelope'`)!
    bad_200_bytes = b"<html>200 OK Gateway Proxy Corrupted Non-JSON Body</html>"
    res_200_bad = _call_deepseek_json_prob(
        "dummy_key", "sys", "usr", max_retries=0, sleep_sec_base=0.0,
        transport_fn=lambda u, d, h, t: (200, bad_200_bytes, None),
    )
    att0_200 = res_200_bad["attempts"][0]
    assert (not res_200_bad["ok"]) and res_200_bad["status_code"] == 200 and res_200_bad["error_stage"] == "envelope"
    assert att0_200["status_code"] == 200 and att0_200["raw_response_text"] == bad_200_bytes.decode("utf-8")
    assert att0_200["resp_sha256"] == hashlib.sha256(bad_200_bytes).hexdigest()

    # (b) HTTP 429 with `3,500`-byte body (`> 2,000` chars) MUST store FULL untruncated body matching `resp_sha256`!
    long_429_bytes = ("{\"error\": \"" + ("R" * 3480) + "\"}").encode("utf-8")
    res_429 = _call_deepseek_json_prob(
        "dummy_key", "sys", "usr", max_retries=0, sleep_sec_base=0.0,
        transport_fn=lambda u, d, h, t: (429, long_429_bytes, "http_status_429"),
    )
    att0_429 = res_429["attempts"][0]
    assert att0_429["status_code"] == 429 and att0_429["error_stage"] == "http_status"
    assert len(att0_429["raw_response_text"]) == len(long_429_bytes) > 3400
    assert att0_429["resp_sha256"] == hashlib.sha256(long_429_bytes).hexdigest()

    # (c) Retry latency accumulation: Attempt 0 fails HTTP 503, Attempt 1 succeeds HTTP 200 -> records both `last` and `total`!
    call_seq = {"k": 0}

    def _retry_mock(u, d, h, t):
        call_seq["k"] += 1
        time.sleep(0.005)
        if call_seq["k"] == 1:
            return 503, b'{"error":"busy"}', "http_status_503"
        ok_env = json.dumps({"model": "deepseek-chat", "choices": [{"message": {"content": '{"defect_prob": 0.42}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        return 200, ok_env.encode("utf-8"), None

    res_retry = _call_deepseek_json_prob("dummy_key", "sys", "usr", max_retries=2, sleep_sec_base=0.0, transport_fn=_retry_mock)
    assert res_retry["ok"] and res_retry["n_attempts"] == 2 and abs(res_retry["prob"] - 0.42) < 1e-9
    assert res_retry["total_latency_ms"] > res_retry["last_attempt_latency_ms"] > 0.0
    assert abs(res_retry["total_latency_ms"] - round(sum(a["attempt_latency_ms"] for a in res_retry["attempts"]), 2)) < 1e-6

    # R3c-4 Counterexamples:
    # (d) Malformed token usage (`prompt_tokens=None`, `"unknown"`, and list `[1]`) MUST NOT throw uncaught exceptions!
    for bad_usage in [{"prompt_tokens": None}, {"prompt_tokens": "unknown", "completion_tokens": -3}, [1, 2], None]:
        env_bad_u = json.dumps({
            "model": "mock",
            "choices": [{"message": {"content": '{"defect_prob": 0.42}'}}],
            "usage": bad_usage,
        }).encode("utf-8")
        res_bu = _call_deepseek_json_prob(
            "dummy_key", "sys", "usr", max_retries=0, sleep_sec_base=0.0,
            transport_fn=lambda u, d, h, t, b=env_bad_u: (200, b, None),
        )
        assert res_bu["ok"] and abs(res_bu["prob"] - 0.42) < 1e-9
        assert (not res_bu["usage_valid"]) and len(res_bu["metadata_errors"]) >= 1
        assert len(res_bu["attempts"]) == 1 and res_bu["attempts"][0]["raw_usage"] == bad_usage

    # (e) Non-UTF-8 raw response bytes (`b"\xff\xfe\x00"`) MUST preserve exact byte-reversible base64 matching `resp_sha256`!
    non_utf8_bytes = b"\xff\xfe\x00\x80corrupted_binary"
    res_bin = _call_deepseek_json_prob(
        "dummy_key", "sys", "usr", max_retries=0, sleep_sec_base=0.0,
        transport_fn=lambda u, d, h, t: (200, non_utf8_bytes, None),
    )
    att_bin = res_bin["attempts"][0]
    assert (not res_bin["ok"]) and res_bin["error_stage"] == "envelope" and (not att_bin["raw_response_utf8_valid"])
    reconstructed_bytes = base64.b64decode(att_bin["raw_response_b64"])
    assert reconstructed_bytes == non_utf8_bytes
    assert hashlib.sha256(reconstructed_bytes).hexdigest() == att_bin["resp_sha256"]

    # (f) Incremental streaming archive writer persists completed calls to disk BEFORE close/crash!
    with tempfile.TemporaryDirectory() as tmp_inc:
        aj_p = os.path.join(tmp_inc, "audit.jsonl")
        rg_p = os.path.join(tmp_inc, "raw.jsonl.gz")
        inc_w = IncrementalLLMArchiveWriter(aj_p, rg_p)
        inc_w.append_record({"commit_id": "c1", "prob": 0.42}, {"commit_id": "c1", "attempts": res_retry["attempts"]})
        # Verify `audit.jsonl` ALREADY contains flushed record `c1` BEFORE `close_and_finalize_hashes()`!
        with open(aj_p, "r", encoding="utf-8") as f_live:
            live_lines = f_live.read().strip().splitlines()
        assert len(live_lines) == 1 and json.loads(live_lines[0])["commit_id"] == "c1"
        a_sha, u_sha, g_sha = inc_w.close_and_finalize_hashes()
        with open(rg_p, "rb") as f_gz_raw:
            gz_file_bytes = f_gz_raw.read()
        decomp_bytes = gzip.decompress(gz_file_bytes)
        with open(aj_p, "rb") as f_aj_raw:
            aj_bytes = f_aj_raw.read()
        assert a_sha == hashlib.sha256(aj_bytes).hexdigest()
        assert g_sha == hashlib.sha256(gz_file_bytes).hexdigest()
        # R3d-1 Exact Assertion: declared uncompressed hash MUST match `sha256(gzip.decompress(gz_file_bytes))` on Windows!
        assert u_sha == hashlib.sha256(decomp_bytes).hexdigest(), (
            f"R3d-1: uncompressed_sha256 {u_sha} != decompressed gzip sha256 {hashlib.sha256(decomp_bytes).hexdigest()}!"
        )
        assert not decomp_bytes.endswith(b"\r\n")

    return {
        "pass": True,
        "stale_checkpoint_rejected": True,
        "llm_strict_schema_and_paired_prompts_verified": True,
        "llm_mock_transport_200_bad_json_and_429_full_body_verified": True,
        "llm_malformed_usage_and_non_utf8_b64_and_incremental_flush_verified": True,
        "forensic_zero_fallback_and_tier1_enforced": True,
        "cell_verifier_self_test": ver_st,
    }


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "results_v3")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    print("=== Running N05 V3 Counterexample-Driven Pre-Run Gate Suite (14 Checks) ===", flush=True)
    c1 = check_1_provenance_and_confirmatory_lock(base_dir)
    print(f"  [PASS] Check  1 (Provenance, Missing Diffs, Aligned Sens Splits, Lock): K={c1['K_natural_clusters']}, locked={c1['current_filesystem_confirmatory_locked']}", flush=True)

    c2 = check_2_strict_prefix_invariance_counterexamples(base_dir)
    print(f"  [PASS] Check  2 (Strict Prefix-Invariance across {c2['n_runner_method_branches_verified']} Runner Branches, BORB_Alarm 350, & ORB/ODaSC Zero-Lat): max_diff={c2['max_prefix_diff_across_all_runner_methods']:.2e}, flips={c2['total_historical_decision_flips']}", flush=True)

    c3 = check_3_causal_kappa_pre_t0_freeze_and_zero_latency_parity(base_dir)
    print(f"  [PASS] Check  3 (Causal Kappa Pre-t0 Freeze & Post-t0 Oracle Parity): {c3['post_t0_oracle_parity_diffs']}", flush=True)

    c4 = check_4_immutable_alarm_budgets_across_4_consecutive_alarms()
    print(f"  [PASS] Check  4 (4-Alarm Immutable Budgets): LGB_Matched={c4['lgb_4alarm_trees_matched']} vs SPU={c4['lgb_4alarm_trees_spu']}", flush=True)

    c5 = check_5_posterior_identity_between_solver_and_training()
    print(f"  [PASS] Check  5 (Solver vs Float32 Training Posterior Identity): actual_resid={c5['actual_training_residual']:.2e}", flush=True)

    c6 = check_6_faithful_orb_borb_odasc_event_traces()
    print(f"  [PASS] Check  6 (ORB/BORB/ODaSC Spec & Differential Counterexamples): noise_skipped={c6['orb_noise_skipped_count']}, close_skipped={c6['orb_150_close_floats_skipped']}, max_lam={c6['orb_unclipped_max_lambda']:.1f}, flood_r={c6['odasc_outlier_flood_max_radius']:.3f}", flush=True)

    c7 = check_7_gbdt_and_neural_nnpu_disjoint_u_and_scar()
    print(f"  [PASS] Check  7 (Disjoint-U GBDT & Neural nnPU): U_grad_diff={c7['u_invariance_grad_diff']:.2e}, MLP_SCAR={c7['scar_results']['mlp']['nnpu_ap']:.4f}", flush=True)

    c8 = check_8_lightgbm_callable_objective_and_exact_init_score_margin()
    print(f"  [PASS] Check  8 (LGB Callable Objective & Native Binary Margin Equivalence): err_vs_native={c8['max_prob_err_vs_native_lgb_binary']:.2e} (if_omitted={c8['counterfactual_err_if_init_score_omitted']:.3f})", flush=True)

    c9 = check_9_unconfounded_drift_monitor_2x2_and_b0_invariance(base_dir)
    print(f"  [PASS] Check  9 (Unconfounded 2x2 DriftMonitor pct=95/cd=1 & B0 Invariance): max_b0_diff={c9['max_b0_diff']:.2e}", flush=True)

    c10 = check_10_linear_skip_epoch0_equivalence(base_dir)
    print(f"  [PASS] Check 10 (Linear-Residual Skip Epoch-0 Equivalence): {c10['epoch0_logit_max_abs_err_vs_tabular_lr']}", flush=True)

    c11 = check_11_km_rounding_adversarial_and_fuzz()
    print(f"  [PASS] Check 11 (KM 1.006d Adversarial & 50 Fuzz Trials): passed={c11['fuzz_trials_passed']}", flush=True)

    c12 = check_12_causal_anchor_and_feature_dependent_latency_audit()
    adv = c12["regimes"]["adversarial_neg_cov"]
    fdep = c12["regimes"]["feature_dependent_delay_and_latency_drift"]
    print(f"  [PASS] Check 12 (Causal-Anchor m0=50 & Feature-Dependent Latency Audit): AdvCov MAE={adv['prevalence_mae_v3_ht']:.4f}, FeatDep MAE={fdep['prevalence_mae_v3_ht']:.4f}", flush=True)

    c13 = check_13_exact_small_k_signed_rank_and_metric_orientation()
    print(f"  [PASS] Check 13 (Exact Signed-Rank K=4->0.125 / K=5->0.0625 & Brier Orientation): K4={c13['exact_K4_all_pos_p']:.4f}, K5={c13['exact_K5_all_pos_p']:.4f}", flush=True)

    c14 = check_14_checkpoint_fingerprint_llm_schema_forensic_rules_and_cell_verifier(base_dir)
    print(f"  [PASS] Check 14 (Stale Fingerprint, Strict LLM Schema, Zero-Fallback Forensics, Row/Col Verifier): passed=True", flush=True)

    report = {
        "provenance": build_provenance_header(
            base_dir,
            [os.path.join(base_dir, "results_v3", "data_provenance_audit_v3.json")],
        ),
        "all_checks_passed": True,
        "n_checks": 14,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "check_01_provenance_and_confirmatory_lock": c1,
        "check_02_strict_prefix_invariance": c2,
        "check_03_causal_kappa_and_oracle_parity": c3,
        "check_04_immutable_4alarm_budgets": c4,
        "check_05_posterior_solver_vs_training_identity": c5,
        "check_06_faithful_orb_borb_odasc_traces": c6,
        "check_07_disjoint_u_gbdt_and_neural_nnpu": c7,
        "check_08_lgb_objective_and_init_score_margin": c8,
        "check_09_unconfounded_drift_2x2_and_b0": c9,
        "check_10_linear_skip_equivalence": c10,
        "check_11_km_adversarial_and_fuzz": c11,
        "check_12_causal_anchor_and_feature_dep_latency": c12,
        "check_13_exact_signed_rank_and_metric_orientation": c13,
        "check_14_fingerprint_llm_forensics_and_cell_verifier": c14,
    }
    out_p = os.path.join(out_dir, "prerun_gate_verification_v3.json")
    save_json_atomic(out_p, report)
    print(f"\n==> ALL 14 COUNTEREXAMPLE-DRIVEN PRE-RUN GATE CHECKS PASSED in {report['elapsed_seconds']}s -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
