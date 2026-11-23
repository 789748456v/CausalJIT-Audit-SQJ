r"""
Post-Campaign Runnable Mechanism & Historical Pre-Unlock Lock Verifier
(`scripts_v3/run_post_campaign_mechanism_checks_v3.py`).

Verifies the completed replication package without modifying `src_v3/verify_mechanisms_and_protocol_v3.py`:
  - `src_v3/verify_mechanisms_and_protocol_v3.py` is the historical PRE-UNLOCK gate executed BEFORE
    unlocking the 5 confirmatory ApacheJIT repositories (`check_1` intentionally asserts that
    `cache/confirmatory_v3` and `data_sync/confirmatory_apachejit_raw` do NOT yet exist).
  - On the completed replication package (where confirmatory bundles are already unlocked), this
    entry point:
      1. Verifies the archived pre-unlock gate report (`results_v3/prerun_gate_archive/prerun_gate_verification_v3.json`)
         against its frozen SHA-256 (`766e3fd64c45a3718ad42c86c44f0b75703d139929d1a72dbd44e0019f492307`),
         verifies that `check_1`'s pre-unlock guard still actively rejects an unlocked workspace,
         and verifies all Benchmark A, Benchmark B, and Confirmatory provenance invariants;
      2. Runs all 13 live counterexample mechanism checks (`Check 2` .. `Check 14`) directly from
         `src_v3/verify_mechanisms_and_protocol_v3.py` on the completed package without overwriting
         any file inside `results_v3/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from config_v3 import EXCLUDED_GERRIT_REVIEW_COLS, NATURAL_STREAM_TO_DONOR_CLUSTER, PURE_COMMIT_COLS_QT_OS, V3_CONFIG
from verify_mechanisms_and_protocol_v3 import (
    check_1_provenance_and_confirmatory_lock,
    check_2_strict_prefix_invariance_counterexamples,
    check_3_causal_kappa_pre_t0_freeze_and_zero_latency_parity,
    check_4_immutable_alarm_budgets_across_4_consecutive_alarms,
    check_5_posterior_identity_between_solver_and_training,
    check_6_faithful_orb_borb_odasc_event_traces,
    check_7_gbdt_and_neural_nnpu_disjoint_u_and_scar,
    check_8_lightgbm_callable_objective_and_exact_init_score_margin,
    check_9_unconfounded_drift_monitor_2x2_and_b0_invariance,
    check_10_linear_skip_epoch0_equivalence,
    check_11_km_rounding_adversarial_and_fuzz,
    check_12_causal_anchor_and_feature_dependent_latency_audit,
    check_13_exact_small_k_signed_rank_and_metric_orientation,
    check_14_checkpoint_fingerprint_llm_schema_forensic_rules_and_cell_verifier,
)

EXPECTED_PRERUN_ARCHIVE_SHA256 = "766e3fd64c45a3718ad42c86c44f0b75703d139929d1a72dbd44e0019f492307"


def verify_historical_lock_and_post_unlock_provenance(base_dir: str) -> dict:
    arch_p = os.path.join(base_dir, "results_v3", "prerun_gate_archive", "prerun_gate_verification_v3.json")
    assert os.path.isfile(arch_p), f"Missing archived pre-run gate report: {arch_p}"
    with open(arch_p, "rb") as fb:
        raw_b = fb.read()
    got_sha = hashlib.sha256(raw_b).hexdigest()
    assert got_sha == EXPECTED_PRERUN_ARCHIVE_SHA256, (
        f"Pre-run gate archive SHA-256 mismatch: got {got_sha}, expected {EXPECTED_PRERUN_ARCHIVE_SHA256}"
    )
    arch_j = json.loads(raw_b.decode("utf-8"))
    assert arch_j["all_checks_passed"] is True and int(arch_j["n_checks"]) == 14
    assert arch_j["check_01_provenance_and_confirmatory_lock"]["current_filesystem_confirmatory_locked"] is True

    # Confirm that `check_1_provenance_and_confirmatory_lock` still enforces the pre-unlock guard on unlocked package
    guard_active = False
    try:
        check_1_provenance_and_confirmatory_lock(base_dir)
    except AssertionError as exc:
        if "Confirmatory cohort path already exists" in str(exc):
            guard_active = True
        else:
            raise
    assert guard_active, "Expected check_1 pre-unlock guard to fire on completed post-unlock package!"

    # Verify post-campaign provenance manifests (`data_provenance_audit_v3.json` & `confirmatory_provenance_v3.json`)
    with open(os.path.join(base_dir, "results_v3", "data_provenance_audit_v3.json"), "r", encoding="utf-8") as f:
        audit = json.load(f)
    with open(os.path.join(base_dir, "results_v3", "confirmatory_provenance_v3.json"), "r", encoding="utf-8") as f:
        conf_prov = json.load(f)

    assert json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True)) == json.loads(
        json.dumps(audit["provenance"]["v3_config"], sort_keys=True)
    )
    for st in audit["benchmark_a"]["primary_streams"]:
        assert st["feature_cols"] == PURE_COMMIT_COLS_QT_OS
        assert len(set(st["feature_cols"]) & set(EXCLUDED_GERRIT_REVIEW_COLS)) == 0
    assert len(set(NATURAL_STREAM_TO_DONOR_CLUSTER.values())) == 25
    assert len(conf_prov["confirmatory_streams"]) == 5 and int(conf_prov["n_umbrella_clusters"]) == 4

    return {
        "archived_prerun_gate_sha256_verified": got_sha,
        "pre_unlock_guard_still_enforced": True,
        "dev_and_confirmatory_provenance_verified": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    print("=== Running Post-Campaign Mechanism & Historical Pre-Unlock Verification Suite ===", flush=True)
    c1 = verify_historical_lock_and_post_unlock_provenance(BASE_DIR)
    print(f"  [PASS] Step  1 (Historical Pre-Unlock Archive SHA-256 & Post-Unlock Provenance): {c1['archived_prerun_gate_sha256_verified'][:16]}...", flush=True)

    c2 = check_2_strict_prefix_invariance_counterexamples(BASE_DIR)
    print(f"  [PASS] Check 2 (Strict Prefix-Invariance across {c2['n_runner_method_branches_verified']} Runner Branches): max_diff={c2['max_prefix_diff_across_all_runner_methods']:.2e}", flush=True)

    c3 = check_3_causal_kappa_pre_t0_freeze_and_zero_latency_parity(BASE_DIR)
    print(f"  [PASS] Check 3 (Causal Kappa Pre-t0 Freeze & Oracle Parity): {c3['post_t0_oracle_parity_diffs']}", flush=True)

    c4 = check_4_immutable_alarm_budgets_across_4_consecutive_alarms()
    print(f"  [PASS] Check 4 (Immutable 4-Alarm Budgets): LGB_Matched={c4['lgb_4alarm_trees_matched']}", flush=True)

    c5 = check_5_posterior_identity_between_solver_and_training()
    print(f"  [PASS] Check 5 (Solver vs Float32 Training Posterior Identity): resid={c5['actual_training_residual']:.2e}", flush=True)

    c6 = check_6_faithful_orb_borb_odasc_event_traces()
    print(f"  [PASS] Check 6 (ORB/BORB/ODaSC Event Traces): noise_skipped={c6['orb_noise_skipped_count']}", flush=True)

    c7 = check_7_gbdt_and_neural_nnpu_disjoint_u_and_scar()
    print(f"  [PASS] Check 7 (Disjoint-U GBDT & Neural nnPU): U_grad_diff={c7['u_invariance_grad_diff']:.2e}", flush=True)

    c8 = check_8_lightgbm_callable_objective_and_exact_init_score_margin()
    print(f"  [PASS] Check 8 (LGB Callable Objective & Native Margin Equivalence): err={c8['max_prob_err_vs_native_lgb_binary']:.2e}", flush=True)

    c9 = check_9_unconfounded_drift_monitor_2x2_and_b0_invariance(BASE_DIR)
    print(f"  [PASS] Check 9 (Unconfounded 2x2 DriftMonitor & B0 Invariance): max_b0_diff={c9['max_b0_diff']:.2e}", flush=True)

    # Check 10: Use full multimodal token + CodeBERT caches when present (Tier-3 Zenodo archive),
    # or exact tabular-real + deterministic multimodal embedding fallback on lightweight GitHub clones.
    tok_p = os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "qt_natural_25k_tok.npz")
    cb_p = os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "qt_natural_25k_cb.npz")
    if os.path.isfile(tok_p) and os.path.isfile(cb_p):
        c10 = check_10_linear_skip_epoch0_equivalence(BASE_DIR)
        print(f"  [PASS] Check 10 (Linear-Residual Skip Epoch-0 Equivalence): {c10['epoch0_logit_max_abs_err_vs_tabular_lr']}", flush=True)
    else:
        import numpy as np
        from models_and_baselines_v3 import (
            build_host_backbone_v3,
            fit_causal_tabular_lr_weights,
            observe_labels_at_time,
            predict_logits_and_embeddings_v3,
        )

        with open(os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
            mm_meta = {m["name"]: m for m in json.load(f)}["qt_natural_25k"]
        b_tab = np.load(os.path.join(BASE_DIR, "cache", "multimodal_v3_pure13", "qt_natural_25k_tab.npz"))
        n_sl = 2000
        t0_c10 = float(b_tab["commit_ts"][n_sl - 1])
        _, y_obs = observe_labels_at_time(b_tab["commit_ts"][:n_sl], b_tab["y_true"][:n_sl], b_tab["fix_ts"][:n_sl], t0_c10)
        x_tab_real = b_tab["X_tab"][:n_sl].astype(np.float32)
        pw = 4.0
        w_lr, b_lr = fit_causal_tabular_lr_weights(x_tab_real, y_obs, pos_weight=pw, seed=42)
        lr_logits = (x_tab_real[:256] @ w_lr) + b_lr

        rng = np.random.RandomState(42)
        tok_meta = mm_meta["tok_meta"]
        cb_meta = mm_meta["cb_meta"]
        tok_dim = int(tok_meta["msg_len"]) + int(tok_meta["code_len"])
        cb_dim = 2 * int(cb_meta["cb_dim"])
        x_tok_syn = np.concatenate(
            [rng.randint(1, int(tok_meta["vocab_msg"]), size=(n_sl, tok_dim)).astype(np.float32), x_tab_real],
            axis=1,
        )
        x_cb_syn = np.concatenate(
            [rng.randn(n_sl, cb_dim).astype(np.float32) * 0.1, x_tab_real],
            axis=1,
        )
        max_errs = {}
        for h_type, x_mat in [("deepjit_cnn", x_tok_syn), ("att_cnn", x_tok_syn), ("codebert", x_cb_syn)]:
            m_h = build_host_backbone_v3(h_type, x_mat[:16], meta=mm_meta, skip_mode="frozen_lr", seed=42)
            m_h.init_linear_skip_from_data(x_mat[:n_sl], y_obs, pos_weight=pw, seed=42)
            lg_m, _ = predict_logits_and_embeddings_v3(m_h, x_mat[:256])
            err = float(np.max(np.abs(lg_m - lr_logits)))
            max_errs[h_type] = err
            assert err < 1e-4, f"Linear skip epoch-0 equivalence failed for {h_type}: err={err}"
        print(f"  [PASS] Check 10 (Linear-Residual Skip Epoch-0 Equivalence [GitHub lightweight mode]): {max_errs}", flush=True)

    c11 = check_11_km_rounding_adversarial_and_fuzz()
    print(f"  [PASS] Check 11 (KM 1.006d Adversarial & 50 Fuzz Trials): passed={c11['fuzz_trials_passed']}", flush=True)

    c12 = check_12_causal_anchor_and_feature_dependent_latency_audit()
    print(f"  [PASS] Check 12 (Causal-Anchor m0=50 & Feature-Dependent Latency Audit): passed=True", flush=True)

    c13 = check_13_exact_small_k_signed_rank_and_metric_orientation()
    print(f"  [PASS] Check 13 (Exact Signed-Rank K=4->0.125 / K=5->0.0625 & Brier Orientation): passed=True", flush=True)

    c14 = check_14_checkpoint_fingerprint_llm_schema_forensic_rules_and_cell_verifier(BASE_DIR)
    print(f"  [PASS] Check 14 (Stale Fingerprint, Strict LLM Schema, Zero-Fallback Forensics, Cell Verifier): passed={c14['pass']}", flush=True)

    print(f"\n==> ALL POST-CAMPAIGN MECHANISM & HISTORICAL LOCK CHECKS PASSED in {time.perf_counter() - t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
