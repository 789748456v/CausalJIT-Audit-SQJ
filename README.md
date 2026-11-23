# Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction

Official replication package and automated verification suite for the manuscript submitted to *Software Quality Journal* (Springer Nature).

**Authors:** Zhijie Li¹ · Shaoyang Xu¹ · Ketong Liu¹ · Shibo Meng¹ · Feiran Gao² · Yuli Zhao¹\*  
¹ Software College, Northeastern University, Shenyang 110169, China  
² School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China  
\* Corresponding author: `zhaoyl@swc.neu.edu.cn`

---

## Study Overview

Unsupervised drift monitors such as DriftLens detect embedding-space shifts in Just-in-Time Software Defect Prediction (JIT-SDP) without defect labels, motivating immediate window retraining. Yet label-free monitoring eliminates label dependence only when deciding *when* to retrain—not when *supervising* weight updates. Under multi-month SZZ verification latency, most genuinely bug-inducing commits inside newly flagged windows remain unfixed at alarm time ($t_i^{\mathrm{fix}} > t_{\mathrm{alarm}}$) and appear clean under strictly causal deployment.

This replication package accompanies our causal supervision audit and cross-paradigm operating-boundary study spanning **37 development streams** and **5 confirmatory ApacheJIT repositories** (**34 natural repository streams** and **8 controlled-shift streams**; **347,122 natural commits**), **29 evaluated methods** (26 deployable + 3 hindsight references across 41 prefix-invariant execution branches), and a **6,400-call paired Code-LLM prompting case study** (`deepseek-flash`).

### Key Headline Findings

- **Severe Alarm-Time Right-Censoring & Hindsight Optimism (RQ1):** Across scheduled 400-commit ($1W$) boundaries in multi-year ecosystems, `68.8%–94.6%` of window defects remain right-censored (`65.8%–95.7%` at fired alarms; `45.1%–80.5%` even at $4W$), whereas the shorter confirmatory cohort exhibits lower censoring (`26.4%` across 5 repositories; `18.0%` on unsplit Standalone-3). Leaking future SZZ fixes into matched window-oracle updates (`B1b*` vs. causal `B2-4W`) significantly inflates whole-stream Average Precision (AP) across $K=25$ independent donor clusters on both MLP (`+0.0237`, `20/0/5`, $p_{\mathrm{Holm}}=6.63\times 10^{-5}$) and LightGBM (`+0.0476`, `23/0/2`, $p_{\mathrm{Holm}}=3.70\times 10^{-5}$).
- **Cross-Paradigm Operating Boundaries, Granularity, & Confirmatory Non-Replication (RQ2):** No single adaptive paradigm dominates across regimes. Frozen 13-metric logistic regression (`Full-LR Frozen = 0.3653` AP) leads every deployable multimodal/tree host on Benchmark A. On gradual $K=25$ whole-stream AP, `Full-LR Adapt-4W` (`0.5108`, $p_{\mathrm{Holm}}=0.0639$) leads in mean while `B2-4W` (`0.5004`, $p_{\mathrm{Holm}}=0.0056$), `Wait-90` (`0.4982`, $p_{\mathrm{Holm}}=0.0069$), and zero-fit `FixedCurve-PU` (`0.4942`, $p_{\mathrm{Holm}}=0.0295$) significantly exceed `Survival-PU` (`0.4866`)—though within-window score standardization on identical 5-seed predictions reverses the mean `B2-4W` gap non-significantly (`Rank-Norm: 0.5027 vs. 0.4989`, unadjusted $p=0.0877$). On the short confirmatory cohort, disclosing upstream `apachejit_total.csv`'s `11,964`-commit clean `apache/hadoop` split, MLP hindsight optimism does not replicate on unsplit Standalone-3 (`0.5951 vs. 0.5985`; `ODaSC = 0.7857`) or frozen Umbrella-4 (`0.6801 vs. 0.6858`; `ODaSC = 0.8289`). Conversely, `Survival-PU` leads full post-shift AP (`0.5764`) and LightGBM Stage-3 distillation (`+0.0379` over `Teacher-Only Matched`, $p_{\mathrm{Holm}}=9.87\times 10^{-4}$).
- **Unrevealed-Sample Audit & Non-Monotonic Cross-Stage Error Propagation (RQ3):** Across 62 fired DriftLens alarms (`74,400` pool evaluations / `59,600` unique commits), adaptive cohort incidence improves Stage-1 maturity $\hat{c}(a)$ MAE over `FixedCurve-PU` (`0.0938` vs. `0.1204`) and reduces strictly unrevealed ($y_{\mathrm{obs}}=0$) soft-target Brier over de-weighted teachers (`0.1303 -> 0.1236`), yet `FixedCurve-PU` attains slightly lower Stage-2 prior MAE (`0.0565` vs. `0.0620`) and unrevealed Brier (`0.1200`).
- **Extreme Latency, Post-Hoc Controls, Block Calibration, & Budget Granularity (RQ4–RQ5):** Under $\kappa=3.0\times$ latency, `FixedCurve-PU` leads AP (`0.5547`) and raw Brier/ECE (`0.1693 / 0.0973`), while `Survival-PU` beats `B2-4W`, alarm-synchronized `PostHoc Logit`, Proposition-1-inspired plug-in `PostHoc Ratio`, and `FixedCurve-PU` on scale-fair threshold-tuned $F_{1,\mathrm{th}}$ (`0.5022` raw; `0.4898` Platt) and avoids `FixedCurve-PU`'s $\kappa=0\times$ misspecification (`0.5760 vs. 0.5544` AP). Standard 90-day Platt calibration worsens calibration relative to raw probabilities under `65.9%` calibration-pool censoring. Circular block-bootstrap DriftLens cuts validation-pool null alarms `4.8x` and detects abrupt shifts `3x` faster than periodic `3W` retraining (`350` vs. `1,050` commits; `0/20` missed). Operational per-window 400-commit budgets reverse most steep whole-stream net-TP losses on Benchmark A while remaining host-dependent.

---

## Manuscript-to-Artifact Crosswalk

Every empirical table cell (`1,452` cells across `22` empirical tables), locked configuration field (`33` fields in `V3_CONFIG`), and key prose comparison (`34` exact-path claims + `60` relational/anti-misstatement guards) is verified automatically against the archived JSON artifacts in this repository.

| Manuscript Element | Backing Artifact File(s) in This Repository | Generator / Verifier Script |
| :--- | :--- | :--- |
| **Table 1** (Paradigm Positioning) | Conceptual positioning table in `manuscript/main.tex` | `scripts_v3/build_manuscript_v3.py` |
| **Table 2** (Locked Protocol Parameters) | `src_v3/config_v3.py` (`V3_CONFIG`, 33 locked fields) | `scripts_v3/verify_prose_claims_v3.py` |
| **Table 3** (RQ1 Verification Latency & Censoring Audit) | `results_v3/data_provenance_audit_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 4** (RQ2 Benchmark A Multi-Host & Track-2 Audit) | `results_v3/benchmark_a_multihost_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 5** (RQ2 Benchmark A Provenance/Feature Sensitivity) | `results_v3/benchmark_a_multihost_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 6** (RQ2 Benchmark B $K=25$ / $K=24$ / LightGBM / Shift) | `results_v3/benchmark_b_main_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 7** (RQ2 Holm–Bonferroni Paired Inference, $K=25$) | `results_v3/benchmark_b_main_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 8** (RQ2 5-Seed Per-Window AP & Scale Removal) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_f_window_vs_whole_ap`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 9** (RQ2 Untouched Confirmatory ApacheJIT Cohort) | `results_v3/confirmatory_cohort_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 10** (RQ3 Maturity Estimators & Component Ablations) | `results_v3/rq3_ablations_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 11** (RQ4a Future-Only Latency Scaling, $\kappa \in [0, 3]$) | `results_v3/rq4a_latency_intervention_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 12** (RQ4a Stage-1 Control: `FixedCurve-PU` vs. `Survival-PU`) | `results_v3/rq4a_fixedcurve_control_v3.json` | `scripts_v3/run_rq4a_fixedcurve_control_v3.py` |
| **Table 13** (RQ4a Raw vs. 90d Platt & Post-Hoc $B_2$ Controls) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_d_calibration_and_posthoc`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 14** (RQ4b $2\times 2$ DriftLens Calibration Factorial) | `results_v3/rq4b_factorial_monitors_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 15** (RQ4c $2\times 2$ Semi-Synthetic PR-SZZ Perturbation) | `results_v3/rq4d_szz_perturbation_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 16** (RQ5 Phase-Decomposed Wall-Clock Runtime) | `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 17** (RQ5 Paired Whole-Stream vs. Per-Window Budgets) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_g_paired_net_gain`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 18** (Synthesis: Observed Patterns vs. Hypotheses) | Synthesized from Tables 3–17 | `scripts_v3/build_manuscript_v3.py` |
| **Table A1** (Executed Host Architectures & Budgets) | `src_v3/models_and_baselines_v3.py` | `scripts_v3/build_manuscript_v3.py` |
| **Table B1** (Confirmatory ApacheJIT Data-Construction Funnel) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_a_apachejit_funnel`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B2** (34-Stream Archive Follow-Up Proxy Audit) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_b_followup_cutoffs`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B3** (Benchmark A $\ge 180$d / $\ge 365$d Sensitivity) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_b_followup_cutoffs`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B4** (62-Alarm Unrevealed Stage 1–3 Mechanism Audit) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_c_stage123_mechanisms`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B5** (Per-Stream Breakdown of 4 Controlled Shifts) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_e_controlled_shifts`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B6** (RQ4e Audited Code-LLM Prompting Study) | `results_v3/rq4f_llm_audit_v3.json`, `logs/llm_raw_exchange_v3.jsonl.gz` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table B7** (RQ5 Pre-Declared Qt Tier-1 SZZ Forensics) | `results_v3/rq5_natural_forensics_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Fig. 1–2** (Causal Timeline & Three-Stage Architecture) | `figures/fig0{a_causal_timeline,b_survival_pu_architecture}_v3.{pdf,png}` | `scripts_v3/generate_conceptual_diagrams_v3.py` |
| **Fig. 3–10** (Empirical RQ1–RQ5 Figures) | `figures/fig{1..8}_*_v3.{pdf,png}` | `scripts_v3/analyze_and_plot_v3.py` |

> **Artifact Naming Conventions:** To preserve exact SHA-256 provenance locks across all immutable `src_v3/` modules and archived `results_v3/*_v3.json` files, internal identifiers retain their frozen execution names: `v3` marks the locked V3 experimental campaign; `n60_*` denotes the five PR-SZZ streams from Bludau & Pretschner (2026); historical JSON filenames `rq4d_szz_perturbation_v3.json`, `rq4e_sensitivity_sweeps_v3.json`, and `rq4f_llm_audit_v3.json` map to manuscript subsections **RQ4c**, **RQ4d**, and **RQ4e**; and `CAUSALJIT_BASE_DIR` is an optional environment variable overriding auto-detected repository root paths in post-processing scripts.

---

## Repository Structure

```text
CausalJIT-Audit-SQJ/
├── README.md
├── LICENSE                                         # MIT License (code/scripts)
├── requirements_lock_v3.txt                        # Exact pinned Python environment (Python 3.13 / CUDA 12.6)
├── src_v3/                                         # [Immutable, SHA-256 provenance-locked core implementation]
│   ├── config_v3.py                                # Frozen V3_CONFIG, donor-cluster maps (K=25/24), codified boundaries
│   ├── data_audit_v3.py                            # Leak-free stream builder, feature filtering, confirmatory unlocker
│   ├── build_codebert_cache_v3.py                  # Offline microsoft/codebert-base feature cache builder
│   ├── survival_estimators_v3.py                   # CohortIncidence, 1e-4d KM cure, Weibull cure, FixedCurve-PU, bisection
│   ├── models_and_baselines_v3.py                  # Att-CNN, CodeBERT, DeepJIT-CNN, MLP, LightGBMHostV3, 2x2 DriftLens
│   ├── stream_competitors_v3.py                    # Prefix-invariant ORB, BORB, ODaSC, Full-LR, 3-branch causal calibration
│   ├── metrics_and_stats_v3.py                     # Horizon evaluator, exact/Pratt paired Wilcoxon, Holm-Bonferroni
│   ├── run_v3_experiments.py                       # Master Phase A / B / mechanisms / confirmatory runner
│   ├── run_v3_llm_and_forensics.py                 # Audited Code-LLM prompting (RQ4e) & Tier-1 forensics (RQ5) runner
│   ├── verify_mechanisms_and_protocol_v3.py        # Historical pre-unlock 14-check gate (archived SHA-256 766e3fd6...)
│   └── verify_manuscript_v3.py                     # [Tier 1] SHA-256 provenance, .npz, BibTeX & 1,452-cell verifier
├── scripts_v3/                                     # [Portable companion audits, plotting, assembly & verification]
│   ├── run_post_campaign_mechanism_checks_v3.py    # [Tier 1] Verifies pre-unlock lock + runs live Checks 2-14 (41 branches)
│   ├── verify_prose_claims_v3.py                   # [Tier 1] Verifies V3_CONFIG, 34 exact-path claims & 60 relational guards
│   ├── generate_conceptual_diagrams_v3.py          # [Tier 1] Regenerates conceptual vector/PNG diagrams (Fig. 1-2)
│   ├── analyze_and_plot_v3.py                      # [Tier 1] Regenerates Fig. 3-10, 22 empirical tables & 1,452-cell registry
│   ├── build_manuscript_v3.py                      # [Tier 1] Assembles manuscript/main.tex, Markdown summary & README.md
│   ├── run_rq4a_fixedcurve_control_v3.py           # [Tier 2] Matched RQ4a zero-fit FixedCurve-PU vs. Survival-PU control
│   ├── run_reviewer_mechanism_audits_v3.py         # [Tier 2] Audits A-G (funnel, follow-up, Stage 1-3, PostHoc, WinAP, budgets)
│   ├── run_parallel_campaign_v3.py                 # [Tier 3] Multi-worker campaign orchestrator over src_v3 runners
│   └── package_sqj_submission_v3.py                # Release bundler & clean-room LaTeX compilation verifier
├── cache/
│   ├── benchmark_b_v3/                             # 27 natural + 4 controlled-shift tabular .npz bundles + SHA-256 manifest
│   ├── confirmatory_v3/                            # 5 confirmatory ApacheJIT .npz bundles + SHA-256 manifest
│   ├── multimodal_v3_pure13/                       # 6 Benchmark A tabular *_tab.npz bundles + manifest (tokens/CB in Zenodo)
│   ├── multimodal_v3_sens/                         # Benchmark A provenance/feature sensitivity manifest
│   └── codebert_v3/                                # CodeBERT cache SHA-256 manifest (768-D embeddings in Zenodo)
├── results_v3/
│   ├── *_v3.json                                   # All 15 provenance-stamped JSON outputs & 1,452-cell registry
│   ├── prerun_gate_archive/                        # Frozen pre-unlock 14-check gate report (SHA-256 766e3fd6...)
│   ├── commit_predictions/                         # 50 commit-level seed-0 .npz archives (385,620 commits)
│   ├── per_cell_a/ & per_stream_b/                 # Per-cell Benchmark A & per-stream Benchmark B archived summaries
├── logs/
│   ├── llm_raw_exchange_v3.jsonl.gz                # Lossless archive of all 6,400 Code-LLM evaluations (6,819 HTTP calls)
│   └── llm_api_audit_v3.jsonl                      # Per-call latency, token, JSON-compliance & retry audit ledger
├── figures/                                        # Vector PDF & 300-DPI PNG figures (Fig. 1-10)
└── manuscript/                                     # Camera-ready Springer sn-jnl LaTeX source, BibTeX, PDF & Markdown summary
```

---

## Reproduction & Verification Guide (Four Tiers)

All post-processing and verification scripts automatically locate the repository root relative to `__file__` (or honor `CAUSALJIT_BASE_DIR`) and contain zero hardcoded host paths.

### Tier 1 — Immediate Offline Verification & Full Figure/Table Regeneration (Zero External Downloads)

Requires only Python 3.10+ (tested on Python `3.13.0`) and the packages in `requirements_lock_v3.txt`. Runs completely offline on CPU directly from a fresh GitHub clone:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements_lock_v3.txt

# 1. Verify all 15 JSON provenance hashes, 50 .npz archives (385,620 commits), 48 citations & 1,452 table cells
python src_v3/verify_manuscript_v3.py

# 2. Verify all 33 V3_CONFIG table fields, 34 exact-path prose claims, 60 relational guards & adversarial self-tests
python scripts_v3/verify_prose_claims_v3.py

# 3. Verify archived pre-unlock gate SHA-256 (766e3fd6...) and run live mechanism Checks 2-14 (41 prefix-invariant branches)
python scripts_v3/run_post_campaign_mechanism_checks_v3.py

# 4. Regenerate all 10 figures (Fig. 1-10), 22 empirical LaTeX tables, and manuscript/main.tex
python scripts_v3/generate_conceptual_diagrams_v3.py
python scripts_v3/analyze_and_plot_v3.py
python scripts_v3/build_manuscript_v3.py
```

Compiling `manuscript/main.tex` via `pdflatex -> bibtex -> pdflatex -> pdflatex` reproduces the camera-ready 52-page Springer SQJ PDF with `0` Errors, `0` Warnings, and `0` Overfull hboxes.

### Tier 2 — Tabular Benchmark & Companion Audit Re-Execution

Because all 31 Benchmark B `.npz` bundles, all 5 confirmatory `.npz` bundles, all 6 Benchmark A `*_tab.npz` bundles, and all 50 seed-0 commit prediction `.npz` archives are included directly in this repository, you can re-execute the tabular campaigns and companion audits offline without extra downloads:

```bash
# Tabular Benchmark B (K=25), RQ3 ablations, RQ4a-d stress tests & confirmatory cohort
python src_v3/run_v3_experiments.py --phase phase_b
python src_v3/run_v3_experiments.py --phase mechanisms
python src_v3/run_v3_experiments.py --phase confirmatory

# Matched RQ4a FixedCurve-PU Stage-1 control (0.00 parity with rq4a_latency_intervention_v3.json)
python scripts_v3/run_rq4a_fixedcurve_control_v3.py

# Companion Audits B-G (follow-up sensitivity, 62-alarm Stage 1-3, PostHoc Logit/Ratio + clipping, 5-seed WinAP, budgets)
python scripts_v3/run_reviewer_mechanism_audits_v3.py
# Fast PostHoc Ratio clipping/tie-rate diagnostic only (~13 min on CPU/GPU):
python scripts_v3/run_reviewer_mechanism_audits_v3.py --update-ratio-clipping-only
```

> **Note on Audit A (Upstream ApacheJIT Funnel):** Re-running Audit A's raw CSV row counts requires the upstream `apachejit_total.csv` (`data_sync/confirmatory_apachejit_raw/`, included in the Tier 3 Zenodo archive).

### Tier 3 — Full Multimodal Re-Execution & Raw-to-Bundle Reconstruction (Full Zenodo Archive)

Re-executing Benchmark A neural hosts (`Attention-CNN`, `CodeBERT`, `DeepJIT-CNN`), rebuilding `.npz` bundles from raw corpora (`data_sync/`), or re-extracting `768`-D `microsoft/codebert-base` embeddings requires the complete `1.55 GB` offline Zenodo replication archive (too large for GitHub's repository limits; available via the corresponding author `zhaoyl@swc.neu.edu.cn` and scheduled for permanent Zenodo DOI deposit upon acceptance):

```bash
# Optional offline CodeBERT feature extraction (local HF cache of microsoft/codebert-base required)
python src_v3/build_codebert_cache_v3.py
# Rebuild all 89 development + 5 confirmatory bundles from raw data_sync/
python src_v3/data_audit_v3.py --rebuild-all --unlock-confirmatory
# Full development campaign (Phase A + B + mechanisms) & Tier-1 commit forensics
python scripts_v3/run_parallel_campaign_v3.py --workers 8
python src_v3/run_v3_llm_and_forensics.py --mode rq5
```

### Tier 4 — Optional Live Code-LLM Prompting Audit (RQ4e)

Offline inspection and verification of all `6,400` logical `deepseek-flash` evaluations (`6,819` HTTP attempts) in `logs/llm_raw_exchange_v3.jsonl.gz` requires **no** API key or network access. Re-issuing live API calls requires placing a DeepSeek key at `../../.llm_api/.deepseek.key` relative to the repository root and running:

```bash
python src_v3/run_v3_llm_and_forensics.py --mode rq4f
```

### Historical Pre-Unlock Gate Audit Record

Prior to unlocking the five confirmatory ApacheJIT repositories (`HDFS`, `MAPREDUCE`, `SPARK`, `ZEPPELIN`, `ZOOKEEPER`), `src_v3/verify_mechanisms_and_protocol_v3.py` was executed on the locked pre-unlock filesystem (`check_1` enforces `assert current_fs_locked` and therefore intentionally raises `AssertionError` once confirmatory data are present). Its exact SHA-256-sealed output is preserved at `results_v3/prerun_gate_archive/prerun_gate_verification_v3.json` (`766e3fd64c45a3718ad42c86c44f0b75703d139929d1a72dbd44e0019f492307`) and verified automatically in Tier 1.

---

## Citation

If you use this replication package, causal benchmark suite, or verification tools in your research, please cite:

```bibtex
@article{li2026causaljit,
  title   = {Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction},
  author  = {Li, Zhijie and Xu, Shaoyang and Liu, Ketong and Meng, Shibo and Gao, Feiran and Zhao, Yuli},
  journal = {Software Quality Journal},
  year    = {2026},
  note    = {Under review. Replication package: \url{https://github.com/789748456v/CausalJIT-Audit-SQJ}}
}
```

Please also credit the upstream datasets and foundational studies integrated into our benchmark:
- Pote et al. (2026), *Software Quality Journal* (unsupervised DriftLens JIT-SDP paradigm)
- McIntosh & Kamei (2018), *IEEE TSE* & Hoang et al. (2019), *MSR* (Qt / OpenStack multimodal JIT corpora)
- Cabral et al. (2019), *ICSE* & Cabral et al. (2023), *EMSE* (`ORB`, `BORB`, and 14 CommitGuru streams)
- Song et al. (2022), *IEEE TSE* (`ODaSC` verification-latency stream learner)
- Keshavarz & Nagappan (2022), *MSR* (`ApacheJIT` corpus)
- Bludau & Pretschner (2026), *Software Quality Journal* (`PR-SZZ` streams and SZZ error rates)

---

## License

- **Code & Scripts (`src_v3/`, `scripts_v3/`):** Licensed under the [MIT License](LICENSE).
- **Structured Results, Manifests, Tables, & Figures (`results_v3/`, `figures/`, `manuscript/`):** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), respecting upstream open-source project licenses (Apache-2.0 / GPL / LGPL / CC-BY-4.0).
