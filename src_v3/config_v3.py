"""
Single Source of Truth for N05 V3 Protocol, Provenance Definitions, Donor Dependency Graph,
Endpoints, Statistical Families, Forensic Selection Rules, and Default Hyperparameters.

All manuscript parameter tables (Table 2, Appendix Host Specs) and verification checks read directly
from `V3_CONFIG` so zero code-vs-paper drift can ever occur.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

SECONDS_PER_DAY: float = 86400.0

# 1. Strictly commit-authoring-time features for Qt & OpenStack (0 post-authoring Gerrit review fields!)
PURE_COMMIT_COLS_QT_OS: List[str] = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc",
    "aexp", "arexp", "asexp", "asawr",
]

# Post-authoring Gerrit review fields (excluded from primary Benchmark A; tested ONLY in pre-merge sensitivity)
EXCLUDED_GERRIT_REVIEW_COLS: List[str] = [
    "revd", "nrev", "rtime", "tcmt", "hcmt", "self", "app",
    "rexp", "oexp", "rrexp", "orexp", "rsexp", "osexp", "rsawr", "osawr",
]

# Bludau & Pretschner (N60, PROMISE 2022 / SQJ 2026) explicitly banned target/post-closure/counter columns
EXCLUDED_N60_POSTCLOSURE_AND_TARGET_COLS: List[str] = [
    "is_bug_fixing_commit", "is_bug_inducing_commit",
    "commit_creation_date", "commit_commitment_date",
    "pr_creation_date", "inner_commit_in_pr_creation_date",
    "pr_finish_date", "pr_duration", "was_merged", "was_squashed", "was_rebased",
    "nth_commit_since_beginning", "nth_commit_since_release",
]

KAMEI_RAW_COLS_CABRAL: List[str] = [
    "fix", "ns", "nd", "nf", "entropy", "la", "ld", "lt",
    "ndev", "age", "nuc", "exp", "rexp", "sexp",
]

APACHEJIT_RAW_COLS: List[str] = [
    "fix", "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]

# Exact standardized column index of log1p(LA) in each Benchmark B cohort for faithful LApredict (Zeng et al. ISSTA'21)
COHORT_LA_COL_INDEX: Dict[str, int] = {
    "Cabral19": 5,      # KAMEI_RAW_COLS_CABRAL.index("la") == 5
    "ApacheJIT": 1,     # APACHEJIT_RAW_COLS.index("la") == 1
    "N60_PRSZZ": 1,     # Bludau M1 features: col 1 == lines_added
    "Multimodal13": 0,  # PURE_COMMIT_COLS_QT_OS.index("la") == 0
}

# 2. Explicit Donor-Project / Ecosystem Dependency Graph for Natural Benchmark B (27 streams -> K=25 clusters)
#    Synthetic / concatenated shift streams are strictly excluded from K=25 cross-project inference!
NATURAL_STREAM_TO_DONOR_CLUSTER: Dict[str, str] = {
    # Cabral ICSE'19 (14 streams)
    "cabral_brackets": "brackets",
    "cabral_broadleaf": "broadleaf",
    "cabral_camel": "apache_camel",       # Merged with apache_camel!
    "cabral_fabric": "fabric",
    "cabral_jgroups": "jgroups",
    "cabral_matplotlib": "matplotlib",
    "cabral_neutron": "openstack",        # Merged with cabral_nova under OpenStack ecosystem!
    "cabral_nova": "openstack",           # Merged with cabral_neutron under OpenStack ecosystem!
    "cabral_npm": "npm",
    "cabral_sentry": "sentry",
    "cabral_spring": "spring_integration",
    "cabral_tomcat": "apache_tomcat",
    "cabral_wagtail": "wagtail",
    "cabral_zulip": "zulip",
    # ApacheJIT MSR'22 Development Streams (8 streams)
    "apache_activemq": "apache_activemq",
    "apache_camel": "apache_camel",       # Merged with cabral_camel!
    "apache_cassandra": "apache_cassandra",
    "apache_flink": "apache_flink",
    "apache_groovy": "apache_groovy",
    "apache_hbase": "apache_hbase",
    "apache_hive": "apache_hive",
    "apache_ignite": "apache_ignite",
    # Bludau & Pretschner SQJ 2026 Automated PR-SZZ Streams (5 streams)
    "n60_airflow": "apache_airflow",
    "n60_angular": "angular",
    "n60_jenkins": "jenkins",
    "n60_kafka": "apache_kafka",
    "n60_pulsar": "apache_pulsar",
}

# Controlled Abrupt/Subsystem Shift Streams (Analyzed Strictly Separately from Natural K=25!)
CONTROLLED_TABULAR_SHIFT_STREAMS: Dict[str, Tuple[str, ...]] = {
    "abrupt_nova_zulip": ("openstack", "zulip"),
    "abrupt_ignite_cassandra": ("apache_ignite", "apache_cassandra"),
    "abrupt_airflow_pulsar": ("apache_airflow", "apache_pulsar"),
    "abrupt_deepjit_nova_neutron_pure13": ("openstack",),
}

# 3. Untouched Confirmatory Cohort (`K_umbrella = 4` independent ecosystems / `5` repositories)
#    PRE-DECLARED STATISTICAL PLAN (Codex R3-C8):
#    Because two-sided exact sign-flip enumeration at K=4 has min attainable p = 2/2^4 = 0.125 (> 0.05)
#    and at K=5 has min p = 0.0625 (> 0.05), `umbrella-4` is pre-registered BEFORE unlocking as an
#    out-of-sample DIRECTIONAL & EFFECT-SIZE REPLICATION check (reporting exact 2^K enumeration p-values,
#    W/T/L, and mean Delta-AP across BOTH Family 1 and Family 2 without claiming alpha=0.05 power),
#    while `K=25` / `K=24` natural clusters serve as the powered cross-project hypothesis-testing cohort.
UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS: List[Tuple[str, str, str, str]] = [
    ("apache/hadoop-hdfs", "HDFS", "conf_apache_hdfs", "apache_hadoop"),
    ("apache/hadoop-mapreduce", "MAPREDUCE", "conf_apache_mapreduce", "apache_hadoop"),
    ("apache/spark", "SPARK", "conf_apache_spark", "apache_spark"),
    ("apache/zeppelin", "ZEPPELIN", "conf_apache_zeppelin", "apache_zeppelin"),
    ("apache/zookeeper", "ZOOKEEPER", "conf_apache_zookeeper", "apache_zookeeper"),
]

# 4. Pre-Declared Holm-Bonferroni Hypothesis Families
FAMILY_1_PARADOX_PAIRS: List[Tuple[str, str]] = [
    ("B1b_4W_Oracle", "B2_4W"),
    ("B0_Frozen", "B2_4W"),
    ("Survival_PU", "B2_4W"),
]

FAMILY_2_COMPETITOR_PAIRS: List[Tuple[str, str]] = [
    ("Survival_PU", "B0_Frozen"),
    ("Survival_PU", "Wait_90"),
    ("Survival_PU", "Pure_nnPU"),
    ("Survival_PU", "Full_LR_Adapt"),
    ("Survival_PU", "BORB_Native"),
    ("Survival_PU", "ORB_OHT_Native"),
    ("Survival_PU", "ODaSC_OHT_Native"),
]

# Pre-Declared Component & Control Family + Frozen Interpretation Boundaries (Codex Round 3b):
# Explicitly tests whether learned survival curves (`Survival_PU`) outperform a zero-fit parametric
# maturity curve (`FixedCurve_PU` / `SPU_Fixed90d`) and a compute-matched reliable teacher
# (`Teacher_Only_Matched`) BEFORE unlocking confirmatory data or viewing test results.
FAMILY_3_COMPONENT_CONTROL_PAIRS: List[Tuple[str, str]] = [
    ("Survival_PU_Default", "SPU_Fixed90d"),
    ("Survival_PU_Default", "Teacher_Only_Matched"),
    ("Survival_PU_Default", "SPU_NoBisection"),
    ("Survival_PU_Default", "SPU_WithLegacyGate"),
]

FAMILY_3_TRACK1_CONTROL_PAIRS: List[Tuple[str, str]] = [
    ("Survival_PU", "FixedCurve_PU"),
    ("Survival_PU", "Teacher_Only_Matched"),
    ("Survival_PU", "Teacher_Only_Exact"),
    ("Survival_PU", "Survival_PU_WithGate"),
]

PRE_REGISTERED_INTERPRETATION_BOUNDARIES: Dict[str, str] = {
    "powered_cohort_scope": (
        "Formal alpha=0.05 Holm-Bonferroni hypothesis testing for component controls (`FAMILY_3_TRACK1_CONTROL_PAIRS`) "
        "is scoped exclusively to the powered `K=25` / `K=24` natural cross-project donor clusters "
        "(min attainable two-sided exact p = 2/2^25 = 5.96e-8 << 0.0125). Small-K cohorts (`RQ3` K=7 natural streams "
        "where p_min = 2/2^7 = 0.015625 > 0.05/4 = 0.0125, and confirmatory `umbrella-4` K=4 / `repo-5` K=5 where "
        "p_min = 0.125 / 0.0625 > 0.05) are pre-registered strictly as descriptive effect-size (Delta-AP, 95% CI) "
        "and directional consistency (W/T/L) summaries within their attainable exact resolution, NOT alpha=0.05 Holm gates."
    ),
    "learned_vs_fixed_survival_curve": (
        "If `Survival_PU` (`CohortIncidence`) does NOT achieve Holm-adjusted p < 0.05 over `FixedCurve_PU` "
        "(`SPU_Fixed90d`) on the powered `K=25` / `K=24` cohort, the manuscript MUST state: "
        "'An incremental benefit over FixedCurve_PU was not established on this cohort', MUST NOT claim "
        "superiority of learned survival curves over simple parametric maturity curves, and MUST NOT treat "
        "statistical non-rejection as proof of equivalence or forced causal attribution."
    ),
    "stage3_pu_vs_matched_teacher": (
        "If `Survival_PU` does NOT achieve Holm-adjusted p < 0.05 over `Teacher_Only_Matched` on the powered "
        "`K=25` / `K=24` cohort, the manuscript MUST state: "
        "'An incremental benefit over Teacher_Only_Matched was not established on this cohort', MUST NOT claim "
        "Stage 3 distillation superiority on that cohort, and MUST NOT treat statistical non-rejection as proof "
        "of equivalence or forced causal attribution."
    ),
}


@dataclass(frozen=True)
class ForensicFilterConfigV3:
    """Single authoritative pre-declared selection rule for RQ5 natural forensics (Codex R3-M4)."""
    allowed_tiers: Tuple[str, ...] = ("Tier1_McIntosh_CSV",)
    min_censoring_latency_days: float = 14.0
    rescue_min_b0: float = 0.45
    rescue_max_b2: float = 0.35
    rescue_min_spu: float = 0.50
    fn_max_b2: float = 0.35
    fn_max_spu: float = 0.35
    fp_min_spu: float = 0.70
    max_cases_per_category: int = 3


@dataclass(frozen=True)
class SurvivalPUConfigV3:
    # Stage 1: Causal Cohort-Exposure Maturity Curve
    max_history_cohort: int = 6000
    support_age_quantile: float = 0.85
    min_risk_set: int = 120
    min_observed_fixes: int = 15
    c_min: float = 0.05
    c_max: float = 0.96
    tail_cap_max: float = 0.94
    tail_cap_min: float = 0.60
    fallback_half_life_days: float = 90.0

    # Stage 2: Reliable-Cohort Covariate-Shifted Teacher & Clipped Inverse-Maturity Prevalence
    reliable_maturity_thresh: float = 0.45
    max_reliable_commits: int = 2200
    imp_weight_min: float = 0.50
    imp_weight_max: float = 2.50
    recency_min: float = 0.75
    recency_max: float = 1.15
    ht_floor_c: float = 0.10
    ht_shrinkage_m0: float = 50.0
    pi_min: float = 0.02
    pi_max: float = 0.70
    teacher_sharpness: float = 1.15
    llm_blend_weight: float = 0.25

    # Stage 3: Bayesian Survival-PU Calibration (Single Shared Posterior Function, Zero Heuristic Gate)
    max_anchor_commits: int = 2000
    drift_emphasis_gamma: float = 1.40
    maturity_weight_floor: float = 0.45
    posterior_p_min: float = 0.003
    posterior_p_max: float = 0.997
    bisection_lo: float = -8.0
    bisection_hi: float = 8.0
    bisection_iters: int = 28

    # Immutable Per-Alarm Optimization & Boosting Budgets (Codex R3-C2: never inherited from mutated models!)
    lgb_init_rounds: int = 90
    lgb_alarm_rounds: int = 75
    neural_teacher_epochs: int = 4
    neural_student_epochs: int = 4
    neural_adapt_lr: float = 4.5e-4

    # DriftLens Monitor Defaults & Unconfounded 2x2 Factorial Parameters (Codex R3-C7)
    window_size_default: int = 400
    drift_pca_dim: int = 24
    drift_block_pct: float = 95.0          # Unified across ALL 4 cells of RQ4b 2x2 factorial!
    drift_cooldown_windows: int = 1        # Unified across ALL 4 cells of RQ4b 2x2 factorial!
    n46_ref_static_pct: float = 98.0       # Separate reference bundle matching Pote et al. (2026)
    n46_ref_cooldown_windows: int = 0
    drift_bootstrap_reps: int = 250
    fixed_transient_commits: int = 1200

    # Pre-declared Primary & Secondary Evaluation Endpoints
    primary_endpoint: str = "ap"  # Average Precision (threshold-free imbalanced ranking)
    secondary_endpoints: Tuple[str, ...] = (
        "roc_auc",
        "f1",
        "f1_th",
        "gmean",
        "brier_cal",
        "ece_cal",
        "recall_at_20pct_effort",
    )

    forensic_rules: ForensicFilterConfigV3 = field(default_factory=ForensicFilterConfigV3)

    def to_dict(self) -> Dict:
        return asdict(self)


V3_CONFIG = SurvivalPUConfigV3()
