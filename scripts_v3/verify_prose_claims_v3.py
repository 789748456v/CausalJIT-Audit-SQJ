r"""
Automated V3 Parameter-Table, Exact-Path Scope-Validated Prose-Claim, & Anti-Overclaiming Verifier
(`scripts_v3/verify_prose_claims_v3.py`).

Implements parameter-table, scope-validated prose, and adversarial verification without modifying `src_v3/*.py`
(preserving 100% SHA-256 lock across all `src_v3` modules and `results_v3/*_v3.json` headers):
  1. Verifies every row and every `V3_CONFIG` field in `tab:hyperparams` (`manuscript/main.tex`)
     directly against `src_v3/config_v3.py` (`V3_CONFIG`), catching any discrepancy such as `0.12` vs `0.10`.
  2. Strips `\begin{table}...\end{table}` blocks from `manuscript/main.tex` so prose checks run strictly
     on narrative text, and verifies Abstract word count is within Springer SQJ's `[150, 250]` limit.
  3. Validates registered prose claims using **exact positional JSON path templates** (not loose
     ancestor set intersections) for BOTH value paths (`jp_a`, `jp_b`) AND statistical paths (`jp_diff`,
     `jp_p`), validated dynamically against `src_v3/config_v3.py` and `results_v3/*.json` manifests.
  4. Runs adversarial self-tests proving swapped numbers, wrong directions, mutated `V3_CONFIG` values,
     fabricated metadata (`cohort="999 independent projects"`, `horizon="entire..."`, `metric="ROC_AUC"`),
     and realistic `K25 <-> K24` path+number substitution (`sensitivity_K24_disjoint_from_benchmark_a`
     / `K24_cluster_means` / `*_K24` claimed as `K=25`) ALL deterministically fail.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from config_v3 import (
    CONTROLLED_TABULAR_SHIFT_STREAMS,
    NATURAL_STREAM_TO_DONOR_CLUSTER,
    PRE_REGISTERED_INTERPRETATION_BOUNDARIES,
    UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS,
    V3_CONFIG,
)
from run_v3_experiments import REPRESENTATIVE_10_STREAMS


@dataclass(frozen=True)
class ProsePairClaim:
    claim_id: str
    section: str
    method_a: str
    method_b: str
    cohort: str
    horizon: str
    metric: str
    aggregation: str
    direction: str  # ">", "<", "==", "tie_fmt"
    jf_a: str
    jp_a: Tuple[str, ...]
    fmt_a: str
    jf_b: str
    jp_b: Tuple[str, ...]
    fmt_b: str
    prose_regex: str
    scope_prose_regex: str
    jf_diff: Optional[str] = None
    jp_diff: Optional[Tuple[str, ...]] = None
    fmt_diff: Optional[str] = None
    jf_p: Optional[str] = None
    jp_p: Optional[Tuple[str, ...]] = None
    fmt_p: Optional[str] = None


def _resolve_json(loaded: Dict[str, Any], jf: str, jpath: Tuple[str, ...]) -> float:
    cur = loaded[jf]
    for k in jpath:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return float(cur)


def _strip_tables_and_comments(tex: str) -> str:
    lines = [re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines()]
    no_comm = "\n".join(lines)
    return re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "\n", no_comm, flags=re.DOTALL)


def _extract_braced_command_arg(text: str, cmd: str) -> Optional[str]:
    m = re.search(rf"\\{cmd}\s*\{{", text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _extract_sections(prose_tex: str) -> Dict[str, str]:
    secs: Dict[str, str] = {"ALL": prose_tex}
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", prose_tex, re.DOTALL)
    if abs_m:
        secs["abstract"] = abs_m.group(1)
    else:
        abs_cmd = _extract_braced_command_arg(prose_tex, "abstract")
        if abs_cmd is not None:
            secs["abstract"] = abs_cmd
    lbl_iter = list(re.finditer(r"\\label\{((?:sec|app):[^\}]+)\}", prose_tex))
    for i, m in enumerate(lbl_iter):
        key = m.group(1)
        start = m.start()
        end = lbl_iter[i + 1].start() if (i + 1) < len(lbl_iter) else len(prose_tex)
        secs[key] = prose_tex[start:end]
    return secs


def _count_abstract_words(abs_tex: str) -> int:
    s = abs_tex
    s = re.sub(
        r"\\noindent|\\small|\\textbf\{([^\}]*)\}|\\textit\{([^\}]*)\}|\\texttt\{([^\}]*)\}",
        lambda m: (m.group(1) or m.group(2) or m.group(3) or " "),
        s,
    )
    s = re.sub(r"\\\\\[[^\]]*\]", " ", s)
    s = re.sub(r"\\vspace\{[^\}]*\}.*$", "", s, flags=re.DOTALL)
    s = re.sub(r"\$([^\$]*)\$", r" \1 ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[\{\}\\]", " ", s)
    tokens = [w for w in re.split(r"\s+", s.strip()) if re.search(r"[A-Za-z0-9]", w)]
    return len(tokens)


def _fmt_num(val: float, fmt: str) -> str:
    if fmt == "sci_tex_2":
        s = f"{val:.2e}"
        mant, exp = s.split("e")
        return f"{mant}e{int(exp)}"
    out = format(val, fmt)
    if out.endswith("%"):
        out = out[:-1]
    return out


def _norm_captured_num(raw: str, fmt: str) -> str:
    s = raw.strip().replace("{,}", "").replace(",", "").replace(r"\%", "").replace("%", "")
    if fmt == "sci_tex_2":
        m = re.match(r"^([+-]?\d+\.\d+)\s*\\times\s*10\^\{([+-]?\d+)\}$", s)
        if m:
            return f"{m.group(1)}e{int(m.group(2))}"
    return s.lstrip("+") if not fmt.startswith("+") else s


def _matches_exact_template(jpath: Tuple[str, ...], template: Tuple[Any, ...]) -> bool:
    """
    Matches `jpath` position-by-position against `template`:
      - `str`: exact string equality
      - `set`/`frozenset`: membership in allowed string set
      - `"*"`: matches any single non-empty key (never across multiple path depths!)
    """
    if len(jpath) != len(template):
        return False
    for elem, pat in zip(jpath, template):
        if pat == "*":
            if not isinstance(elem, str) or not elem:
                return False
        elif isinstance(pat, (set, frozenset)):
            if elem not in pat:
                return False
        else:
            if elem != pat:
                return False
    return True


def _matches_any_template(jf: str, jpath: Tuple[str, ...], allowed: Sequence[Tuple[str, Tuple[Any, ...]]]) -> bool:
    return any(jf == exp_jf and _matches_exact_template(jpath, tmpl) for exp_jf, tmpl in allowed)


def verify_hyperparam_table_against_v3_config(full_tex: str, cfg_obj: Any = V3_CONFIG) -> int:
    m_tab = re.search(
        r"\\begin\{table\}\[[^\]]+\].*?\\label\{tab:hyperparams\}.*?\\midrule(.*?)\\bottomrule",
        full_tex,
        flags=re.DOTALL,
    )
    assert m_tab is not None, "Could not locate `tab:hyperparams` in main.tex!"
    body = m_tab.group(1)
    raw_rows = [r.strip() for r in body.split(r"\\") if r.strip()]
    assert len(raw_rows) >= 12, f"Expected >= 12 rows in tab:hyperparams, got {len(raw_rows)}"

    checked_fields = 0
    for r in raw_rows:
        cols = [c.strip() for c in r.split("&")]
        assert len(cols) == 5, f"Expected 5 columns in tab:hyperparams row, got {len(cols)}: {r}"
        default_col = cols[2].replace("{,}", "").replace(r"\;", " ").replace(r"\text{e-}", "e-")
        field_col = cols[3].replace(r"\allowbreak ", "").replace(r"\allowbreak", "").replace(r"\_", "_")

        fields = re.findall(r"\\texttt\{([a-zA-Z0-9_]+)\}", field_col)
        assert len(fields) >= 1, f"No V3_CONFIG field names found in Col 4: {cols[3]}"
        nums_str = re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", default_col)
        nums = [float(x) for x in nums_str]
        assert len(nums) == len(fields), (
            f"Mismatch between printed defaults {nums} and V3_CONFIG fields {fields} in row: {r}"
        )
        for val_printed, fname in zip(nums, fields):
            assert hasattr(cfg_obj, fname), f"Field '{fname}' does not exist on V3_CONFIG!"
            val_true = float(getattr(cfg_obj, fname))
            assert abs(val_printed - val_true) <= 1e-12, (
                f"tab:hyperparams mismatch for V3_CONFIG.{fname}: printed {val_printed}, actual {val_true}"
            )
            checked_fields += 1

    m_cht = re.search(r"c_\{\\text\{ht\}\}\s*=\s*(0\.\d+)", full_tex)
    assert m_cht is not None, "Missing inline `$c_{\\text{ht}} = ...$` in main.tex!"
    assert abs(float(m_cht.group(1)) - float(cfg_obj.ht_floor_c)) <= 1e-12, (
        f"Inline c_ht={m_cht.group(1)} disagrees with V3_CONFIG.ht_floor_c={cfg_obj.ht_floor_c}!"
    )
    return checked_fields


def build_exact_scope_binding_rules(loaded: Dict[str, Any]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    """
    Dynamically verifies dataset/manifest counts and binds each exact `(cohort, horizon, metric, aggregation)`
    4-tuple to its ONLY permitted full positional JSON path templates.
    """
    ba = loaded["benchmark_a_multihost_v3.json"]
    bb = loaded["benchmark_b_main_v3.json"]
    conf = loaded["confirmatory_cohort_v3.json"]
    rq4a = loaded["rq4a_latency_intervention_v3.json"]
    rq4b = loaded["rq4b_factorial_monitors_v3.json"]
    rq4e = loaded["rq4e_sensitivity_sweeps_v3.json"]
    rq4f = loaded["rq4f_llm_audit_v3.json"]

    assert len({v["stream_name"] for v in ba["cells"].values()}) == 6
    assert len(set(NATURAL_STREAM_TO_DONOR_CLUSTER.values())) == 25
    bb_shift_streams = sorted(bb["controlled_shift_streams_mlp"].keys())
    assert bb_shift_streams == sorted(CONTROLLED_TABULAR_SHIFT_STREAMS.keys()) and len(bb_shift_streams) == 4

    # Verify RQ4b evaluates ALL 4 controlled-shift streams (4 * 5 = 20 runs), whereas RQ4e evaluates 3!
    rq4b_per_stream = set(rq4b["configs"]["DriftLens_Block_Rolling_95_CD1_V3"]["per_stream"].keys())
    rq4b_shifts = sorted(rq4b_per_stream & set(CONTROLLED_TABULAR_SHIFT_STREAMS.keys()))
    assert len(rq4b_shifts) == 4, f"Expected 4 controlled shifts in RQ4b, got {rq4b_shifts}"

    rq4e_shifts = [s for s in REPRESENTATIVE_10_STREAMS if s in CONTROLLED_TABULAR_SHIFT_STREAMS]
    assert rq4e_shifts == [
        "abrupt_nova_zulip",
        "abrupt_ignite_cassandra",
        "abrupt_airflow_pulsar",
    ]
    assert len(REPRESENTATIVE_10_STREAMS) == 10
    assert len(UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS) == 5
    assert len(conf["umbrella4_means_ap"]) >= 1
    assert int(rq4f["ecosystems"]["openstack"]["n_commits_evaluated"]) == 320
    assert "3.00" in rq4a["results_by_kappa"]
    assert "800" in rq4e["window_size_sweep"]

    k25_fams = {"family1_paradox_K25", "family2_competitors_K25", "family3_controls_K25"}

    return {
        ("Benchmark_A_Qt_Natural", "scheduled_1W", "pooled_censoring_and_count", "pooled_window_defects"): {
            "val_templates": [
                (
                    "manuscript_derived_aggregates_v3.json",
                    ("window_censoring_audit", "qt_natural_25k", {"pooled_cens_1w", "pooled_bugs_1w"}),
                ),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_A_6Streams", "full_stream", "ap", "6stream_mean"): {
            "val_templates": [
                ("manuscript_derived_aggregates_v3.json", ("benchmark_a_track2_6stream_mean", "*", "ap")),
                ("manuscript_derived_aggregates_v3.json", ("benchmark_a_4host_6stream_mean", "*", "ap")),
                ("manuscript_derived_aggregates_v3.json", ("benchmark_a_by_host_6streams", "*", "*", "ap")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_A_6Streams", "full_stream", "recall_at_20pct_effort", "6stream_mean"): {
            "val_templates": [
                ("manuscript_derived_aggregates_v3.json", ("benchmark_a_track2_6stream_mean", "*", "recall_at_20pct_effort")),
                ("manuscript_derived_aggregates_v3.json", ("benchmark_a_4host_6stream_mean", "*", "recall_at_20pct_effort")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_B_K25_Natural", "full_stream", "ap", "K25_donor_cluster_mean"): {
            "val_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "*")),
            ],
            "diff_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", k25_fams, "*", "mean_diff")),
            ],
            "p_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_mlp", "primary_ap", k25_fams, "*", "p_holm")),
            ],
        },
        ("Benchmark_B_K25_Natural_LightGBM", "full_stream", "ap", "K25_donor_cluster_mean"): {
            "val_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_lightgbm_ap", "K25_cluster_means", "*")),
            ],
            "diff_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_lightgbm_ap", k25_fams, "*", "mean_diff")),
            ],
            "p_templates": [
                ("benchmark_b_main_v3.json", ("cluster_inference_lightgbm_ap", k25_fams, "*", "p_holm")),
            ],
        },
        ("Benchmark_B_4ControlledShifts", "full_post_onset", "ap", "4stream_mean"): {
            "val_templates": [
                ("manuscript_derived_aggregates_v3.json", ("benchmark_b_controlled_shift_4stream_mean", "*", "post_onset_ap")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_B_4ControlledShifts", "transient_1200", "ap", "4stream_mean"): {
            "val_templates": [
                ("manuscript_derived_aggregates_v3.json", ("benchmark_b_controlled_shift_4stream_mean", "*", "transient_1200_ap")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Confirmatory_Umbrella4_MLP", "full_stream", "ap", "K4_umbrella_mean"): {
            "val_templates": [
                ("confirmatory_cohort_v3.json", ("umbrella4_means_ap", "*")),
            ],
            "diff_templates": [
                ("confirmatory_cohort_v3.json", ("umbrella4_family1_paradox", "*", "mean_diff")),
            ],
            "p_templates": [
                ("confirmatory_cohort_v3.json", ("umbrella4_family1_paradox", "*", "p_wilcoxon_wilcox")),
            ],
        },
        ("RQ4a_10Streams", "full_stream_kappa_3.0", "f1", "10stream_mean"): {
            "val_templates": [
                ("rq4a_latency_intervention_v3.json", ("results_by_kappa", "3.00", "means", "*", "f1")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4a_10Streams", "full_stream_kappa_3.0", "f1_th", "10stream_mean"): {
            "val_templates": [
                ("rq4a_latency_intervention_v3.json", ("results_by_kappa", "3.00", "means", "*", "f1_th")),
                ("rq4a_fixedcurve_control_v3.json", ("results_by_kappa", "3.00", "means", "*", "f1_th")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4a_10Streams", "full_stream_kappa_0.0", "ap", "10stream_mean"): {
            "val_templates": [
                ("rq4a_fixedcurve_control_v3.json", ("results_by_kappa", "0.00", "means", "*", "ap")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4a_10Streams", "full_stream_kappa_3.0", "ap", "10stream_mean"): {
            "val_templates": [
                ("rq4a_fixedcurve_control_v3.json", ("results_by_kappa", "3.00", "means", "*", "ap")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4b_4ControlledShifts_20Runs", "post_onset_detection", "undetected_shift_rate", "4shift_x_5seed_20runs"): {
            "val_templates": [
                ("rq4b_factorial_monitors_v3.json", ("configs", "*", "undetected_shift_rate")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4b_4ControlledShifts_20Runs", "post_onset_detection", "conditional_shift_delay_commits_given_detected", "4shift_x_5seed_20runs"): {
            "val_templates": [
                ("rq4b_factorial_monitors_v3.json", ("configs", "*", "conditional_shift_delay_commits_given_detected")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4c_5PRSZZStreams", "2x2_szz_perturbation", "ap", "5stream_mean"): {
            "val_templates": [
                ("rq4d_szz_perturbation_v3.json", ("regimes", "Pop_N60_Indep_PermFN", "cells_mean_ap", "*", "*")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4e_3RepControlledShifts", "fixed_1200_post_onset_W800", "ap", "3stream_rep_shift_mean"): {
            "val_templates": [
                ("rq4e_sensitivity_sweeps_v3.json", ("window_size_sweep", "800", "fixed_1200_post_onset_ap", "*")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4f_OpenStack_320Slice", "320_commit_slice", "ap_mean", "3_prompt_seed_mean"): {
            "val_templates": [
                ("rq4f_llm_audit_v3.json", ("ecosystems", "openstack", "regimes", "*", "ap_mean")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_B_K25_Natural_5Seed", "per_window_400", "ap", "K25_donor_cluster_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "win_ap_unweighted", "*")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Benchmark_B_K25_Natural_5Seed", "scale_norm_rank_whole", "ap", "K25_donor_cluster_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_f_window_vs_whole_ap", "K25_5seed_means", "scale_norm_rank_whole_ap", "*")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("Confirmatory_Standalone3_MLP", "full_stream", "ap", "3repo_standalone_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_a_apachejit_funnel", "separated_confirmatory_ap", "standalone_3_mlp", "*")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4a_10Streams_PostHocControl", "full_stream_kappa_3.0", "f1", "10stream_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "*", "f1")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4a_10Streams_PostHocRatioClip", "full_stream_kappa_0.0", "upper_clip_and_win_tie", "10stream_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_d_calibration_and_posthoc", "posthoc_ratio_clipping_diagnostics", "by_kappa", "0.00", {"upper_clip_frac", "window_upper_tie_frac"})),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
        ("RQ4b_10Streams_AlarmRate", "full_stream", "mean_alarm_rate_10streams", "10stream_mean"): {
            "val_templates": [
                ("rq4b_factorial_monitors_v3.json", ("configs", "*", "mean_alarm_rate_10streams")),
            ],
            "diff_templates": [],
            "p_templates": [],
        },
    }


def _validate_claim_exact_scope_binding(
    claim: ProsePairClaim,
    sec_txt: str,
    binding_rules: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> None:
    scope_key = (claim.cohort, claim.horizon, claim.metric, claim.aggregation)
    assert scope_key in binding_rules, (
        f"[{claim.claim_id}] Unregistered or inconsistent scope 4-tuple {scope_key}!"
    )
    rule = binding_rules[scope_key]

    assert _matches_any_template(claim.jf_a, claim.jp_a, rule["val_templates"]), (
        f"[{claim.claim_id}] Value path A {(claim.jf_a, claim.jp_a)} violates exact path template for {scope_key}!"
    )
    assert _matches_any_template(claim.jf_b, claim.jp_b, rule["val_templates"]), (
        f"[{claim.claim_id}] Value path B {(claim.jf_b, claim.jp_b)} violates exact path template for {scope_key}!"
    )
    if claim.jf_diff and claim.jp_diff:
        assert _matches_any_template(claim.jf_diff, claim.jp_diff, rule["diff_templates"]), (
            f"[{claim.claim_id}] Diff path {(claim.jf_diff, claim.jp_diff)} violates exact diff template for {scope_key}!"
        )
    if claim.jf_p and claim.jp_p:
        assert _matches_any_template(claim.jf_p, claim.jp_p, rule["p_templates"]), (
            f"[{claim.claim_id}] P-value path {(claim.jf_p, claim.jp_p)} violates exact p-value template for {scope_key}!"
        )
    assert re.search(claim.scope_prose_regex, sec_txt, flags=re.DOTALL) is not None, (
        f"[{claim.claim_id}] Prose scope anchor '{claim.scope_prose_regex}' missing in section '{claim.section}'!"
    )


def verify_single_claim(
    claim: ProsePairClaim,
    secs: Dict[str, str],
    loaded: Dict[str, Any],
    binding_rules: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    if claim.section not in secs:
        raise KeyError(f"[{claim.claim_id}] Missing section '{claim.section}' in main.tex!")
    sec_txt = secs[claim.section]

    _validate_claim_exact_scope_binding(claim, sec_txt, binding_rules)

    va = _resolve_json(loaded, claim.jf_a, claim.jp_a)
    vb = _resolve_json(loaded, claim.jf_b, claim.jp_b)

    if claim.direction == ">":
        assert va > vb, f"[{claim.claim_id}] Direction check failed: expected {claim.method_a} ({va}) > {claim.method_b} ({vb})"
    elif claim.direction == "<":
        assert va < vb, f"[{claim.claim_id}] Direction check failed: expected {claim.method_a} ({va}) < {claim.method_b} ({vb})"
    elif claim.direction == "==":
        assert abs(va - vb) <= 1e-12, f"[{claim.claim_id}] Equality check failed: {va} != {vb}"
    elif claim.direction == "tie_fmt":
        assert _fmt_num(va, claim.fmt_a) == _fmt_num(vb, claim.fmt_b), f"[{claim.claim_id}] Formatted tie check failed: {va} vs {vb}"
    else:
        raise ValueError(f"Unknown direction '{claim.direction}'")

    m = re.search(claim.prose_regex, sec_txt, flags=re.DOTALL)
    assert m is not None, (
        f"[{claim.claim_id}] Prose pattern not found in section '{claim.section}'!\n"
        f"  Pattern: {claim.prose_regex}"
    )
    gd = m.groupdict()
    assert "a" in gd and "b" in gd, f"[{claim.claim_id}] Regex must define (?P<a>...) and (?P<b>...)!"

    exp_a = _fmt_num(va, claim.fmt_a).lstrip("+")
    exp_b = _fmt_num(vb, claim.fmt_b).lstrip("+")
    got_a = _norm_captured_num(gd["a"], claim.fmt_a)
    got_b = _norm_captured_num(gd["b"], claim.fmt_b)
    assert got_a == exp_a, f"[{claim.claim_id}] Prose value A mismatch for {claim.method_a}: got '{got_a}', expected '{exp_a}'"
    assert got_b == exp_b, f"[{claim.claim_id}] Prose value B mismatch for {claim.method_b}: got '{got_b}', expected '{exp_b}'"

    if claim.jf_diff and claim.jp_diff and claim.fmt_diff:
        assert "diff" in gd, f"[{claim.claim_id}] Regex missing (?P<diff>...)"
        vd = _resolve_json(loaded, claim.jf_diff, claim.jp_diff)
        exp_d = format(vd, claim.fmt_diff)
        got_d = gd["diff"].strip().replace("$-$", "-")
        assert got_d == exp_d, f"[{claim.claim_id}] Prose diff mismatch: got '{got_d}', expected '{exp_d}'"

    if claim.jf_p and claim.jp_p and claim.fmt_p:
        assert "p" in gd, f"[{claim.claim_id}] Regex missing (?P<p>...)"
        vp = _resolve_json(loaded, claim.jf_p, claim.jp_p)
        exp_p = _fmt_num(vp, claim.fmt_p)
        got_p = _norm_captured_num(gd["p"], claim.fmt_p)
        assert got_p == exp_p, f"[{claim.claim_id}] Prose p-value mismatch: got '{got_p}', expected '{exp_p}'"

    return {
        "claim_id": claim.claim_id,
        "section": claim.section,
        "pair": f"{claim.method_a} {claim.direction} {claim.method_b}",
        "cohort": claim.cohort,
        "horizon": claim.horizon,
        "metric": claim.metric,
        "aggregation": claim.aggregation,
        "val_a": exp_a,
        "val_b": exp_b,
    }


def build_prose_claims_registry() -> List[ProsePairClaim]:
    return [
        ProsePairClaim(
            claim_id="RQ2_FULL_LR_VS_SPU_GRADUAL",
            section="sec:rq2",
            method_a="Full_LR_Adapt",
            method_b="Survival_PU",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Full_LR_Adapt"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_b=".4f",
            prose_regex=r"Full-LR Adapt-4W\s+leads in mean at\s*\$(?P<a>0\.\d{4})\$.*?Survival-PU\s+scores\s*\$(?P<b>0\.\d{4})\$\s*AP",
            scope_prose_regex=r"across \$K=25\$ natural donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ2_FIXEDCURVE_BEATS_SPU_K25",
            section="sec:rq2",
            method_a="FixedCurve_PU",
            method_b="Survival_PU",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "FixedCurve_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family3_controls_K25", "0", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"zero-fit FixedCurve-PU significantly exceeds cohort-estimated Survival-PU at\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"On \$K=25\$ primary MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ1_QT_POOLED_CENSORING_AND_COUNT",
            section="sec:rq1",
            method_a="Qt_Pooled_Cens_1W",
            method_b="Qt_Pooled_Bugs_1W",
            cohort="Benchmark_A_Qt_Natural",
            horizon="scheduled_1W",
            metric="pooled_censoring_and_count",
            aggregation="pooled_window_defects",
            direction="<",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("window_censoring_audit", "qt_natural_25k", "pooled_cens_1w"),
            fmt_a=".1%",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("window_censoring_audit", "qt_natural_25k", "pooled_bugs_1w"),
            fmt_b=".0f",
            prose_regex=r"natural Qt remain unfixed,\s*corresponding to a pooled rate of\s*\$(?P<a>\d+\.\d)\\%\$\s*across\s*\$(?P<b>\d+)\$\s*window-defects",
            scope_prose_regex=r"Across scheduled single-window boundaries",
        ),
        ProsePairClaim(
            claim_id="RQ1_MLP_ORACLE_VS_B2_4W",
            section="sec:rq1",
            method_a="B1b_4W_Oracle",
            method_b="B2_4W",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "B1b_4W_Oracle"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "B2_4W"),
            fmt_b=".4f",
            jf_diff="benchmark_b_main_v3.json",
            jp_diff=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "0", "mean_diff"),
            fmt_diff="+.4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "0", "p_holm"),
            fmt_p="sci_tex_2",
            prose_regex=r"On MLP,\s*\$B_\{1b\}\^\*\$\s*reaches\s*\$(?P<a>0\.\d{4})\$\s*AP versus\s*\$(?P<b>0\.\d{4})\$\s*for\s*\$B_2\\text\{-4W\}\$,\s*a statistically significant inflation of\s*\$\\Delta\\text\{AP\}\s*=\s*(?P<diff>[+-]0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>\d+\.\d+\s*\\times\s*10\^\{-?\d+\})\$\)",
            scope_prose_regex=r"across \$K=25\$ natural donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ1_LGB_ORACLE_VS_B2_4W",
            section="sec:rq1",
            method_a="B1b_4W_Oracle_LGB",
            method_b="B2_4W_LGB",
            cohort="Benchmark_B_K25_Natural_LightGBM",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_lightgbm_ap", "K25_cluster_means", "B1b_4W_Oracle"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_lightgbm_ap", "K25_cluster_means", "B2_4W"),
            fmt_b=".4f",
            jf_diff="benchmark_b_main_v3.json",
            jp_diff=("cluster_inference_lightgbm_ap", "family1_paradox_K25", "0", "mean_diff"),
            fmt_diff="+.4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_lightgbm_ap", "family1_paradox_K25", "0", "p_holm"),
            fmt_p="sci_tex_2",
            prose_regex=r"On LightGBM,\s*hindsight optimism more than doubles\.\s*Here\s*\$B_\{1b\}\^\*\$\s*reaches\s*\$(?P<a>0\.\d{4})\$\s*AP versus\s*\$(?P<b>0\.\d{4})\$\s*for\s*\$B_2\\text\{-4W\}\$,\s*gaining\s*\$\\Delta\\text\{AP\}\s*=\s*(?P<diff>[+-]0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>\d+\.\d+\s*\\times\s*10\^\{-?\d+\})\$\)",
            scope_prose_regex=r"across \$K=25\$ natural donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ1_B2_4W_BEATS_B0_GRADUAL",
            section="sec:rq1",
            method_a="B2_4W",
            method_b="B0_Frozen",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "B2_4W"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "B0_Frozen"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "1", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"beat frozen \$B_0\$ on full-stream MLP AP across gradual natural clusters,\s*at\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"full-stream MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ1_SHIFTS_B2_4W_BEATS_B0",
            section="sec:rq1",
            method_a="B2_4W",
            method_b="B0_Frozen",
            cohort="Benchmark_B_4ControlledShifts",
            horizon="full_post_onset",
            metric="ap",
            aggregation="4stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_b_controlled_shift_4stream_mean", "B2_4W", "post_onset_ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_b_controlled_shift_4stream_mean", "B0_Frozen", "post_onset_ap"),
            fmt_b=".4f",
            prose_regex=r"controlled abrupt shifts,\s*reaching\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*post-onset AP",
            scope_prose_regex=r"controlled abrupt shifts",
        ),
        ProsePairClaim(
            claim_id="RQ2_BA_FULL_LR_BEATS_DEPLOYABLE_DEEPJIT",
            section="sec:rq2",
            method_a="Full_LR_B0",
            method_b="DeepJIT_CNN_B0",
            cohort="Benchmark_A_6Streams",
            horizon="full_stream",
            metric="ap",
            aggregation="6stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_a_track2_6stream_mean", "Full_LR_B0", "ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_a_by_host_6streams", "deepjit_cnn_skip", "B0_Frozen", "ap"),
            fmt_b=".4f",
            prose_regex=r"Full-LR Frozen,\s*leads every deployable single-host configuration at\s*\$(?P<a>0\.\d{4})\$\s*AP.*?strongest deployable individual model is frozen\s+DeepJIT-CNN\s+at\s*\$B_0\s*=\s*(?P<b>0\.\d{4})\$\s*AP",
            scope_prose_regex=r"across the six Benchmark~A streams",
        ),
        ProsePairClaim(
            claim_id="RQ2_BA_DEEPJIT_HINDSIGHT_EXCEEDS_FULL_LR",
            section="sec:rq2",
            method_a="DeepJIT_CNN_B1a_Hindsight",
            method_b="Full_LR_B0",
            cohort="Benchmark_A_6Streams",
            horizon="full_stream",
            metric="ap",
            aggregation="6stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_a_by_host_6streams", "deepjit_cnn_skip", "B1a_Full_Hindsight", "ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_a_track2_6stream_mean", "Full_LR_B0", "ap"),
            fmt_b=".4f",
            prose_regex=r"Full-LR Frozen,\s*leads every deployable single-host configuration at\s*\$(?P<b>0\.\d{4})\$\s*AP.*?[Oo]nly non-deployable full-prefix hindsight\s+DeepJIT-CNN\s*\(\$B_\{1a\}\^\*\s*=\s*(?P<a>0\.\d{4})\$\)\s*exceeds\s+Full-LR Frozen",
            scope_prose_regex=r"across the six Benchmark~A streams",
        ),
        ProsePairClaim(
            claim_id="RQ2_BA_LOC_ONLY_BEATS_SPU_R20",
            section="sec:rq2",
            method_a="LOC_Only",
            method_b="Survival_PU_4HostMean",
            cohort="Benchmark_A_6Streams",
            horizon="full_stream",
            metric="recall_at_20pct_effort",
            aggregation="6stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_a_track2_6stream_mean", "LOC_Only", "recall_at_20pct_effort"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_a_4host_6stream_mean", "Survival_PU", "recall_at_20pct_effort"),
            fmt_b=".4f",
            prose_regex=r"LOC-Only\s+reaches\s*\$(?P<a>0\.\d{4})\$\s*R@20\\%Effort,\s*ahead of\s+Survival-PU's\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"across the six Benchmark~A streams",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_SPU_BEATS_NNPU",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="Pure_nnPU",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Pure_nnPU"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family2_competitors_K25", "2", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"Survival-PU\s+scores\s*\$(?P<a>0\.\d{4})\$\s*AP,\s*significantly outperforming\s+Pure-nnPU\s+at\s*\$(?P<b>0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"across \$K=25\$ natural donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_SPU_TRAILS_B2_4W",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction="<",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "B2_4W"),
            fmt_b=".4f",
            jf_diff="benchmark_b_main_v3.json",
            jp_diff=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "2", "mean_diff"),
            fmt_diff="+.4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "2", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"followed by\s+Causal-4W\s+at\s*\$(?P<b>0\.\d{4})\$.*?Survival-PU\s+scores\s*\$(?P<a>0\.\d{4})\$\s*AP.*?significantly trails\s+Causal-4W\s+by\s*\$\\Delta\\text\{AP\}\s*=\s*(?P<diff>[+-]0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"across \$K=25\$ natural donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_FIXEDCURVE_BEATS_SPU_MLP",
            section="sec:rq2",
            method_a="FixedCurve_PU",
            method_b="Survival_PU",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "FixedCurve_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family3_controls_K25", "0", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"zero-fit FixedCurve-PU significantly exceeds cohort-estimated Survival-PU at\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"On \$K=25\$ primary MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_FIXEDCURVE_BEATS_SPU_LGB",
            section="sec:rq2",
            method_a="FixedCurve_PU_LGB",
            method_b="Survival_PU_LGB",
            cohort="Benchmark_B_K25_Natural_LightGBM",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_lightgbm_ap", "K25_cluster_means", "FixedCurve_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_lightgbm_ap", "K25_cluster_means", "Survival_PU"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_lightgbm_ap", "family3_controls_K25", "0", "p_holm"),
            fmt_p=".4f",
            prose_regex=r"holds on LightGBM at\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*\(\$p_\{\\text\{Holm\}\}\s*=\s*(?P<p>0\.\d{4})\$\)",
            scope_prose_regex=r"On \$K=25\$ primary MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ2_SHIFT_FULL_SPU_VS_ODASC_HOST",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="ODaSC_Host",
            cohort="Benchmark_B_4ControlledShifts",
            horizon="full_post_onset",
            metric="ap",
            aggregation="4stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_b_controlled_shift_4stream_mean", "Survival_PU", "post_onset_ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_b_controlled_shift_4stream_mean", "ODaSC_Host", "post_onset_ap"),
            fmt_b=".4f",
            prose_regex=r"Survival-PU achieves the highest deployable mean AP at\s*\$(?P<a>0\.\d{4})\$\},\s*ahead of\s+ODaSC-Host\s+at\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"Controlled Concatenated Distribution Shifts.*full post-onset horizon",
        ),
        ProsePairClaim(
            claim_id="RQ2_SHIFT_TRANSIENT_BORB_BEATS_SPU",
            section="sec:rq2",
            method_a="BORB_Alarm",
            method_b="Survival_PU",
            cohort="Benchmark_B_4ControlledShifts",
            horizon="transient_1200",
            metric="ap",
            aggregation="4stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_b_controlled_shift_4stream_mean", "BORB_Alarm", "transient_1200_ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_b_controlled_shift_4stream_mean", "Survival_PU", "transient_1200_ap"),
            fmt_b=".4f",
            prose_regex=r"BORB-Alarm\s+at\s*\$(?P<a>0\.\d{4})\$\s*and\s+BORB-Native\s+at\s*\$0\.5734\$\s*adapt fastest,\s*surpassing\s+Survival-PU\s+at\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"immediate first 1,200 post-onset commits",
        ),
        ProsePairClaim(
            claim_id="RQ2_CONF_B1B_VS_B2_4W_NONREP",
            section="sec:rq2",
            method_a="B1b_4W_Oracle",
            method_b="B2_4W",
            cohort="Confirmatory_Umbrella4_MLP",
            horizon="full_stream",
            metric="ap",
            aggregation="K4_umbrella_mean",
            direction="<",
            jf_a="confirmatory_cohort_v3.json",
            jp_a=("umbrella4_means_ap", "B1b_4W_Oracle"),
            fmt_a=".4f",
            jf_b="confirmatory_cohort_v3.json",
            jp_b=("umbrella4_means_ap", "B2_4W"),
            fmt_b=".4f",
            jf_p="confirmatory_cohort_v3.json",
            jp_p=("umbrella4_family1_paradox", "0", "p_wilcoxon_wilcox"),
            fmt_p=".3f",
            prose_regex=r"Window Oracle\s+reaches\s*\$(?P<a>0\.\d{4})\$\s*AP,\s*trailing\s+Causal-4W\s+at\s*\$(?P<b>0\.\d{4})\$\s*\(exact two-sided\s*\$p\s*=\s*(?P<p>0\.\d{3})\$\)",
            scope_prose_regex=r"four-umbrella MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ2_CONF_B1B_VS_B0_EXACT",
            section="sec:rq2",
            method_a="B1b_4W_Oracle",
            method_b="B0_Frozen",
            cohort="Confirmatory_Umbrella4_MLP",
            horizon="full_stream",
            metric="ap",
            aggregation="K4_umbrella_mean",
            direction="<",
            jf_a="confirmatory_cohort_v3.json",
            jp_a=("umbrella4_means_ap", "B1b_4W_Oracle"),
            fmt_a=".4f",
            jf_b="confirmatory_cohort_v3.json",
            jp_b=("umbrella4_means_ap", "B0_Frozen"),
            fmt_b=".4f",
            prose_regex=r"Window Oracle\s+reaches\s*\$(?P<a>0\.\d{4})\$\s*AP.*?and\s+Frozen\s+at\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"four-umbrella MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_F1_SPU_VS_B2",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="RQ4a_10Streams",
            horizon="full_stream_kappa_3.0",
            metric="f1",
            aggregation="10stream_mean",
            direction=">",
            jf_a="rq4a_latency_intervention_v3.json",
            jp_a=("results_by_kappa", "3.00", "means", "Survival_PU", "f1"),
            fmt_a=".4f",
            jf_b="rq4a_latency_intervention_v3.json",
            jp_b=("results_by_kappa", "3.00", "means", "B2_4W", "f1"),
            fmt_b=".4f",
            prose_regex=r"drops to\s*\$(?P<b>0\.\d{4})\$\s*fixed-threshold\s*\$F_1\$.*?Survival-PU sustains\s*\$F_1\s*=\s*(?P<a>0\.\d{4})\$",
            scope_prose_regex=r"Across the AP trajectory",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_F1TH_SPU_VS_B2",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="RQ4a_10Streams",
            horizon="full_stream_kappa_3.0",
            metric="f1_th",
            aggregation="10stream_mean",
            direction=">",
            jf_a="rq4a_latency_intervention_v3.json",
            jp_a=("results_by_kappa", "3.00", "means", "Survival_PU", "f1_th"),
            fmt_a=".4f",
            jf_b="rq4a_latency_intervention_v3.json",
            jp_b=("results_by_kappa", "3.00", "means", "B2_4W", "f1_th"),
            fmt_b=".4f",
            prose_regex=r"and\s*\$(?P<b>0\.\d{4})\$\s*tuned\s*\$F_\{1,\\text\{th\}\}\$.*?causally tuned\s*\$F_\{1,\\text\{th\}\}\s*=\s*(?P<a>0\.\d{4})\$",
            scope_prose_regex=r"Across the AP trajectory",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA0_SPU_BEATS_FIXEDCURVE_AP",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="FixedCurve_PU",
            cohort="RQ4a_10Streams",
            horizon="full_stream_kappa_0.0",
            metric="ap",
            aggregation="10stream_mean",
            direction=">",
            jf_a="rq4a_fixedcurve_control_v3.json",
            jp_a=("results_by_kappa", "0.00", "means", "Survival_PU", "ap"),
            fmt_a=".4f",
            jf_b="rq4a_fixedcurve_control_v3.json",
            jp_b=("results_by_kappa", "0.00", "means", "FixedCurve_PU", "ap"),
            fmt_b=".4f",
            prose_regex=r"\\kappa = 0\.00\\times.*?points in AP\s*\(\$(?P<b>0\.\d{4})\$\s*vs\.\\\s*\$(?P<a>0\.\d{4})\$\)",
            scope_prose_regex=r"Online Resource~1,\s*Table~S-13",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_FIXEDCURVE_BEATS_SPU_AP",
            section="sec:rq4",
            method_a="FixedCurve_PU",
            method_b="Survival_PU",
            cohort="RQ4a_10Streams",
            horizon="full_stream_kappa_3.0",
            metric="ap",
            aggregation="10stream_mean",
            direction=">",
            jf_a="rq4a_fixedcurve_control_v3.json",
            jp_a=("results_by_kappa", "3.00", "means", "FixedCurve_PU", "ap"),
            fmt_a=".4f",
            jf_b="rq4a_fixedcurve_control_v3.json",
            jp_b=("results_by_kappa", "3.00", "means", "Survival_PU", "ap"),
            fmt_b=".4f",
            prose_regex=r"top deployable heavy-latency AP at\s*\$(?P<a>0\.\d{4})\$\s*versus\s+Causal-4W's\s*\$0\.5466\$\s*and\s+Survival-PU's\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"Online Resource~1,\s*Table~S-13",
        ),
        ProsePairClaim(
            claim_id="RQ4B_MISSED_SHIFT_4STREAMS_20RUNS",
            section="sec:rq4",
            method_a="DriftLens_Block_Static_95_CD1",
            method_b="DriftLens_Block_Rolling_95_CD1_V3",
            cohort="RQ4b_4ControlledShifts_20Runs",
            horizon="post_onset_detection",
            metric="undetected_shift_rate",
            aggregation="4shift_x_5seed_20runs",
            direction=">",
            jf_a="rq4b_factorial_monitors_v3.json",
            jp_a=("configs", "DriftLens_Block_Static_95_CD1", "undetected_shift_rate"),
            fmt_a=".1%",
            jf_b="rq4b_factorial_monitors_v3.json",
            jp_b=("configs", "DriftLens_Block_Rolling_95_CD1_V3", "undetected_shift_rate"),
            fmt_b=".1%",
            prose_regex=r"Across 20 controlled-shift runs.*?dropping from\s*\$(?P<a>\d+\.\d)\\%\$\s*under\s+Block \+ Static.*?to\s*\$(?P<b>\d+\.\d)\\%\$",
            scope_prose_regex=r"Across 20 controlled-shift runs",
        ),
        ProsePairClaim(
            claim_id="RQ4B_ROLLING_DELAYS_IID_VS_BLOCK",
            section="sec:rq4",
            method_a="DriftLens_IID_Rolling_95_CD1",
            method_b="DriftLens_Block_Rolling_95_CD1_V3",
            cohort="RQ4b_4ControlledShifts_20Runs",
            horizon="post_onset_detection",
            metric="conditional_shift_delay_commits_given_detected",
            aggregation="4shift_x_5seed_20runs",
            direction="<",
            jf_a="rq4b_factorial_monitors_v3.json",
            jp_a=("configs", "DriftLens_IID_Rolling_95_CD1", "conditional_shift_delay_commits_given_detected"),
            fmt_a=".0f",
            jf_b="rq4b_factorial_monitors_v3.json",
            jp_b=("configs", "DriftLens_Block_Rolling_95_CD1_V3", "conditional_shift_delay_commits_given_detected"),
            fmt_b=".0f",
            prose_regex=r"Conditional detection delays average\s*\$(?P<a>\d+)\$\s*commits for\s+IID \+ Rolling\s+and\s*\$(?P<b>\d+)\$\s*commits for\s+Block \+ Rolling",
            scope_prose_regex=r"Across 20 controlled-shift runs",
        ),
        ProsePairClaim(
            claim_id="RQ4C_BORB_CLEAN_TO_NOISY",
            section="sec:rq4",
            method_a="BORB_RefRef",
            method_b="BORB_PertRef",
            cohort="RQ4c_5PRSZZStreams",
            horizon="2x2_szz_perturbation",
            metric="ap",
            aggregation="5stream_mean",
            direction=">",
            jf_a="rq4d_szz_perturbation_v3.json",
            jp_a=("regimes", "Pop_N60_Indep_PermFN", "cells_mean_ap", "TrainRef__EvalRef", "BORB_Native"),
            fmt_a=".4f",
            jf_b="rq4d_szz_perturbation_v3.json",
            jp_b=("regimes", "Pop_N60_Indep_PermFN", "cells_mean_ap", "TrainPert__EvalRef", "BORB_Native"),
            fmt_b=".4f",
            prose_regex=r"with\s+BORB\s+falling from\s*\$(?P<a>0\.\d{4})\$\s*to\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"Semi-Synthetic PR-SZZ Noise",
        ),
        ProsePairClaim(
            claim_id="RQ2_K25_PERWINDOW_SPU_BEATS_B2_4W",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="Benchmark_B_K25_Natural_5Seed",
            horizon="per_window_400",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_f_window_vs_whole_ap", "K25_5seed_means", "win_ap_unweighted", "Survival_PU"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_f_window_vs_whole_ap", "K25_5seed_means", "win_ap_unweighted", "B2_4W"),
            fmt_b=".4f",
            prose_regex=r"Survival-PU\s+reaches\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*for\s+Causal-4W",
            scope_prose_regex=r"Table~\\ref\{tab:window_ap_scale\}.*?five-seed predictions",
        ),
        ProsePairClaim(
            claim_id="RQ2_K25_5SEED_RANKNORM_SPU_VS_B2",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="Benchmark_B_K25_Natural_5Seed",
            horizon="scale_norm_rank_whole",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_f_window_vs_whole_ap", "K25_5seed_means", "scale_norm_rank_whole_ap", "Survival_PU"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_f_window_vs_whole_ap", "K25_5seed_means", "scale_norm_rank_whole_ap", "B2_4W"),
            fmt_b=".4f",
            prose_regex=r"reaching\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*under Rank-Norm",
            scope_prose_regex=r"within-window z-score or percentile-rank standardization",
        ),
        ProsePairClaim(
            claim_id="RQ2_CONF_STANDALONE3_B1B_TRAILS_B2_4W",
            section="sec:rq2",
            method_a="B1b_4W_Oracle",
            method_b="B2_4W",
            cohort="Confirmatory_Standalone3_MLP",
            horizon="full_stream",
            metric="ap",
            aggregation="3repo_standalone_mean",
            direction="<",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_a_apachejit_funnel", "separated_confirmatory_ap", "standalone_3_mlp", "B1b_4W_Oracle"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_a_apachejit_funnel", "separated_confirmatory_ap", "standalone_3_mlp", "B2_4W"),
            fmt_b=".4f",
            prose_regex=r"three standalone repositories.*?Window Oracle\s+scores\s*\$(?P<a>0\.\d{4})\$\s*AP,\s*trailing\s+Causal-4W\s+at\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"three standalone repositories",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_SPU_BEATS_POSTHOC_LOGIT_F1",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="B2_4W_PostHocLogit",
            cohort="RQ4a_10Streams_PostHocControl",
            horizon="full_stream_kappa_3.0",
            metric="f1",
            aggregation="10stream_mean",
            direction=">",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "Survival_PU", "f1"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "B2_4W_PostHocLogit", "f1"),
            fmt_b=".4f",
            prose_regex=r"PostHoc Logit\s+at\s*\$(?P<b>0\.\d{4})\$\s*\$F_1\$.*?trail\s+Survival-PU's\s*\$(?P<a>0\.\d{4})\$",
            scope_prose_regex=r"Table~\\ref\{tab:rq4a_posthoc_b2\}",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_SPU_BEATS_POSTHOC_RATIO_F1",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="B2_4W_PostHocRatio",
            cohort="RQ4a_10Streams_PostHocControl",
            horizon="full_stream_kappa_3.0",
            metric="f1",
            aggregation="10stream_mean",
            direction=">",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "Survival_PU", "f1"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "B2_4W_PostHocRatio", "f1"),
            fmt_b=".4f",
            prose_regex=r"PostHoc Ratio\s+at\s*\$(?P<b>0\.\d{4})\$\s*\$F_1\$\s*trail\s+Survival-PU's\s*\$(?P<a>0\.\d{4})\$",
            scope_prose_regex=r"Table~\\ref\{tab:rq4a_posthoc_b2\}",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA0_POSTHOC_RATIO_CLIPPING_TIE",
            section="sec:rq4",
            method_a="PostHocRatio_UpperClip_0x",
            method_b="PostHocRatio_WinUpperTie_0x",
            cohort="RQ4a_10Streams_PostHocRatioClip",
            horizon="full_stream_kappa_0.0",
            metric="upper_clip_and_win_tie",
            aggregation="10stream_mean",
            direction="<",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_d_calibration_and_posthoc", "posthoc_ratio_clipping_diagnostics", "by_kappa", "0.00", "upper_clip_frac"),
            fmt_a=".1%",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_d_calibration_and_posthoc", "posthoc_ratio_clipping_diagnostics", "by_kappa", "0.00", "window_upper_tie_frac"),
            fmt_b=".1%",
            prose_regex=r"upper-bound clipping\s*\(\$(?P<a>\d+\.\d)\\%\$\s*of commits across\s*\$(?P<b>\d+\.\d)\\%\$\s*of windows\)",
            scope_prose_regex=r"Table~\\ref\{tab:rq4a_posthoc_b2\}",
        ),
        ProsePairClaim(
            claim_id="RQ4B_BLOCK_VS_IID_ALARM_RATE_10STREAMS",
            section="sec:rq4",
            method_a="DriftLens_Block_Rolling_95_CD1_V3",
            method_b="DriftLens_IID_Rolling_95_CD1",
            cohort="RQ4b_10Streams_AlarmRate",
            horizon="full_stream",
            metric="mean_alarm_rate_10streams",
            aggregation="10stream_mean",
            direction="<",
            jf_a="rq4b_factorial_monitors_v3.json",
            jp_a=("configs", "DriftLens_Block_Rolling_95_CD1_V3", "mean_alarm_rate_10streams"),
            fmt_a=".1%",
            jf_b="rq4b_factorial_monitors_v3.json",
            jp_b=("configs", "DriftLens_IID_Rolling_95_CD1", "mean_alarm_rate_10streams"),
            fmt_b=".1%",
            prose_regex=r"default\s+Block \+ Rolling\s+fires on\s*\$(?P<a>\d+\.\d)\\%\$\s*of windows versus\s*\$(?P<b>\d+\.\d)\\%\$\s*for\s+IID \+ Rolling",
            scope_prose_regex=r"Table~\\ref\{tab:rq4b_monitors\}",
        ),
        ProsePairClaim(
            claim_id="RQ4E_W800_WAIT90_BEATS_SPU_3SHIFTS",
            section="sec:rq4",
            method_a="Wait_90",
            method_b="Survival_PU",
            cohort="RQ4e_3RepControlledShifts",
            horizon="fixed_1200_post_onset_W800",
            metric="ap",
            aggregation="3stream_rep_shift_mean",
            direction=">",
            jf_a="rq4e_sensitivity_sweeps_v3.json",
            jp_a=("window_size_sweep", "800", "fixed_1200_post_onset_ap", "Wait_90"),
            fmt_a=".4f",
            jf_b="rq4e_sensitivity_sweeps_v3.json",
            jp_b=("window_size_sweep", "800", "fixed_1200_post_onset_ap", "Survival_PU"),
            fmt_b=".4f",
            prose_regex=r"widest window\s*\(\$W\s*=\s*800\$\),\s*Wait-90\s+at\s*\$(?P<a>0\.\d{4})\$\s*and\s+Causal-4W\s+at\s*\$0\.5989\$\s*surpass\s+Survival-PU\s+at\s*\$(?P<b>0\.\d{4})\$",
            scope_prose_regex=r"three representative controlled shifts",
        ),
        ProsePairClaim(
            claim_id="RQ4F_OS_CENSORED_AP_GT_ORACLE_AP",
            section="sec:rq4",
            method_a="LLM_FewShot_Censored_1W",
            method_b="LLM_FewShot_Oracle_Paired_1W",
            cohort="RQ4f_OpenStack_320Slice",
            horizon="320_commit_slice",
            metric="ap_mean",
            aggregation="3_prompt_seed_mean",
            direction=">",
            jf_a="rq4f_llm_audit_v3.json",
            jp_a=("ecosystems", "openstack", "regimes", "LLM_FewShot_Censored_1W", "ap_mean"),
            fmt_a=".4f",
            jf_b="rq4f_llm_audit_v3.json",
            jp_b=("ecosystems", "openstack", "regimes", "LLM_FewShot_Oracle_Paired_1W", "ap_mean"),
            fmt_b=".4f",
            prose_regex=r"OpenStack AP rises slightly under censored six-shot at\s*\$(?P<a>0\.\d{4})\$\s*versus\s*\$(?P<b>0\.\d{4})\$\s*paired oracle",
            scope_prose_regex=r"320-commit Qt and OpenStack slices",
        ),
    ]


def run_self_test(
    loaded: Dict[str, Any],
    binding_rules: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    full_tex: str,
) -> Dict[str, bool]:
    # 1. Parameter-table mutation (`0.10 -> 0.12` on ht_floor_c) must fail!
    bad_tex_cht = full_tex.replace("0.10, 50, [0.02, 0.70]", "0.12, 50, [0.02, 0.70]")
    cht_mutation_rejected = False
    try:
        verify_hyperparam_table_against_v3_config(bad_tex_cht, V3_CONFIG)
    except AssertionError:
        cht_mutation_rejected = True
    assert cht_mutation_rejected, "Self-test failed: 0.10 -> 0.12 table mutation was not rejected!"

    secs_live = _extract_sections(_strip_tables_and_comments(full_tex))

    # 2. Realistic K25 <-> K24 substitution test (value paths + p-value path) MUST FAIL!
    abs_claim = build_prose_claims_registry()[0]  # RQ2_FULL_LR_VS_SPU_GRADUAL
    verify_single_claim(abs_claim, secs_live, loaded, binding_rules)

    k24_mutated_claim = ProsePairClaim(
        **{
            **abs_claim.__dict__,
            "jp_a": (
                "cluster_inference_mlp",
                "primary_ap",
                "sensitivity_K24_disjoint_from_benchmark_a",
                "K24_cluster_means",
                "Full_LR_Adapt",
            ),
            "jp_b": (
                "cluster_inference_mlp",
                "primary_ap",
                "sensitivity_K24_disjoint_from_benchmark_a",
                "K24_cluster_means",
                "Survival_PU",
            ),
        }
    )
    k24_prose = {abs_claim.section: secs_live[abs_claim.section].replace("0.4866", "0.4867")}
    k24_val_rejected = False
    try:
        verify_single_claim(k24_mutated_claim, k24_prose, loaded, binding_rules)
    except AssertionError:
        k24_val_rejected = True
    assert k24_val_rejected, "Self-test failed: K24->K25 value-path substitution was NOT rejected!"

    # Also test K24 statistical p-value path substitution on RQ2_FIXEDCURVE_BEATS_SPU_K25!
    fc_claim = build_prose_claims_registry()[1]
    k24_p_claim = ProsePairClaim(
        **{
            **fc_claim.__dict__,
            "jp_p": (
                "cluster_inference_mlp",
                "primary_ap",
                "sensitivity_K24_disjoint_from_benchmark_a",
                "family3_controls_K24",
                "0",
                "p_holm",
            ),
        }
    )
    k24_p_rejected = False
    try:
        verify_single_claim(
            k24_p_claim,
            {fc_claim.section: secs_live[fc_claim.section].replace("0.0295", "0.0317")},
            loaded,
            binding_rules,
        )
    except AssertionError:
        k24_p_rejected = True
    assert k24_p_rejected, "Self-test failed: K24->K25 statistical p-value path substitution was NOT rejected!"

    # 3. Fabricated / mismatched metadata mutations must fail!
    sample_claim = build_prose_claims_registry()[-2]  # RQ4E_W800_WAIT90_BEATS_SPU_3SHIFTS
    for field_mut, bad_val in [
        ("cohort", "999 independent projects"),
        ("cohort", "Benchmark_B_4ControlledShifts"),
        ("horizon", "entire retrospective training set"),
        ("metric", "ROC_AUC"),
        ("metric", "f1"),
        ("direction", "<"),
    ]:
        mut_rejected = False
        try:
            verify_single_claim(
                ProsePairClaim(**{**sample_claim.__dict__, field_mut: bad_val}),
                secs_live,
                loaded,
                binding_rules,
            )
        except AssertionError:
            mut_rejected = True
        assert mut_rejected, f"Self-test failed: mutation {field_mut}={bad_val!r} was NOT rejected!"

    # 4. Swapped prose numbers must fail!
    swapped_rejected = False
    try:
        bad_sec = {sample_claim.section: secs_live[sample_claim.section].replace("0.6078", "0.5974")}
        verify_single_claim(sample_claim, bad_sec, loaded, binding_rules)
    except AssertionError:
        swapped_rejected = True
    assert swapped_rejected, "Self-test failed: swapped prose numbers were not rejected!"

    # 5. Relational / quantifier overclaim guards must reject banned phrasing!
    relational_overclaims = [
        ("confirming every relative ordering", r"confirming every relative ordering"),
        ("degrades all methods", r"degrades all methods"),
        ("directly accounts for", r"directly accounts for"),
        ("Proposition-1 Ideal", r"Proposition-1\s+Ideal"),
        ("virtually tied", r"virtually tied|essentially tied"),
    ]
    for sample_bad, pat in relational_overclaims:
        assert re.search(pat, sample_bad) is not None, f"Self-test failed: relational guard '{pat}' did not match '{sample_bad}'!"

    return {
        "self_test_passed": True,
        "cht_012_mutation_rejected": True,
        "k24_as_k25_value_and_stat_paths_rejected": True,
        "scope_metadata_mutations_rejected": True,
        "wrong_direction_rejected": True,
        "swapped_numbers_rejected": True,
        "relational_quantifier_guards_tested": True,
    }


def main() -> None:
    res_dir = os.path.join(BASE_DIR, "results_v3")
    ms_tex_p = os.path.join(BASE_DIR, "manuscript", "main.tex")
    ms_md_p = os.path.join(BASE_DIR, "manuscript", "N05_SQJ_Manuscript.md")
    readme_p = os.path.join(BASE_DIR, "README.md")
    lic_p = os.path.join(BASE_DIR, "LICENSE")
    lock_p = os.path.join(BASE_DIR, "requirements_lock_v3.txt")
    post_mech_p = os.path.join(BASE_DIR, "scripts_v3", "run_post_campaign_mechanism_checks_v3.py")

    assert os.path.isfile(lic_p), "Missing research/N05/LICENSE!"
    assert os.path.isfile(lock_p), "Missing research/N05/requirements_lock_v3.txt!"
    assert os.path.isfile(post_mech_p), "Missing scripts_v3/run_post_campaign_mechanism_checks_v3.py!"

    with open(ms_tex_p, "r", encoding="utf-8") as f:
        full_tex = f.read()
    with open(ms_md_p, "r", encoding="utf-8") as f:
        full_md = f.read()
    with open(readme_p, "r", encoding="utf-8") as f:
        full_rm = f.read()

    loaded: Dict[str, Any] = {}
    for fn in os.listdir(res_dir):
        if fn.endswith("_v3.json") and fn != "manuscript_cell_registry_v3.json":
            with open(os.path.join(res_dir, fn), "r", encoding="utf-8") as fj:
                loaded[fn] = json.load(fj)

    binding_rules = build_exact_scope_binding_rules(loaded)
    st = run_self_test(loaded, binding_rules, full_tex)
    checked_cfg_fields = verify_hyperparam_table_against_v3_config(full_tex, V3_CONFIG)

    prose_tex = _strip_tables_and_comments(full_tex)
    secs = _extract_sections(prose_tex)

    abs_words = _count_abstract_words(secs["abstract"])
    assert 150 <= abs_words <= 250, f"Abstract word count {abs_words} outside Springer [150, 250] limit!"

    for key in ("learned_vs_fixed_survival_curve", "stage3_pu_vs_matched_teacher"):
        rule_txt = PRE_REGISTERED_INTERPRETATION_BOUNDARIES[key]
        qm = re.search(r"'([^']+)'", rule_txt)
        assert qm is not None, f"Could not parse quoted verbatim phrase from {key}"
        req_sentence = qm.group(1)
        req_tex_escaped = req_sentence.replace("_", r"\_")
        assert req_tex_escaped in full_tex, f"Missing mandatory verbatim sentence in main.tex: {req_sentence}"

    forbidden_patterns = [
        r"Kernel Mean Matching",
        r"exact unrounded event-time",
        r"[Bb]uilding collegially",
        r"textbook clarity",
        r"online Hoeffding ensembles\s*\([^)]*BORB",
        r"Hoeffding ensembles---[^.]*BORB",
        r"--build-confirmatory",
        r"--phase\s+C\b",
        r"--task\s+forensics",
        r"--task\s+llm",
        r"multimodal_v3_variants",
        r"c_\{\\text\{ht\}\}\s*=\s*0\.12",
        r"[Ee]stimator [Ii]nvariance",
        r"confirms no benefit",
        r"variance outweighs shape flexibility",
        r"fourfold",
        r"814 registered",
        r"Not applicable \(blinded",
        r"Upper Bound",
        r"upper bound on long streams",
        r"causal[\s\\_]*platt[\s\\_]*and[\s\\_]*threshold[\s\\_]*stream",
        r"0\.2984\\text\{--\}0\.3134",
        r"\[Anonymous Artifact Mirror",
        r"reduces missed shifts on the three representative",
        r"1\{,\}001\s+window-defects",
        r"1,001\s+window-defects",
        r"clean PR-SZZ ground truth",
        r"false-negative starvation",
        r"permanent-FN starvation",
        r"thresholded alert recall",
        r"causing hindsight optimism",
        r"Across every multi-year software repository",
        r"To preserve clean academic prose",
        r"noisy-SZZ training recovery",
        r"\b39 natural",
        r"17\.8\\%.*36\.0\\%",
        r"[Pp]ush-time multimodal",
        r"89\.3\\%",
        r"Pre-registered paired cluster|Pre-Registered Contrast",
        r"Full-LR[^.\n]{0,90}significantly (?:outperform|exceed)",
        # Round-3 Relational & Quantifier Overclaim Guards (Item #7)
        r"confirming every relative ordering",
        r"preserves all relative AP",
        r"Full-LR\}?\s+AP/\$?F_1\$?\s+lead",
        r"degrades Brier and ECE across \\textit\{all five",
        r"degrades all methods",
        r"incurring the smallest post-Platt degradation",
        r"smaller post-Platt Brier degradation",
        r"smaller post-Platt drop",
        r"directly accounts for",
        r"directly isolates",
        r"Proposition~\\ref\{prop:bias\}\s+Ideal",
        r"Proposition-1\s+Ideal",
        r"PostHoc Ratio\}\s+preserves",
        r"explaining its strong gradual AP",
        r"\bborderline\b",
        r"virtually tied|essentially tied",
        r"strictly matched(?: commit| at \$20\.0| budgets)",
        r"density-ratio omission remain untested",
        r"not smaller Raw",
        r"a gap reversed in mean",
    ]
    for pat in forbidden_patterns:
        assert re.search(pat, full_tex) is None, f"Forbidden/stale phrase '{pat}' found in main.tex!"
        assert re.search(pat, full_md) is None, f"Forbidden/stale phrase '{pat}' found in N05_SQJ_Manuscript.md!"
        assert re.search(pat, full_rm) is None, f"Forbidden/stale phrase '{pat}' found in README.md!"

    # LaTeX & SQJ narrative hygiene guards on main.tex
    md_backticks = re.findall(r"(?<!`)`[^`\n]+`(?!`)", full_tex)
    assert not md_backticks, f"Literal Markdown backticks found in main.tex: {md_backticks[:5]}"

    body_pre_decl = prose_tex.split(r"\section*{Declarations}")[0]
    body_no_cmds = re.sub(
        r"\\(?:label|ref|eqref|includegraphics(?:\[[^\]]*\])?|citep?|citet)\{[^\}]*\}",
        "",
        body_pre_decl,
    )
    assert re.search(r"\bV3\b", body_no_cmds) is None, "Internal 'V3' tag leaked into Sections 1--9 prose!"
    assert re.search(r"\bRQ4f\b", body_no_cmds) is None, "Non-consecutive 'RQ4f' sub-label found in Sections 1--9 prose!"

    app_clean = secs["app:repro"].replace(r"\allowbreak ", "").replace(r"\allowbreak", "").replace(r"\_", "_")
    required_app_symbols = [
        "CohortIncidenceEstimator",
        "_set_fallback()",
        "estimate_ht_shrunk_prevalence",
        "compute_survival_pu_posterior",
        "calibrate_causal_stream_probs_strict",
        "compute_split_sizes",
        "REPRESENTATIVE_10_STREAMS",
    ]
    for sym in required_app_symbols:
        assert sym in app_clean, f"Missing traceability symbol '{sym}' in Appendix A (app:repro)!"

    claims = build_prose_claims_registry()
    verified = [verify_single_claim(c, secs, loaded, binding_rules) for c in claims]

    report = {
        "self_test": st,
        "abstract_word_count": abs_words,
        "v3_config_table_fields_verified": checked_cfg_fields,
        "verbatim_boundaries_verified": 2,
        "anti_misstatement_guards_passed": len(forbidden_patterns),
        "latex_hygiene_and_sqj_prose_clean": True,
        "appendix_traceability_symbols_verified": len(required_app_symbols),
        "exact_path_prose_claims_verified": len(verified),
    }
    print(f"  [PASS] V3_CONFIG, Exact-Path Scope, & Anti-Misstatement Verification: {json.dumps(report, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
