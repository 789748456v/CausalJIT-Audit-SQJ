r"""
Automated V3 Parameter-Table, Exact-Path Scope-Validated Prose-Claim, & Anti-Overclaiming Verifier
(`scripts_v3/verify_prose_claims_v3.py`).

Decoupled SQJ verification architecture (preserving 100% SHA-256 lock across all `src_v3/*.py`
modules and archived `results_v3/*_v3.json` headers):
  - Exact 4-decimal numerical cells (839 Main + 1,054 Online Resource 1 = 1,893 registered cells)
    are verified cell-by-cell against `results_v3/*.json` by `src_v3/verify_manuscript_v3.py` and
    `scripts_v3/verify_online_resource_v3.py`.
  - This verifier audits narrative prose and protocol hygiene without forcing 4-decimal table
    regurgitation in running text:
      1. Verifies all 33 `V3_CONFIG` fields in `tab:hyperparams` + inline `$c_{\text{ht}} = 0.10$`
         directly against `src_v3/config_v3.py`.
      2. Strips `\begin{table}...\end{table}` blocks so prose checks run strictly on narrative
         text, and checks Abstract length within Springer SQJ's `[150, 250]` word limit.
      3. Validates registered prose claims (`pair`, `delta_pp`, and `direction` modes) using
         **exact positional JSON path templates** for BOTH value paths (`jp_a`, `jp_b`) AND
         statistical paths (`jp_diff`, `jp_p`), validated dynamically against `src_v3/config_v3.py`
         and `results_v3/*.json` manifests.
      4. Enforces semantic compliance with `PRE_REGISTERED_INTERPRETATION_BOUNDARIES`, 60+
         anti-misstatement & relational overclaim guards, LaTeX/SQJ hygiene, and Appendix A
         traceability symbols.
      5. Runs adversarial self-tests proving mutated `V3_CONFIG` values (`0.10 -> 0.12`),
         realistic `K25 <-> K24` value/p-value path substitutions, fabricated metadata,
         wrong directions, mutated prose deltas, and swapped prose pair percentages ALL
         deterministically fail.
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
    claim_mode: str = "delta_pp"  # "pair", "delta_pp", "direction"
    jf_diff: Optional[str] = None
    jp_diff: Optional[Tuple[str, ...]] = None
    fmt_diff: Optional[str] = None
    jf_p: Optional[str] = None
    jp_p: Optional[Tuple[str, ...]] = None
    require_sig: bool = False


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
    out = format(val, fmt)
    if out.endswith("%"):
        out = out[:-1]
    return out


def _norm_captured_num(raw: str, fmt: str) -> str:
    s = raw.strip().replace("{,}", "").replace(",", "").replace(r"\%", "").replace("%", "")
    return s.lstrip("+") if not fmt.startswith("+") else s


def _matches_exact_template(jpath: Tuple[str, ...], template: Tuple[Any, ...]) -> bool:
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
        ("Benchmark_A_MultiYear_Cens", "scheduled_1W", "censoring_range", "cohort_min_max"): {
            "val_templates": [
                ("manuscript_derived_aggregates_v3.json", ("window_censoring_audit", "*", {"cens_1w", "cens_4w"})),
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
        ("Confirmatory_Standalone3_MLP", "full_stream", "ap", "3repo_standalone_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_a_apachejit_funnel", "separated_confirmatory_ap", "standalone_3_mlp", "*")),
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
        ("RQ4a_10Streams_PostHocControl", "full_stream_kappa_3.0", "f1_th", "10stream_mean"): {
            "val_templates": [
                ("reviewer_mechanism_audits_v3.json", ("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "*", "f1_th")),
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
        ("RQ4b_10Streams_NullRate", "val_pool", "stationary_false_alarm_rate", "10stream_mean"): {
            "val_templates": [
                ("rq4b_factorial_monitors_v3.json", ("configs", "*", "stationary_false_alarm_rate")),
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
        ("RQ4b_10Streams_AlarmRate", "full_stream", "mean_alarm_rate_10streams", "10stream_mean"): {
            "val_templates": [
                ("rq4b_factorial_monitors_v3.json", ("configs", "*", "mean_alarm_rate_10streams")),
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
        ("RQ4f_OpenStack_320Slice", "320_commit_slice", "f1_mean", "3_prompt_seed_mean"): {
            "val_templates": [
                ("rq4f_llm_audit_v3.json", ("ecosystems", "openstack", "regimes", "*", "f1_mean")),
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
    else:
        raise ValueError(f"Unknown direction '{claim.direction}'")

    if claim.jf_p and claim.jp_p and claim.require_sig:
        vp = _resolve_json(loaded, claim.jf_p, claim.jp_p)
        assert vp < 0.05, f"[{claim.claim_id}] Significance check failed: p={vp} >= 0.05!"

    if claim.jf_diff and claim.jp_diff:
        vd = _resolve_json(loaded, claim.jf_diff, claim.jp_diff)
        assert abs(abs(vd) - abs(va - vb)) <= 1e-3, (
            f"[{claim.claim_id}] Registered diff path {vd} inconsistent with |va - vb| = {abs(va - vb)}!"
        )

    m = re.search(claim.prose_regex, sec_txt, flags=re.DOTALL)
    assert m is not None, (
        f"[{claim.claim_id}] Prose pattern not found in section '{claim.section}'!\n"
        f"  Pattern: {claim.prose_regex}"
    )
    gd = m.groupdict()

    if claim.claim_mode == "pair":
        assert "a" in gd and "b" in gd, f"[{claim.claim_id}] pair-mode regex must define (?P<a>...) and (?P<b>...)!"
        exp_a = _fmt_num(va, claim.fmt_a).lstrip("+")
        exp_b = _fmt_num(vb, claim.fmt_b).lstrip("+")
        got_a = _norm_captured_num(gd["a"], claim.fmt_a)
        got_b = _norm_captured_num(gd["b"], claim.fmt_b)
        assert got_a == exp_a, f"[{claim.claim_id}] Prose value A mismatch for {claim.method_a}: got '{got_a}', expected '{exp_a}'"
        assert got_b == exp_b, f"[{claim.claim_id}] Prose value B mismatch for {claim.method_b}: got '{got_b}', expected '{exp_b}'"
        checked_summary = f"{exp_a} vs {exp_b}"
    elif claim.claim_mode == "delta_pp":
        assert "delta" in gd, f"[{claim.claim_id}] delta_pp-mode regex must define (?P<delta>...)!"
        exp_pp = f"{abs(va - vb) * 100.0:.1f}"
        got_pp = _norm_captured_num(gd["delta"], ".1f")
        assert got_pp == exp_pp, (
            f"[{claim.claim_id}] Prose delta_pp mismatch ({claim.method_a} vs {claim.method_b}): "
            f"got '{got_pp}pp', expected '{exp_pp}pp' (raw |va-vb|={abs(va - vb):.6f})"
        )
        checked_summary = f"delta={exp_pp}pp"
    elif claim.claim_mode == "direction":
        checked_summary = f"{va:.4f} {claim.direction} {vb:.4f}"
    else:
        raise ValueError(f"Unknown claim_mode '{claim.claim_mode}'")

    return {
        "claim_id": claim.claim_id,
        "section": claim.section,
        "mode": claim.claim_mode,
        "pair": f"{claim.method_a} {claim.direction} {claim.method_b}",
        "cohort": claim.cohort,
        "horizon": claim.horizon,
        "metric": claim.metric,
        "aggregation": claim.aggregation,
        "verified": checked_summary,
    }


def build_prose_claims_registry() -> List[ProsePairClaim]:
    return [
        # --- RQ1 ---
        ProsePairClaim(
            claim_id="RQ1_MULTIYEAR_CENSORING_RANGE",
            section="sec:rq1",
            method_a="Cabral_14_1W",
            method_b="Qt_Natural_1W",
            cohort="Benchmark_A_MultiYear_Cens",
            horizon="scheduled_1W",
            metric="censoring_range",
            aggregation="cohort_min_max",
            direction="<",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("window_censoring_audit", "cohort_Cabral19", "cens_1w"),
            fmt_a=".1%",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("window_censoring_audit", "qt_natural_25k", "cens_1w"),
            fmt_b=".1%",
            claim_mode="pair",
            prose_regex=r"(?P<a>\d+\.\d)\\%\s+to\s+(?P<b>\d+\.\d)\\%\s+of recent defects stay unrevealed at scheduled single-window boundaries",
            scope_prose_regex=r"scheduled single-window boundaries",
        ),
        ProsePairClaim(
            claim_id="RQ1_MULTIYEAR_4W_CENSORING_RANGE",
            section="sec:rq1",
            method_a="ApacheJIT_8_4W",
            method_b="Qt_Natural_4W",
            cohort="Benchmark_A_MultiYear_Cens",
            horizon="scheduled_1W",
            metric="censoring_range",
            aggregation="cohort_min_max",
            direction="<",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("window_censoring_audit", "cohort_ApacheJIT", "cens_4w"),
            fmt_a=".1%",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("window_censoring_audit", "qt_natural_25k", "cens_4w"),
            fmt_b=".1%",
            claim_mode="pair",
            prose_regex=r"still leaving\s+(?P<a>\d+\.\d)\\%\s+to\s+(?P<b>\d+\.\d)\\%\s+of recent defects hidden across multi-year projects",
            scope_prose_regex=r"Widening replay to four windows",
        ),
        ProsePairClaim(
            claim_id="RQ1_CONFIRMATORY_CENSORING_5_VS_3",
            section="sec:rq1",
            method_a="Confirmatory_5_1W",
            method_b="Confirmatory_Standalone3_1W",
            cohort="Benchmark_A_MultiYear_Cens",
            horizon="scheduled_1W",
            metric="censoring_range",
            aggregation="cohort_min_max",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("window_censoring_audit", "cohort_Confirmatory_ApacheJIT", "cens_1w"),
            fmt_a=".1%",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("window_censoring_audit", "cohort_Confirmatory_Standalone3", "cens_1w"),
            fmt_b=".1%",
            claim_mode="pair",
            prose_regex=r"at\s+(?P<a>\d+\.\d)\\%\s+overall and\s+(?P<b>\d+\.\d)\\%\s+across its three standalone repositories",
            scope_prose_regex=r"shorter confirmatory ApacheJIT cohort",
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
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "0", "p_holm"),
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"On MLP,\s*Window Oracle exceeds matched Causal-4W by\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"across the 25 natural donor clusters",
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
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_lightgbm_ap", "family1_paradox_K25", "0", "p_holm"),
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"On LightGBM,\s*hindsight inflation doubles to\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"across the 25 natural donor clusters",
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
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"Causal-4W to outperform a frozen MLP by\s+(?P<delta>\d+\.\d)\s+AP points under gradual natural drift",
            scope_prose_regex=r"gradual natural drift",
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
            claim_mode="direction",
            prose_regex=r"Across the four controlled abrupt shifts,\s*it similarly beats freezing by roughly 5 AP points",
            scope_prose_regex=r"four controlled abrupt shifts",
        ),
        # --- RQ2 ---
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
            claim_mode="direction",
            prose_regex=r"Full-LR Frozen\) edges past single-feature LApredict and beats the strongest deployable neural host,\s*frozen DeepJIT-CNN,\s*by more than a full AP point",
            scope_prose_regex=r"multimodal Benchmark~A",
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
            claim_mode="direction",
            prose_regex=r"only non-deployable Full Hindsight DeepJIT-CNN scores higher",
            scope_prose_regex=r"multimodal Benchmark~A",
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
            claim_mode="direction",
            prose_regex=r"unsupervised LOC-Only outscores Survival-PU by over three points in effort recall",
            scope_prose_regex=r"multimodal Benchmark~A",
        ),
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
            claim_mode="direction",
            prose_regex=r"Full-LR Adapt-4W achieves the highest mean MLP AP,\s*with Causal-4W,\s*Wait-90,\s*and zero-fit FixedCurve-PU all finishing ahead of Survival-PU",
            scope_prose_regex=r"Across the 25 donor clusters",
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
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"Survival-PU significantly beats age-blind Pure-nnPU by\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"Across the 25 donor clusters",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_SPU_BEATS_ORB",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="ORB_OHT_Native",
            cohort="Benchmark_B_K25_Natural",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "Survival_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_mlp", "primary_ap", "K25_cluster_means", "ORB_OHT_Native"),
            fmt_b=".4f",
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family2_competitors_K25", "5", "p_holm"),
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"and Hoeffding ORB by\s+(?P<delta>\d+\.\d)\s+points",
            scope_prose_regex=r"Across the 25 donor clusters",
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
            jf_p="benchmark_b_main_v3.json",
            jp_p=("cluster_inference_mlp", "primary_ap", "family1_paradox_K25", "2", "p_holm"),
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"Survival-PU significantly trails Causal-4W by\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"Across the 25 donor clusters",
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
            require_sig=True,
            claim_mode="delta_pp",
            prose_regex=r"zero-fit FixedCurve-PU significantly exceeds cohort-estimated Survival-PU by\s+(?P<delta>\d+\.\d)\s+points",
            scope_prose_regex=r"On primary MLP AP",
        ),
        ProsePairClaim(
            claim_id="RQ2_BB_SPU_BEATS_TEACHER_MATCHED_LGB",
            section="sec:rq2",
            method_a="Survival_PU_LGB",
            method_b="Teacher_Matched_LGB",
            cohort="Benchmark_B_K25_Natural_LightGBM",
            horizon="full_stream",
            metric="ap",
            aggregation="K25_donor_cluster_mean",
            direction=">",
            jf_a="benchmark_b_main_v3.json",
            jp_a=("cluster_inference_lightgbm_ap", "K25_cluster_means", "Survival_PU"),
            fmt_a=".4f",
            jf_b="benchmark_b_main_v3.json",
            jp_b=("cluster_inference_lightgbm_ap", "K25_cluster_means", "Teacher_Only_Matched"),
            fmt_b=".4f",
            claim_mode="delta_pp",
            prose_regex=r"On LightGBM,\s*it lifts AP over Teacher-Matched by\s+(?P<delta>\d+\.\d)\s+points",
            scope_prose_regex=r"On primary MLP AP",
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
            claim_mode="direction",
            prose_regex=r"per-window AP favors Survival-PU by half a point",
            scope_prose_regex=r"Table~\\ref\{tab:window_ap_scale\}",
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
            claim_mode="direction",
            prose_regex=r"Standardization turns Survival-PU's deficit against Causal-4W into a small exploratory lead",
            scope_prose_regex=r"within-window z-score or percentile-rank standardization",
        ),
        ProsePairClaim(
            claim_id="RQ2_SHIFT_FULL_SPU_BEATS_B2_4W",
            section="sec:rq2",
            method_a="Survival_PU",
            method_b="B2_4W",
            cohort="Benchmark_B_4ControlledShifts",
            horizon="full_post_onset",
            metric="ap",
            aggregation="4stream_mean",
            direction=">",
            jf_a="manuscript_derived_aggregates_v3.json",
            jp_a=("benchmark_b_controlled_shift_4stream_mean", "Survival_PU", "post_onset_ap"),
            fmt_a=".4f",
            jf_b="manuscript_derived_aggregates_v3.json",
            jp_b=("benchmark_b_controlled_shift_4stream_mean", "B2_4W", "post_onset_ap"),
            fmt_b=".4f",
            claim_mode="delta_pp",
            prose_regex=r"Survival-PU leads every deployable learner\},\s*edging past ODaSC-Host and beating Causal-4W by\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"Controlled Concatenated Distribution Shifts",
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
            claim_mode="delta_pp",
            prose_regex=r"ElasticNet BORB-Alarm adapts fastest,\s*outscoring Survival-PU by\s+(?P<delta>\d+\.\d)\s+AP points",
            scope_prose_regex=r"immediate first 1,200 post-onset commits",
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
            claim_mode="direction",
            prose_regex=r"Window Oracle trails Causal-4W slightly",
            scope_prose_regex=r"three standalone repositories",
        ),
        ProsePairClaim(
            claim_id="RQ2_CONF_U4_B1B_TRAILS_B2_4W",
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
            claim_mode="direction",
            prose_regex=r"Window Oracle trails Causal-4W by half a point and lags Frozen Host by more than four points",
            scope_prose_regex=r"four-umbrella view",
        ),
        # --- RQ4 ---
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
            claim_mode="delta_pp",
            prose_regex=r"Survival-PU leads Causal-4W by\s+(?P<delta>\d+\.\d)\s+points in fixed \$F_1\$",
            scope_prose_regex=r"\\kappa = 3\.0\\times",
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
            claim_mode="delta_pp",
            prose_regex=r"and\s+(?P<delta>\d+\.\d)\s+points in tuned \$F_\{1,\\text\{th\}\}\$",
            scope_prose_regex=r"\\kappa = 3\.0\\times",
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
            claim_mode="delta_pp",
            prose_regex=r"leaving FixedCurve-PU\s+(?P<delta>\d+\.\d)\s+AP points and four \$F_1\$ points behind Survival-PU",
            scope_prose_regex=r"Table~S-13",
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
            claim_mode="direction",
            prose_regex=r"FixedCurve-PU consistently outranks Survival-PU,\s*finishing about a point ahead of Causal-4W and Survival-PU",
            scope_prose_regex=r"Table~\\ref\{tab:rq4a_posthoc_b2\}",
        ),
        ProsePairClaim(
            claim_id="RQ4A_KAPPA3_SPU_BEATS_POSTHOC_RATIO_F1TH",
            section="sec:rq4",
            method_a="Survival_PU",
            method_b="B2_4W_PostHocRatio",
            cohort="RQ4a_10Streams_PostHocControl",
            horizon="full_stream_kappa_3.0",
            metric="f1_th",
            aggregation="10stream_mean",
            direction=">",
            jf_a="reviewer_mechanism_audits_v3.json",
            jp_a=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "Survival_PU", "f1_th"),
            fmt_a=".4f",
            jf_b="reviewer_mechanism_audits_v3.json",
            jp_b=("audit_d_calibration_and_posthoc", "posthoc_b2_by_kappa", "3.00", "means", "B2_4W_PostHocRatio", "f1_th"),
            fmt_b=".4f",
            claim_mode="delta_pp",
            prose_regex=r"on scale-fair tuned \$F_\{1,\\text\{th\}\}\$,\s*Survival-PU still leads PostHoc Ratio by\s+(?P<delta>\d+\.\d)\s+points",
            scope_prose_regex=r"Table~\\ref\{tab:rq4a_posthoc_b2\}",
        ),
        ProsePairClaim(
            claim_id="RQ4B_NULL_RATE_STATIC_IID_VS_BLOCK",
            section="sec:rq4",
            method_a="DriftLens_IID_Static_95_CD1",
            method_b="DriftLens_Block_Static_95_CD1",
            cohort="RQ4b_10Streams_NullRate",
            horizon="val_pool",
            metric="stationary_false_alarm_rate",
            aggregation="10stream_mean",
            direction=">",
            jf_a="rq4b_factorial_monitors_v3.json",
            jp_a=("configs", "DriftLens_IID_Static_95_CD1", "stationary_false_alarm_rate"),
            fmt_a=".1%",
            jf_b="rq4b_factorial_monitors_v3.json",
            jp_b=("configs", "DriftLens_Block_Static_95_CD1", "stationary_false_alarm_rate"),
            fmt_b=".1%",
            claim_mode="pair",
            prose_regex=r"dropping from\s+(?P<a>\d+\.\d)\\%\s+to\s+(?P<b>\d+\.\d)\\%\s+under static references",
            scope_prose_regex=r"Table~\\ref\{tab:rq4b_monitors\}",
        ),
        ProsePairClaim(
            claim_id="RQ4B_MISSED_SHIFT_4STREAMS_20RUNS",
            section="sec:rq4",
            method_a="DriftLens_Block_Static_95_CD1",
            method_b="DriftLens_IID_Static_95_CD1",
            cohort="RQ4b_4ControlledShifts_20Runs",
            horizon="post_onset_detection",
            metric="undetected_shift_rate",
            aggregation="4shift_x_5seed_20runs",
            direction=">",
            jf_a="rq4b_factorial_monitors_v3.json",
            jp_a=("configs", "DriftLens_Block_Static_95_CD1", "undetected_shift_rate"),
            fmt_a=".1%",
            jf_b="rq4b_factorial_monitors_v3.json",
            jp_b=("configs", "DriftLens_IID_Static_95_CD1", "undetected_shift_rate"),
            fmt_b=".1%",
            claim_mode="pair",
            prose_regex=r"cutting miss rates from\s+(?P<a>\d+\.\d)\\%\s+under Block \+ Static and\s+(?P<b>\d+\.\d)\\%\s+under IID \+ Static to zero",
            scope_prose_regex=r"twenty controlled-shift runs",
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
            claim_mode="pair",
            prose_regex=r"rolling monitors average\s+(?P<a>\d+)\s+to\s+(?P<b>\d+)\s+commits of delay",
            scope_prose_regex=r"twenty controlled-shift runs",
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
            claim_mode="pair",
            prose_regex=r"default Block \+ Rolling fires on\s+(?P<a>\d+\.\d)\\%\s+of windows versus\s+(?P<b>\d+\.\d)\\%\s+for IID \+ Rolling",
            scope_prose_regex=r"Table~\\ref\{tab:rq4b_monitors\}",
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
            claim_mode="direction",
            prose_regex=r"relative to its clean-label training baseline,\s*BORB drops more than ten AP points to finish six points behind the batch learners,\s*and ORB trails them by ten points",
            scope_prose_regex=r"Semi-Synthetic PR-SZZ Noise",
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
            claim_mode="direction",
            prose_regex=r"widest 800-commit window,\s*Wait-90 and Causal-4W edge slightly ahead",
            scope_prose_regex=r"three controlled shifts",
        ),
        ProsePairClaim(
            claim_id="RQ4F_OS_CENSORED_F1_DROP",
            section="sec:rq4",
            method_a="LLM_FewShot_Oracle_Paired_1W",
            method_b="LLM_FewShot_Censored_1W",
            cohort="RQ4f_OpenStack_320Slice",
            horizon="320_commit_slice",
            metric="f1_mean",
            aggregation="3_prompt_seed_mean",
            direction=">",
            jf_a="rq4f_llm_audit_v3.json",
            jp_a=("ecosystems", "openstack", "regimes", "LLM_FewShot_Oracle_Paired_1W", "f1_mean"),
            fmt_a=".4f",
            jf_b="rq4f_llm_audit_v3.json",
            jp_b=("ecosystems", "openstack", "regimes", "LLM_FewShot_Censored_1W", "f1_mean"),
            fmt_b=".4f",
            claim_mode="delta_pp",
            prose_regex=r"Censoring those exemplar labels drops OpenStack \$F_1\$ by\s+(?P<delta>\d+\.\d)\s+points",
            scope_prose_regex=r"320-commit Qt and OpenStack slices",
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
            claim_mode="direction",
            prose_regex=r"OpenStack AP rises by one point under censored exemplars",
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
    claims_live = {c.claim_id: c for c in build_prose_claims_registry()}

    # 2. Realistic K25 <-> K24 substitution test (value paths + p-value path) MUST FAIL!
    c_spu_c4w = claims_live["RQ2_BB_SPU_TRAILS_B2_4W"]
    verify_single_claim(c_spu_c4w, secs_live, loaded, binding_rules)

    k24_mutated_claim = ProsePairClaim(
        **{
            **c_spu_c4w.__dict__,
            "jp_a": (
                "cluster_inference_mlp",
                "primary_ap",
                "sensitivity_K24_disjoint_from_benchmark_a",
                "K24_cluster_means",
                "Survival_PU",
            ),
            "jp_b": (
                "cluster_inference_mlp",
                "primary_ap",
                "sensitivity_K24_disjoint_from_benchmark_a",
                "K24_cluster_means",
                "B2_4W",
            ),
        }
    )
    k24_val_rejected = False
    try:
        verify_single_claim(k24_mutated_claim, secs_live, loaded, binding_rules)
    except AssertionError:
        k24_val_rejected = True
    assert k24_val_rejected, "Self-test failed: K24->K25 value-path substitution was NOT rejected!"

    c_fc = claims_live["RQ2_BB_FIXEDCURVE_BEATS_SPU_MLP"]
    k24_p_claim = ProsePairClaim(
        **{
            **c_fc.__dict__,
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
        verify_single_claim(k24_p_claim, secs_live, loaded, binding_rules)
    except AssertionError:
        k24_p_rejected = True
    assert k24_p_rejected, "Self-test failed: K24->K25 statistical p-value path substitution was NOT rejected!"

    # 3. Fabricated / mismatched metadata & direction mutations must fail!
    sample_claim = claims_live["RQ4E_W800_WAIT90_BEATS_SPU_3SHIFTS"]
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

    # 4. Mutated prose delta AND swapped prose pair percentages must fail!
    c_delta = claims_live["RQ4A_KAPPA3_F1_SPU_VS_B2"]
    delta_rejected = False
    try:
        bad_sec_d = {c_delta.section: secs_live[c_delta.section].replace("14.3 points", "9.9 points")}
        verify_single_claim(c_delta, bad_sec_d, loaded, binding_rules)
    except AssertionError:
        delta_rejected = True
    assert delta_rejected, "Self-test failed: mutated prose delta_pp was NOT rejected!"

    c_pair = claims_live["RQ4B_BLOCK_VS_IID_ALARM_RATE_10STREAMS"]
    swapped_rejected = False
    try:
        bad_sec_p = {
            c_pair.section: secs_live[c_pair.section]
            .replace("32.1\\% of windows versus 43.5\\%", "43.5\\% of windows versus 32.1\\%")
        }
        verify_single_claim(c_pair, bad_sec_p, loaded, binding_rules)
    except AssertionError:
        swapped_rejected = True
    assert swapped_rejected, "Self-test failed: swapped prose pair numbers were NOT rejected!"

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
        "mutated_delta_and_swapped_pair_rejected": True,
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

    # Verify semantic compliance with both PRE_REGISTERED_INTERPRETATION_BOUNDARIES without robotic verbatim quoting
    assert "learned_vs_fixed_survival_curve" in PRE_REGISTERED_INTERPRETATION_BOUNDARIES
    assert "stage3_pu_vs_matched_teacher" in PRE_REGISTERED_INTERPRETATION_BOUNDARIES
    rq2_txt = secs["sec:rq2"]
    assert re.search(
        r"adaptive Stage~1 estimation adds no whole-stream AP advantage over a fixed 90-day curve",
        rq2_txt,
    ), "Missing pre-registered negative boundary for learned vs. fixed survival curve in sec:rq2!"
    assert re.search(
        r"On MLP,\s*it yields no significant AP gain over Teacher-Matched",
        rq2_txt,
    ), "Missing pre-registered negative boundary for Stage-3 PU vs. Teacher-Matched MLP in sec:rq2!"

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
        # Relational, Quantifier, & Meta-Audit Overclaim Guards
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
        r"Quoting our codified rule verbatim",
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
        "preregistered_boundaries_verified": 2,
        "anti_misstatement_guards_passed": len(forbidden_patterns),
        "latex_hygiene_and_sqj_prose_clean": True,
        "appendix_traceability_symbols_verified": len(required_app_symbols),
        "exact_path_prose_claims_verified": len(verified),
    }
    print(f"  [PASS] V3_CONFIG, Exact-Path Scope, & Anti-Misstatement Verification: {json.dumps(report, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
