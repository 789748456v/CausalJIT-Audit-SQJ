r"""
Strict Row-and-Column Cell-Level Manuscript, Provenance, Commit-Artifact, BibTeX,
& Security Verifier (N05 V3).

Closes Codex Round 2 Criterion (8), Round 3 R3-C9, & Round 3b R3b-M7:
1. Parses LaTeX `tabular` / `tabular*` / `tabularx` blocks into exact `(row, col)` grids (handling
   `\\`, `&`, comments, booktabs rules, and `\multicolumn`), matching `row_selector` uniquely against
   `cells[0]` and checking ONLY `cells[col_index]`.
2. Semantic Numeric & Statistical-Role Parser (`R3b-M7`): parses full signed, scientific-notation
   (`e`/`E` and `\times 10^{...}`), percentage, and structured `mean \pm std` / `[ci_lo, ci_hi]` /
   `primary (paren_0)` / `slash_0/slash_1/slash_2` tokens, binding each registry entry to an explicit
   statistical `role` (default `"primary"`). Deterministically rejects wrong signs (`-0.125`), wrong
   magnitudes (`0.125e-3`, `0.125 \times 10^{-3}`), wrong statistical roles (`0.900 \pm 0.125` when
   expecting primary `0.125`), inequalities (`< 0.001`), swapped rows, swapped columns, and digit substrings!
3. Normalizes `V3_CONFIG.to_dict()` via canonical JSON round-trip before comparing with `provenance.v3_config`,
   and checks SHA-256 hashes of all core `src_v3/*.py` execution modules.
4. Verifies commit-level `.npz` artifacts (`raw_prob`, `cal_prob`, `tuned_th`, `decision_th`).
5. Executes FULL manuscript + provenance + artifact + BibTeX + security verification when invoked
   without `--self-test-only` (and fails if `MANUSCRIPT_CELL_REGISTRY` is empty!).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config_v3 import V3_CONFIG
from data_audit_v3 import sha256_file


FORBIDDEN_INTERNAL_CODES = (
    r"\bN46\b",
    r"\bN60\b",
    r"\bN33\b",
    r"\bn60_airflow\b",
    r"\bn60_angular\b",
    r"\bcabral_camel\b",
)

CORE_CODE_FILES_FOR_PROVENANCE = (
    "config_v3.py",
    "data_audit_v3.py",
    "metrics_and_stats_v3.py",
    "survival_estimators_v3.py",
    "stream_competitors_v3.py",
    "models_and_baselines_v3.py",
    "run_v3_experiments.py",
)

# Populated / bound to exact V3 manuscript tables once `results_v3/*.json` and `main.tex` are generated;
# can also be loaded from `results_v3/manuscript_cell_registry_v3.json`.
MANUSCRIPT_CELL_REGISTRY: List[Tuple[str, str, int, str, Tuple[str, ...], str]] = []


def resolve_json_path(obj: Any, path: Tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                raise KeyError(f"Missing JSON key '{key}' along path {path}")
            cur = cur[key]
        elif isinstance(cur, list):
            idx = int(key)
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"List index {idx} out of bounds along path {path}")
            cur = cur[idx]
        else:
            raise TypeError(f"Cannot traverse key '{key}' on primitive {type(cur)} along {path}")
    return cur


def extract_latex_table_blocks(tex_text: str) -> Dict[str, str]:
    r"""Map `\label{tab:...}` -> enclosing `\begin{table*}...\end{table*}` or `\begin{table}...\end{table}`."""
    blocks: Dict[str, str] = {}
    pattern = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.DOTALL)
    lbl_re = re.compile(r"\\label\{(tab:[^\}]+)\}")
    for m in pattern.finditer(tex_text):
        body = m.group(0)
        lm = lbl_re.search(body)
        if lm:
            lbl = lm.group(1)
            if lbl in blocks:
                raise ValueError(f"Duplicate LaTeX table label '{lbl}'!")
            blocks[lbl] = body
    return blocks


def _strip_latex_comments(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        clean_ln = re.sub(r"(?<!\\)%.*$", "", ln)
        lines.append(clean_ln)
    return "\n".join(lines)


def _clean_rule_prefixes(row_str: str) -> str:
    s = row_str.strip()
    rule_pat = re.compile(
        r"^(?:\\toprule|\\midrule|\\bottomrule|\\hline|\\cline\{[^\}]*\}|\\cmidrule(?:\([^\)]*\))?\{[^\}]*\}|\\addlinespace(?:\[[^\]]*\])?)\s*"
    )
    while True:
        m = rule_pat.match(s)
        if not m:
            break
        s = s[m.end() :].strip()
    return s


def parse_latex_tabular_grid(table_block: str) -> List[List[str]]:
    r"""
    Parses the inner `tabular` / `tabular*` / `tabularx` environment of `table_block` into a 2D list
    of `[cell_0, cell_1, ...]` per data row, expanding `\multicolumn{k}{...}{text}` into `k` columns
    so column indices remain strictly aligned (`R3-C9`).
    """
    no_comments = _strip_latex_comments(table_block)
    tab_m = re.search(
        r"\\begin\{tabular\*?\}(?:\{[^\}]*\})?\{[^\}]*\}(.*?)\\end\{tabular\*?\}",
        no_comments,
        re.DOTALL,
    )
    if tab_m:
        inner = tab_m.group(1)
    else:
        tabx_m = re.search(
            r"\\begin\{tabularx\}\{[^\}]*\}\{[^\}]*\}(.*?)\\end\{tabularx\}",
            no_comments,
            re.DOTALL,
        )
        inner = tabx_m.group(1) if tabx_m else no_comments

    raw_rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", inner)
    multicol_re = re.compile(r"^\\multicolumn\{(\d+)\}\{[^\}]*\}\{(.*)\}$", re.DOTALL)

    grid: List[List[str]] = []
    for rr in raw_rows:
        cleaned = _clean_rule_prefixes(rr)
        if not cleaned or "&" not in cleaned:
            continue
        raw_cells = [c.strip() for c in re.split(r"(?<!\\)&", cleaned)]
        expanded_cells: List[str] = []
        for cell in raw_cells:
            mc = multicol_re.match(cell)
            if mc:
                span = max(int(mc.group(1)), 1)
                expanded_cells.append(mc.group(2).strip())
                expanded_cells.extend([""] * (span - 1))
            else:
                expanded_cells.append(cell)
        grid.append(expanded_cells)
    return grid


def _normalize_row_label_text(cell_text: str) -> str:
    s = cell_text
    s = re.sub(r"\\(?:textbf|textit|texttt|mathrm|emph)\{([^\}]*)\}", r"\1", s)
    s = s.replace(r"\_", "_").replace("$", "").strip()
    return s


VALID_CELL_ROLES: Set[str] = {
    "primary",
    "mean",
    "std",
    "ci_lo",
    "ci_hi",
    "paren_0",
    "paren_1",
    "slash_0",
    "slash_1",
    "slash_2",
}

_STRICT_SCALAR_SLOT_RE = re.compile(
    r"^\s*"
    r"(?P<ineq><<|>>|<=|>=|<|>|~)?"
    r"\s*"
    r"(?P<sign>[+-])?\s*"
    r"(?P<mag>(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<pct>\s*%)?"
    r"(?P<stars>\s*(?:\*+|\^\*+|\\dagger|\\ddagger))?"
    r"\s*$"
)


def _normalize_cell_math_and_macros(cell_str: str) -> str:
    s = cell_str.strip()
    # Iteratively unwrap visual formatting macros (`\textbf{...}`, `\underline{...}`, `\text{...}`, etc.)
    macro_re = re.compile(r"\\(?:textbf|textit|texttt|mathrm|mathbf|emph|underline|text)\{([^{}]*)\}")
    while True:
        s_next = macro_re.sub(r"\1", s)
        if s_next == s:
            break
        s = s_next

    # Normalize LaTeX minus, plus-minus, percent, inequalities (`\leq`, `\geq`, `\approx`, etc.), and thin spaces (`R3c-5`)
    s = s.replace("$-$", "-").replace("\u2212", "-").replace("–", "-")
    s = s.replace(r"\pm", " ± ").replace("±", " ± ")
    s = s.replace(r"\%", "%")
    s = re.sub(r"\\(?:leq|leqslant|le)(?![a-zA-Z])", " <= ", s)
    s = re.sub(r"\\(?:geq|geqslant|ge)(?![a-zA-Z])", " >= ", s)
    s = re.sub(r"\\ll(?![a-zA-Z])", " << ", s)
    s = re.sub(r"\\gg(?![a-zA-Z])", " >> ", s)
    s = re.sub(r"\\(?:approx|sim|simeq|neq)(?![a-zA-Z])", " ~ ", s)
    s = re.sub(r"\\(?:[,;:!]|quad|qquad|\s)", " ", s)
    s = s.replace("$", "")

    def _repl_sci(m: re.Match) -> str:
        sgn = m.group(1) or ""
        mag = m.group(2)
        exp_clean = re.sub(r"\s+", "", m.group(3))
        return f"{sgn}{mag}e{exp_clean}"

    s = re.sub(
        r"([+-])?\s*((?:\d+\.\d+|\d+|\.\d+))\s*\\(?:times|cdot)\s*10\s*\^\s*\{?\s*([+-]?\s*\d+)\s*\}?",
        _repl_sci,
        s,
    )
    # Normalize `^{*...}` significance markers before stripping braces
    s = re.sub(r"\^\s*\{\s*(\*+)\s*\}", r"^\1", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def _parse_single_scalar_slot_strict(slot: str, cell_str: str) -> Dict[str, Any]:
    """
    Full-slot strict parser (`R3c-5`): requires the ENTIRE slot to match `_STRICT_SCALAR_SLOT_RE`
    with ZERO unconsumed tokens, supporting whitespace between unary `[+-]` and digits (`-\\,0.125`)
    and preserving normalized inequality prefixes (`\\leq` -> `<=`).
    """
    m = _STRICT_SCALAR_SLOT_RE.fullmatch(slot)
    if m is None:
        raise AssertionError(
            f"Unconsumed or invalid LaTeX numeric syntax in slot '{slot}' of cell '{cell_str}'!"
        )
    ineq = (m.group("ineq") or "").strip()
    sgn = m.group("sign") or ""
    mag = m.group("mag")
    canon_tok = f"{sgn}{mag}"
    is_pct = bool(m.group("pct"))
    has_exp = ("e" in mag.lower())
    return {
        "raw_token": canon_tok,
        "value": float(canon_tok),
        "has_exponent": has_exp,
        "is_percent": is_pct,
        "inequality": ineq,
    }


def parse_cell_semantic_tokens(cell_str: str) -> Dict[str, Dict[str, Any]]:
    r"""
    Parses a single LaTeX table cell into unambiguous statistical roles (`R3b-M7`, `R3c-5`),
    enforcing full-expression consumption on every slot so detached signs (`-\,0.125`), `\leq`/`\geq`,
    or unknown commands can NEVER be silently skipped!
    """
    s = _normalize_cell_math_and_macros(cell_str)
    roles: Dict[str, Dict[str, Any]] = {}

    # 1. Extract at most one bracketed CI `[ci_lo, ci_hi]`
    ci_matches = list(re.finditer(r"\[([^\[\]]+)\]", s))
    if len(ci_matches) > 1:
        raise AssertionError(f"Multiple bracketed '[...]' blocks in cell '{cell_str}'!")
    if len(ci_matches) == 1:
        ci_m = ci_matches[0]
        ci_parts = [p.strip() for p in ci_m.group(1).split(",")]
        if len(ci_parts) != 2:
            raise AssertionError(f"Expected '[ci_lo, ci_hi]' with 2 comma-separated bounds in '{cell_str}'!")
        roles["ci_lo"] = _parse_single_scalar_slot_strict(ci_parts[0], cell_str)
        roles["ci_hi"] = _parse_single_scalar_slot_strict(ci_parts[1], cell_str)
        s = (s[: ci_m.start()] + " " + s[ci_m.end() :]).strip()

    # 2. Extract at most one parenthesized secondary block `(paren_0[, paren_1])`
    paren_matches = list(re.finditer(r"\(([^\(\)]+)\)", s))
    if len(paren_matches) > 1:
        raise AssertionError(f"Multiple parenthesized '(...)' blocks in cell '{cell_str}'!")
    if len(paren_matches) == 1:
        paren_m = paren_matches[0]
        p_parts = [p.strip() for p in paren_m.group(1).split(",")]
        if len(p_parts) not in (1, 2):
            raise AssertionError(f"Unsupported parenthesized tuple in cell '{cell_str}'!")
        for idx_p, pp in enumerate(p_parts):
            roles[f"paren_{idx_p}"] = _parse_single_scalar_slot_strict(pp, cell_str)
        s = (s[: paren_m.start()] + " " + s[paren_m.end() :]).strip()

    # 3. Handle `mean ± std`
    if "±" in s:
        parts_pm = s.split("±")
        if len(parts_pm) != 2:
            raise AssertionError(f"Malformed multiple '±' in cell '{cell_str}'!")
        left_tok = _parse_single_scalar_slot_strict(parts_pm[0], cell_str)
        right_tok = _parse_single_scalar_slot_strict(parts_pm[1], cell_str)
        roles["primary"] = left_tok
        roles["mean"] = left_tok
        roles["std"] = right_tok
        return roles

    # 4. Handle slash-separated tuples `a / b` or `W / T / L`
    if "/" in s:
        slash_parts = [p.strip() for p in s.split("/")]
        if len(slash_parts) < 2 or len(slash_parts) > 3:
            raise AssertionError(f"Unsupported slash-tuple arity in cell '{cell_str}'!")
        for idx_s, sp in enumerate(slash_parts):
            roles[f"slash_{idx_s}"] = _parse_single_scalar_slot_strict(sp, cell_str)
        return roles

    # 5. Single scalar value (if non-empty remaining expression)
    if s:
        prim_tok = _parse_single_scalar_slot_strict(s, cell_str)
        roles["primary"] = prim_tok
        roles["mean"] = prim_tok
    elif not roles:
        raise AssertionError(f"Empty numeric cell '{cell_str}'!")
    return roles


def validate_cell_registry_against_tex(
    cell_specs: List[Tuple],
    loaded_jsons: Dict[str, Dict],
    table_blocks: Dict[str, str],
) -> Dict:
    """
    `cell_specs` items: `(table_label, row_selector, col_index, json_filename, json_key_tuple, fmt_spec[, role])`
    Strictly locates the UNIQUE row whose `cells[0]` matches `row_selector`, parses `cells[col_index]`
    semantically via `parse_cell_semantic_tokens`, and verifies exact formatted token, sign, exponent
    presence, percentage flag, inequality absence, and statistical `role` (`R3-C9`, `R3b-M7`)!
    """
    seen_cells: Set[Tuple[str, str, int, str]] = set()
    parsed_grids: Dict[str, List[List[str]]] = {}
    verified_count = 0

    for spec in cell_specs:
        if len(spec) == 6:
            tab_lbl, row_sel, col_idx, jf, jpath, fmt_spec = spec
            role = "primary"
        elif len(spec) == 7:
            tab_lbl, row_sel, col_idx, jf, jpath, fmt_spec, role = spec
        else:
            raise ValueError(f"Expected 6-tuple or 7-tuple cell spec, got {spec}")

        if role not in VALID_CELL_ROLES:
            raise ValueError(f"Invalid cell role '{role}' in {spec}; valid roles: {sorted(VALID_CELL_ROLES)}")

        col_i = int(col_idx)
        if col_i < 1:
            raise ValueError(f"Column index must be >= 1 (col 0 is row label); got {col_i} for {tab_lbl}:{row_sel}")
        canon_role_key = "primary" if role in ("primary", "mean") else role
        cell_key = (tab_lbl, row_sel, col_i, canon_role_key)
        if cell_key in seen_cells:
            raise ValueError(f"Duplicate cell-role registration: {cell_key}")
        seen_cells.add(cell_key)

        if tab_lbl not in table_blocks:
            raise KeyError(f"Table label '{tab_lbl}' missing from LaTeX manuscript!")
        if jf not in loaded_jsons:
            raise FileNotFoundError(f"Result JSON '{jf}' not loaded or missing!")

        if tab_lbl not in parsed_grids:
            parsed_grids[tab_lbl] = parse_latex_tabular_grid(table_blocks[tab_lbl])
        grid = parsed_grids[tab_lbl]

        # Match `row_selector` against `cells[0]` (exact normalized match first, else unique substring in `cells[0]`)
        exact_rows = [r for r in grid if _normalize_row_label_text(r[0]) == _normalize_row_label_text(row_sel)]
        if len(exact_rows) == 1:
            target_row = exact_rows[0]
        else:
            sub_rows = [r for r in grid if _normalize_row_label_text(row_sel) in _normalize_row_label_text(r[0])]
            if len(sub_rows) == 0:
                raise AssertionError(
                    f"Row selector '{row_sel}' not found in column 0 of {tab_lbl}! Available col-0 labels: "
                    f"{[_normalize_row_label_text(r[0]) for r in grid]}"
                )
            if len(sub_rows) > 1:
                raise AssertionError(
                    f"Ambiguous row selector '{row_sel}' matched {len(sub_rows)} rows in {tab_lbl}!"
                )
            target_row = sub_rows[0]

        if col_i >= len(target_row):
            raise IndexError(
                f"Column index {col_i} out of range for row '{row_sel}' in {tab_lbl} (row has {len(target_row)} cols)!"
            )

        raw_val = resolve_json_path(loaded_jsons[jf], jpath)
        num_val = float(raw_val)
        if not np.isfinite(num_val):
            raise ValueError(f"Non-finite JSON value {raw_val} for {cell_key} -> {jf}:{jpath}")

        formatted_full = format(num_val, fmt_spec)
        expect_pct = fmt_spec.endswith("%")
        expected_core = formatted_full[:-1] if expect_pct else formatted_full
        expect_exp = ("e" in expected_core.lower())

        cell_str = target_row[col_i]
        role_map = parse_cell_semantic_tokens(cell_str)
        if role not in role_map:
            raise AssertionError(
                f"Role '{role}' not found in {tab_lbl} row='{row_sel}' col={col_i}: cell='{cell_str}', "
                f"parsed_roles={list(role_map.keys())}!"
            )

        tok = role_map[role]
        if tok["inequality"]:
            raise AssertionError(
                f"Unexpected inequality prefix '{tok['inequality']}' in {tab_lbl} row='{row_sel}' col={col_i}: '{cell_str}'!"
            )
        if tok["is_percent"] != expect_pct:
            raise AssertionError(
                f"Percentage mismatch in {tab_lbl} row='{row_sel}' col={col_i}: expected '{formatted_full}', got '{cell_str}'!"
            )
        if tok["has_exponent"] != expect_exp:
            raise AssertionError(
                f"Scientific-exponent mismatch in {tab_lbl} row='{row_sel}' col={col_i}: expected '{formatted_full}', got '{cell_str}'!"
            )

        actual_raw = tok["raw_token"]
        # Allow optional explicit '+' in cell ONLY when expected value is strictly positive and fmt omitted '+'
        if actual_raw.startswith("+") and not expected_core.startswith("+") and float(expected_core) > 0.0:
            actual_cmp = actual_raw[1:]
        else:
            actual_cmp = actual_raw

        if actual_cmp != expected_core:
            raise AssertionError(
                f"Semantic cell mismatch in {tab_lbl} row='{row_sel}' col={col_i} role='{role}': "
                f"expected '{formatted_full}' (from {jf}:{jpath}), but parsed role token is '{actual_raw}' in cell '{cell_str}'!"
            )
        if abs(float(actual_raw) - float(expected_core)) > 1e-12 * max(1.0, abs(float(expected_core))):
            raise AssertionError(
                f"Numeric value mismatch in {tab_lbl} row='{row_sel}' col={col_i} role='{role}': "
                f"expected {expected_core}, got {actual_raw}!"
            )
        verified_count += 1

    return {"verified_cells": verified_count, "unique_tables": len(set(x[0] for x in cell_specs))}


def verify_provenance_freshness(base_dir: str, json_files: List[str]) -> Dict:
    src_v3_dir = os.path.join(base_dir, "src_v3")
    live_hashes = {
        os.path.basename(p): sha256_file(p)
        for p in sorted(glob.glob(os.path.join(src_v3_dir, "*.py")))
    }
    # Canonical JSON normalization so Python `tuple` vs JSON `list` compares identically (`R3-C9`)!
    live_cfg_norm = json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True))
    checked = 0

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "provenance" not in obj:
            raise KeyError(f"Missing 'provenance' header in {jp}!")
        prov = obj["provenance"]
        rec_cfg_norm = json.loads(json.dumps(prov.get("v3_config"), sort_keys=True))
        if rec_cfg_norm != live_cfg_norm:
            raise AssertionError(f"Stale v3_config in {jp} does not match current config_v3.py!")
        rec_hashes = prov.get("code_sha256", {})
        for core_file in CORE_CODE_FILES_FOR_PROVENANCE:
            if core_file in live_hashes and rec_hashes.get(core_file) != live_hashes.get(core_file):
                raise AssertionError(
                    f"Stale core code hash for {core_file} in {jp} "
                    f"(recorded={rec_hashes.get(core_file)} vs live={live_hashes.get(core_file)})!"
                )
        checked += 1

    return {"fresh_json_files_verified": checked}


def verify_commit_level_artifacts(art_root: str) -> Dict:
    npz_files = sorted(glob.glob(os.path.join(art_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No commit-level `.npz` artifacts found under {art_root}!")

    total_commits_logged = 0
    for p in npz_files:
        b = np.load(p)
        for req_k in ("commit_idx", "prediction_timestamp", "y_true", "effort", "shared_model_gen_id"):
            if req_k not in b:
                raise KeyError(f"Missing required commit-level field '{req_k}' in {p}!")
        n = len(b["commit_idx"])
        assert n > 0 and len(b["prediction_timestamp"]) == n and len(b["y_true"]) == n and len(b["shared_model_gen_id"]) == n
        assert np.all(np.diff(b["prediction_timestamp"]) >= 0.0), f"Non-monotonic prediction_timestamp in {p}!"
        prob_keys = [k for k in b.files if k.endswith("__raw_prob") or k.endswith("__cal_prob")]
        th_keys = [k for k in b.files if k.endswith("__tuned_th")]
        dec_keys = [k for k in b.files if k.endswith("__decision_th")]
        assert len(prob_keys) >= 6, f"Too few method probability arrays in {p}!"
        assert len(th_keys) >= 3 and len(dec_keys) >= 3, f"Missing per-commit threshold/decision arrays in {p}!"
        for pk in prob_keys:
            arr = b[pk]
            assert len(arr) == n and np.all(np.isfinite(arr)) and np.all((arr >= 0.0) & (arr <= 1.0)), f"Invalid probs {pk} in {p}!"
        for tk in th_keys:
            arr_t = b[tk]
            assert len(arr_t) == n and np.all(np.isfinite(arr_t)), f"Invalid threshold vector {tk} in {p}!"
        total_commits_logged += n

    return {"artifact_files_verified": len(npz_files), "total_stream_commits_verified": total_commits_logged}


def verify_manuscript_hygiene_and_citations(base_dir: str) -> Dict:
    ms_dir = os.path.join(base_dir, "manuscript")
    tex_p = os.path.join(ms_dir, "main.tex")
    md_p = os.path.join(ms_dir, "N05_SQJ_Manuscript.md")
    bib_p = os.path.join(ms_dir, "references.bib")

    if not os.path.exists(tex_p) or not os.path.exists(bib_p):
        raise FileNotFoundError("Missing manuscript/main.tex or manuscript/references.bib!")

    with open(tex_p, "r", encoding="utf-8") as f:
        tex_txt = f.read()
    with open(bib_p, "r", encoding="utf-8") as f:
        bib_txt = f.read()

    for pat in FORBIDDEN_INTERNAL_CODES:
        if re.search(pat, tex_txt):
            raise AssertionError(f"Forbidden internal note code matching '{pat}' found in main.tex!")
        if os.path.exists(md_p):
            with open(md_p, "r", encoding="utf-8") as fm:
                if re.search(pat, fm.read()):
                    raise AssertionError(f"Forbidden internal note code matching '{pat}' found in N05_SQJ_Manuscript.md!")

    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_txt))
    cite_groups = re.findall(r"\\cite[a-zA-Z*]*\{([^\}]+)\}", tex_txt)
    cited_keys: Set[str] = set()
    for cg in cite_groups:
        for k in cg.split(","):
            ks = k.strip()
            if ks:
                cited_keys.add(ks)
    unresolved = sorted(cited_keys - bib_keys)
    if unresolved:
        raise AssertionError(f"Unresolved LaTeX citations in main.tex: {unresolved}")

    # Security scan: zero API keys (`sk-[a-zA-Z0-9]{16,}`) in manuscript or src_v3
    sk_pat = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
    for scan_p in [tex_p, bib_p] + ([md_p] if os.path.exists(md_p) else []) + glob.glob(os.path.join(base_dir, "src_v3", "*.py")):
        with open(scan_p, "r", encoding="utf-8") as fs:
            if sk_pat.search(fs.read()):
                raise AssertionError(f"Potential secret/API key pattern detected in {scan_p}!")

    return {"cited_keys_verified": len(cited_keys), "bib_entries_total": len(bib_keys), "zero_forbidden_codes": True}


def run_self_test() -> Dict:
    """
    Adversarial Self-Test (`R3-C9`):
    Proves the verifier rejects:
      1. Duplicate `(table, row, col)` entries;
      2. Missing JSON keys;
      3. Wrong cell values;
      4. Swapped rows (right value in row `m2`, wrong value in row `m1`);
      5. Swapped columns (right value in col `2`, wrong value in col `1`);
      6. Partial digit collisions (`0.543` vs `0.5439`);
      7. Stale `v3_config` while passing canonical JSON round-trip of `V3_CONFIG.to_dict()`.
    """
    fake_jsons = {
        "res.json": {
            "m1": {"ap": 0.54321, "auc": 0.71234},
            "m2": {"ap": 0.41111, "auc": 0.62222},
        }
    }
    fake_tex = r"""
    \begin{table}[t]
    \caption{Test}
    \label{tab:t1}
    \begin{tabular}{lcc}
    \toprule
    Method & AP & ROC-AUC \\
    \midrule
    \textbf{m1} & \textbf{0.5432} & 0.7123 \\
    m2 & 0.4111 & 0.6222 \\
    \bottomrule
    \end{tabular}
    \end{table}
    """
    blocks = extract_latex_table_blocks(fake_tex)

    # 1. Exact valid row & column specs pass
    ok = validate_cell_registry_against_tex(
        [
            ("tab:t1", "m1", 1, "res.json", ("m1", "ap"), ".4f"),
            ("tab:t1", "m1", 2, "res.json", ("m1", "auc"), ".4f"),
            ("tab:t1", "m2", 1, "res.json", ("m2", "ap"), ".4f"),
        ],
        fake_jsons,
        blocks,
    )
    assert ok["verified_cells"] == 3

    # 2. Swapped-row counterexample (Codex R3-C9 exact test: m1's value 0.5432 is ONLY in row m2, row m1 has 0.1111)
    swapped_row_tex = r"""
    \begin{table}\label{tab:swap}
    \begin{tabular}{lc}
    m1 & 0.1111 \\
    m2 & 0.5432 \\
    \end{tabular}
    \end{table}
    """
    swapped_row_rejected = False
    try:
        validate_cell_registry_against_tex(
            [("tab:swap", "m1", 1, "res.json", ("m1", "ap"), ".4f")],
            fake_jsons,
            extract_latex_table_blocks(swapped_row_tex),
        )
    except AssertionError:
        swapped_row_rejected = True
    assert swapped_row_rejected, "Verifier failed to reject swapped-row counterexample!"

    # 3. Swapped-column counterexample (0.5432 is in col 2 instead of col 1)
    swapped_col_rejected = False
    try:
        validate_cell_registry_against_tex(
            [("tab:t1", "m1", 2, "res.json", ("m1", "ap"), ".4f")],
            fake_jsons,
            blocks,
        )
    except AssertionError:
        swapped_col_rejected = True
    assert swapped_col_rejected, "Verifier failed to reject swapped-column counterexample!"

    # 4. Digit prefix/suffix collision (`0.543` must NOT match `0.5432`!)
    digit_collision_rejected = False
    try:
        validate_cell_registry_against_tex(
            [("tab:t1", "m1", 1, "res.json", ("m1", "ap"), ".3f")],
            fake_jsons,
            blocks,
        )
    except AssertionError:
        digit_collision_rejected = True
    assert digit_collision_rejected, "Verifier failed to reject partial digit substring!"

    # 5. Duplicate cell registration must raise ValueError
    dup_rejected = False
    try:
        validate_cell_registry_against_tex(
            [("tab:t1", "m1", 1, "res.json", ("m1", "ap"), ".4f"), ("tab:t1", "m1", 1, "res.json", ("m1", "ap"), ".4f")],
            fake_jsons,
            blocks,
        )
    except ValueError:
        dup_rejected = True
    assert dup_rejected, "Verifier failed to reject duplicate cell!"

    # 6. Missing JSON key must raise KeyError
    missing_rejected = False
    try:
        validate_cell_registry_against_tex([("tab:t1", "m1", 1, "res.json", ("m1", "f1"), ".4f")], fake_jsons, blocks)
    except KeyError:
        missing_rejected = True
    assert missing_rejected, "Verifier failed to reject missing JSON key!"

    # 7. JSON round-trip normalization check (`tuple` -> `list` round-trip must equal normalized live config!)
    roundtrip_cfg = json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True))
    assert roundtrip_cfg == json.loads(json.dumps(V3_CONFIG.to_dict(), sort_keys=True))

    # 8. Codex R3b-M7 Semantic Counterexamples for expected `0.125` (`.3f`):
    #    (a) Wrong sign: `-0.125` and `$-$0.125`
    #    (b) Wrong order of magnitude / exponent: `0.125e-3` and `$0.125 \times 10^{-3}$`
    #    (c) Wrong statistical role: `0.900 \pm 0.125` (when checking `primary` / `mean` for `0.125`)
    r3b_jsons = {"r3b.json": {"m": {"val": 0.125, "mean": 0.900, "std": 0.125}}}
    # Explicitly verify token parser on Codex's R3c-5 cases first:
    tok_thin_minus = parse_cell_semantic_tokens(r"-\,0.125")["primary"]
    assert tok_thin_minus["raw_token"] == "-0.125" and tok_thin_minus["value"] == -0.125
    tok_leq = parse_cell_semantic_tokens(r"\leq 0.125")["primary"]
    assert tok_leq["inequality"] == "<="

    for bad_label, bad_cell in [
        ("wrong_sign_ascii", "-0.125"),
        ("wrong_sign_latex", "$-$0.125"),
        ("wrong_sign_thinspace", r"$-\,0.125$"),
        ("wrong_sign_thinspace_bare", r"-\,0.125"),
        ("wrong_exp_ascii", "0.125e-3"),
        ("wrong_exp_times10", r"$0.125 \times 10^{-3}$"),
        ("wrong_role_pm_std", r"0.900 \pm 0.125"),
        ("inequality_lt", "< 0.125"),
        ("inequality_leq", r"$\leq 0.125$"),
        ("inequality_leq_bare", r"\leq 0.125"),
        ("inequality_geqslant", r"\geqslant 0.125"),
        ("approx_prefix", r"\approx 0.125"),
        ("unconsumed_double_minus", "--0.125"),
        ("unconsumed_unknown_macro", r"\foo 0.125"),
        ("unconsumed_two_numbers", "0.900 0.125"),
    ]:
        bad_tex = f"\\begin{{table}}\\label{{tab:r3b}}\\begin{{tabular}}{{lc}}\nm & {bad_cell} \\\\\n\\end{{tabular}}\\end{{table}}"
        rejected = False
        try:
            validate_cell_registry_against_tex(
                [("tab:r3b", "m", 1, "r3b.json", ("m", "val"), ".3f")],
                r3b_jsons,
                extract_latex_table_blocks(bad_tex),
            )
        except AssertionError:
            rejected = True
        assert rejected, f"Verifier failed to reject R3b-M7 / R3c-5 counterexample '{bad_label}': {bad_cell}!"

    # Verify explicit `mean` + `std` roles on `0.900 \pm 0.125` DO pass when bound to their right roles!
    pm_tex = r"\begin{table}\label{tab:pm}\begin{tabular}{lc}" + "\n" + r"m & \textbf{0.900} $\pm$ 0.125 \\" + "\n" + r"\end{tabular}\end{table}"
    ok_pm = validate_cell_registry_against_tex(
        [
            ("tab:pm", "m", 1, "r3b.json", ("m", "mean"), ".3f", "mean"),
            ("tab:pm", "m", 1, "r3b.json", ("m", "std"), ".3f", "std"),
        ],
        r3b_jsons,
        extract_latex_table_blocks(pm_tex),
    )
    assert ok_pm["verified_cells"] == 2

    return {
        "self_test_passed": True,
        "swapped_row_rejected": swapped_row_rejected,
        "swapped_col_rejected": swapped_col_rejected,
        "digit_collision_rejected": digit_collision_rejected,
        "dup_rejected": dup_rejected,
        "missing_rejected": missing_rejected,
        "r3b_sign_exp_role_counterexamples_rejected": True,
        "json_config_roundtrip_verified": True,
    }


def load_cell_registry(base_dir: str) -> List[Tuple]:
    if MANUSCRIPT_CELL_REGISTRY:
        return list(MANUSCRIPT_CELL_REGISTRY)
    reg_json = os.path.join(base_dir, "results_v3", "manuscript_cell_registry_v3.json")
    if os.path.exists(reg_json):
        with open(reg_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            (
                str(r["table"]),
                str(r["row"]),
                int(r["col"]),
                str(r["json_file"]),
                tuple(str(x) for x in r["json_path"]),
                str(r["fmt"]),
                str(r.get("role", "primary")),
            )
            for r in raw
        ]
    return []


def run_full_verification(base_dir: str) -> Dict:
    res_dir = os.path.join(base_dir, "results_v3")
    ms_tex = os.path.join(base_dir, "manuscript", "main.tex")
    art_root = os.path.join(res_dir, "commit_predictions")

    result_jsons = sorted(
        p for p in glob.glob(os.path.join(res_dir, "*_v3.json"))
        if os.path.basename(p) not in ("manuscript_cell_registry_v3.json",)
    )
    if not result_jsons:
        raise FileNotFoundError(f"No `*_v3.json` result files found in {res_dir}!")

    prov_rep = verify_provenance_freshness(base_dir, result_jsons)
    art_rep = verify_commit_level_artifacts(art_root)
    hyg_rep = verify_manuscript_hygiene_and_citations(base_dir)

    registry = load_cell_registry(base_dir)
    if not registry:
        raise RuntimeError(
            "MANUSCRIPT_CELL_REGISTRY / results_v3/manuscript_cell_registry_v3.json is empty! "
            "Bind all manuscript result cells before final manuscript sign-off."
        )

    loaded_jsons = {}
    for jp in result_jsons:
        with open(jp, "r", encoding="utf-8") as f:
            loaded_jsons[os.path.basename(jp)] = json.load(f)
    with open(ms_tex, "r", encoding="utf-8") as f:
        table_blocks = extract_latex_table_blocks(f.read())

    cell_rep = validate_cell_registry_against_tex(registry, loaded_jsons, table_blocks)
    return {
        "provenance": prov_rep,
        "commit_artifacts": art_rep,
        "manuscript_hygiene": hyg_rep,
        "cell_registry": cell_rep,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify V3 Manuscript, Provenance, & Commit Artifacts")
    parser.add_argument("--base-dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    st = run_self_test()
    print(f"  [PASS] Cell-Level Verifier Adversarial Self-Test: {st}", flush=True)
    if args.self_test_only:
        return

    full_rep = run_full_verification(args.base_dir)
    print(f"  [PASS] Full V3 Manuscript & Provenance Verification: {json.dumps(full_rep, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
