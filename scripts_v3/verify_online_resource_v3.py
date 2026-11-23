r"""
Companion Cell-Level Verifier for Springer Online Resource 1 (`manuscript/online_resource.tex`).

Reuses the immutable cell parser and semantic numeric/statistical-role validator from
`src_v3/verify_manuscript_v3.py` to verify 100% of the supplementary empirical tables
(`results_v3/online_resource_cell_registry.json`) directly against the 15 provenance-verified
`results_v3/*_v3.json` experimental outputs without modifying `src_v3/*.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V3_DIR = os.path.join(BASE_DIR, "src_v3")
if SRC_V3_DIR not in sys.path:
    sys.path.insert(0, SRC_V3_DIR)

from verify_manuscript_v3 import (
    FORBIDDEN_INTERNAL_CODES,
    extract_latex_table_blocks,
    validate_cell_registry_against_tex,
    verify_provenance_freshness,
)


def load_supp_registry(base_dir: str) -> List[Tuple]:
    reg_json = os.path.join(base_dir, "results_v3", "online_resource_cell_registry.json")
    if not os.path.exists(reg_json):
        raise FileNotFoundError(f"Missing supplementary cell registry: {reg_json}")
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


def verify_online_resource(base_dir: str) -> Dict:
    res_dir = os.path.join(base_dir, "results_v3")
    supp_tex_path = os.path.join(base_dir, "manuscript", "online_resource.tex")
    if not os.path.exists(supp_tex_path):
        raise FileNotFoundError(f"Missing Online Resource LaTeX file: {supp_tex_path}")

    result_jsons = sorted(
        p for p in glob.glob(os.path.join(res_dir, "*_v3.json"))
        if os.path.basename(p) not in ("manuscript_cell_registry_v3.json",)
    )
    if not result_jsons:
        raise FileNotFoundError(f"No `*_v3.json` files found in {res_dir}")

    prov_rep = verify_provenance_freshness(base_dir, result_jsons)
    loaded_jsons = {}
    for jp in result_jsons:
        with open(jp, "r", encoding="utf-8") as f:
            loaded_jsons[os.path.basename(jp)] = json.load(f)

    with open(supp_tex_path, "r", encoding="utf-8") as f:
        supp_tex = f.read()

    for pat in FORBIDDEN_INTERNAL_CODES:
        if re.search(pat, supp_tex):
            raise AssertionError(f"Forbidden internal identifier '{pat}' leaked into online_resource.tex!")

    table_blocks = extract_latex_table_blocks(supp_tex)
    registry = load_supp_registry(base_dir)
    if not registry:
        raise RuntimeError("online_resource_cell_registry.json is empty!")

    cell_rep = validate_cell_registry_against_tex(registry, loaded_jsons, table_blocks)
    return {
        "provenance": prov_rep,
        "online_resource_tables_parsed": len(table_blocks),
        "online_resource_cell_registry": cell_rep,
        "forbidden_codes_found": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Online Resource 1 empirical tables against results_v3 JSONs.")
    parser.add_argument("--base-dir", type=str, default=os.environ.get("CAUSALJIT_BASE_DIR", BASE_DIR))
    args = parser.parse_args()
    rep = verify_online_resource(args.base_dir)
    print(f"  [PASS] Online Resource 1 Verification: {json.dumps(rep, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
