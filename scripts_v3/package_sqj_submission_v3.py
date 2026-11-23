r"""
Clean Springer *Software Quality Journal* (SQJ) Final Submission Packager
(`scripts_v3/package_sqj_submission_v3.py`).

Produces `research/N05/submission_sqj_final/` containing:
  1. `N05_SQJ_Manuscript_Final.pdf` (Camera-ready official Springer `sn-jnl` PDF)
  2. `N05_SQJ_LaTeX_Source.zip` (Standalone Editorial Manager LaTeX + `sn-jnl.cls` + `sn-basic.bst` + vector figures)
  3. `N05_SQJ_Supplementary_Code_and_Results.zip` (Lightweight <100 MB Editorial Manager supplementary package)
  4. `N05_SQJ_Full_Replication_Artifact_Zenodo.zip` (Complete 100%-offline Zenodo replication archive)
  5. `SUBMISSION_CHECKLIST_AND_MANIFEST.md` (SHA-256 manifest & Editorial Manager upload guide)

Zero edits to `src_v3/*.py` or `results_v3/*_v3.json`. Performs strict secret scanning (`sk-...`)
and verifies that `N05_SQJ_LaTeX_Source.zip` compiles cleanly from scratch in an isolated temp folder.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Iterable, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "submission_sqj_final")

SECRET_PAT = re.compile(rb"\bsk-[A-Za-z0-9]{16,}\b")
SKIP_DIR_NAMES = {"__pycache__", ".venv", ".git", ".pytest_cache", "submission_sqj_final", "build_artifacts"}
SKIP_EXTS = {".pyc", ".pyo", ".aux", ".log", ".out", ".blg", ".bbl", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_no_secrets(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".py", ".tex", ".bib", ".md", ".txt", ".json", ".csv", ".cls", ".bst", ".sh", ".ps1"}:
        with open(path, "rb") as f:
            data = f.read()
        if SECRET_PAT.search(data):
            raise RuntimeError(f"Secret pattern detected in {path}! Aborting packaging.")


def collect_files(
    rel_dir: str,
    only_exts: Tuple[str, ...] | None = None,
    only_names: Tuple[str, ...] | None = None,
    name_substr: str | None = None,
) -> List[Tuple[str, str]]:
    abs_root = os.path.join(BASE_DIR, rel_dir)
    items: List[Tuple[str, str]] = []
    for cur, dirs, files in os.walk(abs_root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXTS:
                continue
            if only_exts is not None and ext not in only_exts:
                continue
            if only_names is not None and fn not in only_names:
                continue
            if name_substr is not None and name_substr not in fn:
                continue
            full_p = os.path.join(cur, fn)
            rel_p = os.path.relpath(full_p, BASE_DIR).replace("\\", "/")
            items.append((full_p, rel_p))
    return items


ALREADY_COMPRESSED_EXTS = {".npz", ".gz", ".png", ".zip", ".jpg", ".jpeg"}


def write_zip(zip_path: str, entries: Iterable[Tuple[str, str]], compresslevel: int = 6) -> Tuple[int, int]:
    seen = set()
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for full_p, arc_p in entries:
            arc_norm = arc_p.replace("\\", "/")
            if arc_norm in seen:
                continue
            seen.add(arc_norm)
            check_no_secrets(full_p)
            ext = os.path.splitext(full_p)[1].lower()
            ctype = zipfile.ZIP_STORED if ext in ALREADY_COMPRESSED_EXTS else zipfile.ZIP_DEFLATED
            zf.write(full_p, arc_norm, compress_type=ctype)
            n_files += 1
    return n_files, os.path.getsize(zip_path)


def test_compile_latex_zip(latex_zip: str) -> Tuple[int, int, int, int]:
    with tempfile.TemporaryDirectory(prefix="n05_sqj_compile_test_") as tmp:
        with zipfile.ZipFile(latex_zip, "r") as zf:
            zf.extractall(tmp)
        for cmd in (
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ):
            res = subprocess.run(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Standalone compile failed on {cmd}:\n{res.stdout[-1500:]}")
        with open(os.path.join(tmp, "main.log"), "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
        errs = [l for l in log.splitlines() if l.startswith("!")]
        warns = [l for l in log.splitlines() if "Warning" in l]
        overfulls = [l for l in log.splitlines() if "Overfull" in l]
        pages = re.findall(r"Output written on main\.pdf \((\d+) pages", log)
        return int(pages[-1]) if pages else 0, len(errs), len(warns), len(overfulls)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Camera-ready PDF
    src_pdf = os.path.join(BASE_DIR, "manuscript", "main.pdf")
    dst_pdf = os.path.join(OUT_DIR, "N05_SQJ_Manuscript_Final.pdf")
    shutil.copy2(src_pdf, dst_pdf)
    print(f"[1/4] Copied {dst_pdf} ({os.path.getsize(dst_pdf)/1e6:.2f} MB)", flush=True)

    # 2. Standalone Springer Editorial Manager LaTeX Source Bundle (strictly V3 figures only!)
    latex_entries: List[Tuple[str, str]] = [
        (os.path.join(BASE_DIR, "manuscript", fn), fn)
        for fn in ("main.tex", "references.bib", "sn-jnl.cls", "sn-basic.bst")
    ]
    v3_pdf_figs = collect_files("figures", only_exts=(".pdf",), name_substr="_v3.")
    assert len(v3_pdf_figs) == 10, f"Expected 10 V3 vector figures, got {len(v3_pdf_figs)}"
    for full_p, rel_p in v3_pdf_figs:
        latex_entries.append((full_p, rel_p))

    latex_zip = os.path.join(OUT_DIR, "N05_SQJ_LaTeX_Source.zip")
    n_tex, sz_tex = write_zip(latex_zip, latex_entries)
    print(f"[2/4] Built {latex_zip} ({n_tex} files, {sz_tex/1e6:.2f} MB) — running clean-room compile test...", flush=True)
    pages, n_err, n_warn, n_over = test_compile_latex_zip(latex_zip)
    print(f"      Clean-room compile test PASS: {pages} pages | Errors={n_err} | Warnings={n_warn} | Overfulls={n_over}", flush=True)
    assert n_err == 0 and n_warn == 0 and n_over == 0, (
        f"Standalone LaTeX zip compile not spotless: err={n_err}, warn={n_warn}, over={n_over}"
    )

    # 3. Lightweight Editorial Manager Supplementary Package (<100 MB, strictly V3 artifacts)
    root_prefix = "N05_SQJ_Replication/"
    supp_entries: List[Tuple[str, str]] = []
    for fn in ("README.md", "LICENSE", "requirements_lock_v3.txt"):
        supp_entries.append((os.path.join(BASE_DIR, fn), root_prefix + fn))
    for rel_dir in ("src_v3", "scripts_v3", "results_v3", "manuscript"):
        for full_p, rel_p in collect_files(rel_dir):
            supp_entries.append((full_p, root_prefix + rel_p))
    for full_p, rel_p in collect_files("figures", name_substr="_v3."):
        supp_entries.append((full_p, root_prefix + rel_p))
    for full_p, rel_p in collect_files("logs", only_names=("llm_raw_exchange_v3.jsonl.gz", "llm_api_audit_v3.jsonl")):
        supp_entries.append((full_p, root_prefix + rel_p))
    for rel_dir in ("cache/benchmark_b_v3", "cache/confirmatory_v3"):
        for full_p, rel_p in collect_files(rel_dir):
            supp_entries.append((full_p, root_prefix + rel_p))
    for rel_dir in ("cache/multimodal_v3_pure13", "cache/multimodal_v3_sens", "cache/codebert_v3"):
        for full_p, rel_p in collect_files(rel_dir, only_exts=(".json",)):
            supp_entries.append((full_p, root_prefix + rel_p))
    for full_p, rel_p in collect_files("cache/multimodal_v3_pure13", name_substr="_tab.npz"):
        supp_entries.append((full_p, root_prefix + rel_p))

    supp_zip = os.path.join(OUT_DIR, "N05_SQJ_Supplementary_Code_and_Results.zip")
    n_supp, sz_supp = write_zip(supp_zip, supp_entries, compresslevel=6)
    print(f"[3/4] Built {supp_zip} ({n_supp} files, {sz_supp/1e6:.2f} MB)", flush=True)

    # 4. Complete Offline Zenodo Archival Replication Package (includes multimodal caches + CodeBERT caches + data_sync)
    zenodo_entries = list(supp_entries)
    for rel_dir in ("data_sync", "cache/multimodal_v3_pure13", "cache/multimodal_v3_sens", "cache/codebert_v3"):
        for full_p, rel_p in collect_files(rel_dir):
            zenodo_entries.append((full_p, root_prefix + rel_p))

    zenodo_zip = os.path.join(OUT_DIR, "N05_SQJ_Full_Replication_Artifact_Zenodo.zip")
    n_zen, sz_zen = write_zip(zenodo_zip, zenodo_entries, compresslevel=1)
    print(f"[4/4] Built {zenodo_zip} ({n_zen} files, {sz_zen/1e6:.2f} MB)", flush=True)

    # 5. Write SHA-256 manifest & Editorial Manager checklist
    manifest_rows = []
    for fn in (
        "N05_SQJ_Manuscript_Final.pdf",
        "N05_SQJ_LaTeX_Source.zip",
        "N05_SQJ_Supplementary_Code_and_Results.zip",
        "N05_SQJ_Full_Replication_Artifact_Zenodo.zip",
    ):
        fp = os.path.join(OUT_DIR, fn)
        manifest_rows.append(f"| `{fn}` | {os.path.getsize(fp)/1e6:.2f} MB | `{sha256_file(fp)}` |")

    md = "\n".join([
        "# Software Quality Journal (SQJ) Final Submission Bundle — N05",
        "",
        "- **Title:** *Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction*",
        "- **Template:** Official Springer Nature `sn-jnl.cls` (`[pdflatex,sn-basic]`, author-year citations)",
        "- **Authors:**",
        "  1. Zhijie Li (`lizhijie@mails.neu.edu.cn`) — Software College, Northeastern University, Shenyang, 110169, China",
        "  2. Shaoyang Xu (`xusy3@mails.neu.edu.cn`) — Software College, Northeastern University, Shenyang, 110169, China",
        "  3. Ketong Liu (`liuketong@mails.neu.edu.cn`) — Software College, Northeastern University, Shenyang, 110169, China",
        "  4. Shibo Meng (`mengsb1@mails.neu.edu.cn`) — Software College, Northeastern University, Shenyang, 110169, China",
        "  5. Feiran Gao (`3120260997@bit.edu.cn`) — School of Computer Science and Technology, Beijing Institute of Technology, Beijing, 100081, China",
        "  6. **Yuli Zhao*** (`zhaoyl@swc.neu.edu.cn`, Corresponding Author) — Software College, Northeastern University, Shenyang, 110169, China",
        f"- **PDF Clean-Room Verification:** `{pages}` pages, `0` Errors, `0` Warnings, `0` Overfulls.",
        "",
        "## Submission Artifacts & SHA-256 Checksums",
        "",
        "| File | Size | SHA-256 |",
        "|---|---:|---|",
        *manifest_rows,
        "",
        "## Springer Editorial Manager Upload Guide",
        "",
        "1. **Main Manuscript (PDF):** Upload `N05_SQJ_Manuscript_Final.pdf` (or upload the contents of `N05_SQJ_LaTeX_Source.zip` if submitting TeX source directly).",
        "2. **LaTeX Source Files:** Upload `N05_SQJ_LaTeX_Source.zip` (verified to compile cleanly out of the box via `pdflatex -> bibtex -> pdflatex x3`).",
        "3. **Supplementary Material (Direct Portal Upload):** Upload `N05_SQJ_Supplementary_Code_and_Results.zip` (compact bundle containing all immutable `src_v3/` code, `scripts_v3/` verifiers and companion audits, all 15 provenance-stamped `results_v3/*_v3.json` files, all 50 commit `.npz` prediction archives, lossless LLM exchange logs, and Benchmark B + Confirmatory bundles).",
        "4. **Long-Term Zenodo / Figshare Deposit:** Upload `N05_SQJ_Full_Replication_Artifact_Zenodo.zip` (adds raw `data_sync/`, multimodal `.npz` bundles, and `CodeBERT` embedding caches for 100%-offline full re-execution).",
        "",
    ])
    with open(os.path.join(OUT_DIR, "SUBMISSION_CHECKLIST_AND_MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote SUBMISSION_CHECKLIST_AND_MANIFEST.md successfully.", flush=True)


if __name__ == "__main__":
    main()
