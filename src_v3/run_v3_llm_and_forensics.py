"""
Audited Live Code-LLM Experiment (`RQ4f`) & Pre-Declared Natural Prequential Commit Forensics (`RQ5`)
for N05 V3 (Software Quality Journal).

Closes Codex Round 2 & Round 3 Findings (`R3-M3`, `R3-M4`):
1. `RQ4f` Live API Audit (`R3-M3`):
   - Reads API key strictly in memory from `.llm_api/.deepseek.key` (zero credential exposure).
   - Saves FULL raw request & raw response payloads for EVERY attempt (including failed retries,
     HTTP status, token usage, and `served_model` recording `"MISSING_MODEL_FIELD"` if absent) to
     `logs/llm_raw_exchange_v3.jsonl.gz` (never storing the API key).
   - Enforces strict JSON schema `{"defect_prob": float in [0.0, 1.0]}` via `parse_strict_llm_json_prob`
     (ZERO regex float scraping from non-compliant text; out-of-range values are logged as protocol
     errors and fall back to pre-declared `0.50`).
   - Separates empty-diff commits (`coverage = None`, `empty_diff_commits_count`) from non-empty diff coverage.
   - Uses EXACT PAIRED few-shot prompts (`FewShot_Censored_1W` vs `FewShot_Oracle_Paired_1W`) with
     the EXACT SAME 6 example commits in the EXACT SAME order and identical prompt template, differing
     SOLELY in `y_obs(t)` vs `y_true` so label poisoning is causally isolated!
   - Evaluates the FULL host & stream comparator set (`LOC_Only`, `LApredict_B0`, `Churn_LR_B0`,
     `Full_LR_Adapt`, `BORB_Native`, `ORB_OHT_Native`, `ODaSC_OHT_Native`, `B0_Frozen`, `B1b_4W_Oracle`,
     `B2_4W`, `Wait_90`, `Pure_nnPU`, `Teacher_Only_Exact`, `Teacher_Only_Matched`, `Survival_PU`,
     and `Hybrid_Survival_PU_plus_ZeroShot_LLM`) on the exact frozen evaluation slice.
2. `RQ5` Out-of-Sample Natural Commit Forensics (`R3-M4`):
   - Reads ALL thresholds and tier filters exclusively from `V3_CONFIG.forensic_rules` (`Tier1_McIntosh_CSV`
     only, `B0 >= 0.45, B2 < 0.35, SPU >= 0.50`, `FP >= 0.70`, ZERO fallback threshold loosening).
   - Reports total matching candidate counts (`n_candidates_matched`) and attaches the exact prior
     alarm & supervision censoring exposure trajectory to every extracted commit.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_v3 import PURE_COMMIT_COLS_QT_OS, SECONDS_PER_DAY, V3_CONFIG
from data_audit_v3 import sha256_file
from metrics_and_stats_v3 import evaluate_horizon_v3, prob_to_logit_safe
from models_and_baselines_v3 import (
    execute_within_host_counterfactual_suite_v3,
    observe_labels_at_time,
    run_borb_competitor_v3,
    run_linear_competitor_v3,
    run_odasc_oht_native_v3,
    run_orb_oht_native_v3,
)
from run_v3_experiments import build_provenance_header, save_json_atomic
from survival_estimators_v3 import CohortIncidenceEstimator


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def compact_unified_diff_and_coverage(raw_diff: str, max_chars: int = 1800) -> Tuple[str, Optional[float], bool]:
    """
    Deterministic diff compaction (`R3-M3`):
      - Empty diffs return `("(empty diff)", None, True)` — NEVER counted as `1.0` coverage!
      - Non-empty diffs with `0` `+`/`-` lines (e.g., pure rename) return `(trunc, None, False)`.
      - Otherwise returns `(trunc, retained_changed_lines / total_changed_lines, False)`.
    """
    if not raw_diff or not raw_diff.strip():
        return "(empty diff)", None, True

    kept_lines: List[str] = []
    total_changed_lines = 0
    for ln in raw_diff.splitlines():
        if ln.startswith(("index ", "old mode", "new mode", "similarity index", "--- a/", "+++ b/")):
            continue
        is_chg = (ln.startswith("+") or ln.startswith("-")) and (not ln.startswith(("+++", "---")))
        if is_chg:
            total_changed_lines += 1
        kept_lines.append(ln)

    joined = "\n".join(kept_lines)
    trunc = joined[:max_chars]
    if total_changed_lines == 0:
        return trunc, None, False

    retained_chg = sum(
        1 for ln in trunc.splitlines()
        if (ln.startswith("+") or ln.startswith("-")) and (not ln.startswith(("+++", "---")))
    )
    cov = float(np.clip(retained_chg / float(total_changed_lines), 0.0, 1.0))
    return trunc, cov, False


def parse_strict_llm_json_prob(content: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Strict JSON Schema Validator (`R3-M3`):
      - Requires valid JSON object with exact key `"defect_prob"` (no extra keys, no regex scraping!).
      - Requires numeric (`int` or `float`, not `bool`), finite value in `[0.0, 1.0]`.
      - Out-of-range values return `(None, "out_of_range_defect_prob")` instead of silent clipping!
    """
    if not isinstance(content, str) or not content.strip():
        return None, "empty_content"
    txt = content.strip()
    try:
        obj = json.loads(txt)
    except Exception:
        return None, "invalid_json"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    if set(obj.keys()) != {"defect_prob"}:
        return None, "unexpected_json_keys"
    raw_v = obj["defect_prob"]
    if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
        return None, "non_numeric_defect_prob"
    val = float(raw_v)
    if not math.isfinite(val):
        return None, "non_finite_defect_prob"
    if val < 0.0 or val > 1.0:
        return None, "out_of_range_defect_prob"
    return val, None


def extract_served_model_strict(parsed_resp: Dict) -> str:
    """Never substitute requested `model_alias` when `response['model']` is absent (`R3-M3`)!"""
    m_val = parsed_resp.get("model")
    if isinstance(m_val, str) and m_val.strip():
        return m_val.strip()
    return "MISSING_MODEL_FIELD"


def _load_api_key_in_memory(base_dir: str) -> str:
    ws_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    key_path = os.path.join(ws_root, ".llm_api", ".deepseek.key")
    if not os.path.exists(key_path):
        raise FileNotFoundError("DeepSeek API key file not found in workspace `.llm_api/.deepseek.key`.")
    with open(key_path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("Empty DeepSeek API key.")
    return key


def _default_urllib_transport(
    url: str,
    data_bytes: bytes,
    headers: Dict[str, str],
    timeout_sec: float,
) -> Tuple[int, bytes, Optional[str]]:
    """
    Pure HTTP transport layer (`R3b-M5`):
      - Returns `(status_code, full_raw_bytes, transport_or_http_error)` BEFORE any JSON parsing;
      - Never truncates HTTP error bodies (`hexc.read()` is captured in full);
      - Distinguishes network/transport exceptions (`status_code == 0`) from HTTP status errors (`status_code > 0`).
    """
    try:
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()
            status_code = int(resp.status)
        if 200 <= status_code < 300:
            return status_code, raw_bytes, None
        return status_code, raw_bytes, f"http_status_{status_code}"
    except urllib.error.HTTPError as hexc:
        try:
            err_bytes = hexc.read()
        except Exception:
            err_bytes = b""
        return int(hexc.code), err_bytes, f"http_status_{int(hexc.code)}"
    except Exception as exc:
        return 0, b"", f"transport_error:{type(exc).__name__}"


def _parse_usage_metadata_safe(raw_usage: Any) -> Tuple[int, int, int, bool, List[str], Any]:
    """
    Lossless, Never-Throwing Token Metadata Validator (`R3c-4`):
      - Retains `raw_usage` verbatim (JSON-safe) and NEVER raises `TypeError` or `ValueError`
        when `usage` or `prompt_tokens`/`completion_tokens`/`total_tokens` is `None`, `"unknown"`,
        `bool`, `float`, negative, `list`, or `dict`!
      - Returns `(p_tok, c_tok, t_tok, usage_valid, metadata_errors, safe_raw_usage)`.
    """
    meta_errs: List[str] = []
    try:
        json.dumps(raw_usage)
        safe_raw = raw_usage
    except Exception:
        safe_raw = repr(raw_usage)

    if raw_usage is None:
        return 0, 0, 0, False, ["missing_usage_field"], safe_raw
    if not isinstance(raw_usage, dict):
        return 0, 0, 0, False, [f"usage_not_object:{type(raw_usage).__name__}"], safe_raw

    def _chk_int(field_name: str, val: Any, required: bool = True) -> Optional[int]:
        if val is None:
            if required:
                meta_errs.append(f"invalid_{field_name}:NoneType")
            return None
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            meta_errs.append(f"invalid_{field_name}:{type(val).__name__}")
            return None
        return int(val)

    p_opt = _chk_int("prompt_tokens", raw_usage.get("prompt_tokens"), required=True)
    c_opt = _chk_int("completion_tokens", raw_usage.get("completion_tokens"), required=True)
    p_tok = p_opt if p_opt is not None else 0
    c_tok = c_opt if c_opt is not None else 0

    if "total_tokens" in raw_usage:
        t_opt = _chk_int("total_tokens", raw_usage.get("total_tokens"), required=True)
        t_tok = t_opt if t_opt is not None else (p_tok + c_tok)
    else:
        t_tok = p_tok + c_tok

    return p_tok, c_tok, t_tok, (len(meta_errs) == 0), meta_errs, safe_raw


class IncrementalLLMArchiveWriter:
    """
    Thread-safe incremental streaming writer (`R3c-4`) that persists and flushes every completed
    LLM audit & raw exchange record immediately upon completion so a later failure can NEVER erase
    previously completed exchanges!
    """

    def __init__(self, audit_jsonl_path: str, raw_gz_path: str):
        os.makedirs(os.path.dirname(audit_jsonl_path), exist_ok=True)
        os.makedirs(os.path.dirname(raw_gz_path), exist_ok=True)
        self.audit_jsonl_path = audit_jsonl_path
        self.raw_gz_path = raw_gz_path
        self._lock = threading.Lock()
        # Use explicit LF (`newline="\n"`) on JSONL and binary mode (`"wb"`) on gzip (`R3d-1`) so
        # Windows NEVER translates `\n` -> `\r\n` and `uncompressed_sha256 == sha256(gzip.decompress(gz_bytes))`!
        self._audit_fp = open(audit_jsonl_path, "w", encoding="utf-8", newline="\n")
        self._raw_gz_fp = gzip.open(raw_gz_path, "wb")
        self._uncompressed_raw_hasher = hashlib.sha256()
        self.records_written = 0

    def append_record(self, audit_rec: Dict, raw_exchange_rec: Dict) -> None:
        audit_line = json.dumps(audit_rec, sort_keys=True) + "\n"
        raw_bytes = (json.dumps(raw_exchange_rec, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            self._audit_fp.write(audit_line)
            self._audit_fp.flush()
            self._raw_gz_fp.write(raw_bytes)
            self._raw_gz_fp.flush()
            self._uncompressed_raw_hasher.update(raw_bytes)
            self.records_written += 1

    def close_and_finalize_hashes(self) -> Tuple[str, str, str]:
        with self._lock:
            self._audit_fp.close()
            self._raw_gz_fp.close()
            uncomp_sha = self._uncompressed_raw_hasher.hexdigest()
        return sha256_file(self.audit_jsonl_path), uncomp_sha, sha256_file(self.raw_gz_path)


def _call_deepseek_json_prob(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model_alias: str = "deepseek-chat",
    timeout_sec: float = 45.0,
    max_retries: int = 2,
    transport_fn: Optional[Callable[[str, bytes, Dict[str, str], float], Tuple[int, bytes, Optional[str]]]] = None,
    sleep_sec_base: float = 0.8,
) -> Dict:
    """
    3-Stage Byte-Lossless & Metadata-Safe Audited LLM Exchange (`R3-M3`, `R3b-M5`, `R3c-4`):
      1. Captures `status_code`, exact `raw_response_b64` (base64 of raw bytes so even non-UTF-8
         binary `b'\\xff'` reconstructs `resp_sha256` bit-for-bit!), `raw_response_utf8_valid`,
         `raw_response_text`, and `resp_sha256` BEFORE any JSON or metadata parsing.
      2. Validates optional `usage` token metadata via `_parse_usage_metadata_safe` (zero uncaught
         `TypeError`/`ValueError` on `prompt_tokens=None` / `"unknown"` / `[1]`), retaining `raw_usage`
         and explicit `metadata_errors`.
      3. Explicitly separates failure stages into `"transport"`, `"http_status"`, `"envelope"`, and
         `"content_schema"`, recording BOTH `last_attempt_latency_ms` and `total_latency_ms`.
    """
    url = "https://api.deepseek.com/chat/completions"
    req_body = {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }
    req_json = json.dumps(req_body, ensure_ascii=False, sort_keys=True)
    req_bytes = req_json.encode("utf-8")
    req_hash = hashlib.sha256(req_bytes).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    active_transport = transport_fn if transport_fn is not None else _default_urllib_transport

    attempts_trace: List[Dict] = []
    last_err = "unknown_error"
    last_stage = "unknown"

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        ts_att = datetime.now(timezone.utc).isoformat()
        try:
            status_code_raw, raw_bytes_in, tp_err = active_transport(url, req_bytes, headers, timeout_sec)
            status_code = int(status_code_raw)
            raw_bytes = bytes(raw_bytes_in) if raw_bytes_in is not None else b""
        except Exception as tp_exc:
            status_code = 0
            raw_bytes = b""
            tp_err = f"transport_error:{type(tp_exc).__name__}"
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Save lossless base64 bytes + strict UTF-8 validity + exact byte SHA-256 BEFORE any parsing (`R3c-4`)!
        has_payload = bool(status_code > 0 or len(raw_bytes) > 0)
        raw_resp_b64 = base64.b64encode(raw_bytes).decode("ascii")
        try:
            raw_resp_text = raw_bytes.decode("utf-8", errors="strict")
            utf8_valid = True
        except UnicodeDecodeError:
            raw_resp_text = raw_bytes.decode("utf-8", errors="replace")
            utf8_valid = False
        resp_hash = hashlib.sha256(raw_bytes).hexdigest() if has_payload else None

        # Stage 1: Transport or HTTP non-2xx error
        if tp_err is not None or not (200 <= status_code < 300):
            err_stage = "transport" if status_code == 0 else "http_status"
            err_detail = tp_err or f"http_status_{status_code}"
            served_m = "TRANSPORT_ERROR" if status_code == 0 else f"HTTP_{status_code}"
            attempts_trace.append({
                "attempt_idx": attempt,
                "ts_utc": ts_att,
                "status_code": status_code,
                "has_http_payload": has_payload,
                "attempt_latency_ms": lat_ms,
                "served_model": served_m,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_valid": False,
                "raw_usage": None,
                "metadata_errors": [],
                "error_stage": err_stage,
                "error_detail": err_detail,
                "resp_sha256": resp_hash,
                "raw_response_utf8_valid": utf8_valid,
                "raw_response_b64": raw_resp_b64,
                "raw_response_text": raw_resp_text,
            })
            last_stage = err_stage
            last_err = err_detail
            if attempt < max_retries and sleep_sec_base > 0.0:
                time.sleep(sleep_sec_base * (attempt + 1))
            continue

        # Stage 2: HTTP 2xx UTF-8 & JSON Envelope Validation
        if not utf8_valid:
            err_stage = "envelope"
            err_detail = "envelope_invalid_utf8"
            attempts_trace.append({
                "attempt_idx": attempt,
                "ts_utc": ts_att,
                "status_code": status_code,
                "has_http_payload": True,
                "attempt_latency_ms": lat_ms,
                "served_model": "ENVELOPE_UTF8_ERROR",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_valid": False,
                "raw_usage": None,
                "metadata_errors": [],
                "error_stage": err_stage,
                "error_detail": err_detail,
                "resp_sha256": resp_hash,
                "raw_response_utf8_valid": False,
                "raw_response_b64": raw_resp_b64,
                "raw_response_text": raw_resp_text,
            })
            last_stage = err_stage
            last_err = err_detail
            if attempt < max_retries and sleep_sec_base > 0.0:
                time.sleep(sleep_sec_base * (attempt + 1))
            continue

        try:
            parsed = json.loads(raw_resp_text)
        except Exception as jexc:
            err_stage = "envelope"
            err_detail = f"envelope_invalid_json:{type(jexc).__name__}"
            attempts_trace.append({
                "attempt_idx": attempt,
                "ts_utc": ts_att,
                "status_code": status_code,
                "has_http_payload": True,
                "attempt_latency_ms": lat_ms,
                "served_model": "ENVELOPE_PARSE_ERROR",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_valid": False,
                "raw_usage": None,
                "metadata_errors": [],
                "error_stage": err_stage,
                "error_detail": err_detail,
                "resp_sha256": resp_hash,
                "raw_response_utf8_valid": True,
                "raw_response_b64": raw_resp_b64,
                "raw_response_text": raw_resp_text,
            })
            last_stage = err_stage
            last_err = err_detail
            if attempt < max_retries and sleep_sec_base > 0.0:
                time.sleep(sleep_sec_base * (attempt + 1))
            continue

        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("choices"), list)
            or len(parsed["choices"]) == 0
            or not isinstance(parsed["choices"][0], dict)
            or not isinstance(parsed["choices"][0].get("message"), dict)
            or "content" not in parsed["choices"][0]["message"]
        ):
            err_stage = "envelope"
            err_detail = "envelope_missing_choices_message_content"
            served_m = extract_served_model_strict(parsed) if isinstance(parsed, dict) else "ENVELOPE_NOT_OBJECT"
            raw_u_env = parsed.get("usage") if isinstance(parsed, dict) else None
            _, _, _, u_ok_env, m_errs_env, safe_u_env = _parse_usage_metadata_safe(raw_u_env)
            attempts_trace.append({
                "attempt_idx": attempt,
                "ts_utc": ts_att,
                "status_code": status_code,
                "has_http_payload": True,
                "attempt_latency_ms": lat_ms,
                "served_model": served_m,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_valid": u_ok_env,
                "raw_usage": safe_u_env,
                "metadata_errors": m_errs_env,
                "error_stage": err_stage,
                "error_detail": err_detail,
                "resp_sha256": resp_hash,
                "raw_response_utf8_valid": True,
                "raw_response_b64": raw_resp_b64,
                "raw_response_text": raw_resp_text,
            })
            last_stage = err_stage
            last_err = err_detail
            if attempt < max_retries and sleep_sec_base > 0.0:
                time.sleep(sleep_sec_base * (attempt + 1))
            continue

        # Stage 3: Safe Metadata Validation + Strict Inner Content Schema Validation (`R3c-4`)
        served_model = extract_served_model_strict(parsed)
        meta_errs: List[str] = []
        if served_model == "MISSING_MODEL_FIELD":
            meta_errs.append("missing_or_empty_served_model")

        p_tok, c_tok, t_tok, u_valid, u_errs, safe_raw_usage = _parse_usage_metadata_safe(parsed.get("usage"))
        meta_errs.extend(u_errs)

        raw_content = parsed["choices"][0]["message"].get("content")
        content = raw_content if isinstance(raw_content, str) else ""
        prob_val, schema_err = parse_strict_llm_json_prob(content)
        err_stage_s3 = None if schema_err is None else "content_schema"
        err_detail_s3 = None if schema_err is None else f"content_schema:{schema_err}"

        attempts_trace.append({
            "attempt_idx": attempt,
            "ts_utc": ts_att,
            "status_code": status_code,
            "has_http_payload": True,
            "attempt_latency_ms": lat_ms,
            "served_model": served_model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "usage_valid": u_valid,
            "raw_usage": safe_raw_usage,
            "metadata_errors": meta_errs,
            "error_stage": err_stage_s3,
            "error_detail": err_detail_s3,
            "resp_sha256": resp_hash,
            "raw_response_utf8_valid": True,
            "raw_response_b64": raw_resp_b64,
            "raw_response_text": raw_resp_text,
        })

        if prob_val is not None and schema_err is None:
            tot_lat = round(sum(float(a["attempt_latency_ms"]) for a in attempts_trace), 2)
            all_meta_errs = [e for a in attempts_trace for e in a.get("metadata_errors", [])]
            return {
                "ok": True,
                "prob": float(prob_val),
                "used_fallback": False,
                "error_stage": None,
                "error_kind": None,
                "metadata_errors": all_meta_errs,
                "usage_valid": bool(u_valid),
                "served_model": served_model,
                "status_code": status_code,
                "last_attempt_latency_ms": lat_ms,
                "total_latency_ms": tot_lat,
                "latency_ms": tot_lat,
                "n_attempts": len(attempts_trace),
                "req_sha256": req_hash,
                "resp_sha256": resp_hash,
                "prompt_tokens_total": sum(a["prompt_tokens"] for a in attempts_trace),
                "completion_tokens_total": sum(a["completion_tokens"] for a in attempts_trace),
                "raw_request_json": req_json,
                "attempts": attempts_trace,
                "ts_utc": ts_att,
            }

        last_stage = "content_schema"
        last_err = str(err_detail_s3)
        if attempt < max_retries and sleep_sec_base > 0.0:
            time.sleep(sleep_sec_base * (attempt + 1))

    tot_lat_fb = round(sum(float(a["attempt_latency_ms"]) for a in attempts_trace), 2)
    last_lat_fb = float(attempts_trace[-1]["attempt_latency_ms"]) if attempts_trace else 0.0
    all_meta_errs_fb = [e for a in attempts_trace for e in a.get("metadata_errors", [])]
    return {
        "ok": False,
        "prob": 0.50,
        "used_fallback": True,
        "error_stage": last_stage,
        "error_kind": last_err,
        "metadata_errors": all_meta_errs_fb,
        "usage_valid": False,
        "served_model": "FALLBACK_0.50",
        "status_code": attempts_trace[-1]["status_code"] if attempts_trace else 0,
        "last_attempt_latency_ms": last_lat_fb,
        "total_latency_ms": tot_lat_fb,
        "latency_ms": tot_lat_fb,
        "n_attempts": len(attempts_trace),
        "req_sha256": req_hash,
        "resp_sha256": attempts_trace[-1]["resp_sha256"] if attempts_trace else None,
        "prompt_tokens_total": sum(a["prompt_tokens"] for a in attempts_trace),
        "completion_tokens_total": sum(a["completion_tokens"] for a in attempts_trace),
        "raw_request_json": req_json,
        "attempts": attempts_trace,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }


def _format_commit_card(metrics_row: Dict[str, float], msg: str, diff_txt: str) -> str:
    m_str = ", ".join(f"{k}={metrics_row[k]:.2f}" for k in PURE_COMMIT_COLS_QT_OS)
    return f"Metrics(13 commit-time): [{m_str}]\nMessage: {msg[:260].strip()}\nDiff:\n{diff_txt}"


def build_paired_fewshot_prompts(
    cards_no_lbl: Dict[int, str],
    example_indices: np.ndarray,
    y_obs_hist: np.ndarray,
    y_true_hist: np.ndarray,
    target_block: str,
) -> Tuple[str, str, int]:
    """
    Constructs EXACT PAIRED prompts (`FewShot_Censored_1W` vs `FewShot_Oracle_Paired_1W`) using
    the EXACT SAME `example_indices` in the EXACT SAME order and identical formatting (`R3-M3`).
    Returns `(censored_prompt, oracle_prompt, n_flipped_labels)`.
    """
    cen_blocks: List[str] = []
    ora_blocks: List[str] = []
    n_flipped = 0
    for j_raw in example_indices:
        j = int(j_raw)
        lbl_obs = int(y_obs_hist[j] > 0.5)
        lbl_ora = int(y_true_hist[j])
        if lbl_obs != lbl_ora:
            n_flipped += 1
        cen_blocks.append(f"{cards_no_lbl[j]}\nLabel: {lbl_obs}")
        ora_blocks.append(f"{cards_no_lbl[j]}\nLabel: {lbl_ora}")

    p_cen = "=== RECENT WINDOW EXAMPLES ===\n" + "\n---\n".join(cen_blocks) + f"\n\n{target_block}"
    p_ora = "=== RECENT WINDOW EXAMPLES ===\n" + "\n---\n".join(ora_blocks) + f"\n\n{target_block}"
    return p_cen, p_ora, n_flipped


def run_rq4f_live_llm_v3(base_dir: str, n_eval_per_eco: int = 320, max_workers: int = 16) -> Dict:
    api_key = _load_api_key_in_memory(base_dir)
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    jl_dir = os.path.join(base_dir, "data_sync", "jitline")
    szz_dir = os.path.join(base_dir, "data_sync", "szz_linked")
    out_path = os.path.join(base_dir, "results_v3", "rq4f_llm_audit_v3.json")
    audit_jsonl = os.path.join(base_dir, "logs", "llm_api_audit_v3.jsonl")
    raw_gz_path = os.path.join(base_dir, "logs", "llm_raw_exchange_v3.jsonl.gz")
    os.makedirs(os.path.dirname(audit_jsonl), exist_ok=True)

    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        mm_meta_map = {m["name"]: m for m in json.load(f)}

    sys_prompt = (
        "You are an expert software quality auditor performing Just-In-Time Software Defect Prediction. "
        "Given ONLY commit-authoring-time change metrics, commit message, and unified Git diff (plus any "
        "labeled historical reference commits), estimate the probability in [0.00, 1.00] that this commit "
        "introduces a software defect. Output strictly a single JSON object with exact key \"defect_prob\": "
        "{\"defect_prob\": 0.XX}."
    )

    served_models_counter: Dict[str, int] = {}
    ecosystem_results: Dict[str, Dict] = {}
    archive_writer = IncrementalLLMArchiveWriter(audit_jsonl, raw_gz_path)

    print(f"\n=== [RQ4f] Running Live Audited Code-LLM Experiment ({n_eval_per_eco} commits/ecosystem, workers={max_workers}) ===", flush=True)

    for sys_name, stream_name in [("qt", "qt_natural_25k"), ("openstack", "openstack_natural_13k")]:
        b_svd = np.load(os.path.join(mm_dir, f"{stream_name}_svd.npz"))
        b_cb = np.load(os.path.join(mm_dir, f"{stream_name}_cb.npz"))
        b_meta = np.load(os.path.join(mm_dir, f"{stream_name}_meta.npz"), allow_pickle=True)
        df_m = pd.read_csv(os.path.join(jl_dir, f"{sys_name}_metrics.csv")).drop_duplicates("commit_id").set_index("commit_id")
        with gzip.open(os.path.join(szz_dir, f"real_diffs_{sys_name}.json.gz"), "rt", encoding="utf-8") as f:
            diffs_map = json.load(f)

        X_svd = b_svd["X"]
        X_cb = b_cb["X"]
        X_tab = b_svd["X_tab"]
        X_la = b_svd["X_la"]
        X_ch = b_svd["X_churn"]
        ts = b_svd["commit_ts"]
        y_true = b_svd["y_true"]
        fix_ts = b_svd["fix_ts"]
        eff = b_svd["effort"]
        ni, nv = int(b_svd["n_init"]), int(b_svd["n_val"])
        s_start = ni + nv

        ed_slice = min(s_start + n_eval_per_eco, len(y_true))
        eval_indices = list(range(s_start, ed_slice))
        cids_all = [str(x) for x in b_meta["commit_ids"]]
        msgs_all = [str(x) for x in b_meta["msgs"]]

        need_idx = list(range(max(0, s_start - 2400), ed_slice))
        cards_no_lbl: Dict[int, str] = {}
        nonempty_cov_pcts: List[float] = []
        n_empty_eval_diffs = 0
        for idx_i in need_idx:
            cid = cids_all[idx_i]
            row_s = df_m.loc[cid]
            m_dict = {k: float(pd.to_numeric(row_s[k], errors="coerce") or 0.0) for k in PURE_COMMIT_COLS_QT_OS}
            d_txt, cov, is_emp = compact_unified_diff_and_coverage(diffs_map.get(cid, ""), max_chars=1800)
            if idx_i >= s_start:
                if is_emp:
                    n_empty_eval_diffs += 1
                elif cov is not None:
                    nonempty_cov_pcts.append(cov)
            cards_no_lbl[idx_i] = _format_commit_card(m_dict, msgs_all[idx_i], d_txt)

        tasks_spec: List[Tuple[str, int, int, str]] = []
        paired_flipped_counts: List[int] = []

        for pos_k, idx_i in enumerate(eval_indices):
            t_now = float(ts[idx_i]) - 1.0
            _, y_obs_hist = observe_labels_at_time(ts[:idx_i], y_true[:idx_i], fix_ts[:idx_i], t_now)
            lat_est = CohortIncidenceEstimator().fit(ts[:idx_i], fix_ts[:idx_i], y_obs_hist, t_now)
            c_hist = lat_est.predict_maturity(np.maximum(t_now - ts[:idx_i], 0.0))

            target_block = f"=== TARGET COMMIT TO AUDIT ===\n{cards_no_lbl[idx_i]}\n\nReturn JSON {{\"defect_prob\": 0.XX}}:"
            tasks_spec.append(("ZeroShot_Causal", 0, pos_k, target_block))

            rec_pool = np.arange(max(s_start - 400, idx_i - 400), idx_i)
            mat_pool = np.where((np.arange(idx_i) >= max(0, s_start - 2400)) & ((c_hist >= 0.70) | (y_obs_hist > 0.5)))[0]
            if len(mat_pool) < 16:
                mat_pool = rec_pool

            for p_seed in range(3):
                rng_p = np.random.RandomState(42 + 100 * p_seed + pos_k)
                # 1. Exact Paired Censored vs Oracle 1W Few-Shot (same 6 commits, same order, ONLY labels differ!)
                sel_paired = rng_p.choice(rec_pool, size=min(6, len(rec_pool)), replace=False)
                p_cen, p_ora, n_flip = build_paired_fewshot_prompts(
                    cards_no_lbl, sel_paired, y_obs_hist, y_true[:idx_i], target_block
                )
                paired_flipped_counts.append(n_flip)
                tasks_spec.append(("FewShot_Censored_1W", p_seed, pos_k, p_cen))
                tasks_spec.append(("FewShot_Oracle_Paired_1W", p_seed, pos_k, p_ora))

                # 2. Causal Mature + Confirmed-Positive Few-Shot (strictly `y_obs_hist`!)
                rng_m = np.random.RandomState(5000 + 100 * p_seed + pos_k)
                pos_mat = mat_pool[y_obs_hist[mat_pool] > 0.5]
                neg_mat = mat_pool[y_obs_hist[mat_pool] <= 0.5]
                if len(pos_mat) >= 2 and len(neg_mat) >= 4:
                    sel_mat = np.concatenate([rng_m.choice(pos_mat, size=2, replace=False), rng_m.choice(neg_mat, size=4, replace=False)])
                else:
                    sel_mat = rng_m.choice(mat_pool, size=min(6, len(mat_pool)), replace=False)
                rng_m.shuffle(sel_mat)
                ex_mat = "\n---\n".join(f"{cards_no_lbl[int(j)]}\nLabel: {int(y_obs_hist[int(j)] > 0.5)}" for j in sel_mat)
                tasks_spec.append(("FewShot_Mature_Causal", p_seed, pos_k, f"=== MATURE REPOSITORY EXAMPLES ===\n{ex_mat}\n\n{target_block}"))

        n_ev = len(eval_indices)
        probs_grid: Dict[Tuple[str, int], np.ndarray] = {
            ("ZeroShot_Causal", 0): np.full(n_ev, 0.50, dtype=np.float64),
        }
        for ps in range(3):
            probs_grid[("FewShot_Censored_1W", ps)] = np.full(n_ev, 0.50, dtype=np.float64)
            probs_grid[("FewShot_Oracle_Paired_1W", ps)] = np.full(n_ev, 0.50, dtype=np.float64)
            probs_grid[("FewShot_Mature_Causal", ps)] = np.full(n_ev, 0.50, dtype=np.float64)

        n_fallbacks = 0
        total_prompt_tok = 0
        total_comp_tok = 0
        error_counts: Dict[str, int] = {}
        metadata_error_counts: Dict[str, int] = {}
        t_api0 = time.perf_counter()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {
                    pool.submit(_call_deepseek_json_prob, api_key, sys_prompt, prompt_txt): (reg_k, sd_k, pk, cids_all[eval_indices[pk]], prompt_txt)
                    for (reg_k, sd_k, pk, prompt_txt) in tasks_spec
                }
                for fut in as_completed(futs):
                    reg_k, sd_k, pk, cid_str, prompt_txt_orig = futs[fut]
                    try:
                        res_api = fut.result()
                    except Exception as worker_exc:
                        err_k = f"worker_exception:{type(worker_exc).__name__}"
                        res_api = {
                            "ok": False,
                            "prob": 0.50,
                            "used_fallback": True,
                            "error_stage": "worker_exception",
                            "error_kind": err_k,
                            "metadata_errors": [str(worker_exc)],
                            "usage_valid": False,
                            "served_model": "WORKER_EXCEPTION",
                            "status_code": 0,
                            "last_attempt_latency_ms": 0.0,
                            "total_latency_ms": 0.0,
                            "n_attempts": 0,
                            "req_sha256": sha256_str(prompt_txt_orig),
                            "resp_sha256": None,
                            "prompt_tokens_total": 0,
                            "completion_tokens_total": 0,
                            "raw_request_json": prompt_txt_orig,
                            "attempts": [],
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                        }

                    probs_grid[(reg_k, sd_k)][pk] = res_api["prob"]
                    n_fallbacks += int(res_api["used_fallback"])
                    total_prompt_tok += int(res_api["prompt_tokens_total"])
                    total_comp_tok += int(res_api["completion_tokens_total"])
                    if res_api["error_kind"] is not None:
                        ek = str(res_api["error_kind"])
                        error_counts[ek] = error_counts.get(ek, 0) + 1
                    for me in res_api.get("metadata_errors", []):
                        metadata_error_counts[str(me)] = metadata_error_counts.get(str(me), 0) + 1
                    sm_name = res_api["served_model"]
                    served_models_counter[sm_name] = served_models_counter.get(sm_name, 0) + 1

                    audit_rec = {
                        "ecosystem": sys_name,
                        "commit_id": cid_str,
                        "regime": reg_k,
                        "prompt_seed": sd_k,
                        "served_model": sm_name,
                        "status_code": res_api["status_code"],
                        "n_attempts": res_api["n_attempts"],
                        "used_fallback": res_api["used_fallback"],
                        "error_stage": res_api["error_stage"],
                        "error_kind": res_api["error_kind"],
                        "metadata_errors": res_api.get("metadata_errors", []),
                        "usage_valid": res_api.get("usage_valid", False),
                        "prob": round(float(res_api["prob"]), 6),
                        "prompt_tokens_total": res_api["prompt_tokens_total"],
                        "completion_tokens_total": res_api["completion_tokens_total"],
                        "last_attempt_latency_ms": res_api["last_attempt_latency_ms"],
                        "total_latency_ms": res_api["total_latency_ms"],
                        "req_sha256": res_api["req_sha256"],
                        "resp_sha256": res_api["resp_sha256"],
                        "ts_utc": res_api["ts_utc"],
                    }
                    raw_rec = {
                        **audit_rec,
                        "raw_request_json": res_api["raw_request_json"],
                        "attempts": res_api["attempts"],
                    }
                    # Incremental flush to disk immediately (`R3c-4`)!
                    archive_writer.append_record(audit_rec, raw_rec)
        except Exception:
            archive_writer.close_and_finalize_hashes()
            raise

        y_sl = y_true[s_start:ed_slice]
        eff_sl = eff[s_start:ed_slice]

        def _eval_p(p_vec: np.ndarray) -> Dict[str, float]:
            z_v = prob_to_logit_safe(p_vec)
            return evaluate_horizon_v3(y_sl, z_v, p_vec, p_vec, eff_sl, batch_window_size=160)

        zs_ev = _eval_p(probs_grid[("ZeroShot_Causal", 0)])
        cen_evs = [_eval_p(probs_grid[("FewShot_Censored_1W", ps)]) for ps in range(3)]
        ora_evs = [_eval_p(probs_grid[("FewShot_Oracle_Paired_1W", ps)]) for ps in range(3)]
        mat_evs = [_eval_p(probs_grid[("FewShot_Mature_Causal", ps)]) for ps in range(3)]

        # Evaluate FULL Host & Stream Comparator Set on the exact frozen slice `[s_start : ed_slice]` (`R3-M3`)!
        sub_end = ed_slice
        host_methods = [
            "B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Pure_nnPU",
            "FixedCurve_PU", "Teacher_Only_Exact", "Teacher_Only_Matched", "Survival_PU",
        ]
        r_lgb = execute_within_host_counterfactual_suite_v3(
            X_svd[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], ni, nv,
            host_type="lightgbm", methods=host_methods, window_size=160, seed=42,
        )
        r_lgb_hyb = execute_within_host_counterfactual_suite_v3(
            X_svd[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], ni, nv,
            host_type="lightgbm", methods=["Survival_PU"], window_size=160,
            llm_commit_probs_stream=probs_grid[("ZeroShot_Causal", 0)], seed=42,
        )
        r_cb = execute_within_host_counterfactual_suite_v3(
            X_cb[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], ni, nv,
            host_type="codebert", meta=mm_meta_map[stream_name], skip_mode="frozen_lr",
            methods=["B0_Frozen", "B2_4W", "Survival_PU"], window_size=160, seed=42,
        )
        r_cb_hyb = execute_within_host_counterfactual_suite_v3(
            X_cb[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], ni, nv,
            host_type="codebert", meta=mm_meta_map[stream_name], skip_mode="frozen_lr",
            methods=["Survival_PU"], window_size=160,
            llm_commit_probs_stream=probs_grid[("ZeroShot_Causal", 0)], seed=42,
        )

        aw_sl = [(w["start"], w["end"], w["fired"]) for w in r_lgb["shared_alarm_windows"]]
        t2_comps = {
            "LOC_Only": run_linear_competitor_v3(X_la[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], eff[:sub_end], s_start, aw_sl, mode="loc_only", window_size=160, seed=42),
            "LApredict_B0": run_linear_competitor_v3(X_la[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], eff[:sub_end], s_start, aw_sl, mode="frozen", window_size=160, seed=42),
            "Churn_LR_B0": run_linear_competitor_v3(X_ch[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], eff[:sub_end], s_start, aw_sl, mode="frozen", window_size=160, seed=42),
            "Full_LR_Adapt": run_linear_competitor_v3(X_tab[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], eff[:sub_end], s_start, aw_sl, mode="adapt_4w", window_size=160, seed=42),
            "BORB_Native": run_borb_competitor_v3(X_tab[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], s_start, update_mode="native", window_size_cal=160, seed=42),
            "ORB_OHT_Native": run_orb_oht_native_v3(X_tab[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], s_start, window_size_cal=160, seed=42),
            "ODaSC_OHT_Native": run_odasc_oht_native_v3(X_tab[:sub_end], ts[:sub_end], y_true[:sub_end], fix_ts[:sub_end], s_start, window_size_cal=160, seed=42),
        }

        def _eval_mo(mo: Dict) -> Dict[str, float]:
            ev = evaluate_horizon_v3(
                y_sl, mo["raw_logits"], mo["raw_probs"], mo["cal_probs"], eff_sl,
                per_commit_thresholds=mo["tuned_th_vec"], per_commit_decisions=mo["decisions_th"], batch_window_size=160,
            )
            return {"ap": ev["ap"], "f1": ev["f1"], "f1_th": ev["f1_th"], "roc_auc": ev["roc_auc"], "recall_at_20pct_effort": ev["recall_at_20pct_effort"]}

        regimes_out = {
            "LLM_ZeroShot_Causal": {"ap": zs_ev["ap"], "f1": zs_ev["f1"], "roc_auc": zs_ev["roc_auc"], "recall_at_20pct_effort": zs_ev["recall_at_20pct_effort"]},
            "LLM_FewShot_Censored_1W": {
                "ap_mean": float(np.mean([e["ap"] for e in cen_evs])),
                "ap_prompt_seed_sd": float(np.std([e["ap"] for e in cen_evs])),
                "f1_mean": float(np.mean([e["f1"] for e in cen_evs])),
                "roc_auc_mean": float(np.mean([e["roc_auc"] for e in cen_evs])),
            },
            "LLM_FewShot_Oracle_Paired_1W": {
                "ap_mean": float(np.mean([e["ap"] for e in ora_evs])),
                "ap_prompt_seed_sd": float(np.std([e["ap"] for e in ora_evs])),
                "f1_mean": float(np.mean([e["f1"] for e in ora_evs])),
                "roc_auc_mean": float(np.mean([e["roc_auc"] for e in ora_evs])),
            },
            "LLM_FewShot_Mature_Causal": {
                "ap_mean": float(np.mean([e["ap"] for e in mat_evs])),
                "ap_prompt_seed_sd": float(np.std([e["ap"] for e in mat_evs])),
                "f1_mean": float(np.mean([e["f1"] for e in mat_evs])),
                "roc_auc_mean": float(np.mean([e["roc_auc"] for e in mat_evs])),
            },
        }
        for hk in host_methods:
            regimes_out[f"LGB_{hk}"] = _eval_mo(r_lgb["methods"][hk])
        regimes_out["LGB_Hybrid_SPU_plus_ZeroShot_LLM"] = _eval_mo(r_lgb_hyb["methods"]["Survival_PU"])
        for hk in ("B0_Frozen", "B2_4W", "Survival_PU"):
            regimes_out[f"CodeBERT_{hk}"] = _eval_mo(r_cb["methods"][hk])
        regimes_out["CodeBERT_Hybrid_SPU_plus_ZeroShot_LLM"] = _eval_mo(r_cb_hyb["methods"]["Survival_PU"])
        for t2k, t2o in t2_comps.items():
            regimes_out[f"Stream_{t2k}"] = _eval_mo(t2o)

        ecosystem_results[sys_name] = {
            "n_commits_evaluated": n_ev,
            "empty_diff_commits_count": n_empty_eval_diffs,
            "empty_diff_rate": float(n_empty_eval_diffs / max(n_ev, 1)),
            "mean_nonempty_diff_changed_line_coverage": float(np.mean(nonempty_cov_pcts)) if nonempty_cov_pcts else None,
            "p25_nonempty_diff_changed_line_coverage": float(np.percentile(nonempty_cov_pcts, 25)) if nonempty_cov_pcts else None,
            "paired_fewshot_mean_censored_flips_per_6examples": float(np.mean(paired_flipped_counts)) if paired_flipped_counts else 0.0,
            "total_api_calls": len(tasks_spec),
            "fallback_count": n_fallbacks,
            "protocol_and_network_error_counts": error_counts,
            "metadata_error_counts": metadata_error_counts,
            "valid_response_rate": float(1.0 - n_fallbacks / max(len(tasks_spec), 1)),
            "prompt_tokens_total": total_prompt_tok,
            "completion_tokens_total": total_comp_tok,
            "elapsed_sec": round(time.perf_counter() - t_api0, 2),
            "regimes": regimes_out,
        }
        cov_s = f"{ecosystem_results[sys_name]['mean_nonempty_diff_changed_line_coverage']:.1%}" if nonempty_cov_pcts else "N/A"
        print(
            f"  [RQ4f {sys_name:9s}] Valid={ecosystem_results[sys_name]['valid_response_rate']:.2%}, "
            f"NonEmptyCov={cov_s}, EmptyDiffs={n_empty_eval_diffs} | "
            f"AP: ZeroShot={zs_ev['ap']:.4f}, CensFS={regimes_out['LLM_FewShot_Censored_1W']['ap_mean']:.4f}, "
            f"PairedOraFS={regimes_out['LLM_FewShot_Oracle_Paired_1W']['ap_mean']:.4f}, "
            f"MatureFS={regimes_out['LLM_FewShot_Mature_Causal']['ap_mean']:.4f}, "
            f"LGB_SPU={regimes_out['LGB_Survival_PU']['ap']:.4f}, CB_SPU={regimes_out['CodeBERT_Survival_PU']['ap']:.4f}",
            flush=True,
        )

    audit_file_sha, raw_uncomp_sha, raw_gz_file_sha = archive_writer.close_and_finalize_hashes()

    payload = {
        "provenance": build_provenance_header(base_dir),
        "served_model_counts": served_models_counter,
        "audit_log_jsonl_sha256": audit_file_sha,
        "raw_exchange_uncompressed_sha256": raw_uncomp_sha,
        "raw_exchange_gzip_file_sha256": raw_gz_file_sha,
        "audit_log_total_records": archive_writer.records_written,
        "caveat": (
            "Served model strings reflect live API routing (`MISSING_MODEL_FIELD` if omitted); "
            "`FewShot_Censored_1W` and `FewShot_Oracle_Paired_1W` share identical examples and order, "
            "differing exclusively in `y_obs(t)` vs `y_true` labels. Modern pretrained LLMs may have "
            "seen public Git histories during pretraining, so RQ4f evaluates in-context label-poisoning "
            "sensitivity and hybrid prior utility rather than zero-knowledge historical forecasting."
        ),
        "ecosystems": ecosystem_results,
    }
    save_json_atomic(out_path, payload)
    print(f"==> Saved Audited RQ4f Results -> {out_path} (Raw Exchange Archive: {raw_gz_path})", flush=True)
    return payload


# ==============================================================================
# RQ5: Out-of-Sample Natural Commit Forensics (`V3_CONFIG.forensic_rules`, `R3-M4`)
# ==============================================================================

def _build_exposure_trace_for_commit(
    eval_offset: int,
    stream_start: int,
    window_size: int,
    shared_alarm_windows: List[Dict],
    commit_ts: np.ndarray,
    y_true: np.ndarray,
    fix_ts: np.ndarray,
) -> Dict:
    """
    Computes the exact causal alarm & right-censored supervision exposure history active when
    predicting commit `abs_idx = stream_start + eval_offset` (`R3-M4`).
    """
    abs_idx = stream_start + int(eval_offset)
    prior_fired = [w for w in shared_alarm_windows if bool(w["fired"]) and int(w["end"]) <= abs_idx]
    if not prior_fired:
        return {
            "active_model_generation_id": 0,
            "n_prior_alarms_before_prediction": 0,
            "last_alarm_end_commit_idx": None,
            "last_alarm_date_utc": None,
            "last_alarm_4W_true_bugs": None,
            "last_alarm_4W_revealed_bugs": None,
            "last_alarm_4W_censored_false_negatives": None,
            "last_alarm_4W_censoring_rate": None,
        }

    w_last = prior_fired[-1]
    w_ed = int(w_last["end"])
    t_al = float(commit_ts[w_ed - 1])
    _, y_obs_al = observe_labels_at_time(commit_ts[:w_ed], y_true[:w_ed], fix_ts[:w_ed], t_al)
    sl_4w = slice(max(0, w_ed - 4 * window_size), w_ed)
    n_true_4w = int(np.sum(y_true[sl_4w] == 1))
    n_rev_4w = int(np.sum(y_obs_al[sl_4w] > 0.5))
    n_cen_fn = max(n_true_4w - n_rev_4w, 0)
    return {
        "active_model_generation_id": len(prior_fired),
        "n_prior_alarms_before_prediction": len(prior_fired),
        "last_alarm_end_commit_idx": w_ed,
        "last_alarm_date_utc": datetime.fromtimestamp(t_al, tz=timezone.utc).strftime("%Y-%m-%d"),
        "last_alarm_4W_true_bugs": n_true_4w,
        "last_alarm_4W_revealed_bugs": n_rev_4w,
        "last_alarm_4W_censored_false_negatives": n_cen_fn,
        "last_alarm_4W_censoring_rate": round(float(n_cen_fn / max(n_true_4w, 1)), 4),
    }


def select_forensic_indices_strict(
    y_ev: np.ndarray,
    ts_ev: np.ndarray,
    fts_ev: np.ndarray,
    ftiers_ev: np.ndarray,
    p_b0: np.ndarray,
    p_b2: np.ndarray,
    p_spu: np.ndarray,
    rules=V3_CONFIG.forensic_rules,
) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Single Authoritative Forensic Filter (`R3-M4`):
      - Reads ALL thresholds & allowed SZZ tiers exclusively from `V3_CONFIG.forensic_rules`;
      - ZERO fallback loosening: if 0 candidates match, returns empty array and `n_matched = 0`!
    """
    tier_ok = np.isin(ftiers_ev, list(rules.allowed_tiers))
    fin_fix = np.isfinite(fts_ev)
    lat_ok = fin_fix & ((fts_ev - ts_ev) >= float(rules.min_censoring_latency_days) * SECONDS_PER_DAY)

    rescue_cand = np.where(
        (y_ev == 1)
        & tier_ok
        & lat_ok
        & (p_b0 >= float(rules.rescue_min_b0))
        & (p_b2 < float(rules.rescue_max_b2))
        & (p_spu >= float(rules.rescue_min_spu))
    )[0]
    rescue_ord = rescue_cand[np.argsort(-(p_spu[rescue_cand] - p_b2[rescue_cand]))[: int(rules.max_cases_per_category)]]

    fn_cand = np.where(
        (y_ev == 1)
        & tier_ok
        & fin_fix
        & (p_b2 < float(rules.fn_max_b2))
        & (p_spu < float(rules.fn_max_spu))
    )[0]
    fn_ord = fn_cand[np.argsort(p_spu[fn_cand])[: int(rules.max_cases_per_category)]]

    fp_cand = np.where((y_ev == 0) & (p_spu >= float(rules.fp_min_spu)))[0]
    fp_ord = fp_cand[np.argsort(-p_spu[fp_cand])[: int(rules.max_cases_per_category)]]

    return {
        "rescues": (rescue_ord, int(len(rescue_cand))),
        "false_negatives": (fn_ord, int(len(fn_cand))),
        "false_positives": (fp_ord, int(len(fp_cand))),
    }


def run_rq5_natural_forensics_v3(base_dir: str) -> Dict:
    mm_dir = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    out_path = os.path.join(base_dir, "results_v3", "rq5_natural_forensics_v3.json")
    with open(os.path.join(mm_dir, "manifest_mm_v3.json"), "r", encoding="utf-8") as f:
        meta_map = {m["name"]: m for m in json.load(f)}

    rules = V3_CONFIG.forensic_rules
    print("\n=== [RQ5] Extracting Strict Pre-Declared Natural Commit Forensics (`V3_CONFIG.forensic_rules`, Zero Fallback) ===", flush=True)
    ecosystems_out: Dict[str, Dict] = {}

    for sys_name, stream_name in [("qt", "qt_natural_25k"), ("openstack", "openstack_natural_13k")]:
        b_cb = np.load(os.path.join(mm_dir, f"{stream_name}_cb.npz"))
        b_meta = np.load(os.path.join(mm_dir, f"{stream_name}_meta.npz"), allow_pickle=True)
        ni, nv = int(b_cb["n_init"]), int(b_cb["n_val"])
        s_start = ni + nv

        res = execute_within_host_counterfactual_suite_v3(
            X=b_cb["X"],
            commit_ts=b_cb["commit_ts"],
            y_true=b_cb["y_true"],
            fix_ts=b_cb["fix_ts"],
            n_init=ni,
            n_val=nv,
            host_type="codebert",
            meta=meta_map[stream_name],
            skip_mode="frozen_lr",
            methods=["B0_Frozen", "B1b_4W_Oracle", "B2_4W", "Wait_90", "Teacher_Only_Matched", "Survival_PU"],
            seed=42,
        )
        borb = run_borb_competitor_v3(b_cb["X_tab"], b_cb["commit_ts"], b_cb["y_true"], b_cb["fix_ts"], s_start, update_mode="native", seed=42)
        orb = run_orb_oht_native_v3(b_cb["X_tab"], b_cb["commit_ts"], b_cb["y_true"], b_cb["fix_ts"], s_start, seed=42)

        p_b0 = res["methods"]["B0_Frozen"]["raw_probs"]
        p_b1b = res["methods"]["B1b_4W_Oracle"]["raw_probs"]
        p_b2 = res["methods"]["B2_4W"]["raw_probs"]
        p_w90 = res["methods"]["Wait_90"]["raw_probs"]
        p_tm = res["methods"]["Teacher_Only_Matched"]["raw_probs"]
        p_borb = borb["raw_probs"]
        p_orb = orb["raw_probs"]
        p_spu = res["methods"]["Survival_PU"]["raw_probs"]

        ts_ev = b_cb["commit_ts"][s_start:]
        fts_ev = b_cb["fix_ts"][s_start:]
        y_ev = b_cb["y_true"][s_start:]
        eff_ev = b_cb["effort"][s_start:]

        cids_ev = np.array([str(x) for x in b_meta["commit_ids"][s_start:]])
        repos_ev = [str(x) for x in b_meta["repos"][s_start:]]
        fshas_ev = [str(x) for x in b_meta["fix_shas"][s_start:]]
        ftiers_ev = np.array([str(x) for x in b_meta["fix_tiers"][s_start:]])
        msgs_ev = [str(x) for x in b_meta["msgs"][s_start:]]
        codes_ev = [str(x) for x in b_meta["codes"][s_start:]]

        sel_map = select_forensic_indices_strict(y_ev, ts_ev, fts_ev, ftiers_ev, p_b0, p_b2, p_spu, rules=rules)

        def _pack_case(k: int, case_type: str) -> Dict:
            c_dt = datetime.fromtimestamp(float(ts_ev[k]), tz=timezone.utc).strftime("%Y-%m-%d")
            has_fin_fix = np.isfinite(fts_ev[k])
            f_dt = datetime.fromtimestamp(float(fts_ev[k]), tz=timezone.utc).strftime("%Y-%m-%d") if has_fin_fix else "unrevealed_in_history"
            lat_d = round(float((fts_ev[k] - ts_ev[k]) / SECONDS_PER_DAY), 1) if has_fin_fix else None
            exp_trace = _build_exposure_trace_for_commit(
                k, s_start, V3_CONFIG.window_size_default, res["shared_alarm_windows"],
                b_cb["commit_ts"], b_cb["y_true"], b_cb["fix_ts"],
            )
            return {
                "case_type": case_type,
                "eval_offset": int(k),
                "repo": repos_ev[k],
                "inducing_sha": str(cids_ev[k]),
                "commit_date_utc": c_dt,
                "fix_sha": fshas_ev[k],
                "fix_tier": str(ftiers_ev[k]),
                "fix_date_utc": f_dt,
                "latency_days": lat_d,
                "y_true": int(y_ev[k]),
                "effort_loc": float(eff_ev[k]),
                "alarm_and_censoring_exposure_trace": exp_trace,
                "subject": msgs_ev[k].splitlines()[0][:140] if msgs_ev[k] else "",
                "diff_excerpt": "\n".join(codes_ev[k].splitlines()[:18]),
                "scores": {
                    "B0_Frozen": round(float(p_b0[k]), 4),
                    "B1b_4W_Oracle": round(float(p_b1b[k]), 4),
                    "B2_4W": round(float(p_b2[k]), 4),
                    "Wait_90": round(float(p_w90[k]), 4),
                    "Teacher_Only_Matched": round(float(p_tm[k]), 4),
                    "BORB_Native": round(float(p_borb[k]), 4),
                    "ORB_OHT_Native": round(float(p_orb[k]), 4),
                    "Survival_PU": round(float(p_spu[k]), 4),
                },
            }

        rescue_ord, n_rescue_cand = sel_map["rescues"]
        fn_ord, n_fn_cand = sel_map["false_negatives"]
        fp_ord, n_fp_cand = sel_map["false_positives"]

        rescues = [_pack_case(int(k), "score_divergence_under_censored_alarm") for k in rescue_ord]
        fn_cases = [_pack_case(int(k), "false_negative_miss") for k in fn_ord]
        fp_cases = [_pack_case(int(k), "false_positive_alarm") for k in fp_ord]

        ecosystems_out[sys_name] = {
            "stream_name": stream_name,
            "authoritative_filter_config": json.loads(json.dumps(V3_CONFIG.to_dict()["forensic_rules"])),
            "candidate_counts": {
                "n_rescue_candidates_matched": n_rescue_cand,
                "n_false_negative_candidates_matched": n_fn_cand,
                "n_false_positive_candidates_matched": n_fp_cand,
            },
            "interpretive_note": (
                "Selected deterministically under frozen `V3_CONFIG.forensic_rules` without fallback "
                "loosening (`Tier1_McIntosh_CSV` required for positive defects). Cases illustrate "
                "prequential score divergence following censored alarm updates alongside honest FN/FP "
                "failure modes, rather than standalone single-commit causal proof."
            ),
            "rescues": rescues,
            "false_negative_failures": fn_cases,
            "false_positive_alarms": fp_cases,
        }
        print(
            f"  [RQ5 {sys_name:9s}] Candidates matched: rescue={n_rescue_cand} (kept {len(rescues)}), "
            f"FN={n_fn_cand} (kept {len(fn_cases)}), FP={n_fp_cand} (kept {len(fp_cases)}).",
            flush=True,
        )

    payload = {"provenance": build_provenance_header(base_dir), "ecosystems": ecosystems_out}
    save_json_atomic(out_path, payload)
    print(f"==> Saved Natural Commit Forensics -> {out_path}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run Audited RQ4f Live LLM & RQ5 Natural Forensics")
    parser.add_argument("--base-dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--mode", choices=["all", "rq4f", "rq5"], default="all")
    parser.add_argument("--n-eval-per-eco", type=int, default=320)
    args = parser.parse_args()

    if args.mode in ("all", "rq4f"):
        run_rq4f_live_llm_v3(args.base_dir, n_eval_per_eco=args.n_eval_per_eco)
    if args.mode in ("all", "rq5"):
        run_rq5_natural_forensics_v3(args.base_dir)


if __name__ == "__main__":
    main()
