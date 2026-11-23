"""
Strict Temporal & Provenance Audit + Clean V3 Benchmark Cache Builder (N05 V3).

Closes Codex Round 2 Criterion (1) & Criterion (7):
1. Explicit (t_feature_available, t_prediction, t_label_available) provenance registry across
   Benchmark A (Qt, OpenStack) and Benchmark B (Cabral19, ApacheJIT, N60 PR-SZZ, Shift Suite).
2. Builds Benchmark A (`cache/multimodal_v3_pure13/`) using ONLY the 13 commit-time metrics
   (`PURE_COMMIT_COLS_QT_OS`, zero Gerrit review fields), separate message/code vocabularies
   fit strictly on `[0 : n_init]`, and Tier-1 + Tier-2 Git SZZ fix timestamps (`+inf` if unlinked),
   plus explicit sensitivity bundles (`tier1_only`, `drop_unlinked_pos`, `premerge_28col`).
3. Cleans Benchmark B (`cache/benchmark_b_v3/`):
   - Replaces post-CSV-cutoff `max_csv_ts + 86400` sentinels in `n60_*` / `abrupt_airflow_pulsar`
     with exact `+inf` (unrevealed within observation history);
   - Excludes Gerrit-contaminated `deepjit_openstack_mm` from natural Benchmark B (since OpenStack
     is already represented in Benchmark A and in `cabral_nova`/`cabral_neutron`), and rebuilds
     `abrupt_deepjit_nova_neutron_pure13` with ONLY the 13 pure commit-time metrics;
   - Computes exact donor-project dependency clusters (`27` natural streams -> `K = 25` independent
     ecosystem clusters; `K = 24` completely disjoint from Benchmark A).
4. Keeps the 5 confirmatory ApacheJIT projects (`HDFS`, `MAPREDUCE`, `SPARK`, `ZEPPELIN`, `ZOOKEEPER`)
   LOCKED behind `--unlock-confirmatory` until Pre-Run Gate sign-off is completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import gzip
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from config_v3 import (
    APACHEJIT_RAW_COLS,
    COHORT_LA_COL_INDEX,
    CONTROLLED_TABULAR_SHIFT_STREAMS,
    EXCLUDED_GERRIT_REVIEW_COLS,
    EXCLUDED_N60_POSTCLOSURE_AND_TARGET_COLS,
    KAMEI_RAW_COLS_CABRAL,
    NATURAL_STREAM_TO_DONOR_CLUSTER,
    PURE_COMMIT_COLS_QT_OS,
    SECONDS_PER_DAY,
    UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS,
    V3_CONFIG,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def standardize_on_init(X: np.ndarray, n_init: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    mu = np.mean(X[:n_init], axis=0, keepdims=True)
    sd = np.std(X[:n_init], axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.clip((X - mu) / sd, -8.0, 8.0).astype(np.float32)


def compute_split_sizes(n_total: int) -> Tuple[int, int]:
    if n_total < 4000:
        # Preserve exact 25% init / 8% val proportions on smaller (<4k) streams where 1500+500 exceeds `n_total`
        n_init = int(np.clip(round(0.25 * n_total / 25.0) * 25, 150, 1500))
        n_val = int(np.clip(round(0.08 * n_total / 25.0) * 25, 50, 500))
        return n_init, n_val
    n_init = int(np.clip(round(0.25 * n_total / 100.0) * 100, 1500, 3500))
    n_val = int(np.clip(round(0.08 * n_total / 50.0) * 50, 500, 1000))
    return n_init, n_val


def engineer_jit_features(df: pd.DataFrame, raw_cols: List[str], la_col: str = "la", ld_col: str = "ld", nf_col: str = "nf") -> Tuple[np.ndarray, np.ndarray]:
    X_raw = df[raw_cols].fillna(0.0).astype(np.float64).values.copy()
    X_log = np.sign(X_raw) * np.log1p(np.abs(X_raw))

    la = np.maximum(df[la_col].fillna(0.0).astype(np.float64).values, 0.0)
    ld = np.maximum(df[ld_col].fillna(0.0).astype(np.float64).values, 0.0)
    nf = np.maximum(df[nf_col].fillna(0.0).astype(np.float64).values, 1.0)
    churn = la + ld
    effort = np.maximum(churn, 1.0)

    extra = np.column_stack([
        np.log1p(churn),
        la / (churn + 1.0),
        churn / nf,
        np.log1p(churn / nf),
    ])
    X = np.hstack([X_log, extra]).astype(np.float32)
    return X, effort.astype(np.float32)


def tokenize_strictly_on_init(texts: List[str], n_init: int, max_len: int, max_vocab: int = 8000) -> Tuple[np.ndarray, int]:
    tok_re = re.compile(r"[A-Za-z0-9_]+")
    tokenized = [tok_re.findall(t.lower())[:max_len] for t in texts]
    cnt: Counter = Counter()
    for toks in tokenized[:n_init]:
        cnt.update(toks)
    # 0 = PAD, 1 = UNK, 2..max_vocab-1 = top frequent tokens strictly in [0 : n_init]
    vocab = {w: idx + 2 for idx, (w, _) in enumerate(cnt.most_common(max_vocab - 2))}
    arr = np.zeros((len(texts), max_len), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        for j, w in enumerate(toks[:max_len]):
            arr[i, j] = float(vocab.get(w, 1))
    return arr, max_vocab


def load_ecosystem_v3(
    base_dir: str,
    sys_name: str,
    szz_mode: str = "tier12",        # "tier12", "tier1_only", "drop_unlinked_pos"
    feature_mode: str = "pure13",    # "pure13" or "premerge28"
) -> Dict:
    dj_dir = os.path.join(base_dir, "data_sync", "deepjit")
    jl_dir = os.path.join(base_dir, "data_sync", "jitline")
    cb_path = os.path.join(base_dir, "cache", "codebert_v3", f"codebert_{sys_name}_v3.npz")
    assert os.path.exists(cb_path), (
        f"Missing clean V3 CodeBERT cache {cb_path}! Run `build_codebert_cache_v3.py` first (`R3b-M1`)."
    )
    szz_path = os.path.join(base_dir, "data_sync", "szz_linked", f"szz_{sys_name}_exact.json")

    with open(os.path.join(dj_dir, f"{sys_name}_train.pkl"), "rb") as f:
        id_tr, y_tr, m_tr, _ = pickle.load(f)
    with open(os.path.join(dj_dir, f"{sys_name}_test.pkl"), "rb") as f:
        id_te, y_te, m_te, _ = pickle.load(f)

    raw_ids = list(id_tr) + list(id_te)
    raw_ys = np.array(list(y_tr) + list(y_te), dtype=np.int32)
    raw_msgs = [str(m) for m in (list(m_tr) + list(m_te))]

    diff_gz = os.path.join(base_dir, "data_sync", "szz_linked", f"real_diffs_{sys_name}.json.gz")
    with gzip.open(diff_gz, "rt", encoding="utf-8") as f:
        real_diffs_map = json.load(f)
    # Codex R3-M2: NEVER copy commit message into empty code diff! Preserve empty diff as "" (`PAD`)
    # and explicitly count missing-diff modality commits (`Qt: 262`, `OpenStack: 10`).
    raw_codes: List[str] = []
    missing_diff_flags = np.zeros(len(raw_ids), dtype=bool)
    for idx_d, cid in enumerate(raw_ids):
        d_str = str(real_diffs_map.get(str(cid), "") or "")
        if not d_str.strip():
            missing_diff_flags[idx_d] = True
            raw_codes.append("")
        else:
            raw_codes.append(d_str)

    cb_npz = np.load(cb_path, allow_pickle=True)
    cb_ids = [str(x) for x in cb_npz["commit_ids"]]
    assert cb_ids == [str(x) for x in raw_ids], f"CodeBERT commit_id misalignment on {sys_name}!"
    cb_msg_all = cb_npz["msg_emb"].astype(np.float32)
    cb_code_all = cb_npz["code_emb"].astype(np.float32)
    if "missing_diff_mask" in cb_npz:
        assert np.array_equal(cb_npz["missing_diff_mask"].astype(bool), missing_diff_flags), (
            f"CodeBERT missing_diff_mask mismatch on {sys_name}!"
        )

    df_m = pd.read_csv(os.path.join(jl_dir, f"{sys_name}_metrics.csv")).drop_duplicates("commit_id").set_index("commit_id")
    df_sub = df_m.loc[raw_ids].reset_index()

    with open(szz_path, "r", encoding="utf-8") as f:
        szz = json.load(f)
    linked_fixes = szz["linked_fixes"]
    repo_map = szz["commit_repo_map"]

    if feature_mode == "pure13":
        use_cols = list(PURE_COMMIT_COLS_QT_OS)
        assert len(set(use_cols) & set(EXCLUDED_GERRIT_REVIEW_COLS)) == 0, "Gerrit review column leaked!"
    elif feature_mode == "premerge28":
        use_cols = [c for c in df_sub.columns if c not in ("commit_id", "author_date", "bugcount", "fixcount")]
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    tab_df = df_sub[use_cols].copy()
    for col in tab_df.columns:
        if tab_df[col].dtype == object:
            tab_df[col] = tab_df[col].map({"true": 1.0, "True": 1.0, "false": 0.0, "False": 0.0}).fillna(0.0)
        else:
            tab_df[col] = pd.to_numeric(tab_df[col], errors="coerce").fillna(0.0)
    tab_raw = tab_df.values.astype(np.float64)
    tab_log = (np.sign(tab_raw) * np.log1p(np.abs(tab_raw))).astype(np.float32)

    la_raw = np.maximum(pd.to_numeric(df_sub["la"], errors="coerce").fillna(0.0).values, 0.0).astype(np.float32)
    ld_raw = np.maximum(pd.to_numeric(df_sub["ld"], errors="coerce").fillna(0.0).values, 0.0).astype(np.float32)
    effort_arr = np.maximum(la_raw + ld_raw, 1.0).astype(np.float32)

    ts_arr = df_sub["author_date"].values.astype(np.float64)
    fix_ts_arr = np.full(len(raw_ids), np.inf, dtype=np.float64)
    fix_sha_list = [""] * len(raw_ids)
    fix_tier_list = ["clean"] * len(raw_ids)
    repos_list = [repo_map.get(cid, "other") for cid in raw_ids]

    allowed_tiers = ("Tier1_McIntosh_CSV",) if szz_mode == "tier1_only" else ("Tier1_McIntosh_CSV", "Tier2_Git_BugFix")
    keep_mask = np.ones(len(raw_ids), dtype=bool)
    n_tier1, n_tier2, n_tier3_dropped, n_unlinked, n_retroactive_quarantined = 0, 0, 0, 0, 0

    for i, (cid, yv) in enumerate(zip(raw_ids, raw_ys)):
        if yv == 1:
            info = linked_fixes.get(cid)
            if info is not None:
                tier = info.get("tier", "")
                if tier in allowed_tiers:
                    fts = float(info["fix_ts"])
                    # Codex R3-M2: Never clamp retroactive `fts < ts` to `ts + 60s`; quarantine strictly to `+inf`!
                    if fts < ts_arr[i]:
                        fix_ts_arr[i] = np.inf
                        fix_tier_list[i] = "quarantined_retroactive_fix_ts"
                        n_retroactive_quarantined += 1
                        if szz_mode == "drop_unlinked_pos":
                            keep_mask[i] = False
                    else:
                        fix_ts_arr[i] = fts
                        fix_sha_list[i] = str(info.get("fix_sha", ""))
                        fix_tier_list[i] = tier
                        if tier == "Tier1_McIntosh_CSV":
                            n_tier1 += 1
                        else:
                            n_tier2 += 1
                else:
                    fix_ts_arr[i] = np.inf
                    fix_tier_list[i] = f"unrevealed_{tier}"
                    if tier == "Tier3_Line_Modification":
                        n_tier3_dropped += 1
                    else:
                        n_unlinked += 1
                    if szz_mode == "drop_unlinked_pos":
                        keep_mask[i] = False
            else:
                fix_ts_arr[i] = np.inf
                fix_tier_list[i] = "unlinked_inf"
                n_unlinked += 1
                if szz_mode == "drop_unlinked_pos":
                    keep_mask[i] = False

    valid_idx = np.where(keep_mask)[0]
    order = valid_idx[np.argsort(ts_arr[valid_idx], kind="mergesort")]

    pos_revealed = (raw_ys[order] == 1) & np.isfinite(fix_ts_arr[order])
    lats_d = (fix_ts_arr[order][pos_revealed] - ts_arr[order][pos_revealed]) / SECONDS_PER_DAY
    n_missing_diff = int(np.sum(missing_diff_flags[order]))

    return {
        "ecosystem": sys_name,
        "feature_cols": use_cols,
        "szz_mode": szz_mode,
        "ids": [raw_ids[k] for k in order],
        "repos": [repos_list[k] for k in order],
        "fix_shas": [fix_sha_list[k] for k in order],
        "fix_tiers": [fix_tier_list[k] for k in order],
        "msgs": [raw_msgs[k] for k in order],
        "codes": [raw_codes[k] for k in order],
        "missing_diff_mask": missing_diff_flags[order],
        "ts": ts_arr[order],
        "y": raw_ys[order],
        "fix_ts": fix_ts_arr[order],
        "la_raw": la_raw[order],
        "effort": effort_arr[order],
        "tab_log": tab_log[order],
        "cb_msg": cb_msg_all[order],
        "cb_code": cb_code_all[order],
        "audit_stats": {
            "total_commits": int(len(order)),
            "total_buggy": int(np.sum(raw_ys[order] == 1)),
            "prevalence": float(np.mean(raw_ys[order])),
            "missing_diff_commits_count": n_missing_diff,
            "missing_diff_rate": float(n_missing_diff / max(len(order), 1)),
            "revealed_tier1": n_tier1,
            "revealed_tier2": n_tier2,
            "retroactive_links_quarantined_to_inf": n_retroactive_quarantined,
            "tier3_excluded_to_inf": n_tier3_dropped,
            "unlinked_kept_as_inf": n_unlinked if szz_mode != "drop_unlinked_pos" else 0,
            "unlinked_dropped": int(np.sum(~keep_mask)),
            "verified_reveal_coverage_of_buggy": float((n_tier1 + n_tier2) / max(int(np.sum(raw_ys[order] == 1)), 1)),
            "median_latency_days": float(np.median(lats_d)) if len(lats_d) > 0 else 0.0,
            "p75_latency_days": float(np.percentile(lats_d, 75)) if len(lats_d) > 0 else 0.0,
        },
    }


def save_multimodal_stream_v3(
    out_dir: str,
    stream_name: str,
    stream_kind: str,
    eco_data: Dict,
    indices: List[int],
    n_init: int,
    n_val: int,
    drift_onset_idx: int = -1,
    ts_override: Optional[np.ndarray] = None,
    fix_override: Optional[np.ndarray] = None,
    save_meta_text: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    s_msgs = [eco_data["msgs"][i] for i in indices]
    s_codes = [eco_data["codes"][i] for i in indices]
    s_ids = np.array([eco_data["ids"][i] for i in indices])
    s_repos = np.array([eco_data["repos"][i] for i in indices])
    s_fix_shas = np.array([eco_data["fix_shas"][i] for i in indices])
    s_fix_tiers = np.array([eco_data["fix_tiers"][i] for i in indices])
    s_miss_diff = eco_data["missing_diff_mask"][indices].astype(bool)

    ts = ts_override if ts_override is not None else eco_data["ts"][indices]
    fix_ts = fix_override if fix_override is not None else eco_data["fix_ts"][indices]
    y = eco_data["y"][indices]
    eff = eco_data["effort"][indices]
    la_raw = eco_data["la_raw"][indices]

    tab_scaled = standardize_on_init(eco_data["tab_log"][indices], n_init)
    X_la = standardize_on_init(np.log1p(la_raw)[:, None], n_init)
    X_churn = standardize_on_init(np.log1p(eff)[:, None], n_init)

    # 1. Dual-vocabulary token sequence bundle (`tok`: msg_len=64, code_len=128, tab_dim=D)
    msg_ids, vocab_msg = tokenize_strictly_on_init(s_msgs, n_init, max_len=64, max_vocab=8000)
    code_ids, vocab_code = tokenize_strictly_on_init(s_codes, n_init, max_len=128, max_vocab=8000)
    X_tok = np.hstack([msg_ids, code_ids, tab_scaled]).astype(np.float32)

    # 2. Pretrained CodeBERT semantic bundle (`cb`: 768 + 768 + D)
    cb_m = standardize_on_init(eco_data["cb_msg"][indices], n_init)
    cb_c = standardize_on_init(eco_data["cb_code"][indices], n_init)
    X_cb = np.hstack([cb_m, cb_c, tab_scaled]).astype(np.float32)

    # 3. Causal TF-IDF SVD + Tabular bundle (`svd`: 48 + 64 + D)
    vec_m = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    svd_m = TruncatedSVD(n_components=48, random_state=42)
    M_f = np.vstack([
        svd_m.fit_transform(vec_m.fit_transform(s_msgs[:n_init])),
        svd_m.transform(vec_m.transform(s_msgs[n_init:])),
    ])
    vec_c = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    svd_c = TruncatedSVD(n_components=64, random_state=42)
    C_f = np.vstack([
        svd_c.fit_transform(vec_c.fit_transform(s_codes[:n_init])),
        svd_c.transform(vec_c.transform(s_codes[n_init:])),
    ])
    X_svd = np.hstack([
        standardize_on_init(np.hstack([M_f, C_f]), n_init),
        tab_scaled,
    ]).astype(np.float32)

    bundle_hashes = {}
    for rep_key, X_mat in [("tok", X_tok), ("cb", X_cb), ("svd", X_svd), ("tab", tab_scaled)]:
        out_p = os.path.join(out_dir, f"{stream_name}_{rep_key}.npz")
        np.savez_compressed(
            out_p,
            X=X_mat,
            X_tab=tab_scaled,
            X_la=X_la,
            X_churn=X_churn,
            commit_ts=ts.astype(np.float64),
            y_true=y.astype(np.int32),
            fix_ts=fix_ts.astype(np.float64),
            effort=eff.astype(np.float32),
            la_raw=la_raw.astype(np.float32),
            n_init=np.int32(n_init),
            n_val=np.int32(n_val),
            drift_onset_idx=np.int32(drift_onset_idx),
            missing_diff_mask=s_miss_diff,
        )
        bundle_hashes[rep_key] = sha256_file(out_p)

    if save_meta_text:
        meta_p = os.path.join(out_dir, f"{stream_name}_meta.npz")
        np.savez_compressed(
            meta_p,
            commit_ids=s_ids,
            repos=s_repos,
            fix_shas=s_fix_shas,
            fix_tiers=s_fix_tiers,
            missing_diff_mask=s_miss_diff,
            msgs=np.array(s_msgs, dtype=object),
            codes=np.array([c[:2400] for c in s_codes], dtype=object),
        )
        bundle_hashes["meta"] = sha256_file(meta_p)

    pos_fin = (y == 1) & np.isfinite(fix_ts)
    med_lat = float(np.median((fix_ts[pos_fin] - ts[pos_fin]) / SECONDS_PER_DAY)) if np.any(pos_fin) else 0.0
    info = {
        "name": stream_name,
        "ecosystem": eco_data["ecosystem"],
        "stream_kind": stream_kind,
        "feature_cols": eco_data["feature_cols"],
        "szz_mode": eco_data["szz_mode"],
        "N": int(len(indices)),
        "n_init": int(n_init),
        "n_val": int(n_val),
        "stream_eval_N": int(len(indices) - n_init - n_val),
        "prevalence": float(np.mean(y)),
        "median_latency_days": med_lat,
        "drift_onset_idx": int(drift_onset_idx),
        "tok_meta": {
            "vocab_msg": vocab_msg,
            "vocab_code": vocab_code,
            "msg_len": 64,
            "code_len": 128,
            "tab_dim": int(tab_scaled.shape[1]),
            "n_msg_chunks": 8,
            "n_code_hunks": 8,
        },
        "cb_meta": {"cb_dim": 768, "tab_dim": int(tab_scaled.shape[1])},
        "svd_meta": {"svd_dim": 112, "tab_dim": int(tab_scaled.shape[1])},
        "sha256": bundle_hashes,
    }
    print(
        f"  [MM-V3 {stream_kind:24s}] {stream_name:28s}: N={info['N']:6d} "
        f"({n_init}/{n_val}/{info['stream_eval_N']}), tab_D={tab_scaled.shape[1]}, "
        f"prev={info['prevalence']:.3f}, med_lat={med_lat:5.1f}d",
        flush=True,
    )
    return info


def build_abrupt_pair_v3(
    eco_data: Dict,
    pre_repos: Tuple[str, ...],
    post_repos: Tuple[str, ...],
    max_pre: int,
    max_post: int,
    pre_offset: int = 0,
    post_offset: int = 0,
) -> Tuple[List[int], np.ndarray, np.ndarray, int]:
    pre_all = [i for i, r in enumerate(eco_data["repos"]) if r in pre_repos]
    post_all = [i for i, r in enumerate(eco_data["repos"]) if r in post_repos]
    pre_idx = pre_all[pre_offset : pre_offset + max_pre]
    post_idx = post_all[post_offset : post_offset + max_post]
    comb = pre_idx + post_idx
    dt = float(eco_data["ts"][pre_idx[-1]]) - float(eco_data["ts"][post_idx[0]]) + 3600.0
    ts_c = np.concatenate([eco_data["ts"][pre_idx], eco_data["ts"][post_idx] + dt])
    # Note: np.inf + dt == np.inf cleanly in IEEE-754!
    fix_c = np.concatenate([eco_data["fix_ts"][pre_idx], eco_data["fix_ts"][post_idx] + dt])
    return comb, ts_c, fix_c, len(pre_idx)


def build_all_benchmark_a_v3(base_dir: str, force_rebuild: bool = False) -> Dict:
    out_pure = os.path.join(base_dir, "cache", "multimodal_v3_pure13")
    out_sens = os.path.join(base_dir, "cache", "multimodal_v3_sens")
    os.makedirs(out_pure, exist_ok=True)
    os.makedirs(out_sens, exist_ok=True)
    man_a_p = os.path.join(out_pure, "manifest_mm_v3.json")
    man_s_p = os.path.join(out_sens, "manifest_sens_v3.json")

    qt_pure = load_ecosystem_v3(base_dir, "qt", szz_mode="tier12", feature_mode="pure13")
    os_pure = load_ecosystem_v3(base_dir, "openstack", szz_mode="tier12", feature_mode="pure13")
    if (not force_rebuild) and os.path.exists(man_a_p) and os.path.exists(man_s_p):
        print("=== [Step 1-2] Reusing Verified Benchmark A Primary & Sensitivity V3 Caches ===", flush=True)
        with open(man_a_p, "r", encoding="utf-8") as f:
            manifest_a = json.load(f)
        with open(man_s_p, "r", encoding="utf-8") as f:
            sens_manifest = json.load(f)
        return {
            "qt_audit": qt_pure["audit_stats"],
            "openstack_audit": os_pure["audit_stats"],
            "primary_streams": manifest_a,
            "sensitivity_streams": sens_manifest,
        }

    print("=== [Step 1] Auditing & Building Benchmark A Primary V3 Cache (13 Pure Commit-Time Metrics, Tier 1+2 SZZ) ===", flush=True)

    manifest_a: List[Dict] = []
    # Natural Ecosystem Streams (2)
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "qt_natural_25k", "natural_ecosystem",
            qt_pure, list(range(len(qt_pure["ids"]))), n_init=6000, n_val=1500, save_meta_text=True,
        )
    )
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "openstack_natural_13k", "natural_ecosystem",
            os_pure, list(range(len(os_pure["ids"]))), n_init=8600, n_val=1000, save_meta_text=True,
        )
    )

    # Controlled Intra-Ecosystem Subsystem Shifts (4)
    idx_qd, ts_qd, fix_qd, onset_qd = build_abrupt_pair_v3(qt_pure, ("qtbase",), ("qtdeclarative",), 5500, 4000, 0, 0)
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "qt_abrupt_qtbase_qtdecl", "controlled_subsystem_shift",
            qt_pure, idx_qd, n_init=4500, n_val=1000, drift_onset_idx=onset_qd,
            ts_override=ts_qd, fix_override=fix_qd, save_meta_text=True,
        )
    )
    idx_qux, ts_qux, fix_qux, onset_qux = build_abrupt_pair_v3(
        qt_pure, ("qtbase", "qttools"), ("qtlocation", "qtmultimedia", "qtquickcontrols", "qtserialport", "qtconnectivity"), 5200, 2800, 0, 0
    )
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "qt_abrupt_core_ux", "controlled_subsystem_shift",
            qt_pure, idx_qux, n_init=4200, n_val=1000, drift_onset_idx=onset_qux,
            ts_override=ts_qux, fix_override=fix_qux, save_meta_text=False,
        )
    )
    idx_on, ts_on, fix_on, onset_on = build_abrupt_pair_v3(os_pure, ("nova",), ("neutron",), 4200, 1800, 3000, 372)
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "openstack_abrupt_nova_neut", "controlled_subsystem_shift",
            os_pure, idx_on, n_init=3400, n_val=800, drift_onset_idx=onset_on,
            ts_override=ts_on, fix_override=fix_on, save_meta_text=True,
        )
    )
    idx_osg, ts_osg, fix_osg, onset_osg = build_abrupt_pair_v3(os_pure, ("nova",), ("cinder", "glance", "swift"), 4200, 2900, 3000, 697)
    manifest_a.append(
        save_multimodal_stream_v3(
            out_pure, "openstack_abrupt_comp_stor", "controlled_subsystem_shift",
            os_pure, idx_osg, n_init=3400, n_val=800, drift_onset_idx=onset_osg,
            ts_override=ts_osg, fix_override=fix_osg, save_meta_text=False,
        )
    )

    with open(os.path.join(out_pure, "manifest_mm_v3.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_a, f, indent=2)

    print("\n=== [Step 2] Building Benchmark A Sensitivity Bundles (Exact Temporal Cutoffs & Common Eval IDs) ===", flush=True)
    sens_manifest: List[Dict] = []
    primary_map = {"qt": (qt_pure, 6000, 1500), "openstack": (os_pure, 8600, 1000)}
    for sys_name, (prim_eco, n_init_s, n_val_s) in primary_map.items():
        # Exact calendar & commit-membership partition from primary `tier12` stream (Codex R3-M2)!
        init_id_set = set(prim_eco["ids"][:n_init_s])
        val_id_set = set(prim_eco["ids"][n_init_s : n_init_s + n_val_s])
        eval_id_set = set(prim_eco["ids"][n_init_s + n_val_s :])
        for s_tag, szz_m, feat_m in [
            ("tier1_only", "tier1_only", "pure13"),
            ("drop_unlinked", "drop_unlinked_pos", "pure13"),
            ("premerge28", "tier12", "premerge28"),
        ]:
            eco_s = load_ecosystem_v3(base_dir, sys_name, szz_mode=szz_m, feature_mode=feat_m)
            # Because `order` is chronological (`mergesort`) over a subset of `prim_eco['ids']`,
            # membership in `init_id_set`, `val_id_set`, `eval_id_set` forms three contiguous blocks
            # preserving the EXACT SAME calendar cutoffs and common evaluation IDs!
            ni = sum(1 for cid in eco_s["ids"] if cid in init_id_set)
            nv = sum(1 for cid in eco_s["ids"] if cid in val_id_set)
            ne = sum(1 for cid in eco_s["ids"] if cid in eval_id_set)
            assert ni + nv + ne == len(eco_s["ids"]), "Sensitivity split partition mismatch!"
            assert all(cid in eval_id_set for cid in eco_s["ids"][ni + nv :]), "Eval ID leaked across time cutoff!"
            sm_entry = save_multimodal_stream_v3(
                out_sens, f"{sys_name}_{s_tag}", f"sensitivity_{s_tag}",
                eco_s, list(range(len(eco_s["ids"]))), n_init=ni, n_val=nv, save_meta_text=True,
            )
            sm_entry["aligned_to_primary_eval_N"] = len(eval_id_set)
            sens_manifest.append(sm_entry)
    with open(os.path.join(out_sens, "manifest_sens_v3.json"), "w", encoding="utf-8") as f:
        json.dump(sens_manifest, f, indent=2)

    return {
        "qt_audit": qt_pure["audit_stats"],
        "openstack_audit": os_pure["audit_stats"],
        "primary_streams": manifest_a,
        "sensitivity_streams": sens_manifest,
    }


def build_and_audit_benchmark_b_v3(base_dir: str) -> Dict:
    src_dir = os.path.join(base_dir, "cache", "full32")
    out_dir = os.path.join(base_dir, "cache", "benchmark_b_v3")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(src_dir, "manifest.json"), "r", encoding="utf-8") as f:
        old_manifest = {item["name"]: item for item in json.load(f)}

    print("\n=== [Step 3] Auditing & Cleaning Benchmark B (27 Natural Streams -> K=25 Donor Clusters + 4 Controlled Shifts) ===", flush=True)
    natural_entries: List[Dict] = []
    shift_entries: List[Dict] = []
    sentinel_fixes_cleaned: Dict[str, int] = {}

    # 1. Process all 27 natural streams
    for sname, donor_cluster in NATURAL_STREAM_TO_DONOR_CLUSTER.items():
        meta_old = old_manifest[sname]
        cohort = meta_old["cohort"]
        b = np.load(os.path.join(src_dir, f"{sname}.npz"))
        X = b["X"].astype(np.float32)
        ts = b["commit_ts"].astype(np.float64)
        y = b["y_true"].astype(np.int32)
        fts = b["fix_ts"].astype(np.float64).copy()
        eff = b["effort"].astype(np.float32)
        n_init = int(b["n_init"])
        n_val = int(b["n_val"])

        # Strict causal fix_ts hygiene (Codex R3-M2): quarantine ANY `fix_ts < commit_ts` strictly to `+inf` (zero clamping!)
        assert np.all(np.diff(ts) >= 0.0), f"Non-monotonic commit_ts in {sname}!"
        fin_mask = np.isfinite(fts)
        retro_corrupt = fin_mask & (fts < ts)
        n_retro_cleaned = int(np.sum(retro_corrupt))
        if n_retro_cleaned > 0:
            fts[retro_corrupt] = np.inf
            fin_mask = np.isfinite(fts)
        assert np.all(fts[fin_mask] >= ts[fin_mask]), f"Retroactive fix_ts < commit_ts in {sname}!"

        # Clean post-CSV-cutoff `max_csv_ts + 86400` sentinel in N60_PRSZZ to exact `+inf`
        n_cleaned = 0
        if cohort == "N60_PRSZZ" and np.any(fin_mask):
            mx_fin = float(np.max(fts[fin_mask]))
            # In build_full_benchmarks.py, unmatched post-2021H1 fixes were assigned `max_csv_ts + 86400.0` (~1625178324)
            is_sentinel = fin_mask & (np.abs(fts - mx_fin) < 1.0)
            n_cleaned = int(np.sum(is_sentinel))
            fts[is_sentinel] = np.inf
            sentinel_fixes_cleaned[sname] = n_cleaned

        la_idx = COHORT_LA_COL_INDEX[cohort]
        X_la = X[:, [la_idx]].astype(np.float32)
        X_churn = standardize_on_init(np.log1p(eff)[:, None], n_init)

        out_p = os.path.join(out_dir, f"{sname}.npz")
        np.savez_compressed(
            out_p,
            X=X,
            X_tab=X,
            X_la=X_la,
            X_churn=X_churn,
            commit_ts=ts,
            y_true=y,
            fix_ts=fts,
            effort=eff,
            n_init=np.int32(n_init),
            n_val=np.int32(n_val),
            drift_onset_idx=np.int32(-1),
        )
        pos_rev = (y == 1) & np.isfinite(fts)
        med_lat = float(np.median((fts[pos_rev] - ts[pos_rev]) / SECONDS_PER_DAY)) if np.any(pos_rev) else 0.0

        # Explicit temporal semantics per cohort (Codex Criterion 1)
        if cohort in ("Cabral19", "ApacheJIT"):
            pred_time_sem = "git_author_date_historical_replay"
            feat_avail_sem = "commit_author_time_kamei_metrics_no_review_cols"
            lbl_avail_sem = "earliest_szz_bugfix_commit_timestamp"
        else:
            pred_time_sem = "git_committer_integration_date"
            feat_avail_sem = "post_pr_review_integration_time_ast_and_pr_workflow_metrics"
            lbl_avail_sem = "earliest_automated_prszz_bugfix_commit_timestamp"

        entry = {
            "name": sname,
            "cohort": cohort,
            "stream_kind": "natural_project",
            "donor_cluster": donor_cluster,
            "overlaps_benchmark_a": bool(donor_cluster == "openstack"),
            "N": int(len(X)),
            "dim": int(X.shape[1]),
            "n_init": n_init,
            "n_val": n_val,
            "stream_eval_N": int(len(X) - n_init - n_val),
            "prevalence": float(np.mean(y)),
            "revealed_buggy_count": int(np.sum(pos_rev)),
            "unrevealed_inf_buggy_count": int(np.sum((y == 1) & (~np.isfinite(fts)))),
            "retroactive_fixes_cleaned_to_inf": n_retro_cleaned,
            "sentinel_fixes_cleaned_to_inf": n_cleaned,
            "median_latency_days": med_lat,
            "prediction_time_semantics": pred_time_sem,
            "feature_availability_semantics": feat_avail_sem,
            "label_availability_semantics": lbl_avail_sem,
            "sha256": sha256_file(out_p),
        }
        natural_entries.append(entry)
        print(
            f"  [Nat-B {cohort:9s}] {sname:20s} -> cluster={donor_cluster:18s} | "
            f"N={entry['N']:6d}, D={entry['dim']:2d}, prev={entry['prevalence']:.3f}, med_lat={med_lat:5.1f}d",
            flush=True,
        )

    # 2. Process 3 tabular concatenated shifts + 1 rebuilt Gerrit-free OpenStack Nova->Neutron SVD+13col shift
    for sname, cohort_src in [
        ("abrupt_nova_zulip", "Cabral19"),
        ("abrupt_ignite_cassandra", "ApacheJIT"),
        ("abrupt_airflow_pulsar", "N60_PRSZZ"),
    ]:
        b = np.load(os.path.join(src_dir, f"{sname}.npz"))
        X = b["X"].astype(np.float32)
        ts = b["commit_ts"].astype(np.float64)
        y = b["y_true"].astype(np.int32)
        fts = b["fix_ts"].astype(np.float64).copy()
        eff = b["effort"].astype(np.float32)
        n_init = int(b["n_init"])
        n_val = int(b["n_val"])
        onset = int(b["drift_onset_idx"])

        if cohort_src == "N60_PRSZZ":
            # Clean shifted post-cutoff sentinels in both halves of abrupt_airflow_pulsar
            fin_pre = np.isfinite(fts[:onset])
            if np.any(fin_pre):
                mx1 = float(np.max(fts[:onset][fin_pre]))
                fts[:onset][fin_pre & (np.abs(fts[:onset] - mx1) < 1.0)] = np.inf
            fin_post = np.isfinite(fts[onset:])
            if np.any(fin_post):
                mx2 = float(np.max(fts[onset:][fin_post]))
                fts[onset:][fin_post & (np.abs(fts[onset:] - mx2) < 1.0)] = np.inf

        la_idx = COHORT_LA_COL_INDEX[cohort_src]
        X_la = X[:, [la_idx]].astype(np.float32)
        X_churn = standardize_on_init(np.log1p(eff)[:, None], n_init)

        out_p = os.path.join(out_dir, f"{sname}.npz")
        np.savez_compressed(
            out_p,
            X=X,
            X_tab=X,
            X_la=X_la,
            X_churn=X_churn,
            commit_ts=ts,
            y_true=y,
            fix_ts=fts,
            effort=eff,
            n_init=np.int32(n_init),
            n_val=np.int32(n_val),
            drift_onset_idx=np.int32(onset),
        )
        shift_entries.append({
            "name": sname,
            "cohort": "ControlledShift",
            "stream_kind": "synthetic_concatenated_shift",
            "donor_clusters": list(CONTROLLED_TABULAR_SHIFT_STREAMS[sname]),
            "N": int(len(X)),
            "dim": int(X.shape[1]),
            "n_init": n_init,
            "n_val": n_val,
            "drift_onset_idx": onset,
            "prevalence": float(np.mean(y)),
            "sha256": sha256_file(out_p),
        })

    # Also copy the clean 13-col OpenStack Nova->Neutron SVD stream into ControlledShift so zero Gerrit cols exist!
    mm_nn = np.load(os.path.join(base_dir, "cache", "multimodal_v3_pure13", "openstack_abrupt_nova_neut_svd.npz"))
    sname_nn = "abrupt_deepjit_nova_neutron_pure13"
    out_nn = os.path.join(out_dir, f"{sname_nn}.npz")
    np.savez_compressed(
        out_nn,
        X=mm_nn["X"],
        X_tab=mm_nn["X_tab"],
        X_la=mm_nn["X_la"],
        X_churn=mm_nn["X_churn"],
        commit_ts=mm_nn["commit_ts"],
        y_true=mm_nn["y_true"],
        fix_ts=mm_nn["fix_ts"],
        effort=mm_nn["effort"],
        n_init=mm_nn["n_init"],
        n_val=mm_nn["n_val"],
        drift_onset_idx=mm_nn["drift_onset_idx"],
    )
    shift_entries.append({
        "name": sname_nn,
        "cohort": "ControlledShift",
        "stream_kind": "controlled_subsystem_shift_pure13",
        "donor_clusters": ["openstack"],
        "N": int(len(mm_nn["X"])),
        "dim": int(mm_nn["X"].shape[1]),
        "n_init": int(mm_nn["n_init"]),
        "n_val": int(mm_nn["n_val"]),
        "drift_onset_idx": int(mm_nn["drift_onset_idx"]),
        "prevalence": float(np.mean(mm_nn["y_true"])),
        "sha256": sha256_file(out_nn),
    })

    unique_clusters = sorted(set(NATURAL_STREAM_TO_DONOR_CLUSTER.values()))
    disjoint_from_a = [c for c in unique_clusters if c != "openstack"]
    cluster_members = {
        c: sorted([s for s, dc in NATURAL_STREAM_TO_DONOR_CLUSTER.items() if dc == c])
        for c in unique_clusters
    }

    manifest_b = {
        "natural_streams": natural_entries,
        "controlled_shift_streams": shift_entries,
        "donor_cluster_summary": {
            "n_natural_streams": len(natural_entries),
            "K_independent_ecosystem_clusters": len(unique_clusters),
            "K_benchmark_a_disjoint_clusters": len(disjoint_from_a),
            "merged_multi_stream_clusters": {k: v for k, v in cluster_members.items() if len(v) > 1},
            "cluster_members": cluster_members,
        },
        "sentinel_fixes_cleaned_to_inf": sentinel_fixes_cleaned,
    }
    with open(os.path.join(out_dir, "manifest_b_v3.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_b, f, indent=2)

    print(
        f"  [Benchmark B Summary] {len(natural_entries)} natural streams -> "
        f"K={len(unique_clusters)} independent clusters (K={len(disjoint_from_a)} disjoint from Benchmark A); "
        f"{len(shift_entries)} controlled shift streams separated.",
        flush=True,
    )
    return manifest_b


def download_and_build_confirmatory_apache_v3(base_dir: str) -> Dict:
    """
    Strictly Post-Sign-Off Confirmatory Cohort Builder (`HDFS`, `MAPREDUCE`, `SPARK`, `ZEPPELIN`, `ZOOKEEPER`).
    Must ONLY be invoked with `--unlock-confirmatory` AFTER Codex Round 3 Pre-Run Gate sign-off!
    """
    raw_dir = os.path.join(base_dir, "data_sync", "confirmatory_apachejit_raw")
    out_dir = os.path.join(base_dir, "cache", "confirmatory_v3")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    gh_base = "https://raw.githubusercontent.com/hosseinkshvrz/apachejit/main"
    total_csv = os.path.join(raw_dir, "apachejit_total.csv")
    need_clone = not os.path.exists(total_csv)
    if need_clone:
        try:
            url_tot = f"{gh_base}/dataset/apachejit_total.csv"
            print(f"  [Download Confirmatory] Fetching {url_tot} ...", flush=True)
            urllib.request.urlretrieve(url_tot, total_csv)
            for _, short_code, _, _ in UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS:
                lf_local = os.path.join(raw_dir, f"commit_links_{short_code}.csv")
                if not os.path.exists(lf_local):
                    urllib.request.urlretrieve(f"{gh_base}/data/commit_links_{short_code}.csv", lf_local)
        except Exception:
            tmp_clone = os.path.join(raw_dir, "_git_tmp")
            shutil.rmtree(tmp_clone, ignore_errors=True)
            print("  [Download Confirmatory] Using shallow git clone of hosseinkshvrz/apachejit ...", flush=True)
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hosseinkshvrz/apachejit.git", tmp_clone], check=True)
            shutil.copy2(os.path.join(tmp_clone, "dataset", "apachejit_total.csv"), total_csv)
            for _, short_code, _, _ in UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS:
                shutil.copy2(os.path.join(tmp_clone, "data", f"commit_links_{short_code}.csv"), os.path.join(raw_dir, f"commit_links_{short_code}.csv"))
            shutil.rmtree(tmp_clone, ignore_errors=True)

    df_aj = pd.read_csv(total_csv)
    df_aj["fix"] = df_aj["fix"].astype(float)

    link_dfs = [pd.read_csv(os.path.join(raw_dir, f"commit_links_{sc}.csv")) for _, sc, _, _ in UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS]

    df_links = pd.concat(link_dfs, ignore_index=True)
    earliest_fix = df_links.groupby("bug_hash")["fix_date"].min().to_dict()
    df_aj["fix_date"] = df_aj["commit_id"].map(earliest_fix)
    cutoff_aj = pd.Timestamp("2019-01-01").timestamp()

    conf_entries: List[Dict] = []
    for proj_full, short_code, sname, umbrella_cluster in UNTOUCHED_CONFIRMATORY_APACHE_PROJECTS:
        grp = df_aj[(df_aj["project"] == proj_full) & (df_aj["author_date"] <= cutoff_aj)].sort_values("author_date", kind="mergesort").reset_index(drop=True)
        if len(grp) > 15000:
            grp = grp.iloc[-15000:].reset_index(drop=True)

        ts = grp["author_date"].values.astype(np.float64)
        y = grp["buggy"].values.astype(np.int32)
        raw_f = grp["fix_date"].values.astype(np.float64)
        # Strict quarantine of any retroactive `raw_f < ts` to `+inf` (zero clamping!)
        valid_reveal = (y == 1) & np.isfinite(raw_f) & (raw_f >= ts)
        fts = np.where(valid_reveal, raw_f, np.inf)
        n_retro_conf = int(np.sum((y == 1) & np.isfinite(raw_f) & (raw_f < ts)))

        X_unscaled, eff = engineer_jit_features(grp, APACHEJIT_RAW_COLS)
        n_init, n_val = compute_split_sizes(len(grp))
        X = standardize_on_init(X_unscaled, n_init)
        X_la = X[:, [COHORT_LA_COL_INDEX["ApacheJIT"]]].astype(np.float32)
        X_churn = standardize_on_init(np.log1p(eff)[:, None], n_init)

        out_p = os.path.join(out_dir, f"{sname}.npz")
        np.savez_compressed(
            out_p,
            X=X,
            X_tab=X,
            X_la=X_la,
            X_churn=X_churn,
            commit_ts=ts,
            y_true=y,
            fix_ts=fts,
            effort=eff,
            n_init=np.int32(n_init),
            n_val=np.int32(n_val),
            drift_onset_idx=np.int32(-1),
        )
        pos_rev = (y == 1) & np.isfinite(fts)
        med_lat = float(np.median((fts[pos_rev] - ts[pos_rev]) / SECONDS_PER_DAY)) if np.any(pos_rev) else 0.0
        entry = {
            "name": sname,
            "project": proj_full,
            "short_code": short_code,
            "cohort": "Confirmatory_ApacheJIT",
            "stream_kind": "untouched_confirmatory",
            "repo_cluster": sname,
            "umbrella_ecosystem_cluster": umbrella_cluster,
            "N": int(len(X)),
            "dim": int(X.shape[1]),
            "n_init": n_init,
            "n_val": n_val,
            "stream_eval_N": int(len(X) - n_init - n_val),
            "prevalence": float(np.mean(y)),
            "revealed_buggy_count": int(np.sum(pos_rev)),
            "unrevealed_inf_buggy_count": int(np.sum((y == 1) & (~np.isfinite(fts)))),
            "retroactive_fixes_cleaned_to_inf": n_retro_conf,
            "median_latency_days": med_lat,
            "sha256": sha256_file(out_p),
        }
        conf_entries.append(entry)
        print(
            f"  [Confirmatory] {sname:22s} ({proj_full:24s}): N={entry['N']:6d}, "
            f"prev={entry['prevalence']:.3f}, med_lat={med_lat:5.1f}d",
            flush=True,
        )

    payload = {
        "confirmatory_streams": conf_entries,
        "n_repositories": len(conf_entries),
        "n_umbrella_clusters": len(set(e["umbrella_ecosystem_cluster"] for e in conf_entries)),
    }
    with open(os.path.join(out_dir, "manifest_confirmatory_v3.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build & Audit V3 Benchmarks")
    parser.add_argument("--base-dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--rebuild-all", action="store_true")
    parser.add_argument("--unlock-confirmatory", action="store_true", help="ONLY set after Pre-Run Gate sign-off!")
    args = parser.parse_args()

    res_dir = os.path.join(args.base_dir, "results_v3")
    os.makedirs(res_dir, exist_ok=True)
    src_v3_dir = os.path.join(args.base_dir, "src_v3")
    code_hashes = {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(src_v3_dir, "*.py")))}

    conf_cache = os.path.join(args.base_dir, "cache", "confirmatory_v3")
    if not args.unlock_confirmatory:
        audit_a = build_all_benchmark_a_v3(args.base_dir, force_rebuild=args.rebuild_all)
        audit_b = build_and_audit_benchmark_b_v3(args.base_dir)
        cb_man_path = os.path.join(args.base_dir, "cache", "codebert_v3", "manifest_codebert_v3.json")
        cb_man_obj = json.load(open(cb_man_path, "r", encoding="utf-8")) if os.path.exists(cb_man_path) else {}
        full_audit = {
            "provenance": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "v3_config": json.loads(json.dumps(V3_CONFIG.to_dict())),
                "code_sha256": code_hashes,
                "codebert_v3_encoder_provenance": cb_man_obj,
            },
            "confirmatory_locked_untouched": not os.path.exists(conf_cache),
            "temporal_provenance_notes": {
                "Benchmark_A_Primary": (
                    "Archival author_date historical replay of merged Qt/OpenStack artifacts using ONLY 13 "
                    "commit-time change/diffusion/history/author-experience features (0 Gerrit review fields); "
                    "not certified as initial unreviewed patch-set-1 push capture because merged diffs/messages "
                    "can incorporate review revisions while retaining Git author_date. Empty diffs (`Qt: 262`, "
                    "`OpenStack: 10`) are preserved as empty PAD sequences without message substitution. "
                    "Retroactive SZZ links (`fix_ts < commit_ts`) are strictly quarantined to `+inf`."
                ),
                "Benchmark_B_Cabral19_ApacheJIT": (
                    "Archival Git author_date historical replay with Kamei CommitGuru change/diffusion/history/"
                    "author-experience metrics (0 review fields) and earliest SZZ bug-fixing commit reveal timestamps "
                    "(any `fix_ts < commit_ts` strictly quarantined to `+inf`)."
                ),
                "Benchmark_B_N60_PRSZZ": (
                    "Post-PR-review main-branch integration triage ordered by Git committer date (`commit_commitment_date`), "
                    "retaining Bludau & Pretschner (PROMISE 2022 / SQJ 2026) M1 AST + PR-workflow + reviewer metrics "
                    "computed up to the inner commit while explicitly stripping post-closure/target columns "
                    f"{EXCLUDED_N60_POSTCLOSURE_AND_TARGET_COLS} and converting post-2021H1 sentinels to `+inf`."
                ),
            },
            "benchmark_a": audit_a,
            "benchmark_b": audit_b,
        }
        out_audit = os.path.join(res_dir, "data_provenance_audit_v3.json")
        with open(out_audit, "w", encoding="utf-8") as f:
            json.dump(full_audit, f, indent=2)
        print(f"\n==> Saved Complete Provenance & Donor-Cluster Audit -> {out_audit}", flush=True)
    else:
        conf_payload = download_and_build_confirmatory_apache_v3(args.base_dir)
        conf_payload["provenance"] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "v3_config": json.loads(json.dumps(V3_CONFIG.to_dict())),
            "code_sha256": code_hashes,
        }
        out_conf = os.path.join(res_dir, "confirmatory_provenance_v3.json")
        with open(out_conf, "w", encoding="utf-8") as f:
            json.dump(conf_payload, f, indent=2)
        print(f"\n==> Saved Confirmatory Cohort Audit -> {out_conf}", flush=True)


if __name__ == "__main__":
    main()
