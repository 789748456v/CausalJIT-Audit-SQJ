"""
Versioned Causal Per-Commit `microsoft/codebert-base` Feature Extractor (N05 V3, `R3b-M1`).

Fixes legacy `src/cache_codebert_embeddings.py:93` (`real_diffs.get(cid, "") or str(m)`), which
copied the commit message into `code_emb` whenever a commit's unified diff was empty (`Qt: 262`,
`OpenStack: 10`).

V3 Missing-Modality Policy:
  - `codes = [str(real_diffs.get(str(cid), "")) for cid in ids]` — empty diffs remain `""` (`<s></s>`),
    NEVER substituted with the commit message!
  - Saves `cache/codebert_v3/codebert_{openstack,qt}_v3.npz` + `manifest_codebert_v3.json` recording
    exact `torch` / `transformers` versions, `microsoft/codebert-base` config hash, SHA-256 hashes of
    `commit_ids`, `msgs`, and `codes`, missing-diff counts, and verification that all empty-diff commits
    share the canonical empty-diff embedding (`unique_empty_code_vectors == 1`).
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def sha256_str_list(items: List[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        b = s.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


@torch.no_grad()
def encode_texts_fp16(
    texts: List[str],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 256,
) -> np.ndarray:
    n = len(texts)
    out = np.zeros((n, 768), dtype=np.float16)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        enc = tokenizer(
            texts[s:e],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attn_mask = enc["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            hs = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            cls_v = hs[:, 0, :]
            mask_f = attn_mask.unsqueeze(-1).float()
            mean_v = (hs.float() * mask_f).sum(dim=1) / torch.clamp(mask_f.sum(dim=1), min=1.0)
            pooled = 0.5 * cls_v.float() + 0.5 * mean_v
        out[s:e] = pooled.detach().cpu().numpy().astype(np.float16)
    return out


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dj_dir = os.path.join(base_dir, "data_sync", "deepjit")
    out_dir = os.path.join(base_dir, "cache", "codebert_v3")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "microsoft/codebert-base"
    print(f"[CodeBERT-V3] Loading {model_id} on {device}...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device)
    model.eval()
    cfg_json = json.dumps(model.config.to_dict(), sort_keys=True)
    cfg_sha = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    print(f"[CodeBERT-V3] Loaded in {time.perf_counter() - t0:.2f}s (config_sha256={cfg_sha[:16]}...)", flush=True)

    # Encode canonical empty string `""` once so we can verify exact empty-diff vector identity
    empty_ref_vec = encode_texts_fp16([""], tokenizer, model, device, max_length=192, batch_size=1)[0]

    manifest_datasets: Dict[str, Dict] = {}
    for ds_name in ("openstack", "qt"):
        ts0 = time.perf_counter()
        with open(os.path.join(dj_dir, f"{ds_name}_train.pkl"), "rb") as f:
            id_tr, y_tr, m_tr, _ = pickle.load(f)
        with open(os.path.join(dj_dir, f"{ds_name}_test.pkl"), "rb") as f:
            id_te, y_te, m_te, _ = pickle.load(f)

        ids_list = [str(x) for x in (list(id_tr) + list(id_te))]
        ids = np.array(ids_list)
        ys = np.array(list(y_tr) + list(y_te), dtype=np.int32)
        msgs = [str(m) for m in (list(m_tr) + list(m_te))]

        diff_gz = os.path.join(base_dir, "data_sync", "szz_linked", f"real_diffs_{ds_name}.json.gz")
        with gzip.open(diff_gz, "rt", encoding="utf-8") as f:
            real_diffs = json.load(f)

        # STRICT V3 MISSING-MODALITY POLICY: preserve empty diff as `""` (ZERO `or str(m)` fallback!)
        codes: List[str] = []
        missing_diff_mask = np.zeros(len(ids_list), dtype=bool)
        for idx_i, cid in enumerate(ids_list):
            raw_d = str(real_diffs.get(cid, ""))
            if not raw_d.strip():
                codes.append("")
                missing_diff_mask[idx_i] = True
            else:
                codes.append(raw_d)

        print(
            f"  [{ds_name}] Encoding {len(ids_list)} commits (missing_diffs={int(np.sum(missing_diff_mask))}) on {device}...",
            flush=True,
        )
        msg_emb = encode_texts_fp16(msgs, tokenizer, model, device, max_length=96, batch_size=256)
        code_emb = encode_texts_fp16(codes, tokenizer, model, device, max_length=192, batch_size=256)

        # Enforce exact bit-identical `empty_ref_vec` across all `missing_diff_mask` rows (eliminating fp16 batch-padding drift)
        code_emb[missing_diff_mask] = empty_ref_vec

        # Verify empty-diff `code_emb` is 1 unique vector and NOT equal to `msg_emb`!
        emp_c = code_emb[missing_diff_mask].astype(np.float64)
        emp_m = msg_emb[missing_diff_mask].astype(np.float64)
        uniq_emp = np.unique(code_emb[missing_diff_mask], axis=0)
        assert len(uniq_emp) == 1, f"Expected 1 canonical empty-diff vector, got {len(uniq_emp)}"
        cos_sim = np.sum(emp_c * emp_m, axis=1) / np.maximum(np.linalg.norm(emp_c, axis=1) * np.linalg.norm(emp_m, axis=1), 1e-12)
        med_cos = float(np.median(cos_sim))
        assert med_cos < 0.995, f"Empty-diff code_emb still matches msg_emb (median cos={med_cos})!"

        out_path = os.path.join(out_dir, f"codebert_{ds_name}_v3.npz")
        np.savez_compressed(
            out_path,
            commit_ids=ids,
            y_deepjit=ys,
            missing_diff_mask=missing_diff_mask,
            msg_emb=msg_emb,
            code_emb=code_emb,
        )
        file_sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        dt = time.perf_counter() - ts0
        manifest_datasets[ds_name] = {
            "npz_file": f"codebert_{ds_name}_v3.npz",
            "npz_sha256": file_sha,
            "n_commits": int(len(ids_list)),
            "n_missing_diffs": int(np.sum(missing_diff_mask)),
            "unique_empty_diff_code_vectors": int(len(uniq_emp)),
            "empty_diff_code_vs_msg_median_cosine": round(med_cos, 6),
            "commit_ids_sha256": sha256_str_list(ids_list),
            "msgs_text_sha256": sha256_str_list(msgs),
            "codes_text_sha256": sha256_str_list(codes),
            "elapsed_sec": round(dt, 2),
        }
        print(
            f"  [{ds_name}] Saved {out_path} in {dt:.1f}s | missing_diffs={int(np.sum(missing_diff_mask))}, "
            f"unique_empty_vecs={len(uniq_emp)}, empty_code_msg_cos={med_cos:.4f}",
            flush=True,
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_model_id": model_id,
        "encoder_config_sha256": cfg_sha,
        "pooling_rule": "0.5 * CLS + 0.5 * masked_token_mean (msg_max_len=96, code_max_len=192, fp16)",
        "missing_modality_policy": "preserve_empty_string_no_message_fallback",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets": manifest_datasets,
    }
    man_path = os.path.join(out_dir, "manifest_codebert_v3.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"==> Saved Versioned CodeBERT V3 Manifest -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
