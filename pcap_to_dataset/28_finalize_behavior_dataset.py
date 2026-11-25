"""Finalize v6 from the per-capture cache written by 26_build_behavior_dataset.py.

Why: the final in-memory merge in script 26 (windows x repeated evidence columns for 768,927
windows) needs ~71 GB. This script performs the same merge session by session and streams the
result to one Parquet file. Script 26 is left untouched, so its hash and the cache stay valid.

Same checks, same output names and same manifest fields as 26 (validated by 27).
Never assigns labels. Refuses to run if inputs changed since the build.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# mimalloc (pyarrow default on Windows) keeps freed pages; the system pool returns them to the OS.
pa.set_memory_pool(pa.system_memory_pool())
HERE = Path(__file__).resolve().parent


def rss_mb():
    """Working set of this process in MB (Windows psapi; -1 if unavailable)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize // 2**20 if ok else -1
    except Exception:
        return -1


spec = importlib.util.spec_from_file_location("behavior_builder", HERE / "26_build_behavior_dataset.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

ROOT, KEY, W = B.ROOT, B.KEY, B.W
TRAIN, EXPER = B.TRAINING_FEATURES, B.EXPERIMENTAL_FEATURES
DROP_FROM_EVIDENCE = {"packet_count", "observed_seconds", "first_observed", "last_observed", "feature_version"}
# Per device/session lists: kept only in the review table (one row per pair), never replicated per window.
LONG_LIST_COLS = ["device_ips", "peer_ips", "peer_macs", "service_ports", "evidence_packet_refs", "identity_packet_refs"]
COUNT_COLS = {"n_device_ips": "device_ips", "n_peer_ips": "peer_ips", "n_peer_macs": "peer_macs", "n_service_ports": "service_ports"}


def json_list_len(series):
    """Element count of a JSON-list string without parsing (elements are quoted strings)."""
    return series.map(lambda x: 0 if x in ("", "[]") else x.count('", "') + 1).astype("int64")


LABEL_COLS = {"label", "label_reason", "label_quality", "review_status", "reviewer", "reviewed_at"}


def unified(paths):
    schemas = [pq.read_schema(p) for p in paths]
    return pa.unify_schemas(schemas, promote_options="permissive")


def fix_null(t):
    return pa.large_string() if pa.types.is_null(t) else t


def main():
    features = ROOT / "Dataset/features"
    out = features / "v6_behavior_build"
    registry_path = ROOT / "Dataset/metadata/devices.csv"
    run_config = json.loads((out / "build_config.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / "captures" / config_hash
    if not cache.is_dir():
        raise RuntimeError(f"Cache directory not found: {cache}")
    # Refuse if anything the build depended on changed.
    if B.sha256(HERE / "26_build_behavior_dataset.py") != run_config["script_sha256"]:
        raise RuntimeError("Script 26 changed since the build; cache no longer matches")
    if B.sha256(registry_path) != run_config["registry_sha256"]:
        raise RuntimeError("devices.csv changed since the build")
    pairs = B.behavior_pairs(registry_path)
    if hashlib.sha256(pairs.to_csv(index=False).encode()).hexdigest() != run_config["pairs_sha256"]:
        raise RuntimeError("device_flows pairs changed since the build")
    sessions = sorted(pairs.session_id.unique())
    missing = [s for s in sessions if not all((cache / (s + x)).exists() for x in (".windows.parquet", ".review.parquet", ".audit.json"))]
    if missing:
        raise RuntimeError(f"{len(missing)} captures not built yet, e.g. {missing[:3]}")

    review_path = features / "device_session_review_v6.csv"
    master_path = features / "master_device_windows_v6_behavior.parquet"
    if review_path.exists():
        old = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        edited = old.get("review_status", pd.Series("pending", index=old.index)).ne("pending")
        for field in ["label", "label_reason", "reviewer", "reviewed_at", "proposed_label"]:
            if field in old:
                edited |= old[field].str.strip().ne("")
        if edited.any():
            raise RuntimeError("Review file contains human work; preserve it and use a new version")

    wschema = unified([cache / (s + ".windows.parquet") for s in sessions])
    rschema = unified([cache / (s + ".review.parquet") for s in sessions])
    evidence_cols = [c for c in rschema.names if c not in DROP_FROM_EVIDENCE and c not in LONG_LIST_COLS]
    order = list(wschema.names) + [c for c in evidence_cols if c not in wschema.names] + list(COUNT_COLS)
    target = pa.schema([pa.field(c, pa.int64() if c in COUNT_COLS else fix_null(wschema.field(c).type if c in wschema.names else rschema.field(c).type)) for c in order])

    tmp = master_path.with_suffix(".tmp")
    writer = pq.ParquetWriter(tmp, target)
    review_keys, preview_parts = [], []
    review_rows = 0
    rows = 0
    devices, miss = set(), pd.Series(0, index=TRAIN + EXPER, dtype="int64")
    seen_pairs = 0
    try:
        for i, s in enumerate(sessions, 1):
            w = pd.read_parquet(cache / (s + ".windows.parquet"))
            r = pd.read_parquet(cache / (s + ".review.parquet"))
            r = r.reindex(columns=rschema.names)
            r.to_csv(review_path, mode="w" if i == 1 else "a", header=(i == 1), index=False,
                     encoding="utf-8-sig" if i == 1 else "utf-8")
            review_keys.append(r[["device_id", "session_id"]])
            review_rows += len(r)
            rc = r[evidence_cols].copy()
            for new, src in COUNT_COLS.items():
                rc[new] = json_list_len(r[src])
            m = w.merge(rc, on=["device_id", "session_id"], how="left", validate="many_to_one")
            if len(m) != len(w) or m.duplicated(KEY).any() or r.duplicated(["device_id", "session_id"]).any():
                raise ValueError(f"Duplicate/lost keys in {s}")
            if m.label.fillna("").ne("").any():
                raise ValueError("Label assignment is outside this build")
            if np.isinf(m[TRAIN].to_numpy(float)).any():
                raise ValueError("Infinite training feature")
            if (m.flow_dur_median.dropna() > W + 1e-6).any():
                raise ValueError("Duration outside window")
            writer.write_table(pa.Table.from_pandas(m[order], preserve_index=False).cast(target))
            rows += len(m)
            devices.update(m.device_id.unique())
            miss += m[TRAIN + EXPER].isna().sum()
            if seen_pairs < 100:
                p = m.drop_duplicates(["device_id", "session_id"]).head(100 - seen_pairs)
                preview_parts.append(p)
                seen_pairs += len(p)
            del w, r, rc, m
            gc.collect()
            if i % 50 == 0 or i == len(sessions):
                print(f"merged {i}/{len(sessions)} sessions, {rows:,} rows | rss {rss_mb()} MB | arrow {pa.total_allocated_bytes() // 2**20} MB", flush=True)
    finally:
        writer.close()
    tmp.replace(master_path)

    review = pd.concat(review_keys, ignore_index=True)
    if review.duplicated(["device_id", "session_id"]).any():
        raise ValueError("Duplicate review keys")
    coverage = review.merge(pairs, on=["device_id", "session_id"], how="outer", indicator=True)
    if coverage._merge.ne("both").any():
        raise ValueError("Device/session coverage differs from v6 scope (device_flows pairs); master written but NOT valid")
    pd.concat(preview_parts).to_csv(features / "preview_100_device_windows_v6.csv", index=False, encoding="utf-8-sig")
    column_roles = {c: "training_feature" if c in TRAIN else "experimental_not_training" if c in EXPER else "label_not_input" if c in LABEL_COLS else "metadata_not_training" for c in order}
    manifest = {**run_config, "rows": rows, "devices": len(devices), "sessions": len(sessions),
                "review_rows": review_rows, "columns": column_roles, "training_features": TRAIN,
                "labeling": "deferred; all label fields empty; legacy labels are metadata only",
                "scope": "devices.csv include_in_behavior_dataset=True (>=1 flow, not spoofed); pairs from processed/device_flows, __l2 merged into raw session; includes packet-only windows",
                "feature_semantics": "packet timestamps exact; per-window observed conversations (5tuple+VLAN, no idle timeout); duration=max(timestamp)-min(timestamp) within window; small/oneway within window; carryover=earlier observed packet in same conversation; roles as of window end",
                "raw_counter_policy": "raw frame counters include capture duplicates; flow features use directed packet duplicate policy",
                "missing_policy": "undefined ratios and no-flow features remain NaN; counts are zero only if observed absence",
                "limitations": ["same MAC across sources not verified as one physical asset", "protocol sampling is not full dissection", "synthetic frames may limit TCP protocol detection", "OT port list is heuristic; never a label", "IPv6 header evidence limited by legacy raw parser", "window rows from same conversation must not be randomly split"],
                "missing_rates": (miss / rows).to_dict(),
                "source_pair_coverage": len(coverage), "expected_source_pairs": len(pairs),
                "finalized_by": "28_finalize_behavior_dataset.py (streamed per-session merge; same checks as 26)",
                "master_excludes_long_lists": LONG_LIST_COLS + [" -> use device_session_review_v6.csv (join on device_id, session_id); master has n_* counts instead"],
                "output_sha256": B.sha256(master_path), "review_sha256": B.sha256(review_path)}
    (features / "feature_manifest_v6.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {rows:,} rows, {review_rows:,} review pairs, {len(order)} columns, {len(devices):,} devices -> {master_path}", flush=True)


if __name__ == "__main__":
    main()
