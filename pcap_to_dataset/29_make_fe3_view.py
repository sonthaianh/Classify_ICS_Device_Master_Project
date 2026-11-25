"""FE-3 derived view of v6 (v6 itself is never modified).

Adds, without dropping any row or changing any raw D17 value:
  * label / evidence_tier / flood_like from device_labels_v6_rule.csv (rule labels, weak)
  * n_windows_device, n_sessions_device
  * cap_rank: deterministic 0..n-1 order inside each device, round-robin across sessions
    (first window of every session, then the second of every session, ...), so
    `cap_rank < K` keeps at most K windows per device AND keeps session diversity.
    K is chosen at training time (a hyper-parameter), not frozen here.
  * log1p_<feature> for heavy-tailed count/rate D17 features (stateless transform, no fitted
    statistics, so no train/test leakage; NaN stays NaN). Raw columns stay untouched.
Scaling/normalisation must be fitted on the training split only (not done here).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_contract import TRAINING_FEATURES  # noqa: E402

F = HERE.parent / "Dataset/features"
KEY = ["device_id", "session_id", "window_start"]
LOG_COLS = ["pkts_per_flow", "bytes_per_packet", "packet_rate", "unique_local_port_count", "svc_port_unique",
            "frames_sent", "arp_frame_count", "arp_target_unique", "icmp_frame_count", "nonip_frame_count", "ip_count"]
SEED = 20261108


def main():
    assert set(LOG_COLS) <= set(TRAINING_FEATURES)
    m = pd.read_parquet(F / "master_device_windows_v6_behavior.parquet", columns=KEY + TRAINING_FEATURES)
    rows = len(m)
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False)
    lab["flood_like"] = lab.flood_like.eq("True")
    m = m.merge(lab[["device_id", "label", "evidence_tier", "flood_like", "label_quality"]], on="device_id", how="left", validate="many_to_one")
    assert len(m) == rows and m.label.notna().all()
    m["n_windows_device"] = m.groupby("device_id").device_id.transform("size")
    m["n_sessions_device"] = m.groupby("device_id").session_id.transform("nunique")
    rng = np.random.default_rng(SEED)
    m["_r"] = rng.random(rows)
    m = m.sort_values(["device_id", "session_id", "_r"], kind="mergesort")
    m["_k"] = m.groupby(["device_id", "session_id"]).cumcount()
    m = m.sort_values(["device_id", "_k", "_r"], kind="mergesort")
    m["cap_rank"] = m.groupby("device_id").cumcount().astype("int32")
    m = m.drop(columns=["_r", "_k"]).sort_values(KEY).reset_index(drop=True)
    for c in LOG_COLS:
        assert (m[c].dropna() >= 0).all(), c
        m["log1p_" + c] = np.log1p(m[c])
    out = F / "master_device_windows_v6_fe3.parquet"
    m.to_parquet(out, index=False)
    caps = {k: int((m.cap_rank < k).sum()) for k in (10, 20, 50, 100)}
    manifest = {"source": "master_device_windows_v6_behavior.parquet (unchanged)", "rows": rows,
                "devices": int(m.device_id.nunique()), "seed": SEED, "log1p_columns": LOG_COLS,
                "cap_rows_by_K": caps, "labels": "rule-based weak labels from device_labels_v6_rule.csv",
                "notes": ["no row dropped; choose K via cap_rank < K at training time",
                          "raw D17 values untouched; log1p_* are extra columns, use one version per model",
                          "fit scalers on train split only; split by device/source group, never by row",
                          "evidence_tier / flood_like / label_quality are metadata, not model inputs"]}
    (F / "feature_manifest_v6_fe3.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
