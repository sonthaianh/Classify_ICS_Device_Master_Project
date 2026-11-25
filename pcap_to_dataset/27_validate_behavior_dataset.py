"""Validate complete v6 (expanded behaviour scope) artifacts; no model fit, no labels.

Copy of 24_validate_review_dataset.py adapted to 26_build_behavior_dataset.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.util
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from feature_contract import TRAINING_FEATURES, EXPERIMENTAL_FEATURES, select_training_features

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "Dataset/features"
KEY = ["device_id", "session_id", "window_start"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    master_path = FEATURES / "master_device_windows_v6_behavior.parquet"
    review_path = FEATURES / "device_session_review_v6.csv"
    manifest = json.loads((FEATURES / "feature_manifest_v6.json").read_text(encoding="utf-8"))
    require(sha256(master_path) == manifest["output_sha256"], "Master hash mismatch")
    require(sha256(review_path) == manifest["review_sha256"], "Review hash mismatch")
    require(sha256(ROOT / "Dataset/metadata/devices.csv") == manifest["registry_sha256"], "Registry changed during/after build")
    builder_path = Path(__file__).with_name("26_build_behavior_dataset.py")
    require(sha256(builder_path) == manifest["script_sha256"], "Builder differs from recorded version")
    spec = importlib.util.spec_from_file_location("behavior_builder", builder_path)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    base_pairs = builder.behavior_pairs(ROOT / "Dataset/metadata/devices.csv")
    require(hashlib.sha256(base_pairs.to_csv(index=False).encode()).hexdigest() == manifest["pairs_sha256"], "device_flows pairs changed after build")
    registry = pd.read_csv(ROOT / "Dataset/metadata/devices.csv", dtype=str, keep_default_na=False)
    in_scope = set(registry.loc[registry.include_in_behavior_dataset.eq("True"), "device_id"])
    spoofed = set(registry.loc[registry.spoofed_identity.eq("True"), "device_id"])
    columns = KEY + TRAINING_FEATURES + EXPERIMENTAL_FEATURES + ["packet_count", "pkt_alloc", "flows_active", "has_flow_packets", "window_end"]
    data = pd.read_parquet(master_path, columns=columns)
    review = pd.read_csv(review_path, keep_default_na=False, dtype=str)
    require(not data.duplicated(KEY).any(), "Duplicate window keys")
    require(not review.duplicated(["device_id", "session_id"]).any(), "Duplicate review keys")
    require(len(data) == manifest["rows"], "Row count differs from manifest")
    require(list(select_training_features(data).columns) == TRAINING_FEATURES, "Training contract changed")
    require(manifest["training_features"] == TRAINING_FEATURES, "Manifest training allowlist mismatch")
    for c in TRAINING_FEATURES + EXPERIMENTAL_FEATURES:
        values = data[c].dropna().to_numpy(float)
        require(np.isfinite(values).all(), f"Non-finite values in {c}")
        require((values >= 0).all(), f"Negative values in {c}")
        if c.endswith("ratio") or c == "ot_pkt_share":
            require((values <= 1 + 1e-12).all(), f"Ratio outside [0,1]: {c}")
    require(data.flow_dur_median.dropna().lt(45 + 1e-6).all(), "Duration exceeds window")
    require(data.window_end.sub(data.window_start).eq(45).all(), "Window size mismatch")
    require(data.window_start.mod(45).eq(0).all(), "Window grid mismatch")
    require(np.allclose(data.pkt_alloc, np.round(data.pkt_alloc)), "Packet counts are still interpolated")
    require(data.loc[~data.has_flow_packets, "pkts_per_flow"].isna().all(), "No-flow windows imputed silently")
    require(data.has_flow_packets.eq(data.flows_active.gt(0)).all(), "Flow eligibility flag mismatch")
    for c in ["label", "proposed_label", "label_reason", "reviewer", "reviewed_at"]:
        require(review[c].eq("").all(), f"Unexpected label/reviewer assignment: {c}")
    require(review.review_status.eq("pending").all(), "Unexpected review status")
    require(review.label_quality.eq("unlabeled").all(), "Unexpected label quality")
    # Master labels must also be empty, not merely the exported CSV.
    master_labels = pd.read_parquet(master_path, columns=["label", "review_status"])
    require(master_labels.label.eq("").all() and master_labels.review_status.eq("pending").all(), "Master contains assigned labels")
    require(set(data.device_id) == in_scope, "Device set differs from include_in_behavior_dataset")
    require(not (set(data.device_id) & spoofed), "Spoofed identity present in dataset")
    coverage = review[["device_id", "session_id"]].merge(base_pairs, how="outer", indicator=True)
    require(coverage._merge.eq("both").all(), "Missing/extra source device-session pairs")
    rolled = data.groupby(["device_id", "session_id"]).agg(packet_count=("packet_count", "sum"), n_windows=("window_start", "size")).reset_index()
    checked = review[["device_id", "session_id", "packet_count", "n_windows"]].copy()
    checked[["packet_count", "n_windows"]] = checked[["packet_count", "n_windows"]].astype("int64")
    checked = checked.merge(rolled, on=["device_id", "session_id"], suffixes=("_review", "_windows"), validate="one_to_one")
    require(checked.packet_count_review.eq(checked.packet_count_windows).all(), "Review packet totals differ from windows")
    require(checked.n_windows_review.eq(checked.n_windows_windows).all(), "Review window totals differ")
    config = json.loads((FEATURES / "v6_behavior_build/build_config.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cache = FEATURES / "v6_behavior_build/captures" / config_hash
    audits = [json.loads(p.read_text(encoding="utf-8")) for p in cache.glob("*.audit.json")]
    require(len(audits) == data.session_id.nunique(), "Capture audit count mismatch")
    require(sum(a["windows"] for a in audits) == len(data), "Audit window totals differ")
    schema = pq.ParquetFile(master_path).schema_arrow
    report = {
        "status": "passed", "rows": len(data), "columns": len(schema),
        "devices": data.device_id.nunique(), "captures": len(audits), "review_rows": len(review),
        "raw_packets_scanned": sum(a["raw_packets"] for a in audits),
        "scoped_packets": sum(a["scoped_packets"] for a in audits),
        "duplicate_flow_packets_excluded": sum(a["duplicate_flow_packets_excluded"] for a in audits),
        "invalid_timestamp_packets_excluded": sum(a["invalid_timestamps"] for a in audits),
        "packet_only_windows": int((~data.has_flow_packets).sum()),
        "model_identity_review_rows": int(review.model.ne("[]").sum()),
        "protocol_evidence_review_rows": int(review.protocols_seen.ne("[]").sum()),
        "training_features": TRAINING_FEATURES,
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__},
        "support_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__).with_name("feature_contract.py"), ROOT / "utils/frame_builder.py",
            ROOT / "utils/tshark_dpi.py", Path(__file__)]},
        "column_types": {field.name: str(field.type) for field in schema},
        "feature_summary": {c: {"missing": int(data[c].isna().sum()), "min": float(data[c].min()) if data[c].notna().any() else None,
                                "max": float(data[c].max()) if data[c].notna().any() else None} for c in TRAINING_FEATURES + EXPERIMENTAL_FEATURES},
        "verified": ["source files unchanged", "unique keys", "complete device_flows device/session coverage", "device set == include_in_behavior_dataset", "no spoofed MAC",
                     "exact integer packet counts", "bounded duration/ratios", "no-flow NaN preserved",
                     "metadata excluded from D17", "all new labels empty", "review/window totals agree", "output hashes"],
        "not_claimed": ["label correctness", "physical identity across sites", "full protocol dissection", "model accuracy or calibration"],
    }
    out = FEATURES / "feature_quality_report_v6.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "rows", "columns", "devices", "captures", "review_rows", "raw_packets_scanned", "packet_only_windows"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
