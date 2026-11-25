"""Manifest nguon - capture - session - thiet bi cho bo 186 nhan (2026-11-22). Chi doc du lieu co san, khong ghi de ket qua.

Dau ra (Dataset/features):
  capture_manifest_v6.csv     1 dong / capture (session): tep parquet, nhom nguon (suy tu TEN session, chua duyet), so thiet bi, so thiet bi co nhan,
                              thoi gian quan sat dau-cuoi, kich thuoc, sha256 (chi capture co it nhat 1 thiet bi co nhan), nhom trung lap
  label_manifest_186.csv      1 dong / thiet bi co nhan: MAC, lop, nguon nhan, do tin cay, so capture, nhom nguon, ghep tai san (IP/OUI), co ho so payload
  manifest_186_summary.json   tong hop va cac kiem tra (trung hash, trung chu ky, MAC ghep bang IP...)
Luu y: PCAP goc (2.230 tep) khong co tren may nay; hash tinh tren tep parquet da chuyen doi tu PCAP, khong phai tren PCAP goc.
Chay:  py -3.14 102_build_manifest_186.py [--nohash]
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent / "Dataset"
F = D / "features"
RAW = D / "raw_parquet"


def sha256(p, chunk=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    nohash = "--nohash" in sys.argv
    lab = pd.read_csv(F / "phase2_type_labels_4class_186_devices.csv", encoding="utf-8-sig")
    rev = pd.read_csv(F / "device_session_review_v6.csv", encoding="utf-8-sig", usecols=["device_id", "session_id", "capture_file", "device_mac", "device_ips", "packet_count", "first_observed", "last_observed"])
    smap = pd.read_csv(F / "session_source_map_draft.csv", encoding="utf-8-sig")[["session_id", "source_family", "basis"]]
    prof = pd.read_csv(F / "phase2_train4_preds_v6_186.csv", encoding="utf-8-sig") if (F / "phase2_train4_preds_v6_186.csv").exists() else None
    fa = pd.read_csv(F / "phase2_fold_assignments_186.csv", encoding="utf-8-sig")
    inv = pd.read_csv(F / "inventory_match_v6.csv", encoding="utf-8-sig", usecols=["device_id", "ip", "vendor", "confidence", "oui_matches_inventory_mac", "vendor_consistent", "Name", "Type"]).drop_duplicates("device_id").set_index("device_id")
    inv = inv.rename(columns={"confidence": "asset_match_confidence", "ip": "asset_ip", "vendor": "oui_vendor", "Name": "asset_name", "Type": "asset_type"})
    inA = set(fa.device_id)
    lab_ids = set(lab.device_id)

    # ---- capture manifest
    r = rev.merge(smap, on="session_id", how="left")
    r["is_labelled"] = r.device_id.isin(lab_ids)
    r["in_A"] = r.device_id.isin(inA)
    cap = r.groupby(["session_id", "capture_file"]).agg(n_devices=("device_id", "nunique"), n_labelled_devices=("is_labelled", "sum"), n_A_devices=("in_A", "sum"),
                                                        first_observed=("first_observed", "min"), last_observed=("last_observed", "max"), source_family=("source_family", "first"), source_basis=("basis", "first")).reset_index()
    iso = lambda t: datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(t) and 0 < float(t) < 4e9 else ""
    cap["first_observed_utc"] = cap.first_observed.map(iso)
    cap["last_observed_utc"] = cap.last_observed.map(iso)
    cap["duration_s"] = (cap.last_observed - cap.first_observed).round(1)
    cap["file_size_bytes"] = [(RAW / f).stat().st_size if (RAW / f).exists() else -1 for f in cap.capture_file]
    cap["sha256_parquet"] = ""
    need = cap.index[cap.n_labelled_devices > 0]
    print("captures", len(cap), "with labelled devices", len(need), "bytes to hash", int(cap.loc[need, "file_size_bytes"].clip(lower=0).sum() / 1e9), "GB", flush=True)
    if not nohash:
        t0 = time.time()
        for k, i in enumerate(need):
            f = RAW / cap.at[i, "capture_file"]
            if f.exists():
                cap.at[i, "sha256_parquet"] = sha256(f)
            if k % 50 == 0:
                print(f"hash {k}/{len(need)}  {time.time() - t0:.0f}s", flush=True)
    sig = cap[["n_devices", "first_observed", "last_observed"]].astype(str).agg("|".join, axis=1)
    cap["same_signature_group"] = sig.map(sig.value_counts())
    cap["unreviewed_note"] = "source_family inferred from the capture name only; no provenance file exists"
    cap.drop(columns=["first_observed", "last_observed"]).to_csv(F / "capture_manifest_v6.csv", index=False, encoding="utf-8-sig")

    # ---- label manifest
    g = r[r.is_labelled].groupby("device_id").agg(device_mac=("device_mac", "first"), n_captures=("session_id", "nunique"), first_capture=("session_id", "min"),
                                                    n_source_families=("source_family", "nunique"), source_families=("source_family", lambda s: ";".join(sorted(set(s.dropna())))))
    lm = lab.merge(g, left_on="device_id", right_index=True, how="left").merge(inv, left_on="device_id", right_index=True, how="left")
    lm["in_stage2_set_A_profiled"] = lm.device_id.isin(inA)
    lm["mapping_basis"] = lm.source.map(lambda s: "IP + OUI (MAC in asset table differs from capture MAC)" if s in ("customer_site_inventory",) else ("public 4SICS roster, IP + OUI" if s == "public_roster_4sics_netresec" else "device MAC seen in capture"))
    cols = ["device_id", "device_mac", "type4", "sub_type", "source", "confidence", "detail", "mapping_basis", "asset_match_confidence", "asset_ip", "oui_vendor", "oui_matches_inventory_mac", "n_captures", "first_capture", "n_source_families", "source_families", "in_stage2_set_A_profiled", "tier1_role", "role_evidence"]
    lm[cols].to_csv(F / "label_manifest_186.csv", index=False, encoding="utf-8-sig")

    # ---- summary
    out = {"n_labels": int(len(lm)), "n_captures_total": int(len(cap)), "n_captures_with_labelled_devices": int(len(need)),
           "labels_by_source_and_confidence": {f"{a}|{b}": int(n) for (a, b), n in lm.groupby(["source", "confidence"]).size().items()},
           "labels_missing_in_review_table": int(lm.n_captures.isna().sum()),
           "labelled_devices_in_one_capture_only": int((lm.n_captures == 1).sum()),
           "labelled_devices_median_captures": float(lm.n_captures.median()),
           "source_family_unassigned_captures": int((cap.source_family == "unassigned").sum()),
           "labelled_captures_by_family": cap.loc[need].source_family.fillna("NA").value_counts().to_dict(),
           "mapping_basis": lm.mapping_basis.value_counts().to_dict(),
           "asset_match_confidence_of_asset_labels": lm[lm.source.isin(["customer_site_inventory", "lab_inventory"])].asset_match_confidence.fillna("none").value_counts().to_dict()}
    if not nohash:
        h = cap.loc[need, "sha256_parquet"]
        h = h[h != ""]
        out["hash_checked_captures"] = int(len(h))
        out["duplicate_hash_groups"] = int((h.value_counts() > 1).sum())
        out["captures_in_duplicate_hash_groups"] = int(h[h.duplicated(keep=False)].shape[0])
    out["captures_sharing_signature_with_another_capture"] = int((cap.same_signature_group > 1).sum())
    (F / "manifest_186_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
