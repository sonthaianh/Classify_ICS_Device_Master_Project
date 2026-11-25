"""Phase 2 - thu nho taxonomy con 4 lop: PLC / HMI / EWS / OTHER (2026-11-10, theo yeu cau chu de tai).

Chi anh xa lai bang nhan da co (phase2_type_labels_v6.csv); KHONG tao nhan moi, KHONG sua file goc.
IED, IO va OTHER cu gop thanh OTHER; giu cot `sub_type` de truy nguoc. Nhan EXCLUDED_* giu nguyen (khong tinh).
Chay: py -3.14 Tools/pcap_to_dataset/55_phase2_taxonomy4.py
"""
import json
from pathlib import Path

import pandas as pd

FEAT = Path(__file__).resolve().parents[1] / "Dataset" / "features"
SRC = FEAT / "phase2_type_labels_v6.csv"
OUT_CSV = FEAT / "phase2_type_labels_4class_v6.csv"
OUT_JSON = FEAT / "phase2_type_labels_4class_summary_v6.json"

MAP = {"PLC": "PLC", "HMI": "HMI", "EWS": "EWS", "IED": "OTHER", "IO": "OTHER", "OTHER": "OTHER"}

# Gan them 2026-11-10: chu de tai duyet 5/6 ca bi loai (client + chuoi ma hang bo dieu khien) la EWS,
# dua tren ten session (STEP7/TIA/RSLogix5000/backup/download/firmware/clock write-read). Chua chuyen gia xac nhan.
# DEV_0226 (SEL RTAC, DNP3 master) KHONG duoc gan - giu loai.
EWS_ADDED = {
    "DEV_3229": "session S1930",
    "DEV_0160": "sessions backup/connect/download_reset_plc_program/firmware_update",
    "DEV_0235": "sessions S7commUploadBlock/tia downloadHwConfig/downloadOb1/flashLed",
    "DEV_0091": "sessions RsLogix5000 change_mode program/run",
    "DEV_0089": "sessions Clock_Write-Read/Cycle_time_read (confidence thap hon)",
}

d = pd.read_csv(SRC, encoding="utf-8-sig")
d["sub_type"] = d["type"]
d["type4"] = d["type"].map(MAP)  # EXCLUDED_* -> NaN
for dev, why in EWS_ADDED.items():
    m = d["device_id"] == dev
    assert m.sum() == 1 and str(d.loc[m, "type"].iloc[0]).startswith("EXCLUDED"), dev
    d.loc[m, "type4"] = "EWS"
    d.loc[m, "source"] = "owner_approved_session_evidence"
    d.loc[m, "detail"] = d.loc[m, "detail"].astype(str) + " || owner-approved EWS: " + why
usable = d[d["type4"].notna()].copy()

counts = usable["type4"].value_counts().reindex(["PLC", "HMI", "EWS", "OTHER"]).fillna(0).astype(int)
by_src = usable.groupby(["type4", "source"]).size().unstack(fill_value=0)
summary = {
    "total_usable": int(len(usable)),
    "excluded": int(d["type4"].isna().sum()),
    "counts": counts.to_dict(),
    "other_breakdown": usable.loc[usable.type4 == "OTHER", "sub_type"].value_counts().to_dict(),
    "by_source": {k: {s: int(v) for s, v in row.items()} for k, row in by_src.iterrows()},
    "bar_total_100_each_class_ge_10": bool(len(usable) >= 100 and (counts >= 10).all()),
    "each_class_ge_10": bool((counts >= 10).all()),
}
cols = ["device_id", "type4", "sub_type", "source", "confidence", "detail", "tier1_role", "role_evidence"]
usable[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

# tu kiem
assert len(usable) == len(d) - summary["excluded"]
assert usable["device_id"].is_unique
assert sum(summary["counts"].values()) == len(usable)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK ->", OUT_CSV.name)
