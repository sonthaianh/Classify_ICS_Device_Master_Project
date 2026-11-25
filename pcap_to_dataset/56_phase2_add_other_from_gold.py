"""Phase 2 - them vi du OTHER (scanner / may IT) tu gold non-OT ma Phase 1 gan nham OT (2026-11-10, chu de tai dong y).

Nguon: gold_union_v6.csv (non-OT) join diem OOF Phase 1 (phase1_oof_scores_v6.csv): thiet bi non-OT co diem >= 0,5 = 33 thiet bi.
Tat ca da nam trong danh sach Phase 2 (phase2_candidate_pool_v6.csv) nhung chua co nhan 4 lop.
Dau vao : phase2_type_labels_4class_v6.csv (script 55). Dau ra: phase2_type_labels_4class_plus_v6.csv + _summary.json.
Khong sua file goc. Nhan moi la nhan CHUA chuyen gia xac nhan: giu `gold_source` de bao cao tach rieng.
Chay: py -3.14 pcap_to_dataset/56_phase2_add_other_from_gold.py
"""
import json
from pathlib import Path

import pandas as pd

F = Path(__file__).resolve().parents[1] / "Dataset" / "features"
rd = lambda n: pd.read_csv(F / n, encoding="utf-8-sig")
THR = 0.5

base = rd("phase2_type_labels_4class_v6.csv")
gold = rd("gold_union_v6.csv").set_index("device_id")
oof = rd("phase1_oof_scores_v6.csv").set_index("device_id")
pool = set(rd("phase2_candidate_pool_v6.csv")["device_id"])
roles = set(rd("phase2_role_tier1_v6.csv")["device_id"])

g = gold[gold.gold_label == "non-OT"].join(oof[["lightgbm"]])
fp = g[g.lightgbm >= THR].copy()

# ---- kiem tra dau vao (dung nhu mo ta) ----
assert len(fp) == 33, len(fp)
assert fp.index.isin(pool).all(), "co thiet bi khong nam trong danh sach Phase 2"
assert not fp.index.isin(base.device_id).any(), "co thiet bi da co nhan 4 lop"
assert base.device_id.is_unique

conf = {"lab_inventory": "high", "blind_sheet_reviewed_sonthai": "medium", "capture_scenario_provisional": "low"}
new = pd.DataFrame({
    "device_id": fp.index,
    "type4": "OTHER",
    "sub_type": "FP_scanner_or_IT",
    "source": "gold_nonOT_phase1_false_positive",
    "confidence": [conf[s] for s in fp.source],
    "detail": ["gold non-OT (" + s + "), Phase 1 score " + format(v, ".2f") for s, v in zip(fp.source, fp.lightgbm)],
    "tier1_role": None, "role_evidence": None,
})
new["gold_source"] = fp.source.values
base2 = base.copy()
base2["gold_source"] = None
out = pd.concat([base2, new], ignore_index=True)

# ---- tu kiem dau ra ----
assert out.device_id.is_unique and len(out) == len(base) + 33
counts = out.type4.value_counts().reindex(["PLC", "HMI", "EWS", "OTHER"]).astype(int).to_dict()
assert counts == {"PLC": 40, "HMI": 11, "EWS": 15, "OTHER": 56}, counts
assert (base.set_index("device_id").type4 == out.set_index("device_id").loc[base.device_id, "type4"]).all(), "nhan cu bi doi"

out.to_csv(F / "phase2_type_labels_4class_plus_v6.csv", index=False, encoding="utf-8-sig")
summary = {
    "total": int(len(out)), "counts": counts, "added": 33,
    "added_by_gold_source": fp.source.value_counts().to_dict(),
    "added_with_role_profile": int(fp.index.isin(roles).sum()),
    "added_without_role_profile": int((~fp.index.isin(roles)).sum()),
    "meets_minimum_each_class_ge_10_total_ge_100": bool(out.shape[0] >= 100 and min(counts.values()) >= 10),
    "note": "33 nhan OTHER moi: 18 sonthai ra soat + 13 capture mau tam thoi + 2 lab; chua chuyen gia xac nhan",
}
(F / "phase2_type_labels_4class_plus_summary_v6.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK")
