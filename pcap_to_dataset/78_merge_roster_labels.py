"""Hop nhat nhan tu danh sach 4SICS vao BAN GOLD MOI (khong ghi de file dong bang) (2026-11-12).

Dau vao : gold_union_v6.csv + gold_roster_add_v6.csv ; phase2_type_labels_4class_plus_v6.csv + phase2_roster_add_v6.csv (script 77)
Dau ra  : gold_union_v6_plus_4sics.csv ; phase2_type_labels_4class_plus_v6_plus_4sics.csv ; phase1_gold_plus_4sics_eval.json
Cac file goc (gold_union_v6.csv, phase2_type_labels_4class_plus_v6.csv) KHONG doi, nen FREEZE va 38_phase1_freeze.py --verify van dung.
De cac script (36,37,39...) dung ban moi: tro ten file gold trong script, hoac copy ban moi thanh ten cu SAU KHI cap nhat FREEZE.

  py -3.14 78_merge_roster_labels.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"


def metrics(g, oof):
    y = (g.gold_label == "OT").to_numpy()
    s = oof.loc[g.device_id].to_numpy()
    pred = s >= 0.5
    return {"n": int(len(g)), "ot": int(y.sum()), "non_ot": int((~y).sum()), "roc_auc": round(float(roc_auc_score(y, s)), 4), "pr_auc": round(float(average_precision_score(y, s)), 4),
            "recall_ot@0.5": round(float((pred & y).sum() / y.sum()), 4), "non_ot_flagged@0.5": round(float((pred & ~y).sum() / (~y).sum()), 4)}


def main():
    g0 = pd.read_csv(F / "gold_union_v6.csv", dtype=str)
    ga = pd.read_csv(F / "gold_roster_add_v6.csv", dtype=str)
    assert not set(g0.device_id) & set(ga.device_id), "trung device_id voi gold cu"
    g1 = pd.concat([g0, ga], ignore_index=True)
    g1.to_csv(F / "gold_union_v6_plus_4sics.csv", index=False)
    p0 = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", dtype=str)
    pa = pd.read_csv(F / "phase2_roster_add_v6.csv", dtype=str)
    assert not set(p0.device_id) & set(pa.device_id)
    assert list(pa.columns) == list(p0.columns), (list(pa.columns), list(p0.columns))
    p1 = pd.concat([p0, pa], ignore_index=True)
    p1.to_csv(F / "phase2_type_labels_4class_plus_v6_plus_4sics.csv", index=False)
    oof = pd.read_csv(F / "phase1_oof_scores_v6.csv").set_index("device_id").lightgbm
    prof = json.loads((F / "phase2_payload_profiles_v6_full.json").read_text(encoding="utf-8"))["records"]
    have = {r["device_id"] for r in prof if r.get("device_id")}
    out = {"gold_old": metrics(g0, oof), "gold_new": metrics(g1, oof), "gold_added_only": metrics(ga, oof) if ga.gold_label.nunique() > 1 else {"n": len(ga), "note": "chi mot lop"},
           "gold_source_counts_new": g1.source.value_counts().to_dict(),
           "phase2_old": p0.type4.value_counts().to_dict(), "phase2_new": p1.type4.value_counts().to_dict(),
           "phase2_new_with_payload_profile": int(pa.device_id.isin(have).sum()), "phase2_new_total": int(len(pa)),
           "note": "diem Phase 1 la diem out-of-fold cua mo hinh dong bang (phase1_oof_scores_v6.csv); 8 nhan moi gan nhu deu la OT de doan nen khong the dung de noi mo hinh tot hon"}
    (F / "phase1_gold_plus_4sics_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
