"""Phase 2: phan tich loi theo nguon nhan, bang chung vai tro va loai con tren bo 186 nhan (Phu luc G.5; 2026-11-20).
Dung du doan ngoai mau hat 0 cua Random Forest 64 cot (phase2_oof_seed0_186.csv, script 93 --ci) va file nhan 186. KHONG ghi de ket qua dong bang.
Cach nhom giong ban 117 (kiem tra: tren phase2_oof_seed0_v6.csv va file nhan 122 cho dung 20/3/42/14/38 va 84/117).
Dau ra: phase2_errors_186.json
Chay: py -3.14 95_phase2_errors_186.py
"""
import json
from pathlib import Path

import pandas as pd

F = Path(__file__).resolve().parent.parent / "Dataset/features"


def tables(oof_file, lab_file):
    oof = pd.read_csv(F / oof_file, encoding="utf-8-sig").set_index("device_id")
    lab = pd.read_csv(F / lab_file, encoding="utf-8-sig").set_index("device_id")
    df = oof.join(lab, how="left")
    df["ok"] = df.RF_64 == df.true
    df["ev"] = df.role_evidence.fillna("NA")
    f = lambda g: {str(k): {"n": int(len(v)), "correct": int(v.ok.sum()), "acc": round(float(v.ok.mean()), 3)} for k, v in g}
    res = {"n": int(len(df)), "correct": int(df.ok.sum()), "acc": round(float(df.ok.mean()), 3),
           "majority": round(float(df.true.value_counts().max() / len(df)), 3),
           "by_source": f(df.groupby("source")), "by_evidence": f(df.groupby("ev")), "by_subtype_class": f(df.groupby(["sub_type", "true"]))}
    res["other_in_NA"] = int(((df.ev == "NA") & (df.true == "OTHER")).sum())
    hmi = df[(df.true == "HMI") & (~df.ok)][["RF_64", "sub_type", "source", "detail"]]
    res["hmi_wrong"] = hmi.reset_index().to_dict("records")
    return res


if __name__ == "__main__":
    chk = tables("phase2_oof_seed0_v6.csv", "phase2_type_labels_4class_plus_v6.csv")
    assert chk["n"] == 117 and chk["correct"] == 84, chk["correct"]   # tai hien ban dong bang
    out = {"labels_file": "phase2_type_labels_4class_186_devices.csv", **tables("phase2_oof_seed0_186.csv", "phase2_type_labels_4class_186_devices.csv")}
    (F / "phase2_errors_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
