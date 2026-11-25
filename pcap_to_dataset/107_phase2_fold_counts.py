"""Dem so thiet bi huan luyen / kiem tra theo lop o tung fold cua giai doan 2 (2026-11-22). KHONG huan luyen, KHONG ghi de ket qua dong bang.

Dung lai build() cua script 58 va nhan 186 (nhu script 101/91): tap A (co ho so payload) va tap T (co thong diep OT, tinh tren 186 nhan).
Cach chia: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=hat), nhom = capture dau tien cua thiet bi; moi tap chia rieng.
Dau ra: Dataset/features/phase2_fold_counts_186.json
Chay:  py -3.14 107_phase2_fold_counts.py
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR
CUR["path"] = str(F / "phase2_type_labels_4class_186_devices.csv")
S61 = M._load("s61", "61_phase2_dl_mlp.py")
S58 = S61.S58
CLASSES = ["PLC", "HMI", "EWS", "OTHER"]


def counts(y):
    s = pd.Series(y).value_counts()
    return {c: int(s.get(c, 0)) for c in CLASSES}


def main():
    lab, sets = S58.build()
    eng = M._orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    out = {"n_labels": int(len(lab)), "class_counts_186": counts(lab.type4)}
    for name, mask in (("A", lab.has_profile), ("T", tok)):
        sub = lab[mask]
        y, g = sub.type4.to_numpy(), sub.group.to_numpy()
        blk = {"n": int(len(sub)), "class_counts": counts(y), "n_groups_first_capture": int(len(set(g))), "seeds": []}
        for seed in range(5):
            folds = []
            for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(sub, y, g)):
                folds.append({"fold": k, "train_n": int(len(tr)), "train": counts(y[tr]), "test_n": int(len(te)), "test": counts(y[te]),
                              "test_groups": int(len(set(g[te])))})
            blk["seeds"].append(folds)
        out[name] = blk
        f0 = pd.DataFrame([{"fold": f["fold"], "train_n": f["train_n"], **{"tr_" + c: f["train"][c] for c in CLASSES}, "test_n": f["test_n"], **{"te_" + c: f["test"][c] for c in CLASSES}} for f in blk["seeds"][0]])
        print(name, blk["n"], blk["class_counts"])
        print(f0.to_string())
        tr_all = [f["train_n"] for s in blk["seeds"] for f in s]
        te_all = [f["test_n"] for s in blk["seeds"] for f in s]
        print(" train n range", min(tr_all), max(tr_all), " test n range", min(te_all), max(te_all))
    (F / "phase2_fold_counts_186.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
