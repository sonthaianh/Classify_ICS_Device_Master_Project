"""Phase 2: MLP 64 cot tren TAP T (thiet bi co token OT) cua bo 186 nhan, ca nhan day du va nhan nghiem (2026-11-20). KHONG ghi de ket qua dong bang.
Ly do: script 61 chi chay MLP tren tap T voi bo 39 va 49 cot (khong co dac trung hanh vi), nen bang so sanh mo hinh khong cung so cot voi Random Forest 64 cot.
Chay: py -3.14 98_phase2_mlp_T64_186.py        Dau ra: phase2_dl_mlp_T64_v6_186.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR
CUR["path"] = str(F / "phase2_type_labels_4class_186_devices.csv")


def main():
    S61 = M._load("s61", "61_phase2_dl_mlp.py")
    S58 = S61.S58
    lab, sets = S58.build()
    eng = M._orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    out = {"status": "MLP 64 cot, tap T, bo 186 nhan, 5 hat"}
    for strict in (False, True):
        sub = lab[tok & (~lab.owner_approved if strict else True)]
        y, g = sub.type4.to_numpy(), sub.group.to_numpy()
        X = np.nan_to_num(sets["behaviour_payload_role_eng"].loc[sub.index].to_numpy(dtype=float), nan=0.0)
        res, _ = S61.cv_generic(lambda tr, te, seed, yy: S61.mlp_fit_predict(X[tr], yy[tr], X[te], seed), X, y, g)
        out["strict" if strict else "all_labels"] = {"n": int(len(y)), **res}
        print("strict" if strict else "all", len(y), res["balanced_acc"], res["balanced_acc_std"], flush=True)
    (F / "phase2_dl_mlp_T64_v6_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
