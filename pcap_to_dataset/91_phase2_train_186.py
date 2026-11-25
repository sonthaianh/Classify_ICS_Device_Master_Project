"""Phase 2: huan luyen thu (RF + MLP) tren bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang.

Nhan 186 = 165 nhan cu + 21 nhan moi (14 reviewed_session_evidence, 7 reviewed_provisional). File nguon: phase2_type_labels_4class_186_devices.csv
Cac bien the:
  x186            186 nhan
  x186_noLow      bo 7 dong reviewed_provisional (do tin cay thap)
  x186_noReviewed bo ca 21 dong reviewed_* (de tach hanh vi lam nhan khoi ket qua)
Cung quy trinh dong bang: StratifiedGroupKFold (nhom = session dau), 5 hat, balanced accuracy; tap A = co ho so payload, T = co token OT.
Dung lai ham cua script 80/61/58. Dau ra: phase2_train_186_tabular.json.
Chay:  py -3.14 91_phase2_train_186.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, _orig, CUR = M.F, M._orig, M.CUR

SRC = F / "phase2_type_labels_4class_186_devices.csv"
OLD165 = F / "phase2_type_labels_4class_plus_v6_plus_all.csv"


def variants():
    d = _orig(SRC, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out = {"x186": SRC}
    for name, mask in (("x186_noLow", d.source != "reviewed_provisional"), ("x186_noReviewed", ~d.source.str.startswith("reviewed_"))):
        p = F / f"phase2_type_labels_4class_186_devices_{name}.csv"
        d[mask].to_csv(p, index=False, encoding="utf-8-sig")
        out[name] = p
    return out


def main():
    S61 = M._load("s61", "61_phase2_dl_mlp.py")
    S58 = S61.S58
    old_ids = set(_orig(OLD165, encoding="utf-8-sig").device_id)
    res = {"status": "EXPLORATORY retrain on 186 labels; frozen results untouched", "chance_balanced_acc": 0.25, "variants": {}}
    eng = _orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    for name, path in variants().items():
        CUR["path"] = str(path)
        lab, sets = S58.build()
        tok = pd.Series({dv: (dv in eng.index and int(eng.loc[dv, "has_tokens"]) == 1) for dv in lab.index})
        blk = {"n_labels": int(len(lab)), "class_counts": lab.type4.value_counts().to_dict(), "n_profiled": int(lab.has_profile.sum()), "n_tokens": int(tok.sum())}
        for rname, mask in (("A_profiled", lab.has_profile), ("T_tokens", tok)):
            sub = lab[mask]
            y, g = sub.type4.to_numpy(), sub.group.to_numpy()
            r = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(S58.CLASSES).fillna(0).astype(int).to_dict()}
            for ks in ("behaviour_payload_role_eng", "payload_role_eng"):
                if ks == "behaviour_payload_role_eng" or rname == "A_profiled":
                    X = sets[ks].loc[sub.index]
                    rf, oof = S58.cv(X, y, g)
                    r[f"RF|{ks}"] = rf
                    if ks == "behaviour_payload_role_eng":
                        r["RF_subsets_seed0"] = M.subset_report(sub, oof, old_ids)
                    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
                    mres, _ = S61.cv_generic(lambda tr, te, seed, yy, Xn=Xn: S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), Xn, y, g)
                    r[f"MLP|{ks}"] = mres
            blk[rname] = r
            print(name, rname, "n=", len(y), {k: v["balanced_acc"] for k, v in r.items() if isinstance(v, dict) and "balanced_acc" in v}, flush=True)
        res["variants"][name] = blk
    (F / "phase2_train_186_tabular.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-186")


if __name__ == "__main__":
    main()
