"""Phase 2: Transformer tren TAP T (thiet bi co token OT) cua bo 186 nhan (2026-11-20). Boc script 80/92, KHONG ghi de ket qua dong bang.
Giong 92 nhung mat na la thiet bi co token (has_tokens = 1) thay vi thiet bi co ho so payload.
Chay (cham):
  py -3.14 96_phase2_tf_T_186.py --variant scratch --seeds 3
  py -3.14 96_phase2_tf_T_186.py --variant pretrained --seeds 3
  them --labels x186_noLow de bo 7 nhan tam thoi
Dau ra: phase2_retrain_plus_all_tfT_<labels>_<variant>.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m92", HERE / "92_phase2_tf_186.py")
M92 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M92)
M = M92.M
F, CUR, arg = M.F, M.CUR, M.arg


def main():
    name, variant, seeds = arg("--labels", "x186"), arg("--variant", "scratch"), int(arg("--seeds", "3"))
    S71 = M._load("s71", "71_phase2_dl_transformer.py")
    S58, S61, DEV = S71.S58, S71.S61, S71.DEV
    CUR["path"] = str(M92.label_file(name))
    t0 = time.time()
    pre = None
    if variant == "pretrained":
        pre, macc, mbase = S71.pretrain()
        print(f"pretrain {time.time() - t0:.0f}s masked acc {macc:.3f} vs {mbase:.3f}", flush=True)
    lab, _ = S58.build()
    eng = M._orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    sub = lab[tok & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)

    def fp(tr, te, seed, yy):
        return S71.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, pre)
    res, _ = S61.cv_generic(fp, None, y, g, seeds=seeds)
    out = {"labels": name, "regime": "T_tokens", "variant": variant, "seeds": seeds, "n": int(len(y)),
           "class_counts": pd.Series(y).value_counts().reindex(S58.CLASSES).fillna(0).astype(int).to_dict(), "result": res, "seconds": round(time.time() - t0)}
    (F / f"phase2_retrain_plus_all_tfT_{name}_{variant}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TF-T", name, variant, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
    print("DONE-TF-T")


if __name__ == "__main__":
    main()
