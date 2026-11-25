"""Phase 2: them Transformer (huan luyen tu dau, hat 0) vao bang khoang tin cay bootstrap cua bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang.
Doc phase2_oof_seed0_186.csv (script 93 --ci: RF_64, MLP_64, CNN_scratch, Luat_vai_tro, Da_so; cung fold hat 0), chay Transformer tu dau 1 hat tren cung tap A (cung thu tu thiet bi),
roi tinh lai bootstrap (B = 2000) cho ca sau mo hinh va chenh lech RF - mo hinh khac.
Dau ra: phase2_oof_seed0_186_tf.csv, phase2_baselines_ci_186_tf.json
Chay (cham, khoang 20 phut): py -3.14 97_phase2_tf_ci_186.py
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m92", HERE / "92_phase2_tf_186.py")
M92 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M92)
M = M92.M
F, CUR = M.F, M.CUR
CUR["path"] = str(M92.label_file("x186"))


def main():
    S71 = M._load("s71", "71_phase2_dl_transformer.py")
    M72 = M._load("s72", "72_phase2_baselines_ci.py")
    S58, S61, DEV = S71.S58, S71.S61, S71.DEV
    lab, _ = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    oof = pd.read_csv(F / "phase2_oof_seed0_186.csv", encoding="utf-8-sig")
    assert oof.device_id.tolist() == devs and oof.true.tolist() == y.tolist(), "thu tu thiet bi khac file out-of-fold hat 0"
    t0 = time.time()
    res, oof0 = S61.cv_generic(lambda tr, te, seed, yy: S71.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None), None, y, g, seeds=1)
    print("Transformer seed0", res["balanced_acc"], f"{time.time() - t0:.0f}s", flush=True)
    oof["Transformer_scratch"] = np.array(oof0, dtype=object)
    oof.to_csv(F / "phase2_oof_seed0_186_tf.csv", index=False, encoding="utf-8-sig")
    models = ["RF_64", "MLP_64", "CNN_scratch", "Transformer_scratch", "Luat_vai_tro", "Da_so"]
    preds = {k: oof[k].to_numpy(dtype=object) for k in models}
    rng = np.random.default_rng(M72.SEED)
    r, diffs = M72.boot_ci(y, preds, rng)
    out = {"labels_file": "phase2_type_labels_4class_186_devices.csv", "n": len(y), "bootstrap_B": M72.B, "balanced_acc_seed0_with_bootstrap_ci": r, "paired_bootstrap_rf_minus_other": diffs,
           "per_model_recall_seed0": {k: dict(zip(S58.CLASSES, np.round(recall_score(y, p, labels=S58.CLASSES, average=None, zero_division=0), 3).tolist())) for k, p in preds.items()},
           "transformer_seed0_balanced_acc": res["balanced_acc"]}
    rec_ci, n = {}, len(y)
    idx = [rng.integers(0, n, n) for _ in range(M72.B)]
    for c in S58.CLASSES:
        vals = []
        for i in idx:
            m = y[i] == c
            if m.sum():
                vals.append(float((preds["RF_64"][i][m] == c).mean()))
        rec_ci[c] = [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
    out["rf64_recall_bootstrap_ci95"] = rec_ci
    (F / "phase2_baselines_ci_186_tf.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["balanced_acc_seed0_with_bootstrap_ci"], indent=1), json.dumps(diffs, indent=1), "DONE-TF-CI", flush=True)


if __name__ == "__main__":
    main()
