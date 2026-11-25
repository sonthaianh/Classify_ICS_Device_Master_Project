"""Phase 2: chay lai phan con thieu cua quy trinh dong bang tren bo 186 nhan (2026-11-19). KHONG ghi de ket qua dong bang.

Da co (script 91/92): RF + MLP (RF/MLP, 3 bien the nhan), Transformer scratch/pretrained.
Script nay them:
  --cnn   CNN 1D scratch + pretrained, 5 hat, che do A (co ho so) va T (co token), nhan day du va nhan nghiem (bo owner_approved)  -> phase2_dl_seq_186.json
  --ci    seed-0 out-of-fold cho RF_64, MLP_64, CNN_scratch + 2 baseline; bootstrap CI (B=2000) va chenh lech RF - mo hinh khac       -> phase2_baselines_ci_186.json, phase2_oof_seed0_186.csv
  --perm  kiem tra hoan vi nhan: RF (200 lan) va CNN scratch (30 lan) tren tap A                                                  -> phase2_permutation_186.json
Nhan: phase2_type_labels_4class_186_devices.csv (chuyen huong pd.read_csv nhu script 80/91). Cung StratifiedGroupKFold (nhom = session dau), balanced accuracy.
Chay: py -3.14 93_phase2_full_186.py --cnn      (cham)   |   py -3.14 93_phase2_full_186.py --ci --perm
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR
LABELS = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "x186"
SRC = F / ("phase2_type_labels_4class_186_devices.csv" if LABELS == "x186" else f"phase2_type_labels_4class_186_devices_{LABELS}.csv")
SUFFIX = "" if LABELS == "x186" else f"_{LABELS}"
ONLY_ALL = "--only-all" in sys.argv  # bo che do strict (chi chay nhan day du cua bien the)
CUR["path"] = str(SRC)
S62 = M._load("s62", "62_phase2_dl_seq.py")
S61, S58 = S62.S61, S62.S58
CLASSES = S58.CLASSES
DEV = S62.DEV


def regimes(lab):
    eng = M._orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    return {"A_profiled": lab.has_profile, "T_tokens": tok}


def run_cnn():
    t0 = time.time()
    pre, macc, mbase = S62.pretrain()
    print(f"pretrain {time.time() - t0:.0f}s masked acc {macc:.3f} vs {mbase:.3f}", flush=True)
    assert macc > mbase + 0.05
    lab, _ = S58.build()
    out = {"status": "EXPLORATORY retrain on 186 labels; frozen results untouched", "labels_file": SRC.name, "chance_balanced_acc": 0.25,
           "pretrain": {"masked_symbol_acc": round(macc, 3), "most_common_baseline": round(mbase, 3)}}
    for rname, mask in regimes(lab).items():
        for strict in ((False,) if ONLY_ALL else (False, True)):
            sub = lab[mask & (~lab.owner_approved if strict else True)]
            sub = sub[sub.index.isin(DEV)]
            y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
            tag = f"{rname}|{'strict' if strict else 'all_labels'}"
            block = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()}
            for vname, st in (("scratch", None), ("pretrained", pre)):
                def fp(tr, te, seed, yy, st=st, devs=devs):
                    return S62.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, st)
                res, _ = S61.cv_generic(fp, None, y, g, seeds=5)
                block[vname] = res
                print(tag, vname, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
            out[tag] = block
            (F / f"phase2_dl_seq_186{SUFFIX}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-CNN")


def run_ci():
    M72 = M._load("s72", "72_phase2_baselines_ci.py")
    lab, sets = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    X = sets["behaviour_payload_role_eng"].loc[sub.index]
    preds = {}
    res, oof = S58.cv(X, y, g, seeds=1, n_est=300)
    preds["RF_64"] = np.array(oof, dtype=object)
    print("RF seed0", res["balanced_acc"], flush=True)
    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
    res, oof = S61.cv_generic(lambda tr, te, seed, yy: S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), Xn, y, g, seeds=1)
    preds["MLP_64"] = np.array(oof, dtype=object)
    print("MLP seed0", res["balanced_acc"], flush=True)
    res, oof = S61.cv_generic(lambda tr, te, seed, yy: S62.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None), None, y, g, seeds=1)
    preds["CNN_scratch"] = np.array(oof, dtype=object)
    print("CNN seed0", res["balanced_acc"], flush=True)
    role = M._orig(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    fr = role.reindex(sub.index).final_role.fillna("unknown")
    preds["Luat_vai_tro"] = np.array(np.where(fr == "server", "PLC", np.where(fr == "client", "HMI", "OTHER")), dtype=object)
    preds["Da_so"] = np.array(["OTHER"] * len(y), dtype=object)
    out = {"status": "EXPLORATORY retrain on 186 labels; seed-0 out-of-fold, same folds for all models", "labels_file": SRC.name, "bootstrap_B": M72.B, "n": len(y),
           "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
           "note": "Transformer khong co trong bang nay (script 80 khong luu out-of-fold); xem phase2_retrain_plus_all_tf_x186_*.json"}
    out["per_model_recall_seed0"] = {k: dict(zip(CLASSES, np.round(recall_score(y, p, labels=CLASSES, average=None, zero_division=0), 3).tolist())) for k, p in preds.items()}
    rng = np.random.default_rng(M72.SEED)
    res, diffs = M72.boot_ci(y, {k: np.asarray(v) for k, v in preds.items()}, rng)
    out["balanced_acc_seed0_with_bootstrap_ci"], out["paired_bootstrap_rf_minus_other"] = res, diffs
    rec_ci, n = {}, len(y)
    idx = [rng.integers(0, n, n) for _ in range(M72.B)]
    for c in CLASSES:
        vals = []
        for i in idx:
            m = y[i] == c
            if m.sum():
                vals.append(float((preds["RF_64"][i][m] == c).mean()))
        rec_ci[c] = [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
    out["rf64_recall_bootstrap_ci95"] = rec_ci
    pd.DataFrame({"device_id": devs, "true": y, **preds}).to_csv(F / "phase2_oof_seed0_186.csv", index=False, encoding="utf-8-sig")
    (F / "phase2_baselines_ci_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["balanced_acc_seed0_with_bootstrap_ci"], indent=1), "DONE-CI", flush=True)


def run_perm(n_rf=200, n_cnn=30):
    lab, sets = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    X = sets["behaviour_payload_role_eng"].loc[sub.index]
    rng = np.random.default_rng(20261119)
    obs_rf = S58.cv(X, y, g, seeds=1)[0]["balanced_acc"]
    null = []
    for i in range(n_rf):
        r, _ = S58.cv(X, y, g, seeds=1, shuffle_y=rng.permutation(y))
        null.append(r["balanced_acc"])
    out = {"labels_file": SRC.name, "n": int(len(y)),
           "RF_64": {"observed_seed0": obs_rf, "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                     "p_value": round(float((1 + sum(v >= obs_rf for v in null)) / (1 + len(null))), 4), "n_perm": n_rf}}
    print(out["RF_64"], flush=True)
    (F / "phase2_permutation_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    obs_cnn, null = None, []
    for i in range(n_cnn):
        ys = rng.permutation(y)
        fp = lambda tr, te, seed, yy: S62.ft_predict([devs[j] for j in tr], yy[tr], [devs[j] for j in te], seed, None)
        if obs_cnn is None:
            obs_cnn = S61.cv_generic(fp, None, y, g, seeds=1)[0]["balanced_acc"]
        null.append(S61.cv_generic(fp, None, y, g, seeds=1, shuffle_y=ys)[0]["balanced_acc"])
        print("perm cnn", i, null[-1], flush=True)
    out["CNN_scratch"] = {"observed_seed0": obs_cnn, "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                          "p_value": round(float((1 + sum(v >= obs_cnn for v in null)) / (1 + len(null))), 4), "n_perm": n_cnn}
    (F / "phase2_permutation_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out, "DONE-PERM")


if __name__ == "__main__":
    if "--cnn" in sys.argv:
        run_cnn()
    if "--ci" in sys.argv:
        run_ci()
    if "--perm" in sys.argv:
        run_perm()
    if not any(a in sys.argv for a in ("--cnn", "--ci", "--perm")):
        print(__doc__)
