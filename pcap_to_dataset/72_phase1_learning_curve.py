"""Phase 1: duong cong hoc tren du lieu THAT (khong dung du lieu sinh) (2026-11-12).

Hoi: neu them thiet bi that (cung phan phoi voi hien tai) thi ROC co con tang khong?
Cach lam: giu nguyen giao thuc dong bang (LightGBM tham so mac dinh, 15 dac trung, K = 20, 5 fold). Trong moi fold, chi lay ngau nhien
f% thiet bi huan luyen (phan tang theo nhan luat) de huan luyen; cham tren TOAN BO thiet bi fold giu lai (khong doi). f = 10, 25, 50, 75, 100%.
Lap 3 hat lay mau (f < 100%). Chay voi hai cach chia: fold_device (mac dinh) va fold_component (chat hon: thanh phan lien thong nam tron o mot fold).
Ghi ra: Dataset/features/phase1_learning_curve_v6.csv / .json

  py -3.14 72_phase1_learning_curve.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_contract import TRAINING_FEATURES as D17  # noqa: E402

warnings.filterwarnings("ignore")
F = HERE.parent / "Dataset/features"
SEED, K = 20261108, 20
USED = [c for c in D17 if c not in ("ip_count", "frames_sent")]
OTL = "OT_confirmed"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1)
FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
N_SEEDS = 3


def main():
    t0 = time.time()
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "cap_rank", "label"] + USED)
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype=str, keep_default_na=False)
    for c in ("fold_device", "fold_component"):
        dev[c] = dev[c].astype(int)
    gold = pd.read_csv(F / "gold_union_v6.csv").set_index("device_id")
    lab = dev.set_index("device_id").label
    yfull = (dev.label == OTL).to_numpy()
    gi = [d for d in dev.device_id if d in gold.index]
    yg = (gold.loc[gi, "gold_label"] == "OT").to_numpy()
    dev_codes = x.device_id.astype("category")
    codes = dev_codes.cat.codes.to_numpy()
    cat_index = {d: i for i, d in enumerate(dev_codes.cat.categories)}
    capmask = x.cap_rank.to_numpy() < K
    ywin = (x.label == OTL).to_numpy().astype(int)
    rows, oofs = [], {}
    for split in ("fold_device", "fold_component"):
        fold_of = dict(zip(dev.device_id, dev[split]))
        fx = np.array([fold_of[d] for d in dev_codes.cat.categories])[codes]
        for frac in FRACS:
            for s in range(1 if frac == 1.0 else N_SEEDS):
                oof = pd.Series(np.nan, index=dev.device_id)
                n_tr_dev = []
                for f in range(5):
                    tr_dev = dev[dev[split] != f]
                    if frac < 1.0:
                        rng = np.random.default_rng(SEED + 7919 * s + 131 * f + int(frac * 1000))
                        keep = pd.concat([g.sample(frac=frac, random_state=int(rng.integers(1 << 30))) for _, g in tr_dev.groupby("label")]).device_id
                    else:
                        keep = tr_dev.device_id
                    n_tr_dev.append(len(keep))
                    keep_codes = np.array([cat_index[d] for d in keep])
                    in_keep = np.zeros(len(dev_codes.cat.categories), bool)
                    in_keep[keep_codes] = True
                    tr = in_keep[codes] & capmask
                    te = fx == f
                    spw = float((ywin[tr] == 0).sum() / max((ywin[tr] == 1).sum(), 1))
                    m = lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1, **PARAMS).fit(x.loc[tr, USED], ywin[tr])
                    p = pd.Series(m.predict_proba(x.loc[te, USED])[:, 1], index=x.index[te]).groupby(x.device_id[te]).mean()
                    oof.loc[p.index] = p.values
                assert not oof.isna().any()
                sc = oof.loc[dev.device_id].to_numpy()
                sg = oof.loc[gi].to_numpy()
                r = {"split": split, "fraction": frac, "sample_seed": s, "train_devices_mean": round(float(np.mean(n_tr_dev))),
                     "rule_roc": roc_auc_score(yfull, sc), "rule_pr": average_precision_score(yfull, sc),
                     "gold_roc": roc_auc_score(yg, sg), "gold_pr": average_precision_score(yg, sg),
                     "gold_recall@0.5": float(((sg >= 0.5) & yg).sum() / yg.sum()), "gold_nonOT_flag@0.5": float(((sg >= 0.5) & ~yg).sum() / (~yg).sum())}
                rows.append(r)
                oofs[(split, frac, s)] = oof
                print(f"{split:15s} frac={frac:<5} seed={s} train_dev={r['train_devices_mean']:5d} rule ROC {r['rule_roc']:.4f} PR {r['rule_pr']:.4f} | gold ROC {r['gold_roc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(F / "phase1_learning_curve_v6_runs.csv", index=False)
    g = d.groupby(["split", "fraction"]).agg(train_devices=("train_devices_mean", "mean"), n_runs=("sample_seed", "size"),
                                              rule_roc=("rule_roc", "mean"), rule_roc_sd=("rule_roc", "std"), rule_pr=("rule_pr", "mean"), rule_pr_sd=("rule_pr", "std"),
                                              gold_roc=("gold_roc", "mean"), gold_roc_sd=("gold_roc", "std"), gold_pr=("gold_pr", "mean"),
                                              gold_recall=("gold_recall@0.5", "mean"), gold_nonOT_flag=("gold_nonOT_flag@0.5", "mean")).round(4).reset_index()
    g.to_csv(F / "phase1_learning_curve_v6.csv", index=False)
    # bootstrap cap thiet bi: 100% so voi 50% (hat lay mau 0), cung thiet bi cham
    boot = {}
    rng = np.random.default_rng(SEED)
    for split in ("fold_device", "fold_component"):
        full = oofs[(split, 1.0, 0)].loc[dev.device_id].to_numpy()
        half = oofs[(split, 0.5, 0)].loc[dev.device_id].to_numpy()
        fg, hg = oofs[(split, 1.0, 0)].loc[gi].to_numpy(), oofs[(split, 0.5, 0)].loc[gi].to_numpy()
        dr, dg = [], []
        for _ in range(1000):
            i = rng.integers(0, len(yfull), len(yfull))
            if 0 < yfull[i].sum() < len(i):
                dr.append(roc_auc_score(yfull[i], full[i]) - roc_auc_score(yfull[i], half[i]))
            j = rng.integers(0, len(yg), len(yg))
            if 0 < yg[j].sum() < len(j):
                dg.append(roc_auc_score(yg[j], fg[j]) - roc_auc_score(yg[j], hg[j]))
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
        boot[split] = {"rule_roc_100_minus_50[2.5,mean,97.5]": ci(dr), "gold_roc_100_minus_50[2.5,mean,97.5]": ci(dg)}
    out = {"protocol": "LightGBM frozen default params, 15 features, K=20, 5 folds; train subsample stratified by rule label; test = all held-out devices; real data only",
           "fractions": list(FRACS), "sample_seeds": N_SEEDS, "summary": g.to_dict(orient="records"), "bootstrap": boot, "seconds": round(time.time() - t0)}
    (F / "phase1_learning_curve_v6.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(g.to_string(index=False))
    print(json.dumps(boot, indent=1))


if __name__ == "__main__":
    main()
