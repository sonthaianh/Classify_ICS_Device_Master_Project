"""Phase 1 - doi chung bo sung cho script 63: LightGBM tren feature GOP THEO THIET BI (khong phai DL), 2026-11-10.

Cau hoi: DeepSets thang LightGBM cua so co phai vi la deep learning, hay chi vi no nhin ca tap cua so cua thiet bi? Neu LightGBM tren thong ke gop
(mean, std, min, max cua tung feature tren toi da K=20 cua so dau cua thiet bi) cung dat tuong duong DeepSets thi loi the den tu CACH GOP, khong phai DL.
Cung giao thuc voi script 63: cung feature log, cung K, cung fold (fold_device va fold_component), 3 hat, cung gold, cung bootstrap theo thiet bi.
KHONG dung so cua so cua thiet bi lam feature (tranh nhiem "do dai capture").
Doc diem cua DeepSets/LightGBM-cua-so tu phase1_dl_vs_lgbm_scores_v6.csv (script 63).
Dau ra: phase1_device_agg_gbm_v6.json     Chay: py -3.14 pcap_to_dataset/64_phase1_device_agg_gbm.py
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("s63", HERE / "63_phase1_dl_vs_lgbm.py")
S63 = importlib.util.module_from_spec(s)
s.loader.exec_module(S63)
dev, F, K, SEED, B = S63.dev, S63.F, S63.K, S63.SEED, S63.B


def agg_table():
    v = pd.DataFrame(S63.XR[S63.CAPX], columns=S63.COLS)
    v["device_id"] = S63.DEV_OF[S63.CAPX]
    g = v.groupby("device_id")
    parts = [g.mean().add_suffix("__mean"), g.std().fillna(0.0).add_suffix("__std"), g.min().add_suffix("__min"), g.max().add_suffix("__max")]
    return pd.concat(parts, axis=1)


VARIANTS = {"mean_only": ["__mean"], "mean_min_max": ["__mean", "__min", "__max"], "all_mean_std_min_max": ["__mean", "__std", "__min", "__max"], "n_windows_only": None}


def sel(T, suffixes):
    if suffixes is None:      # doi chung nhiem: chi dung so cua so (log) cua thiet bi, de do muc "do dai capture"
        return pd.DataFrame({"log_n_windows": np.log1p(dev.set_index("device_id").loc[T.index, "n_windows"].to_numpy())}, index=T.index)
    return T[[c for c in T.columns if any(c.endswith(x) for x in suffixes)]]


def roc_len(score_arr, mask):
    y = (dev.label == "OT_confirmed").to_numpy()
    return round(float(roc_auc_score(y[mask], np.asarray(score_arr)[mask])), 3)


def main():
    Tall = agg_table().loc[dev.device_id]
    y = (dev.label == "OT_confirmed").astype(int).to_numpy()
    long_mask = (dev.n_windows >= 5).to_numpy()
    out = {"status": "EXPLORATORY control for script 63", "note": "std/min/max cua thiet bi chi co 1 cua so deu bang 0 -> co the ma hoa so cua so; bien the mean_only va n_windows_only de tach phan nay",
           "n_devices_ge5_windows": int(long_mask.sum())}
    ref = pd.read_csv(F / "phase1_dl_vs_lgbm_scores_v6.csv").set_index("device_id")
    for tag, fold_col in (("fold_device", "fold_device"), ("strict_component", "fold_component")):
        out[tag] = {"reference": {m: {"roc_all": round(float(roc_auc_score(y, ref[f"{tag}|{m}"].loc[dev.device_id])), 3),
                                      "roc_devices_ge5_windows": roc_len(ref[f"{tag}|{m}"].loc[dev.device_id].to_numpy(), long_mask)} for m in ("lightgbm", "deepsets_device")}}
        for vname, suf in VARIANTS.items():
            T = sel(Tall, suf)
            per = []
            for sd in range(S63.SEEDS):
                sc = pd.Series(np.nan, index=dev.device_id)
                for f in range(5):
                    te = (dev[fold_col] == f).to_numpy()
                    ytr = y[~te]
                    m = lgb.LGBMClassifier(scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)), subsample_freq=1, random_state=SEED + sd, n_jobs=8, verbose=-1,
                                           **B.DEFAULT["lightgbm"]).fit(T[~te], ytr)
                    sc.loc[dev.device_id[te]] = m.predict_proba(T[te])[:, 1]
                assert not sc.isna().any()
                per.append(sc)
            ens_arr = np.mean([p.loc[dev.device_id].to_numpy() for p in per], axis=0)
            ens = S63.evaluate(ens_arr)
            blk = {"n_features": int(T.shape[1]), "roc_all": ens["all"]["roc_auc"], "pr_all": ens["all"]["pr_auc"], "portonly_roc": ens["portonly_OT_vs_unknown"]["roc_auc"],
                   "roc_devices_ge5_windows": roc_len(ens_arr, long_mask), "gold": ens["gold"],
                   "per_seed_roc": [round(float(roc_auc_score(y, p.loc[dev.device_id])), 3) for p in per]}
            blk["bootstrap_minus_lightgbm_window"] = S63.boot_diff(ens_arr, ref[f"{tag}|lightgbm"].loc[dev.device_id].to_numpy(), n=1000)
            out[tag][vname] = blk
            print(tag, vname, {k: blk[k] for k in ("n_features", "roc_all", "pr_all", "roc_devices_ge5_windows")}, blk["gold"], flush=True)
    (F / "phase1_device_agg_gbm_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
