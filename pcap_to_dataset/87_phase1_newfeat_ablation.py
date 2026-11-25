"""Phase 1: danh gia dac trung moi (script 84) bang ablation theo nhom, giao thuc DONG BANG (2026-11-12). KHONG ghi de ket qua dong bang.

Moi bien the dung cung LightGBM tham so dong bang (script 81), K = 20, cung bo chia (fold_device, fold_component), 3 hat (trung binh diem thiet bi), cung chi so:
  base15   15 dac trung hanh vi dong bang
  +T       + 5 dac trung thoi gian (iat_cv, iat_log_median_ms, iat_entropy, iat_burst, active_slot_ratio)
  +L       + 3 dac trung do dai goi (len_cv, len_entropy, len_unique_ratio)
  +G       + 5 dac trung do thi (peer_count_log, fanout_ratio, reciprocity, top_peer_share, peer_entropy)
  +TLG     + 13 dac trung moi
  new13    chi 13 dac trung moi (khong co 15 cu: xem thong tin rieng cua chung)
Cham tren: nhan luat (3.923 thiet bi), gold 184, gold mo rong 243; va bo-mot-nhom-nguon (LOFO) tren 4 nhom nguon capture.
  py -3.14 87_phase1_newfeat_ablation.py
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
T = ["iat_cv", "iat_log_median_ms", "iat_entropy", "iat_burst", "active_slot_ratio"]
Lg = ["len_cv", "len_entropy", "len_unique_ratio"]
G = ["peer_count_log", "fanout_ratio", "reciprocity", "top_peer_share", "peer_entropy"]
SETS = {"base15": USED, "+T": USED + T, "+L": USED + Lg, "+G": USED + G, "+TLG": USED + T + Lg + G, "new13": T + Lg + G}
OTL = "OT_confirmed"
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1, subsample_freq=1, n_jobs=8, verbose=-1)
SEEDS = (SEED, SEED + 1, SEED + 2)
FAMS = ["eth2dump_attack_lab", "anonymized_hourly", "switch_series_siw_swi", "4sics"]


def fit_predict(Xtr, ytr, Xte, seeds=SEEDS):
    s = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    out = 0
    for sd in seeds:
        m = lgb.LGBMClassifier(scale_pos_weight=s, random_state=sd, **P).fit(Xtr, ytr)
        out = out + m.predict_proba(Xte)[:, 1]
    return out / len(seeds)


def met(s, ids, y):
    sc = s.loc[ids].to_numpy()
    pred = sc >= 0.5
    d = {"n": int(len(ids)), "ot": int(y.sum()), "roc": round(float(roc_auc_score(y, sc)), 4), "pr": round(float(average_precision_score(y, sc)), 4), "recall@0.5": round(float((pred & y).sum() / max(y.sum(), 1)), 4)}
    if (~y).sum():
        d["nonOT_flag@0.5"] = round(float((pred & ~y).sum() / (~y).sum()), 4)
    return d


def main():
    t0 = time.time()
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "window_start", "cap_rank"] + USED)
    nf = pd.read_parquet(F / "phase1_newfeat_v6.parquet")
    n0 = len(x)
    x = x.merge(nf, on=["device_id", "session_id", "window_start"], how="left", validate="one_to_one")
    cov = {c: round(float(x[c].notna().mean()), 4) for c in T + Lg + G}
    print("windows", n0, "coverage", cov, flush=True)
    src = pd.read_csv(F / "session_source_map_draft.csv", usecols=["session_id", "source_family"])
    x = x.merge(src, on="session_id", how="left")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype=str, keep_default_na=False)
    for c in ("fold_device", "fold_component"):
        dev[c] = dev[c].astype(int)
    yrule = (dev.label == OTL).to_numpy()
    g184 = pd.read_csv(F / "gold_union_v6.csv").set_index("device_id")
    g243 = pd.read_csv(F / "gold_union_v6_plus_all.csv").set_index("device_id")
    dev_ids = x.device_id.to_numpy()
    cap = x.cap_rank.to_numpy() < K
    y_win = np.isin(dev_ids, dev.device_id[yrule]).astype(int)
    res = {"protocol": "LightGBM frozen params, K=20, 3 seeds averaged, device score = mean window prob; rule labels / gold184 / gold243", "coverage": cov, "n_windows": int(n0), "splits": {}, "lofo": {}, "sets": {k: v for k, v in SETS.items()}}
    oof_all = {}
    for split in ("fold_device", "fold_component"):
        fold_of = dict(zip(dev.device_id, dev[split]))
        fx = x.device_id.map(fold_of).to_numpy()
        blk = {}
        for name, cols in SETS.items():
            Xa = x[cols].to_numpy(dtype=np.float32)
            out = pd.Series(np.nan, index=dev.device_id)
            for f in range(5):
                tr, te = (fx != f) & cap, fx == f
                p = fit_predict(Xa[tr], y_win[tr], Xa[te])
                s = pd.Series(p, index=x.index[te]).groupby(x.device_id[te]).mean()
                out.loc[s.index] = s.values
            oof_all[f"{split}|{name}"] = out
            blk[name] = {"rule": met(out, list(dev.device_id), yrule), "gold184": met(out, list(g184.index), (g184.gold_label == "OT").to_numpy()),
                         "gold243": met(out, list(g243.index), (g243.gold_label == "OT").to_numpy())}
            print(split, name, "rule", blk[name]["rule"]["roc"], "g184", blk[name]["gold184"]["roc"], "g243", blk[name]["gold243"]["roc"], f"{time.time() - t0:.0f}s", flush=True)
        # bootstrap theo thiet bi: hieu ROC so voi base15
        rng = np.random.default_rng(SEED)
        yg = (g243.gold_label == "OT").to_numpy()
        b = oof_all[f"{split}|base15"]
        ci = {}
        for name in SETS:
            if name == "base15":
                continue
            s = oof_all[f"{split}|{name}"]
            sr, br = s.loc[dev.device_id].to_numpy(), b.loc[dev.device_id].to_numpy()
            sg, bg = s.loc[g243.index].to_numpy(), b.loc[g243.index].to_numpy()
            dr, dg = [], []
            for _ in range(500):
                i = rng.integers(0, len(yrule), len(yrule))
                j = rng.integers(0, len(yg), len(yg))
                if 0 < yrule[i].sum() < len(i):
                    dr.append(roc_auc_score(yrule[i], sr[i]) - roc_auc_score(yrule[i], br[i]))
                if 0 < yg[j].sum() < len(j):
                    dg.append(roc_auc_score(yg[j], sg[j]) - roc_auc_score(yg[j], bg[j]))
            q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
            ci[name] = {"rule_roc_diff_vs_base15": q(dr), "gold243_roc_diff_vs_base15": q(dg)}
        blk["bootstrap_vs_base15"] = ci
        res["splits"][split] = blk
    pd.DataFrame(oof_all).to_csv(F / "phase1_newfeat_ablation_oof.csv")
    # LOFO
    for fam in FAMS:
        fam_devices = set(x.loc[x.source_family == fam, "device_id"])
        test_ids = set(dev.loc[dev.primary_source_family == fam, "device_id"])
        tr = (~x.device_id.isin(fam_devices)).to_numpy() & cap
        te = x.device_id.isin(test_ids).to_numpy()
        d = dev[dev.device_id.isin(test_ids)]
        yt = (d.label == OTL).to_numpy()
        blk = {"n_test": int(len(d)), "ot": int(yt.sum()), "train_devices": int(x.loc[tr, "device_id"].nunique())}
        for name, cols in SETS.items():
            Xa = x[cols].to_numpy(dtype=np.float32)
            p = fit_predict(Xa[tr], y_win[tr], Xa[te])
            s = pd.Series(p, index=x.index[te]).groupby(x.device_id[te]).mean().reindex(d.device_id)
            blk[name] = {"roc": round(float(roc_auc_score(yt, s.to_numpy())), 4) if 0 < yt.sum() < len(yt) else None,
                         "recall@0.5": round(float(((s.to_numpy() >= 0.5) & yt).sum() / max(yt.sum(), 1)), 4),
                         "flag_rate@0.5": round(float((s.to_numpy() >= 0.5).mean()), 4)}
        res["lofo"][fam] = blk
        print("LOFO", fam, {k: v["roc"] for k, v in blk.items() if isinstance(v, dict)}, f"{time.time() - t0:.0f}s", flush=True)
    # do quan trong (gain) cua mo hinh +TLG tren toan bo du lieu
    cols = SETS["+TLG"]
    Xa = x[cols].to_numpy(dtype=np.float32)
    m = lgb.LGBMClassifier(scale_pos_weight=float((y_win[cap] == 0).sum() / (y_win[cap] == 1).sum()), random_state=SEED, importance_type="gain", **P).fit(Xa[cap], y_win[cap])
    imp = pd.Series(m.feature_importances_, index=cols)
    res["gain_share_TLG"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()
    res["seconds"] = round(time.time() - t0)
    (F / "phase1_newfeat_ablation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-ABLATION", res["seconds"], "s")


if __name__ == "__main__":
    main()
