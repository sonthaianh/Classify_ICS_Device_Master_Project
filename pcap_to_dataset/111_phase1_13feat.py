"""Nhom A2 cua so theo doi (2026-11-25): giai doan 1 voi 13 dac trung (bo 2 dac trung dem cong svc_port_unique, unique_local_port_count). KHONG ghi de ket qua dong bang.

Cung giao thuc voi mo hinh chinh (script 34/36/39): LightGBM tham so mac dinh, huan luyen tren K = 20 cua so dau cua moi thiet bi, cham diem tren MOI cua so cua thiet bi ngoai fold,
diem thiet bi = trung binh xac suat cua so; 5 fold theo thiet bi (fold_device) va 5 fold theo thanh phan lien thong (fold_component).
Chay ca 15 dac trung (de kiem tra tai hien so da bao cao: 0,928 / 0,858 / 0,819) va 13 dac trung; so sanh ghep cap tren tap gold 184 (bootstrap theo thiet bi, B = 2000).
Dau ra: phase1_13feat_eval.json, phase1_oof_scores_13feat.csv.     Chay: py -3.14 111_phase1_13feat.py
"""
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t39", HERE / "39_phase1_scanner_strict_obstime.py")
T39 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T39)
BM = T39.B                      # module script 34 (make, DEFAULT, K, FEATS)
F = Path(HERE).parent / "Dataset/features"
PORT = ["svc_port_unique", "unique_local_port_count"]
F15 = list(BM.FEATS)
F13 = [c for c in F15 if c not in PORT]
BOOT = 2000
THR = 0.5


def gold_block(s, gold):
    y = (gold.gold_label == "OT").to_numpy()
    sg = s.loc[gold.index].to_numpy()
    pr = sg >= THR
    tp, fn, fp, tn = int((pr & y).sum()), int((~pr & y).sum()), int((pr & ~y).sum()), int((~pr & ~y).sum())
    rec, fpr = tp / (tp + fn), fp / (fp + tn)
    out = {"n_OT": int(y.sum()), "n_nonOT": int((~y).sum()), "roc_auc": round(float(roc_auc_score(y, sg)), 3), "pr_auc": round(float(average_precision_score(y, sg)), 3),
           "at_0.5": {"OT_recall": round(rec, 3), "nonOT_flagged": round(fpr, 3), "balanced_acc": round((rec + 1 - fpr) / 2, 3), "confusion_TP_FN_FP_TN": [tp, fn, fp, tn]},
           "threshold_table": {}}
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        p = sg >= t
        r, f = float(p[y].mean()), float(p[~y].mean())
        out["threshold_table"][str(t)] = {"OT_recall": round(r, 3), "nonOT_flagged": round(f, 3), "balanced_acc": round((r + 1 - f) / 2, 3), "youden_J": round(r - f, 3)}
    return out


def main():
    t0 = time.time()
    assert len(F15) == 15 and len(F13) == 13, (len(F15), len(F13))
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    ytrue = (dev.set_index("device_id").label == "OT_confirmed")
    gold = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig").set_index("device_id")
    frozen = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    out = {"status": "A2: stage 1 with 13 features (2 port-count features removed); frozen results untouched", "features_15": F15, "features_13": F13, "removed": PORT,
           "n_devices": int(len(dev)), "n_windows": int(len(x)), "K_train_windows": int(BM.K), "threshold": THR, "variants": {}}
    scores = {}
    for name, feats in (("15_features", F15), ("13_features", F13)):
        BM.FEATS = feats                      # fit_oof cua script 39 doc B.FEATS
        blk = {}
        for fold_col in ("fold_device", "fold_component"):
            s = T39.fit_oof(x, dev, fold_col)
            scores[(name, fold_col)] = s
            yt = ytrue.loc[s.index].to_numpy()
            per_fold = []
            for f in range(5):
                ids = dev.loc[dev[fold_col] == f, "device_id"]
                yy = ytrue.loc[ids].to_numpy()
                if 0 < yy.sum() < len(yy):
                    per_fold.append(float(roc_auc_score(yy, s.loc[ids])))
            blk[fold_col] = {"reference_label_roc_auc_pooled": round(float(roc_auc_score(yt, s)), 3), "reference_label_pr_auc_pooled": round(float(average_precision_score(yt, s)), 3),
                             "roc_auc_per_fold": [round(v, 3) for v in per_fold], "roc_auc_fold_mean": round(float(np.mean(per_fold)), 3),
                             "share_devices_score_ge_0.5": round(float((s >= THR).mean()), 3), "n_devices_score_ge_0.5": int((s >= THR).sum()),
                             "gold_184": gold_block(s, gold)}
            print(name, fold_col, {k: v for k, v in blk[fold_col].items() if k != "gold_184"}, "| gold", blk[fold_col]["gold_184"]["roc_auc"], blk[fold_col]["gold_184"]["at_0.5"], round(time.time() - t0), "s", flush=True)
        out["variants"][name] = blk
    BM.FEATS = F15
    s15, s13 = scores[("15_features", "fold_device")], scores[("13_features", "fold_device")]
    out["reproduction_check_15_features_vs_frozen_oof"] = {"max_abs_diff": float(np.abs(s15 - frozen.loc[s15.index]).max()), "frozen_gold_roc_auc": round(float(roc_auc_score((gold.gold_label == "OT").to_numpy(), frozen.loc[gold.index])), 3)}
    # so sanh ghep cap 15 vs 13 dac trung
    rng = np.random.default_rng(20261125)
    yg = (gold.gold_label == "OT").to_numpy()
    a, b = s15.loc[gold.index].to_numpy(), s13.loc[gold.index].to_numpy()
    d_roc, d_rec, d_fpr = [], [], []
    n = len(yg)
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        if 0 < yg[i].sum() < n:
            d_roc.append(roc_auc_score(yg[i], b[i]) - roc_auc_score(yg[i], a[i]))
            d_rec.append((b[i][yg[i]] >= THR).mean() - (a[i][yg[i]] >= THR).mean())
            d_fpr.append((b[i][~yg[i]] >= THR).mean() - (a[i][~yg[i]] >= THR).mean())
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    ya = ytrue.loc[s15.index].to_numpy()
    d_pop = []
    A_, B_ = s15.to_numpy(), s13.loc[s15.index].to_numpy()
    for _ in range(500):
        i = rng.integers(0, len(ya), len(ya))
        d_pop.append(roc_auc_score(ya[i], B_[i]) - roc_auc_score(ya[i], A_[i]))
    out["paired_13_minus_15"] = {"gold_184_roc_auc_diff": {"point": round(float(roc_auc_score(yg, b) - roc_auc_score(yg, a)), 3), "ci95": ci(d_roc)},
                                 "gold_184_OT_recall_diff_at_0.5": {"point": round(float((b[yg] >= THR).mean() - (a[yg] >= THR).mean()), 3), "ci95": ci(d_rec)},
                                 "gold_184_nonOT_flagged_diff_at_0.5": {"point": round(float((b[~yg] >= THR).mean() - (a[~yg] >= THR).mean()), 3), "ci95": ci(d_fpr)},
                                 "reference_label_roc_auc_diff_fold_device": {"point": round(float(roc_auc_score(ya, B_) - roc_auc_score(ya, A_)), 3), "ci95": ci(d_pop), "bootstrap_B": 500},
                                 "devices_changing_side_of_0.5": int(((A_ >= THR) != (B_ >= THR)).sum())}
    pd.DataFrame({"device_id": s15.index, "lightgbm_15_fold_device": s15.values, "lightgbm_13_fold_device": s13.loc[s15.index].values,
                  "lightgbm_15_fold_component": scores[("15_features", "fold_component")].loc[s15.index].values,
                  "lightgbm_13_fold_component": scores[("13_features", "fold_component")].loc[s15.index].values}).to_csv(F / "phase1_oof_scores_13feat.csv", index=False, encoding="utf-8-sig")
    (F / "phase1_13feat_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["paired_13_minus_15"], indent=1), out["reproduction_check_15_features_vs_frozen_oof"], "DONE-P1", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
