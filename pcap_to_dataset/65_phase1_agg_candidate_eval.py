"""Phase 1 - ung vien chinh: LightGBM tren feature GOP THEO THIET BI (mean/std/min/max cua <= K=20 cua so dau), 2026-11-11.

Script 64 chi bao cao so tong hop; script nay tao DIEM OOF thiet bi cua ung vien (3 hat, trung binh), danh gia tren GOLD HOP NHAT (gold_union_v6.csv),
chon nguong theo cung quy tac da dung truoc do (cuc dai Youden J tren gold hop nhat), va so voi mo hinh cua so dong bang (phase1_oof_scores_v6.csv).
Bien the ung vien duoc CHON THEO CHI SO TREN NHAN LUAT (ROC/PR theo thiet bi va chia chat), KHONG theo gold: all_mean_std_min_max (60 cot);
mean_only (15 cot) la doi chung gon nhat.
Dau ra (Dataset/features): phase1_agg_oof_scores_v6.csv, phase1_agg_threshold_union_v6.csv, phase1_agg_vs_window_v6.json
Chay: py -3.14 Tools/pcap_to_dataset/65_phase1_agg_candidate_eval.py
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
s = importlib.util.spec_from_file_location("s64", HERE / "64_phase1_device_agg_gbm.py")
S64 = importlib.util.module_from_spec(s)
s.loader.exec_module(S64)
S63, dev, F, SEED, B = S64.S63, S64.dev, S64.F, S64.SEED, S64.B
THRS = [round(t, 2) for t in np.arange(0.20, 0.751, 0.05)]
NBOOT = 2000


def oof(T, fold_col, y):
    per = []
    for sd in range(S63.SEEDS):
        sc = pd.Series(np.nan, index=dev.device_id)
        for f in range(5):
            te = (dev[fold_col] == f).to_numpy()
            ytr = y[~te]
            m = lgb.LGBMClassifier(scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)), subsample_freq=1, random_state=SEED + sd, n_jobs=8,
                                   verbose=-1, **B.DEFAULT["lightgbm"]).fit(T[~te], ytr)
            sc.loc[dev.device_id[te]] = m.predict_proba(T[te])[:, 1]
        assert not sc.isna().any()
        per.append(sc)
    return pd.concat(per, axis=1).mean(axis=1)


def rates(score, yg, t):
    p = score >= t
    return float(p[yg == 1].mean()), float(p[yg == 0].mean())  # recall, non-OT flagged


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def table(score, yg, tag):
    rows = []
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(yg), len(yg)) for _ in range(NBOOT)]
    for t in THRS:
        r, f = rates(score, yg, t)
        bs = [rates(score[i], yg[i], t) for i in idx if 0 < yg[i].sum() < len(i)]
        rows.append({"model": tag, "threshold": t, "ot_recall": round(r, 3), "ot_recall_ci95": ci([b[0] for b in bs]), "nonot_flagged": round(f, 3),
                     "nonot_flagged_ci95": ci([b[1] for b in bs]), "youden_j": round(r - f, 3), "balanced_accuracy": round((r + 1 - f) / 2, 3)})
    return pd.DataFrame(rows)


def recall_at_fpr(score, yg, target):
    """OT recall at the threshold whose non-OT flagged rate is the largest value <= target (apples-to-apples comparison between models)."""
    best = (0.0, 1.0)
    for t in np.unique(score):
        r, f = rates(score, yg, t)
        if f <= target and r >= best[0]:
            best = (r, t)
    return round(best[0], 3)


def main():
    y = (dev.label == "OT_confirmed").astype(int).to_numpy()
    T = S64.sel(S64.agg_table().loc[dev.device_id], S64.VARIANTS["all_mean_std_min_max"])
    Tm = S64.sel(S64.agg_table().loc[dev.device_id], S64.VARIANTS["mean_only"])
    sc = {"agg60_device": oof(T, "fold_device", y), "agg60_strict": oof(T, "fold_component", y),
          "agg15mean_device": oof(Tm, "fold_device", y), "agg15mean_strict": oof(Tm, "fold_component", y)}
    old = pd.read_csv(F / "phase1_oof_scores_v6.csv").set_index("device_id").loc[dev.device_id]
    out = pd.DataFrame({"device_id": dev.device_id.to_numpy(), "label": dev.label.to_numpy(), "fold_device": dev.fold_device.to_numpy(),
                        "window_lightgbm_frozen": old.lightgbm.to_numpy()})
    for k, v in sc.items():
        out[k] = v.loc[dev.device_id].to_numpy()
    out.round(5).to_csv(F / "phase1_agg_oof_scores_v6.csv", index=False, encoding="utf-8-sig")

    g = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig").set_index("device_id")
    o = out.set_index("device_id")
    yg = (g.gold_label == "OT").astype(int).to_numpy()
    res = {"status": "CANDIDATE evaluation (selection of variant on rule-label metrics, not on gold)", "n_gold": int(len(g)), "gold_ot": int(yg.sum()), "gold_nonot": int((1 - yg).sum()),
           "population": {}, "gold": {}}
    yp = y
    for k in ("window_lightgbm_frozen", "agg60_device", "agg15mean_device", "agg60_strict", "agg15mean_strict"):
        res["population"][k] = {"roc": round(float(roc_auc_score(yp, o[k])), 3), "pr": round(float(average_precision_score(yp, o[k])), 3)}
    tabs = []
    for k in ("window_lightgbm_frozen", "agg60_device", "agg15mean_device", "agg60_strict"):
        sg = o.loc[g.index, k].to_numpy()
        t = table(sg, yg, k)
        tabs.append(t)
        jbest = t.loc[t.youden_j.idxmax()]
        res["gold"][k] = {"roc": round(float(roc_auc_score(yg, sg)), 3), "pr": round(float(average_precision_score(yg, sg)), 3),
                          "best_threshold_youden": float(jbest.threshold), "best_youden_j": float(jbest.youden_j),
                          "recall_at_nonot_flagged_<=0.20": recall_at_fpr(sg, yg, 0.20), "recall_at_nonot_flagged_<=0.28": recall_at_fpr(sg, yg, 0.28)}
        by = {}
        for src, gi in g.groupby("source").groups.items():
            ii = g.index.get_indexer(gi)
            if 0 < yg[ii].sum() < len(ii):
                by[src] = round(float(roc_auc_score(yg[ii], sg[ii])), 3)
        res["gold"][k]["roc_by_source_when_both_classes"] = by
    pd.concat(tabs).to_csv(F / "phase1_agg_threshold_union_v6.csv", index=False, encoding="utf-8-sig")

    # paired bootstrap on gold: candidate - frozen window model
    rng = np.random.default_rng(SEED)
    s_old = o.loc[g.index, "window_lightgbm_frozen"].to_numpy()
    for k in ("agg60_device", "agg15mean_device"):
        s_new = o.loc[g.index, k].to_numpy()
        dr, dp = [], []
        for _ in range(NBOOT):
            i = rng.integers(0, len(yg), len(yg))
            if 0 < yg[i].sum() < len(i):
                dr.append(roc_auc_score(yg[i], s_new[i]) - roc_auc_score(yg[i], s_old[i]))
                dp.append(average_precision_score(yg[i], s_new[i]) - average_precision_score(yg[i], s_old[i]))
        res["gold"][k]["delta_vs_frozen_window"] = {"roc": [round(float(np.mean(dr)), 3), *ci(dr)], "pr": [round(float(np.mean(dp)), 3), *ci(dp)]}

    # population consequences at the candidate's own chosen threshold
    tb = pd.concat(tabs)
    tstar = float(tb[tb.model == "agg60_device"].sort_values(["youden_j", "threshold"], ascending=[False, True]).iloc[0].threshold)
    flag_old = (o.window_lightgbm_frozen >= 0.5)
    flag_new = (o.agg60_device >= tstar)
    res["population_flagged"] = {"threshold_candidate": tstar, "frozen_window@0.5": int(flag_old.sum()), "agg60@candidate": int(flag_new.sum()),
                                 "both": int((flag_old & flag_new).sum()), "only_old": int((flag_old & ~flag_new).sum()), "only_new": int((~flag_old & flag_new).sum()),
                                 "rule_OT_recall_candidate": round(float(flag_new[y == 1].mean()), 3), "rule_precision_candidate": round(float(y[flag_new.to_numpy()].mean()), 3)}
    (F / "phase1_agg_vs_window_v6.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(pd.concat(tabs)[["model", "threshold", "ot_recall", "nonot_flagged", "youden_j"]].to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
