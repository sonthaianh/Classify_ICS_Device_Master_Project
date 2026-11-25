"""Phase 1: baseline cong trinh truoc + hoc voi nhan duong va chua gan nhan (PU) + nhan hop nhat (2026-11-12). KHONG ghi de ket qua dong bang.

Moi bien the dung cung bo chia (fold_device va fold_component), cung K = 20, cung chi so; cham tren: nhan luat (3.923 thiet bi), gold 184, gold mo rong 243.
Bien the:
  V0_lgbm            LightGBM tham so dong bang (tai hien mo hinh dong bang)
  V1_pu              LightGBM sau khi sua trong so theo Elkan-Noto (nhan unknown = chua gan nhan)
  V2_merged          nhan duong = nhan luat OT_confirmed UNION 48 thiet bi OT tu bang tai san/danh sach 4SICS (chi dung o thiet bi huan luyen cua fold)
  V3_pu_merged       V1 + V2
  B1_svm15           SVM RBF tren trung vi 15 dac trung hanh vi cua thiet bi (phien ban gian luoc theo tinh than Chakraborty 2021: SVM + dac trung luu luong, khong cong/giao thuc)
  B2_oui             chi OUI (3 byte dau cua MAC) -> LightGBM tren ma hoa OUI (theo tinh than Niedermaier 2018)
Phien ban B1/B2 la CAI DAT LAI GIAN LUOC, khong phai ket qua goc cua tac gia.
  py -3.14 81_phase1_baselines_pu.py
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_contract import TRAINING_FEATURES as D17  # noqa: E402

warnings.filterwarnings("ignore")
F = HERE.parent / "Dataset/features"
SEED, K = 20261108, 20
USED = [c for c in D17 if c not in ("ip_count", "frames_sent")]
OTL = "OT_confirmed"
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1, subsample_freq=1, n_jobs=8, verbose=-1)


def fit_lgb(X, y, w=None, spw=True, seed=SEED):
    s = float((y == 0).sum() / max((y == 1).sum(), 1)) if spw else 1.0
    return lgb.LGBMClassifier(scale_pos_weight=s, random_state=seed, **P).fit(X, y, sample_weight=w)


def pu_fit(Xp, Xu, rng, seed=SEED):
    """Elkan-Noto (trong so): Xp = cua so cua thiet bi nhan duong (da tach 20% thiet bi de uoc luong c), Xu = cua so chua gan nhan. Tra ve (mo hinh f, c)."""
    return None


def run_split(split, x, dev, extra_ot, gold184, gold243, dcs_ids):
    fold_of = dict(zip(dev.device_id, dev[split]))
    fx = x.device_id.map(fold_of).to_numpy()
    rule_pos = set(dev.loc[dev.label == OTL, "device_id"])
    merged_pos = rule_pos | extra_ot
    cap = x.cap_rank.to_numpy() < K
    dev_ids = x.device_id.to_numpy()
    Xall = x[USED].to_numpy()
    out = {k: pd.Series(np.nan, index=dev.device_id) for k in ("V0_lgbm", "V1_pu", "V2_merged", "V3_pu_merged")}
    cs = {}
    for f in range(5):
        tr = (fx != f) & cap
        te = fx == f
        tr_dev = pd.unique(dev_ids[tr])
        rng = np.random.default_rng(SEED + f)
        for name, pos_set, use_pu in (("V0_lgbm", rule_pos, False), ("V1_pu", rule_pos, True), ("V2_merged", merged_pos, False), ("V3_pu_merged", merged_pos, True)):
            y = np.isin(dev_ids[tr], list(pos_set)).astype(int)
            Xtr = Xall[tr]
            if not use_pu:
                m = fit_lgb(Xtr, y)
            else:
                pos_dev = np.array([d for d in tr_dev if d in pos_set])
                hold = set(rng.choice(pos_dev, size=max(1, int(0.2 * len(pos_dev))), replace=False))
                in_hold = np.isin(dev_ids[tr], list(hold))
                g_mask = ~in_hold                                  # g: duong (khong thuoc nhom giu lai) so voi chua gan nhan
                g = fit_lgb(Xtr[g_mask], y[g_mask], spw=False)
                gv = g.predict_proba(Xtr[in_hold & (y == 1)])[:, 1]
                c = float(np.clip(gv.mean(), 0.05, 1.0))
                gu = np.clip(g.predict_proba(Xtr[y == 0])[:, 1], 1e-3, 0.99)
                w1 = np.clip(((1 - c) / c) * gu / (1 - gu), 0.0, 1.0)
                Xf = np.vstack([Xtr[y == 1], Xtr[y == 0], Xtr[y == 0]])
                yf = np.r_[np.ones((y == 1).sum()), np.ones((y == 0).sum()), np.zeros((y == 0).sum())].astype(int)
                wf = np.r_[np.ones((y == 1).sum()), w1, 1 - w1]
                m = fit_lgb(Xf, yf, w=wf, spw=False)
                cs[(name, f)] = round(c, 3)
            p = pd.Series(m.predict_proba(Xall[te])[:, 1], index=x.index[te]).groupby(x.device_id[te]).mean()
            out[name].loc[p.index] = p.values
        print(split, "fold", f, "done", flush=True)
    return out, cs


def metrics(s, ids, y):
    sc = s.loc[ids].to_numpy()
    pred = sc >= 0.5
    d = {"n": int(len(ids)), "ot": int(y.sum()), "roc": round(float(roc_auc_score(y, sc)), 4), "pr": round(float(average_precision_score(y, sc)), 4),
         "recall@0.5": round(float((pred & y).sum() / y.sum()), 4)}
    if (~y).sum():
        d["nonOT_flag@0.5"] = round(float((pred & ~y).sum() / (~y).sum()), 4)
    return d


def main():
    t0 = time.time()
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "cap_rank"] + USED)
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype=str, keep_default_na=False)
    for c in ("fold_device", "fold_component"):
        dev[c] = dev[c].astype(int)
    labs = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False).set_index("device_id")
    g184 = pd.read_csv(F / "gold_union_v6.csv").set_index("device_id")
    g243 = pd.read_csv(F / "gold_union_v6_plus_all.csv").set_index("device_id")
    extra = g243[~g243.index.isin(g184.index) & ~g243.index.isin(pd.read_csv(F / "gold_union_v6_plus_4sics.csv").device_id)]
    extra_all = g243[~g243.index.isin(g184.index)]
    extra_ot = set(extra_all.index[extra_all.gold_label == "OT"])
    inv = pd.read_csv(F / "inventory_match_v6.csv", dtype=str, keep_default_na=False).set_index("device_id")
    dcs_ids = [d for d in extra_all.index if d in inv.index and inv.loc[d, "Subtype"] == "DCS"]
    print("extra OT", len(extra_ot), "dcs", len(dcs_ids))
    res = {"protocol": "LightGBM default params, 15 features, K=20; test = devices of held-out fold; PU = Elkan-Noto weighted", "extra_OT_labels": len(extra_ot), "dcs": len(dcs_ids), "splits": {}}
    for split in ("fold_device", "fold_component"):
        out, cs = run_split(split, x, dev, extra_ot, g184, g243, dcs_ids)
        # B1: SVM RBF tren trung vi dac trung thiet bi (luat nhan)
        med = x[x.cap_rank < K].groupby("device_id")[USED].median()
        med = med.reindex(dev.device_id).fillna(-1)
        Xm = np.log1p(np.clip(med.to_numpy(), 0, None)) * 1.0
        yrule = (dev.label == OTL).to_numpy().astype(int)
        fo = dev[split].to_numpy()
        b1 = np.full(len(dev), np.nan)
        for f in range(5):
            tr, te = fo != f, fo == f
            m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)).fit(Xm[tr], yrule[tr])
            b1[te] = m.predict_proba(Xm[te])[:, 1]
        out["B1_svm15"] = pd.Series(b1, index=dev.device_id)
        # B2: chi OUI
        oui = labs.reindex(dev.device_id).device_mac.str.lower().str[:8].fillna("na")
        top = oui.value_counts()
        oui_c = oui.where(oui.map(top) >= 5, "rare")
        codes = oui_c.astype("category").cat.codes.to_numpy()
        b2 = np.full(len(dev), np.nan)
        ohe = pd.get_dummies(oui_c).to_numpy().astype(float)
        for f in range(5):
            tr, te = fo != f, fo == f
            m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=5, scale_pos_weight=float((yrule[tr] == 0).sum() / (yrule[tr] == 1).sum()),
                                   random_state=SEED, n_jobs=8, verbose=-1).fit(ohe[tr], yrule[tr])
            b2[te] = m.predict_proba(ohe[te])[:, 1]
        out["B2_oui"] = pd.Series(b2, index=dev.device_id)
        blk = {"pu_c_estimates": {f"{k[0]}|fold{k[1]}": v for k, v in cs.items()}, "variants": {}}
        for name, s in out.items():
            ids_rule = list(dev.device_id)
            r = {"rule": metrics(s, ids_rule, yrule.astype(bool)),
                 "gold184": metrics(s, list(g184.index), (g184.gold_label == "OT").to_numpy()),
                 "gold243": metrics(s, list(g243.index), (g243.gold_label == "OT").to_numpy())}
            if dcs_ids:
                r["dcs_recall@0.5"] = round(float((s.loc[dcs_ids] >= 0.5).mean()), 3)
                r["dcs_median_score"] = round(float(s.loc[dcs_ids].median()), 3)
            ex_ids = list(extra_all.index)
            r["added_51"] = metrics(s, ex_ids, (extra_all.gold_label == "OT").to_numpy())
            blk["variants"][name] = r
            print(split, name, "rule roc", r["rule"]["roc"], "| g184", r["gold184"]["roc"], r["gold184"]["recall@0.5"], r["gold184"].get("nonOT_flag@0.5"),
                  "| g243", r["gold243"]["roc"], r["gold243"]["recall@0.5"], "| dcs", r.get("dcs_recall@0.5"), flush=True)
        # bootstrap theo thiet bi: hieu ROC (nhan luat va gold 243) so voi V0
        rng = np.random.default_rng(SEED)
        ci = {}
        yg = (g243.gold_label == "OT").to_numpy()
        base_r, base_g = out["V0_lgbm"].loc[dev.device_id].to_numpy(), out["V0_lgbm"].loc[g243.index].to_numpy()
        for name, s in out.items():
            if name == "V0_lgbm":
                continue
            sr, sg = s.loc[dev.device_id].to_numpy(), s.loc[g243.index].to_numpy()
            dr, dg = [], []
            for _ in range(500):
                i = rng.integers(0, len(yrule), len(yrule)); j = rng.integers(0, len(yg), len(yg))
                if 0 < yrule[i].sum() < len(i):
                    dr.append(roc_auc_score(yrule[i], sr[i]) - roc_auc_score(yrule[i], base_r[i]))
                if 0 < yg[j].sum() < len(j):
                    dg.append(roc_auc_score(yg[j], sg[j]) - roc_auc_score(yg[j], base_g[j]))
            q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
            ci[name] = {"rule_roc_diff_vs_V0": q(dr), "gold243_roc_diff_vs_V0": q(dg)}
        blk["bootstrap_vs_V0"] = ci
        res["splits"][split] = blk
        pd.DataFrame(out).to_csv(F / f"phase1_wp3_oof_{split}.csv")
    res["seconds"] = round(time.time() - t0)
    (F / "phase1_wp3_baselines_pu.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("DONE-WP3", res["seconds"], "s")


if __name__ == "__main__":
    main()
