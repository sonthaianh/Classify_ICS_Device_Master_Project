"""Phase 1: K comparison, drop-one feature ablation, global explanation, metadata-only baseline.

Same protocol as scripts 34/36 (LightGBM default, 5-fold by device, K=20 unless varied). Every variant is scored on
  * population (weak rule labels): ROC-AUC, PR-AUC
  * independent gold (gold_lab_inventory_v6.csv + gold_nonot_session_gold_v6.csv): ROC-AUC, OT recall@0.3, non-OT FPR@0.3
Global explanation = mean |SHAP| (LightGBM pred_contrib on held-out windows, averaged per device) from the 5 fold models,
plus gain importance from a full fit.  Metadata-only baseline uses n_windows_device only (capture-length artefact check).

Outputs (Dataset/features): phase1_k_compare_v6.csv, phase1_drop_one_v6.csv, phase1_global_importance_v6.csv,
phase1_explain_summary_v6.json.  Window length (45 s vs longer) needs a dataset rebuild and is NOT covered here.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("abl", "32_fe4_ablation.py")
B = _load("cmp", "34_compare_lightgbm_xgboost.py")
F = A.F
THR = 0.3


def load():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label.isin(["OT", "non-OT"])][["device_id_v6", "gold_label"]].rename(columns={"device_id_v6": "device_id"})
    ses = pd.read_csv(F / "gold_nonot_session_gold_v6.csv", encoding="utf-8-sig")[["device_id", "gold_label"]]
    gold = pd.concat([lab, ses]).drop_duplicates("device_id").set_index("device_id").gold_label
    return x, dev, gold


def oof(x, dev, feats, K, shap=False):
    s = pd.Series(np.nan, index=dev.device_id)
    contrib = []
    for f in range(5):
        te = set(dev.loc[dev.fold_device == f, "device_id"])
        tr = x[~x.device_id.isin(te) & (x.cap_rank < K)]
        y = (tr.label == "OT_confirmed").to_numpy().astype(int)
        m = B.make("lightgbm", B.DEFAULT["lightgbm"], y).fit(tr[feats], y)
        t = x[x.device_id.isin(te)]
        p = pd.Series(m.predict_proba(t[feats])[:, 1], index=t.index).groupby(t.device_id).mean()
        s.loc[p.index] = p.values
        if shap:
            c = pd.DataFrame(np.abs(m.predict(t[feats], pred_contrib=True)[:, :-1]), columns=feats)
            c["device_id"] = t.device_id.values
            contrib.append(c.groupby("device_id").mean())
    assert not s.isna().any()
    return (s, pd.concat(contrib)) if shap else s


def metrics(s, dev, gold):
    yp = (dev.set_index("device_id").label == "OT_confirmed").loc[s.index].to_numpy()
    g = gold.loc[gold.index.intersection(s.index)]
    yg = (g == "OT").to_numpy()
    sg = s.loc[g.index]
    return {"pop_roc": round(roc_auc_score(yp, s), 4), "pop_pr": round(average_precision_score(yp, s), 4),
            "gold_roc": round(roc_auc_score(yg, sg), 3), "gold_ot_recall@0.3": round(float((sg[yg] >= THR).mean()), 3),
            "gold_nonot_fpr@0.3": round(float((sg[~yg] >= THR).mean()), 3)}


def main():
    x, dev, gold = load()
    feats = B.FEATS
    out = {"n_gold_ot": int((gold == "OT").sum()), "n_gold_nonot": int((gold == "non-OT").sum())}
    # 1. K comparison
    rows = [{"K": K, "n_train_windows": int((x.cap_rank < K).sum()), **metrics(oof(x, dev, feats, K), dev, gold)} for K in (10, 20, 50, 100)]
    kt = pd.DataFrame(rows)
    kt.to_csv(F / "phase1_k_compare_v6.csv", index=False, encoding="utf-8-sig")
    print("K comparison\n", kt.to_string(index=False))
    # 2. drop-one ablation
    base = metrics(oof(x, dev, feats, B.K), dev, gold)
    rows = [{"dropped": "(none: full 15)", **base}]
    for f in feats:
        rows.append({"dropped": f, **metrics(oof(x, dev, [c for c in feats if c != f], B.K), dev, gold)})
    dt = pd.DataFrame(rows)
    for c in ("pop_roc", "pop_pr", "gold_roc"):
        dt["d_" + c] = (dt[c] - base[c]).round(4)
    dt.to_csv(F / "phase1_drop_one_v6.csv", index=False, encoding="utf-8-sig")
    print("\nDrop-one\n", dt.to_string(index=False))
    # 3. global explanation
    s_full, contrib = oof(x, dev, feats, B.K, shap=True)
    shap_imp = contrib.mean().sort_values(ascending=False)
    tr = x[x.cap_rank < B.K]
    y = (tr.label == "OT_confirmed").to_numpy().astype(int)
    full = B.make("lightgbm", B.DEFAULT["lightgbm"], y).fit(tr[feats], y)
    gain = pd.Series(full.booster_.feature_importance("gain"), index=feats)
    gi = pd.DataFrame({"mean_abs_shap": shap_imp, "shap_share": shap_imp / shap_imp.sum(), "gain_share": (gain / gain.sum()).reindex(shap_imp.index)}).round(4)
    gi.index.name = "feature"
    gi.to_csv(F / "phase1_global_importance_v6.csv", encoding="utf-8-sig")
    print("\nGlobal importance\n", gi.to_string())
    # 4. single-feature and metadata-only baselines (capture-length / source artefacts)
    yp = (dev.set_index("device_id").label == "OT_confirmed")
    single = {f: round(max(roc_auc_score(yp.loc[v.index], v), 1 - roc_auc_score(yp.loc[v.index], v)), 3)
              for f in ("udp_ratio", "pkts_per_flow", "packet_rate") for v in [x.groupby("device_id")[f].median().fillna(0)]}
    meta_feats = ["n_windows_device"]
    meta_s = oof(x, dev, meta_feats, B.K)
    meta = metrics(meta_s, dev, gold)
    out.update({"base_full15": base, "single_feature_device_median_auc": single, "metadata_only_n_windows_device": meta,
                "top5_shap_share": gi.shap_share.head(5).round(3).to_dict()})
    (F / "phase1_explain_summary_v6.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSummary\n", json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
