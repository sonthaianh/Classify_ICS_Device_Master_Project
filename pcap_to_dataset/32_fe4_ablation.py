"""FE-4 ablation on the device-level folds (diagnostic, not the final model).

Train: windows of training-fold devices, capped to cap_rank < K, class_weight=balanced.
Test : ALL windows of held-out devices; window probabilities averaged per device.
Metrics are device-level, pooled over the 5 out-of-fold predictions, plus per-fold ROC-AUC spread.
Labels are weak rule labels (protocol/port); label/tier/flood columns are never model inputs.
Subsets: all | protocol OT vs unknown | port-only OT (peer_port, local_port_only) vs unknown | non flood-like.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_contract import TRAINING_FEATURES as D17  # noqa: E402

warnings.filterwarnings("ignore")
F = HERE.parent / "Dataset/features"
K, SEED = 20, 20261108
LOG = ["pkts_per_flow", "bytes_per_packet", "packet_rate", "unique_local_port_count", "svc_port_unique",
       "frames_sent", "arp_frame_count", "arp_target_unique", "icmp_frame_count", "nonip_frame_count", "ip_count"]
PORT = ["svc_port_unique", "unique_local_port_count"]
SCALE = ["ip_count", "frames_sent"]


def model(kind):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=SEED)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))


def cols_for(feats, kind):
    return [("log1p_" + c if (kind == "lr" and c in LOG) else c) for c in feats]


def metrics(d, score):
    y = (d.label == "OT_confirmed").to_numpy()
    out = {"n": int(len(d)), "ot": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["roc_auc"], out["pr_auc"] = round(roc_auc_score(y, score), 3), round(average_precision_score(y, score), 3)
        pred = score >= 0.5
        out["recall@0.5"] = round(float((pred & y).sum() / y.sum()), 3)
        out["precision@0.5"] = round(float((pred & y).sum() / max(pred.sum(), 1)), 3)
    return out


def subsets(dev, score):
    t = dev.evidence_tier
    rows = {"all": dev.index == dev.index,
            "protocol_OT_vs_unknown": t.isin(["protocol", ""]),
            "portonly_OT_vs_unknown": t.isin(["peer_port", "local_port_only", ""]),
            "non_flood": ~dev.flood_like}
    return {k: metrics(dev[m.values if hasattr(m, "values") else m], score[np.asarray(m)]) for k, m in rows.items()}


def run_cv(x, dev, feats, kind, drop_flood_train=False):
    cols = cols_for(feats, kind)
    oof = pd.Series(np.nan, index=dev.device_id)
    per_fold = []
    for f in range(5):
        test_ids = set(dev.loc[dev.fold_device == f, "device_id"])
        tr = x[~x.device_id.isin(test_ids) & (x.cap_rank < K)]
        if drop_flood_train:
            tr = tr[~tr.flood_like]
        te = x[x.device_id.isin(test_ids)]
        m = model(kind).fit(tr[cols], (tr.label == "OT_confirmed").astype(int))
        p = pd.Series(m.predict_proba(te[cols])[:, 1], index=te.index).groupby(te.device_id).mean()
        oof.loc[p.index] = p.values
        d = dev[dev.fold_device == f]
        per_fold.append(roc_auc_score(d.label == "OT_confirmed", oof.loc[d.device_id].to_numpy()))
    score = oof.loc[dev.device_id].to_numpy()
    res = subsets(dev, score)
    res["fold_roc_auc_mean_std"] = [round(float(np.mean(per_fold)), 3), round(float(np.std(per_fold)), 3)]
    return res


def run_lofo(x, dev, family, feats):
    cols = cols_for(feats, "hgb")
    fam_devices = set(x.loc[x.source_family == family, "device_id"])
    test_ids = set(dev.loc[dev.primary_source_family == family, "device_id"])
    tr = x[~x.device_id.isin(fam_devices) & (x.cap_rank < K)]
    te = x[x.device_id.isin(test_ids)]
    m = model("hgb").fit(tr[cols], (tr.label == "OT_confirmed").astype(int))
    p = pd.Series(m.predict_proba(te[cols])[:, 1], index=te.index).groupby(te.device_id).mean()
    d = dev[dev.device_id.isin(test_ids)].set_index("device_id").loc[p.index].reset_index()
    return {"train_devices": int(tr.device_id.nunique()), **metrics(d, p.loc[d.device_id].to_numpy())}


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    src = pd.read_csv(F / "session_source_map_draft.csv", usecols=["session_id", "source_family"])
    x = x.merge(src, on="session_id", how="left")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    dev["flood_like"] = dev.flood_like.astype(str).eq("True")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev["evidence_tier"] = dev.evidence_tier.fillna("")
    prevalence = float((dev.label == "OT_confirmed").mean())
    sets = {"D17_all": D17,
            "D17_minus_port_features": [c for c in D17 if c not in PORT],
            "D17_minus_scale_features": [c for c in D17 if c not in SCALE],
            "D17_minus_port_and_scale": [c for c in D17 if c not in PORT + SCALE],
            "baseline_only_svc_port_unique": ["svc_port_unique"]}
    results = {"prevalence_pr_auc_baseline": round(prevalence, 3), "K": K, "cv": {}, "lofo": {}}
    for name, feats in sets.items():
        for kind in (("hgb", "lr") if name.startswith("D17") else ("hgb",)):
            results["cv"][f"{name}|{kind}"] = run_cv(x, dev, feats, kind)
            print(name, kind, results["cv"][f"{name}|{kind}"]["all"], flush=True)
    results["cv"]["D17_all|hgb|train_without_flood"] = run_cv(x, dev, D17, "hgb", drop_flood_train=True)
    for fam in ["eth2dump_attack_lab", "anonymized_hourly", "switch_series_siw_swi", "4sics"]:
        results["lofo"][fam] = run_lofo(x, dev, fam, D17)
        print("LOFO", fam, results["lofo"][fam], flush=True)
    (F / "fe4_ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for k, v in results["cv"].items():
        for s in ("all", "protocol_OT_vs_unknown", "portonly_OT_vs_unknown", "non_flood"):
            rows.append({"experiment": k, "subset": s, **v[s]})
    pd.DataFrame(rows).to_csv(F / "fe4_ablation_results.csv", index=False)
    print(json.dumps(results["lofo"], indent=1))


if __name__ == "__main__":
    main()
