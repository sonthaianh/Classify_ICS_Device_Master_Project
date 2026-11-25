"""Phase 1: LightGBM vs XGBoost under one identical protocol (15 provisional features).

* Outer: 5-fold by device (device_folds_v6.csv fold_device). Train windows capped at cap_rank < K=20,
  test = all windows of held-out devices, window probabilities averaged per device.
* Tuning (identical budget for both): random search, N_CONFIG configs from comparable spaces, scored by
  3-fold device-level inner CV (PR-AUC) INSIDE the outer training devices only. Test folds never seen.
* Imbalance: scale_pos_weight = negatives / positives on the training data (same for both).
* Reports default and tuned versions, subsets (all / protocol / port-only / non flood-like),
  per-fold ROC-AUC, training time, and a paired device-level bootstrap of the difference.
Labels are weak rule labels; this compares algorithms, it does not validate the labels.
"""
import importlib.util
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "32_fe4_ablation.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

FEATS = [c for c in A.D17 if c not in A.SCALE]
SEED, K, N_CONFIG, N_JOBS = A.SEED, A.K, 15, 8
DEFAULT = {"xgboost": dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1),
           "lightgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1)}
SPACE = {"xgboost": dict(n_estimators=[200, 400], learning_rate=[0.03, 0.05, 0.1], max_depth=[3, 4, 6, 8], min_child_weight=[1, 5, 20],
                         subsample=[0.7, 0.9], colsample_bytree=[0.7, 1.0], reg_lambda=[1, 5, 20]),
         "lightgbm": dict(n_estimators=[200, 400], learning_rate=[0.03, 0.05, 0.1], num_leaves=[8, 15, 31, 63, 127], min_child_samples=[5, 20, 50],
                          subsample=[0.7, 0.9], colsample_bytree=[0.7, 1.0], reg_lambda=[1, 5, 20])}


def make(algo, params, y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    if algo == "xgboost":
        return XGBClassifier(scale_pos_weight=spw, tree_method="hist", random_state=SEED, n_jobs=N_JOBS, **params)
    return lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=SEED, n_jobs=N_JOBS, verbose=-1, **params)


def device_scores(model, te):
    return pd.Series(model.predict_proba(te[FEATS])[:, 1], index=te.index).groupby(te.device_id).mean()


def inner_score(algo, params, x, dev_tr, rng_seed):
    """3-fold device-level inner CV, stratified round-robin on label; returns mean PR-AUC."""
    rng = np.random.default_rng(rng_seed)
    d = dev_tr.assign(_f=-1)
    for _, g in d.groupby("label"):
        idx = rng.permutation(g.index.to_numpy())
        d.loc[idx, "_f"] = np.arange(len(idx)) % 3
    scores = []
    for f in range(3):
        va_ids = set(d.loc[d._f == f, "device_id"])
        tr = x[x.device_id.isin(set(d.device_id) - va_ids) & (x.cap_rank < K)]
        va = x[x.device_id.isin(va_ids)]
        y = (tr.label == "OT_confirmed").to_numpy().astype(int)
        m = make(algo, params, y).fit(tr[FEATS], y)
        p = device_scores(m, va)
        yv = (d.set_index("device_id").loc[p.index, "label"] == "OT_confirmed").to_numpy()
        scores.append(average_precision_score(yv, p.to_numpy()))
    return float(np.mean(scores))


def sample_configs(algo, n, seed):
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SPACE[algo].items()} for _ in range(n)]


def run(algo, x, dev, tune):
    oof = pd.Series(np.nan, index=dev.device_id)
    per_fold, fit_secs, chosen = [], [], []
    configs = sample_configs(algo, N_CONFIG, SEED) if tune else None
    for f in range(5):
        test_ids = set(dev.loc[dev.fold_device == f, "device_id"])
        dev_tr = dev[~dev.device_id.isin(test_ids)].reset_index(drop=True)
        params = DEFAULT[algo]
        if tune:
            xtr_all = x[~x.device_id.isin(test_ids)]
            cand = [(inner_score(algo, c, xtr_all, dev_tr, SEED + f), i) for i, c in enumerate(configs)]
            best = max(cand)[1]
            params = configs[best]
            chosen.append({"fold": f, "inner_pr_auc": round(max(cand)[0], 3), **params})
        tr = x[~x.device_id.isin(test_ids) & (x.cap_rank < K)]
        y = (tr.label == "OT_confirmed").to_numpy().astype(int)
        t0 = time.time()
        m = make(algo, params, y).fit(tr[FEATS], y)
        fit_secs.append(time.time() - t0)
        p = device_scores(m, x[x.device_id.isin(test_ids)])
        oof.loc[p.index] = p.values
        d = dev[dev.fold_device == f]
        per_fold.append(roc_auc_score(d.label == "OT_confirmed", oof.loc[d.device_id].to_numpy()))
        print(f"  {algo} tune={tune} fold {f}: auc {per_fold[-1]:.3f}", flush=True)
    score = oof.loc[dev.device_id].to_numpy()
    res = A.subsets(dev, score)
    y = (dev.label == "OT_confirmed").to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    res["recall_at_fpr_5pct"] = round(float(np.interp(0.05, fpr, tpr)), 3)
    res["fold_roc_auc_mean_std"] = [round(float(np.mean(per_fold)), 3), round(float(np.std(per_fold)), 3)]
    res["final_fit_seconds_mean"] = round(float(np.mean(fit_secs)), 2)
    res["chosen_params"] = chosen
    return res, score


def bootstrap(dev, s_a, s_b, n=2000):
    y = (dev.label == "OT_confirmed").to_numpy()
    rng = np.random.default_rng(SEED)
    d_roc, d_pr = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        d_roc.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
        d_pr.append(average_precision_score(y[i], s_a[i]) - average_precision_score(y[i], s_b[i]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"roc_auc_diff_lgbm_minus_xgb[2.5%,mean,97.5%]": ci(d_roc), "pr_auc_diff_lgbm_minus_xgb[2.5%,mean,97.5%]": ci(d_pr)}


def main():
    x = pd.read_parquet(A.F / "master_device_windows_v6_fe3.parquet")
    dev = pd.read_csv(A.F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    dev["flood_like"] = dev.flood_like.astype(str).eq("True")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev["evidence_tier"] = dev.evidence_tier.fillna("")
    out, scores = {"features": FEATS, "K": K, "n_config": N_CONFIG, "prevalence": round(float((dev.label == "OT_confirmed").mean()), 3)}, {}
    for tune in (False, True):
        for algo in ("xgboost", "lightgbm"):
            key = f"{algo}|{'tuned' if tune else 'default'}"
            out[key], scores[key] = run(algo, x, dev, tune)
    out["bootstrap_default"] = bootstrap(dev, scores["lightgbm|default"], scores["xgboost|default"])
    out["bootstrap_tuned"] = bootstrap(dev, scores["lightgbm|tuned"], scores["xgboost|tuned"])
    (A.F / "phase1_lgbm_vs_xgb_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    rows = [{"model": k, "subset": s, **v[s], "recall_at_fpr_5pct": v["recall_at_fpr_5pct"], "fold_auc_mean": v["fold_roc_auc_mean_std"][0],
             "fold_auc_std": v["fold_roc_auc_mean_std"][1], "fit_s": v["final_fit_seconds_mean"]}
            for k, v in out.items() if isinstance(v, dict) and "all" in v for s in ("all", "protocol_OT_vs_unknown", "portonly_OT_vs_unknown", "non_flood")]
    t = pd.DataFrame(rows)
    t.to_csv(A.F / "phase1_lgbm_vs_xgb_results.csv", index=False)
    print(t.to_string(index=False))
    print(json.dumps({k: out[k] for k in ("bootstrap_default", "bootstrap_tuned")}, indent=1))


if __name__ == "__main__":
    main()
