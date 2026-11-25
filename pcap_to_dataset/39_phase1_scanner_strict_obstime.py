"""Phase 1 follow-ups (2026-11-09):

  (1) gold union  : lab inventory + capture-scenario non-OT + blind-sheet annotation (reviewed by sonthai, medium/high confidence only).
                    Written to gold_union_v6.csv (source + confidence kept; priority lab > annotation > capture-scenario).
  (2) scanner taxonomy: OT = PLC/RTU/IED/DCS/IO/HMI/SCADA/engineering WS; scanners / monitoring / attack tools are NOT OT.
                    Experiment: devices with >= SCAN_PORTS service ports in one session are 'scanner suspects'; compare training
                    labels A = original, B = drop suspects from training, C = relabel suspects as non-OT (0).
  (3) strict split: 5-fold by connected component of shared sessions (fold_component) instead of fold_device.
  (4) observation time: mean of the first n 45-s window probabilities per device (n = 1,2,3,5,10,all), as a stand-in for longer
                    windows (which need a dataset rebuild).
Outputs (Dataset/features): gold_union_v6.csv, phase1_scanner_variants_v6.csv, phase1_strict_split_v6.csv,
phase1_obstime_v6.csv, phase1_followups_v6.json.
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
SCAN_PORTS = 50
THRS = (0.3, 0.5)


def build_gold_union():
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label.isin(["OT", "non-OT"])]
    g1 = pd.DataFrame({"device_id": lab.device_id_v6, "gold_label": lab.gold_label, "source": "lab_inventory", "confidence": lab.gold_confidence})
    ann = pd.read_csv(F / "gold_candidates_v6_blind_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_label.isin(["OT", "non-OT"]) & ann.gold_confidence.isin(["high", "medium"])]
    g2 = pd.DataFrame({"device_id": ann.device_id, "gold_label": ann.gold_label, "source": "blind_sheet_reviewed_sonthai", "confidence": ann.gold_confidence})
    ses = pd.read_csv(F / "gold_nonot_session_gold_v6.csv", encoding="utf-8-sig")
    g3 = pd.DataFrame({"device_id": ses.device_id, "gold_label": "non-OT", "source": "capture_scenario_provisional", "confidence": ses.gold_confidence})
    g = pd.concat([g1, g2, g3], ignore_index=True)
    conflicts = g.groupby("device_id").gold_label.nunique()
    conflicts = conflicts[conflicts > 1].index.tolist()
    g = g.drop_duplicates("device_id", keep="first")  # priority order of concat
    g = g[~g.device_id.isin(conflicts)]
    g.to_csv(F / "gold_union_v6.csv", index=False, encoding="utf-8-sig")
    return g.set_index("device_id"), conflicts


def fit_oof(x, dev, fold_col, train_mask_fn=None, relabel=None, window_probs=False):
    """OOF device scores; optional training-label variant. Returns scores (and window-level probs)."""
    s = pd.Series(np.nan, index=dev.device_id)
    wp = []
    for f in range(5):
        te = set(dev.loc[dev[fold_col] == f, "device_id"])
        tr = x[~x.device_id.isin(te) & (x.cap_rank < B.K)]
        if train_mask_fn is not None:
            tr = tr[train_mask_fn(tr)]
        lab = tr.label == "OT_confirmed"
        if relabel is not None:
            lab = lab & ~tr.device_id.isin(relabel)
        y = lab.to_numpy().astype(int)
        m = B.make("lightgbm", B.DEFAULT["lightgbm"], y).fit(tr[B.FEATS], y)
        t = x[x.device_id.isin(te)]
        p = pd.Series(m.predict_proba(t[B.FEATS])[:, 1], index=t.index)
        s.loc[p.groupby(t.device_id).mean().index] = p.groupby(t.device_id).mean().values
        if window_probs:
            wp.append(pd.DataFrame({"device_id": t.device_id.values, "window_start": t.window_start.values, "p": p.values}))
    assert not s.isna().any()
    return (s, pd.concat(wp)) if window_probs else s


def gold_metrics(s, gold, label):
    g = gold.loc[gold.index.intersection(s.index)]
    y = (g.gold_label == "OT").to_numpy()
    sg = s.loc[g.index].to_numpy()
    r = {"variant": label, "gold_n_ot": int(y.sum()), "gold_n_nonot": int((~y).sum()), "gold_roc": round(roc_auc_score(y, sg), 3),
         "gold_pr": round(average_precision_score(y, sg), 3)}
    for t in THRS:
        r[f"ot_recall@{t}"] = round(float((sg[y] >= t).mean()), 3)
        r[f"nonot_flagged@{t}"] = round(float((sg[~y] >= t).mean()), 3)
    return r


def main():
    gold, conflicts = build_gold_union()
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig").set_index("device_id")
    suspects = set(lab.index[lab.max_service_ports_in_a_session >= SCAN_PORTS])
    ytrue = (dev.set_index("device_id").label == "OT_confirmed")
    out = {"gold": {"n": int(len(gold)), "ot": int((gold.gold_label == "OT").sum()), "nonot": int((gold.gold_label == "non-OT").sum()),
                    "by_source": gold.groupby(["source", "gold_label"]).size().to_dict().__repr__(), "conflicts_dropped": conflicts},
           "scanner_suspects": {"threshold_ports": SCAN_PORTS, "n": len(suspects), "pct_rule_ot": round(float(ytrue.loc[list(suspects)].mean()), 3),
                                "gold_ot_among_suspects": int(gold.index.isin(suspects)[(gold.gold_label == "OT").to_numpy()].sum()),
                                "gold_nonot_among_suspects": int(gold.index.isin(suspects)[(gold.gold_label == "non-OT").to_numpy()].sum())}}
    # (2) scanner variants
    rows = []
    sc = {}
    sc["A_original"] = fit_oof(x, dev, "fold_device")
    sc["B_drop_suspects"] = fit_oof(x, dev, "fold_device", train_mask_fn=lambda t: ~t.device_id.isin(suspects))
    sc["C_suspects_as_nonOT"] = fit_oof(x, dev, "fold_device", relabel=suspects)
    keep = ~dev.device_id.isin(suspects).to_numpy()
    for name, s in sc.items():
        r = gold_metrics(s, gold, name)
        yy = ytrue.loc[s.index]
        r["pop_roc_excl_suspects"] = round(roc_auc_score(yy[~s.index.isin(suspects)], s[~s.index.isin(suspects)]), 4)
        r["pop_pr_excl_suspects"] = round(average_precision_score(yy[~s.index.isin(suspects)], s[~s.index.isin(suspects)]), 4)
        r["suspects_scored>=0.5"] = round(float((s[list(suspects)] >= 0.5).mean()), 3)
        rows.append(r)
    sv = pd.DataFrame(rows)
    sv.to_csv(F / "phase1_scanner_variants_v6.csv", index=False, encoding="utf-8-sig")
    print("SCANNER VARIANTS\n", sv.to_string(index=False))
    # (3) strict split
    s_strict, _ = fit_oof(x, dev, "fold_component", window_probs=True)
    per = []
    for f in range(5):
        d = dev[dev.fold_component == f]
        yy = (d.label == "OT_confirmed").to_numpy()
        per.append({"fold_component": f, "n_devices": len(d), "ot_share": round(float(yy.mean()), 3),
                    "roc": round(roc_auc_score(yy, s_strict.loc[d.device_id]), 3) if 0 < yy.sum() < len(yy) else None})
    st = pd.DataFrame(per)
    st.to_csv(F / "phase1_strict_split_v6.csv", index=False, encoding="utf-8-sig")
    yy = ytrue.loc[s_strict.index]
    out["strict_component_split"] = {"pooled_roc": round(roc_auc_score(yy, s_strict), 4), "pooled_pr": round(average_precision_score(yy, s_strict), 4),
                                     **{k: v for k, v in gold_metrics(s_strict, gold, "strict").items() if k != "variant"}}
    print("\nSTRICT (fold_component)\n", st.to_string(index=False), "\n", out["strict_component_split"])
    # (4) observation time (fold_device models; window-level probabilities)
    sA, wp = fit_oof(x, dev, "fold_device", window_probs=True)
    wp = wp.sort_values(["device_id", "window_start"])
    wp["rank"] = wp.groupby("device_id").cumcount()
    rows = []
    for n in (1, 2, 3, 5, 10, 999):
        sub = wp[wp["rank"] < n]
        cnt = sub.groupby("device_id").size()
        ids = cnt.index if n == 999 else cnt[cnt >= n].index
        s = sub[sub.device_id.isin(ids)].groupby("device_id").p.mean()
        yy = ytrue.loc[s.index]
        r = {"first_n_windows": "all" if n == 999 else n, "approx_seconds": "all" if n == 999 else 45 * n, "n_devices": len(s),
             "pop_roc": round(roc_auc_score(yy, s), 4) if 0 < yy.sum() < len(yy) else None}
        g = gold.loc[gold.index.intersection(s.index)]
        yg = (g.gold_label == "OT").to_numpy()
        r["n_gold"] = len(g)
        r["gold_roc"] = round(roc_auc_score(yg, s.loc[g.index]), 3) if 0 < yg.sum() < len(yg) else None
        rows.append(r)
    ot_ = pd.DataFrame(rows)
    ot_.to_csv(F / "phase1_obstime_v6.csv", index=False, encoding="utf-8-sig")
    print("\nOBS TIME\n", ot_.to_string(index=False))
    # (5) final threshold table on the union gold (model A = original labels, fold_device OOF)
    rng = np.random.default_rng(20261109)
    g = gold.loc[gold.index.intersection(sA.index)]
    yg = (g.gold_label == "OT").to_numpy()
    sg = sA.loc[g.index].to_numpy()
    rows = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        tp = (sg[yg] >= t)
        fp = (sg[~yg] >= t)
        bt = [(tp[rng.integers(0, len(tp), len(tp))].mean(), fp[rng.integers(0, len(fp), len(fp))].mean()) for _ in range(2000)]
        tpr, fpr = float(tp.mean()), float(fp.mean())
        rows.append({"threshold": t, "ot_recall": round(tpr, 3), "ot_recall_ci95": [round(float(np.percentile([b[0] for b in bt], 2.5)), 3), round(float(np.percentile([b[0] for b in bt], 97.5)), 3)],
                     "nonot_flagged": round(fpr, 3), "nonot_flagged_ci95": [round(float(np.percentile([b[1] for b in bt], 2.5)), 3), round(float(np.percentile([b[1] for b in bt], 97.5)), 3)],
                     "balanced_accuracy": round((tpr + 1 - fpr) / 2, 3), "youden_j": round(tpr - fpr, 3),
                     "est_precision_at_prev_0.237": round(tpr * 0.237 / max(tpr * 0.237 + fpr * 0.763, 1e-9), 3)})
    thr = pd.DataFrame(rows)
    thr.to_csv(F / "phase1_threshold_union_v6.csv", index=False, encoding="utf-8-sig")
    print("\nTHRESHOLDS on union gold\n", thr.to_string(index=False))
    by_src = []
    for (src, lbl), d in g.groupby(["source", "gold_label"]):
        sc_ = sA.loc[d.index]
        by_src.append({"source": src, "gold_label": lbl, "n": len(d), "pct_scored>=0.5": round(float((sc_ >= 0.5).mean()), 3), "median_score": round(float(sc_.median()), 3)})
    out["by_source_at_0.5"] = by_src
    print(pd.DataFrame(by_src).to_string(index=False))
    (F / "phase1_followups_v6.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
