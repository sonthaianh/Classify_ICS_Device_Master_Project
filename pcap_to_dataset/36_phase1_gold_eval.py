"""Phase 1: out-of-fold scores, independent-gold evaluation and threshold table (reproducible; replaces scratch scripts).

Protocol = script 34 default: 15 features, K=20 windows/device for training, 5-fold by device (fold_device), device
score = mean window probability of a device that was NOT in the training fold. LightGBM (main) and XGBoost (control).

Gold sets (all independent of the protocol/port rule):
  * gold_lab_inventory_v6.csv  : lab device table given by the owner (22 OT, 2 non-OT Neuralyzer servers; .141 = unknown, excluded).
  * capture-scenario non-OT    : devices that appear ONLY in generic IT-protocol demo captures where the rule found no OT device.
    PROVISIONAL: candidates from `gold_nonot_session_candidates_v6.csv` minus sessions judged ambiguous (AMBIGUOUS regex below).
    The owner has not reviewed the session list yet -> status PROVISIONAL_AUTO_PENDING_USER_REVIEW.

Outputs (Dataset/features): phase1_oof_scores_v6.csv, gold_nonot_session_gold_v6.csv, phase1_gold_eval_v6.json,
phase1_threshold_table_v6.csv.  Labels in the population are weak rule labels; gold numbers are small-sample (see CI).
"""
import importlib.util
import json
import re
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
SEED = 20261109
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
AMBIGUOUS = re.compile(r"(snmp_ntp|syslog|dicom|vrrp|openvpn)", re.I)  # sessions that may come from ICS/medical gear
PREVALENCE = 0.237  # share of rule-OT among the 3,923 devices (used for the model-based precision estimate only)


def oof_scores(x, dev, algo):
    s = pd.Series(np.nan, index=dev.device_id)
    for f in range(5):
        te = set(dev.loc[dev.fold_device == f, "device_id"])
        tr = x[~x.device_id.isin(te) & (x.cap_rank < B.K)]
        y = (tr.label == "OT_confirmed").to_numpy().astype(int)
        m = B.make(algo, B.DEFAULT[algo], y).fit(tr[B.FEATS], y)
        p = B.device_scores(m, x[x.device_id.isin(te)])
        s.loc[p.index] = p.values
    assert not s.isna().any(), "every device must be scored by exactly one held-out fold"
    return s


def build_gold(x):
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label.isin(["OT", "non-OT"])]
    gold = lab[["device_id_v6", "gold_label"]].rename(columns={"device_id_v6": "device_id"}).assign(source="lab_inventory")
    sess = x.drop_duplicates(["device_id", "session_id"]).groupby("device_id").session_id.apply(sorted)
    cand = pd.read_csv(F / "gold_nonot_session_candidates_v6.csv", encoding="utf-8-sig")
    keep = [d for d in cand.device_id if not any(AMBIGUOUS.search(s) for s in sess[d]) and d not in set(gold.device_id)]
    ses = pd.DataFrame({"device_id": keep})
    ses["device_mac"] = ses.device_id.map(cand.set_index("device_id").device_mac)
    ses["sessions"] = ses.device_id.map(lambda d: " | ".join(sess[d]))
    ses["gold_label"] = "non-OT"
    ses["gold_basis"] = "capture_scenario"
    ses["gold_confidence"] = "medium"
    ses["status"] = "PROVISIONAL_AUTO_PENDING_USER_REVIEW"
    ses.to_csv(F / "gold_nonot_session_gold_v6.csv", index=False, encoding="utf-8-sig")
    gold = pd.concat([gold, ses[["device_id", "gold_label"]].assign(source="capture_scenario")], ignore_index=True)
    return gold, len(cand), len(keep)


def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def boot_rate(flags, n=2000):
    rng = np.random.default_rng(SEED)
    flags = np.asarray(flags, float)
    return ci([flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(n)])


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    x["evidence_tier"] = x.evidence_tier.fillna("")
    sc = pd.DataFrame({a: oof_scores(x, dev, a) for a in ("lightgbm", "xgboost")})
    sc.index.name = "device_id"
    out = sc.join(dev.set_index("device_id")[["label", "evidence_tier", "fold_device", "n_windows", "n_sessions"]])
    out.round(5).reset_index().to_csv(F / "phase1_oof_scores_v6.csv", index=False, encoding="utf-8-sig")

    gold, n_cand, n_keep = build_gold(x)
    gold = gold.drop_duplicates("device_id").set_index("device_id")
    assert gold.index.isin(sc.index).all(), "gold device missing from v6"
    s = sc.lightgbm
    ot_ids = gold.index[gold.gold_label == "OT"]
    no_ids = gold.index[gold.gold_label == "non-OT"]
    scan_ids = ["DEV_0691", "DEV_0265"]  # Neuralyzer servers that actively probe OT ports
    no_clean = [d for d in no_ids if d not in scan_ids]
    yp = (dev.set_index("device_id").label == "OT_confirmed").loc[s.index]
    rows = []
    for t in THRESHOLDS:
        tpr = float((s[ot_ids] >= t).mean())
        fpr_all = float((s[no_ids] >= t).mean())
        fpr_clean = float((s[no_clean] >= t).mean())
        prec_est = tpr * PREVALENCE / max(tpr * PREVALENCE + fpr_all * (1 - PREVALENCE), 1e-9)
        pred = s >= t
        tp = int((pred & yp).sum()); fp = int((pred & ~yp).sum()); fn = int((~pred & yp).sum()); tn = int((~pred & ~yp).sum())
        rows.append({"threshold": t,
                     "gold_ot_recall": round(tpr, 3), "gold_ot_recall_ci95": boot_rate((s[ot_ids] >= t)),
                     "gold_nonot_fpr_all": round(fpr_all, 3), "gold_nonot_fpr_ci95": boot_rate((s[no_ids] >= t)),
                     "gold_nonot_fpr_excl_scanner_servers": round(fpr_clean, 3),
                     "gold_balanced_accuracy": round((tpr + 1 - fpr_all) / 2, 3),
                     "est_precision_at_prevalence_0.237": round(prec_est, 3),
                     "rule_recall": round(tp / (tp + fn), 3), "rule_precision": round(tp / max(tp + fp, 1), 3),
                     "rule_accuracy": round((tp + tn) / len(s), 3), "rule_f1": round(2 * tp / (2 * tp + fp + fn), 3),
                     "pct_devices_flagged_OT": round(float(pred.mean()), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(F / "phase1_threshold_table_v6.csv", index=False, encoding="utf-8-sig")

    yg = (gold.gold_label == "OT").astype(int).to_numpy()
    both = gold.index
    res = {"protocol": "script 34 default; 15 features; K=20; 5-fold by device; LightGBM main",
           "n_devices": int(len(s)), "gold": {"ot_lab": int(len(ot_ids)), "nonot_lab_servers": int(len(scan_ids)),
                                             "nonot_capture_scenario_provisional": int(len(no_ids) - len(scan_ids)),
                                             "candidates_before_ambiguity_filter": n_cand, "kept_after_filter": n_keep},
           "gold_roc_auc": round(roc_auc_score(yg, s[both]), 3), "gold_pr_auc": round(average_precision_score(yg, s[both]), 3),
           "gold_roc_auc_without_scanner_servers": round(roc_auc_score(yg[~both.isin(scan_ids)], s[both][~both.isin(scan_ids)]), 3),
           "gold_nonot_score_median": round(float(s[no_ids].median()), 3), "gold_ot_score_median": round(float(s[ot_ids].median()), 3),
           "population_roc_auc": round(roc_auc_score(yp, s), 3), "population_pr_auc": round(average_precision_score(yp, s), 3),
           "xgboost_gold_roc_auc": round(roc_auc_score(yg, sc.xgboost[both]), 3),
           "caveat": "gold small (22 OT / few non-OT); capture-scenario non-OT is provisional; lab devices share sessions in training folds"}
    (F / "phase1_gold_eval_v6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(res, indent=2))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
