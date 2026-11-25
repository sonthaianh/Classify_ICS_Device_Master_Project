"""Train and save the Phase 1 model used by the web app (LightGBM, 15 behaviour features).

Trained on ALL devices of v6 (K=20 windows per device), default parameters of script 34. Because every training
device is included, scores on the project's own captures are NOT an honest test; the app warns about MAC overlap.
Cross-validated numbers belong to script 34 (see model_meta.json for the reference values).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS / "pcap_to_dataset"))
from feature_contract import TRAINING_FEATURES as D17  # noqa: E402

F = TOOLS / "Dataset/features"
FEATURES = [c for c in D17 if c not in ("ip_count", "frames_sent")]
K, SEED = 20, 20261108
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "label", "cap_rank"] + FEATURES)
    tr = x[x.cap_rank < K]
    y = (tr.label == "OT_confirmed").astype(int)
    spw = float((y == 0).sum() / (y == 1).sum())
    t = time.time()
    model = lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1, **PARAMS).fit(tr[FEATURES], y)
    (HERE / "model").mkdir(exist_ok=True)
    joblib.dump(model, HERE / "model/phase1_lightgbm.joblib")
    reg = pd.read_csv(TOOLS / "Dataset/metadata/devices.csv", dtype=str, keep_default_na=False)
    train_macs = sorted(reg.loc[reg.include_in_behavior_dataset == "True", "mac"])
    cv = json.loads((F / "phase1_lgbm_vs_xgb_results.json").read_text(encoding="utf-8"))["lightgbm|default"]
    meta = {"algorithm": "LightGBM (default parameters of script 34)", "params": PARAMS, "scale_pos_weight": round(spw, 3), "features": FEATURES,
            "window_seconds": 45, "device_score": "mean of window probabilities", "default_threshold": 0.5,
            "train_devices": int(x.device_id.nunique()), "train_windows": int(len(tr)), "K": K, "seed": SEED, "fit_seconds": round(time.time() - t, 2),
            "labels": "nhãn tham chiếu gán theo bằng chứng giao thức và cổng: có giao thức OT hoặc cổng OT ở một trong hai phía thì là OT_confirmed, còn lại là unknown (unknown không có nghĩa là không-OT)",
            "cv_reference_device_level": {"roc_auc": cv["all"]["roc_auc"], "pr_auc": cv["all"]["pr_auc"], "recall@0.5": cv["all"]["recall@0.5"],
                                          "precision@0.5": cv["all"]["precision@0.5"], "portonly_roc_auc": cv["portonly_OT_vs_unknown"]["roc_auc"],
                                          "portonly_pr_auc": cv["portonly_OT_vs_unknown"]["pr_auc"], "prevalence": 0.237,
                                          "note": "5-fold by device, rule-based reference labels, no independent gold set yet"},
            "train_macs": train_macs, "data_sha16": {"fe3_parquet": sha(F / "master_device_windows_v6_fe3.parquet")}}
    try:  # independent-gold / strict-split reference numbers (scripts 36-39); optional
        fu = json.loads((F / "phase1_followups_v6.json").read_text(encoding="utf-8"))
        thr = pd.read_csv(F / "phase1_threshold_union_v6.csv").set_index("threshold").loc[0.5]
        meta["honest_reference"] = {"strict_component_split_pooled_roc": fu["strict_component_split"]["pooled_roc"], "gold_n_ot": fu["gold"]["ot"], "gold_n_nonot": fu["gold"]["nonot"],
                                    "gold_ot_recall@0.5": float(thr.ot_recall), "gold_nonot_flagged@0.5": float(thr.nonot_flagged),
                                    "note": "gold = lab inventory + sonthai-reviewed blind sheet + provisional capture-scenario non-OT; not reviewed by a third party"}
    except FileNotFoundError:
        pass
    (HERE / "model/model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("trained on", meta["train_devices"], "devices,", meta["train_windows"], "windows; scale_pos_weight", meta["scale_pos_weight"])
    print((imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
