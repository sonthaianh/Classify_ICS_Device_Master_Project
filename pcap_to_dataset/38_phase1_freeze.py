"""Phase 1 freeze (FE-5, PROVISIONAL): hash data/labels/folds/gold/features/params so results can be reproduced and audited.

  py -3.14 38_phase1_freeze.py          -> writes Dataset/features/phase1_freeze_manifest.json
  py -3.14 38_phase1_freeze.py --verify -> recomputes every hash and runs consistency checks (exit 1 on any failure)

Status stays 'PROVISIONAL' until the owner (a) reviews the capture-scenario non-OT session list, (b) annotates the blind sheet,
(c) reviews source groups.  Not frozen: window length (45 s) comparison, source-group leave-one-out.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
MANIFEST = F / "phase1_freeze_manifest.json"
FILES = ["master_device_windows_v6_behavior.parquet", "master_device_windows_v6_fe3.parquet", "device_labels_v6_rule.csv", "device_folds_v6.csv",
         "gold_lab_inventory_v6.csv", "gold_nonot_session_candidates_v6.csv", "gold_nonot_session_gold_v6.csv", "phase1_oof_scores_v6.csv",
         "phase1_threshold_table_v6.csv", "phase1_gold_eval_v6.json", "phase1_k_compare_v6.csv", "phase1_drop_one_v6.csv",
         "phase1_global_importance_v6.csv", "phase1_explain_summary_v6.json",
         "gold_candidates_v6_blind_reviewed_sonthai.csv", "gold_union_v6.csv", "phase1_scanner_variants_v6.csv", "phase1_strict_split_v6.csv",
         "phase1_obstime_v6.csv", "phase1_threshold_union_v6.csv", "phase1_followups_v6.json"]
SCRIPTS = ["29_make_fe3_view.py", "31_make_folds.py", "32_fe4_ablation.py", "34_compare_lightgbm_xgboost.py", "36_phase1_gold_eval.py",
           "37_phase1_explain_k_ablation.py", "39_phase1_scanner_strict_obstime.py", "38_phase1_freeze.py"]
BIG = {"device_session_review_v6.csv"}  # 2.6 GB: size only


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def make():
    import lightgbm, numpy, sklearn, xgboost
    B = HERE / "34_compare_lightgbm_xgboost.py"
    ns = {}
    # read feature list/params without importing heavy modules twice: execute only for constants via import
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmp", B)
    cmp_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_)
    m = {"status": "PROVISIONAL", "created_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
         "protocol": {"features": cmp_.FEATS, "K": cmp_.K, "seed": cmp_.SEED, "default_params": cmp_.DEFAULT, "folds": "fold_device (5-fold by device)",
                      "window_seconds": 45, "threshold_candidates": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "threshold_chosen": 0.5, "threshold_rationale": "max Youden J on union gold (0.569); 0.3 adds only +1.6 pt recall but nearly doubles non-OT flagged (0.28 -> 0.52); 0.3 kept only as optional high-recall review list"},
         "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__,
                 "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
         "files": {f: {"sha256": sha(F / f), "bytes": (F / f).stat().st_size} for f in FILES},
         "scripts": {s: sha(HERE / s) for s in SCRIPTS},
         "large_files_size_only": {f: (F / f).stat().st_size for f in BIG},
         "open_items": ["domain-owner review of sonthai-reviewed blind sheet (118 labelled / 102 khong ro) and of the capture-scenario non-OT sessions", "source provenance of PCAPs (3,618 devices unassigned)", "new PCAPs outside the 2,230 files for an independent app test", "true longer-window comparison needs a dataset rebuild (obs-time study used as proxy)", "scanner detector: no behaviour-only rule separates scanners from OT (taxonomy: scanner = non-OT; manual list only)"]}
    MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print("manifest written:", MANIFEST, "|", len(FILES), "files,", len(SCRIPTS), "scripts")


def verify():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for f, d in m["files"].items():
        chk(sha(F / f) == d["sha256"], f"hash {f}")
    for s, h in m["scripts"].items():
        chk(sha(HERE / s) == h, f"hash script {s}")
    for f, n in m["large_files_size_only"].items():
        chk((F / f).stat().st_size == n, f"size {f}")
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "window_start", "label", "cap_rank"] + m["protocol"]["features"])
    dev = pd.read_csv(F / "device_folds_v6.csv", keep_default_na=False)
    oof = pd.read_csv(F / "phase1_oof_scores_v6.csv")
    gold_l = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    gold_s = pd.read_csv(F / "gold_nonot_session_gold_v6.csv", encoding="utf-8-sig")
    thr = pd.read_csv(F / "phase1_threshold_table_v6.csv")
    chk(len(x) == 768927 and x.device_id.nunique() == 3923, "v6 fe3: 768,927 windows, 3,923 devices")
    chk(not x.duplicated(["device_id", "session_id", "window_start"]).any(), "no duplicate window keys")
    chk(len(m["protocol"]["features"]) == 15 and not {"ip_count", "frames_sent"} & set(m["protocol"]["features"]), "15 features, no ip_count/frames_sent")
    forbidden = {"protocol", "ot_flow_ratio", "ot_pkt_share", "evidence_tier", "label", "flood_like", "vendor", "mac"}
    chk(not forbidden & set(m["protocol"]["features"]), "no protocol/port/label/vendor columns in model inputs")
    chk(set(dev.device_id) == set(x.device_id) == set(oof.device_id), "device sets identical across fe3 / folds / OOF")
    chk(dev.fold_device.isin(range(5)).all() and dev.device_id.is_unique, "fold_device in 0..4, one fold per device")
    chk(oof.lightgbm.between(0, 1).all() and oof.lightgbm.notna().all(), "OOF probabilities in [0,1], no NaN")
    chk((dev.set_index("device_id").fold_device.loc[oof.device_id].to_numpy() == oof.fold_device.to_numpy()).all(), "OOF fold column matches folds file")
    ids = set(gold_l[gold_l.gold_label.isin(["OT", "non-OT"])].device_id_v6) | set(gold_s.device_id)
    chk(ids <= set(dev.device_id), "all gold devices exist in v6")
    chk(not (set(gold_l[gold_l.gold_label == "OT"].device_id_v6) & set(gold_s.device_id)), "no device is both lab-OT and capture non-OT")
    chk((gold_l.gold_label == "OT").sum() == 22 and (gold_l.gold_label == "non-OT").sum() == 2, "lab gold = 22 OT + 2 non-OT")
    chk(gold_s.status.eq("PROVISIONAL_AUTO_PENDING_USER_REVIEW").all(), "capture-scenario gold flagged provisional")
    chk(thr.gold_ot_recall.is_monotonic_decreasing and thr.pct_devices_flagged_OT.is_monotonic_decreasing, "recall and %flagged fall as threshold rises")
    r05 = thr[thr.threshold == 0.5].iloc[0]
    chk(abs(r05.rule_recall - 0.851) < 0.002 and abs(r05.rule_precision - 0.703) < 0.002, "threshold table @0.5 reproduces earlier recall 0.851 / precision 0.703")
    ev = json.loads((F / "phase1_gold_eval_v6.json").read_text(encoding="utf-8"))
    chk(abs(ev["population_roc_auc"] - 0.928) < 0.002, "population ROC-AUC 0.928 matches script 34/32")
    gu = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig")
    ann = pd.read_csv(F / "gold_candidates_v6_blind_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    bl = pd.read_csv(F / "gold_candidates_v6_blind.csv", encoding="utf-8-sig", dtype=str).fillna("")
    chk(gu.device_id.is_unique and set(gu.device_id) <= set(dev.device_id), "gold_union: unique devices, all exist in v6")
    chk(gu.gold_label.isin(["OT", "non-OT"]).all() and gu.confidence.isin(["high", "medium"]).all(), "gold_union: labels OT/non-OT, confidence high|medium only")
    chk(list(ann.device_id) == list(bl.device_id) and len(ann) == 220, "annotated sheet keeps the 220 blind rows in the same order")
    chk(bl.gold_label.eq("").all(), "original blind sheet untouched (annotation is a separate file)")
    chk(ann.gold_label.isin(["OT", "non-OT", "khong ro"]).all() and (ann.gold_label == "khong ro").sum() > 0, "annotation labels valid; uncertain rows kept as 'khong ro'")
    chk(not ann[ann.gold_label.isin(["OT", "non-OT"])].notes.str.contains("reviewed_by=sonthai").eq(False).any(), "every annotated row is attributed to its annotator")
    tu = pd.read_csv(F / "phase1_threshold_union_v6.csv")
    chk(tu.ot_recall.is_monotonic_decreasing and tu.nonot_flagged.is_monotonic_decreasing, "union threshold table monotonic")
    chk(float(tu.set_index("threshold").youden_j.idxmax()) == m["protocol"]["threshold_chosen"], "chosen threshold 0.5 = argmax Youden J on union gold")
    blind = pd.read_csv(F / "gold_candidates_v6_blind.csv", encoding="utf-8-sig")
    print("INFO blind sheet annotated rows:", int(blind.gold_label.fillna("").astype(str).str.strip().ne("").sum()), "/", len(blind))
    print("VERIFY", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv else (make() or 0))
