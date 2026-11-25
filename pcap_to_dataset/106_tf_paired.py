"""So sanh ghep cap Random Forest 64 cot - Transformer tu dau (3 hat) tren bo 186 nhan (2026-11-22). Doc du doan tung hat da luu:
phase2_perseed_preds_186.npz (script 105 --seeds) va phase2_tf_scratch_perseed_186.npz (script 105 --tf). Cung thiet bi, cung fold (StratifiedGroupKFold theo hat).
Bootstrap theo thiet bi 2.000 lan, hieu so trung binh cac hat (hat 0-2). Ghi them vao phase2_groupc_rerun_186.json (khoa tf_vs_rf).
Chay: py -3.14 106_tf_paired.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

F = Path(__file__).resolve().parent.parent / "Dataset/features"
B = 2000
CLASSES = ["PLC", "HMI", "EWS", "OTHER"]


def main():
    a = np.load(F / "phase2_perseed_preds_186.npz", allow_pickle=True)
    t = np.load(F / "phase2_tf_scratch_perseed_186.npz", allow_pickle=True)
    assert list(a["devices"]) == list(t["devices"]) and list(a["y"]) == list(t["y"])
    y = a["y"].astype(str)
    seeds = sorted(int(k.split("_")[1]) for k in t.files if k.startswith("TF_"))
    rf = [a[f"RF64_{s}"].astype(str) for s in seeds]
    tf = [t[f"TF_{s}"].astype(str) for s in seeds]
    bal = lambda ps, idx=None: float(np.mean([balanced_accuracy_score(y if idx is None else y[idx], p if idx is None else p[idx]) for p in ps]))
    rng = np.random.default_rng(20261122)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(bal(rf, i) - bal(tf, i))
    res = {"seeds": seeds, "n": int(n), "RF64_mean_bal": round(bal(rf), 3), "TF_scratch_mean_bal": round(bal(tf), 3),
           "per_seed_RF64": [round(balanced_accuracy_score(y, p), 3) for p in rf], "per_seed_TF": [round(balanced_accuracy_score(y, p), 3) for p in tf],
           "RF64_minus_TF": {"point": round(bal(rf) - bal(tf), 3), "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)], "share_le_0": round(float(np.mean(np.array(d) <= 0)), 3)},
           "TF_recall": {c: round(float(np.mean([(p[y == c] == c).mean() for p in tf])), 3) for c in CLASSES}}
    print(json.dumps(res, indent=1))
    out = F / "phase2_groupc_rerun_186.json"
    dd = json.loads(out.read_text(encoding="utf-8"))
    dd["tf_vs_rf"] = res
    out.write_text(json.dumps(dd, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
