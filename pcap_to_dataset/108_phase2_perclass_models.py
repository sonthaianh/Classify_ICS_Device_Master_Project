"""Bang precision / do phu / F1 theo lop va ma tran nham lan cho RF64, MLP64, CNN 1D, Transformer (tu dau) tren tap A (2026-11-22).
Chi doc du doan tung hat da luu (phase2_perseed_preds_186.npz, phase2_tf_scratch_perseed_186.npz); KHONG huan luyen lai.
Gop luot du doan cua cac hat (RF/MLP/CNN: 5 hat; Transformer: 3 hat). Dau ra: phase2_perclass_models_186.json
Chay: py -3.14 108_phase2_perclass_models.py"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
F = Path(__file__).resolve().parent.parent / "Dataset/features"
C = ["PLC", "HMI", "EWS", "OTHER"]
def summarize(y, preds):
    cm = np.zeros((4, 4), int); ba = []
    for p in preds:
        cm += confusion_matrix(y, p, labels=C); ba.append(balanced_accuracy_score(y, p))
    rec = cm.diagonal() / cm.sum(1); prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"n_seeds": len(preds), "balanced_acc_mean": round(float(np.mean(ba)), 3), "balanced_acc_std": round(float(np.std(ba)), 3),
            "accuracy": round(float(cm.diagonal().sum() / cm.sum()), 3), "confusion": cm.tolist(),
            "recall": [round(float(x), 3) for x in rec], "precision": [round(float(x), 3) for x in prec], "f1": [round(float(x), 3) for x in f1],
            "macro_precision": round(float(prec.mean()), 3), "macro_recall": round(float(rec.mean()), 3), "macro_f1": round(float(f1.mean()), 3)}
def main():
    a = np.load(F / "phase2_perseed_preds_186.npz", allow_pickle=True)
    t = np.load(F / "phase2_tf_scratch_perseed_186.npz", allow_pickle=True)
    assert list(a["devices"]) == list(t["devices"])
    y = a["y"].astype(str)
    out = {"classes": C, "n": int(len(y))}
    for name, pre, ns in (("RF64", "RF64_", 5), ("MLP64", "MLP64_", 5), ("CNN1D", "CNN1D_", 5)):
        out[name] = summarize(y, [a[f"{pre}{s}"].astype(str) for s in range(ns)])
    out["Transformer_scratch"] = summarize(y, [t[f"TF_{s}"].astype(str) for s in range(3)])
    (F / "phase2_perclass_models_186.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in ("RF64", "MLP64", "CNN1D", "Transformer_scratch"):
        v = out[k]; print(k, v["n_seeds"], v["balanced_acc_mean"], "acc", v["accuracy"], "R", v["recall"], "P", v["precision"], "F1", v["f1"], "macroF1", v["macro_f1"])
        print("  cm", v["confusion"])
if __name__ == "__main__":
    sys.exit(main())
