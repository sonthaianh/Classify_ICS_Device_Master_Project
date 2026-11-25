"""Phase 2 - mo hinh deep learning DOI CHUNG: MLP tren cac bo feature cua script 58/59 (2026-11-10). THAM DO.

Muc dich: thay RandomForest bang MLP (PyTorch, CPU) trong CUNG nhan, CUNG chia fold (StratifiedGroupKFold, nhom = session dau, 5 hat x 5 fold),
CUNG chi so (balanced accuracy) de so sanh cong bang. Sieu tham so CO DINH truoc, KHONG chinh theo ket qua fold test:
  chuan hoa theo thong ke fold train; MLP 2 lop an (64, 32) + ReLU + dropout 0,3; AdamW lr 2e-3, weight decay 1e-2, 200 epoch full-batch;
  CE co trong so lop nghich dao tan suat (chi tinh tren train); gop 3 khoi tao ngau nhien (trung binh xac suat).
Che do: A = 101 thiet bi co ho so payload; T = 65 thiet bi co token (nhan day du va nhan nghiem bo 5 EWS duyet tay).
Dau ra: phase2_dl_mlp_v6.json (kem so RandomForest tu script 58/59 de doi chieu)
Chay: py -3.14 pcap_to_dataset/61_phase2_dl_mlp.py
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S58 = _load("s58", "58_phase2_train4.py")
F, CLASSES = S58.F, S58.CLASSES
C2I = {c: i for i, c in enumerate(CLASSES)}


def mlp_fit_predict(Xtr, ytr, Xte, seed, epochs=200, n_init=3):
    """Tra ve xac suat (n_te, 4). ytr la chi so lop."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(np.where(cnt > 0, len(ytr) / (4 * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    P = 0
    for k in range(n_init):
        torch.manual_seed(seed * 100 + k)
        net = nn.Sequential(nn.Linear(A.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 4))
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss(weight=w)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(net(A), y).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(B), 1).numpy()
    return P / n_init


def cv_generic(fit_predict, X, y, g, seeds=5, shuffle_y=None):
    """fit_predict(train_idx, test_idx, seed) -> proba (n_te,4). Dung chung cho MLP va mo hinh chuoi."""
    bal, f1s, rec, cm, oof0 = [], [], [], np.zeros((4, 4)), None
    yi = np.array([C2I[c] for c in y])
    for seed in range(seeds):
        pred = np.empty(len(y), dtype=int)
        yy = yi if shuffle_y is None else np.array([C2I[c] for c in shuffle_y])
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), yy, g):
            pred[te] = fit_predict(tr, te, seed, yy).argmax(1)
        p = np.array(CLASSES)[pred]
        bal.append(balanced_accuracy_score(y, p))
        f1s.append(f1_score(y, p, average="macro"))
        rec.append(recall_score(y, p, labels=CLASSES, average=None, zero_division=0))
        cm += confusion_matrix(y, p, labels=CLASSES)
        if seed == 0:
            oof0 = p.copy()
    return {"balanced_acc": round(float(np.mean(bal)), 3), "balanced_acc_std": round(float(np.std(bal)), 3), "macro_f1": round(float(np.mean(f1s)), 3),
            "recall": dict(zip(CLASSES, np.round(np.mean(rec, axis=0), 3).tolist())), "confusion_sum_over_seeds": cm.astype(int).tolist()}, oof0


def main():
    lab, sets = S58.build()
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    out = {"status": "EXPLORATORY - same labels/folds/metric as RandomForest (scripts 58/59); hyper-parameters fixed a priori", "chance_balanced_acc": 0.25}
    regimes = {"A_profiled_101": (lab.has_profile, ["behaviour", "payload_role", "payload_role_eng", "behaviour_payload_role_eng"]),
               "T_tokens_65": (tok, ["payload_role", "payload_role_eng"])}
    for rname, (mask, keysets) in regimes.items():
        for strict in (False, True):
            sub = lab[mask & (~lab.owner_approved if strict else True)]
            y, g = sub.type4.to_numpy(), sub.group.to_numpy()
            tag = f"{rname}|{'strict' if strict else 'all_labels'}"
            block = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()}
            for ks in keysets:
                X = sets[ks].loc[sub.index].to_numpy(dtype=float)
                X = np.nan_to_num(X, nan=0.0)

                def fp(tr, te, seed, yy, X=X):
                    return mlp_fit_predict(X[tr], yy[tr], X[te], seed)
                res, _ = cv_generic(fp, X, y, g)
                block[ks] = res
                print(tag, ks, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], flush=True)
            out[tag] = block
    # doi chieu RandomForest (doc tu file da co, khong chay lai)
    rf58 = json.loads((F / "phase2_train4_v6.json").read_text(encoding="utf-8"))
    rf59 = json.loads((F / "phase2_train4_tokens_only_v6.json").read_text(encoding="utf-8"))
    ref = {}
    for strict, k58, k59 in ((False, "all_labels", "all_labels"), (True, "strict", "strict")):
        ref[f"A_profiled_101|{'strict' if strict else 'all_labels'}"] = {ks: rf58[k58]["A_profiled_only"][ks]["balanced_acc"] for ks in ("behaviour", "payload_role", "payload_role_eng", "behaviour_payload_role_eng")}
        ref[f"T_tokens_65|{'strict' if strict else 'all_labels'}"] = {ks: rf59[k59][ks]["balanced_acc"] for ks in ("payload_role", "payload_role_eng")}
    out["randomforest_reference_balanced_acc"] = ref
    # tu kiem
    assert out["A_profiled_101|all_labels"]["n"] == int(lab.has_profile.sum())
    assert out["T_tokens_65|all_labels"]["n"] == int(tok.sum())
    (F / "phase2_dl_mlp_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
