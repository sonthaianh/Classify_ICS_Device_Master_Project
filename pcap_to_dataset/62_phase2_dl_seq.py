"""Phase 2 - mo hinh deep learning tren CHUOI su kien payload: 1D-CNN + tien huan luyen khong nhan (2026-11-10). THAM DO.

Dau vao : phase2_sequences_v6.pkl (script 60): moi thiet bi = chuoi su kien (ky hieu thong diep/ho giao thuc x vai tro x huong, bin do dai, bin khoang cach thoi gian).
Kien truc: embedding(ky hieu 32 + do dai 8 + thoi gian 8) -> Conv1d(48->64,k5) -> Conv1d(64->64,k5,dilation 2) -> gop mean+max (co mat na pad) -> Linear(128->4).
  Cua so 256 su kien; luc huan luyen lay ngau nhien 3 cua so/thiet bi/epoch; luc du doan lay 8 cua so cach deu roi trung binh xac suat.
Tien huan luyen (khong nhan): doan ky hieu bi che (15%) tren cua so cua TOAN BO 864 thiet bi (ke ca thiet bi chua gan nhan). Khong dung nhan loai nen khong ro ri nhan,
  nhung thiet bi o fold test co mat trong tien huan luyen (kieu transductive) - ghi ro trong bao cao. Kiem tra hop le: do chinh xac doan ky hieu che phai > baseline "ky hieu pho bien nhat".
So sanh: (a) tu dau (khong tien huan luyen) vs (b) tinh chinh tu trong so tien huan luyen; CUNG nhan/fold/chi so voi RandomForest (58/59) va MLP (61).
Sieu tham so CO DINH truoc, khong chinh theo fold test. Chi dung chuoi su kien: KHONG dung feature vai tro/payload tong hop, KHONG dung chuoi danh tinh.
Che do: A = 101 thiet bi co ho so; T = 65 thiet bi co token; nhan day du va nhan nghiem (bo 5 EWS duyet tay).
Dau ra: phase2_dl_seq_v6.json     Chay: py -3.14 Tools/pcap_to_dataset/62_phase2_dl_seq.py
"""
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S61 = _load("s61", "61_phase2_dl_mlp.py")
S58 = S61.S58
F, CLASSES = S58.F, S58.CLASSES
SEQ = pickle.loads((F / os.environ.get("P2_SEQ", "phase2_sequences_v6.pkl")).read_bytes())
VOCAB = len(SEQ["vocab"])      # chi so 0 = pad; chi so VOCAB = MASK
MASK_ID = VOCAB
L, NWIN_TRAIN, NWIN_TEST = 256, 3, 8
DEV = {}
for dev, sess in SEQ["devices"].items():
    DEV[dev] = (np.concatenate([s["sym"] for s in sess]).astype(np.int64), np.concatenate([s["lenb"] for s in sess]).astype(np.int64) + 1,
                np.concatenate([s["dtb"] for s in sess]).astype(np.int64) + 1)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.es, self.el, self.ed = nn.Embedding(VOCAB + 1, 32, padding_idx=0), nn.Embedding(14, 8, padding_idx=0), nn.Embedding(9, 8, padding_idx=0)
        self.c1 = nn.Conv1d(48, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=4, dilation=2)

    def forward(self, s, l, d):
        x = torch.cat([self.es(s), self.el(l), self.ed(d)], -1).transpose(1, 2)
        return torch.relu(self.c2(torch.relu(self.c1(x))))       # (B,64,L)


def pool(H, s):
    m = (s > 0).float().unsqueeze(1)
    mean = (H * m).sum(2) / m.sum(2).clamp(min=1)
    mx = (H - 1e4 * (1 - m)).max(2).values
    mx = torch.where(m.sum(2) > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], 1)


def window(dev, start):
    s, l, d = DEV[dev]
    a, b = s[start:start + L], (l[start:start + L], d[start:start + L])
    pad = L - len(a)
    return (np.pad(a, (0, pad)), np.pad(b[0], (0, pad)), np.pad(b[1], (0, pad)))


def rand_windows(devs, n_per, rng):
    out = []
    for dv in devs:
        n = len(DEV[dv][0])
        for _ in range(n_per):
            out.append(window(dv, int(rng.integers(0, max(1, n - L + 1)))))
    return [torch.tensor(np.stack(z)) for z in zip(*out)]


def test_windows(dev):
    n = len(DEV[dev][0])
    starts = np.unique(np.linspace(0, max(0, n - L), NWIN_TEST).astype(int))
    return [torch.tensor(np.stack(z)) for z in zip(*[window(dev, int(st)) for st in starts])]


def pretrain(seed=0, steps=700, bs=96):
    """Doan ky hieu bi che tren cua so cua toan bo thiet bi; tra ve (state_dict encoder, do chinh xac che, baseline pho bien nhat)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Enc(), nn.Linear(64, VOCAB)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-2)
    devs = sorted(DEV)
    allsym = np.concatenate([DEV[d][0] for d in devs])
    top = np.bincount(allsym, minlength=VOCAB + 1).argmax()
    accs, base = [], []
    for step in range(steps):
        pick = [devs[i] for i in rng.integers(0, len(devs), bs)]
        s, l, d = rand_windows(pick, 1, rng)
        real = s > 0
        mk = (torch.rand(s.shape) < 0.15) & real
        if mk.sum() == 0:
            continue
        sin = torch.where(mk, torch.full_like(s, MASK_ID), s)
        H = enc(sin, l, d)                                         # (B,64,L)
        logits = head(H.transpose(1, 2)[mk])
        loss = nn.functional.cross_entropy(logits, s[mk] - 0)      # ky hieu goc (id>=1) -> lop id (0 khong xuat hien vi chi che vi tri that)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= steps - 100:
            accs.append((logits.argmax(1) == s[mk]).float().mean().item())
            base.append((s[mk] == top).float().mean().item())
    return copy.deepcopy(enc.state_dict()), float(np.mean(accs)), float(np.mean(base))


def ft_predict(tr_devs, ytr, te_devs, seed, pre_state, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, clf = Enc(), nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 4))
    if pre_state is not None:
        enc.load_state_dict(pre_state)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(clf.parameters()), lr=1e-3, weight_decay=1e-2)
    cnt = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(np.where(cnt > 0, len(ytr) / (4 * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    yt = torch.tensor(np.repeat(ytr, NWIN_TRAIN), dtype=torch.long)
    for _ in range(epochs):
        enc.train(); clf.train()
        s, l, d = rand_windows(tr_devs, NWIN_TRAIN, rng)
        opt.zero_grad()
        lossf(clf(pool(enc(s, l, d), s)), yt).backward()
        opt.step()
    enc.eval(); clf.eval()
    P = []
    with torch.no_grad():
        for dv in te_devs:
            s, l, d = test_windows(dv)
            P.append(torch.softmax(clf(pool(enc(s, l, d), s)), 1).mean(0).numpy())
    return np.stack(P)


def perm_only(n_perm=30):
    """Kiem tra hoan vi nhan cho mo hinh chuoi 'tu dau' tren tap A (101 thiet bi, day du nhan); ghi them vao JSON da co."""
    lab, _ = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    path = F / "phase2_dl_seq_v6.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    obs = out["A_profiled_101|all_labels"]["scratch"]["balanced_acc"]
    rng = np.random.default_rng(20261110)
    null = []
    for i in range(n_perm):
        ys = rng.permutation(y)

        def fp(tr, te, seed, yy):
            return ft_predict([devs[j] for j in tr], yy[tr], [devs[j] for j in te], seed, None)
        res, _ = S61.cv_generic(fp, None, y, g, seeds=1, shuffle_y=ys)
        null.append(res["balanced_acc"])
        print("perm", i, res["balanced_acc"], flush=True)
    out["permutation_A_all_labels_scratch"] = {"observed": obs, "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                                               "p_value": round(float((1 + sum(v >= obs for v in null)) / (1 + len(null))), 3), "n_perm": n_perm}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out["permutation_A_all_labels_scratch"], "DONE-PERM")


def main():
    if "--perm" in sys.argv:
        return perm_only()
    t0 = time.time()
    pre, macc, mbase = pretrain()
    print(f"pretrain done {time.time() - t0:.0f}s | masked-symbol acc {macc:.3f} vs most-common baseline {mbase:.3f}", flush=True)
    assert macc > mbase + 0.05, "tien huan luyen khong hoc duoc"
    lab, sets = S58.build()
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    out = {"status": "EXPLORATORY - sequence CNN; pretraining unlabeled & transductive; hyper-parameters fixed a priori",
           "pretrain": {"masked_symbol_acc": round(macc, 3), "most_common_baseline": round(mbase, 3), "steps": 700, "devices": len(DEV), "vocab": VOCAB},
           "chance_balanced_acc": 0.25}
    regimes = {"A_profiled_101": lab.has_profile, "T_tokens_65": tok}
    for rname, mask in regimes.items():
        for strict in (False, True):
            sub = lab[mask & (~lab.owner_approved if strict else True)]
            sub = sub[sub.index.isin(DEV)]
            y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
            tag = f"{rname}|{'strict' if strict else 'all_labels'}"
            block = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()}
            for vname, st in (("scratch", None), ("pretrained", pre)):
                def fp(tr, te, seed, yy, st=st):
                    return ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, st)
                res, _ = S61.cv_generic(fp, None, y, g, seeds=5)
                block[vname] = res
                print(tag, vname, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
            out[tag] = block
    ref = {}
    for fn, key in (("phase2_dl_mlp_v6.json", "mlp"), ("phase2_train4_v6.json", "rf58"), ("phase2_train4_tokens_only_v6.json", "rf59")):
        p = F / fn
        ref[key] = "available" if p.exists() else "missing"
    out["reference_files"] = ref
    assert out["A_profiled_101|all_labels"]["n"] == int(lab.has_profile.sum())
    (F / "phase2_dl_seq_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
