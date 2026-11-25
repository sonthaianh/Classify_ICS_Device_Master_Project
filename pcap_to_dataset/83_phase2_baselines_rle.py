"""Phase 2: baseline cong trinh truoc (OUI; khop mo chuoi) va thu nen lap chuoi su kien cho CNN 1D (2026-11-12). KHONG ghi de ket qua dong bang.

  --baselines : B2 (chi OUI -> Random Forest 4 lop, theo tinh than Niedermaier 2018) va B5 (1-NN khop mo: khoang cach chinh sua chuan hoa tren chuoi ky hieu
                da nen lap, theo tinh than Tao 2023). CUNG tap A (thiet bi co ho so), cung nhom session, 5 hat x 5 fold, balanced accuracy.
  --rle       : CNN 1D tu dau voi 3 bien the CUNG kien truc/ngan sach: raw (khong nen), rle_rep (nen lap + kenh so lan lap), rle_norep (nen lap, khong kenh lap).
Chay voi P2_PROFILE=phase2_payload_profiles_v6_full.json P2_SEQ=phase2_sequences_v6_full.pkl (mac dinh trong script). Day la CAI DAT LAI GIAN LUOC, khong phai ket qua goc cua tac gia.
"""
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("P2_PROFILE", "phase2_payload_profiles_v6_full.json")
os.environ.setdefault("P2_SEQ", "phase2_sequences_v6_full.pkl")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s62", HERE / "62_phase2_dl_seq.py")
S62 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S62)
S61, S58 = S62.S61, S62.S58
F, CLASSES, DEV, VOCAB, L = S62.F, S62.CLASSES, S62.DEV, S62.VOCAB, S62.L
C2I = {c: i for i, c in enumerate(CLASSES)}
torch.set_num_threads(int(os.environ.get("P2_THREADS", "8")))
NW_TRAIN, NW_TEST = S62.NWIN_TRAIN, S62.NWIN_TEST


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def get_sub():
    lab, _ = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    return sub, list(sub.index)


# ---------------------------------------------------------------------------------------------------- baselines
def lev_norm(a, b):
    if len(a) == 0 or len(b) == 0:
        return 1.0
    prev = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (b != ca)
        cur[1:] = np.minimum(sub, prev[1:] + 1)
        # chen: cur[j] = min(cur[j], cur[j-1] + 1) can lap tuan tu
        for j in range(1, len(b) + 1):
            if cur[j - 1] + 1 < cur[j]:
                cur[j] = cur[j - 1] + 1
        prev = cur
    return prev[-1] / max(len(a), len(b))


def baselines():
    sub, devs = get_sub()
    y = sub.type4.to_numpy()
    g = sub.group.to_numpy()
    macs = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False).set_index("device_id").device_mac
    oui = macs.reindex(devs).str.lower().str[:8].fillna("na")
    top = oui.value_counts()
    oui_c = oui.where(oui.map(top) >= 3, "rare")
    X = pd.get_dummies(oui_c).to_numpy().astype(float)

    def rf_fp(tr, te, seed, yy):
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=1, random_state=seed, n_jobs=-1).fit(X[tr], yy[tr])
        P = np.zeros((len(te), 4))
        P[:, m.classes_.astype(int)] = m.predict_proba(X[te])
        return P
    res = {}
    res["B2_oui_rf"], _ = S61.cv_generic(rf_fp, X, y, g)
    print("B2", res["B2_oui_rf"]["balanced_acc"], res["B2_oui_rf"]["recall"], flush=True)
    # B5: khop mo
    seqs = []
    for d in devs:
        s = DEV[d][0]
        keep = np.r_[True, s[1:] != s[:-1]]
        seqs.append(s[keep][:200])
    n = len(devs)
    t0 = time.time()
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = lev_norm(seqs[i], seqs[j])
    print(f"khoang cach chinh sua {n}x{n}: {time.time() - t0:.0f}s", flush=True)

    def nn_fp(tr, te, seed, yy):
        P = np.zeros((len(te), 4))
        for k, t in enumerate(te):
            j = tr[np.argmin(D[t, tr])]
            P[k, yy[j]] = 1.0
        return P
    res["B5_fuzzy_1nn"], _ = S61.cv_generic(nn_fp, D, y, g)
    print("B5", res["B5_fuzzy_1nn"]["balanced_acc"], res["B5_fuzzy_1nn"]["recall"], flush=True)
    out = {"n": n, "class_counts": pd.Series(y).value_counts().to_dict(), "baselines": res, "note": "cai dat lai gian luoc; chance 0,25; so sanh voi RF 64 cot 0,618 va Transformer/CNN trong cac file dong bang"}
    (F / "phase2_wp2_baselines.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-BASELINES")


# ---------------------------------------------------------------------------------------------------- RLE CNN
def rle(s, l, d):
    keep = np.r_[True, (s[1:] != s[:-1]) | (l[1:] != l[:-1])]
    idx = np.flatnonzero(keep)
    run = np.diff(np.r_[idx, len(s)])
    rep = np.minimum(np.floor(np.log2(run)).astype(np.int64) + 1, 9)
    return s[idx], l[idx], d[idx], rep


class Enc2(nn.Module):
    def __init__(self):
        super().__init__()
        self.es, self.el, self.ed, self.er = nn.Embedding(VOCAB + 1, 32, padding_idx=0), nn.Embedding(14, 8, padding_idx=0), nn.Embedding(9, 8, padding_idx=0), nn.Embedding(11, 8, padding_idx=0)
        self.c1 = nn.Conv1d(56, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=4, dilation=2)

    def forward(self, s, l, d, r):
        x = torch.cat([self.es(s), self.el(l), self.ed(d), self.er(r)], -1).transpose(1, 2)
        return torch.relu(self.c2(torch.relu(self.c1(x))))


def make_arrays(variant):
    out = {}
    for dv in DEV:
        s, l, d = DEV[dv]
        if variant == "raw":
            out[dv] = (s, l, d, np.ones_like(s))
        else:
            s2, l2, d2, r2 = rle(s, l, d)
            out[dv] = (s2, l2, d2, r2 if variant == "rle_rep" else np.ones_like(s2))
    return out


def win(arrs, dev, start):
    a = arrs[dev]
    cut = [x[start:start + L] for x in a]
    pad = L - len(cut[0])
    return [np.pad(c, (0, pad)) for c in cut]


def rand_w(arrs, devs, n_per, rng):
    out = []
    for dv in devs:
        n = len(arrs[dv][0])
        for _ in range(n_per):
            out.append(win(arrs, dv, int(rng.integers(0, max(1, n - L + 1)))))
    return [torch.tensor(np.stack(z)) for z in zip(*out)]


def test_w(arrs, dev):
    n = len(arrs[dev][0])
    starts = np.unique(np.linspace(0, max(0, n - L), NW_TEST).astype(int))
    return [torch.tensor(np.stack(z)) for z in zip(*[win(arrs, dev, int(s)) for s in starts])]


def ft_predict2(arrs, tr_devs, ytr, te_devs, seed, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, clf = Enc2(), nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 4))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(clf.parameters()), lr=1e-3, weight_decay=1e-2)
    cnt = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(np.where(cnt > 0, len(ytr) / (4 * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    yt = torch.tensor(np.repeat(ytr, NW_TRAIN), dtype=torch.long)
    for _ in range(epochs):
        enc.train(); clf.train()
        s, l, d, r = rand_w(arrs, tr_devs, NW_TRAIN, rng)
        opt.zero_grad()
        lossf(clf(S62.pool(enc(s, l, d, r), s)), yt).backward()
        opt.step()
    enc.eval(); clf.eval()
    P = []
    with torch.no_grad():
        for dv in te_devs:
            s, l, d, r = test_w(arrs, dv)
            P.append(torch.softmax(clf(S62.pool(enc(s, l, d, r), s)), 1).mean(0).numpy())
    return np.stack(P)


def rle_run():
    sub, devs = get_sub()
    y, g = sub.type4.to_numpy(), sub.group.to_numpy()
    seeds = int(arg("--seeds", "5"))
    out = {"n": len(devs), "class_counts": pd.Series(y).value_counts().to_dict(), "seeds": seeds, "variants": {}, "events_mean": {}}
    t0 = time.time()
    store = {}
    for variant in ("raw", "rle_rep", "rle_norep"):
        arrs = make_arrays(variant)
        out["events_mean"][variant] = round(float(np.mean([len(arrs[d][0]) for d in devs])), 1)

        def fp(tr, te, seed, yy, arrs=arrs):
            return ft_predict2(arrs, [devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed)
        res, oof0 = S61.cv_generic(fp, None, y, g, seeds=seeds)
        out["variants"][variant] = res
        store[variant] = oof0
        print(variant, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
        (F / arg("--out", "phase2_wp6_rle_cnn.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(store, index=devs).assign(true=y).to_csv(F / arg("--out", "phase2_wp6_rle_cnn.json").replace(".json", "_oof_seed0.csv"))
    print("DONE-RLE")


if __name__ == "__main__":
    if "--baselines" in sys.argv:
        baselines()
    elif "--rle" in sys.argv:
        rle_run()
    else:
        print(__doc__)
