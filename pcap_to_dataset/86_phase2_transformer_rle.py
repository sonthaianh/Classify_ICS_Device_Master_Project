"""Phase 2: nen lap chuoi su kien cho TRANSFORMER (2026-11-12). KHONG ghi de ket qua dong bang.
Cung thiet ke voi script 83 (--rle, CNN 1D): 3 bien the CUNG kien truc/ngan sach/bo chia, Transformer huan luyen tu dau tren tap A (117 thiet bi co ho so):
  raw        chuoi goc (doi chung; tai hien cau hinh scratch cua script 71 voi so hat nho hon)
  rle_rep    nen lap (gop cac su kien lien tiep cung ky hieu va cung bin do dai) + kenh so lan lap (log2)
  rle_norep  nen lap, khong kenh so lan lap
Kien truc: nhu TEnc cua script 71 (embedding 32+8+8 [+8 kenh lap] -> Linear -> 2 lop encoder -> mean+max), cua so 256 su kien SAU nen -> cua so bao phu nhieu su kien goc hon.
Chay: py -3.14 86_phase2_transformer_rle.py [--seeds 3] [--seq phase2_sequences_v6_full.pkl] [--out phase2_wp6_rle_transformer.json]
"""
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


os.environ.setdefault("P2_PROFILE", "phase2_payload_profiles_v6_full.json")
os.environ["P2_SEQ"] = _arg("--seq", os.environ.get("P2_SEQ", "phase2_sequences_v6_full.pkl"))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S83 = _load("s83", "83_phase2_baselines_rle.py")
S71 = _load("s71", "71_phase2_dl_transformer.py")
S62, S61, S58, F = S83.S62, S83.S61, S83.S58, S83.F
VOCAB, L, NW_TRAIN, NW_TEST, DEV = S83.VOCAB, S83.L, S83.NW_TRAIN, S83.NW_TEST, S83.DEV
D = S71.D
torch.set_num_threads(int(os.environ.get("P2_THREADS", "8")))


class TEnc2(nn.Module):
    def __init__(self):
        super().__init__()
        self.es, self.el, self.ed, self.er = nn.Embedding(VOCAB + 1, 32, padding_idx=0), nn.Embedding(14, 8, padding_idx=0), nn.Embedding(9, 8, padding_idx=0), nn.Embedding(11, 8, padding_idx=0)
        self.inp = nn.Linear(56, D)
        self.pos = nn.Embedding(L, D)
        layer = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, s, l, d, r):
        x = self.inp(torch.cat([self.es(s), self.el(l), self.ed(d), self.er(r)], -1)) + self.pos(torch.arange(s.shape[1]))[None]
        return self.norm(self.enc(x, src_key_padding_mask=(s == 0))).transpose(1, 2)


def ft_predict_t(arrs, tr_devs, ytr, te_devs, seed, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, clf = TEnc2(), nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * D, 4))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(clf.parameters()), lr=1e-3, weight_decay=1e-2)
    cnt = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(np.where(cnt > 0, len(ytr) / (4 * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    yt = torch.tensor(np.repeat(ytr, NW_TRAIN), dtype=torch.long)
    for _ in range(epochs):
        enc.train(); clf.train()
        s, l, d, r = S83.rand_w(arrs, tr_devs, NW_TRAIN, rng)
        opt.zero_grad()
        lossf(clf(S62.pool(enc(s, l, d, r), s)), yt).backward()
        opt.step()
    enc.eval(); clf.eval()
    P = []
    with torch.no_grad():
        for dv in te_devs:
            s, l, d, r = S83.test_w(arrs, dv)
            P.append(torch.softmax(clf(S62.pool(enc(s, l, d, r), s)), 1).mean(0).numpy())
    return np.stack(P)


def main():
    sub, devs = S83.get_sub()
    y, g = sub.type4.to_numpy(), sub.group.to_numpy()
    seeds = int(_arg("--seeds", "3"))
    outp = F / _arg("--out", "phase2_wp6_rle_transformer.json")
    out = {"n": len(devs), "class_counts": pd.Series(y).value_counts().to_dict(), "seeds": seeds, "seq": os.environ["P2_SEQ"], "variants": {}, "events_mean": {}}
    t0, store = time.time(), {}
    for variant in ("raw", "rle_rep", "rle_norep"):
        arrs = S83.make_arrays(variant)
        out["events_mean"][variant] = round(float(np.mean([len(arrs[d][0]) for d in devs])), 1)

        def fp(tr, te, seed, yy, arrs=arrs):
            return ft_predict_t(arrs, [devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed)
        res, oof0 = S61.cv_generic(fp, None, y, g, seeds=seeds)
        out["variants"][variant] = res
        store[variant] = oof0
        print(variant, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
        outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(store, index=devs).assign(true=y).to_csv(outp.with_suffix(".oof_seed0.csv"))
    print("DONE-RLE-TF")


if __name__ == "__main__":
    main()
