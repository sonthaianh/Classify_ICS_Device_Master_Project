"""Phase 2 - Transformer tren chuoi su kien payload (de cuong hua hen Transformer-based), 2026-11-11. THAM DO, cung giao thuc voi CNN (script 62).

Cung nhan (122 -> 117 co ho so / 67 co token), cung chia fold (StratifiedGroupKFold, nhom = session dau, 5 hat x 5 fold), cung chi so (balanced accuracy),
cung dau vao (chuoi su kien: ky hieu vai tro/huong/thong diep + bin do dai + bin thoi gian; cua so 256 su kien; huan luyen 3 cua so ngau nhien/thiet bi/epoch;
du doan 8 cua so cach deu roi lay trung binh xac suat) va cung ngan sach huan luyen cua CNN (30 buoc, lr 1e-3, weight decay 1e-2, loss co trong so lop).
Kien truc (CHOT TRUOC khi chay, khong chinh theo fold test): embedding ky hieu 32 + do dai 8 + thoi gian 8 -> Linear(48 -> 64) + embedding vi tri hoc duoc;
2 lop TransformerEncoder (4 dau chu y, ff 128, dropout 0,1, pre-norm, mat na pad) -> gop mean + max (128) -> Dropout 0,3 -> Linear(4).
Bien the: tu dau (scratch) va tien huan luyen doan ky hieu bi che (15%) tren chuoi cua TOAN BO thiet bi chua nhan (giong script 62, ghi ro la transductive).
Che do: A = thiet bi co ho so (117); T = thiet bi co token (67); khong ro ri nhan (nhan loai khong vao tien huan luyen).
Chay: set P2_PROFILE=phase2_payload_profiles_v6_full.json & P2_SEQ=phase2_sequences_v6_full.pkl ;  py -3.14 71_phase2_dl_transformer.py [--bench] [--strict] [--perm]
Dau ra: phase2_dl_transformer_v6.json
"""
import copy
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

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s62", HERE / "62_phase2_dl_seq.py")
S62 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S62)
S61, S58 = S62.S61, S62.S58
F, CLASSES = S62.F, S62.CLASSES
DEV, VOCAB, MASK_ID, L, NWIN_TRAIN = S62.DEV, S62.VOCAB, S62.MASK_ID, S62.L, S62.NWIN_TRAIN
rand_windows, test_windows, pool = S62.rand_windows, S62.test_windows, S62.pool
D = 64
torch.set_num_threads(int(os.environ.get("P2_THREADS", "8")))


class TEnc(nn.Module):
    """Tra ve (B, D, L) giong Enc cua CNN de dung chung ham gop pool()."""

    def __init__(self):
        super().__init__()
        self.es, self.el, self.ed = nn.Embedding(VOCAB + 1, 32, padding_idx=0), nn.Embedding(14, 8, padding_idx=0), nn.Embedding(9, 8, padding_idx=0)
        self.inp = nn.Linear(48, D)
        self.pos = nn.Embedding(L, D)
        layer = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, s, l, d):
        x = self.inp(torch.cat([self.es(s), self.el(l), self.ed(d)], -1)) + self.pos(torch.arange(s.shape[1]))[None]
        h = self.norm(self.enc(x, src_key_padding_mask=(s == 0)))
        return h.transpose(1, 2)


def pretrain(seed=0, steps=700, bs=96):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = TEnc(), nn.Linear(D, VOCAB)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)
    devs = sorted(DEV)
    top = np.bincount(np.concatenate([DEV[d][0] for d in devs]), minlength=VOCAB + 1).argmax()
    accs, base = [], []
    for step in range(steps):
        pick = [devs[i] for i in rng.integers(0, len(devs), bs)]
        s, l, d = rand_windows(pick, 1, rng)
        mk = (torch.rand(s.shape) < 0.15) & (s > 0)
        if mk.sum() == 0:
            continue
        sin = torch.where(mk, torch.full_like(s, MASK_ID), s)
        enc.train()
        logits = head(enc(sin, l, d).transpose(1, 2)[mk])
        loss = nn.functional.cross_entropy(logits, s[mk])
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
    enc, clf = TEnc(), nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * D, 4))
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


def bench():
    lab, _ = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y = np.array([S61.C2I[c] for c in sub.type4])
    devs = list(sub.index)
    idx = np.arange(len(devs))
    tr, te = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    t0 = time.time()
    P = ft_predict([devs[i] for i in tr], y[tr], [devs[i] for i in te], 0, None)
    print(f"one fold (scratch, {len(tr)} train devices): {time.time() - t0:.1f}s; probs {P.shape}; params {sum(p.numel() for p in TEnc().parameters())}")
    t0 = time.time()
    st, a, b = pretrain(steps=20)
    print(f"20 pretrain steps: {time.time() - t0:.1f}s")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def merge():
    out = {"status": "EXPLORATORY - Transformer on event sequences; same labels/folds/metric/budget as the CNN (script 62); architecture fixed a priori",
           "architecture": {"d_model": D, "layers": 2, "heads": 4, "ff": 128, "dropout": 0.1, "window": L, "params": int(sum(p.numel() for p in TEnc().parameters())), "train_steps": 30},
           "chance_balanced_acc": 0.25}
    for fp_ in sorted(F.glob("phase2_dl_transformer_part_*.json")):
        d = json.loads(fp_.read_text(encoding="utf-8"))
        tag, vname = d["tag"], d["variant"]
        out.setdefault(tag, {"n": d["n"], "class_counts": d["class_counts"]})[vname] = d["result"]
        if d.get("pretrain"):
            out["pretrain"] = d["pretrain"]
    ref = {}
    for fn, key in (("phase2_dl_seq_v6.json", "cnn"), ("phase2_dl_mlp_v6.json", "mlp")):
        p = F / fn
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            ref[key] = {k: {v: d[k][v]["balanced_acc"] for v in d[k] if isinstance(d[k][v], dict) and "balanced_acc" in d[k][v]} for k in d if isinstance(d[k], dict) and "n" in d[k]}
    out["reference_balanced_acc"] = ref
    (F / "phase2_dl_transformer_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("reference_balanced_acc",)}, indent=2, ensure_ascii=False)[:3500])


def main():
    if "--bench" in sys.argv:
        return bench()
    if "--merge" in sys.argv:
        return merge()
    regime, variant = arg("--regime", "A"), arg("--variant", "scratch")      # A = co ho so; T = co token
    strict = "--strict" in sys.argv
    t0 = time.time()
    pre, pinfo = None, None
    if variant == "pretrained":
        pre, macc, mbase = pretrain()
        pinfo = {"masked_symbol_acc": round(macc, 3), "most_common_baseline": round(mbase, 3), "steps": 700, "devices": len(DEV), "vocab": VOCAB}
        print(f"pretrain done {time.time() - t0:.0f}s | masked-symbol acc {macc:.3f} vs most-common baseline {mbase:.3f}", flush=True)
        assert macc > mbase + 0.05, "tien huan luyen khong hoc duoc"
    lab, sets = S58.build()
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    mask = lab.has_profile if regime == "A" else tok
    sub = lab[mask & (~lab.owner_approved if strict else True)]
    sub = sub[sub.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    tag = f"{'A_profiled' if regime == 'A' else 'T_tokens'}|{'strict' if strict else 'all_labels'}"

    def fp(tr, te, seed, yy):
        return ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, pre)
    res, oof0 = S61.cv_generic(fp, None, y, g, seeds=5)
    print(tag, variant, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
    part = {"tag": tag, "variant": variant, "n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(), "result": res, "pretrain": pinfo, "oof_seed0": {"device_id": devs, "true": y.tolist(), "pred": [str(v) for v in oof0]}}
    (F / f"phase2_dl_transformer_part_{regime}_{variant}{'_strict' if strict else ''}.json").write_text(json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
