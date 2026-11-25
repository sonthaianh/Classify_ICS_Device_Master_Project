"""Phase 1 - so sanh deep learning voi LightGBM duoi CUNG giao thuc (2026-11-10, chu de tai yeu cau them). THAM DO, khong doi mo hinh Phase 1 da dong bang.

Giao thuc (giong script 34/39): 15 feature hanh vi (cot log1p cho cac feature log nhu ban 'lr'), nhan luat yeu, huan luyen tren toi da K=20 cua so/thiet bi
cua cac thiet bi fold train, du doan = trung binh xac suat cac cua so (hoac cac bo K cua so) cua thiet bi test; diem OOF theo thiet bi.
Hai cach chia: fold_device (chinh) va fold_component (chat - chia theo thanh phan lien thong cua session chung).
Mo hinh (siêu tham so CO DINH truoc; dung som tren 10% thiet bi train, KHONG dung fold test):
  LightGBM  : mac dinh cua script 34 (300 cay, lr 0.05, 31 la, scale_pos_weight) - doi chung; 3 hat.
  MLP cua so: 15->64->32->1, dropout 0.2, AdamW, BCE pos_weight = neg/pos; huan luyen tren cac cua so; diem thiet bi = trung binh xac suat tat ca cua so test.
  DeepSets  : encoder cua so 15->64->64, gop mean+max tren tap <=20 cua so cua thiet bi -> MLP -> 1; tap bi bo ngau nhien 25% cua so khi huan luyen;
              luc du doan: tap <=20 cua so (thiet bi nhieu cua so: 5 lan boc 20 cua so ngau nhien roi trung binh).
Moi mo hinh DL chay 3 hat; bao cao (a) trung binh +- do lech theo hat (mo hinh don) va (b) diem trung binh 3 hat (ensemble).
Danh gia: ROC/PR tren nhan luat (tat ca/giao thuc/chi cong/khong flood) + tren gold hop nhat 184 thiet bi (ROC, recall/non-OT bi gan nham @0,5) + bootstrap hieu ROC theo thiet bi.
Tu kiem: LightGBM hat dau phai tai tao ROC da bao cao truoc do (0,928 theo thiet bi; ~0,858 chat).
Dau ra: phase1_dl_vs_lgbm_v6.json, phase1_dl_vs_lgbm_scores_v6.csv     Chay: py -3.14 Tools/pcap_to_dataset/63_phase1_dl_vs_lgbm.py
"""
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


B = _load("cmp", "34_compare_lightgbm_xgboost.py")
A = B.A
F, K, SEED = A.F, A.K, A.SEED
FEATS = B.FEATS
COLS = A.cols_for(FEATS, "lr")
SEEDS = 3
MAX_EP, PATIENCE, BS = 60, 6, 512

x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
dev["flood_like"] = dev.flood_like.astype(str).eq("True")
dev["evidence_tier"] = dev.evidence_tier.fillna("")
x["evidence_tier"] = x.evidence_tier.fillna("")
gold = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig").set_index("device_id")
XR = x[COLS].to_numpy(dtype=np.float32)
ROWS = x.groupby("device_id").indices                                   # device -> vi tri hang
CAPX = np.where((x.cap_rank < K).to_numpy())[0]                          # vi tri (trong x day du) cua cac hang cap_rank < K
CAP = {d: CAPX[i] for d, i in x.iloc[CAPX].groupby("device_id").indices.items()}   # device -> hang K cua so dau (vi tri TRONG x day du)
YW = (x.label == "OT_confirmed").to_numpy().astype(np.float32)
DEV_OF = x.device_id.to_numpy()
YDEV = dict(zip(dev.device_id, (dev.label == "OT_confirmed").astype(int)))


# ---------------- tien xu ly (thong ke fold train) ----------------
def prep_stats(train_rows):
    v = XR[train_rows]
    med = np.nanmedian(v, axis=0)
    v = np.where(np.isnan(v), med, v)
    return med, v.mean(0), v.std(0) + 1e-6


def tf(rows, st):
    med, mu, sd = st
    v = XR[rows]
    return ((np.where(np.isnan(v), med, v) - mu) / sd).astype(np.float32)


# ---------------- LightGBM doi chung ----------------
def lgbm_scores(tr_rows, te_devs, seed):
    y = YW[tr_rows].astype(int)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    m = lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=SEED + seed, n_jobs=8, verbose=-1, **B.DEFAULT["lightgbm"])
    m.fit(x.iloc[tr_rows][FEATS], y)
    rows = np.concatenate([ROWS[d] for d in te_devs])
    p = m.predict_proba(x.iloc[rows][FEATS])[:, 1]
    return pd.Series(p).groupby(DEV_OF[rows]).mean()


# ---------------- MLP cua so ----------------
def make_mlp(d):
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))


def fit_es(net, step_fn, val_fn, seed):
    """Huan luyen voi dung som tren tap validation noi bo."""
    best, state, bad = 1e9, None, 0
    for ep in range(MAX_EP):
        net.train()
        step_fn()
        net.eval()
        with torch.no_grad():
            v = val_fn()
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state)
    return net


def mlp_scores(tr_dev, va_dev, te_devs, seed):
    tr_rows = np.concatenate([CAP[d] for d in tr_dev])
    va_rows = np.concatenate([CAP[d] for d in va_dev])
    st = prep_stats(tr_rows)
    Xt, yt = torch.tensor(tf(tr_rows, st)), torch.tensor(YW[tr_rows])
    Xv, yv = torch.tensor(tf(va_rows, st)), torch.tensor(YW[va_rows])
    torch.manual_seed(SEED + seed)
    net = make_mlp(Xt.shape[1])
    pw = torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)

    def step():
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            lossf(net(Xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
    fit_es(net, step, lambda: lossf(net(Xv).squeeze(1), yv).item(), seed)
    rows = np.concatenate([ROWS[d] for d in te_devs])
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 65536):
            out.append(torch.sigmoid(net(torch.tensor(tf(rows[i:i + 65536], st)))).squeeze(1).numpy())
    return pd.Series(np.concatenate(out)).groupby(DEV_OF[rows]).mean()


# ---------------- DeepSets thiet bi ----------------
class DS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, X, m):
        h = self.enc(X)
        mean = (h * m.unsqueeze(2)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        mx = (h - 1e4 * (1 - m.unsqueeze(2))).max(1).values
        return self.head(torch.cat([mean, mx], 1)).squeeze(1)


def pad_sets(sets, st):
    """sets: list of arrays of row positions (<=K). -> X (n,K,d), mask (n,K)."""
    n, d = len(sets), len(COLS)
    X, M = np.zeros((n, K, d), np.float32), np.zeros((n, K), np.float32)
    for i, r in enumerate(sets):
        X[i, :len(r)] = tf(r, st)
        M[i, :len(r)] = 1
    return torch.tensor(X), torch.tensor(M)


def ds_scores(tr_dev, va_dev, te_devs, seed):
    st = prep_stats(np.concatenate([CAP[d] for d in tr_dev]))
    Xt, Mt = pad_sets([CAP[d] for d in tr_dev], st)
    Xv, Mv = pad_sets([CAP[d] for d in va_dev], st)
    yt = torch.tensor([YDEV[d] for d in tr_dev], dtype=torch.float32)
    yv = torch.tensor([YDEV[d] for d in va_dev], dtype=torch.float32)
    torch.manual_seed(SEED + seed)
    net = DS(Xt.shape[2])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1))))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)

    def step():
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            keep = (torch.rand(Mt[idx].shape) > 0.25).float() * Mt[idx]
            keep = torch.where(keep.sum(1, keepdim=True) == 0, Mt[idx], keep)
            opt.zero_grad()
            lossf(net(Xt[idx], keep), yt[idx]).backward()
            opt.step()
    fit_es(net, step, lambda: lossf(net(Xv, Mv), yv).item(), seed)
    rng = np.random.default_rng(SEED + seed)
    out = {}
    with torch.no_grad():
        for d in te_devs:
            r = ROWS[d]
            sets = [r] if len(r) <= K else [rng.choice(r, K, replace=False) for _ in range(5)]
            X, M = pad_sets(sets, st)
            out[d] = float(torch.sigmoid(net(X, M)).mean())
    return pd.Series(out)


# ---------------- khung chay ----------------
def run_split(fold_col):
    scores = {m: [pd.Series(np.nan, index=dev.device_id) for _ in range(SEEDS)] for m in ("lightgbm", "mlp_window", "deepsets_device")}
    t0 = time.time()
    for f in range(5):
        te_devs = list(dev.loc[dev[fold_col] == f, "device_id"])
        tr = dev[dev[fold_col] != f]
        for s in range(SEEDS):
            rng = np.random.default_rng(SEED + 100 * f + s)
            va = set()
            for _, g in tr.groupby("label"):
                ids = rng.permutation(g.device_id.to_numpy())
                va |= set(ids[: max(1, int(0.1 * len(ids)))])
            tr_dev = [d for d in tr.device_id if d not in va]
            va_dev = sorted(va)
            tr_rows = np.concatenate([CAP[d] for d in tr.device_id])        # LightGBM dung toan bo thiet bi train nhu script 34
            for name, fn in (("lightgbm", lambda: lgbm_scores(tr_rows, te_devs, s)), ("mlp_window", lambda: mlp_scores(tr_dev, va_dev, te_devs, s)),
                             ("deepsets_device", lambda: ds_scores(tr_dev, va_dev, te_devs, s))):
                p = fn()
                scores[name][s].loc[p.index] = p.values
            print(f"  {fold_col} fold {f} seed {s} done {time.time() - t0:.0f}s", flush=True)
    for m in scores:
        for s in scores[m]:
            assert not s.isna().any(), m
    return scores


def evaluate(score_arr):
    res = A.subsets(dev, score_arr)
    s = pd.Series(score_arr, index=dev.device_id)
    g = gold.loc[gold.index.intersection(s.index)]
    y = (g.gold_label == "OT").to_numpy()
    sg = s.loc[g.index].to_numpy()
    res["gold"] = {"roc": round(float(roc_auc_score(y, sg)), 3), "pr": round(float(average_precision_score(y, sg)), 3),
                   "ot_recall@0.5": round(float((sg[y] >= 0.5).mean()), 3), "nonot_flagged@0.5": round(float((sg[~y] >= 0.5).mean()), 3), "n": int(len(y))}
    return res


def boot_diff(sa, sb, n=2000):
    y = (dev.label == "OT_confirmed").to_numpy()
    rng = np.random.default_rng(SEED)
    d_roc, d_pr = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        d_roc.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        d_pr.append(average_precision_score(y[i], sa[i]) - average_precision_score(y[i], sb[i]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"roc_diff[2.5%,mean,97.5%]": ci(d_roc), "pr_diff[2.5%,mean,97.5%]": ci(d_pr)}


def main():
    out = {"features": COLS, "K": K, "seeds": SEEDS, "status": "EXPLORATORY comparison; Phase 1 production model (LightGBM) unchanged"}
    allscores = {}
    for tag, fold_col in (("fold_device", "fold_device"), ("strict_component", "fold_component")):
        sc = run_split(fold_col)
        blk = {}
        for m, lst in sc.items():
            per = [evaluate(s.loc[dev.device_id].to_numpy()) for s in lst]
            ens = evaluate(np.mean([s.loc[dev.device_id].to_numpy() for s in lst], axis=0))
            blk[m] = {"per_seed_all_roc": [p["all"]["roc_auc"] for p in per], "per_seed_all_pr": [p["all"]["pr_auc"] for p in per],
                      "roc_mean_std": [round(float(np.mean([p["all"]["roc_auc"] for p in per])), 3), round(float(np.std([p["all"]["roc_auc"] for p in per])), 3)],
                      "pr_mean_std": [round(float(np.mean([p["all"]["pr_auc"] for p in per])), 3), round(float(np.std([p["all"]["pr_auc"] for p in per])), 3)],
                      "gold_roc_mean": round(float(np.mean([p["gold"]["roc"] for p in per])), 3), "seed0": per[0], "ensemble": ens}
            allscores[f"{tag}|{m}"] = np.mean([s.loc[dev.device_id].to_numpy() for s in lst], axis=0)
        ens_l = allscores[f"{tag}|lightgbm"]
        blk["bootstrap_vs_lightgbm_ensemble"] = {m: boot_diff(allscores[f"{tag}|{m}"], ens_l) for m in ("mlp_window", "deepsets_device")}
        out[tag] = blk
        print(tag, {m: blk[m]["roc_mean_std"] for m in sc}, flush=True)
    # ---- tu kiem tai lap ----
    r0 = out["fold_device"]["lightgbm"]["per_seed_all_roc"][0]
    r1 = out["strict_component"]["lightgbm"]["per_seed_all_roc"][0]
    out["reproduction_check"] = {"lightgbm_fold_device_seed0_roc": r0, "expected": 0.928, "lightgbm_strict_seed0_roc": r1, "expected_approx": 0.858}
    assert abs(r0 - 0.928) <= 0.003, r0
    assert abs(r1 - 0.858) <= 0.012, r1
    (F / "phase1_dl_vs_lgbm_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({k: v for k, v in allscores.items()}, index=dev.device_id).to_csv(F / "phase1_dl_vs_lgbm_scores_v6.csv")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
