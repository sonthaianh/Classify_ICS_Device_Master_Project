"""Phase 1: sinh them thiet bi tong hop (synthetic) va thu nghiem xem co giup mo hinh khong (2026-11-12).

KHONG sua bat ky file v6 nao. Moi dau ra co hau to `_syn10k` (hoac thu muc/ten rieng).

Phuong phap sinh (smoothed bootstrap o muc thiet bi, theo tung lop nhan luat):
  1. Chon thiet bi that "hat giong" cung lop (moi hat duoc dung deu nhau).
  2. Bien doi dac trung sang khong gian log1p (dac trung dem/toc do/thoi luong) hoac giu nguyen (ti le 0-1).
  3. Dich ca "dam cua so" cua thiet bi mot vector ngau nhien ~ N(0, tau^2 * Sigma_lop), Sigma_lop = hiep phuong sai GIUA cac thiet bi cung lop
     (giu tuong quan giua cac dac trung); cong them nhieu nho tung cua so (jit * sigma_trong_thiet_bi).
  4. Giu: gia tri 0 cau truc (0 o hat thi van 0), mat na NaN cua hat, thu tu cap_rank/phien, so nguyen cho dac trung dem,
     ti le trong [0,1], cac gioi han min/max cua lop, arp_target_unique <= arp_frame_count.
  Nhan, evidence_tier, flood_like, nhom nguon, thanh phan, fold duoc KE THUA tu hat giong (con luon cung fold/thanh phan voi hat).
Vi khong them thong tin moi ngoai cac thiet bi that, ket luan chi co gia tri khi cham tren thiet bi THAT chua tung dung de sinh (che do `experiment`:
moi fold chi sinh tu thiet bi huan luyen cua fold do).

  py -3.14 71_phase1_synthetic_devices.py build        # tao dataset 10.000 thiet bi (3.923 that + 6.077 sinh), can bang 5.000/5.000
  py -3.14 71_phase1_synthetic_devices.py quality      # so sanh phan phoi that vs sinh, bo phan biet, khoang cach hat
  py -3.14 71_phase1_synthetic_devices.py experiment   # huan luyen that vs that+sinh vs nhan ban nguyen mau; cham tren thiet bi that
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_contract import TRAINING_FEATURES as D17  # noqa: E402

warnings.filterwarnings("ignore")
F = HERE.parent / "Dataset/features"
SEED, K = 20261108, 20
SCALE = ["ip_count", "frames_sent"]
USED = [c for c in D17 if c not in SCALE]            # 15 dac trung cua mo hinh dong bang
RATIO = ["udp_ratio", "small_flow_ratio", "one_way_flow_ratio", "bcast_frame_ratio", "carryover_flow_ratio"]
LOGT = [c for c in D17 if c not in RATIO]
INTCOL = ["frames_sent", "arp_frame_count", "arp_target_unique", "icmp_frame_count", "nonip_frame_count", "ip_count",
          "unique_local_port_count", "svc_port_unique"]
LOG1P_COLS = ["pkts_per_flow", "bytes_per_packet", "packet_rate", "unique_local_port_count", "svc_port_unique",
              "frames_sent", "arp_frame_count", "arp_target_unique", "icmp_frame_count", "nonip_frame_count", "ip_count"]
OTL, UNK = "OT_confirmed", "unknown"
DEFAULT_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1)
IX = {c: i for i, c in enumerate(D17)}
IS_RATIO = np.array([c in RATIO for c in D17])
IS_INT = np.array([c in INTCOL for c in D17])


def load():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x = x.sort_values(["device_id", "cap_rank"], kind="mergesort").reset_index(drop=True)
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False)
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str, "primary_source_family": str}, keep_default_na=False)
    dev["flood_like"] = dev.flood_like.astype(str).eq("True")
    return x, lab, dev


class Gen:
    """Chua ma tran cua so that da bien doi, thong ke theo lop va ham sinh thiet bi con."""

    def __init__(self, x, dev):
        self.x = x
        self.dev = dev.set_index("device_id")
        ids = x.device_id.to_numpy()
        change = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        self.start = dict(zip(ids[change], change))
        self.count = dict(zip(ids[change], np.diff(np.r_[change, len(ids)])))
        self.R = x[D17].to_numpy(dtype=float)
        self.T = np.where(IS_RATIO, self.R, np.log1p(np.where(np.isnan(self.R), 0, self.R)))
        self.T[np.isnan(self.R)] = np.nan
        self.cap = x.cap_rank.to_numpy()
        self.sess = x.session_id.to_numpy()
        self.wstart = x.window_start.to_numpy()

    def rows(self, d, cap):
        s = self.start[d]
        return slice(s, s + min(self.count[d], cap))

    def class_stats(self, ids, cap=100):
        n = len(ids)
        M = np.full((n, len(D17)), np.nan)
        W = []
        lo = np.full(len(D17), np.inf)
        hi = np.full(len(D17), -np.inf)
        for i, d in enumerate(ids):
            r = self.rows(d, cap)
            Tt, Rr = self.T[r], self.R[r]
            with np.errstate(all="ignore"):
                M[i] = np.nanmean(Tt, axis=0)
                if len(Tt) >= 3:
                    W.append(np.nanstd(Tt, axis=0))
                pos = np.where(Rr > 0, Rr, np.nan)
                lo = np.fmin(lo, np.nanmin(pos, axis=0)) if np.isfinite(pos).any() else lo
                hi = np.fmax(hi, np.nanmax(Rr, axis=0)) if np.isfinite(Rr).any() else hi
        med = np.nanmedian(M, axis=0)
        M = np.where(np.isnan(M), med, M)
        cov = np.cov(M.T) + 1e-6 * np.eye(len(D17))
        sigw = np.nanmedian(np.array(W), axis=0) if W else np.zeros(len(D17))
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        return {"L": np.linalg.cholesky(cov), "sigw": np.nan_to_num(sigw), "lo": lo, "hi": hi}

    def child(self, seed, st, tau, jit, rng, cap):
        r = self.rows(seed, cap)
        Rs, Ts = self.R[r], self.T[r]
        if tau == 0 and jit == 0:
            return Rs.copy(), r
        absent = np.all((np.nan_to_num(Rs) == 0), axis=0)          # dac trung 0 cau truc o hat -> giu nguyen
        delta = tau * (st["L"] @ rng.standard_normal(len(D17)))
        delta[absent] = 0
        eps = jit * st["sigw"] * rng.standard_normal(Ts.shape)
        eps[:, absent] = 0
        Tn = Ts + delta + eps
        interior = (Rs > 0) & (Rs < 1)                               # ti le: gia tri bien 0/1 la cau truc (hai dinh), chi dich phan o giua
        ratio_new = np.where(interior, np.clip(Tn, 0.001, 0.999), Rs)
        out = np.where(IS_RATIO, ratio_new, np.expm1(np.minimum(Tn, 50)))
        out = np.clip(out, 0, None)
        out = np.minimum(out, st["hi"])
        pos_lo = np.where(IS_RATIO, 0.0, st["lo"])
        out = np.where(out > 0, np.maximum(out, pos_lo), out)
        out = np.where(Rs == 0, 0.0, out)                            # 0 cau truc tung cua so
        out[:, IS_INT] = np.rint(out[:, IS_INT])
        a, b = IX["arp_target_unique"], IX["arp_frame_count"]
        out[:, a] = np.minimum(out[:, a], out[:, b])
        s_, u_ = IX["svc_port_unique"], IX["unique_local_port_count"]
        obey = Rs[:, s_] <= Rs[:, u_]                                # ep dieu kien svc <= cong cuc bo o cua so ma hat da thoa
        out[:, s_] = np.where(obey, np.minimum(out[:, s_], out[:, u_]), out[:, s_])
        cols3 = [IX["arp_frame_count"], IX["icmp_frame_count"], IX["nonip_frame_count"]]
        fs = IX["frames_sent"]
        tot3 = out[:, cols3].sum(axis=1)
        obey3 = Rs[:, cols3].sum(axis=1) <= Rs[:, fs]                # ep arp+icmp+nonip <= frames_sent o cua so hat da thoa
        scale = np.where(obey3 & (tot3 > out[:, fs]) & (tot3 > 0), out[:, fs] / np.maximum(tot3, 1), 1.0)
        out[:, cols3] = np.floor(out[:, cols3] * scale[:, None])
        out[:, IX["arp_target_unique"]] = np.minimum(out[:, IX["arp_target_unique"]], out[:, IX["arp_frame_count"]])
        out[:, IX["flow_dur_median"]] = np.minimum(out[:, IX["flow_dur_median"]], 45.0)
        out[np.isnan(Rs)] = np.nan                                   # giu mat na NaN cua hat
        return out, r


def seed_plan(ids_by_class, n_new, rng):
    """Moi hat duoc dung deu nhau: moi hat sinh floor(n_new/n) con, phan du chon ngau nhien."""
    plan = {}
    for c, ids in ids_by_class.items():
        n = len(ids)
        need = n_new[c]
        seeds = list(np.repeat(ids, need // n)) if n else []
        extra = list(rng.choice(ids, size=need - len(seeds), replace=False)) if n and need - len(seeds) > 0 else []
        plan[c] = seeds + extra
    return plan


# ----------------------------------------------------------------------------------------------------------------- build
def build(tau=0.5, jit=0.15, total=10000, cap=100):
    t0 = time.time()
    x, lab, dev = load()
    g = Gen(x, dev)
    rng = np.random.default_rng(SEED + 71)
    ids_by_class = {c: dev.loc[dev.label == c, "device_id"].to_numpy() for c in (OTL, UNK)}
    n_new = {c: total // 2 - len(ids_by_class[c]) for c in ids_by_class}
    assert all(v > 0 for v in n_new.values())
    plan = seed_plan(ids_by_class, n_new, rng)
    stats = {c: g.class_stats(ids_by_class[c]) for c in ids_by_class}
    labidx = lab.set_index("device_id")
    devidx = dev.set_index("device_id")
    parts, lab_rows, dev_rows, meta = [], [], [], []
    k = 0
    for c in (OTL, UNK):
        for seed in plan[c]:
            k += 1
            sid = f"SYN_{k:05d}"
            Rn, r = g.child(seed, stats[c], tau, jit, rng, cap)
            n = Rn.shape[0]
            sub = x.iloc[r.start:r.stop]
            sess_map = {s: f"SYN_S_{k:05d}_{i}" for i, s in enumerate(pd.unique(sub.session_id))}
            df = pd.DataFrame(Rn, columns=D17)
            df.insert(0, "window_start", sub.window_start.to_numpy())
            df.insert(0, "session_id", sub.session_id.map(sess_map).to_numpy())
            df.insert(0, "device_id", sid)
            for col in ("label", "evidence_tier", "flood_like", "label_quality"):
                df[col] = sub[col].iloc[0]
            df["n_windows_device"] = n
            df["n_sessions_device"] = len(sess_map)
            df["cap_rank"] = sub.cap_rank.to_numpy()
            for col in LOG1P_COLS:
                df["log1p_" + col] = np.log1p(df[col])
            parts.append(df)
            lr = labidx.loc[seed].copy()
            lr["device_mac"] = f"SYNTHETIC:{sid}"
            lr["label_reason"] = f"synthetic child of {seed} (smoothed bootstrap, tau={tau}, jit={jit}); label inherited from seed"
            lr["label_method"] = "synthetic_smoothed_bootstrap"
            lr["device_id"] = sid
            lab_rows.append(lr)
            dr = devidx.loc[seed].copy()
            dr["device_id"] = sid
            dr["n_windows"] = n
            dev_rows.append(dr)
            meta.append((sid, seed, c))
    syn = pd.concat(parts, ignore_index=True)
    real = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")[list(syn.columns)]   # dung thu tu goc cua file that
    for c in syn.columns:                                           # cung kieu du lieu voi file that
        try:
            syn[c] = syn[c].astype(real[c].dtype)
        except (ValueError, TypeError):
            pass
    out = pd.concat([real, syn], ignore_index=True)
    out = out.sort_values(["device_id", "session_id", "window_start"], kind="mergesort").reset_index(drop=True)   # nhu fe3 goc
    out.to_parquet(F / "master_device_windows_v6_fe3_syn10k.parquet", index=False)
    labs = pd.concat([lab, pd.DataFrame(lab_rows)], ignore_index=True)
    labs.to_csv(F / "device_labels_v6_rule_syn10k.csv", index=False)
    devs_new = pd.DataFrame(dev_rows)[dev.columns]
    devs = pd.concat([dev, devs_new], ignore_index=True)
    devs.to_csv(F / "device_folds_v6_syn10k.csv", index=False)
    pd.DataFrame(meta, columns=["device_id", "seed_device_id", "class"]).to_csv(F / "syn10k_seed_map.csv", index=False)
    man = {"created": pd.Timestamp.now().isoformat(timespec="seconds"), "method": "smoothed bootstrap by device and class", "tau": tau, "jit": jit,
           "window_cap_per_synthetic_device": cap, "seed": SEED + 71, "real_devices": int(len(dev)), "synthetic_devices": int(len(devs_new)),
           "total_devices": int(devs.device_id.nunique()), "label_counts": devs.label.value_counts().to_dict(),
           "rows_real": int(len(real)), "rows_synthetic": int(len(syn)), "synthetic_id_prefix": "SYN_",
           "notes": ["labels are the weak rule labels inherited from the seed device (unknown != non-OT)",
                     "synthetic devices share fold_device/fold_component/component_id with their seed; never evaluate on synthetic devices",
                     "class statistics (covariance, ranges) estimated from ALL real devices: for leakage-free evaluation use `experiment` (per-fold generation)",
                     "real rows are byte-identical to master_device_windows_v6_fe3.parquet"]}
    (F / "syn10k_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man, indent=2), f"\n{time.time() - t0:.0f}s")


# --------------------------------------------------------------------------------------------------------------- quality
def ks(a, b):
    a, b = np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    v = np.sort(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, side="right") / len(a) - np.searchsorted(b, v, side="right") / len(b))))


def quality():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3_syn10k.parquet")
    seedmap = pd.read_csv(F / "syn10k_seed_map.csv")
    syn_mask = x.device_id.str.startswith("SYN_")
    real, syn = x[~syn_mask], x[syn_mask]
    out = {"ks_device_mean_by_class": {}, "ks_window_by_class": {}}
    rng = np.random.default_rng(1)
    dm = x.groupby("device_id")[D17].mean()
    lab = x.groupby("device_id").label.first()
    for c in (OTL, UNK):
        r_ids, s_ids = lab[(lab == c) & ~lab.index.str.startswith("SYN_")].index, lab[(lab == c) & lab.index.str.startswith("SYN_")].index
        out["ks_device_mean_by_class"][c] = {f: round(ks(dm.loc[r_ids, f].to_numpy(), dm.loc[s_ids, f].to_numpy()), 3) for f in D17}
        rw, sw = real[real.label == c], syn[syn.label == c]
        # lay mau cua so, moi thiet bi toi da 20 cua so dau (cap_rank<20) de khong bi thiet bi lon chi phoi
        rw, sw = rw[rw.cap_rank < K], sw[sw.cap_rank < K]
        out["ks_window_by_class"][c] = {f: round(ks(rw[f].to_numpy(), sw[f].to_numpy()), 3) for f in D17}
    # bo phan biet that-sinh o muc thiet bi (nhom theo hat giong de con khong ro ri sang fold khac)
    seed_of = dict(zip(seedmap.device_id, seedmap.seed_device_id))
    ids = dm.index.to_numpy()
    grp = np.array([seed_of.get(d, d) for d in ids])
    y = np.array([d.startswith("SYN_") for d in ids]).astype(int)
    Xd = np.where(IS_RATIO, dm[D17].to_numpy(), np.log1p(np.clip(dm[D17].to_numpy(), 0, None)))
    Xd = np.nan_to_num(Xd, nan=-1)
    ug = pd.unique(grp)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % 5))
    f_arr = np.array([fold_of[g] for g in grp])
    # phan biet TRONG TUNG LOP (neu gop lop thi ty le sinh/that khac nhau theo lop se bi bo phan biet khai thac nham)
    cls = lab.loc[ids].to_numpy()
    out["real_vs_synthetic_discriminator_auc_within_class"] = {}
    for c in (OTL, UNK):
        pr = np.full(len(y), np.nan)
        for f in range(5):
            tr, te = (f_arr != f) & (cls == c), (f_arr == f) & (cls == c)
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEED, n_jobs=8,
                                   class_weight="balanced").fit(Xd[tr], y[tr])
            pr[te] = m.predict_proba(Xd[te])[:, 1]
        k = cls == c
        out["real_vs_synthetic_discriminator_auc_within_class"][c] = round(float(roc_auc_score(y[k], pr[k])), 3)
    # khoang cach hat -> con va khoang cach den thiet bi that gan nhat khac
    sd = Xd.std(axis=0) + 1e-9
    Z = (Xd - Xd.mean(axis=0)) / sd
    idx = {d: i for i, d in enumerate(ids)}
    real_i = np.flatnonzero(y == 0)
    ratios, own = [], []
    sample = rng.choice(np.flatnonzero(y == 1), size=min(1500, int(y.sum())), replace=False)
    for i in sample:
        seed_i = idx[seed_of[ids[i]]]
        d_all = np.linalg.norm(Z[real_i] - Z[i], axis=1)
        d_other = np.min(d_all[real_i != seed_i])
        d_own = np.linalg.norm(Z[seed_i] - Z[i])
        own.append(d_own)
        ratios.append(d_own / (d_other + 1e-9))
    out["child_to_seed_vs_nearest_other_real"] = {"median_ratio": round(float(np.median(ratios)), 3), "share_child_closer_to_seed_than_any_other_real": round(float(np.mean(np.array(ratios) < 1)), 3),
                                                   "median_dist_to_seed": round(float(np.median(own)), 3)}
    r_ot = real_i[lab.loc[ids[real_i]].to_numpy() == OTL]
    nn = []
    for i in rng.choice(r_ot, size=min(300, len(r_ot)), replace=False):
        d = np.linalg.norm(Z[r_ot] - Z[i], axis=1)
        nn.append(np.min(d[r_ot != i]))
    out["real_ot_nearest_other_real_median_dist"] = round(float(np.median(nn)), 3)
    out["exact_copy_windows_share"] = None
    out["invariants"] = {
        "arp_target<=arp_frame": float((syn.arp_target_unique <= syn.arp_frame_count).mean()),
        "ratios_in_0_1": bool(all(((syn[c].dropna() >= 0) & (syn[c].dropna() <= 1)).all() for c in RATIO)),
        "int_features_integer": bool(all((syn[c].dropna() % 1 == 0).all() for c in INTCOL)),
        "nan_share_real_vs_syn_pkts_per_flow": [round(float(real.pkts_per_flow.isna().mean()), 3), round(float(syn.pkts_per_flow.isna().mean()), 3)],
        "svc<=ulp_real_vs_syn": [round(float((real.svc_port_unique <= real.unique_local_port_count)[real.svc_port_unique.notna()].mean()), 3),
                                 round(float((syn.svc_port_unique <= syn.unique_local_port_count)[syn.svc_port_unique.notna()].mean()), 3)]}
    out["max_ks_device_mean"] = {c: round(max(v for v in out["ks_device_mean_by_class"][c].values() if v == v), 3) for c in out["ks_device_mean_by_class"]}
    (F / "syn10k_quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------------------------------------------------- experiment
def fit_score(tr_X, tr_y, te_x_frame, te_ids):
    spw = float((tr_y == 0).sum() / max((tr_y == 1).sum(), 1))
    m = lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1, **DEFAULT_LGB).fit(tr_X, tr_y)
    p = pd.Series(m.predict_proba(te_x_frame[USED])[:, 1], index=te_x_frame.index).groupby(te_x_frame.device_id).mean()
    return p


def metric_block(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)), "recall_at_fpr5": float(np.interp(0.05, fpr, tpr))}


def experiment(gen_seeds=(0, 1, 2), total_train=8000, B=1000):
    t0 = time.time()
    x, lab, dev = load()
    g = Gen(x, dev)
    gold = pd.read_csv(F / "gold_union_v6.csv")
    gold = gold.set_index("device_id")
    variants = [("real_only", None, None, None), ("dup_balance", "balance", 0.0, 0.0), ("syn_balance_t0.5", "balance", 0.5, 0.15),
                ("dup_scale8000", "scale", 0.0, 0.0), ("syn_scale8000_t0.25", "scale", 0.25, 0.15), ("syn_scale8000_t0.5", "scale", 0.5, 0.15),
                ("syn_scale8000_t1.0", "scale", 1.0, 0.15)]
    dev_ids = dev.device_id.to_numpy()
    yfull = (dev.label == OTL).to_numpy()
    scores = {}                                                      # (variant, gseed) -> Series(device -> score)
    trcols = USED
    cap_mask = x.cap_rank.to_numpy() < K
    for vname, mode, tau, jit in variants:
        for gs in (gen_seeds if mode else (0,)):
            oof = pd.Series(np.nan, index=dev_ids)
            for f in range(5):
                te_ids = set(dev.loc[dev.fold_device == f, "device_id"])
                tr_dev = dev[~dev.device_id.isin(te_ids)]
                te_mask = x.device_id.isin(te_ids).to_numpy()
                tr_mask = (~te_mask) & cap_mask
                Xtr = x.loc[tr_mask, trcols].to_numpy()
                ytr = (x.loc[tr_mask, "label"] == OTL).to_numpy().astype(int)
                if mode:
                    rng = np.random.default_rng(SEED + 1000 * gs + f)
                    ids_by_class = {c: tr_dev.loc[tr_dev.label == c, "device_id"].to_numpy() for c in (OTL, UNK)}
                    if mode == "balance":
                        n_new = {OTL: max(0, len(ids_by_class[UNK]) - len(ids_by_class[OTL])), UNK: 0}
                    else:
                        n_new = {c: max(0, total_train // 2 - len(ids_by_class[c])) for c in (OTL, UNK)}
                    plan = seed_plan(ids_by_class, n_new, rng)
                    add_X, add_y = [], []
                    for c in (OTL, UNK):
                        if not plan[c]:
                            continue
                        st = g.class_stats(ids_by_class[c])
                        for seed in plan[c]:
                            Rn, _ = g.child(seed, st, tau, jit, rng, K)
                            add_X.append(Rn[:, [IX[c2] for c2 in trcols]])
                            add_y.append(np.full(len(Rn), int(c == OTL)))
                    Xtr = np.vstack([Xtr] + add_X)
                    ytr = np.concatenate([ytr] + add_y)
                p = fit_score(pd.DataFrame(Xtr, columns=trcols), ytr, x[te_mask], te_ids)
                oof.loc[p.index] = p.values
            assert not oof.isna().any()
            scores[(vname, gs)] = oof
            m_rule = metric_block(yfull, oof.to_numpy())
            gi = gold.index.intersection(oof.index)
            yg = (gold.loc[gi, "gold_label"] == "OT").to_numpy()
            m_gold = metric_block(yg, oof.loc[gi].to_numpy())
            sg = oof.loc[gi].to_numpy() >= 0.5
            m_gold["recall_at_0.5"], m_gold["non_ot_flag_rate_at_0.5"] = float((sg & yg).sum() / yg.sum()), float((sg & ~yg).sum() / (~yg).sum())
            print(f"{vname:22s} gseed={gs} rule roc {m_rule['roc_auc']:.4f} pr {m_rule['pr_auc']:.4f} | gold roc {m_gold['roc_auc']:.4f} pr {m_gold['pr_auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            scores[(vname, gs, "m")] = (m_rule, m_gold)
    # tong hop
    rows = []
    for vname, mode, tau, jit in variants:
        ms = [scores[(vname, gs, "m")] for gs in (gen_seeds if mode else (0,))]
        row = {"variant": vname, "n_gen_seeds": len(ms)}
        for tag, k in (("rule", 0), ("gold", 1)):
            for met in ("roc_auc", "pr_auc", "recall_at_fpr5"):
                v = [m[k][met] for m in ms]
                row[f"{tag}_{met}_mean"], row[f"{tag}_{met}_sd"] = round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
        v = [m[1]["recall_at_0.5"] for m in ms]
        w = [m[1]["non_ot_flag_rate_at_0.5"] for m in ms]
        row["gold_recall_at_0.5_mean"], row["gold_non_ot_flag_at_0.5_mean"] = round(float(np.mean(v)), 3), round(float(np.mean(w)), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    # bootstrap cap thiet bi: chenh lech so voi real_only (diem hat 0)
    rng = np.random.default_rng(SEED)
    base = scores[("real_only", 0)]
    gi = gold.index.intersection(base.index)
    yg = (gold.loc[gi, "gold_label"] == "OT").to_numpy()
    boot = {}
    for vname, mode, *_ in variants[1:]:
        s = scores[(vname, 0)]
        dr, dg = [], []
        for _ in range(B):
            i = rng.integers(0, len(yfull), len(yfull))
            if 0 < yfull[i].sum() < len(i):
                dr.append(roc_auc_score(yfull[i], s.to_numpy()[i]) - roc_auc_score(yfull[i], base.to_numpy()[i]))
            j = rng.integers(0, len(yg), len(yg))
            if 0 < yg[j].sum() < len(j):
                dg.append(roc_auc_score(yg[j], s.loc[gi].to_numpy()[j]) - roc_auc_score(yg[j], base.loc[gi].to_numpy()[j]))
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
        boot[vname] = {"rule_roc_diff[2.5,mean,97.5]": ci(dr), "gold_roc_diff[2.5,mean,97.5]": ci(dg)}
    pd.DataFrame({f"{k[0]}|g{k[1]}": v for k, v in scores.items() if len(k) == 2}).to_csv(F / "phase1_syn10k_oof_scores.csv")
    summary.to_csv(F / "phase1_syn10k_experiment.csv", index=False)
    res = {"protocol": {"model": "LightGBM default (frozen Phase 1 params)", "features": USED, "K": K, "train_devices_target_for_scale": total_train,
                        "folds": "fold_device; synthetic generated per fold from TRAINING devices only; test = real devices only", "gen_seeds": list(gen_seeds),
                        "gold": "gold_union_v6.csv (184)", "bootstrap_B": B}, "summary": rows, "bootstrap_diff_vs_real_only_seed0": boot,
           "seconds": round(time.time() - t0)}
    (F / "phase1_syn10k_experiment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(boot, indent=1))


def cv_on_file(x, dev, test_real_only, model_seed):
    """Huan luyen tren file (fold_device cua chinh file do), cap_rank<K; cham tren thiet bi cua fold giu lai."""
    x = x.sort_values(["device_id", "cap_rank"], kind="mergesort").reset_index(drop=True)
    fold_of = dict(zip(dev.device_id, dev.fold_device.astype(int)))
    fx = x.device_id.map(fold_of).to_numpy()
    is_syn = x.device_id.str.startswith("SYN_").to_numpy()
    y_win = (x.label == OTL).to_numpy().astype(int)
    oof = {}
    for f in range(5):
        tr = (fx != f) & (x.cap_rank.to_numpy() < K)
        te = (fx == f) & (~is_syn if test_real_only else True)
        spw = float((y_win[tr] == 0).sum() / max((y_win[tr] == 1).sum(), 1))
        m = lgb.LGBMClassifier(scale_pos_weight=spw, subsample_freq=1, random_state=model_seed, n_jobs=8, verbose=-1, **DEFAULT_LGB).fit(x.loc[tr, USED], y_win[tr])
        p = pd.Series(m.predict_proba(x.loc[te, USED])[:, 1], index=x.index[te]).groupby(x.device_id[te]).mean()
        oof.update(p.to_dict())
    return pd.Series(oof)


def compare_files(B=1000):
    t0 = time.time()
    x_old = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    d_old = pd.read_csv(F / "device_folds_v6.csv", dtype=str, keep_default_na=False)
    x_new = pd.read_parquet(F / "master_device_windows_v6_fe3_syn10k.parquet")
    d_new = pd.read_csv(F / "device_folds_v6_syn10k.csv", dtype=str, keep_default_na=False)
    gold = pd.read_csv(F / "gold_union_v6.csv").set_index("device_id")
    lab_all = d_new.set_index("device_id").label
    runs = {"old_file_test_real": (x_old, d_old, True), "new_file_test_real": (x_new, d_new, True), "new_file_test_all10k": (x_new, d_new, False)}
    seeds = (SEED, SEED + 1, SEED + 2)
    sc = {}
    for k, (x, d, ro) in runs.items():
        sc[k] = [cv_on_file(x, d, ro, s) for s in seeds]
        print(k, "done", f"{time.time() - t0:.0f}s", flush=True)

    def mets(s, ids):
        y = (lab_all.loc[ids] == OTL).to_numpy()
        out = {"n": int(len(ids)), "rule_roc": float(roc_auc_score(y, s.loc[ids].to_numpy())), "rule_pr": float(average_precision_score(y, s.loc[ids].to_numpy()))}
        fpr, tpr, _ = roc_curve(y, s.loc[ids].to_numpy())
        out["rule_recall_fpr5"] = float(np.interp(0.05, fpr, tpr))
        pred = s.loc[ids].to_numpy() >= 0.5
        out["rule_recall@0.5"], out["rule_precision@0.5"] = float((pred & y).sum() / y.sum()), float((pred & y).sum() / max(pred.sum(), 1))
        gi = [i for i in ids if i in gold.index]
        if gi:
            yg = (gold.loc[gi, "gold_label"] == "OT").to_numpy()
            sg = s.loc[gi].to_numpy()
            out["gold_n"], out["gold_roc"], out["gold_pr"] = int(len(gi)), float(roc_auc_score(yg, sg)), float(average_precision_score(yg, sg))
            out["gold_recall@0.5"], out["gold_nonOT_flag@0.5"] = float(((sg >= 0.5) & yg).sum() / yg.sum()), float(((sg >= 0.5) & ~yg).sum() / (~yg).sum())
        return out

    real_ids = [i for i in d_old.device_id]
    all_ids = list(d_new.device_id)
    rows = []
    for k in runs:
        ids = all_ids if k.endswith("all10k") else real_ids
        per = [mets(s, ids) for s in sc[k]]
        row = {"run": k}
        for key in per[0]:
            v = [p[key] for p in per]
            row[key] = round(float(np.mean(v)), 4)
            if isinstance(v[0], float):
                row[key + "_sd"] = round(float(np.std(v)), 4)
        rows.append(row)
    # tach theo nguon cua nhom 10k: rieng thiet bi sinh, rieng thiet bi that
    syn_ids = [i for i in all_ids if i.startswith("SYN_")]
    per = [mets(s, syn_ids) for s in sc["new_file_test_all10k"]]
    rows.append({"run": "new_file_test_syn_only", **{k: round(float(np.mean([p[k] for p in per])), 4) for k in per[0] if not k.startswith("gold")}})
    # bootstrap cap: that-only cu vs moi (diem hat mo hinh 0) tren cung 3.923 thiet bi that
    a, b = sc["old_file_test_real"][0].loc[real_ids].to_numpy(), sc["new_file_test_real"][0].loc[real_ids].to_numpy()
    y = (lab_all.loc[real_ids] == OTL).to_numpy()
    gi = [i for i in real_ids if i in gold.index]
    yg = (gold.loc[gi, "gold_label"] == "OT").to_numpy()
    ga, gb = sc["old_file_test_real"][0].loc[gi].to_numpy(), sc["new_file_test_real"][0].loc[gi].to_numpy()
    rng = np.random.default_rng(SEED)
    dr, dg = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            dr.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
        j = rng.integers(0, len(yg), len(yg))
        if 0 < yg[j].sum() < len(j):
            dg.append(roc_auc_score(yg[j], gb[j]) - roc_auc_score(yg[j], ga[j]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.mean(v)), 4), round(float(np.percentile(v, 97.5)), 4)]
    res = {"protocol": "LightGBM frozen params, 15 features, K=20, fold_device of each file, 3 model seeds; test on held-out fold devices",
           "rows": rows, "paired_bootstrap_new_minus_old_seed0": {"rule_roc[2.5,mean,97.5]": ci(dr), "gold_roc[2.5,mean,97.5]": ci(dg)}, "seconds": round(time.time() - t0)}
    (F / "phase1_syn10k_file_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(F / "phase1_syn10k_file_compare.csv", index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(json.dumps(res["paired_bootstrap_new_minus_old_seed0"], indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"build": build, "quality": quality, "experiment": experiment, "compare_files": compare_files}.get(cmd, lambda: print(__doc__))()
