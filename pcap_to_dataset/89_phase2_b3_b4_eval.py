"""Phase 2: danh gia baseline B3 (Al Ghazo & Kumar 2019) va B4 (Pospisil & Fujdiak 2024) tren CUNG tap A (117 thiet bi co ho so), cung chi so (2026-11-13).
KHONG ghi de ket qua dong bang. Day la CAI DAT LAI GIAN LUOC tren du lieu cua luan van, khong phai ket qua goc cua tac gia.

B3: thuat toan nhan dien tang dieu khien (Algorithm 1 cua bai bao) theo tung session tren do thi giao tiep unicast giua cac MAC; tang 0 -> PLC, tang >= 1 -> HMI (SCADA gop vao HMI); thiet bi khong xuat hien
    trong do thi -> OTHER; EWS khong bao gio duoc du doan (bai bao khong co lop nay). Khong huan luyen. Gop nhieu session: PLC neu >= 50% so session o tang 0.
    V1_all      moi canh unicast, hai huong (dung mo ta cua bai bao)
    V2_ot       chi canh OT (it nhat mot cong thuoc danh muc cong OT), hai huong
    V3_ot_req   chi canh huong yeu cau OT (client -> server); may chi la dich (khong gui yeu cau) o tang 0, may khach xep tang theo thuat toan o tang 1 tro len (BIEN DOI, khong co trong bai bao)
B4: dac trung entropy/tong byte 24 byte dau payload theo mau 15 phut (10 dac trung), XGBoost; 3 cau hinh khai bao truoc: P1 (sieu tham so kich ban 3 cua bai bao: 2 cay, do sau 1, lr 0,01),
    P2 (200 cay, do sau 3, lr 0,1) tren 10 dac trung, P2e tren 5 dac trung entropy. StratifiedGroupKFold (nhom = session dau), 5 hat x 5 fold; diem thiet bi = trung binh xac suat cac mau.
Chay: py -3.14 89_phase2_b3_b4_eval.py
"""
import importlib.util
import json
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

os.environ.setdefault("P2_PROFILE", "phase2_payload_profiles_v6_full.json")
os.environ.setdefault("P2_SEQ", "phase2_sequences_v6_full.pkl")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
CACHE = F / "b3b4_cache_v6"
CLASSES = ["PLC", "HMI", "EWS", "OTHER"]


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def levels(D):
    """Algorithm 1 (Al Ghazo & Kumar): D: {nguon: frozenset(dich)} -> ({nguon: tang}, fallback)."""
    D = {s: frozenset(x for x in v if x != s) for s, v in D.items()}
    D = {s: v for s, v in D.items() if v}
    groups = defaultdict(list)
    for s, v in D.items():
        groups[v].append(s)
    gl = list(groups.items())
    lvl, fallback = {}, False
    cur = [i for i, (vi, _) in enumerate(gl) if not any(vj < vi for j, (vj, _) in enumerate(gl) if j != i)]
    k = 0
    while cur:
        for i in cur:
            for s in gl[i][1]:
                lvl[s] = k
        dests = set().union(*[gl[i][0] for i in cur])
        un = [i for i in range(len(gl)) if gl[i][1][0] not in lvl]
        nxt = [i for i in un if any(s in dests for s in gl[i][1])]
        if not nxt and un:
            nxt, fallback = un, True
        cur, k = nxt, k + 1
    return lvl, fallback


def b3(variant, sessions):
    """Tra ve {device: [tang theo tung session]}, thong ke thoai hoa."""
    per_dev, stats = defaultdict(list), {"sessions": 0, "single_level": 0, "fallback": 0, "sources_median": []}
    for sess, d in sessions.items():
        E = d["edges"]
        if variant == "V1_all":
            sel = [(s, t) for (s, t, f), c in E.items()]
        elif variant == "V2_ot":
            sel = [(s, t) for (s, t, f), c in E.items() if f > 0]
        else:
            sel = [(s, t) for (s, t, f), c in E.items() if f == 1]
        if not sel:
            continue
        D = defaultdict(set)
        for s, t in sel:
            D[s].add(t)
        lv, fb = levels({s: frozenset(v) for s, v in D.items()})
        if variant == "V3_ot_req":
            servers = {t for s, t in sel} - set(D)
            lv = {s: k + 1 for s, k in lv.items()}
            lv.update({s: 0 for s in servers})
        stats["sessions"] += 1
        stats["fallback"] += int(fb)
        stats["single_level"] += int(len(set(lv.values())) <= 1)
        stats["sources_median"].append(len(lv))
        for mac, dev in d["targets"].items():
            if mac in lv:
                per_dev[dev].append(lv[mac])
    stats["sources_median"] = float(np.median(stats["sources_median"])) if stats["sources_median"] else 0
    return per_dev, stats


def b3_predict(per_dev, devs):
    out = []
    for d in devs:
        v = per_dev.get(d)
        if not v:
            out.append("OTHER")
        else:
            out.append("PLC" if np.mean(np.array(v) == 0) >= 0.5 else "HMI")
    return np.array(out)


def metr(y, p):
    return {"n": int(len(y)), "balanced_acc": round(float(balanced_accuracy_score(y, p)), 3),
            "recall": dict(zip(CLASSES, np.round(recall_score(y, p, labels=CLASSES, average=None, zero_division=0), 3).tolist())),
            "confusion": confusion_matrix(y, p, labels=CLASSES).tolist()}


def boot(y, p, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    v = [balanced_accuracy_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main():
    S62 = _load("s62", "62_phase2_dl_seq.py")
    S61, S58 = S62.S61, S62.S58
    lab, _ = S58.build()
    subA = lab[lab.has_profile & lab.index.isin(S62.DEV)]
    lab144 = pd.read_csv(F / "phase2_type_labels_4class_plus_v6_plus_all.csv").set_index("device_id")
    sessions = {p.stem: pickle.loads(p.read_bytes()) for p in CACHE.glob("*.pkl")}
    print("sessions in cache:", len(sessions), flush=True)
    out = {"status": "EXPLORATORY re-implementations; frozen results untouched", "sessions": len(sessions), "setA_n": int(len(subA)), "B3": {}, "B4": {}}
    # ---------------------------------------------------------------- B3
    devsA, yA = list(subA.index), subA.type4.to_numpy()
    devs144 = [d for d in lab144.index if d in set(sum([list(s["targets"].values()) for s in sessions.values()], []))]
    y144 = lab144.loc[devs144, "type4"].to_numpy()
    for var in ("V1_all", "V2_ot", "V3_ot_req"):
        per, st = b3(var, sessions)
        pA, p144 = b3_predict(per, devsA), b3_predict(per, devs144)
        blk = {"graph_stats": st, "setA": metr(yA, pA), "setA_ci95": boot(yA, pA), "set144": metr(y144, p144), "devices_in_graph_setA": int(sum(d in per for d in devsA)),
               "pred_counts_setA": dict(Counter(pA))}
        m = np.isin(yA, ["PLC", "HMI"])
        pm = pA[m]
        blk["plc_vs_hmi_only"] = {"n": int(m.sum()), "accuracy": round(float((pm == yA[m]).mean()), 3), "predicted_as_PLC_or_HMI": int(np.isin(pm, ["PLC", "HMI"]).sum())}
        out["B3"][var] = blk
        print("B3", var, blk["setA"]["balanced_acc"], blk["setA_ci95"], blk["setA"]["recall"], blk["plc_vs_hmi_only"], flush=True)
        pd.DataFrame({"device_id": devsA, "true": yA, "pred": pA, "levels": [per.get(d, []) for d in devsA]}).to_csv(F / f"phase2_wp2b_b3_{var}_setA.csv", index=False)
    # ---------------------------------------------------------------- B4
    import xgboost as xgb
    X, nsamp = {}, {}
    for d in devsA:
        rows = []
        for s in sessions.values():
            v = s["samples"].get(d)
            if v:
                rows += [r[:10] for r in v["samples"]]
        X[d] = np.array(rows, dtype=float) if rows else np.full((1, 10), np.nan)
        nsamp[d] = len(rows)
    out["B4"]["devices_with_samples_setA"] = int(sum(v > 0 for v in nsamp.values()))
    out["B4"]["samples_per_device_median"] = float(np.median([v for v in nsamp.values() if v > 0])) if any(nsamp.values()) else 0
    gA = subA.group.to_numpy()
    cfgs = {"P1_paper_scenario3": (dict(n_estimators=2, max_depth=1, learning_rate=0.01, subsample=0.8, gamma=0.1, max_delta_step=5, reg_lambda=1, colsample_bytree=1.0), slice(0, 10)),
            "P2_standard": (dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), slice(0, 10)),
            "P2e_entropy_only": (dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), slice(0, 5))}
    for name, (params, cols) in cfgs.items():
        def fp(tr, te, seed, yy, params=params, cols=cols):
            Xtr = np.vstack([X[devsA[i]][:, cols] for i in tr])
            ytr = np.concatenate([[yy[i]] * len(X[devsA[i]]) for i in tr])
            cnt = np.bincount(ytr, minlength=4).astype(float)
            w = np.where(cnt[ytr] > 0, len(ytr) / (4 * np.maximum(cnt[ytr], 1)), 1.0)
            m = xgb.XGBClassifier(objective="multi:softprob", num_class=4, random_state=seed, n_jobs=8, verbosity=0, **params).fit(Xtr, ytr, sample_weight=w)
            P = np.zeros((len(te), 4))
            for k, i in enumerate(te):
                pr = m.predict_proba(X[devsA[i]][:, cols])
                full = np.zeros((len(pr), 4))
                full[:, m.classes_.astype(int)] = pr
                P[k] = full.mean(0)
            return P
        res, oof0 = S61.cv_generic(fp, None, yA, gA, seeds=5)
        out["B4"][name] = res
        print("B4", name, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], flush=True)
    (F / "phase2_wp2b_b3_b4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-B3B4")


if __name__ == "__main__":
    sys.exit(main())
