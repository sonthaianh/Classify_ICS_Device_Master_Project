"""Nhom A cua so theo doi (2026-11-25): chay lai giai doan 2 va baseline tren BO NHAN DOC LAP. KHONG ghi de ket qua dong bang.

Bo nhan doc lap (indep160) = 186 nhan - 5 EWS duyet theo ten session (owner_approved_session_evidence)
                                      - 21 nhan ra soat tu bang chung session (reviewed_session_evidence 14 + reviewed_provisional 7) = 160 nhan.
  --set indep160 | x186     bo nhan (x186 chi de kiem tra script tai hien dung so cu)
  --tab   A1 + A3: RF 64 cot (tap A, tap T), luat vai tro theo cong, MLP 64 cot, B2 (OUI), B3 (3 bien the), B4 (3 cau hinh), B5 (1-NN khop mo),
          chia theo thanh phan lien thong, dau-cuoi hai giai doan, doi chung mot giai doan. Luu du doan tung hat.
  --cnn   CNN 1D tu dau, 5 hat (tap A).            --tf   Transformer tu dau, 3 hat (tap A; cham).
  --report  gop du doan da luu: trung binh +- do lech chuan, khoang tin cay bootstrap (B = 2000, theo thiet bi), hieu so ghep cap RF - mo hinh khac,
            do phu / precision / F1 theo lop, ma tran nham lan.
Cung giao thuc voi bo 186: StratifiedGroupKFold 5 fold (nhom = capture dau tien), RF 300 cay, min_samples_leaf 2, class_weight balanced.
Dau ra: phase2_groupA_preds_<set>_<stage>.json, phase2_groupA_<set>.json.
Chay: py -3.14 110_group_A_independent.py --set indep160 --tab   (roi --cnn, --tf, --report)
"""
import importlib.util
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SET = arg("--set", "indep160")
SRC186 = F / "phase2_type_labels_4class_186_devices.csv"
DROP = ["owner_approved_session_evidence", "reviewed_session_evidence", "reviewed_provisional"]
if SET == "indep160":
    LABEL = F / "phase2_type_labels_4class_186_devices_indep160.csv"
    raw = M._orig(SRC186, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    keep = raw[~raw.source.isin(DROP)]
    assert len(raw) == 186 and len(keep) == 160, (len(raw), len(keep))
    keep.to_csv(LABEL, index=False, encoding="utf-8-sig")
elif SET == "x186":
    LABEL = SRC186
else:
    raise SystemExit("--set indep160 | x186")
CUR["path"] = str(LABEL)
S62 = M._load("s62", "62_phase2_dl_seq.py")
S61, S58 = S62.S61, S62.S58
CLASSES = S58.CLASSES
C2I = {c: i for i, c in enumerate(CLASSES)}
KS = "behaviour_payload_role_eng"
OUT = F / f"phase2_groupA_{SET}.json"
B = 2000
THR = 0.5


def pred_path(stage):
    return F / f"phase2_groupA_preds_{SET}_{stage}.json"


def save_res(key, val):
    d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def bal4(y, p):
    """Do chinh xac can bang = trung binh do phu cua cac lop co mat trong y (du doan co the la UNKNOWN)."""
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean([(p[y == c] == c).mean() for c in CLASSES if (y == c).any()]))


def rf_preds(X, y, g, seed, n_est=300):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        m = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def generic(fp, y, g, seeds, tag=""):
    yi = np.array([C2I[c] for c in y])
    out = []
    for seed in range(seeds):
        pred = np.empty(len(y), dtype=int)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), yi, g):
            pred[te] = fp(tr, te, seed, yi).argmax(1)
        out.append(np.array(CLASSES, dtype=object)[pred])
        print(f"  {tag} seed {seed}: {balanced_accuracy_score(y, out[-1]):.3f}", flush=True)
    return out


def ms(v):
    return {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3), "per_seed": [round(float(x), 3) for x in v]}


def base():
    lab, sets = S58.build()
    eng = M._orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    A = lab[lab.has_profile & lab.index.isin(S62.DEV)]
    assert len(A) == int(lab.has_profile.sum()), "mot so thiet bi co ho so nhung khong co chuoi su kien"
    T = lab[tok]
    return lab, sets, A, T


def components_of(devs, ds):
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for d in devs:
        parent[("d", d)] = ("d", d)
        for s_ in ds[d]:
            parent.setdefault(("s", s_), ("s", s_))
            ra, rb = find(("d", d)), find(("s", s_))
            if ra != rb:
                parent[ra] = rb
    return {d: str(find(("d", d))) for d in devs}


def do_tab():
    t0 = time.time()
    lab, sets, A, T = base()
    cnt = lambda s: s.type4.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
    info = {"label_file": LABEL.name, "n_labels": int(len(lab)), "labels_by_class": cnt(lab), "set_A_n": int(len(A)), "set_A_by_class": cnt(A),
            "set_T_n": int(len(T)), "set_T_by_class": cnt(T), "n_first_capture_groups_A": int(A.group.nunique())}
    print(info, flush=True)
    save_res("sets", info)
    y, g, devs = A.type4.to_numpy(), A.group.to_numpy(), list(A.index)
    X = sets[KS].loc[A.index]
    P = {"A": {"devices": devs, "y": list(y), "preds": {}}, "T": {"devices": list(T.index), "y": list(T.type4), "preds": {}}}
    pa = P["A"]["preds"]
    # ---- RF 64 cot, tap A va tap T
    pa["RF64"] = [rf_preds(X, y, g, s) for s in range(5)]
    print("RF64 A", ms([balanced_accuracy_score(y, p) for p in pa["RF64"]]), round(time.time() - t0), "s", flush=True)
    yT, gT = T.type4.to_numpy(), T.group.to_numpy()
    P["T"]["preds"]["RF64"] = [rf_preds(sets[KS].loc[T.index], yT, gT, s) for s in range(5)]
    print("RF64 T", ms([balanced_accuracy_score(yT, p) for p in P["T"]["preds"]["RF64"]]), flush=True)
    # ---- luat vai tro theo cong (khong huan luyen)
    role = M._orig(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    for nm, sub in (("A", A), ("T", T)):
        fr = role.reindex(sub.index).final_role.fillna("unknown")
        P[nm]["preds"]["Luat_vai_tro"] = [np.array(np.where(fr == "server", "PLC", np.where(fr == "client", "HMI", "OTHER")), dtype=object)]
    print("Luat vai tro A", round(balanced_accuracy_score(y, pa["Luat_vai_tro"][0]), 3), flush=True)
    # ---- MLP 64 cot
    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
    pa["MLP64"] = generic(lambda tr, te, seed, yy: S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), y, g, 5, "MLP64")
    print("MLP done", round(time.time() - t0), "s", flush=True)
    # ---- B2: chi OUI -> Random Forest (script 83)
    macs = M._orig(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False).set_index("device_id").device_mac
    oui = macs.reindex(devs).str.lower().str[:8].fillna("na")
    oui_c = oui.where(oui.map(oui.value_counts()) >= 3, "rare")
    Xo = pd.get_dummies(oui_c).to_numpy().astype(float)

    def b2_fp(tr, te, seed, yy):
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=1, random_state=seed, n_jobs=-1).fit(Xo[tr], yy[tr])
        Pm = np.zeros((len(te), 4))
        Pm[:, m.classes_.astype(int)] = m.predict_proba(Xo[te])
        return Pm
    pa["B2_OUI_RF"] = generic(b2_fp, y, g, 5, "B2")
    # ---- B5: 1-NN khop mo tren chuoi ky hieu da nen lap (script 83)
    S83 = M._load("s83", "83_phase2_baselines_rle.py")
    seqs = []
    for d in devs:
        s = S62.DEV[d][0]
        seqs.append(s[np.r_[True, s[1:] != s[:-1]]][:200])
    n = len(devs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = S83.lev_norm(seqs[i], seqs[j])

    def b5_fp(tr, te, seed, yy):
        Pm = np.zeros((len(te), 4))
        for k, t in enumerate(te):
            Pm[k, yy[tr[np.argmin(D[t, tr])]]] = 1.0
        return Pm
    pa["B5_fuzzy_1NN"] = generic(b5_fp, y, g, 5, "B5")
    print("B2/B5 done", round(time.time() - t0), "s", flush=True)
    # ---- B3 (Al Ghazo & Kumar) va B4 (Pospisil & Fujdiak) (script 89)
    S89 = M._load("s89", "89_phase2_b3_b4_eval.py")
    sessions = {p.stem: pickle.loads(p.read_bytes()) for p in S89.CACHE.glob("*.pkl")}
    b3info = {}
    for var in ("V1_all", "V2_ot", "V3_ot_req"):
        per, st = S89.b3(var, sessions)
        p3 = S89.b3_predict(per, devs)
        pa[f"B3_{var}"] = [np.array(p3, dtype=object)]
        b3info[var] = {"devices_in_graph": int(sum(d in per for d in devs)), "pred_counts": dict(Counter(p3))}
        print("B3", var, round(balanced_accuracy_score(y, p3), 3), flush=True)
    save_res("B3_info", b3info)
    import xgboost as xgb
    X4, nsamp = {}, {}
    for d in devs:
        rows = []
        for s in sessions.values():
            v = s["samples"].get(d)
            if v:
                rows += [r[:10] for r in v["samples"]]
        X4[d] = np.array(rows, dtype=float) if rows else np.full((1, 10), np.nan)
        nsamp[d] = len(rows)
    save_res("B4_info", {"devices_with_samples": int(sum(v > 0 for v in nsamp.values())), "n": len(devs)})
    cfgs = {"P1_paper_scenario3": (dict(n_estimators=2, max_depth=1, learning_rate=0.01, subsample=0.8, gamma=0.1, max_delta_step=5, reg_lambda=1, colsample_bytree=1.0), slice(0, 10)),
            "P2_standard": (dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), slice(0, 10)),
            "P2e_entropy_only": (dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), slice(0, 5))}
    for name, (params, cols) in cfgs.items():
        def b4_fp(tr, te, seed, yy, params=params, cols=cols):
            Xtr = np.vstack([X4[devs[i]][:, cols] for i in tr])
            ytr = np.concatenate([[yy[i]] * len(X4[devs[i]]) for i in tr])
            c = np.bincount(ytr, minlength=4).astype(float)
            w = np.where(c[ytr] > 0, len(ytr) / (4 * np.maximum(c[ytr], 1)), 1.0)
            m = xgb.XGBClassifier(objective="multi:softprob", num_class=4, random_state=seed, n_jobs=8, verbosity=0, **params).fit(Xtr, ytr, sample_weight=w)
            Pm = np.zeros((len(te), 4))
            for k, i in enumerate(te):
                pr = m.predict_proba(X4[devs[i]][:, cols])
                full = np.zeros((len(pr), 4))
                full[:, m.classes_.astype(int)] = pr
                Pm[k] = full.mean(0)
            return Pm
        pa[f"B4_{name}"] = generic(b4_fp, y, g, 5, f"B4_{name}")
    print("B3/B4 done", round(time.time() - t0), "s", flush=True)
    dump = {k: {"devices": v["devices"], "y": v["y"], "preds": {m: [list(map(str, p)) for p in pp] for m, pp in v["preds"].items()}} for k, v in P.items()}
    pred_path("tab").write_text(json.dumps(dump), encoding="utf-8")
    # ---- chia theo thanh phan lien thong (12 lan danh so lai thanh phan x 5 hat)
    win = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    ds = win[win.device_id.isin(A.index)].groupby("device_id").session_id.agg(lambda s: set(s)).to_dict()
    comp = pd.Series(components_of(devs, ds))
    ids = sorted(comp.unique())
    rel = []
    for r in range(12):
        rng = np.random.default_rng(1000 + r)
        perm = dict(zip(ids, rng.permutation(len(ids))))
        gg = comp.loc[A.index].map(perm).to_numpy()
        rel.append(float(np.mean([balanced_accuracy_score(y, rf_preds(X, y, gg, s)) for s in range(5)])))
        print("  component relabel", r, round(rel[-1], 3), flush=True)
    # ti le thiet bi kiem tra dung chung capture voi tap huan luyen khi nhom = capture dau tien
    share = []
    for s in range(5):
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s).split(X, y, g):
            trs = set().union(*[ds[devs[i]] for i in tr])
            share += [bool(ds[devs[i]] & trs) for i in te]
    save_res("component_split", {"n_components": len(ids), "relabelings": [round(x, 3) for x in rel], "mean": round(float(np.mean(rel)), 3), "sd": round(float(np.std(rel)), 3),
                                 "min": round(float(np.min(rel)), 3), "max": round(float(np.max(rel)), 3),
                                 "share_test_devices_sharing_a_capture_with_train_first_capture_grouping": round(float(np.mean(share)), 3)})
    print("component done", round(time.time() - t0), "s", flush=True)
    # ---- dau-cuoi hai giai doan (cong loc giai doan 1, 15 dac trung, diem out-of-fold dong bang) va doi chung mot giai doan
    p1 = M._orig(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    L = lab.copy()
    L["p1"] = p1.reindex(L.index)
    assert L.p1.notna().all()
    yL = L.type4.to_numpy()
    ot = np.isin(yL, ["PLC", "HMI", "EWS"])
    blocked = (L.p1 < THR).to_numpy()
    e2e, oracle, one = [], [], []
    for seed in range(5):
        pr = pd.Series(pa["RF64"][seed], index=A.index)
        fin, orc = [], []
        for d in L.index:
            b = L.p1[d] < THR
            if not L.has_profile[d]:
                fin.append("OTHER" if b else "UNKNOWN")
                orc.append("UNKNOWN")
            else:
                fin.append("OTHER" if b else pr[d])
                orc.append(pr[d])
        e2e.append(np.array(fin, dtype=object))
        oracle.append(np.array(orc, dtype=object))
        one.append(rf_preds(sets[KS].loc[L.index], yL, L.group.to_numpy(), seed))
    st = []
    for i, d in enumerate(L.index):
        if not L.has_profile[d]:
            s_ = "no_profile_blocked" if blocked[i] else "no_profile_passed"
        elif blocked[i]:
            s_ = "blocked_by_stage1"
        else:
            s_ = "stage2_correct" if pa["RF64"][0][devs.index(d)] == yL[i] else "stage2_wrong_role"
        st.append(s_)
    rec = lambda preds: {c: round(float(np.mean([(p[yL == c] == c).mean() for p in preds])), 3) for c in CLASSES}
    res = {"n_labels": int(len(L)), "labels_by_class": cnt(L), "n_profiled": int(len(A)), "threshold": THR,
           "blocked_by_stage1_by_class": {c: f"{int(blocked[yL == c].sum())}/{int((yL == c).sum())}" for c in CLASSES},
           "blocked_OT_devices": f"{int(blocked[ot].sum())}/{int(ot.sum())}",
           "two_stage_e2e_bal_acc": ms([bal4(yL, p) for p in e2e]), "two_stage_e2e_recall": rec(e2e), "two_stage_e2e_ot3_accuracy": ms([float((p[ot] == yL[ot]).mean()) for p in e2e]),
           "oracle_gate_bal_acc": ms([bal4(yL, p) for p in oracle]),
           "one_stage_bal_acc": ms([bal4(yL, p) for p in one]), "one_stage_recall": rec(one), "one_stage_ot3_accuracy": ms([float((p[ot] == yL[ot]).mean()) for p in one]),
           "seed0_status_by_true_class": pd.crosstab(pd.Series(yL, name="true"), pd.Series(st, name="status")).to_dict("index")}
    rng = np.random.default_rng(20261125)
    dd = []
    for _ in range(B):
        i = rng.integers(0, len(yL), len(yL))
        dd.append(np.mean([bal4(yL[i], p[i]) for p in e2e]) - np.mean([bal4(yL[i], p[i]) for p in one]))
    res["two_stage_minus_one_stage_paired_bootstrap"] = {"point": round(res["two_stage_e2e_bal_acc"]["mean"] - res["one_stage_bal_acc"]["mean"], 3),
                                                         "ci95": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)], "share_le_0": round(float(np.mean(np.array(dd) <= 0)), 3)}
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    save_res("end_to_end", res)
    pred_path("e2e").write_text(json.dumps({"devices": list(L.index), "y": list(yL), "e2e": [list(map(str, p)) for p in e2e], "one_stage": [list(map(str, p)) for p in one]}), encoding="utf-8")
    print("DONE-TAB", round(time.time() - t0), "s", flush=True)


def do_seq(kind):
    t0 = time.time()
    lab, sets, A, T = base()
    y, g, devs = A.type4.to_numpy(), A.group.to_numpy(), list(A.index)
    if kind == "cnn":
        mod, key, seeds = S62, "CNN1D", 5
    else:
        mod, key, seeds = M._load("s71", "71_phase2_dl_transformer.py"), "Transformer", 3
    yi = np.array([C2I[c] for c in y])
    preds = []
    for seed in range(seeds):
        pred = np.empty(len(y), dtype=int)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), yi, g):
            pred[te] = mod.ft_predict([devs[i] for i in tr], yi[tr], [devs[i] for i in te], seed, None).argmax(1)
        preds.append(list(np.array(CLASSES)[pred]))
        print(key, "seed", seed, round(balanced_accuracy_score(y, preds[-1]), 3), round(time.time() - t0), "s", flush=True)
        pred_path(kind).write_text(json.dumps({"A": {"devices": devs, "y": list(y), "preds": {key: preds}}}), encoding="utf-8")
    print("DONE-" + kind.upper(), flush=True)


def summarize(y, preds):
    cm = np.zeros((4, 4), int)
    for p in preds:
        cm += confusion_matrix(y, p, labels=CLASSES)
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    r = lambda v: [round(float(x), 3) for x in v]
    return {"n_seeds": len(preds), **ms([balanced_accuracy_score(y, p) for p in preds]), "accuracy": round(float(cm.diagonal().sum() / cm.sum()), 3),
            "recall": dict(zip(CLASSES, r(rec))), "precision": dict(zip(CLASSES, r(prec))), "f1": dict(zip(CLASSES, r(f1))), "macro_f1": round(float(f1.mean()), 3),
            "confusion_rows_true_cols_pred_sum_over_seeds": cm.tolist()}


def do_report():
    tab = json.loads(pred_path("tab").read_text(encoding="utf-8"))
    y = np.array(tab["A"]["y"])
    preds = {k: [np.array(p) for p in v] for k, v in tab["A"]["preds"].items()}
    for stage in ("cnn", "tf"):
        if pred_path(stage).exists():
            d = json.loads(pred_path(stage).read_text(encoding="utf-8"))["A"]
            assert d["devices"] == tab["A"]["devices"]
            preds.update({k: [np.array(p) for p in v] for k, v in d["preds"].items()})
    n = len(y)
    out = {"n": int(n), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(), "bootstrap_B": B, "models": {}}
    rng = np.random.default_rng(20261111)                 # cung hat va cung cach rut voi script 72 (bang hat 0)
    idx0 = [rng.integers(0, n, n) for _ in range(B)]
    rng2 = np.random.default_rng(20261122)                # cung hat voi script 105 (trung binh cac hat)
    idxm = [rng2.integers(0, n, n) for _ in range(B)]
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    bal_mean = lambda pp, i: float(np.mean([balanced_accuracy_score(y[i], p[i]) for p in pp]))
    s0 = {k: np.array([balanced_accuracy_score(y[i], pp[0][i]) for i in idx0]) for k, pp in preds.items()}
    sm = {k: np.array([bal_mean(pp, i) for i in idxm]) for k, pp in preds.items()}
    for k, pp in preds.items():
        blk = summarize(y, pp)
        blk["seed0"] = {"point": round(float(balanced_accuracy_score(y, pp[0])), 3), "ci95": ci(s0[k])}
        blk["mean_over_seeds_bootstrap_ci95"] = ci(sm[k])
        if k != "RF64":
            d0 = s0["RF64"] - s0[k]
            blk["RF64_minus_this_seed0_paired"] = {"point": round(float(balanced_accuracy_score(y, preds["RF64"][0]) - balanced_accuracy_score(y, pp[0])), 3), "mean_diff": round(float(d0.mean()), 3),
                                                   "ci95": ci(d0), "share_le_0": round(float((d0 <= 0).mean()), 3)}
            ns = len(pp)
            rf_same = preds["RF64"] if ns == 1 else preds["RF64"][:ns]
            dm = np.array([bal_mean(rf_same, i) for i in idxm]) - sm[k] if ns not in (1, 5) else sm["RF64"] - sm[k]
            blk["RF64_minus_this_mean_over_seeds_paired"] = {"rf_seeds_used": 5 if ns in (1, 5) else ns, "point": round(float(np.mean([balanced_accuracy_score(y, p) for p in (preds["RF64"] if ns in (1, 5) else rf_same)]) - blk["mean"]), 3),
                                                             "ci95": ci(dm), "share_le_0": round(float((dm <= 0).mean()), 3)}
        out["models"][k] = blk
        print(k, blk["n_seeds"], blk["mean"], "+-", blk["sd"], "seed0", blk["seed0"], "| diff", blk.get("RF64_minus_this_mean_over_seeds_paired"), flush=True)
    yT = np.array(tab["T"]["y"])
    out["set_T"] = {"n": int(len(yT)), "class_counts": pd.Series(yT).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
                    "models": {k: summarize(yT, [np.array(p) for p in v]) for k, v in tab["T"]["preds"].items()}}
    print("T", {k: (v["mean"], v["sd"]) for k, v in out["set_T"]["models"].items()})
    save_res("report_set_A", out)
    print("DONE-REPORT")


if __name__ == "__main__":
    if "--tab" in sys.argv:
        do_tab()
    if "--cnn" in sys.argv:
        do_seq("cnn")
    if "--tf" in sys.argv:
        do_seq("tf")
    if "--report" in sys.argv:
        do_report()
