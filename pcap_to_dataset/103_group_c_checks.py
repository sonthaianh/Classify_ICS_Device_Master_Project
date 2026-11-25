"""Nhom C theo nhan xet cua Thay (2026-11-22). Chi dung du lieu co san; KHONG ghi de ket qua dong bang.

  --e2e     Danh gia dau-cuoi tren DU 186 nhan, cau hinh chinh (RF 64 cot, 5 hat, nguong giai doan 1 = 0,5). Ghi ro trang thai tung thiet bi:
            bi giai doan 1 chan / chua co ho so payload / qua cong nhung sai vai tro / dung. Hai cach xu ly thiet bi chua co ho so:
            U = tinh la chua nhan dung; D = mac dinh du doan OTHER. So sanh "cong that" va "cong ly tuong (oracle)" tren cung quan the.
  --nested  (a) Giai doan 2: chon bo dac trung va min_samples_leaf bang CV long (vong trong 3 fold theo nhom) trong moi fold ngoai, danh gia tren fold ngoai.
            (b) Giai doan 1: chon nguong (luoi 0,2..0,7 buoc 0,05, tieu chi Youden J) tren phan huan luyen cua tap gold 184, danh gia tren phan con lai (lap 20 lan 5 fold).
  --extra   (c) nhay cam nhan DCS (giu HMI / doi PLC / bo) ; (d) ket qua theo muc tin cay va nguon nhan ; (e) tap kiem tra DONG BANG (nhan do tin cay cao,
            chia theo thanh phan lien thong) de so sanh cac phien ban nhan huan luyen 122/165/186 tren CUNG tap kiem tra ;
            (f) doi chung MOT giai doan: RF 64 cot huan luyen thang tren 186 thiet bi (khong cong giai doan 1), cung nhan, cung cach chia.
Dau ra: phase2_e2e_full186.json, phase2_nested_186.json, phase2_groupc_extra_186.json.   Chay: py -3.14 103_group_c_checks.py --e2e --nested --extra
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR
CUR["path"] = str(F / "phase2_type_labels_4class_186_devices.csv")
S61 = M._load("s61", "61_phase2_dl_mlp.py")
S58 = S61.S58
CLASSES = S58.CLASSES
KS = "behaviour_payload_role_eng"
THR = 0.5


def rf_oof(X, y, g, seed, n_est=300, msl=2):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        m = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=msl, random_state=seed, n_jobs=-1).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def e2e():
    lab, sets = S58.build()
    p1 = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    lab = lab.copy()
    lab["p1"] = p1.reindex(lab.index)
    assert lab.p1.notna().all() and len(lab) == 186
    A = lab[lab.has_profile]
    y, g = A.type4.to_numpy(), A.group.to_numpy()
    X = sets[KS].loc[A.index]
    out = {"status": "EXPLORATORY; labels not expert-verified; OTHER labels partly selected from stage-1 false positives; no frozen result overwritten",
           "n_labels": int(len(lab)), "n_profiled": int(len(A)), "n_no_profile": int((~lab.has_profile).sum()), "threshold": THR,
           "no_profile_by_class": lab[~lab.has_profile].type4.value_counts().to_dict(), "no_profile_pass_stage1": int((lab[~lab.has_profile].p1 >= THR).sum())}
    res = {"U": {"oracle": [], "e2e": []}, "D": {"oracle": [], "e2e": []}}
    percls = {k: {"oracle": [], "e2e": []} for k in ("U", "D")}
    ot3 = {k: {"oracle": [], "e2e": []} for k in ("U", "D")}
    stage2_only_profiled = []
    seed0_status = None
    for seed in range(5):
        pred = pd.Series(rf_oof(X, y, g, seed), index=A.index)
        stage2_only_profiled.append(balanced_accuracy_score(y, pred.to_numpy()))
        for scen in ("U", "D"):
            for mode in ("oracle", "e2e"):
                fin = {}
                for d in lab.index:
                    t = lab.type4[d]
                    if not lab.has_profile[d]:
                        p = "OTHER" if (scen == "D" or (mode == "e2e" and lab.p1[d] < THR)) else "UNKNOWN"
                    else:
                        p = pred[d]
                        if mode == "e2e" and lab.p1[d] < THR:
                            p = "OTHER"
                    fin[d] = p
                fin = pd.Series(fin)
                rec = {c: float((fin[lab.type4 == c] == c).mean()) for c in CLASSES}
                res[scen][mode].append(float(np.mean(list(rec.values()))))
                percls[scen][mode].append(rec)
                o = lab.type4.isin(["PLC", "HMI", "EWS"])
                ot3[scen][mode].append(float((fin[o] == lab.type4[o]).mean()))
        if seed == 0:
            st = []
            for d in lab.index:
                t = lab.type4[d]
                if not lab.has_profile[d]:
                    s = "no_profile_blocked" if lab.p1[d] < THR else "no_profile_passed"
                elif lab.p1[d] < THR:
                    s = "blocked_by_stage1"
                else:
                    s = "stage2_correct" if pred[d] == t else "stage2_wrong_role"
                st.append((t, s))
            seed0_status = pd.crosstab(pd.Series([a for a, _ in st], name="true"), pd.Series([b for _, b in st], name="status")).to_dict("index")
    ms = lambda v: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)}
    out["stage2_only_balanced_acc_profiled_165"] = ms(stage2_only_profiled)
    out["scenario_U_missing_profile_counts_as_wrong"] = {m: {"balanced_acc_186": ms(res["U"][m]), "ot3_accuracy_106": ms(ot3["U"][m]),
                                                              "recall_by_class": {c: ms([r[c] for r in percls["U"][m]]) for c in CLASSES}} for m in ("oracle", "e2e")}
    out["scenario_D_missing_profile_defaults_to_OTHER"] = {m: {"balanced_acc_186": ms(res["D"][m]), "ot3_accuracy_106": ms(ot3["D"][m]),
                                                                "recall_by_class": {c: ms([r[c] for r in percls["D"][m]]) for c in CLASSES}} for m in ("oracle", "e2e")}
    out["seed0_status_by_true_class"] = seed0_status
    (F / "phase2_e2e_full186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)


def nested():
    lab, sets = S58.build()
    A = lab[lab.has_profile]
    y, g = A.type4.to_numpy(), A.group.to_numpy()
    cfgs = [(ks, msl) for ks in ("behaviour", "payload_role", "payload_role_eng", "behaviour_payload_role_eng") for msl in (1, 2, 5)]
    outer_scores, chosen, fixed_scores = [], [], []
    for seed in range(5):
        pred = np.empty(len(y), dtype=object)
        for otr, ote in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(A, y, g):
            best, bs = None, -1
            for ks, msl in cfgs:
                X = sets[ks].loc[A.index]
                sc = []
                for itr, ite in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(X.iloc[otr], y[otr], g[otr]):
                    m = RandomForestClassifier(n_estimators=100, class_weight="balanced", min_samples_leaf=msl, random_state=seed, n_jobs=-1).fit(X.iloc[otr].iloc[itr], y[otr][itr])
                    sc.append(balanced_accuracy_score(y[otr][ite], m.predict(X.iloc[otr].iloc[ite])))
                if np.mean(sc) > bs:
                    bs, best = np.mean(sc), (ks, msl)
            chosen.append(f"{best[0]}|leaf{best[1]}")
            X = sets[best[0]].loc[A.index]
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=best[1], random_state=seed, n_jobs=-1).fit(X.iloc[otr], y[otr])
            pred[ote] = m.predict(X.iloc[ote])
        outer_scores.append(balanced_accuracy_score(y, pred))
        print("seed", seed, "nested bal acc", round(outer_scores[-1], 3), flush=True)
    sd = lambda v: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)}
    out = {"stage2_nested": {"n": int(len(A)), "balanced_acc_outer": sd(outer_scores), "per_seed": [round(x, 3) for x in outer_scores],
                             "configs_chosen_counts": pd.Series(chosen).value_counts().to_dict(), "note": "fixed RF 64 columns, leaf 2: 0.641 +/- 0.030 (frozen)"}}
    # stage 1 threshold
    gold = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig")
    p1 = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    gold = gold[gold.device_id.isin(p1.index)].copy()
    gold["y"] = (gold.gold_label == "OT").astype(int)
    gold["s"] = p1.reindex(gold.device_id).to_numpy()
    grid = np.round(np.arange(0.2, 0.7001, 0.05), 2)
    yy, ss = gold.y.to_numpy(), gold.s.to_numpy()

    def metr(yt, st, t):
        pr = st >= t
        rec = float(pr[yt == 1].mean()); fpr = float(pr[yt == 0].mean())
        return rec, fpr, (rec + 1 - fpr) / 2
    ins = {t: metr(yy, ss, t) for t in grid}
    best_in = max(grid, key=lambda t: ins[t][0] - ins[t][1])
    rs, thrs = [], []
    for rep in range(20):
        P, T = np.zeros(len(yy), bool), np.zeros(len(yy), bool)
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=rep).split(ss, yy):
            t = max(grid, key=lambda t: metr(yy[tr], ss[tr], t)[0] - metr(yy[tr], ss[tr], t)[1])
            thrs.append(float(t))
            P[te] = ss[te] >= t
        rec = float(P[yy == 1].mean()); fpr = float(P[yy == 0].mean())
        rs.append((rec, fpr, (rec + 1 - fpr) / 2))
    rs = np.array(rs)
    out["stage1_threshold_nested"] = {"n_gold": int(len(gold)), "n_OT": int(yy.sum()), "grid": [float(x) for x in grid],
                                      "in_sample_best_threshold_youden": float(best_in), "in_sample_at_best": [round(x, 3) for x in ins[best_in]],
                                      "in_sample_at_0.5": [round(x, 3) for x in ins[0.5]],
                                      "nested_recall_fpr_balacc_mean": [round(float(x), 3) for x in rs.mean(0)], "nested_sd": [round(float(x), 3) for x in rs.std(0)],
                                      "chosen_threshold_counts": pd.Series(thrs).value_counts().sort_index().to_dict(), "note": "tuples are (recall OT, false-positive rate, balanced accuracy)"}
    (F / "phase2_nested_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str), flush=True)


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


def e2e_summary(lab, sets, p1):
    A = lab[lab.has_profile]
    y, g = A.type4.to_numpy(), A.group.to_numpy()
    X = sets[KS].loc[A.index]
    st, bal_u, bal_d, ot3, recs = [], [], [], [], []
    for seed in range(5):
        pred = pd.Series(rf_oof(X, y, g, seed), index=A.index)
        st.append(balanced_accuracy_score(y, pred.to_numpy()))
        fu, fd = {}, {}
        for d in lab.index:
            blocked = p1[d] < THR
            if not lab.has_profile[d]:
                fu[d] = "OTHER" if blocked else "UNKNOWN"
                fd[d] = "OTHER"
            else:
                fu[d] = fd[d] = "OTHER" if blocked else pred[d]
        fu, fd = pd.Series(fu), pd.Series(fd)
        rec_u = {c: float((fu[lab.type4 == c] == c).mean()) for c in CLASSES if (lab.type4 == c).any()}
        bal_u.append(float(np.mean(list(rec_u.values()))))
        bal_d.append(float(np.mean([float((fd[lab.type4 == c] == c).mean()) for c in CLASSES if (lab.type4 == c).any()])))
        o = lab.type4.isin(["PLC", "HMI", "EWS"])
        ot3.append(float((fu[o] == lab.type4[o]).mean()))
        recs.append(rec_u)
    m = lambda v: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]
    return {"n_labels": int(len(lab)), "n_profiled": int(len(A)), "stage2_only_profiled_bal_acc": m(st), "e2e_bal_acc_missing_wrong": m(bal_u), "e2e_bal_acc_missing_OTHER": m(bal_d),
            "e2e_ot3_accuracy": m(ot3), "e2e_recall_by_class": {c: round(float(np.mean([r[c] for r in recs])), 3) for c in recs[0]}}


def extra():
    out = {}
    base_csv = F / "phase2_type_labels_4class_186_devices.csv"
    p1 = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    raw = M._orig(base_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    # (c) DCS sensitivity
    var = {"as_is_HMI": raw}
    d2 = raw.copy()
    d2.loc[d2.sub_type == "SCADA_DCS_valmet", "type4"] = "PLC"
    var["DCS_as_PLC"] = d2
    var["DCS_excluded"] = raw[raw.sub_type != "SCADA_DCS_valmet"].copy()
    out["dcs_sensitivity"] = {}
    for name, dfv in var.items():
        pth = F / f"phase2_type_labels_4class_186_devices_dcs_{name}.csv"
        dfv.to_csv(pth, index=False, encoding="utf-8-sig")
        CUR["path"] = str(pth)
        lab, sets = S58.build()
        lab = lab.copy()
        out["dcs_sensitivity"][name] = e2e_summary(lab, sets, p1)
        print("dcs", name, out["dcs_sensitivity"][name], flush=True)
    CUR["path"] = str(base_csv)
    lab, sets = S58.build()
    A = lab[lab.has_profile]
    y, g = A.type4.to_numpy(), A.group.to_numpy()
    X = sets[KS].loc[A.index]
    # (d) by confidence / source (pooled over 5 seeds)
    preds = [pd.Series(rf_oof(X, y, g, seed), index=A.index) for seed in range(5)]
    meta = M._orig(base_csv, encoding="utf-8-sig").set_index("device_id")
    rows = []
    for col in ("confidence", "source"):
        for k, idx in A.groupby(meta.loc[A.index, col]).groups.items():
            acc = [float((p[idx] == A.type4[idx]).mean()) for p in preds]
            rows.append({"by": col, "value": k, "n": int(len(idx)), "accuracy_mean": round(float(np.mean(acc)), 3), "accuracy_sd": round(float(np.std(acc)), 3),
                         "classes": A.type4[idx].value_counts().to_dict()})
    out["accuracy_by_confidence_and_source"] = rows
    print("byconf", rows, flush=True)
    # (e) frozen test set
    win = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    ds = win[win.device_id.isin(A.index)].groupby("device_id").session_id.agg(lambda s: set(s)).to_dict()
    comp = pd.Series(components_of(list(A.index), ds))
    ids122 = set(M._orig(F / "phase2_type_labels_4class_plus_v6.csv", encoding="utf-8-sig").device_id)
    ids165 = set(M._orig(F / "phase2_type_labels_4class_plus_v6_plus_all.csv", encoding="utf-8-sig").device_id)
    ids186 = set(A.index)
    versions = {"122": ids122, "165": ids165, "186": ids186}
    high = meta.loc[A.index, "confidence"] == "high"
    res = {k: [] for k in versions}
    res["186_without_21_reviewed"] = []
    rev = set(meta.index[meta.source.isin(["reviewed_session_evidence", "reviewed_provisional"])])
    sizes = []
    for rep in range(30):
        rng = np.random.default_rng(rep)
        cl = list(comp.unique())
        rng.shuffle(cl)
        test_c, n = set(), 0
        for c in cl:
            test_c.add(c)
            n += int((comp == c).sum())
            if n >= 0.3 * len(A):
                break
        te = [d for d in A.index if comp[d] in test_c and high[d]]
        tr_all = [d for d in A.index if comp[d] not in test_c]
        if len(set(A.type4[te])) < 3:
            continue
        sizes.append(len(te))
        Xte = X.loc[te]
        for k, ids in list(versions.items()) + [("186_without_21_reviewed", ids186 - rev)]:
            tr = [d for d in tr_all if d in ids]
            if len(set(A.type4[tr])) < 4:
                continue
            m_ = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=rep, n_jobs=-1).fit(X.loc[tr], A.type4[tr])
            res[k].append(balanced_accuracy_score(A.type4[te], m_.predict(Xte)))
    out["frozen_test_high_confidence"] = {"n_repeats_used": len(sizes), "mean_test_size": round(float(np.mean(sizes)), 1),
                                          "balanced_acc_by_training_version": {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3), len(v)] for k, v in res.items() if v},
                                          "note": "test = high-confidence devices inside ~30% of connected components; train = devices in the other components that belong to the label version"}
    print("frozen", out["frozen_test_high_confidence"], flush=True)
    # (f) one-stage control
    L = lab.copy()
    X186 = sets[KS].loc[L.index]
    y186, g186 = L.type4.to_numpy(), L.group.to_numpy()
    bal, ot3, recs = [], [], []
    for seed in range(5):
        pred = pd.Series(rf_oof(X186, y186, g186, seed), index=L.index)
        rec = {c: float((pred[L.type4 == c] == c).mean()) for c in CLASSES}
        recs.append(rec)
        bal.append(float(np.mean(list(rec.values()))))
        o = L.type4.isin(["PLC", "HMI", "EWS"])
        ot3.append(float((pred[o] == L.type4[o]).mean()))
    out["one_stage_RF64_on_all_186"] = {"balanced_acc": [round(float(np.mean(bal)), 3), round(float(np.std(bal)), 3)], "ot3_accuracy_106": [round(float(np.mean(ot3)), 3), round(float(np.std(ot3)), 3)],
                                        "recall_by_class": {c: round(float(np.mean([r[c] for r in recs])), 3) for c in CLASSES},
                                        "note": "no stage-1 gate; devices without payload profile keep behaviour columns only; same labels and grouped folds; OTHER holds only 33 stage-1 false positives plus IT/network labels, not all non-OT devices"}
    print("one-stage", out["one_stage_RF64_on_all_186"], flush=True)
    (F / "phase2_groupc_extra_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    if "--extra" in sys.argv:
        extra()
    if "--e2e" in sys.argv:
        e2e()
    if "--nested" in sys.argv:
        nested()
