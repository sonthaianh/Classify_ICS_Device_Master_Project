"""Nhom C (bao cao review toan bo, 2026-11-22): cac phan tich can chay them tren bo 186 nhan. KHONG ghi de ket qua dong bang.

  --role   Doi chung chi dung cot vai tro ket noi (RF 5 hat) va bootstrap ghep cap RF64 - doi chung (trung binh 5 hat).
  --seeds  Du doan tung hat (5 hat) cua RF64, MLP64, CNN 1D tu dau; bootstrap ghep cap theo thiet bi cho hieu so trung binh 5 hat.
  --tf     Transformer tu dau, hat 0-2, du doan tung hat (RAT CHAM, ~20 phut/hat); luu tung hat ra phase2_tf_scratch_perseed_186.npz.
  --lofo   Kiem tra lai nhom 'capture an danh theo gio' trong bo tung nhom nguon (so diem trung nhau, PR-AUC, ROC-AUC).
  --obs    ROC-AUC theo so cua so quan sat tren CUNG tap thiet bi (>= n cua so).
  --eng    Tac vu ky thuat EWS vs HMI (AUC mot chieu, KTC bootstrap) tren bo 186 nhan.
  --depth  Muc doc du lieu: chay voi bien moi truong P2_PROFILE=phase2_payload_profiles_v6.json (3 session) hoac mac dinh (day du); ghi phase2_depth_<tag>.json.
  --twin   Gop cac capture 'sinh doi' (cung so thiet bi, cung khoang thoi gian) thanh mot truoc khi lay thanh phan lien thong; RF64 5 hat.
Dau ra: phase2_groupc_rerun_186.json (gop theo khoa), phase2_depth_*.json, phase2_tf_scratch_perseed_186.npz.
Chay: py -3.14 105_group_c_rerun.py --role --seeds --lofo --obs --eng --twin  |  py -3.14 105_group_c_rerun.py --tf  |  [env P2_PROFILE=...] py -3.14 105_group_c_rerun.py --depth
"""
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

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
C2I = {c: i for i, c in enumerate(CLASSES)}
KS = "behaviour_payload_role_eng"
OUT = F / "phase2_groupc_rerun_186.json"
B = 2000


def save(key, val):
    d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def rf_preds(X, y, g, seed, n_est=300):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        m = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def per_seed_generic(fp, y, g, seeds):
    yi = np.array([C2I[c] for c in y])
    out = []
    for seed in range(seeds):
        pred = np.empty(len(y), dtype=int)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), yi, g):
            pred[te] = fp(tr, te, seed, yi).argmax(1)
        out.append(np.array(CLASSES, dtype=object)[pred])
        print("  seed", seed, round(balanced_accuracy_score(y, out[-1]), 3), flush=True)
    return out


def bal_seeds(y, preds, idx=None):
    if idx is None:
        return float(np.mean([balanced_accuracy_score(y, p) for p in preds]))
    return float(np.mean([balanced_accuracy_score(y[idx], p[idx]) for p in preds]))


def paired_boot(y, pa, pb, seed=20261122):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(bal_seeds(y, pa, i) - bal_seeds(y, pb, i))
    pt = bal_seeds(y, pa) - bal_seeds(y, pb)
    return {"point": round(pt, 3), "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)], "share_le_0": round(float(np.mean(np.array(d) <= 0)), 3)}


def base():
    lab, sets = S58.build()
    A = lab[lab.has_profile]
    return lab, sets, A, A.type4.to_numpy(), A.group.to_numpy()


def do_role():
    lab, sets, A, y, g = base()
    X = sets[KS].loc[A.index]
    cols4 = ["ot_server_share", "hs_client_share", "log_syn", "log_synack"]
    cols7 = cols4 + ["tx_share", "n_clients", "n_servers"]
    miss = [c for c in cols7 if c not in X.columns]
    assert not miss, miss
    p64 = [rf_preds(X, y, g, s) for s in range(5)]
    res = {"RF64": {"bal_acc_mean": round(bal_seeds(y, p64), 3), "per_seed": [round(balanced_accuracy_score(y, p), 3) for p in p64]}}
    for name, cols in (("role4_connection_role_only", cols4), ("role7_plus_tx_share_and_counts", cols7)):
        pp = [rf_preds(X[cols], y, g, s) for s in range(5)]
        rec = {c: round(float(np.mean([(p[y == c] == c).mean() for p in pp])), 3) for c in CLASSES}
        res[name] = {"columns": cols, "bal_acc_mean": round(bal_seeds(y, pp), 3), "per_seed": [round(balanced_accuracy_score(y, p), 3) for p in pp], "recall": rec,
                     "RF64_minus_this_paired_bootstrap": paired_boot(y, p64, pp)}
        print(name, res[name], flush=True)
    save("role_only_control", res)


def do_seeds():
    S62 = M._load("s62", "62_phase2_dl_seq.py")
    lab, sets = S58.build()
    DEV = S62.DEV
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    X = sets[KS].loc[sub.index]
    print("n", len(y), flush=True)
    t0 = time.time()
    preds = {"RF64": [rf_preds(X, y, g, s) for s in range(5)]}
    print("RF done", round(time.time() - t0), flush=True)
    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
    preds["MLP64"] = per_seed_generic(lambda tr, te, seed, yy: S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), y, g, 5)
    print("MLP done", round(time.time() - t0), flush=True)
    preds["CNN1D"] = per_seed_generic(lambda tr, te, seed, yy: S62.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None), y, g, 5)
    print("CNN done", round(time.time() - t0), flush=True)
    np.savez(F / "phase2_perseed_preds_186.npz", devices=np.array(devs), y=y, **{f"{k}_{s}": np.array(v[s], dtype=str) for k, v in preds.items() for s in range(5)})
    res = {"n": int(len(y)), "mean_bal_acc_5_seeds": {k: round(bal_seeds(y, v), 3) for k, v in preds.items()},
           "per_seed": {k: [round(balanced_accuracy_score(y, p), 3) for p in v] for k, v in preds.items()}}
    for k in ("MLP64", "CNN1D"):
        res[f"RF64_minus_{k}_paired_bootstrap_5seed_mean"] = paired_boot(y, preds["RF64"], preds[k])
    print(json.dumps(res, indent=1), flush=True)
    save("per_seed_paired_bootstrap", res)


def do_tf():
    S71 = M._load("s71", "71_phase2_dl_transformer.py")
    lab, sets = S58.build()
    DEV = S71.DEV
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    t0 = time.time()
    allp = {}
    for seed in range(3):
        p = per_seed_generic_one(S71, devs, y, g, seed)
        allp[f"TF_{seed}"] = p
        np.savez(F / "phase2_tf_scratch_perseed_186.npz", devices=np.array(devs), y=y, **{k: np.array(v, dtype=str) for k, v in allp.items()})
        print("TF seed", seed, round(balanced_accuracy_score(y, p), 3), round(time.time() - t0), "s", flush=True)
    print("DONE-TF")


def per_seed_generic_one(S71, devs, y, g, seed):
    yi = np.array([C2I[c] for c in y])
    pred = np.empty(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), yi, g):
        P = S71.ft_predict([devs[i] for i in tr], yi[tr], [devs[i] for i in te], seed, None)
        pred[te] = P.argmax(1)
    return np.array(CLASSES, dtype=object)[pred]


def do_lofo():
    A32 = M._load("a32", "32_fe4_ablation.py")
    B34 = M._load("b34", "34_compare_lightgbm_xgboost.py")
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    src = pd.read_csv(F / "session_source_map_draft.csv", usecols=["session_id", "source_family"])
    x = x.merge(src, on="session_id", how="left")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev["evidence_tier"] = dev.evidence_tier.fillna("")
    res = {}
    for fam in ["eth2dump_attack_lab", "anonymized_hourly", "switch_series_siw_swi", "4sics"]:
        cols = A32.cols_for(A32.D17, "hgb")
        fam_devices = set(x.loc[x.source_family == fam, "device_id"])
        test_ids = set(dev.loc[dev.primary_source_family == fam, "device_id"])
        tr = x[~x.device_id.isin(fam_devices) & (x.cap_rank < A32.K)]
        te = x[x.device_id.isin(test_ids)]
        m = A32.model("hgb").fit(tr[cols], (tr.label == "OT_confirmed").astype(int))
        p = pd.Series(m.predict_proba(te[cols])[:, 1], index=te.index).groupby(te.device_id).mean()
        d = dev[dev.device_id.isin(test_ids)].set_index("device_id").loc[p.index]
        yy = (d.label == "OT_confirmed").to_numpy()
        s = p.loc[d.index].to_numpy()
        r = {"n": int(len(s)), "n_OT": int(yy.sum()), "prevalence": round(float(yy.mean()), 3), "n_unique_scores": int(len(np.unique(np.round(s, 6)))),
             "score_min": round(float(s.min()), 3), "score_median": round(float(np.median(s)), 3), "score_max": round(float(s.max()), 3),
             "share_score_ge_0.5": round(float((s >= 0.5).mean()), 3), "roc_auc": round(float(roc_auc_score(yy, s)), 3), "pr_auc_average_precision": round(float(average_precision_score(yy, s)), 3),
             "mean_score_OT": round(float(s[yy].mean()), 3), "mean_score_nonOT": round(float(s[~yy].mean()), 3)}
        res[fam] = r
        print(fam, r, flush=True)
    save("lofo_recheck", res)


def do_obs():
    T39 = M._load("t39", "39_phase1_scanner_strict_obstime.py")
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    ytrue = dev.set_index("device_id").label == "OT_confirmed"
    gold = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig").set_index("device_id")
    sA, wp = T39.fit_oof(x, dev, "fold_device", window_probs=True)
    wp = wp.sort_values(["device_id", "window_start"])
    wp["rank"] = wp.groupby("device_id").cumcount()
    cnt_all = wp.groupby("device_id").size()
    res = {}
    for need in (2, 3, 5, 10):
        ids = cnt_all[cnt_all >= need].index
        gids = [d for d in ids if d in gold.index]
        yg = (gold.loc[gids].gold_label == "OT").to_numpy()
        row = {"n_devices": int(len(ids)), "n_gold": int(len(gids)), "pop_roc": {}, "gold_roc": {}}
        for n in sorted({1, 2, 3, 5, 10} & set(range(1, need + 1)) | {need}):
            sub = wp[(wp["rank"] < n) & wp.device_id.isin(ids)].groupby("device_id").p.mean()
            row["pop_roc"][str(n)] = round(float(roc_auc_score(ytrue.loc[sub.index], sub)), 4)
            if 0 < yg.sum() < len(yg):
                row["gold_roc"][str(n)] = round(float(roc_auc_score(yg, sub.loc[gids])), 3)
        res[f"devices_with_at_least_{need}_windows"] = row
        print(need, row, flush=True)
    save("obs_time_same_devices", res)


def do_eng():
    E57 = M._load("e57", "57_phase2_eng_features.py")
    X = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    lab = M._orig(F / "phase2_type_labels_4class_186_devices.csv", encoding="utf-8-sig").set_index("device_id")
    both = lab[lab.type4.isin(["EWS", "HMI"])].join(X, how="left")
    prof = both[both.has_tokens.fillna(0).astype(int) == 1].copy()
    prof["owner_approved"] = prof.source == "owner_approved_session_evidence"
    FE = ["s7_prog_n", "s7_userdata_n", "mb_diag_n", "fins_ctrl_n", "enip_disc_n", "eng_task_n", "eng_task_share", "c_tx_distinct", "n_client_fams", "log_servers", "tx_share"]
    res = {"coverage": {"EWS_labelled": int((lab.type4 == "EWS").sum()), "HMI_labelled": int((lab.type4 == "HMI").sum()),
                        "EWS_with_tokens": int((prof.type4 == "EWS").sum()), "HMI_with_tokens": int((prof.type4 == "HMI").sum()),
                        "EWS_not_owner_approved": int(((prof.type4 == "EWS") & ~prof.owner_approved).sum())}}

    def evaluate(sub):
        y = (sub.type4 == "EWS").astype(int).values
        out = {"n_EWS": int(y.sum()), "n_HMI": int((1 - y).sum()), "auc": {}}
        for f in FE:
            out["auc"][f] = E57.auc_ci(y, sub[f].astype(float).fillna(0).values)
        best = max(out["auc"].items(), key=lambda kv: kv[1][0] if isinstance(kv[1], (list, tuple)) else kv[1]["auc"] if isinstance(kv[1], dict) else 0)
        out["max_auc_feature"] = best[0]
        return out
    res["EWS_not_owner_vs_HMI"] = evaluate(prof[~prof.owner_approved])
    res["EWS_all_vs_HMI"] = evaluate(prof)
    res["EWS_identity_only_vs_HMI"] = evaluate(prof[~prof.owner_approved & ~prof.source.str.startswith("reviewed_")])
    res["coverage"]["EWS_identity_only"] = int(((prof.type4 == "EWS") & ~prof.owner_approved & ~prof.source.str.startswith("reviewed_")).sum())
    res["within_S7_family_only"] = evaluate(prof[prof.dominant_family.isin(["s7_iso"])])
    for k in ("EWS_not_owner_vs_HMI", "EWS_identity_only_vs_HMI", "EWS_all_vs_HMI", "within_S7_family_only"):
        print(k, res[k]["n_EWS"], res[k]["n_HMI"], res[k]["max_auc_feature"], res[k]["auc"][res[k]["max_auc_feature"]], flush=True)
    save("eng_tasks_186", res)


def do_depth():
    tag = "3session" if "v6.json" in os.environ.get("P2_PROFILE", "") and "full" not in os.environ.get("P2_PROFILE", "") else "full"
    lab, sets, A, y, g = base()
    X = sets[KS].loc[A.index]
    p = [rf_preds(X, y, g, s) for s in range(5)]
    sd = [balanced_accuracy_score(y, q) for q in p]
    res = {"tag": tag, "profile_file": os.environ.get("P2_PROFILE"), "n_profiled": int(len(A)), "class_counts": pd.Series(y).value_counts().to_dict(),
           "bal_acc_mean": round(float(np.mean(sd)), 3), "bal_acc_sd": round(float(np.std(sd)), 3), "device_ids": list(A.index)}
    other = F / ("phase2_depth_3session.json" if tag == "full" else "phase2_depth_full.json")
    if other.exists():
        oth = json.loads(other.read_text(encoding="utf-8"))
        common = sorted(set(A.index) & set(oth["device_ids"]))
        A2 = A.loc[common]
        X2 = sets[KS].loc[common]
        y2, g2 = A2.type4.to_numpy(), A2.group.to_numpy()
        q = [rf_preds(X2, y2, g2, s) for s in range(5)]
        sd2 = [balanced_accuracy_score(y2, z) for z in q]
        res["on_common_devices"] = {"n": len(common), "bal_acc_mean": round(float(np.mean(sd2)), 3), "bal_acc_sd": round(float(np.std(sd2)), 3)}
    print({k: v for k, v in res.items() if k != "device_ids"}, flush=True)
    (F / f"phase2_depth_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def components_of(devs, ds, canon):
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for d in devs:
        parent[("d", d)] = ("d", d)
        for s_ in ds[d]:
            s2 = canon.get(s_, s_)
            parent.setdefault(("s", s2), ("s", s2))
            ra, rb = find(("d", d)), find(("s", s2))
            if ra != rb:
                parent[ra] = rb
    return {d: str(find(("d", d))) for d in devs}


def do_twin():
    lab, sets, A, y, g = base()
    X = sets[KS].loc[A.index]
    win = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    ds = win[win.device_id.isin(A.index)].groupby("device_id").session_id.agg(lambda s: set(s)).to_dict()
    rv = pd.read_csv(F / "device_session_review_v6.csv", encoding="utf-8-sig", usecols=["device_id", "session_id", "first_observed", "last_observed"])
    cm = rv.groupby("session_id").agg(n_devices=("device_id", "nunique"), first_observed=("first_observed", "min"), last_observed=("last_observed", "max")).reset_index()
    cm["sig"] = cm[["n_devices", "first_observed", "last_observed"]].astype(str).agg("|".join, axis=1)
    grp = cm.groupby("sig").session_id.apply(list)
    canon, twin_sessions = {}, set()
    for lst in grp:
        if len(lst) > 1:
            for s_ in lst:
                canon[s_] = lst[0]
                twin_sessions.add(s_)
    affected = [d for d in A.index if ds[d] & twin_sessions]
    res = {"n_twin_groups": int(sum(1 for l in grp if len(l) > 1)), "n_twin_sessions": len(twin_sessions), "devices_in_A_with_twin_capture": len(affected),
           "affected_by_class": pd.Series(A.type4.loc[affected]).value_counts().to_dict()}
    comp_plain = pd.Series(components_of(list(A.index), ds, {}))
    comp_twin = pd.Series(components_of(list(A.index), ds, canon))
    res["n_components_plain"] = int(comp_plain.nunique())
    res["n_components_with_twin_merge"] = int(comp_twin.nunique())
    for name, comp in (("component_plain", comp_plain), ("component_twin_merged", comp_twin)):
        p = [rf_preds(X, y, comp.loc[A.index].to_numpy(), s) for s in range(5)]
        sd = [balanced_accuracy_score(y, q) for q in p]
        res[f"RF64_group_{name}"] = {"bal_acc_mean": round(float(np.mean(sd)), 3), "bal_acc_sd": round(float(np.std(sd)), 3)}
    keep = [d for d in A.index if d not in set(affected)]
    A2 = A.loc[keep]
    p = [rf_preds(sets[KS].loc[keep], A2.type4.to_numpy(), A2.group.to_numpy(), s) for s in range(5)]
    sd = [balanced_accuracy_score(A2.type4.to_numpy(), q) for q in p]
    res["RF64_first_capture_group_without_affected_devices"] = {"n": len(keep), "bal_acc_mean": round(float(np.mean(sd)), 3), "bal_acc_sd": round(float(np.std(sd)), 3)}
    print(res, flush=True)
    save("twin_captures", res)


def do_comp():
    lab, sets, A, y, g = base()
    X = sets[KS].loc[A.index]
    win = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    ds = win[win.device_id.isin(A.index)].groupby("device_id").session_id.agg(lambda s: set(s)).to_dict()
    comp = pd.Series(components_of(list(A.index), ds, {}))
    ids = sorted(comp.unique())
    res = {"n_components": len(ids), "relabelings": []}
    for r in range(12):
        rng = np.random.default_rng(1000 + r)
        perm = dict(zip(ids, rng.permutation(len(ids))))
        gg = comp.loc[A.index].map(perm).to_numpy()
        p = [rf_preds(X, y, gg, s) for s in range(5)]
        res["relabelings"].append(round(bal_seeds(y, p), 3))
        print("relabel", r, res["relabelings"][-1], flush=True)
    v = np.array(res["relabelings"])
    res["mean"], res["sd"], res["min"], res["max"] = round(float(v.mean()), 3), round(float(v.std()), 3), round(float(v.min()), 3), round(float(v.max()), 3)
    save("component_split_relabel_robustness", res)


if __name__ == "__main__":
    for flag, fn in (("--role", do_role), ("--seeds", do_seeds), ("--lofo", do_lofo), ("--obs", do_obs), ("--eng", do_eng), ("--twin", do_twin), ("--comp", do_comp), ("--depth", do_depth), ("--tf", do_tf)):
        if flag in sys.argv:
            fn()
