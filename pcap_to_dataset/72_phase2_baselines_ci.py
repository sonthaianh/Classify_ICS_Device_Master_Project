"""Luan van, Chuong 4: co so doi chieu (baseline), khoang tin cay bootstrap va so sanh ghep cap (2026-11-11/12). KHONG doi mo hinh/nhan.

(1) Giai doan 1: luat gan nhan theo giao thuc/cong (device_labels_v6_rule.csv) cham tren gold hop nhat (gold_union_v6.csv) - baseline "luat theo cong/giao thuc".
(2) Giai doan 2 (tap A = 117 thiet bi co ho so): baseline da so (balanced accuracy = 0,25) va baseline LUAT VAI TRO co dinh truoc (final_role tu bat tay TCP + luat so huu cong:
    server -> PLC, client -> HMI, con lai -> OTHER; khong chinh theo ket qua).
(3) Du doan out-of-fold HAT 0 cua RF 64 cot, MLP 64 cot, CNN 1D tu dau (va Transformer tu dau neu da chay, doc tu phase2_dl_transformer_part_A_scratch.json); cung fold (random_state = 0)
    nen so sanh ghep cap hop le. Bootstrap theo thiet bi (B = 2000) cho balanced accuracy (khoang tin cay 95%) va cho chenh lech RF - mo hinh khac.
Dau ra: phase2_baselines_ci_v6.json, phase2_oof_seed0_v6.csv.    Chay: P2_PROFILE=phase2_payload_profiles_v6_full.json P2_SEQ=phase2_sequences_v6_full.pkl py -3.14 72_phase2_baselines_ci.py
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("P2_PROFILE", "phase2_payload_profiles_v6_full.json")
os.environ.setdefault("P2_SEQ", "phase2_sequences_v6_full.pkl")
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
B = 2000
SEED = 20261111


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S62 = _load("s62", "62_phase2_dl_seq.py")
S61, S58 = S62.S61, S62.S58
F, CLASSES = S58.F, S58.CLASSES


def phase1_baseline():
    """Baseline luat giao thuc/cong tren gold hop nhat. Luu y: 47 thiet bi non-OT 'capture mau' duoc CHON la thiet bi ma luat khong gan OT (rule-negative theo cach xay),
    nen so sanh cong bang hon la bo nguon nay (gold con 65 OT / 72 non-OT)."""
    from sklearn.metrics import roc_auc_score
    g = pd.read_csv(F / "gold_union_v6.csv", encoding="utf-8-sig").set_index("device_id")
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id")
    oof = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id")
    g["y"] = (g.gold_label == "OT").astype(int)
    g["rule"] = (lab.loc[g.index, "label"] == "OT_confirmed").astype(int)
    g["score"] = oof.loc[g.index, "lightgbm"]
    g["model"] = (g.score >= 0.5).astype(int)

    def rep(df):
        ot, no = df[df.y == 1], df[df.y == 0]
        out = {"n_ot": int(len(ot)), "n_nonot": int(len(no)), "model_roc": round(float(roc_auc_score(df.y, df.score)), 3)}
        for c, nm in (("rule", "luat_giao_thuc_cong"), ("model", "lightgbm_0.5")):
            rec, fpr = float(ot[c].mean()), float(no[c].mean())
            out[nm] = {"ot_recall": round(rec, 3), "nonot_flagged": round(fpr, 3), "balanced_acc": round((rec + 1 - fpr) / 2, 3)}
        return out
    res = {"gold_184_all_sources": rep(g)}
    f = g[g.source != "capture_scenario_provisional"]
    res["gold_137_excl_capture_scenario"] = rep(f)
    miss = f[(f.y == 1) & (f.rule == 0)]
    no_neg = f[(f.y == 0) & (f.rule == 0)]
    no_pos = f[(f.y == 0) & (f.rule == 1)]
    res["added_value_of_model_over_rule"] = {"gold_ot_missed_by_rule": int(len(miss)), "gold_ot_missed_by_rule_model_flags_ot": int(miss.model.sum()),
                                             "gold_nonot_rule_negative": int(len(no_neg)), "gold_nonot_rule_negative_model_flags_ot": int(no_neg.model.sum()),
                                             "gold_nonot_rule_positive": int(len(no_pos)), "model_flags_among_rule_positive_nonot": int(no_pos.model.sum())}
    cs = g[g.source == "capture_scenario_provisional"]
    res["capture_scenario_nonot_47"] = {"rule_flags": int(cs.rule.sum()), "model_flags": int(cs.model.sum()), "n": int(len(cs))}
    return res


def boot_ci(y, preds, rng):
    """preds: dict name -> array. Tra ve CI 95% cua balanced accuracy cho tung mo hinh va chenh lech RF - mo hinh khac (cung mau bootstrap)."""
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    sc = {k: [] for k in preds}
    for i in idx:
        for k, p in preds.items():
            sc[k].append(balanced_accuracy_score(y[i], p[i]))
    res = {k: {"point": round(float(balanced_accuracy_score(y, p)), 3), "ci95": [round(float(np.percentile(sc[k], 2.5)), 3), round(float(np.percentile(sc[k], 97.5)), 3)]} for k, p in preds.items()}
    diffs = {}
    if "RF_64" in preds:
        for k in preds:
            if k == "RF_64":
                continue
            d = np.array(sc["RF_64"]) - np.array(sc[k])
            diffs[f"RF_64 - {k}"] = {"mean_diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                                       "share_boot_diff_le_0": round(float((d <= 0).mean()), 3)}
    return res, diffs


def main():
    if "--phase1-only" in sys.argv:
        pth = F / "phase2_baselines_ci_v6.json"
        out = json.loads(pth.read_text(encoding="utf-8"))
        out["phase1_baseline_on_union_gold"] = phase1_baseline()
        pth.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out["phase1_baseline_on_union_gold"], indent=1, ensure_ascii=False))
        return 0
    out = {"status": "EXPLORATORY; labels not expert-verified; seed-0 out-of-fold predictions, same folds for all models", "bootstrap_B": B}
    out["phase1_baseline_on_union_gold"] = phase1_baseline()

    lab, sets = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(S62.DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)
    assert len(devs) == 117
    X = sets["behaviour_payload_role_eng"].loc[sub.index]
    preds = {}
    res, oof = S58.cv(X, y, g, seeds=1, n_est=300)
    preds["RF_64"] = np.array(oof, dtype=object)
    print("RF seed0", res["balanced_acc"], flush=True)
    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)

    def fp_mlp(tr, te, seed, yy):
        return S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed)
    res, oof = S61.cv_generic(fp_mlp, Xn, y, g, seeds=1)
    preds["MLP_64"] = np.array(oof, dtype=object)
    print("MLP seed0", res["balanced_acc"], flush=True)

    def fp_cnn(tr, te, seed, yy):
        return S62.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None)
    res, oof = S61.cv_generic(fp_cnn, None, y, g, seeds=1)
    preds["CNN_scratch"] = np.array(oof, dtype=object)
    print("CNN seed0", res["balanced_acc"], flush=True)

    tp = F / "phase2_dl_transformer_part_A_scratch.json"
    if tp.exists():
        d = json.loads(tp.read_text(encoding="utf-8"))["oof_seed0"]
        assert d["device_id"] == devs and d["true"] == y.tolist(), "Transformer part file uses different device order"
        preds["Transformer_scratch"] = np.array(d["pred"], dtype=object)
    # baselines
    role = pd.read_csv(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    fr = role.reindex(sub.index).final_role.fillna("unknown")
    preds["Luat_vai_tro"] = np.array(np.where(fr == "server", "PLC", np.where(fr == "client", "HMI", "OTHER")), dtype=object)
    preds["Da_so"] = np.array(["OTHER"] * len(y), dtype=object)

    out["n"] = len(y)
    out["class_counts"] = pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
    out["per_model_recall_seed0"] = {k: dict(zip(CLASSES, np.round(recall_score(y, p, labels=CLASSES, average=None, zero_division=0), 3).tolist())) for k, p in preds.items()}
    rng = np.random.default_rng(SEED)
    res, diffs = boot_ci(y, {k: np.asarray(v) for k, v in preds.items()}, rng)
    out["balanced_acc_seed0_with_bootstrap_ci"] = res
    out["paired_bootstrap_rf_minus_other"] = diffs
    # CI cho recall tung lop cua RF_64
    rec_ci = {}
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    for c in CLASSES:
        vals = []
        for i in idx:
            yy, pp = y[i], preds["RF_64"][i]
            m = yy == c
            if m.sum():
                vals.append(float((pp[m] == c).mean()))
        rec_ci[c] = [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
    out["rf64_recall_bootstrap_ci95"] = rec_ci
    pd.DataFrame({"device_id": devs, "true": y, **{k: v for k, v in preds.items()}}).to_csv(F / "phase2_oof_seed0_v6.csv", index=False, encoding="utf-8-sig")
    (F / "phase2_baselines_ci_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False)[:3500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
