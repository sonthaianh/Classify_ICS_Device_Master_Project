"""Phase 2: huan luyen lai tren bo nhan MOI (165 nhan = 122 cu + 6 tu 4SICS + 37 tu bang tai san/4SICS) (2026-11-12). KHONG ghi de ket qua dong bang.

Cach lam: dung nguyen ham cua cac script 58 (RF), 61 (MLP), 62/71 (chuoi su kien) nhung doi duong dan file nhan bang cach chuyen huong pd.read_csv
(khong sua script nao). Dau ra nam trong file RIENG: phase2_retrain_plus_all_*.json.
Bien the nhan:
  old    122 nhan cu (tai hien ket qua dong bang de kiem tra)
  all    165 nhan (them 37)
  noDCS  165 nhan, bo 9 bo dieu khien DCS Valmet (sub_type SCADA_DCS_valmet)
  noSCADA 165 nhan, bo moi nhan SCADA (9 DCS + 5 may chu + 3 tram van hanh)
Che do (cung giao thuc dong bang): A = thiet bi co ho so payload; T = thiet bi co token OT. StratifiedGroupKFold (nhom = session dau), balanced accuracy.

  py -3.14 80_phase2_retrain_plus_all.py --tab                       # RF + MLP, 4 bien the nhan, 5 hat (nhanh)
  py -3.14 80_phase2_retrain_plus_all.py --tf --labels all --variant pretrained --seeds 3   # Transformer (cham: ~2 phut/fold)
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
from sklearn.metrics import balanced_accuracy_score, recall_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
OLD = "phase2_type_labels_4class_plus_v6.csv"
CUR = {"path": None}
_orig = pd.read_csv


def _patched(p, *a, **k):
    if CUR["path"] and str(p).endswith(OLD):
        p = CUR["path"]
    return _orig(p, *a, **k)


pd.read_csv = _patched


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def label_file(name):
    if name == "old":
        return F / OLD
    d = pd.read_csv(F / "phase2_type_labels_4class_plus_v6_plus_all.csv", dtype=str, keep_default_na=False)
    if name == "all":
        return F / "phase2_type_labels_4class_plus_v6_plus_all.csv"
    if name == "noDCS":
        d = d[d.sub_type != "SCADA_DCS_valmet"]
    elif name == "noSCADA":
        d = d[~d.sub_type.str.startswith("SCADA")]
    else:
        raise SystemExit(f"bien the nhan khong biet: {name}")
    out = F / f"phase2_type_labels_4class_plus_v6_plus_all_{name}.csv"
    d.to_csv(out, index=False)
    return out


def subset_report(sub, oof, old_ids):
    """oof: mang nhan du doan hat 0 cung thu tu sub. Bao cao theo tap cu / tap moi them."""
    y = sub.type4.to_numpy()
    res = {}
    for nm, m in (("old_labels", sub.index.isin(old_ids)), ("new_added", ~sub.index.isin(old_ids))):
        if m.sum() == 0:
            continue
        ys, ps = y[m], np.asarray(oof)[m]
        res[nm] = {"n": int(m.sum()), "accuracy": round(float((ys == ps).mean()), 3), "balanced_acc_present_classes": round(float(balanced_accuracy_score(ys, ps)), 3),
                   "recall_by_class": {c: round(float((ps[ys == c] == c).mean()), 3) for c in ["PLC", "HMI", "EWS", "OTHER"] if (ys == c).any()}}
    if "sub_type" in sub:
        st = {}
        for s_, g in pd.Series(np.asarray(oof) == y, index=sub.index).groupby(sub.sub_type):
            if str(s_).startswith("SCADA") or str(s_) in ("gateway_or_field_device",):
                st[str(s_)] = {"n": int(len(g)), "accuracy": round(float(g.mean()), 3)}
        res["accuracy_by_new_subtype"] = st
    return res


def tab():
    S61 = _load("s61", "61_phase2_dl_mlp.py")
    S58 = S61.S58
    old_ids = set(_orig(F / OLD, encoding="utf-8-sig").device_id)
    out = {"status": "EXPLORATORY retrain; frozen results untouched", "chance_balanced_acc": 0.25, "variants": {}}
    for name in ("old", "all", "noDCS", "noSCADA"):
        CUR["path"] = label_file(name)
        lab, sets = S58.build()
        eng = _orig(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
        tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
        blk = {"n_labels": int(len(lab)), "class_counts": lab.type4.value_counts().to_dict(), "n_profiled": int(lab.has_profile.sum()), "n_tokens": int(tok.sum())}
        for rname, mask in (("A_profiled", lab.has_profile), ("T_tokens", tok)):
            sub = lab[mask]
            y, g = sub.type4.to_numpy(), sub.group.to_numpy()
            r = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(S58.CLASSES).fillna(0).astype(int).to_dict()}
            for ks in ("behaviour_payload_role_eng", "payload_role_eng"):
                if ks == "behaviour_payload_role_eng" or rname == "A_profiled":
                    X = sets[ks].loc[sub.index]
                    res, oof = S58.cv(X, y, g)
                    r[f"RF|{ks}"] = res
                    if ks == "behaviour_payload_role_eng":
                        r["RF_subsets_seed0"] = subset_report(sub, oof, old_ids)
                    Xn = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
                    mres, moof = S61.cv_generic(lambda tr, te, seed, yy, Xn=Xn: S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), Xn, y, g)
                    r[f"MLP|{ks}"] = mres
            blk[rname] = r
            print(name, rname, "n=", len(y), {k: v["balanced_acc"] for k, v in r.items() if isinstance(v, dict) and "balanced_acc" in v}, flush=True)
        out["variants"][name] = blk
    (F / "phase2_retrain_plus_all_tabular.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-TAB")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def tf():
    name, variant, seeds = arg("--labels", "all"), arg("--variant", "pretrained"), int(arg("--seeds", "3"))
    S71 = _load("s71", "71_phase2_dl_transformer.py")
    S58, S61, DEV = S71.S58, S71.S61, S71.DEV
    CUR["path"] = label_file(name)
    old_ids = set(_orig(F / OLD, encoding="utf-8-sig").device_id)
    t0 = time.time()
    pre = None
    if variant == "pretrained":
        pre, macc, mbase = S71.pretrain()
        print(f"pretrain {time.time() - t0:.0f}s masked acc {macc:.3f} vs {mbase:.3f}", flush=True)
    lab, _ = S58.build()
    sub = lab[lab.has_profile & lab.index.isin(DEV)]
    y, g, devs = sub.type4.to_numpy(), sub.group.to_numpy(), list(sub.index)

    def fp(tr, te, seed, yy):
        return S71.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, pre)
    res, oof0 = S61.cv_generic(fp, None, y, g, seeds=seeds)
    out = {"labels": name, "variant": variant, "seeds": seeds, "n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(S58.CLASSES).fillna(0).astype(int).to_dict(),
           "result": res, "subsets_seed0": subset_report(sub, oof0, old_ids), "seconds": round(time.time() - t0)}
    (F / f"phase2_retrain_plus_all_tf_{name}_{variant}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TF", name, variant, res["balanced_acc"], "+-", res["balanced_acc_std"], res["recall"], f"{time.time() - t0:.0f}s", flush=True)
    print("DONE-TF")


if __name__ == "__main__":
    if "--tab" in sys.argv:
        tab()
    elif "--tf" in sys.argv:
        tf()
    else:
        print(__doc__)
