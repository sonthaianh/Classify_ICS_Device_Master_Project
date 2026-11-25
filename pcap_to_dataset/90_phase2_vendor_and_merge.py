"""Phase 2: thu nghiem (1) them dac trung hang (OUI) cho phan loai vai tro, (2) gop SCADA va DCS vao EWS (2026-11-20).

KHONG ghi de ket qua dong bang. Dung nguyen quy trinh cua script 58 (RF, StratifiedGroupKFold nhom = session dau, 5 hat, balanced accuracy,
tap A = thiet bi co ho so payload). Dau ra: phase2_vendor_merge_exp_v6.json va phase2_type_labels_4class_plus_v6_plus_all_scadaEWS.csv.

Bien the nhan:
  old        122 nhan dong bang
  all        165 nhan (SCADA/DCS dang nam trong HMI)
  scadaEWS   165 nhan, moi nhan co sub_type SCADA* chuyen thanh EWS
Bo dac trung: 49 (payload+vai tro+tac vu ky thuat), 64 (them hanh vi), moi bo them hoac khong them 6 cot hang (nhom OUI).
Chay:  py -3.14 90_phase2_vendor_and_merge.py
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

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
sys.path.insert(0, str(HERE.parent / "utils"))
from oui_lookup import OuiLookup  # noqa: E402

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


S58 = _load("s58", "58_phase2_train4.py")
CLASSES = S58.CLASSES

VM = ["vmware", "virtualbox", "qemu", "xen", "parallels", "hyper-v"]
IND = ["siemens", "rockwell", "allen", "mitsubishi", "schneider", "omron", "abb", "phoenix", "beckhoff", "wago", "hirschmann", "moxa", "yokogawa",
       "honeywell", "emerson", "general electric", "schweitzer", "weidmuller", "pilz", "bernecker", "fuji", "hitachi", "valmet", "metso", "keyence",
       "lenze", "sick", "festo", "pepperl", "turck", "red lion", "prosoft", "advantech", "kuka", "fanuc", "yaskawa", "bosch", "eaton", "danfoss",
       "endress", "krohne", "ifm", "sel", "hms", "anybus", "ge ", "pro-face", "digital electronics"]
NET = ["cisco", "fortinet", "juniper", "netgear", "d-link", "tp-link", "ubiquiti", "aruba", "extreme", "mikrotik", "zyxel", "arista"]
PC = ["intel", "dell", "hewlett", "hp ", "lenovo", "realtek", "broadcom", "apple", "microsoft", "asus", "gigabyte", "micro-star", "acer", "toshiba",
      "samsung", "sony", "liteon", "azurewave", "qualcomm", "nvidia"]
CATS = ["v_vm", "v_industrial", "v_network", "v_pc", "v_local", "v_other"]
O = OuiLookup()
MAC = _orig(F / "device_labels_v6_rule.csv", encoding="utf-8-sig").set_index("device_id").device_mac


def vendor_cat(mac):
    r = O.lookup(mac)
    nm = ((r.get("full_name") or "") + " " + (r.get("short_name") or "")).lower()
    if r.get("is_locally_administered") and not nm.strip():
        return "v_local", ""
    if not nm.strip():
        return "v_other", ""
    for key, lst in (("v_vm", VM), ("v_industrial", IND), ("v_network", NET), ("v_pc", PC)):
        if any(k in nm for k in lst):
            return key, nm.strip()[:40]
    return "v_other", nm.strip()[:40]


def vendor_frame(index):
    rows, names = {}, {}
    for d in index:
        c, nm = vendor_cat(MAC.get(d))
        rows[d] = [1.0 if c == k else 0.0 for k in CATS]
        names[d] = (c, nm)
    return pd.DataFrame.from_dict(rows, orient="index", columns=CATS), names


def label_files():
    allp = F / "phase2_type_labels_4class_plus_v6_plus_all.csv"
    d = _orig(allp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    m = d.copy()
    changed = m.sub_type.str.startswith("SCADA")
    m.loc[changed, "type4"] = "EWS"
    out = F / "phase2_type_labels_4class_plus_v6_plus_all_scadaEWS.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    return {"old": F / OLD, "all": allp, "scadaEWS": out}, int(changed.sum())


def run_variant(name, path):
    CUR["path"] = str(path)
    lab, sets = S58.build()
    sub = lab[lab.has_profile]
    y, g = sub.type4.to_numpy(), sub.group.to_numpy()
    V, names = vendor_frame(sub.index)
    X49 = sets["payload_role_eng"].loc[sub.index]
    X64 = sets["behaviour_payload_role_eng"].loc[sub.index]
    configs = {"49": X49, "49+hang": pd.concat([X49, V], axis=1), "64": X64, "64+hang": pd.concat([X64, V], axis=1), "chi_hang": V}
    out = {"n_tap_A": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(), "ket_qua": {}}
    for k, X in configs.items():
        res, _ = S58.cv(X, y, g, seeds=5)
        out["ket_qua"][k] = {"balanced_acc": res["balanced_acc"], "std": res["balanced_acc_std"], "recall": res["recall"]}
        print(name, k, res["balanced_acc"], res["recall"], flush=True)
    ct = pd.crosstab(pd.Series(y, index=sub.index, name="lop"), pd.Series([names[d][0] for d in sub.index], index=sub.index, name="nhom_hang"))
    out["bang_lop_x_nhom_hang"] = ct.reindex(CLASSES).fillna(0).astype(int).to_dict(orient="index")
    he = sub[sub.type4.isin(["HMI", "EWS"])]
    Vhe, _ = vendor_frame(he.index)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    yh = (he.type4 == "HMI").to_numpy().astype(int)
    if yh.sum() >= 5 and (1 - yh).sum() >= 5:
        aucs = []
        for seed in range(5):
            p = np.zeros(len(yh))
            for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Vhe, yh, he.group.to_numpy()):
                m = RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(Vhe.iloc[tr], yh[tr])
                p[te] = m.predict_proba(Vhe.iloc[te])[:, 1] if len(m.classes_) == 2 else 0.5
            aucs.append(roc_auc_score(yh, p))
        out["HMI_vs_EWS_chi_hang_auc"] = {"n_HMI": int(yh.sum()), "n_EWS": int((1 - yh).sum()), "auc_mean": round(float(np.mean(aucs)), 3)}
        print(name, "HMI vs EWS chi hang AUC", out["HMI_vs_EWS_chi_hang_auc"], flush=True)
    return out, lab


def main():
    files, n_changed = label_files()
    res = {"ghi_chu": "THU NGHIEM BO SUNG - khong ghi de ket qua dong bang; RF, tap A, 5 hat, nhom = session dau", "so_nhan_SCADA_chuyen_sang_EWS": n_changed}
    labs = {}
    for name, p in files.items():
        res[name], labs[name] = run_variant(name, p)
    lab = labs["scadaEWS"]
    sc = lab[lab.sub_type.str.startswith("SCADA")]
    res["SCADA_nhan_co_ho_so_payload"] = {"tong": int(len(sc)), "co_ho_so": int(sc.has_profile.sum())}
    (F / "phase2_vendor_merge_exp_v6.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SCADA labels:", res["SCADA_nhan_co_ho_so_payload"])


if __name__ == "__main__":
    main()
