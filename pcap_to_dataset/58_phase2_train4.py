"""Phase 2 - huan luyen THU mo hinh cay 4 lop PLC / HMI / EWS / OTHER (2026-11-10). THAM DO, khong phai ket qua cuoi.

Nhan  : phase2_type_labels_4class_plus_v6.csv (122 nhan; EWS co 5 nhan duyet tay, OTHER co 33 nhan gold non-OT bi Phase 1 gan nham).
Feature: behaviour (15 trung vi Phase 1) | payload+vai tro (script 49) | tac vu ky thuat (script 57).  KHONG dung chuoi danh tinh, KHONG dung co has_profile.
Cach danh gia (de tranh anh huong "co hay khong co payload"):
  A) tap CO ho so payload (n~65): tat ca bo feature, KHONG co confound do phu.   <- ket qua chinh
  B) toan bo 122 chi voi feature hanh vi (khong thieu du lieu).
  C) toan bo 122 voi payload (thieu -> 0): CHI de CHO THAY confound (OTHER thieu payload nhieu) - khong dung de ket luan.
  Moi bien the chay them che do "nhan nghiem": bo 5 EWS duyet tay (chon tu ten session = khong doc lap voi feature tac vu).
CV   : StratifiedGroupKFold (nhom = session dau cua thiet bi), 5 hat x 5 fold; RandomForest class_weight=balanced.
Kiem tra: hoan vi nhan (100 lan) cho bien the tot nhat de lay p-value so voi ngau nhien.
Dau ra: phase2_train4_v6.json  (+ phase2_train4_preds_v6.csv: du doan out-of-fold hat 0)
Chay: py -3.14 pcap_to_dataset/58_phase2_train4.py
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S43 = _load("s43", "43_phase2_similarity_labels.py")
B = _load("cmp", "34_compare_lightgbm_xgboost.py")
F = S43.F
CLASSES = ["PLC", "HMI", "EWS", "OTHER"]
ENG = ["s7_prog_n", "s7_userdata_n", "mb_diag_n", "fins_ctrl_n", "enip_disc_n", "eng_task_n", "eng_task_share", "c_tx_distinct", "n_client_fams", "log_servers"]


def build():
    lab = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", encoding="utf-8-sig").set_index("device_id")
    agg = S43.load_profiles()
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"] + B.FEATS)
    beh = x.groupby("device_id")[B.FEATS].median()
    first = x.sort_values("session_id").groupby("device_id").session_id.first()
    lab = lab[lab.index.isin(beh.index)].copy()
    role = pd.read_csv(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    cells = [f"{f}|{r}" for f in S43.FAMS for r in ("S", "C")]
    nn = lambda z: 0.0 if z is None or (isinstance(z, float) and np.isnan(z)) else float(np.log1p(z))
    pay, has = {}, {}
    for d in lab.index:
        a = agg.get(d)
        has[d] = a is not None
        if a is None:
            pay[d] = [0.0] * (len(cells) + 7)
            continue
        tot = max(sum(a["fam_role"].values()), 1)
        v = [a["fam_role"].get(c, 0) / tot for c in cells]
        v += [nn(a["tx_med"]), nn(a["tx_p90"]), nn(a["rx_med"]), nn(a["rx_p90"]), a["tx_share"], np.log1p(a["n_clients"]), np.log1p(a["n_servers"])]
        pay[d] = v
    P = pd.DataFrame.from_dict(pay, orient="index", columns=cells + ["tx_med", "tx_p90", "rx_med", "rx_p90", "tx_share", "n_clients", "n_servers"])
    rs = role.ot_server_share_port if "ot_server_share_port" in role else role.port_rule_server_share
    R = pd.DataFrame({"ot_server_share": rs, "hs_client_share": role.ot_client_share_handshake, "log_syn": np.log1p(role.ot_syn_sent), "log_synack": np.log1p(role.ot_synack_sent)}).reindex(lab.index).fillna(-1)
    E = eng.reindex(lab.index)[ENG].fillna(0)
    sets = {"behaviour": beh.loc[lab.index].fillna(-1), "payload_role": pd.concat([P, R], axis=1),
            "payload_role_eng": pd.concat([P, R, E], axis=1), "behaviour_payload_role_eng": pd.concat([beh.loc[lab.index].fillna(-1), P, R, E], axis=1)}
    lab["has_profile"] = pd.Series(has)
    lab["group"] = first.loc[lab.index].values
    lab["owner_approved"] = lab.source == "owner_approved_session_evidence"
    return lab, sets


def cv(X, y, g, seeds=5, n_est=300, shuffle_y=None):
    bal, f1s, rec, cm = [], [], [], np.zeros((4, 4))
    oof0 = None
    for seed in range(seeds):
        pred = np.empty(len(y), dtype=object)
        yy = y if shuffle_y is None else shuffle_y
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, yy, g):
            m = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(X.iloc[tr], yy[tr])
            pred[te] = m.predict(X.iloc[te])
        bal.append(balanced_accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro"))
        rec.append(recall_score(y, pred, labels=CLASSES, average=None, zero_division=0))
        cm += confusion_matrix(y, pred, labels=CLASSES)
        if seed == 0:
            oof0 = pred.copy()
    return {"balanced_acc": round(float(np.mean(bal)), 3), "balanced_acc_std": round(float(np.std(bal)), 3), "macro_f1": round(float(np.mean(f1s)), 3),
            "recall": dict(zip(CLASSES, np.round(np.mean(rec, axis=0), 3).tolist())), "confusion_sum_over_seeds": cm.astype(int).tolist()}, oof0


def main():
    lab, sets = build()
    out = {"status": "EXPLORATORY - small n, labels partly owner-reviewed, not expert verified", "chance_balanced_acc": 0.25, "classes": CLASSES}
    preds = lab[["type4"]].copy()
    regimes = {"A_profiled_only": lab.has_profile, "B_all_behaviour_only": pd.Series(True, index=lab.index), "C_all_with_payload_CONFOUNDED": pd.Series(True, index=lab.index)}
    for strict in (False, True):
        tag = "strict" if strict else "all_labels"
        for rname, mask in regimes.items():
            sub = lab[mask & (~lab.owner_approved if strict else True)]
            y, g = sub.type4.to_numpy(), sub.group.to_numpy()
            keysets = ["behaviour"] if rname.startswith("B") else (["payload_role", "payload_role_eng", "behaviour_payload_role_eng"] if rname.startswith("A") else ["payload_role", "payload_role_eng"])
            if rname.startswith("A"):
                keysets = ["behaviour"] + keysets
            block = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
                     "majority_baseline_acc": round(float(pd.Series(y).value_counts().max() / len(y)), 3)}
            for ks in keysets:
                res, oof = cv(sets[ks].loc[sub.index], y, g)
                block[ks] = res
                if (not strict) and rname.startswith("A") and ks == "payload_role_eng":
                    preds.loc[sub.index, "pred_A_payload_role_eng"] = oof
            out.setdefault(tag, {})[rname] = block
            print(tag, rname, {k: (v["balanced_acc"] if isinstance(v, dict) and "balanced_acc" in v else v) for k, v in block.items() if k not in ("class_counts",)}, flush=True)

    # hoan vi nhan cho bien the A / payload_role_eng / all_labels
    sub = lab[lab.has_profile]
    y, g = sub.type4.to_numpy(), sub.group.to_numpy()
    obs = out["all_labels"]["A_profiled_only"]["payload_role_eng"]["balanced_acc"]
    rng = np.random.default_rng(20261110)
    null = []
    for i in range(100):
        ys = rng.permutation(y)
        res, _ = cv(sets["payload_role_eng"].loc[sub.index], y, g, seeds=1, n_est=100, shuffle_y=ys)
        null.append(res["balanced_acc"])
    out["permutation_test_A_payload_role_eng"] = {"observed": obs, "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                                                  "p_value": round(float((1 + sum(n >= obs for n in null)) / (1 + len(null))), 3), "n_perm": 100}
    # tam quan trong feature (RandomForest day du, tap A)
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2, random_state=0, n_jobs=-1).fit(sets["payload_role_eng"].loc[sub.index], y)
    imp = pd.Series(m.feature_importances_, index=sets["payload_role_eng"].columns).sort_values(ascending=False)
    out["top_features_A_payload_role_eng_(fit on all A, descriptive only)"] = {k: round(float(v), 3) for k, v in imp.head(12).items()}
    # tung loai con cua OTHER (theo du doan out-of-fold hat 0)
    p = preds.dropna(subset=["pred_A_payload_role_eng"]).join(lab[["sub_type"]])
    oth = p[p.type4 == "OTHER"]
    out["OTHER_recall_by_subtype_A_seed0"] = {k: {"n": int(len(v)), "recall": round(float((v.pred_A_payload_role_eng == "OTHER").mean()), 3)} for k, v in oth.groupby("sub_type")}
    # tu kiem
    assert sum(out["all_labels"]["A_profiled_only"]["class_counts"].values()) == int(lab.has_profile.sum())
    assert out["all_labels"]["B_all_behaviour_only"]["n"] == len(lab)
    assert out["strict"]["A_profiled_only"]["n"] == int((lab.has_profile & ~lab.owner_approved).sum())
    preds.to_csv(F / "phase2_train4_preds_v6.csv", encoding="utf-8-sig")
    (F / "phase2_train4_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
