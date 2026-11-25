"""Phase 2: kiem tra cach chia fold va cot trung tren bo 186 nhan (2026-11-22). KHONG ghi de ket qua dong bang.

Tra loi ba cau hoi cua phan bien (Thay):
  1) Moi fold: danh sach thiet bi train/test; phan giao giua train va test THEO SESSION (capture) khi nhom = session dau.
  2) Cac thiet bi cua tap A noi voi nhau qua session nao (thanh phan lien thong); neu chia theo thanh phan lien thong thi diem con bao nhieu.
  3) Bo cot trung (log_servers trung voi n_servers) co lam diem Random Forest thay doi khong.
Dung lai build() va cv() cua script 58, cung nhan 186, tap A (co ho so payload, n = 165), RF 64 cot, 5 hat.
Dau ra: phase2_fold_audit_186.json, phase2_fold_assignments_186.csv (moi dong = thiet bi x hat x fold).
Chay:  py -3.14 101_phase2_fold_audit.py
"""
import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
KS = "behaviour_payload_role_eng"


def components(devs, dev_sessions):
    """Thanh phan lien thong cua do thi thiet bi - session (chi tinh thiet bi trong devs)."""
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for d in devs:
        parent[("d", d)] = ("d", d)
        for s in dev_sessions[d]:
            parent.setdefault(("s", s), ("s", s))
            union(("d", d), ("s", s))
    return {d: find(("d", d)) for d in devs}


def main():
    lab, sets = S58.build()
    A = lab[lab.has_profile]
    y, g = A.type4.to_numpy(), A.group.to_numpy()
    X = sets[KS].loc[A.index]
    print("n A =", len(A), "cols =", X.shape[1], flush=True)

    win = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    ds = win[win.device_id.isin(A.index)].groupby("device_id").session_id.agg(lambda s: set(s)).to_dict()
    n_sess = pd.Series({d: len(v) for d, v in ds.items()})
    out = {"status": "AUDIT on frozen 186-label set; no frozen result overwritten", "n_devices_A": int(len(A)), "n_columns": int(X.shape[1])}
    out["sessions_per_device"] = {"median": float(n_sess.median()), "mean": round(float(n_sess.mean()), 2), "share_devices_with_gt1_session": round(float((n_sess > 1).mean()), 3),
                                  "n_distinct_sessions_in_A": int(len(set().union(*ds.values()))), "n_distinct_first_sessions": int(len(set(g)))}

    # ---- 1) phan giao train/test theo session
    rows, per = [], []
    for seed in range(5):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g)):
            dtr, dte = A.index[tr], A.index[te]
            s_tr = set().union(*[ds[d] for d in dtr])
            share = [bool(ds[d] & s_tr) for d in dte]
            only_first = [bool(ds[d] - {g[A.index.get_loc(d)]}) for d in dte]  # thiet bi con xuat hien o capture khac ngoai capture dau
            s_te = set().union(*[ds[d] for d in dte])
            per.append({"seed": seed, "fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)), "test_devices_sharing_a_session_with_train": int(sum(share)),
                        "share_of_test_devices": round(float(np.mean(share)), 3), "sessions_in_both": int(len(s_tr & s_te)), "sessions_in_test": int(len(s_te)),
                        "first_session_overlap": int(len(set(g[tr]) & set(g[te])))})
            for d in dte:
                rows.append({"seed": seed, "fold": fold, "device_id": d, "type4": A.type4[d], "first_session": g[A.index.get_loc(d)], "n_sessions": len(ds[d])})
    pd.DataFrame(rows).to_csv(F / "phase2_fold_assignments_186.csv", index=False, encoding="utf-8-sig")
    P = pd.DataFrame(per)
    out["train_test_overlap"] = {"first_session_overlap_total": int(P.first_session_overlap.sum()),
                                 "mean_share_of_test_devices_sharing_a_session_with_train": round(float(P.share_of_test_devices.mean()), 3),
                                 "min": round(float(P.share_of_test_devices.min()), 3), "max": round(float(P.share_of_test_devices.max()), 3),
                                 "mean_sessions_in_both": round(float(P.sessions_in_both.mean()), 1), "mean_sessions_in_test": round(float(P.sessions_in_test.mean()), 1)}
    out["per_fold"] = per
    print("overlap", out["train_test_overlap"], flush=True)

    # ---- 2) thanh phan lien thong tren tap A
    comp = components(list(A.index), ds)
    cs = pd.Series(comp).map(lambda c: str(c))
    sizes = cs.value_counts()
    out["components"] = {"n_components": int(len(sizes)), "sizes_top10": sizes.head(10).tolist(), "largest_share": round(float(sizes.iloc[0] / len(A)), 3),
                         "n_singletons": int((sizes == 1).sum())}
    print("components", out["components"], flush=True)
    gc = cs.to_numpy()
    cls_in_large = pd.Series(y[cs.values == sizes.index[0]]).value_counts().to_dict()
    out["components"]["class_counts_in_largest"] = {k: int(v) for k, v in cls_in_large.items()}
    try:
        res_cc, _ = S58.cv(X, y, gc)
        out["rf64_group_by_component"] = res_cc
        print("RF64 group=component", res_cc["balanced_acc"], res_cc["recall"], flush=True)
    except Exception as e:  # noqa: BLE001
        out["rf64_group_by_component"] = {"error": str(e)}
        print("component CV failed:", e, flush=True)

    # ---- 3) bo cot trung
    dup = [c for c in X.columns if c != "log_servers" and np.allclose(X[c].to_numpy(float), X["log_servers"].to_numpy(float))]
    out["duplicate_of_log_servers"] = dup
    base, _ = S58.cv(X, y, g)
    out["rf64_baseline_recomputed"] = base
    print("RF64 baseline recomputed", base["balanced_acc"], base["balanced_acc_std"], flush=True)
    if dup:
        X63 = X.drop(columns=["log_servers"])
        r63, _ = S58.cv(X63, y, g)
        out["rf63_without_log_servers"] = r63
        out["n_columns_after_drop"] = int(X63.shape[1])
        print("RF63", r63["balanced_acc"], r63["balanced_acc_std"], flush=True)
    (F / "phase2_fold_audit_186.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
