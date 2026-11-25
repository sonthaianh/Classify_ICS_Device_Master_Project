"""Nhom A (2026-11-25), phan bo sung cho muc 4.3 tren bo nhan doc lap: bo dac trung, doi chung chi vai tro ket noi, hoan vi nhan, phan tich loi hat 0,
khoang tin cay do phu theo lop, do quan trong dac trung. KHONG ghi de ket qua dong bang.  Chay: py -3.14 113_group_A_extra.py [--set indep160|x186]
Dau ra: phase2_groupA_extra_<set>.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g110", HERE / "110_group_A_independent.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
F, C = G.F, G.CLASSES


def main():
    lab, sets, A, T = G.base()
    out = {"set": G.SET}
    ms = G.ms
    # 1) bo dac trung
    fs = {}
    for nm, sub in (("A", A), ("T", T)):
        y, g = sub.type4.to_numpy(), sub.group.to_numpy()
        for ks in ("behaviour", "payload_role", "payload_role_eng", "behaviour_payload_role_eng"):
            X = sets[ks].loc[sub.index]
            fs[f"{nm}|{ks}"] = {"n_cols": int(X.shape[1]), **ms([balanced_accuracy_score(y, G.rf_preds(X, y, g, s)) for s in range(5)])}
            print(nm, ks, fs[f"{nm}|{ks}"], flush=True)
    out["feature_sets"] = fs
    y, g, devs = A.type4.to_numpy(), A.group.to_numpy(), list(A.index)
    X = sets[G.KS].loc[A.index]
    p64 = [G.rf_preds(X, y, g, s) for s in range(5)]
    # 2) doi chung chi cot vai tro ket noi
    cols4 = ["ot_server_share", "hs_client_share", "log_syn", "log_synack"]
    cols7 = cols4 + ["tx_share", "n_clients", "n_servers"]
    out["role_only_control"] = {k: ms([balanced_accuracy_score(y, G.rf_preds(X[c], y, g, s)) for s in range(5)]) for k, c in (("role4", cols4), ("role7", cols7))}
    print(out["role_only_control"], flush=True)
    # 3) khoang tin cay do phu theo lop (hat 0) va loi hat 0
    rng = np.random.default_rng(20261111)
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(2000)]
    rec_ci = {}
    for c in C:
        v = [float((p64[0][i][y[i] == c] == c).mean()) for i in idx if (y[i] == c).any()]
        rec_ci[c] = {"seed0": round(float((p64[0][y == c] == c).mean()), 3), "ci95": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
    out["recall_seed0_ci"] = rec_ci
    meta = G.M._orig(G.LABEL, encoding="utf-8-sig").set_index("device_id")
    st = meta.loc[A.index, "sub_type"].to_numpy()
    wrong_other = (y == "OTHER") & (p64[0] != "OTHER")
    out["seed0_OTHER_errors"] = {"n_OTHER": int((y == "OTHER").sum()), "n_wrong": int(wrong_other.sum()), "pred_counts": pd.Series(p64[0][wrong_other]).value_counts().to_dict(),
                                 "wrong_as_PLC_by_subtype": pd.Series(st[wrong_other & (p64[0] == "PLC")]).value_counts().to_dict(),
                                 "OTHER_subtypes_in_A": pd.Series(st[y == "OTHER"]).value_counts().to_dict()}
    tot_other_plc = int(sum(((y == "OTHER") & (p == "PLC")).sum() for p in p64))
    out["OTHER_as_PLC_all_seeds"] = f"{tot_other_plc}/{int((y == 'OTHER').sum()) * 5}"
    print(out["recall_seed0_ci"], out["seed0_OTHER_errors"], out["OTHER_as_PLC_all_seeds"], flush=True)
    # 4) hoan vi nhan (200 lan, hat 0)
    obs = float(balanced_accuracy_score(y, p64[0]))
    rng = np.random.default_rng(20261119)
    null = []
    for i in range(200):
        ys = rng.permutation(y)
        null.append(float(balanced_accuracy_score(ys, G.rf_preds(X, ys, g, 0, n_est=300))))
    out["permutation_RF64"] = {"observed_seed0": round(obs, 3), "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                               "p_value": round(float((1 + sum(v >= obs for v in null)) / 201), 4), "n_perm": 200}
    print(out["permutation_RF64"], flush=True)
    # 5) do quan trong dac trung (bo 49 cot va bo 64 cot, huan luyen tren toan tap A)
    for ks in ("payload_role_eng", "behaviour_payload_role_eng"):
        Xk = sets[ks].loc[A.index]
        imp = np.mean([RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=s, n_jobs=-1).fit(Xk, y).feature_importances_ for s in range(5)], axis=0)
        out[f"importance_top8|{ks}"] = [(c, round(float(v), 3)) for c, v in sorted(zip(Xk.columns, imp), key=lambda t: -t[1])[:8]]
        print(ks, out[f"importance_top8|{ks}"], flush=True)
    # 6) so thiet bi tap A khong co thong diep OT
    out["A_without_OT_tokens"] = int((~A.index.isin(T.index)).sum())
    out["A_without_OT_tokens_by_class"] = A[~A.index.isin(T.index)].type4.value_counts().to_dict()
    (F / f"phase2_groupA_extra_{G.SET}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("DONE-EXTRA", out["A_without_OT_tokens"], out["A_without_OT_tokens_by_class"])


if __name__ == "__main__":
    sys.exit(main())
