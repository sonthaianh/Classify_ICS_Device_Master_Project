"""Kiem tra lai hai cho so o Phu luc G cua luan van (2026-11-25). KHONG ghi de ket qua dong bang; chi ghi appendix_G_recheck.json.

(1) G.1 - thoi gian quan sat tren CUNG mot tap thiet bi: lay cac thiet bi co it nhat 10 cua so, tinh ROC-AUC cua giai doan 1 khi
    chi dung n cua so dau (n = 1, 2, 3, 5, 10). Cung cach tinh voi script 39 (trung binh xac suat ngoai mau cua n cua so dau,
    mo hinh LightGBM chia fold theo thiet bi); script 39 chi luu so tren cac tap thiet bi KHAC nhau (phase1_obstime_v6.csv).
(2) G.3 - diem giai doan 1 cua hai may chu phong thi nghiem mang nhan gold khong-OT (may chay cong cu quet).
Chay: py -3.14 115_recheck_appendix_G.py
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t39", HERE / "39_phase1_scanner_strict_obstime.py")
T39 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T39)
F = T39.F


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    ytrue = (dev.set_index("device_id").label == "OT_confirmed")
    gold = pd.read_csv(F / "gold_union_v6.csv", dtype=str).set_index("device_id")
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id")
    sA, wp = T39.fit_oof(x, dev, "fold_device", window_probs=True)
    wp = wp.sort_values(["device_id", "window_start"])
    wp["rank"] = wp.groupby("device_id").cumcount()
    nwin = wp.groupby("device_id").size()
    out = {"check_all_windows_roc": round(roc_auc_score(ytrue.loc[sA.index], sA), 4)}

    # ---- (1) cung tap thiet bi co >= 10 cua so
    ids10 = nwin[nwin >= 10].index
    y10 = ytrue.loc[ids10]
    g10 = gold.loc[gold.index.intersection(ids10)]
    rows = []
    for n in (1, 2, 3, 5, 10):
        s = wp[(wp["rank"] < n) & wp.device_id.isin(ids10)].groupby("device_id").p.mean().loc[ids10]
        r = {"first_n_windows": n, "approx_seconds": 45 * n, "n_devices": int(len(ids10)), "pop_roc_same_devices": round(roc_auc_score(y10, s), 4),
             "n_gold": int(len(g10)), "gold_roc_same_devices": round(roc_auc_score((g10.gold_label == "OT").to_numpy(), s.loc[g10.index]), 3)}
        rows.append(r)
    out["obstime_same_1556_devices"] = rows
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- (2) may quet trong tap gold
    inv = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sc = inv[inv.gold_label == "non-OT"]          # thiet bi phong thi nghiem duoc xac nhan la khong-OT (hai may chu chay cong cu quet)
    rec = []
    for r in sc.itertuples():
        d = r.device_id_v6
        if d in sA.index:
            rec.append({"device_id": d, "lab": r.lab, "device_type": r.device_type, "gold_label": r.gold_label, "stage1_score": round(float(sA.loc[d]), 3),
                        "rule_label": lab.label.get(d, ""), "max_service_ports_in_a_session": lab.max_service_ports_in_a_session.get(d, ""), "n_windows": int(nwin.get(d, 0))})
    out["scanner_like_lab_devices"] = rec
    print(pd.DataFrame(rec).to_string(index=False) if rec else "khong thay thiet bi phong thi nghiem nao co ghi chu may quet")
    g = gold[gold.gold_label == "non-OT"]
    top = sA.loc[g.index.intersection(sA.index)].sort_values(ascending=False).head(8)
    out["top_scored_gold_nonOT"] = [{"device_id": d, "stage1_score": round(float(v), 3), "source": gold.source[d], "rule_label": lab.label.get(d, "")} for d, v in top.items()]
    print(pd.DataFrame(out["top_scored_gold_nonOT"]).to_string(index=False))
    (F / "appendix_G_recheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("check all-window ROC (so luu san: 0,9286):", out["check_all_windows_roc"])


if __name__ == "__main__":
    sys.exit(main())
