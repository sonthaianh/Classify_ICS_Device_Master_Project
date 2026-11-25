"""Phase 1 -> Phase 2 error propagation (RQ5), 2026-11-11. Chi dung ket qua da co, KHONG huan luyen lai.

Mo hinh Phase 1 = ban dong bang (phase1_oof_scores_v6.csv, cot lightgbm, nguong 0,5, diem out-of-fold).
Phase 2 = du doan out-of-fold 4 lop (phase2_train4_preds_v6.csv, bo A payload+vai tro+tac vu) tren 101/122 thiet bi co nhan va co ho so payload (phase2_type_labels_4class_plus_v6.csv).
Ket qua cuoi (end-to-end) cua mot thiet bi: Phase 1 < 0,5 -> "non-OT/OTHER" (khong di tiep); nguoc lai -> nhan Phase 2.
CANH BAO thien lech: 33/56 nhan OTHER la thiet bi non-OT bi Phase 1 gan nham (chon tu danh sach da qua Phase 1) -> moi OTHER deu "qua cong"; phan OTHER
cua end-to-end KHONG danh gia duoc cong. Chi so dang tin hon la 3 lop OT (PLC/HMI/EWS): ty le qua cong va ty le dung sau cong. Nhan chua chuyen gia xac nhan.
Dau ra: endtoend_phase1_phase2_v6.csv, endtoend_phase1_phase2_v6.json     Chay: py -3.14 Tools/pcap_to_dataset/66_endtoend_phase1_phase2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

F = Path(__file__).resolve().parent.parent / "Dataset/features"
THR = 0.5


def main():
    lab = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", encoding="utf-8-sig")[["device_id", "type4", "source"]]
    pr = pd.read_csv(F / "phase2_train4_preds_v6.csv", encoding="utf-8-sig")
    p1 = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig")[["device_id", "lightgbm"]].rename(columns={"lightgbm": "p1_score"})
    d = lab.merge(pr[["device_id", "pred_A_payload_role_eng"]], on="device_id", how="left").merge(p1, on="device_id", how="left")
    assert len(d) == 122 and d.p1_score.notna().all()
    nopred = d[d.pred_A_payload_role_eng.isna()]   # 21 thiet bi khong co ho so payload -> Phase 2 khong co du doan; bao rieng, khong tinh vao ty le
    no_pred_info = {"n": int(len(nopred)), "by_class": nopred.type4.value_counts().to_dict(), "pass_phase1": int((nopred.p1_score >= THR).sum())}
    d = d[d.pred_A_payload_role_eng.notna()].copy()
    d["pass_p1"] = d.p1_score >= THR
    d["p2_correct"] = d.pred_A_payload_role_eng == d.type4
    d["final"] = np.where(d.pass_p1, d.pred_A_payload_role_eng, "OTHER(blocked)")
    d["final_correct"] = np.where(d.pass_p1, d.p2_correct, d.type4 == "OTHER")
    d.to_csv(F / "endtoend_phase1_phase2_v6.csv", index=False, encoding="utf-8-sig")

    res = {"status": "EXPLORATORY; labels not expert-verified; OTHER biased (selected from Phase 1 false positives)", "threshold": THR, "n": int(len(d)), "devices_without_phase2_prediction_(no payload profile)": no_pred_info, "per_class": {}}
    for c in ("PLC", "HMI", "EWS", "OTHER"):
        s = d[d.type4 == c]
        ps = s[s.pass_p1]
        res["per_class"][c] = {"n": int(len(s)), "pass_phase1": int(s.pass_p1.sum()), "pass_rate": round(float(s.pass_p1.mean()), 3),
                               "phase2_correct_given_pass": int(ps.p2_correct.sum()), "phase2_acc_given_pass": round(float(ps.p2_correct.mean()), 3) if len(ps) else None,
                               "end_to_end_correct": int(s.final_correct.sum()), "end_to_end_recall": round(float(s.final_correct.mean()), 3),
                               "phase2_only_recall_all_devices(oracle gate)": round(float(s.p2_correct.mean()), 3)}
    ot = d[d.type4.isin(["PLC", "HMI", "EWS"])]
    res["ot_types_pooled"] = {"n": int(len(ot)), "pass_rate": round(float(ot.pass_p1.mean()), 3), "phase2_correct_given_pass": round(float(ot[ot.pass_p1].p2_correct.mean()), 3),
                              "end_to_end_correct": round(float(ot.final_correct.mean()), 3), "oracle_gate_correct": round(float(ot.p2_correct.mean()), 3)}
    res["ot_types_blocked_by_phase1"] = d[(~d.pass_p1) & d.type4.isin(["PLC", "HMI", "EWS"])][["device_id", "type4", "p1_score"]].round(3).to_dict("records")
    # Phase 2 tipping in the other direction: OT-type devices predicted OTHER inside Phase 2 (second kind of loss)
    res["ot_types_passed_but_predicted_OTHER"] = int(((ot.pass_p1) & (ot.pred_A_payload_role_eng == "OTHER")).sum())
    # confusion after the gate (passed devices only)
    ps = d[d.pass_p1]
    res["confusion_passed_only"] = pd.crosstab(ps.type4, ps.pred_A_payload_role_eng).to_dict("index")
    (F / "endtoend_phase1_phase2_v6.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
