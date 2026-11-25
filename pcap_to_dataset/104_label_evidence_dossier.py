"""Ho so bang chung cho 9 nhan DCS va 21 nhan ra soat tu bang chung session (2026-11-22). Chi doc du lieu co san.

Muc dich: giup nguoi thuc hien de tai (va Thay) kiem tra tung nhan: thiet bi la gi, bang chung nao, bang chung do co trung voi thu ma dac trung do hay khong.
Cot 'xac_minh_cua_de_tai' va 'ket_luan' de TRONG: chi nguoi thuc hien de tai (co bang tai san, nguoi van hanh) moi dien duoc.
Dau ra: Dataset/features/label_evidence_dossier_186.csv va .xlsx ; ban sao xlsx o thu muc Luan_Van.
Chay: py -3.14 104_label_evidence_dossier.py
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
OUT_COPY = HERE.parent.parent / "Luan_Van" / "BANG_RA_SOAT_NHAN_DCS_VA_21_NHAN.xlsx"


def main():
    lab = pd.read_csv(F / "phase2_type_labels_4class_186_devices.csv", encoding="utf-8-sig")
    rev = pd.read_csv(F / "device_session_review_v6.csv", encoding="utf-8-sig", usecols=["device_id", "session_id", "device_mac", "device_ips"])
    inv = pd.read_csv(F / "inventory_match_v6.csv", encoding="utf-8-sig").drop_duplicates("device_id").set_index("device_id")
    preds = pd.read_csv(F / "phase2_train4_preds_v6_186.csv", encoding="utf-8-sig") if (F / "phase2_train4_preds_v6_186.csv").exists() else None
    prof_ids = set(pd.read_csv(F / "phase2_fold_assignments_186.csv", encoding="utf-8-sig").device_id)
    sess = rev.groupby("device_id").agg(device_mac=("device_mac", "first"), n_captures=("session_id", "nunique"), captures_sample=("session_id", lambda s: "; ".join(sorted(set(s))[:6])))

    dcs = lab[lab.sub_type == "SCADA_DCS_valmet"].copy()
    new = lab[lab.source.isin(["reviewed_session_evidence", "reviewed_provisional"])].copy()
    rows = []
    for grp, df in (("DCS (nhãn HMI theo quy tắc gộp SCADA)", dcs), ("Nhãn rà soát từ bằng chứng session", new)):
        for _, r in df.iterrows():
            d = r.device_id
            e = {"nhom": grp, "device_id": d, "mac": sess.device_mac.get(d, ""), "nhan_hien_tai": r.type4, "nguon_nhan": r.source, "do_tin_cay": r.confidence,
                 "mo_ta_bang_chung": r.detail, "vai_tro_tang1_ket_noi": r.tier1_role, "bang_chung_vai_tro_ket_noi": r.role_evidence,
                 "so_capture": int(sess.n_captures.get(d, 0)), "capture_mau": sess.captures_sample.get(d, ""), "co_ho_so_payload_tap_A": d in prof_ids}
            if d in inv.index:
                for k_out, k_in in (("tai_san_ten", "Name"), ("tai_san_loai", "Type"), ("tai_san_hang", "Mfr/Brand"), ("tai_san_model", "Product/Model"), ("tai_san_muc_purdue", "Purdue Model"),
                                    ("tai_san_ip", "ip"), ("ghep_oui_khop", "oui_matches_inventory_mac"), ("ghep_do_tin_cay", "confidence")):
                    e[k_out] = inv.at[d, k_in]
            txt = str(r.detail)
            if grp.startswith("DCS"):
                e["loai_bang_chung"] = "bảng tài sản (loại và model), ghép theo IP + OUI"
                e["nguy_co_trung_dac_trung"] = "thấp (không dùng hành vi để gán nhãn)"
                e["van_de_can_xac_minh"] = "Loại 'DCS' và model 'CAN RT DC' chỉ tên hệ thống/phần cứng. Thiết bị này là bộ điều khiển, máy chủ điều khiển hay trạm vận hành? Quy ước gộp SCADA vào HMI có áp dụng được không?"
            else:
                ident = bool(re.search(r"string|banner|SIMATIC-ROOT-HMI|CPU", txt, re.I))
                e["loai_bang_chung"] = "chuỗi danh tính trong payload (mẫu thiết bị) + vai trò trong session" if ident else "chỉ hành vi trong session (máy khách/máy chủ, tác vụ)"
                e["nguy_co_trung_dac_trung"] = "trung bình (có chuỗi danh tính nhưng vai trò vẫn lấy từ hành vi)" if ident else "cao (bằng chứng là hành vi mà đặc trưng giai đoạn 2 cũng đo)"
                e["van_de_can_xac_minh"] = "Bằng chứng này có độc lập với lưu lượng dùng làm đặc trưng không? Có nguồn cấu hình hoặc người vận hành xác nhận không?" if not ident else "Chuỗi danh tính có đủ để xác định vai trò (PLC/HMI/EWS), hay chỉ xác định model?"
            e["xac_minh_cua_de_tai"] = ""
            e["ket_luan"] = ""
            rows.append(e)
    out = pd.DataFrame(rows)
    out.to_csv(F / "label_evidence_dossier_186.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(F / "label_evidence_dossier_186.xlsx", engine="openpyxl") as w:
            out.to_excel(w, index=False, sheet_name="ho_so_bang_chung")
            ws = w.sheets["ho_so_bang_chung"]
            for col in ws.columns:
                width = min(60, max(len(str(c.value)) if c.value is not None else 0 for c in col[:30]) + 2)
                ws.column_dimensions[col[0].column_letter].width = max(10, width)
            ws.freeze_panes = "C2"
        shutil.copy(F / "label_evidence_dossier_186.xlsx", OUT_COPY)
        print("xlsx ->", OUT_COPY)
    except Exception as ex:  # noqa: BLE001
        print("xlsx failed:", ex)
    print(out.groupby(["nhom", "loai_bang_chung"]).size())
    print(len(out), "rows")


if __name__ == "__main__":
    sys.exit(main())
