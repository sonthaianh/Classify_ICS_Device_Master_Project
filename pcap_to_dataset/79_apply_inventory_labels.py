"""Ap quyet dinh cua chu de tai (2026-11-12) de gan nhan Phase 1 / Phase 2 cho thiet bi khop bang tai san nha may (script 75) va danh sach 4SICS (script 77).

Quyet dinh cua chu de tai:
  1. Ha tang mang cong nghiep (Moxa, RUGGEDCOM, Hirschmann, Westermo) = OT; Cisco, Fortinet = non-OT (theo mo ta lua chon); hang khac (Planet, TXOne...) = KHONG gan.
  2. DCS Valmet DNA dung nhu SCADA -> Phase 1: OT; Phase 2: HMI (SCADA da gop vao HMI), sub_type 'SCADA_DCS_valmet' de co the quay lai.
  3. May chu va tram van hanh cua nha may (Dell, Purdue 2) coi la SCADA -> Phase 1: OT; Phase 2: HMI (sub_type 'SCADA_server'/'SCADA_operator_ws').
  4. Gateway noi tiep / bo chuyen doi giao thuc = OTHER (Phase 2), OT (Phase 1).
Quy tac chung: chi dung cap IP-thiet bi muc 'high' (customer_site) ; 4SICS 'high' + 'medium' (xLogic, bo chuyen doi, Digi) ; khong dung 'low' / 'mismatch'.
KHONG ghi de gold_union_v6.csv, phase2_type_labels_4class_plus_v6.csv, gold_union_v6_plus_4sics.csv, ... ; dau ra la file MOI co hau to _plus_all.

  py -3.14 79_apply_inventory_labels.py <inventory.csv>
"""
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
IND_NET = ("moxa", "ruggedcom", "hirschmann", "westermo")
ENT_NET = ("cisco", "fortinet")
INFRA_TYPES = {"Switch", "Firewall", "Network Device", "Router"}
TAG = {"customer_site": "customer_site_inventory", "4sics": "public_roster_4sics_netresec"}


def has(s, kws):
    s = str(s).lower()
    return any(k in s for k in kws)


def decide_inventory(r):
    """Tra ve (gold_label, phase2_type, sub_type, ly_do) hoac (None, None, None, ly_do_bo)."""
    t, st, mfr, pur = r.Type, r.Subtype, r["Mfr/Brand"], str(r["Purdue Model"])
    if t in INFRA_TYPES:
        if has(mfr, IND_NET):
            return "OT", None, None, "ha_tang cong nghiep = OT (quyet dinh 1)"
        if has(mfr, ENT_NET):
            return "non-OT", None, None, "ha_tang doanh nghiep (Cisco/Fortinet) = non-OT (quyet dinh 1)"
        return None, None, None, "ha_tang hang khac: chua quyet dinh"
    if t == "Industrial Device":
        if st == "DCS":
            return "OT", "HMI", "SCADA_DCS_valmet", "DCS Valmet = SCADA (quyet dinh 2)"
        if st == "PLC":
            return "OT", "PLC", "PLC", "PLC theo bang tai san"
        if st == "HMI":
            return "OT", "HMI", "HMI", "HMI theo bang tai san"
        return "OT", "OTHER", "gateway_or_field_device", "thiet bi cong nghiep khac/gateway = OTHER (quyet dinh 4)"
    if t in ("Server", "Workstation") and "dell" in str(mfr).lower() and pur.strip() == "2":
        return "OT", "HMI", "SCADA_server" if t == "Server" else "SCADA_operator_ws", "may chu/tram van hanh Dell Purdue 2 = SCADA (quyet dinh 3)"
    return None, None, None, f"{t}: chua quyet dinh"


def main():
    inv_path = sys.argv[1]
    inv = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
    mm = pd.read_csv(F / "inventory_match_v6.csv", dtype=str, keep_default_na=False)
    rost = pd.read_csv(F / "4sics_roster_match_v6.csv", dtype=str, keep_default_na=False)
    g_prev = pd.read_csv(F / "gold_union_v6_plus_4sics.csv", dtype=str)
    p_prev = pd.read_csv(F / "phase2_type_labels_4class_plus_v6_plus_4sics.csv", dtype=str)
    prof = json.loads((F / "phase2_payload_profiles_v6_full.json").read_text(encoding="utf-8"))["records"]
    have = {r["device_id"] for r in prof if r.get("device_id")}
    rows, excl = [], []
    for _, r in mm.iterrows():
        dcs_hw = r.confidence == "medium" and r.Subtype == "DCS" and has(r.vendor, ("portwell", "ibase"))   # phan cung Portwell/IBASE cua bo DCS; chu de tai xac nhan nhom Valmet DNA
        if r.confidence != "high" and not dcs_hw:
            excl.append((r.device_id, "customer_site", r.Name, f"muc {r.confidence}"))
            continue
        g, p2, sub, why = decide_inventory(r)
        if g is None:
            excl.append((r.device_id, "customer_site", r.Name, why))
            continue
        if dcs_hw:
            why += "; OUI phan cung Portwell/IBASE (khong khop dung cap IP-OUI): muc medium"
        rows.append({"device_id": r.device_id, "src": "customer_site", "name": r.Name, "gold_label": g, "type4": p2, "sub_type": sub, "why": why, "confidence": "medium" if dcs_hw else "high", "detail": f"{r.Type}/{r.Subtype} {r['Mfr/Brand']} {r['Product/Model']}".strip()})
    for _, r in rost[rost.device_id != ""].iterrows():
        if r.confidence not in ("high", "medium"):
            excl.append((r.device_id, "4sics", r.roster_name, f"muc {r.confidence}"))
            continue
        low = r.roster_name.lower()
        if r.proposed_OT == "ha_tang":
            g, p2, sub = ("OT", None, None) if has(low, IND_NET) else ((("non-OT", None, None)) if has(low, ENT_NET) else (None, None, None))
            why = "ha_tang (quyet dinh 1)"
        elif r.proposed_OT in ("OT", "non-OT"):
            g, p2, sub, why = r.proposed_OT, (r.proposed_phase2 or None), (r.proposed_phase2 or None), "danh sach 4SICS"
        else:
            g = None
            why = "chua ro vai tro"
        if g is None:
            excl.append((r.device_id, "4sics", r.roster_name, why))
            continue
        rows.append({"device_id": r.device_id, "src": "4sics", "name": r.roster_name, "gold_label": g, "type4": p2, "sub_type": sub, "why": why, "confidence": r.confidence, "detail": r.roster_name})
    add = pd.DataFrame(rows).drop_duplicates("device_id")
    dup = add[add.device_id.duplicated(keep=False)]
    # bo cac thiet bi da co trong gold/phase2 truoc do
    g_new = add[~add.device_id.isin(g_prev.device_id)]
    gold_add = pd.DataFrame({"device_id": g_new.device_id, "gold_label": g_new.gold_label, "source": g_new.src.map(TAG), "confidence": g_new.confidence})
    p_new = add[add.type4.notna() & ~add.device_id.isin(p_prev.device_id) & (add.gold_label == "OT")]
    p2_add = pd.DataFrame({"device_id": p_new.device_id, "type4": p_new.type4, "sub_type": p_new.sub_type, "source": p_new.src.map(TAG), "confidence": p_new.confidence, "detail": p_new.detail,
                           "tier1_role": "", "role_evidence": "", "gold_source": p_new.src.map(TAG)})
    gold_add.to_csv(F / "gold_inventory_add_v6.csv", index=False)
    p2_add.to_csv(F / "phase2_inventory_add_v6.csv", index=False)
    pd.concat([g_prev, gold_add], ignore_index=True).to_csv(F / "gold_union_v6_plus_all.csv", index=False)
    pd.concat([p_prev, p2_add], ignore_index=True).to_csv(F / "phase2_type_labels_4class_plus_v6_plus_all.csv", index=False)
    # dien file Excel duyet
    xl = openpyxl.load_workbook(F / "inventory_review_sheet_v6.xlsx")
    dec = {r.device_id: r for r in add.itertuples() if r.src == "customer_site"}
    note = "Dien theo quyet dinh chu de tai 2026-11-12 (chua duyet tung cap)"
    for sheet in ("review_high", "review_medium", "low_khong_dung"):
        ws = xl[sheet]
        hdr = {c.value: c.column for c in ws[1]}
        for row in range(2, ws.max_row + 1):
            dev = ws.cell(row, hdr["device_id"]).value
            conf = {"review_high": "high", "review_medium": "medium", "low_khong_dung": "low"}[sheet]
            d = dec.get(dev)
            ws.cell(row, hdr["xac_nhan_cap_IP_thiet_bi"]).value = "dung" if conf == "high" else "khong_chac"
            if d is not None:
                ws.cell(row, hdr["loai_OT"]).value = d.gold_label
                ws.cell(row, hdr["loai_phase2"]).value = d.type4 if d.type4 else "khong_ap_dung"
                ws.cell(row, hdr["ghi_chu"]).value = f"{note}; {d.why}"
            else:
                reason = next((e[3] for e in excl if e[0] == dev and e[1] == "customer_site"), "")
                ws.cell(row, hdr["loai_OT"]).value = "khong_ro"
                ws.cell(row, hdr["loai_phase2"]).value = "khong_ap_dung"
                ws.cell(row, hdr["ghi_chu"]).value = f"{note}; khong gan: {reason}"
    q = xl["quyet_dinh"]
    answers = ["OT cho ha tang cong nghiep (Moxa/RUGGEDCOM/Hirschmann/Westermo); Cisco/Fortinet non-OT; hang khac khong gan", "DCS Valmet = SCADA -> HMI (Phase 2), OT (Phase 1)",
               "OTHER (gateway noi tiep, bo chuyen doi)", "May chu/tram van hanh Dell Purdue 2 = SCADA -> HMI, OT", "May chu/tram van hanh Dell Purdue 2 = SCADA -> HMI, OT (cac may chu/tram khac chua quyet dinh)"]
    for i, a in enumerate(answers, start=2):
        q.cell(i, 2).value = a
    xl.save(F / "inventory_review_sheet_v6_filled.xlsx")
    # bang tai san chua co luu luong
    inv["ip1"] = inv.IP.str.split(",").str[0].str.strip()
    un = inv[~inv.ip1.isin(set(mm.ip))].copy()
    out = []
    for _, r in un.iterrows():
        g, p2, sub, why = decide_inventory(r)
        out.append({"Name": r.Name, "IP": r.IP, "Type": r.Type, "Subtype": r.Subtype, "Mfr/Brand": r["Mfr/Brand"], "Product/Model": r["Product/Model"], "Purdue Model": r["Purdue Model"],
                    "proposed_gold_label": g or "", "proposed_phase2": p2 or "", "sub_type": sub or "", "note": why, "in_dataset": "khong (khong co luu luong trong parquet hien co)"})
    pd.DataFrame(out).to_csv(F / "inventory_assets_without_traffic_v6.csv", index=False, encoding="utf-8-sig")
    rep = {"customer_site_high_pairs": int((mm.confidence == "high").sum()), "gold_added_new": len(gold_add), "gold_added_by_label": gold_add.gold_label.value_counts().to_dict(),
           "gold_added_by_source": gold_add.source.value_counts().to_dict(), "gold_total_new": int(len(g_prev) + len(gold_add)),
           "phase2_added_new": len(p2_add), "phase2_added_by_type": p2_add.type4.value_counts().to_dict(), "phase2_added_by_sub_type": p2_add.sub_type.value_counts().to_dict(),
           "phase2_added_with_payload_profile": int(p2_add.device_id.isin(have).sum()),
           "phase2_total_new": int(len(p_prev) + len(p2_add)), "phase2_class_counts_new": pd.concat([p_prev.type4, p2_add.type4]).value_counts().to_dict(),
           "excluded": pd.DataFrame(excl, columns=["device_id", "src", "name", "reason"]).reason.str.replace(r":.*", "", regex=True).value_counts().to_dict(),
           "assets_without_traffic": len(out), "assets_without_traffic_with_proposed_label": int(sum(1 for o in out if o["proposed_gold_label"]))}
    (F / "inventory_labels_applied_v6.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(pd.DataFrame(excl, columns=["device_id", "src", "name", "reason"]).to_string(index=False)[:3500])


if __name__ == "__main__":
    main()
