"""Gan nhan thiet bi 4SICS GeekLounge (151020-151022) tu danh sach thiet bi cong bo (Netresec PCAP4SICS) (2026-11-12).

Nguon danh sach: trang netresec.com/?page=PCAP4SICS (cong bo, chu de tai dan lai). Ghep theo IP voi thiet bi v6 co session 4SICS, kiem tra hang theo OUI.
KHONG sua gold_union_v6.csv / phase2_type_labels_4class_plus_v6.csv: ghi file RIENG:
  4sics_roster_match_v6.csv   moi dong danh sach -> thiet bi v6 (neu co), nhan de xuat, do tin cay, nhan luat va diem Phase 1 hien tai
  gold_roster_add_v6.csv      (device_id,gold_label,source,confidence) cho Phase 1: CHI muc 'high'
  phase2_roster_add_v6.csv    (cot nhu phase2_type_labels_4class_plus_v6.csv) cho Phase 2: CHI muc 'high', chua co nhan
Ha tang mang (switch, router, firewall) de 'ha_tang' (chua chot phan loai) nen KHONG vao gold.
  py -3.14 77_roster_labels_4sics.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"

# (ip, ten theo danh sach, tu khoa hang OUI ky vong, nhan OT, loai phase2, ghi chu)
ROSTER = [
    ("192.168.88.15", "DirectLogic 205 (PLC)", ["host eng"], "OT", "PLC", ""),
    ("192.168.88.20", "Phoenix Contact FL IL 24 BK-PAC (Ethernet bus coupler)", ["phoenix"], "OT", "OTHER", "bo ghep bus/IO"),
    ("192.168.88.25", "Advantech ADAM-5500 (Ethernet I/O module)", ["advantech"], "OT", "OTHER", "module IO"),
    ("192.168.88.49", "AXIS 206 Network Camera", ["axis"], "non-OT", "OTHER", "camera mang"),
    ("192.168.88.75", "Hirschmann EAGLE 20 Tofino (Firewall)", ["hirschmann"], "ha_tang", "", "firewall"),
    ("192.168.88.30", "Siemens SIMATIC S7-1200 (PLC)", ["siemens"], "OT", "PLC", ""),
    ("192.168.88.91", "RUGGEDCOM RS910", ["ruggedcom"], "ha_tang", "", "serial device server + switch"),
    ("192.168.88.92", "RUGGEDCOM RS910", ["ruggedcom"], "ha_tang", "", ""),
    ("192.168.88.93", "RUGGEDCOM RS910", ["ruggedcom"], "ha_tang", "", ""),
    ("192.168.88.95", "RUGGEDCOM RS910", ["ruggedcom"], "ha_tang", "", ""),
    ("192.168.89.1", "RUGGEDCOM RX1100 (Router)", ["ruggedcom"], "ha_tang", "", "router"),
    ("192.168.88.50", "Red Lion DSP (protocol converter)", ["red lion"], "OT", "OTHER", "bo chuyen doi giao thuc"),
    ("192.168.88.60", "Moxa EDS-508A (Switch)", ["moxa"], "ha_tang", "", ""),
    ("192.168.88.61", "Moxa EDS-508A (Switch)", ["moxa"], "ha_tang", "", ""),
    ("192.168.88.70", "Cisco Catalyst 2955 (Switch)", ["cisco"], "ha_tang", "", ""),
    ("192.168.88.80", "Moxa UC-7112 (Embedded computer)", ["moxa"], "khong_ro", "", "may tinh nhung, chua ro vai tro"),
    ("192.168.88.100", "HOST Engineering MB-gateway (Modbus gateway)", ["host eng"], "OT", "OTHER", "gateway Modbus"),
    ("192.168.88.51", "Beckhoff CX1010 (Win CE)", ["beckhoff"], "OT", "PLC", "danh sach noi Beckhoff nhung OUI trong capture la hang khac -> can kiem tra"),
    ("192.168.88.85", "xLogic x-Messenger EXM-12DC-DA-RT-WiFi (WiFi PLC)", ["hi-flying", "xlogic"], "OT", "PLC", "OUI la hang module WiFi"),
    ("192.168.88.105", "TCP/IP to RS-232/422/485 Converter", ["cimsys"], "OT", "OTHER", "bo chuyen doi noi tiep; OUI chua xac minh"),
    ("192.168.88.1", "Westermo Lynx 3643-0105 (Router)", ["westermo"], "ha_tang", "", "router"),
    ("192.168.87.1", "Westermo Lynx 3643-0105 (Router)", ["westermo"], "ha_tang", "", "router"),
    ("192.168.88.115", "Westermo Digi (serial/terminal server)", ["digi"], "OT", "OTHER", "may chu noi tiep (cong 7000-7025); chua xac minh"),
    ("10.10.10.10", "Siemens SIMATIC S7 (PLC) - Svenska Kraftnat", ["siemens"], "OT", "PLC", ""),
    ("10.10.10.30", "Windows PC - Svenska Kraftnat", ["wistron", "microsoft"], "khong_ro", "", "PC; vai tro (HMI/EWS/khac) chua ro, nhan luat OT_confirmed"),
]


def main():
    dev = pd.read_csv(HERE.parent / "Dataset/metadata/devices.csv", dtype=str, keep_default_na=False).rename(columns={"device_id": "dev_old"})
    dev["mac"] = dev.mac.str.lower()
    d4 = dev[dev.session_ids.str.contains("4SICS", case=False)]
    v6 = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False)
    oof = pd.read_csv(F / "phase1_oof_scores_v6.csv").set_index("device_id")
    gold = pd.read_csv(F / "gold_union_v6.csv", dtype=str)
    p2 = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", dtype=str)
    m = d4.merge(v6[["device_id", "device_mac", "label", "evidence_tier"]], left_on="mac", right_on="device_mac", how="inner")
    rows = []
    for ip, name, kws, ot, t2, note in ROSTER:
        c = m[m.ip == ip]
        if c.empty:
            rows.append({"roster_ip": ip, "roster_name": name, "device_id": "", "mac": "", "oui_vendor": "", "match": "khong_thay_trong_v6", "proposed_OT": ot, "proposed_phase2": t2,
                         "confidence": "", "note": note})
            continue
        for _, r in c.iterrows():
            ok = any(k in r.vendor.lower() for k in kws)
            conf = "high" if ok and "chua xac minh" not in note and "OUI la hang" not in note else ("medium" if ok else "mismatch")
            rows.append({"roster_ip": ip, "roster_name": name, "device_id": r.device_id, "mac": r.mac, "oui_vendor": r.vendor, "match": "ip+oui" if ok else "ip_only", "proposed_OT": ot,
                         "proposed_phase2": t2, "confidence": conf, "note": note, "rule_label": r.label, "evidence_tier": r.evidence_tier,
                         "phase1_oof_score": round(float(oof.loc[r.device_id, "lightgbm"]), 3), "in_gold": r.device_id in set(gold.device_id), "in_phase2_labels": r.device_id in set(p2.device_id)})
    out = pd.DataFrame(rows)
    for c in ("in_gold", "in_phase2_labels"):
        out[c] = out[c].fillna(False).astype(bool)
    out.to_csv(F / "4sics_roster_match_v6.csv", index=False, encoding="utf-8-sig")
    hi = out[(out.confidence == "high") & out.proposed_OT.isin(["OT", "non-OT"])]
    g = hi[~hi.in_gold][["device_id", "proposed_OT", "roster_name"]].rename(columns={"proposed_OT": "gold_label"})
    g["source"], g["confidence"] = "public_roster_4sics_netresec", "high"
    g[["device_id", "gold_label", "source", "confidence"]].to_csv(F / "gold_roster_add_v6.csv", index=False)
    p = hi[(hi.proposed_phase2 != "") & ~hi.in_phase2_labels]
    p2add = pd.DataFrame({"device_id": p.device_id, "type4": p.proposed_phase2, "sub_type": p.proposed_phase2, "source": "public_roster_4sics_netresec", "confidence": "high",
                          "detail": p.roster_name, "tier1_role": "", "role_evidence": "", "gold_source": "public_roster_4sics_netresec"})
    p2add.to_csv(F / "phase2_roster_add_v6.csv", index=False)
    print(out.match.value_counts().to_dict(), out.confidence.value_counts().to_dict())
    print("gold add:", len(g), g.gold_label.value_counts().to_dict(), "| phase2 add:", len(p2add), p2add.type4.value_counts().to_dict())
    print(out[out.confidence.isin(["medium", "mismatch"]) | (out.match != "ip+oui")][["roster_ip", "roster_name", "oui_vendor", "match", "confidence"]].to_string(index=False))
    h = out[out.device_id != ""]
    print("Phase 1 OOF score vs de xuat (cac muc khong ha_tang/khong_ro):")
    print(h[h.proposed_OT.isin(["OT", "non-OT"])].groupby("proposed_OT").phase1_oof_score.agg(["count", "min", "median", "max"]).round(3))


if __name__ == "__main__":
    main()
