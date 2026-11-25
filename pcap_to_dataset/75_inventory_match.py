"""Doi chieu bang tai san cua mot nha may (CSV import-asset) voi du lieu v6 (2026-11-12). Chi doc; ghi 1 file dau ra.

Khop theo IP (MAC trong bang tai san KHONG khop MAC trong capture; xem bao cao) + kiem tra nhat quan hang (OUI vs hang trong bang).
Muc tin cay:
  high   : IP thuoc dai dac thu cua site (khong phai dai chung kieu 192.168.0.x / 10.0.0.x) VA (3 byte dau MAC trong capture trung 3 byte dau MAC trong bang tai san HOAC hang OUI nhat quan voi hang trong bang)
  medium : IP thuoc dai dac thu nhung hang OUI khong ro / khac
  low    : IP thuoc dai chung de trung (192.168.0.*, 10.0.0.*) hoac dong mau mac dinh cua template -> KHONG dung de gan nhan
Dau ra KHONG ghi vao gold. Day la ung vien cho chu de tai/chuyen gia xac nhan.

  py -3.14 75_inventory_match.py <inventory.csv> [--out ten.csv]
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
GENERIC = re.compile(r"^(192\.168\.0\.|10\.0\.0\.|10\.0\.1\.|10\.0\.2\.)")
OT_TYPES = {"Industrial Device"}


def brand(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else F / "inventory_match_v6.csv"
    inv = pd.read_csv(args[0], dtype=str, keep_default_na=False)
    inv = inv.assign(ip=inv.IP.str.split(",")).explode("ip")
    inv["ip"] = inv.ip.str.strip()
    inv = inv[inv.ip != ""].drop_duplicates("ip")
    dev = pd.read_csv(HERE.parent / "Dataset/metadata/devices.csv", dtype=str, keep_default_na=False)
    v6 = pd.read_csv(F / "device_labels_v6_rule.csv", dtype=str, keep_default_na=False)
    fo = pd.read_csv(F / "device_folds_v6.csv", dtype=str, keep_default_na=False)
    gold = pd.read_csv(F / "gold_union_v6.csv", dtype=str)
    p2 = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", dtype=str)
    d = dev[dev.ip.isin(set(inv.ip))][["mac", "ip", "vendor"]].copy()
    d["mac"] = d.mac.str.lower()
    j = d.merge(v6[["device_id", "device_mac", "label", "evidence_tier"]], left_on="mac", right_on="device_mac", how="inner")
    j = j.merge(fo[["device_id", "primary_source_family", "fold_device"]], on="device_id", how="left")
    j = j.merge(inv[["ip", "Name", "Type", "Subtype", "Mfr/Brand", "Product/Model", "Purdue Model", "Criticality"]], on="ip", how="left")
    j["in_gold"] = j.device_id.isin(gold.device_id)
    j["in_phase2_labels"] = j.device_id.isin(p2.device_id)
    vend_ok = [brand(a)[:5] in brand(b) or brand(b)[:5] in brand(a) if brand(a) and brand(b) else False for a, b in zip(j.vendor, j["Mfr/Brand"])]
    # hang phan cung hay khac hang phan mem (Valmet chay tren phan cung Portwell/IBASE) -> chi coi la 'nhat quan' khi OUI trung hang; con lai de medium
    inv_oui = inv.set_index("ip").MAC.map(lambda s: {m.strip().lower().replace("-", ":")[:8] for m in str(s).split(",") if m.strip()})
    oui_ok = [m[:8] in inv_oui.get(i, set()) for m, i in zip(j.mac if "mac" in j else j.device_mac, j.ip)]
    j["oui_matches_inventory_mac"] = oui_ok
    j["vendor_consistent"] = [a or b for a, b in zip(vend_ok, oui_ok)]
    generic = j.ip.map(lambda s: bool(GENERIC.match(s)))
    template = j.Name.str.contains("Device Test", case=False)
    j["confidence"] = "medium"
    j.loc[j.vendor_consistent, "confidence"] = "high"
    j.loc[generic | template, "confidence"] = "low"
    j["inventory_is_OT"] = j.Type.isin(OT_TYPES)
    j["rule_label_agrees"] = (j.label == "OT_confirmed") == j.inventory_is_OT
    cols = ["device_id", "device_mac", "ip", "vendor", "primary_source_family", "fold_device", "label", "evidence_tier", "Name", "Type", "Subtype", "Mfr/Brand", "Product/Model",
            "Purdue Model", "Criticality", "inventory_is_OT", "rule_label_agrees", "vendor_consistent", "oui_matches_inventory_mac", "confidence", "in_gold", "in_phase2_labels"]
    j = j[cols].sort_values(["confidence", "Type", "ip"])
    j.to_csv(out, index=False, encoding="utf-8-sig")
    print(out, len(j))
    print(j.confidence.value_counts().to_dict())
    h = j[j.confidence != "low"]
    print("khong-low:", len(h), "| inventory OT:", int(h.inventory_is_OT.sum()), "| rule OT_confirmed:", int((h.label == "OT_confirmed").sum()))
    print(pd.crosstab(h.inventory_is_OT, h.label, rownames=["inv_is_OT"], colnames=["rule_label"]))
    print(h[h.inventory_is_OT].groupby(["Subtype"]).label.value_counts().to_string())


if __name__ == "__main__":
    main()
