"""Phase 2 tier 2: map payload identity evidence (script 46) to device TYPE with explicit, auditable rules; validate on the lab seeds.

Classes: PLC | HMI | EWS | IED | IO | OTHER   (OTHER = network infrastructure / IT host / camera / generic server)
Evidence weights: strong = 3 (device describes itself: LLDP/SNMP system description, DCP role/type, CIP device type, CPU/IM order number in a
self-description), medium = 2 (catalog/model string, station name convention, relay menu text), weak = 1 (server banners).
Decision per device: top class if score >= 2 and at least 1.0 above the runner-up; confidence 'high' if top >= 3 and no rival >= 2,
'medium' otherwise.  Margin < 1 or score < 2 -> 'undecided'.  Order-number logic: a device that emits a CPU order number is a PLC even if it
also lists I/O modules (S7 configuration listings); it is an IO device only if it lists IM/I-O modules and no CPU.
Outputs: phase2_identity_types_v6.csv, phase2_identity_validation_v6.json.  Rules are in RULES below; nothing is learned.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
CPU_S7 = re.compile(r"6ES7 ?(21[1-5]|5[01][0-9]|31[0-9]|41[0-9]|6[0-9]{2})-")  # S7-1200 / S7-1500 / S7-300 / S7-400 CPU family (CPUs only)
IO_S7 = re.compile(r"6ES7 ?(155|153|13[1-8])-")  # IM / ET200 I/O modules
RULES = [  # (source, key, regex on value, class, weight, label)
    ("lldp", "sys_desc", r"SIMATIC S7, (CPU|S7)", "PLC", 3, "LLDP: SIMATIC CPU"),
    ("lldp", "sys_desc", r"SIMATIC S7, IM ?155|ET ?200", "IO", 3, "LLDP: ET200/IM155"),
    ("lldp", "sys_desc", r"SIMATIC HMI|\bKTP\b|Comfort Panel", "HMI", 3, "LLDP: SIMATIC HMI"),
    ("lldp", "sys_desc", r"\+\s*engineering|PRONETA", "EWS", 2, "LLDP: PC + engineering / PRONETA"),
    ("lldp", "sys_desc", r"\+\s*HMI", "HMI", 2, "LLDP: PC + HMI"),
    ("lldp", "sys_name", r"MGate|NPort|Gateway", "IO", 3, "LLDP: protocol gateway"),
    ("lldp", "sys_desc", r"Cisco IOS|Catalyst|SCALANCE|switch|router", "OTHER", 3, "LLDP: network device"),
    ("lldp", "caps_enabled", r"router|wlan_ap", "OTHER", 3, "LLDP caps: router/AP"),
    ("pn_dcp", "type_of_station", r"S7-(300|400|1200|1500)|AXC|CPU", "PLC", 3, "DCP type: PLC/CPU"),
    ("pn_dcp", "type_of_station", r"ET ?200|AXL F|BK PN|IM ?155", "IO", 3, "DCP type: remote I/O"),
    ("pn_dcp", "type_of_station", r"SIMATIC-PC", "EWS", 2, "DCP type: SIMATIC PC station"),
    ("pn_dcp", "device_role", r"io_controller", "PLC", 3, "DCP role: IO controller"),
    ("pn_dcp", "device_role", r"^io_device$", "IO", 2, "DCP role: IO device only"),
    ("pn_dcp", "device_role", r"io_supervisor", "EWS", 3, "DCP role: IO supervisor"),
    ("enip_ident", "device_type", r":plc$", "PLC", 3, "CIP device type: PLC"),
    ("enip_ident", "device_type", r":hmi$", "HMI", 3, "CIP device type: HMI"),
    ("enip_ident", "device_type", r":(communications_adapter|general_purpose_discrete_io|ac_drive|dc_drive|generic_device_keyable)$", "IO", 3, "CIP device type: adapter/IO/drive"),
    ("enip_ident", "device_type", r":managed_ethernet_switch$", "OTHER", 3, "CIP device type: switch"),
    ("enip_ident", "product_name", r"^17(56|69|76|63|66|62|68)-L", "PLC", 2, "CIP product name: controller catalog"),
    ("enip_ident", "product_name", r"^1756-EN", "IO", 2, "CIP product name: EtherNet/IP bridge"),
    ("snmp", "sysDescr", r"SIMATIC S7, CPU|CPU-?1[0-9]{3}|M2[0-9]{2}|CJ2M|CP1[LH]|MicroLogix|PLC|ControlLogix", "PLC", 3, "SNMP sysDescr: controller"),
    ("snmp", "sysDescr", r"IM ?155|ET ?200|WAGO 750-3[0-9]{2}|fieldbus coupler", "IO", 3, "SNMP sysDescr: remote I/O / coupler"),
    ("snmp", "sysDescr", r"Cisco IOS|Catalyst|ProCurve|SCALANCE|switch|router", "OTHER", 3, "SNMP sysDescr: network device"),
    ("ftp", "banner", r"Camera|AXIS", "OTHER", 3, "FTP banner: camera"),
    ("ftp", "banner", r"vsFTPd|ProFTPD|Pure-FTPd", "OTHER", 1, "FTP banner: generic server"),
    ("ssh", "banner", r"OpenSSH.*(Ubuntu|Debian)|Bitvise|WinSSHD", "OTHER", 1, "SSH banner: generic server"),
    ("http", "server", r"Microsoft-IIS|Apache|nginx|Tomcat|GWS|lighttpd", "OTHER", 1, "HTTP server: generic web server"),
    ("telnet", "banner", r"IL[123]RMS|THD[IU]|Directional power|GOOSE_N|UL[LN]min|Ucomm|Prg[1-4]", "IED", 2, "telnet menu: protection-relay quantities"),
    ("ascii_kw", "match", r"^SEL-[0-9]{3}", "IED", 2, "string: SEL relay model"),
    ("ascii_kw", "match", r"^17(63|66|69|56|62)-L", "PLC", 2, "string: Rockwell controller catalog"),
    ("ascii_kw", "match", r"^(FX5U|CJ2[MH]|CP1[LH]|MELSEC|MicroLogix|ControlLogix|CompactLogix|S7-(300|400|1200|1500)|TM2[0-9]{2})", "PLC", 2, "string: PLC model"),
    ("ascii_kw", "match", r"SIMATIC-ROOT-ES|SIMATIC Automation Tool|SIMATIC NET PC", "EWS", 2, "string: SIMATIC engineering/PC"),
    ("ascii_kw", "match", r"SIMATIC-ROOT-HMI", "HMI", 2, "string: SIMATIC HMI station"),
    ("ascii_kw", "match", r"^(Cisco IOS|Hirschmann|RuggedCom|ROS-CF)", "OTHER", 1, "string: network vendor/OS"),
]
RULES = [(s, k, re.compile(p, re.I), c, w, lab) for s, k, p, c, w, lab in RULES]
LABEL_NAMES = {"PLC": "PLC/controller", "HMI": "HMI/SCADA/operator station", "EWS": "Engineering workstation/tool", "IED": "IED/RTU/protection relay",
               "IO": "IO/gateway/other field device", "OTHER": "khac/khong ro"}


def classify(items):
    votes, ev = defaultdict(float), defaultdict(list)
    seen = set()
    for src, key, val, _n in items:
        for s, k, rx, cls, w, lab in RULES:
            if s == src and k == key and rx.search(val) and (lab, val) not in seen:
                seen.add((lab, val))
                votes[cls] = max(votes[cls], w) + (0.25 if votes[cls] and (lab, val) not in ev[cls] else 0)  # strongest evidence + small bonus per extra item
                ev[cls].append((lab, val[:50]))
    kw = [v for s, k, v, _ in items if s == "ascii_kw"]
    cpu = any(CPU_S7.search(v) for v in kw)
    io = any(IO_S7.search(v) for v in kw)
    if cpu:
        votes["PLC"] = max(votes["PLC"], 2)
        ev["PLC"].append(("order number: S7 CPU", next(v for v in kw if CPU_S7.search(v))[:30]))
    elif io:
        votes["IO"] = max(votes["IO"], 2)
        ev["IO"].append(("order number: S7 I/O module (no CPU)", next(v for v in kw if IO_S7.search(v))[:30]))
    if not votes:
        return "none", 0, "", "", ""
    order = sorted(votes.items(), key=lambda kv: -kv[1])
    vs = ";".join(f"{c}:{v:g}" for c, v in order)
    top, sc = order[0]
    second = order[1][1] if len(order) > 1 else 0
    if sc < 2 or sc - second < 1:  # margin rule: a decision must not rest on the small per-extra-item bonus
        return "undecided", sc, "", " | ".join(f"{l}={v}" for c in votes for l, v in ev[c][:2])[:300], vs
    conf = "high" if sc >= 3 and second < 2 else "medium"
    return top, sc, conf, " | ".join(f"{l}={v}" for l, v in ev[top][:3])[:300], vs


def main():
    full = F / "phase2_identity_raw_v6_full.json"
    raw_path = full if full.exists() else F / "phase2_identity_raw_v6.json"  # full = every session (script 52); fallback = first pass (script 46)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))["records"]
    per = defaultdict(dict)
    for r in raw:
        for src, key, val, n in r["items"]:
            per[r["device_id"]][(src, key, val)] = n
    rows = []
    for dev, items in per.items():
        cls, sc, conf, ev, vs = classify([(a, b, c, n) for (a, b, c), n in items.items()])
        rows.append({"device_id": dev, "identity_type": cls, "score": sc, "confidence": conf if cls not in ("none", "undecided") else "", "evidence": ev, "votes_all_classes": vs,
                     "n_identity_items": len(items), "strong_sources": ",".join(sorted({a for a, _, _ in items if a != "ascii_kw"}))})
    T = pd.DataFrame(rows)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"].set_index("device_id_v6")
    T["seed_type"] = T.device_id.map(lab.device_type).fillna("")
    T["seed_ip"] = T.device_id.map(lab.ip).fillna("")
    T.to_csv(F / "phase2_identity_types_v6.csv", index=False, encoding="utf-8-sig")
    tmap = {"PLC": "PLC", "HMI": "HMI", "IO": "IO"}
    seeds = lab.reset_index().rename(columns={"device_id_v6": "device_id"})[["device_id", "device_type", "ip", "model"]].merge(T[["device_id", "identity_type", "confidence", "evidence"]], on="device_id", how="left")
    seeds["identity_type"] = seeds.identity_type.fillna("no_identity_items")
    dec = seeds[seeds.identity_type.isin(["PLC", "HMI", "EWS", "IED", "IO", "OTHER"])]
    ok = dec[dec.identity_type == dec.device_type]
    ann = pd.read_csv(F / "phase2_type_labeling_sheet_v6_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_confidence.isin(["high", "medium"])]
    ann["a"] = ann.gold_type.map({v: k for k, v in LABEL_NAMES.items()})
    m = ann.merge(T[T.identity_type.isin(LABEL_NAMES)][["device_id", "identity_type", "confidence"]], on="device_id")
    val = {"raw_input": raw_path.name, "seeds_total": int(len(seeds)), "seeds_with_any_identity": int((seeds.identity_type != "no_identity_items").sum()), "seeds_decided": int(len(dec)),
           "seeds_decided_correct": int(len(ok)), "seed_accuracy_on_decided": round(len(ok) / len(dec), 3) if len(dec) else None,
           "seed_rows": seeds.to_dict("records"),
           "type_counts_all_devices": T.identity_type.value_counts().to_dict(),
           "type_counts_decided_by_confidence": T[T.identity_type.isin(LABEL_NAMES)].groupby(["identity_type", "confidence"]).size().unstack(fill_value=0).to_dict(),
           "agreement_with_sonthai_review": {"n_overlap": int(len(m)), "agree": int((m.a == m.identity_type).sum()),
                                                "confusion": pd.crosstab(m.a, m.identity_type).to_dict() if len(m) else {}}}
    (F / "phase2_identity_validation_v6.json").write_text(json.dumps(val, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(seeds.to_string(index=False))
    print(json.dumps({k: v for k, v in val.items() if k != "seed_rows"}, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
