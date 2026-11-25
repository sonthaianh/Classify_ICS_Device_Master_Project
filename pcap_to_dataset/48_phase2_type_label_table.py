"""Phase 2 tier 2: consolidated type-label table + readiness check (no training).

Source priority (highest first): lab inventory (owner table, MAC verified)  >  payload identity high  >  payload identity medium.
sonthai-reviewed sheet labels (scenario/vendor based) are NOT merged into the training table; they are reported only as a cross-check.
Consistency check: identity-derived type vs tier-1 role (PLC/IO/IED expected 'server', HMI/EWS expected 'client'); a mismatch is reported, not
corrected (PLCs can be Modbus masters, HMIs can be slaves).
Readiness rule: >= 100 labelled devices over the five OT classes and >= 10 per class.
Outputs: phase2_type_labels_v6.csv, phase2_type_labels_summary_v6.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

F = Path(__file__).resolve().parent.parent / "Dataset/features"
CLASSES = ["PLC", "HMI", "EWS", "IED", "IO"]


def main():
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"]
    g1 = pd.DataFrame({"device_id": lab.device_id_v6, "type": lab.device_type, "source": "lab_inventory", "confidence": "high", "detail": lab.model})
    idt = pd.read_csv(F / "phase2_identity_types_v6.csv", encoding="utf-8-sig", dtype=str).fillna("")
    idt = idt[idt.identity_type.isin(CLASSES + ["OTHER"])]
    g2 = pd.DataFrame({"device_id": idt.device_id, "type": idt.identity_type, "source": "payload_identity", "confidence": idt.confidence, "detail": idt.evidence.str[:160]})
    g = pd.concat([g1, g2], ignore_index=True)
    g["rank"] = g.source.map({"lab_inventory": 0, "payload_identity": 1}) * 10 + g.confidence.map({"high": 0, "medium": 1})
    conflicts = []
    for dev, grp in g.groupby("device_id"):
        if grp.type.nunique() > 1:
            conflicts.append({"device_id": dev, "labels": dict(zip(grp.source + ":" + grp.confidence, grp.type))})
    g = g.sort_values("rank").drop_duplicates("device_id").drop(columns="rank")
    role = pd.read_csv(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    g["tier1_role"] = g.device_id.map(role.final_role).fillna("not_profiled")
    g["role_evidence"] = g.device_id.map(role.evidence).fillna("")
    # Controller/relay strings (order numbers, model names) emitted by a device that acts as TCP CLIENT without any self-description
    # (LLDP/DCP/SNMP/CIP) are most likely configuration transfers from an engineering tool, not the device's own identity -> excluded.
    strong = idt.set_index("device_id").strong_sources
    g["strong_sources"] = g.device_id.map(strong).fillna("")
    sus = (g.source == "payload_identity") & (g.strong_sources == "") & (g.tier1_role == "client") & g.type.isin(["PLC", "IO", "IED"])
    excluded = g[sus][["device_id", "type", "detail"]].to_dict("records")
    g.loc[sus, "type"] = "EXCLUDED_client_with_controller_strings"
    exp = {"PLC": "server", "IO": "server", "IED": "server", "HMI": "client", "EWS": "client"}
    chk = g[g.type.isin(exp) & g.tier1_role.isin(["server", "client", "both"])]
    g["role_consistent"] = [("" if t not in exp or r not in ("server", "client", "both") else ("yes" if r == exp[t] else ("both" if r == "both" else "no"))) for t, r in zip(g.type, g.tier1_role)]
    g.to_csv(F / "phase2_type_labels_v6.csv", index=False, encoding="utf-8-sig")
    ot = g[g.type.isin(CLASSES)]
    four = ot.type.replace({"IED": "FIELD_OTHER", "IO": "FIELD_OTHER"}).value_counts().to_dict()
    counts = ot.type.value_counts().reindex(CLASSES, fill_value=0).to_dict()
    by_src = ot.groupby(["type", "source"]).size().unstack(fill_value=0).to_dict()
    ann = pd.read_csv(F / "phase2_type_labeling_sheet_v6_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_confidence.isin(["high", "medium"])]
    m = ann.merge(g[["device_id", "type"]], on="device_id")
    amap = {"PLC/controller": "PLC", "HMI/SCADA/operator station": "HMI", "Engineering workstation/tool": "EWS", "IED/RTU/protection relay": "IED", "IO/gateway/other field device": "IO", "khac/khong ro": "OTHER"}
    m["a"] = m.gold_type.map(amap)
    summ = {"labelled_total_OT_classes": int(len(ot)), "per_class": counts, "per_class_by_source": by_src, "other_labelled(OTHER)": int((g.type == "OTHER").sum()),
            "four_class_counts(PLC,HMI,EWS,IED+IO)": four, "excluded_suspect_identity": excluded, "meets_total_100": bool(len(ot) >= 100), "meets_each_10": {c: bool(counts[c] >= 10) for c in CLASSES},
            "ready_for_5_class_training": bool(len(ot) >= 100 and all(counts[c] >= 10 for c in CLASSES)),
            "conflicts_between_sources": conflicts,
            "role_consistency": {"checked": int(len(chk)), "consistent": int((g.role_consistent == "yes").sum()), "both_roles": int((g.role_consistent == "both").sum()),
                                 "inconsistent": int((g.role_consistent == "no").sum()),
                                 "inconsistent_rows": g[g.role_consistent == "no"][["device_id", "type", "tier1_role", "role_evidence", "source"]].to_dict("records")},
            "cross_check_vs_sonthai_review": {"overlap": int(len(m)), "agree": int((m.a == m.type).sum()),
                                                 "disagree_rows": m[m.a != m.type][["device_id", "a", "type"]].rename(columns={"a": "sonthai", "type": "table"}).to_dict("records")}}
    (F / "phase2_type_labels_summary_v6.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
