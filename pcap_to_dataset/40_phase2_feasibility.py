"""Phase 2 step 1 (feasibility, NO training): candidate pool + type-labelling sheet.

Pool = devices Phase 1 calls OT (OOF LightGBM score >= 0.5, script 36).  Phase 2 will classify device type on these, with an extra
class 'khac/khong ro' that absorbs Phase-1 false positives (scanners, IT hosts, infrastructure).
Outputs (Dataset/features): phase2_candidate_pool_v6.csv (all pool devices, with evidence summary),
phase2_type_labeling_sheet_v6.csv (stratified blind sheet for the owner), phase2_feasibility_v6.json.
The sheet hides the Phase-1 score and rule label so the type judgement is not anchored by them.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
SEED, N_PER_BUCKET = 20261109, 20
BUCKETS = [("siemens_s7", r"s7comm|cotp|\b102\b"), ("modbus", r"modbus|mbtcp|\b502\b"), ("ethernet_ip_cip", r"enip|cip|44818|2222"),
           ("bacnet", r"bacnet|bacapp|bvlc|47808"), ("iec61850_goose_mms", r"goose|mms|\bsv\b|iec61850"),
           ("telecontrol_dnp3_104", r"dnp3|iec60870|\b20000\b|2404"), ("profinet_other_fieldbus", r"profinet|epl|esio|omron|9600|34964|18246"),
           ("no_decoded_protocol_or_port", r"^$")]
TAXONOMY = ["PLC/controller", "HMI/SCADA/operator station", "Engineering workstation/tool", "IED/RTU/protection relay", "IO/gateway/other field device", "khac/khong ro"]


def bucket(txt):
    for name, pat in BUCKETS:
        if re.search(pat, txt, re.I):
            return name
    return "other_protocol_or_port"


def main():
    oof = pd.read_csv(F / "phase1_oof_scores_v6.csv").set_index("device_id")
    lab = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).fillna("").set_index("device_id")
    scaf = pd.read_csv(F.parent / "metadata/devices_mac_scaffold.csv", encoding="utf-8-sig", dtype=str).drop_duplicates("mac").set_index("mac")
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "pkts_per_flow", "udp_ratio", "packet_rate", "svc_port_unique"])
    sess = x.drop_duplicates(["device_id", "session_id"]).groupby("device_id").session_id.apply(lambda s: " | ".join(sorted(s)[:4]))
    med = x.groupby("device_id")[["pkts_per_flow", "udp_ratio", "packet_rate", "svc_port_unique"]].median().round(2)

    pool_ids = oof.index[oof.lightgbm >= 0.5]
    p = lab.loc[pool_ids, ["device_mac", "label", "evidence_tier", "ot_protocols", "ot_peer_ports", "ot_local_ports", "flood_like"]].copy()
    p["phase1_score"] = oof.loc[pool_ids, "lightgbm"].round(3)
    p["vendor_oui_unverified"] = p.device_mac.map(scaf.vendor)
    p["top_sessions"] = sess.reindex(pool_ids).values
    p = p.join(med)
    p["bucket"] = (p.ot_protocols + ";" + p.ot_peer_ports + ";" + p.ot_local_ports).map(lambda t: bucket(t.strip(";")))
    p["port_side"] = np.where((p.ot_peer_ports != "") & (p.ot_local_ports != ""), "both",
                              np.where(p.ot_local_ports != "", "device_listens", np.where(p.ot_peer_ports != "", "device_initiates", "none")))
    p.reset_index().to_csv(F / "phase2_candidate_pool_v6.csv", index=False, encoding="utf-8-sig")

    lab_gold = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab_gold = lab_gold[lab_gold.gold_label == "OT"]
    in_pool = lab_gold.device_id_v6.isin(pool_ids)
    # stratified blind sheet (lab-known devices excluded: their type is already known)
    rng = np.random.default_rng(SEED)
    cand = p[~p.index.isin(lab_gold.device_id_v6)]
    parts = [g.sample(min(len(g), N_PER_BUCKET), random_state=int(rng.integers(1 << 30))) for _, g in cand.groupby("bucket")]
    sh = pd.concat(parts).sample(frac=1, random_state=SEED)
    sheet = sh.reset_index()[["device_id", "device_mac", "vendor_oui_unverified", "top_sessions", "ot_protocols", "ot_peer_ports", "ot_local_ports",
                              "pkts_per_flow", "udp_ratio", "packet_rate", "svc_port_unique"]].rename(columns={"device_id": "device_id"})
    for c, v in (("gold_type", ""), ("gold_basis", ""), ("gold_confidence", ""), ("notes", "")):
        sheet[c] = v
    sheet.to_csv(F / "phase2_type_labeling_sheet_v6.csv", index=False, encoding="utf-8-sig")

    lab_types = lab_gold.groupby("device_type").size().to_dict()
    res = {"pool_size_threshold_0.5": int(len(pool_ids)), "pool_rule_OT": int((p.label == "OT_confirmed").sum()), "pool_rule_unknown": int((p.label != "OT_confirmed").sum()),
           "pool_by_bucket": p.bucket.value_counts().to_dict(), "pool_by_port_side": p.port_side.value_counts().to_dict(),
           "type_labels_available": {"devices_csv_ground_truth_device_type_nonempty": 0, "lab_inventory_types": lab_types, "lab_in_pool": int(in_pool.sum()),
                                     "lab_total": int(len(lab_gold))},
           "taxonomy_proposed": TAXONOMY, "labeling_sheet_rows": int(len(sheet)), "labeling_sheet_by_bucket": sh.bucket.value_counts().to_dict(),
           "min_labels_rule_of_thumb": "about 30-50 labelled devices per class for a stable baseline -> 150-250 for 5 classes + 'khac'"}
    (F / "phase2_feasibility_v6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
