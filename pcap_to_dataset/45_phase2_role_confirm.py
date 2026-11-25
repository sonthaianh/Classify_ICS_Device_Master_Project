"""Phase 2 tier 1: confirm client/server role of each device with TCP handshakes, and compare with the port-ownership rule (script 44).

Evidence ladder (per device, OT-protocol service ports only):
  1. handshake : SYN sent by the device = it is the CLIENT of that connection; SYN-ACK sent by the device = it is the SERVER.
  2. port_rule : (script 44) the device owns the service port; used only when too few handshakes are visible (UDP protocols, flows that
                 started before the capture) -> weaker evidence.
Decision: handshake events >= MIN_HS -> role from handshake share (client >= 0.8, server <= 0.2, else both); otherwise the port rule,
marked 'port_rule_only'.  Agreement between the two evidence sources is reported (this is the 'confirmation').
Same device/session selection as script 42 (so profiles and roles describe the same traffic).
Outputs: phase2_handshake_roles_raw_v6.json, phase2_role_tier1_v6.csv, phase2_role_tier1_validation_v6.json.
Run: py -3.14 45_phase2_role_confirm.py [--workers 4]
"""
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s42", HERE / "42_phase2_payload_profiles.py")
S42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S42)
F, DS = S42.F, S42.DS
MIN_HS, CLI, SRV, CAP_ROWS = 3, 0.8, 0.2, 60000
OT_FAMS = [f for f in S42.FAM_PORTS]


def fam_of(port):
    return S42.PORT2FAM.get(port, "it_service" if port in S42.IT_PORTS else "other")


def process_file(job):
    session, targets = job
    out = []
    try:
        t = pq.read_table(DS / "raw_parquet" / f"{session}.parquet", columns=["src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "tcp_flags"])
        t = t.filter(pc.equal(t["ip_proto"], 6))
        fl = pc.bit_wise_and(pc.cast(t["tcp_flags"], pa.int32()), 0x12)
        syn = t.filter(pc.equal(fl, 0x02))
        synack = t.filter(pc.equal(fl, 0x12))
        for mac, dev in targets.items():
            rec = {"device_id": dev, "session_id": session, "syn_sent": Counter(), "synack_sent": Counter(), "peers_as_client": defaultdict(set), "peers_as_server": defaultdict(set)}
            s1 = syn.filter(pc.equal(syn["src_mac"], mac)).slice(0, CAP_ROWS)
            for dip, dp in zip(s1["dst_ip"].to_pylist(), s1["dst_port"].to_pylist()):
                f = fam_of(dp)
                rec["syn_sent"][f] += 1
                rec["peers_as_client"][f].add(dip)
            s2 = synack.filter(pc.equal(synack["src_mac"], mac)).slice(0, CAP_ROWS)
            for dip, sp in zip(s2["dst_ip"].to_pylist(), s2["src_port"].to_pylist()):
                f = fam_of(sp)
                rec["synack_sent"][f] += 1
                rec["peers_as_server"][f].add(dip)
            if rec["syn_sent"] or rec["synack_sent"]:
                out.append({"device_id": dev, "session_id": session, "syn_sent": dict(rec["syn_sent"]), "synack_sent": dict(rec["synack_sent"]),
                            "peers_as_client": {k: len(v) for k, v in rec["peers_as_client"].items()}, "peers_as_server": {k: len(v) for k, v in rec["peers_as_server"].items()}})
    except Exception as e:
        out.append({"device_id": None, "session_id": session, "error": f"{type(e).__name__}: {e}"})
    return out


def extract(workers):
    if os.environ.get("P2_ROLE_ALL") == "1":   # doc moi cap thiet bi-session (script 67), khong gioi han 3 session / 400 MB / 30 frame
        s67 = importlib.util.spec_from_file_location("s67", HERE / "67_phase2_payload_profiles_full.py")
        S67 = importlib.util.module_from_spec(s67)
        s67.loader.exec_module(S67)
        jobs, ids, n_cov = S67.build_jobs()
        n_ids = len(ids)
    else:
        jobs, n_ids, n_cov = S42.select_jobs()
    print(f"{len(jobs)} files, {n_cov}/{n_ids} devices", flush=True)
    res, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
            res.extend(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    errs = [r for r in res if r.get("error")]
    (F / "phase2_handshake_roles_raw_v6.json").write_text(json.dumps({"n_errors": len(errs), "errors": errs[:10], "records": [r for r in res if not r.get("error")]}), encoding="utf-8")
    print("extract done, errors:", len(errs))


def merge():
    raw = json.loads((F / "phase2_handshake_roles_raw_v6.json").read_text(encoding="utf-8"))["records"]
    hs = defaultdict(lambda: {"syn": Counter(), "synack": Counter(), "pc": Counter(), "ps": Counter(), "sessions": 0})
    for r in raw:
        h = hs[r["device_id"]]
        h["syn"].update(r["syn_sent"])
        h["synack"].update(r["synack_sent"])
        h["sessions"] += 1
        for k, v in r["peers_as_client"].items():
            h["pc"][k] = max(h["pc"][k], v)
        for k, v in r["peers_as_server"].items():
            h["ps"][k] = max(h["ps"][k], v)
    port = pd.read_csv(F / "phase2_role_labels_v6.csv", encoding="utf-8-sig").set_index("device_id")
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"].set_index("device_id_v6")
    rows = []
    for dev, p in port.iterrows():
        h = hs.get(dev)
        ot_syn = sum(v for k, v in h["syn"].items() if k in OT_FAMS) if h else 0
        ot_ack = sum(v for k, v in h["synack"].items() if k in OT_FAMS) if h else 0
        n_hs = ot_syn + ot_ack
        cshare = ot_syn / n_hs if n_hs else np.nan
        hs_role = "unknown" if n_hs < MIN_HS else ("client" if cshare >= CLI else ("server" if cshare <= SRV else "both"))
        pr = {"responder_like": "server", "initiator_like": "client", "mixed": "both", "no_ot_payload": "unknown"}[p.role_class]
        if hs_role != "unknown":
            final, ev = hs_role, ("handshake_confirms_port_rule" if pr == hs_role else ("handshake_only(no port-rule payload)" if pr == "unknown" else "handshake_overrides_port_rule"))
        elif pr != "unknown":
            final, ev = pr, "port_rule_only(weaker)"
        else:
            final, ev = "unknown", "no_evidence"
        fam = max(((k, v) for k, v in (h["syn"] + h["synack"]).items() if k in OT_FAMS), key=lambda kv: kv[1], default=("", 0))[0] if h else ""
        rows.append({"device_id": dev, "final_role": final, "evidence": ev, "handshake_role": hs_role, "port_rule_role": pr, "ot_syn_sent": ot_syn, "ot_synack_sent": ot_ack,
                     "ot_client_share_handshake": round(cshare, 3) if n_hs else np.nan, "port_rule_server_share": p.ot_server_share, "dominant_ot_family": p.dominant_ot_family or fam,
                     "ot_peers_when_client": max([v for k, v in h["pc"].items() if k in OT_FAMS], default=0) if h else 0,
                     "ot_peers_when_server": max([v for k, v in h["ps"].items() if k in OT_FAMS], default=0) if h else 0,
                     "all_syn_sent": sum(h["syn"].values()) if h else 0, "all_synack_sent": sum(h["synack"].values()) if h else 0,
                     "is_seed": dev in lab.index, "seed_type": lab.device_type.get(dev, ""), "seed_ip": lab.ip.get(dev, "")})
    T = pd.DataFrame(rows)
    T.to_csv(F / "phase2_role_tier1_v6.csv", index=False, encoding="utf-8-sig")
    both = T[(T.handshake_role != "unknown") & (T.port_rule_role != "unknown")]
    val = {"devices": int(len(T)), "final_role_counts": T.final_role.value_counts().to_dict(), "evidence_counts": T.evidence.value_counts().to_dict(),
           "devices_with_both_evidences": int(len(both)),
           "port_rule_vs_handshake_agreement": round(float((both.handshake_role == both.port_rule_role).mean()), 3) if len(both) else None,
           "confusion_handshake_rows_x_portrule_cols": pd.crosstab(both.handshake_role, both.port_rule_role).to_dict() if len(both) else {},
           "disagreements": both[both.handshake_role != both.port_rule_role][["device_id", "handshake_role", "port_rule_role", "ot_syn_sent", "ot_synack_sent", "port_rule_server_share", "dominant_ot_family", "seed_type"]].head(30).to_dict("records"),
           "seed_table": T[T.is_seed][["device_id", "seed_type", "seed_ip", "final_role", "evidence", "handshake_role", "port_rule_role", "ot_syn_sent", "ot_synack_sent", "dominant_ot_family"]].to_dict("records")}
    s = T[T.is_seed & T.seed_type.isin(["PLC", "HMI"]) & (T.final_role.isin(["client", "server", "both"]))]
    val["seed_plc_server_rate"] = round(float((s[s.seed_type == "PLC"].final_role == "server").mean()), 3) if (s.seed_type == "PLC").any() else None
    val["seed_hmi_client_rate"] = round(float((s[s.seed_type == "HMI"].final_role == "client").mean()), 3) if (s.seed_type == "HMI").any() else None
    ann = pd.read_csv(F / "phase2_type_labeling_sheet_v6_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_confidence.isin(["high", "medium"])].set_index("device_id").gold_type
    ns = T[~T.is_seed].set_index("device_id")
    val["vs_sonthai_review_final_role"] = {n: ns.loc[[i for i in ann[ann == n].index if i in ns.index]].final_role.value_counts().to_dict() for n in ann.unique()}
    (F / "phase2_role_tier1_validation_v6.json").write_text(json.dumps(val, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in val.items() if k not in ("seed_table",)}, indent=2, default=str))
    print(pd.DataFrame(val["seed_table"]).to_string(index=False))


if __name__ == "__main__":
    if "--merge-only" not in sys.argv:
        extract(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4)
    merge()
