"""Phase 2: payload communication profiles per device (protocol family x role, message-type tokens, payload sizes, peers).

Scope: the 22 lab reference devices (PLC/HMI/IO, owner-provided) + the 1,123 devices Phase 1 flags OT (+ lab devices not flagged).
Per device: up to 3 sessions with the most packets among raw files <= 400 MB; up to MAX_PKTS payload-carrying TCP/UDP packets per
device-session (chronological).  Protocol parsing is shallow on purpose (header bytes only) for S7comm/COTP, Modbus/TCP, EtherNet/IP,
BACnet/IP, DNP3, IEC 60870-5-104, Omron FINS; every other packet only counts toward its port-based family and role.
Role: owner of the service port (known OT/service port, else lower port < 1024, else higher port = client).
Output: Dataset/features/phase2_payload_profiles_v6.json (counts per device; vectors are built in script 43).
Run:  py -3.14 42_phase2_payload_profiles.py [--limit N] [--workers 4]
"""
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "Dataset"
F = DS / "features"
MAX_PKTS, MAX_FILE_BYTES, SESS_PER_DEV = 20000, 400e6, 3

FAM_PORTS = {"s7_iso": [102], "modbus": [502, 503], "enip": [44818, 2222], "bacnet": [47808], "dnp3": [20000, 20547], "iec104": [2404],
             "profinet_ethercat": [34962, 34963, 34964, 34980], "omron_fins": [9600], "fox_niagara": [1911, 1962, 4911, 41100], "opcua": [4840, 4843],
             "mqtt": [1883, 8883], "codesys_ge_misc": [2455, 11740, 851, 18245, 18246, 5007, 48898, 4000, 4001, 4002, 55000, 55001, 55003, 1089, 1090, 1091, 1541, 789, 2000, 4712, 3671, 5094]}
IT_PORTS = {80: "http", 443: "tls", 8080: "http", 53: "dns", 123: "ntp", 22: "ssh", 23: "telnet", 21: "ftp", 25: "smtp", 445: "smb", 139: "smb", 137: "nbns", 138: "nbns",
            3389: "rdp", 161: "snmp", 162: "snmp", 5353: "mdns", 1900: "ssdp", 5355: "llmnr", 67: "dhcp", 68: "dhcp", 389: "ldap", 88: "kerberos", 514: "syslog", 135: "msrpc"}
PORT2FAM = {p: f for f, ps in FAM_PORTS.items() for p in ps}
FAMS = list(FAM_PORTS) + ["it_service", "unknown"]


def family_and_role(sp, dp, sender_is_dev):
    """Return (family, device_role) where device_role in {'S','C'} (device is server / client of this flow)."""
    def known(p):
        return p in PORT2FAM or p in IT_PORTS
    if known(sp) and not known(dp):
        svc, sender_is_server = sp, True
    elif known(dp) and not known(sp):
        svc, sender_is_server = dp, False
    elif known(sp) and known(dp):
        if sp in PORT2FAM and dp not in PORT2FAM:
            svc, sender_is_server = sp, True
        elif dp in PORT2FAM and sp not in PORT2FAM:
            svc, sender_is_server = dp, False
        else:
            svc, sender_is_server = (sp, True) if sp < dp else (dp, False)
    else:
        lo, hi = min(sp, dp), max(sp, dp)
        if lo < 1024:
            svc, sender_is_server = lo, sp == lo
        else:
            svc, sender_is_server = None, sp < dp  # higher port = client
    fam = PORT2FAM.get(svc, "it_service" if svc in IT_PORTS else "unknown")
    dev_is_server = sender_is_server if sender_is_dev else (not sender_is_server)
    return fam, ("S" if dev_is_server else "C")


def msg_token(fam, b, sender_is_server, tcp):
    """Shallow message-type token from header bytes; None if not parseable. req/resp refers to the sender's role."""
    n = len(b)
    d = "resp" if sender_is_server else "req"
    try:
        if fam == "s7_iso" and tcp and n >= 7 and b[0] == 3:
            li = b[4]
            pdu = b[5] if n > 5 else 0
            if pdu == 0xE0:
                return "iso_cr"
            if pdu == 0xD0:
                return "iso_cc"
            if pdu == 0xF0:
                s = 5 + li
                if n > s + 1 and b[s] == 0x32:
                    ros = b[s + 1]
                    if ros in (1, 3) and n > s + 10 + (2 if ros == 3 else 0):
                        fc = b[s + 10 + (2 if ros == 3 else 0)]
                        return f"s7_{'job' if ros == 1 else 'ackdata'}_fc{fc:02x}"
                    return f"s7_rosctr{ros}"
                if n > s and b[s] == 0x72:
                    return "s7plus"
                return "iso_data_other(mms?)"
            return "tpkt_other"
        if fam == "modbus" and tcp and n >= 8 and b[2] == 0 and b[3] == 0:
            fc = b[7]
            return f"mb_{d}_exc" if fc >= 0x80 else f"mb_{d}_fc{fc:02x}"
        if fam == "enip" and n >= 24 and tcp:
            cmd = b[0] | (b[1] << 8)
            return f"enip_cmd{cmd:04x}_{d}"
        if fam == "enip" and not tcp:
            return "enip_udp_io" if n < 200 else "enip_udp_other"
        if fam == "bacnet" and n >= 6 and b[0] == 0x81:
            fn = b[1]
            ctl = b[5]
            if ctl & 0x80:
                return f"bac_f{fn:02x}_nlm"
            pos = 6
            if ctl & 0x20:
                pos += 3 + b[pos + 2]
                pos += 0
            if ctl & 0x08:
                pos += 3 + b[pos + 2]
            if ctl & 0x20:
                pos += 1
            at = b[pos] >> 4
            if at == 0 and n > pos + 3:
                return f"bac_creq_svc{b[pos + 3]:02x}"
            if at == 1 and n > pos + 1:
                return f"bac_ureq_svc{b[pos + 1]:02x}"
            return f"bac_apdu{at}"
        if fam == "dnp3" and n >= 13 and b[0] == 5 and b[1] == 0x64:
            return f"dnp3_{d}_fc{b[12]:02x}"
        if fam == "dnp3" and n >= 10 and b[0] == 5 and b[1] == 0x64:
            return "dnp3_link"
        if fam == "iec104" and n >= 6 and b[0] == 0x68:
            c = b[2]
            if c & 1 == 0:
                return f"iec104_I_t{b[6]:02x}" if n > 6 else "iec104_I"
            return "iec104_S" if c & 3 == 1 else "iec104_U"
        if fam == "omron_fins" and n >= 12 and not tcp:
            return f"fins_{b[10]:02x}{b[11]:02x}_{d}"
    except IndexError:
        return None
    return None


def profile_device(rows, mac):
    """rows: list of tuples (src_mac,dst_mac,src_ip,dst_ip,sp,dp,proto,plen,payload)."""
    fam_role, tokens = Counter(), Counter()
    tx_len, rx_len = [], []
    peers_srv, peers_cli = set(), set()
    n = 0
    for sm, dm, sip, dip, sp, dp, proto, plen, pay in rows:
        if sp is None or dp is None:
            continue
        sender_is_dev = sm == mac
        fam, role = family_and_role(int(sp), int(dp), sender_is_dev)
        n += 1
        fam_role[f"{fam}|{role}"] += 1
        (tx_len if sender_is_dev else rx_len).append(plen)
        peer = dip if sender_is_dev else sip
        (peers_cli if role == "S" else peers_srv).add(peer)  # role S: peers are clients of the device
        if pay:
            sender_is_server = (role == "S") == sender_is_dev
            t = msg_token(fam, pay, sender_is_server, proto == 6)
            if t:
                tokens[f"{role}:{'tx' if sender_is_dev else 'rx'}:{t}"] += 1
    q = lambda a, p: float(np.percentile(a, p)) if a else None
    return {"n_pkts": n, "fam_role": dict(fam_role), "tokens": dict(tokens), "tx_n": len(tx_len), "rx_n": len(rx_len),
            "tx_len_med": q(tx_len, 50), "tx_len_p90": q(tx_len, 90), "rx_len_med": q(rx_len, 50), "rx_len_p90": q(rx_len, 90),
            "n_clients_of_device": len(peers_cli), "n_servers_contacted": len(peers_srv)}


def process_file(job):
    session, targets = job  # targets: {mac: device_id}
    path = DS / "raw_parquet" / f"{session}.parquet"
    out = []
    try:
        t = pq.read_table(path, columns=["src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "payload_len", "app_payload"])
        m = pc.and_(pc.greater(t["payload_len"], 0), pc.is_in(t["ip_proto"], value_set=__import__("pyarrow").array([6, 17], type=t["ip_proto"].type)))
        t = t.filter(m)
        for mac, dev in targets.items():
            sel = t.filter(pc.or_(pc.equal(t["src_mac"], mac), pc.equal(t["dst_mac"], mac)))
            if sel.num_rows == 0:
                continue
            sel = sel.slice(0, MAX_PKTS)
            cols = [sel[c].to_pylist() for c in ("src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "payload_len", "app_payload")]
            prof = profile_device(list(zip(*cols)), mac)
            prof.update(device_id=dev, session_id=session, n_rows_total=int(sel.num_rows))
            out.append(prof)
    except Exception as e:  # keep going; record failure
        out.append({"device_id": None, "session_id": session, "error": f"{type(e).__name__}: {e}"})
    return out


def select_jobs():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "frames_sent"])
    pool = pd.read_csv(F / "phase2_candidate_pool_v6.csv", encoding="utf-8-sig", dtype=str)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"]
    mac = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id").device_mac
    ids = sorted(set(pool.device_id) | set(lab.device_id_v6))
    ds = x[x.device_id.isin(ids)].groupby(["device_id", "session_id"]).frames_sent.sum().reset_index()
    ds["bytes"] = ds.session_id.map(lambda s: (DS / "raw_parquet" / f"{s}.parquet").stat().st_size)
    ds = ds[(ds.frames_sent >= 30) & (ds.bytes <= MAX_FILE_BYTES)]
    ds = ds.sort_values(["device_id", "frames_sent"], ascending=[True, False]).groupby("device_id").head(SESS_PER_DEV)
    jobs = defaultdict(dict)
    for r in ds.itertuples():
        jobs[r.session_id][mac[r.device_id].lower()] = r.device_id
    return sorted(jobs.items(), key=lambda kv: -len(kv[1])), len(ids), ds.device_id.nunique()


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    jobs, n_ids, n_cov = select_jobs()
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} files, {n_cov}/{n_ids} devices covered", flush=True)
    res, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
            res.extend(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files, {len(res)} device-sessions, {time.time() - t0:.0f}s", flush=True)
    errs = [r for r in res if r.get("error")]
    out = {"params": {"MAX_PKTS": MAX_PKTS, "MAX_FILE_BYTES": MAX_FILE_BYTES, "SESS_PER_DEV": SESS_PER_DEV}, "n_files": len(jobs), "n_errors": len(errs),
           "errors": errs[:20], "records": [r for r in res if not r.get("error")]}
    (F / ("phase2_payload_profiles_v6.json" if not limit else "phase2_payload_profiles_v6_TEST.json")).write_text(json.dumps(out), encoding="utf-8")
    print("done; errors:", len(errs), "| device-sessions:", len(out["records"]))


if __name__ == "__main__":
    main()
