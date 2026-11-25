"""Phase 2 tier 2: extract device IDENTITY evidence from payloads sent BY the device (independent of behaviour features).

Sources (all parsed from packet bytes, nothing from vendor OUI):
  lldp       : sys_name, sys_desc, capabilities (eth 0x88cc)
  pn_dcp     : PROFINET DCP identify/get responses: type_of_station, name_of_station, vendor/device id, DEVICE ROLE (eth 0x8892)
  enip_ident : EtherNet/IP ListIdentity: vendor id, CIP device type, product code, revision, product name (port 44818)
  snmp       : sysDescr / sysName (UDP 161 responses)
  http/ftp/telnet/ssh banners : Server header, <title>, 220 / SSH banners
  modbus_id  : Read Device Identification (FC 43/14) objects (port 502)
  bacnet_iam : I-Am vendor id (UDP 47808)
  ascii_kw   : product/catalog strings found in device-sent payloads (S7 order numbers 6ES7..., Rockwell catalog 17xx-..., SIMATIC, Modicon,
               SEL-..., MELSEC, ...) -- weaker evidence, reported separately
Selection: up to SESS_PER_DEV sessions per device (most frames), raw files <= 400 MB, any frame count >= 1.
Output: Dataset/features/phase2_identity_raw_v6.json  (records per device: list of [source, key, value]).
Run: py -3.14 46_phase2_identity_extract.py [--workers 4] ; parsers are unit-tested by `--selftest`.
"""
import importlib.util
import json
import re
import struct
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s42", HERE / "42_phase2_payload_profiles.py")
S42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S42)
F, DS = S42.F, S42.DS
SESS_PER_DEV, MAX_FILE, MAX_PKTS = 6, 400e6, 40000
LLDP_CAPS = ["other", "repeater", "bridge", "wlan_ap", "router", "telephone", "docsis", "station_only"]
CIP_TYPES = {0x00: "generic", 0x02: "ac_drive", 0x03: "dc_drive", 0x07: "general_purpose_discrete_io", 0x0C: "communications_adapter", 0x0E: "plc",
             0x10: "position_controller", 0x12: "motor_overload", 0x13: "limit_switch", 0x15: "proximity_sensor", 0x18: "hmi", 0x1A: "mass_flow_controller",
             0x1B: "pneumatic_valve", 0x1C: "vacuum_pressure_gauge", 0x1D: "process_control_valve", 0x22: "residual_gas_analyzer", 0x23: "dc_power_generator",
             0x24: "rfid_reader", 0x25: "bar_code_scanner", 0x26: "encoder", 0x27: "photoelectric_sensor", 0x28: "ethernet_ip_to_x_gateway(?)",
             0x2B: "generic_device_keyable", 0x2C: "managed_ethernet_switch", 0x2D: "cip_motion_drive", 0x32: "safety_discrete_io"}
KW = re.compile(rb"(6(?:ES7|AV2|AV7|GK7|ES5)[ \x00]?[0-9A-Z]{3}-[0-9A-Z]{4,5}-[0-9A-Z]{4}|\b17[0-9]{2}-[A-Z0-9]{3,12}\b|SIMATIC[ \-A-Za-z0-9]{0,24}|S7-(?:200|300|400|1200|1500)|"
                rb"MicroLogix[ 0-9A-Za-z]{0,10}|ControlLogix|CompactLogix|Modicon[ A-Za-z0-9]{0,12}|\bTM2[0-9]{2}[A-Z0-9]{3,10}\b|\bSEL-[0-9]{3,4}[A-Z0-9]{0,6}\b|"
                rb"MELSEC[ A-Za-z0-9\-]{0,10}|\bFX5U[A-Z0-9\-/]{0,12}\b|\bCJ2[MH]\b|\bCP1[LH]\b|\bCX[0-9]{4}\b|WAGO[ 0-9\-A-Za-z]{0,12}|Desigo|Niagara|JACE|"
                rb"Hirschmann|RuggedCom|ROS-CF|Moxa|MOXA|Cisco IOS|Linux|Windows|Ubuntu|Hewlett|PowerConnect)")


def _text(b):
    return b.decode("latin-1", "replace").strip("\x00 ").replace("\x00", " ")


def parse_lldp(p):
    out, i = [], 0
    while i + 2 <= len(p):
        h = (p[i] << 8) | p[i + 1]
        t, ln = h >> 9, h & 0x1FF
        v = p[i + 2:i + 2 + ln]
        i += 2 + ln
        if t == 0:
            break
        if t == 5:
            out.append(("lldp", "sys_name", _text(v)))
        elif t == 6:
            out.append(("lldp", "sys_desc", _text(v)))
        elif t == 7 and len(v) >= 4:
            caps = struct.unpack("!HH", v[:4])[1]  # enabled capabilities
            out.append(("lldp", "caps_enabled", ",".join(n for k, n in enumerate(LLDP_CAPS) if caps >> k & 1)))
    return out


def parse_dcp(p):
    out = []
    if len(p) < 12 or p[0] != 0xFE or p[1] not in (0xFF, 0xFD):
        return out
    if p[3] != 1:  # service type: 1 = response success
        return out
    dlen = (p[10] << 8) | p[11]
    i, end = 12, min(12 + dlen, len(p))
    while i + 4 <= end:
        o, so, ln = p[i], p[i + 1], (p[i + 2] << 8) | p[i + 3]
        d = p[i + 4:i + 4 + ln]
        i += 4 + ln + (ln & 1)
        d = d[2:] if o != 0xFF and len(d) >= 2 else d  # strip BlockInfo
        if o == 2 and so == 1:
            out.append(("pn_dcp", "type_of_station", _text(d)))
        elif o == 2 and so == 2:
            out.append(("pn_dcp", "name_of_station", _text(d)))
        elif o == 2 and so == 3 and len(d) >= 4:
            out.append(("pn_dcp", "vendor_device_id", f"{d[0]:02x}{d[1]:02x}:{d[2]:02x}{d[3]:02x}"))
        elif o == 2 and so == 4 and len(d) >= 1:
            r = d[0]
            names = [n for bit, n in ((1, "io_device"), (2, "io_controller"), (4, "io_multidevice"), (8, "io_supervisor")) if r & bit]
            out.append(("pn_dcp", "device_role", "+".join(names) or f"0x{r:02x}"))
    return out


def parse_enip_ident(p):
    if len(p) < 24 + 2 or (p[0] | p[1] << 8) != 0x0063 or len(p) < 24 + 2 + 4:
        return []
    try:
        pos = 24
        n = p[pos] | p[pos + 1] << 8
        pos += 2
        if n < 1 or (p[pos] | p[pos + 1] << 8) != 0x000C:
            return []
        pos += 4 + 2 + 16  # item type+len, enc proto version, sockaddr
        vendor, dtype, prod = struct.unpack("<HHH", p[pos:pos + 6])
        rev = (p[pos + 6], p[pos + 7])
        pos += 8 + 2 + 4
        nl = p[pos]
        name = _text(p[pos + 1:pos + 1 + nl])
        return [("enip_ident", "vendor_id", str(vendor)), ("enip_ident", "device_type", f"{dtype}:{CIP_TYPES.get(dtype, 'other')}"),
                ("enip_ident", "product_code", str(prod)), ("enip_ident", "revision", f"{rev[0]}.{rev[1]}"), ("enip_ident", "product_name", name)]
    except (IndexError, struct.error):
        return []


def _ber_string_after(p, oid_tail):
    k = p.find(oid_tail)
    if k < 0:
        return None
    j = k + len(oid_tail)
    if j + 2 > len(p) or p[j] != 0x04:
        return None
    ln = p[j + 1]
    s = j + 2
    if ln & 0x80:
        nb = ln & 0x7F
        ln = int.from_bytes(p[j + 2:j + 2 + nb], "big")
        s = j + 2 + nb
    return _text(p[s:s + ln])


def parse_snmp(p):
    out = []
    d = _ber_string_after(p, b"\x2b\x06\x01\x02\x01\x01\x01\x00")
    if d:
        out.append(("snmp", "sysDescr", d))
    n = _ber_string_after(p, b"\x2b\x06\x01\x02\x01\x01\x05\x00")
    if n:
        out.append(("snmp", "sysName", n))
    return out


def parse_http(p):
    if not p.startswith(b"HTTP/"):
        return []
    head, _, body = p.partition(b"\r\n\r\n")
    out = []
    m = re.search(rb"(?im)^Server:\s*(.+)$", head)
    if m:
        out.append(("http", "server", _text(m.group(1))[:100]))
    m = re.search(rb"(?is)<title>(.*?)</title>", body[:2000])
    if m:
        out.append(("http", "title", _text(m.group(1))[:100]))
    return out


def parse_modbus_id(p):
    if len(p) < 15 or p[7] != 0x2B or p[8] != 0x0E:
        return []
    out, n, i = [], p[13], 14
    names = {0: "vendor_name", 1: "product_code", 2: "revision", 3: "vendor_url", 4: "product_name", 5: "model_name", 6: "user_app_name"}
    for _ in range(n):
        if i + 2 > len(p):
            break
        oid, ln = p[i], p[i + 1]
        out.append(("modbus_id", names.get(oid, f"obj{oid}"), _text(p[i + 2:i + 2 + ln])))
        i += 2 + ln
    return out


def parse_bacnet_iam(p):
    if len(p) < 12 or p[0] != 0x81:
        return []
    try:
        pos = 6  # BVLC(4) + NPDU version, control
        ctl = p[5]
        if ctl & 0x80:
            return []
        if ctl & 0x20:
            pos += 3 + p[pos + 2] + 1
        if ctl & 0x08:
            pos += 3 + p[pos + 2]
        if p[pos] >> 4 != 1 or p[pos + 1] != 0:
            return []
        q = pos + 2
        if p[q] != 0xC4:
            return []
        q += 5
        q += 1 + (p[q] & 7)  # max APDU
        q += 2  # segmentation
        ln = p[q] & 7
        vid = int.from_bytes(p[q + 1:q + 1 + ln], "big")
        return [("bacnet_iam", "vendor_id", str(vid))]
    except IndexError:
        return []


def parse_banner(p, sport):
    if sport == 22 and p.startswith(b"SSH-"):
        return [("ssh", "banner", _text(p.split(b"\n")[0])[:100])]
    if sport in (21, 23, 25) and len(p) >= 4 and p[:4].isascii():
        line = _text(p.split(b"\n")[0])
        if sport == 21 and line.startswith("220"):
            return [("ftp", "banner", line[:100])]
        if sport == 23 and len(line) > 3 and line.isprintable():
            return [("telnet", "banner", line[:100])]
    return []


def extract_packet(eth, proto, sp, dp, p, ascii_kw=True):
    out = []
    if eth == 0x88CC:
        return parse_lldp(p)
    if eth == 0x8892:
        return parse_dcp(p)
    if sp is None:
        return out
    sp = int(sp)
    if sp == 44818:
        out += parse_enip_ident(p)
    if proto == 17 and sp == 161:
        out += parse_snmp(p)
    if proto == 6 and sp in (80, 8080, 8000, 443):
        out += parse_http(p)
    if proto == 6 and sp == 502:
        out += parse_modbus_id(p)
    if proto == 17 and sp == 47808:
        out += parse_bacnet_iam(p)
    if proto == 6 and sp in (21, 22, 23):
        out += parse_banner(p, sp)
    if ascii_kw and len(p) >= 12:
        for m in KW.finditer(p[:1500]):
            out.append(("ascii_kw", "match", _text(m.group(0))[:60]))
    return out


def process_file(job):
    session, targets = job
    res = []
    try:
        t = pq.read_table(DS / "raw_parquet" / f"{session}.parquet", columns=["src_mac", "eth_type", "ip_proto", "src_port", "dst_port", "payload_len", "app_payload"])
        t = t.filter(pc.and_(pc.is_in(t["src_mac"], value_set=pa.array(list(targets))), pc.greater(t["payload_len"], 0)))
        for mac, dev in targets.items():
            s = t.filter(pc.equal(t["src_mac"], mac)).slice(0, MAX_PKTS)
            if not s.num_rows:
                continue
            seen = {}
            for eth, proto, sp, dp, pay in zip(s["eth_type"].to_pylist(), s["ip_proto"].to_pylist(), s["src_port"].to_pylist(), s["dst_port"].to_pylist(), s["app_payload"].to_pylist()):
                if not pay:
                    continue
                for it in extract_packet(eth, proto, sp, dp, pay):
                    k = tuple(it)
                    seen[k] = seen.get(k, 0) + 1
            if seen:
                res.append({"device_id": dev, "session_id": session, "items": [[a, b, c, n] for (a, b, c), n in sorted(seen.items(), key=lambda kv: -kv[1])[:60]]})
    except Exception as e:
        res.append({"device_id": None, "session_id": session, "error": f"{type(e).__name__}: {e}"})
    return res


def select_jobs():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "frames_sent"])
    pool = pd.read_csv(F / "phase2_candidate_pool_v6.csv", encoding="utf-8-sig", dtype=str)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    ids = sorted(set(pool.device_id) | set(lab[lab.gold_label == "OT"].device_id_v6))
    mac = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id").device_mac
    ds = x[x.device_id.isin(ids)].groupby(["device_id", "session_id"]).frames_sent.sum().reset_index()
    ds["bytes"] = ds.session_id.map(lambda s: (DS / "raw_parquet" / f"{s}.parquet").stat().st_size)
    ds = ds[ds.bytes <= MAX_FILE].sort_values(["device_id", "frames_sent"], ascending=[True, False]).groupby("device_id").head(SESS_PER_DEV)
    jobs = defaultdict(dict)
    for r in ds.itertuples():
        jobs[r.session_id][mac[r.device_id].lower()] = r.device_id
    return sorted(jobs.items(), key=lambda kv: -len(kv[1])), len(ids), ds.device_id.nunique()


def selftest():
    import glob
    t = pq.read_table(glob.glob(str(DS / "raw_parquet/S1549.parquet"))[0], columns=["eth_type", "app_payload"])
    lld = [p for e, p in zip(t["eth_type"].to_pylist(), t["app_payload"].to_pylist()) if e == 0x88CC][:1]
    print("LLDP:", parse_lldp(lld[0]))
    t2 = pq.read_table(glob.glob(str(DS / "raw_parquet/S1309.parquet"))[0], columns=["eth_type", "app_payload"])
    print("LLDP(Cisco):", [x for p in [p for e, p in zip(t2["eth_type"].to_pylist(), t2["app_payload"].to_pylist()) if e == 0x88CC][:1] for x in parse_lldp(p)])
    n_dcp = 0
    for e, p in zip(t["eth_type"].to_pylist(), t["app_payload"].to_pylist()):
        if e == 0x8892 and p and p[0] == 0xFE and p[1] == 0xFF:
            r = parse_dcp(p)
            if r:
                n_dcp += 1
                if n_dcp <= 2:
                    print("DCP:", r)
    print("DCP responses parsed:", n_dcp)
    # synthetic checks
    ident = bytes([0x63, 0, 0, 0] + [0] * 20) + bytes([1, 0, 0x0C, 0, 0, 0, 1, 0]) + bytes(16) + struct.pack("<HHH", 1, 14, 55) + bytes([20, 11]) + bytes(2) + bytes(4) + bytes([5]) + b"1756-" + bytes([3])
    ident = ident[:24] + bytes([1, 0, 0x0C, 0, 0x30, 0]) + bytes([1, 0]) + bytes(16) + struct.pack("<HHH", 1, 14, 55) + bytes([20, 11]) + bytes(2) + bytes(4) + bytes([5]) + b"1756-" + bytes([3])
    print("ENIP ident:", parse_enip_ident(ident))
    snmp = b"\x30\x20\x2b\x06\x01\x02\x01\x01\x01\x00\x04\x0bCisco Cat29"
    print("SNMP:", parse_snmp(snmp))
    mb = bytes(7) + bytes([0x2B, 0x0E, 1, 1, 0, 0, 2, 0, 3]) + b"ABC" + bytes([1, 2]) + b"XY"
    print("Modbus ID:", parse_modbus_id(mb))
    print("KW:", [m.group(0) for m in KW.finditer(b"..6ES7 211-1AE40-0XB0  S V-P4..1756-L61 SIMATIC S7-1200")])


def main():
    if "--selftest" in sys.argv:
        return selftest()
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    jobs, n_ids, n_cov = select_jobs()
    print(f"{len(jobs)} files, {n_cov}/{n_ids} devices", flush=True)
    res, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
            res.extend(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    errs = [r for r in res if r.get("error")]
    (F / "phase2_identity_raw_v6.json").write_text(json.dumps({"n_files": len(jobs), "n_errors": len(errs), "errors": errs[:10], "records": [r for r in res if not r.get("error")]}), encoding="utf-8")
    print("done; errors", len(errs), "| records", len(res) - len(errs))


if __name__ == "__main__":
    main()
