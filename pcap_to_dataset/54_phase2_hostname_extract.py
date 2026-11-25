"""Phase 2: hostname evidence from name-service packets sent by the in-scope devices (all sessions, streaming, resumable).

Sources : DHCP (UDP 67/68: option 12 host name, option 60 vendor class), NBNS (UDP 137: NetBIOS name + suffix),
          LLMNR (UDP 5355) and mDNS (UDP 5353): DNS question / answer names.
Scope   : same device-session jobs as script 52 (1,473 captures).  Only packets SENT by the device (src_mac match).
Cache   : Dataset/features/hostname_cache_v6/{session}.json (atomic); output phase2_hostname_raw_v6.json
Item    : [source, kind, value, count]
Run     : py -3.14 54_phase2_hostname_extract.py [--workers 3] | --selftest
"""
import importlib.util
import json
import os
import struct
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s52", HERE / "52_phase2_identity_full.py")
S52 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S52)
F, DS = S52.F, S52.DS
CACHE = F / "hostname_cache_v6"
BATCH = 100_000
COLS = ["src_mac", "ip_proto", "src_port", "dst_port", "payload_len", "app_payload"]
NAME_PORTS = {67, 68, 137, 5353, 5355}


def _t(b):
    return b.decode("ascii", "replace").strip("\x00 \r\n\t")


def parse_dhcp(p):
    out = []
    if len(p) < 241 or p[236:240] != b"\x63\x82\x53\x63":
        return out
    i = 240
    while i < len(p):
        o = p[i]
        if o == 255:
            break
        if o == 0:
            i += 1
            continue
        if i + 1 >= len(p):
            break
        ln = p[i + 1]
        v = p[i + 2:i + 2 + ln]
        if o == 12:
            out.append(("dhcp", "hostname", _t(v)[:60]))
        elif o == 60:
            out.append(("dhcp", "vendor_class", _t(v)[:60]))
        i += 2 + ln
    return out


def parse_nbns(p):
    out = []
    if len(p) < 13 or p[12] != 0x20:
        return out
    enc = p[13:45]
    if len(enc) < 32:
        return out
    try:
        raw = bytes(((enc[2 * k] - 65) << 4) | (enc[2 * k + 1] - 65) for k in range(16))
    except Exception:
        return out
    name, suf = _t(raw[:15]), raw[15]
    if name and name.isprintable():
        out.append(("nbns", f"name_{suf:02x}", name[:40]))
    return out


def _labels(p, i, depth=0):
    parts = []
    while i < len(p) and depth < 10:
        n = p[i]
        if n == 0:
            return ".".join(parts), i + 1
        if n & 0xC0 == 0xC0:
            if i + 1 >= len(p):
                return None, i
            sub, _ = _labels(p, ((n & 0x3F) << 8) | p[i + 1], depth + 1)
            if sub:
                parts.append(sub)
            return ".".join(parts), i + 2
        parts.append(_t(p[i + 1:i + 1 + n]))
        i += 1 + n
    return None, i


def parse_dns(p, src):
    out = []
    if len(p) < 12:
        return out
    qd, an = struct.unpack(">HH", p[4:8])
    if qd > 10 or an > 20:
        return out
    i = 12
    names = []
    for _ in range(qd):
        nm, i = _labels(p, i)
        if nm is None:
            return out
        i += 4
        names.append(("query", nm))
    for _ in range(an):
        nm, i = _labels(p, i)
        if nm is None or i + 10 > len(p):
            break
        rdlen = struct.unpack(">H", p[i + 8:i + 10])[0]
        names.append(("answer", nm))
        i += 10 + rdlen
    for kind, nm in names:
        if nm and nm.isprintable():
            out.append((src, kind, nm[:80]))
    return out


def extract_packet(proto, sp, dp, p):
    if proto != 17 or not p:
        return []
    sp, dp = int(sp or 0), int(dp or 0)
    if sp in (67, 68) or dp in (67, 68):
        return parse_dhcp(p)
    if sp == 137 or dp == 137:
        return parse_nbns(p)
    if dp == 5355 or sp == 5355:
        return parse_dns(p, "llmnr")
    if dp == 5353 or sp == 5353:
        return parse_dns(p, "mdns")
    return []


def process_file(job):
    session, targets = job
    out_path = CACHE / f"{session}.json"
    if out_path.exists():
        return {"session": session, "status": "cached"}
    t0 = time.time()
    try:
        pf = pq.ParquetFile(DS / "raw_parquet" / f"{session}.parquet")
        macs = pa.array(list(targets))
        ports = pa.array(sorted(NAME_PORTS), type=pa.int64())
        items = {m: defaultdict(int) for m in targets}
        n_pk = {m: 0 for m in targets}
        for batch in pf.iter_batches(batch_size=BATCH, columns=COLS):
            tb = pa.Table.from_batches([batch])
            sp = tb["src_port"].cast(pa.int64())
            dp = tb["dst_port"].cast(pa.int64())
            m = pc.and_(pc.equal(tb["ip_proto"].cast(pa.int64()), 17),
                        pc.and_(pc.greater(tb["payload_len"], 0),
                                pc.and_(pc.is_in(tb["src_mac"], value_set=macs),
                                        pc.or_(pc.is_in(sp, value_set=ports), pc.is_in(dp, value_set=ports)))))
            tb = tb.filter(m)
            if not tb.num_rows:
                continue
            for mac, pr, s_, d_, p in zip(tb["src_mac"].to_pylist(), tb["ip_proto"].to_pylist(), tb["src_port"].to_pylist(),
                                          tb["dst_port"].to_pylist(), tb["app_payload"].to_pylist()):
                n_pk[mac] += 1
                for it in extract_packet(pr, s_, d_, p):
                    items[mac][tuple(it)] += 1
        recs = []
        for mac, dev in targets.items():
            it = sorted(items[mac].items(), key=lambda kv: (-kv[1], kv[0]))[:60]
            recs.append({"device_id": dev, "session_id": session, "n_name_pkts": n_pk[mac], "items": [[a, b, c, n] for (a, b, c), n in it]})
        payload = {"session": session, "seconds": round(time.time() - t0, 1), "records": recs}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, out_path)
        return {"session": session, "status": "done"}
    except Exception as e:
        return {"session": session, "status": "error", "error": f"{type(e).__name__}: {e}"}


def merge():
    jobs, _, _ = S52.build_jobs()
    recs, missing = [], []
    for s, _t_ in jobs:
        p = CACHE / f"{s}.json"
        if not p.exists():
            missing.append(s)
            continue
        recs.extend(json.loads(p.read_text(encoding="utf-8"))["records"])
    out = {"n_files_expected": len(jobs), "n_files_missing": len(missing), "records": [r for r in recs if r["items"]], "n_records_all": len(recs)}
    (F / "phase2_hostname_raw_v6.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"merged {len(jobs) - len(missing)}/{len(jobs)} files; records {len(recs)}; with names {len(out['records'])}; missing {len(missing)}")
    return len(missing)


def selftest():
    # DHCP
    d = bytearray(240)
    d[236:240] = b"\x63\x82\x53\x63"
    d += bytes([53, 1, 1, 12, 6]) + b"HMI-01" + bytes([60, 8]) + b"MSFT 5.0" + bytes([255])
    assert extract_packet(17, 68, 67, bytes(d)) == [("dhcp", "hostname", "HMI-01"), ("dhcp", "vendor_class", "MSFT 5.0")], extract_packet(17, 68, 67, bytes(d))
    # NBNS
    name = b"SCADA-PC".ljust(15) + b"\x00"
    enc = b"".join(bytes([65 + (c >> 4), 65 + (c & 15)]) for c in name)
    pk = bytes(12) + b"\x20" + enc + b"\x00" + bytes(4)
    assert extract_packet(17, 137, 137, pk) == [("nbns", "name_00", "SCADA-PC")], extract_packet(17, 137, 137, pk)
    # mDNS query
    q = struct.pack(">HHHHHH", 0, 0, 1, 0, 0, 0) + b"\x07plc-one\x05local\x00" + b"\x00\x01\x00\x01"
    assert extract_packet(17, 5353, 5353, q) == [("mdns", "query", "plc-one.local")], extract_packet(17, 5353, 5353, q)
    print("selftest OK")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    CACHE.mkdir(exist_ok=True)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    jobs, n_ids, n_pairs = S52.build_jobs()
    todo = [j for j in jobs if not (CACHE / f"{j[0]}.json").exists()]
    print(f"{n_ids} devices, {n_pairs} pairs, {len(jobs)} files; {len(jobs) - len(todo)} cached, {len(todo)} to do", flush=True)
    t0, errs = time.time(), []
    with Pool(workers, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, todo), 1):
            if r["status"] == "error":
                errs.append(r)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} files, {time.time() - t0:.0f}s, errors {len(errs)}", flush=True)
    for e in errs[:10]:
        print("ERROR", e)
    miss = merge()
    print("DONE" if not errs and not miss else "INCOMPLETE - rerun")


if __name__ == "__main__":
    main()
