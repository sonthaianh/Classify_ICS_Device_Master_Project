"""Phase 2: identity extraction over EVERY session of the in-scope devices (careful, resumable, memory-safe version of script 46).

Scope   : the 1,126 devices of the Phase-2 pool (+ lab seeds), every session in which they have windows (7,543 device-session pairs,
          1,473 raw files, ~31 GB).  Nothing is skipped for size: each parquet file is streamed in row batches.
Method  : same parsers as script 46 (imported, not copied).  Strong sources (LLDP, PROFINET DCP, ENIP ListIdentity, SNMP, HTTP/FTP/SSH/Telnet
          banners, Modbus device id, BACnet I-Am) are parsed on EVERY device-sent packet that can carry them (pre-selected by EtherType / source
          port); the weak ascii-keyword scan is applied to the first ASCII_CAP device-sent payload packets of each device-session
          (same cap as script 46, so the new result is a superset).
Resume  : one cache file per raw capture in Dataset/features/identity_cache_v6/ (written atomically); re-running skips finished captures.
Output  : Dataset/features/phase2_identity_raw_v6_full.json  (same record format as script 46 + coverage counters).
Run     : py -3.14 52_phase2_identity_full.py [--workers 3]
"""
import importlib.util
import json
import os
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
spec = importlib.util.spec_from_file_location("s46", HERE / "46_phase2_identity_extract.py")
S46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S46)
F, DS = S46.F, S46.DS
CACHE = F / "identity_cache_v6"
ASCII_CAP, BATCH = 40000, 100_000
IDENT_SRC_PORTS = [21, 22, 23, 80, 161, 443, 502, 8000, 8080, 44818, 47808]
COLS = ["src_mac", "eth_type", "ip_proto", "src_port", "dst_port", "payload_len", "app_payload"]


def process_file(job):
    session, targets = job
    out_path = CACHE / f"{session}.json"
    if out_path.exists():
        return {"session": session, "status": "cached"}
    t0 = time.time()
    try:
        pf = pq.ParquetFile(DS / "raw_parquet" / f"{session}.parquet")
        macs = pa.array(list(targets))
        items = {m: defaultdict(int) for m in targets}
        n_pay = {m: 0 for m in targets}      # device-sent payload packets seen
        n_strong = {m: 0 for m in targets}   # packets passed to the strong parsers
        n_ascii = {m: 0 for m in targets}    # packets passed to the ascii scan
        rows_total = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=COLS):
            tb = pa.Table.from_batches([batch])
            rows_total += tb.num_rows
            tb = tb.filter(pc.and_(pc.greater(tb["payload_len"], 0), pc.is_in(tb["src_mac"], value_set=macs)))
            if not tb.num_rows:
                continue
            smac = tb["src_mac"].to_pylist()
            eth = tb["eth_type"].to_pylist()
            proto = tb["ip_proto"].to_pylist()
            sp = tb["src_port"].to_pylist()
            dp = tb["dst_port"].to_pylist()
            pay = tb["app_payload"].to_pylist()
            for m, e, pr, s_, d_, p in zip(smac, eth, proto, sp, dp, pay):
                if not p:
                    continue
                n_pay[m] += 1
                strong_candidate = e in (0x88CC, 0x8892) or (s_ is not None and int(s_) in IDENT_SRC_PORTS)
                do_ascii = n_pay[m] <= ASCII_CAP
                if not strong_candidate and not do_ascii:
                    continue
                if strong_candidate:
                    n_strong[m] += 1
                if do_ascii:
                    n_ascii[m] += 1
                for it in S46.extract_packet(e, pr, s_, d_, p, ascii_kw=do_ascii):
                    items[m][tuple(it)] += 1
        recs = []
        for m, dev in targets.items():
            it = sorted(items[m].items(), key=lambda kv: (-kv[1], kv[0]))[:80]
            recs.append({"device_id": dev, "session_id": session, "items": [[a, b, c, n] for (a, b, c), n in it], "n_device_payload_pkts": n_pay[m],
                         "n_strong_scanned": n_strong[m], "n_ascii_scanned": n_ascii[m]})
        payload = {"session": session, "rows_total": rows_total, "seconds": round(time.time() - t0, 1), "records": recs}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, out_path)
        return {"session": session, "status": "done", "seconds": payload["seconds"]}
    except Exception as e:
        return {"session": session, "status": "error", "error": f"{type(e).__name__}: {e}"}


def build_jobs():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"]).drop_duplicates()
    pool = pd.read_csv(F / "phase2_candidate_pool_v6.csv", encoding="utf-8-sig", dtype=str)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    ids = sorted(set(pool.device_id) | set(lab[lab.gold_label == "OT"].device_id_v6))
    mac = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id").device_mac
    pairs = x[x.device_id.isin(ids)]
    jobs = defaultdict(dict)
    for r in pairs.itertuples():
        jobs[r.session_id][mac[r.device_id].lower()] = r.device_id
    size = {s: (DS / "raw_parquet" / f"{s}.parquet").stat().st_size for s in jobs}
    return sorted(jobs.items(), key=lambda kv: size[kv[0]]), len(ids), len(pairs)  # small files first


def merge():
    recs, errors = [], []
    jobs, _, _ = build_jobs()
    missing = []
    for s, _t in jobs:
        p = CACHE / f"{s}.json"
        if not p.exists():
            missing.append(s)
            continue
        recs.extend(json.loads(p.read_text(encoding="utf-8"))["records"])
    out = {"n_files_expected": len(jobs), "n_files_missing": len(missing), "missing": missing[:20], "ascii_cap": ASCII_CAP,
           "records": [r for r in recs if r["items"]], "records_all_incl_empty": len(recs)}
    (F / "phase2_identity_raw_v6_full.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"merged: {len(jobs) - len(missing)}/{len(jobs)} files; device-session records {len(recs)}; with items {len(out['records'])}; missing {len(missing)}")
    return len(missing)


def main():
    CACHE.mkdir(exist_ok=True)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    jobs, n_ids, n_pairs = build_jobs()
    todo = [j for j in jobs if not (CACHE / f"{j[0]}.json").exists()]
    print(f"{n_ids} devices, {n_pairs} device-session pairs, {len(jobs)} files; {len(jobs) - len(todo)} already cached, {len(todo)} to do", flush=True)
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
    print("DONE" if not errs and not miss else "INCOMPLETE - rerun to retry failed files")


if __name__ == "__main__":
    main()
