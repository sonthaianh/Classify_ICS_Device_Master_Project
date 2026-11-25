"""Phase 2: payload profiles over EVERY session and EVERY packet of the in-scope devices (full re-read, 2026-11-11).

Replaces the limits of script 42 (kept untouched as pass 1): no 3-session cap, no 400 MB file cap, no frames_sent >= 30 filter, no MAX_PKTS cap.
Scope   : same 1,126 devices as script 52 (Phase-2 pool + lab OT seeds), every (device, session) pair in the v6 windows (7,543 pairs, 1,473 raw files, ~31 GB).
Method  : script 42's own functions are imported, not copied (family_and_role, msg_token, same record format), so the result is a strict superset:
          for any device-session that script 42 read completely (n_rows_total < 20,000) the new record must be IDENTICAL (checked by --verify).
          Files are streamed in row batches (memory-safe even for the 2.1 GB capture); payload bytes are cut to the first 300 bytes (msg_token never
          reads beyond byte ~272); payload-length percentiles are computed exactly from a length histogram (same linear interpolation as np.percentile).
Resume  : one cache file per capture in Dataset/features/payload_full_cache_v6/ (atomic write); re-running skips finished captures.
Output  : Dataset/features/phase2_payload_profiles_v6_full.json (same record format as script 42 + coverage fields). Script 42's file is NOT overwritten.
Run     : py -3.14 67_phase2_payload_profiles_full.py [--workers 6] [--limit-files N]   |  --merge-only  |  --verify
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
CACHE = F / "payload_full_cache_v6"
OUT = F / "phase2_payload_profiles_v6_full.json"
BATCH, CUT = 100_000, 300
COLS = ["src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "payload_len", "app_payload"]


def hist_percentile(h, p):
    """Exact np.percentile(method='linear') from a {value: count} histogram."""
    if not h:
        return None
    vals = np.array(sorted(h), dtype=float)
    cnt = np.array([h[v] for v in sorted(h)], dtype=np.int64)
    cum = np.cumsum(cnt)
    n = int(cum[-1])
    pos = p / 100.0 * (n - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    vlo = vals[np.searchsorted(cum, lo, side="right")]
    vhi = vals[np.searchsorted(cum, hi, side="right")]
    return float(vlo + (vhi - vlo) * (pos - lo))


def process_file(job):
    session, targets = job  # targets: {mac: device_id}
    out_path = CACHE / f"{session}.json"
    if out_path.exists():
        return {"session": session, "status": "cached"}
    t0 = time.time()
    try:
        pf = pq.ParquetFile(DS / "raw_parquet" / f"{session}.parquet")
        macs = pa.array(list(targets))
        A = {m: {"fam_role": Counter(), "tokens": Counter(), "tx": Counter(), "rx": Counter(), "ps": set(), "pc": set(), "n": 0, "rows": 0} for m in targets}
        rows_total = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=COLS):
            tb = pa.Table.from_batches([batch])
            rows_total += tb.num_rows
            keep = pc.and_(pc.greater(tb["payload_len"], 0), pc.is_in(tb["ip_proto"], value_set=pa.array([6, 17], type=tb["ip_proto"].type)))
            keep = pc.and_(keep, pc.or_(pc.is_in(tb["src_mac"], value_set=macs), pc.is_in(tb["dst_mac"], value_set=macs)))
            tb = tb.filter(keep)
            if not tb.num_rows:
                continue
            pay_col = tb["app_payload"]
            try:
                pay_col = pc.binary_slice(pay_col, 0, CUT)
            except Exception:
                pass
            sm, dm, sip, dip = (tb[c].to_pylist() for c in ("src_mac", "dst_mac", "src_ip", "dst_ip"))
            sp, dp, proto, plen = (tb[c].to_pylist() for c in ("src_port", "dst_port", "ip_proto", "payload_len"))
            pay = pay_col.to_pylist()
            for a, b, sa, da, s_, d_, pr, pl, p in zip(sm, dm, sip, dip, sp, dp, proto, plen, pay):
                for mac, sender_is_dev in ((a, True), (b, False)):
                    if mac not in A:
                        continue
                    if not sender_is_dev and a == b:   # src_mac == dst_mac (seen in attack captures): count once, as sent (script 42 semantics)
                        continue
                    acc = A[mac]
                    acc["rows"] += 1
                    if s_ is None or d_ is None:
                        continue
                    fam, role = S42.family_and_role(int(s_), int(d_), sender_is_dev)
                    acc["n"] += 1
                    acc["fam_role"][f"{fam}|{role}"] += 1
                    acc["tx" if sender_is_dev else "rx"][int(pl)] += 1
                    (acc["pc"] if role == "S" else acc["ps"]).add(da if sender_is_dev else sa)
                    if p:
                        sender_is_server = (role == "S") == sender_is_dev
                        t = S42.msg_token(fam, p, sender_is_server, pr == 6)
                        if t:
                            acc["tokens"][f"{role}:{'tx' if sender_is_dev else 'rx'}:{t}"] += 1
        recs = []
        for mac, dev in targets.items():
            acc = A[mac]
            if acc["rows"] == 0:
                continue
            recs.append({"n_pkts": acc["n"], "fam_role": dict(acc["fam_role"]), "tokens": dict(acc["tokens"]), "tx_n": sum(acc["tx"].values()), "rx_n": sum(acc["rx"].values()),
                         "tx_len_med": hist_percentile(acc["tx"], 50), "tx_len_p90": hist_percentile(acc["tx"], 90),
                         "rx_len_med": hist_percentile(acc["rx"], 50), "rx_len_p90": hist_percentile(acc["rx"], 90),
                         "n_clients_of_device": len(acc["pc"]), "n_servers_contacted": len(acc["ps"]),
                         "device_id": dev, "session_id": session, "n_rows_total": acc["rows"], "truncated": False})
        payload = {"session": session, "file_rows": rows_total, "seconds": round(time.time() - t0, 1), "records": recs}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, out_path)
        return {"session": session, "status": "done", "seconds": payload["seconds"], "rows": rows_total}
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
    return sorted(jobs.items(), key=lambda kv: size[kv[0]]), ids, len(pairs)  # small files first


def merge(jobs=None):
    jobs = jobs or build_jobs()[0]
    recs, missing = [], []
    for s, _t in jobs:
        p = CACHE / f"{s}.json"
        if not p.exists():
            missing.append(s)
            continue
        recs.extend(json.loads(p.read_text(encoding="utf-8"))["records"])
    out = {"params": {"MAX_PKTS": None, "MAX_FILE_BYTES": None, "SESS_PER_DEV": None, "MIN_FRAMES": 0, "payload_cut_bytes": CUT},
           "n_files": len(jobs) - len(missing), "n_files_expected": len(jobs), "n_files_missing": len(missing), "missing": missing[:20], "n_errors": 0, "errors": [], "records": recs}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"merged {out['n_files']}/{len(jobs)} files; {len(recs)} device-session records; missing {len(missing)}")
    return len(missing)


def verify():
    """Superset check against script 42's pass-1 file, plus coverage numbers."""
    new = json.loads(OUT.read_text(encoding="utf-8"))
    old = json.loads((F / "phase2_payload_profiles_v6.json").read_text(encoding="utf-8"))
    nk = {(r["device_id"], r["session_id"]): r for r in new["records"]}
    ok, n_cmp, bad = True, 0, []
    for r in old["records"]:
        k = (r["device_id"], r["session_id"])
        if k not in nk:
            if new["n_files_missing"] == 0:
                bad.append(("missing_in_new", k))
            continue
        if r["n_rows_total"] >= old["params"]["MAX_PKTS"]:   # old record was truncated -> new one must contain MORE or equal, not identical
            if nk[k]["n_rows_total"] < r["n_rows_total"]:
                bad.append(("new_smaller_than_truncated_old", k))
            continue
        n_cmp += 1
        for f in ("n_pkts", "fam_role", "tokens", "tx_n", "rx_n", "n_clients_of_device", "n_servers_contacted", "n_rows_total"):
            if nk[k][f] != r[f]:
                bad.append((f, k, r[f], nk[k][f]))
        for f in ("tx_len_med", "tx_len_p90", "rx_len_med", "rx_len_p90"):
            a, b = r[f], nk[k][f]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                bad.append((f, k, a, b))
    print(f"compared {n_cmp} untruncated pass-1 device-sessions; mismatches: {len(bad)}")
    for b in bad[:15]:
        print("  MISMATCH", b)
    dev_old = {r["device_id"] for r in old["records"]}
    dev_new = {r["device_id"] for r in new["records"]}
    print(f"devices with a record: pass-1 {len(dev_old)} -> full {len(dev_new)} (pass-1 subset of full: {dev_old <= dev_new}); device-sessions {len(old['records'])} -> {len(new['records'])}")
    print("pairs/records with n_pkts total: pass-1", sum(r["n_pkts"] for r in old["records"]), "-> full", sum(r["n_pkts"] for r in new["records"]))
    print("VERIFY", "OK" if not bad else "FAILED")
    return 0 if not bad else 1


def main():
    if "--verify" in sys.argv:
        return verify()
    CACHE.mkdir(exist_ok=True)
    jobs, ids, n_pairs = build_jobs()
    if "--merge-only" in sys.argv:
        return merge(jobs)
    lim = int(sys.argv[sys.argv.index("--limit-files") + 1]) if "--limit-files" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    todo = [j for j in jobs if not (CACHE / f"{j[0]}.json").exists()]
    if "--only" in sys.argv:   # text file with one session name per line (used for the pass-1 overlap test)
        only = set(Path(sys.argv[sys.argv.index("--only") + 1]).read_text(encoding="utf-8").split())
        todo = [j for j in jobs if j[0] in only and not (CACHE / f"{j[0]}.json").exists()]
    if lim:
        todo = todo[:lim]
    print(f"{len(ids)} devices, {n_pairs} device-session pairs, {len(jobs)} files; {len(jobs) - len([j for j in jobs if not (CACHE / (j[0] + '.json')).exists()])} cached, {len(todo)} to do now", flush=True)
    t0, errs, rows = time.time(), [], 0
    with Pool(workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, todo), 1):
            if r["status"] == "error":
                errs.append(r)
            rows += r.get("rows", 0)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} files, {rows / 1e6:.1f}M rows, {time.time() - t0:.0f}s, errors {len(errs)}", flush=True)
    for e in errs[:10]:
        print("ERROR", e)
    miss = merge(jobs)
    print("DONE" if not errs and not miss else "INCOMPLETE - rerun to retry failed/missing files")


if __name__ == "__main__":
    sys.exit(main())
