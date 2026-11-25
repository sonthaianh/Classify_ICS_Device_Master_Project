"""Phase 2 (huong deep learning) - trich CHUOI su kien payload theo thiet bi (2026-11-10).

Muc dich: mo hinh chuoi (1D-CNN/GRU) can THU TU thong diep, trong khi script 42 chi luu so dem. Script nay dung lai CHINH XAC cach chon file/session
va cach tach thong diep cua script 42 (import, khong sao chep): cung danh sach thiet bi-session, cung bo loc (payload_len>0, TCP/UDP), cung gioi han
MAX_PKTS goi dau moi thiet bi-session -> so dem token phai KHOP script 42 (kiem tra tu dong, xem merge()).
Moi su kien = 1 goi mang payload cua thiet bi:  ky hieu = "<vai tro S|C>:<tx|rx>:<token>" neu tach duoc thong diep, nguoc lai "<vai tro>:<tx|rx>:~<ho giao thuc>";
kem bin do dai payload (log2) va bin khoang cach thoi gian (so voi su kien truoc, trong cung thiet bi-session).
Luu toi da CAP_EVENTS su kien dau moi thiet bi-session (de nho), nhung dem token tren TOAN BO (<= MAX_PKTS) de doi chieu.
Cache: Dataset/features/seq_cache_v6/<session>.pkl (ghi nguyen tu; chay lai se bo qua file da xong).  Dau ra gop: phase2_sequences_v6.pkl
Chay: py -3.14 pcap_to_dataset/60_phase2_sequences.py [--workers 6] [--limit N]     (them --merge-only de chi gop + kiem tra)
"""
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s42", HERE / "42_phase2_payload_profiles.py")
S42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S42)
F, DS = S42.F, S42.DS
CACHE = F / "seq_cache_v6"
CAP_EVENTS = 2048
COLS = ["timestamp", "src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "payload_len", "app_payload"]
DT_EDGES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def dt_bin(dt):
    if dt is None:
        return 6
    for i, e in enumerate(DT_EDGES):
        if dt < e:
            return i
    return 5


def len_bin(n):
    return min(int(math.log2(n + 1)), 11)


def device_events(rows, mac):
    """rows: tuples (ts, src_mac, dst_mac, src_ip, dst_ip, sp, dp, proto, plen, payload) -> (events, token_counter_full)."""
    ev, tokens, prev = [], Counter(), None
    for ts, sm, dm, sip, dip, sp, dp, proto, plen, pay in rows:
        if sp is None or dp is None:
            continue
        sender_is_dev = sm == mac
        fam, role = S42.family_and_role(int(sp), int(dp), sender_is_dev)
        direc = "tx" if sender_is_dev else "rx"
        sym = f"{role}:{direc}:~{fam}"
        if pay:
            sender_is_server = (role == "S") == sender_is_dev
            t = S42.msg_token(fam, pay, sender_is_server, proto == 6)
            if t:
                sym = f"{role}:{direc}:{t}"
                tokens[sym] += 1
        if len(ev) < CAP_EVENTS:
            ev.append((sym, len_bin(plen), dt_bin(None if prev is None else max(ts - prev, 0.0))))
        prev = ts
    return ev, tokens


def process_file(job):
    session, targets = job
    out = CACHE / f"{session}.pkl"
    if out.exists():
        return {"session": session, "status": "cached"}
    res = {}
    try:
        t = pq.read_table(DS / "raw_parquet" / f"{session}.parquet", columns=COLS)
        m = pc.and_(pc.greater(t["payload_len"], 0), pc.is_in(t["ip_proto"], value_set=pa.array([6, 17], type=t["ip_proto"].type)))
        t = t.filter(m)
        for mac, dev in targets.items():
            sel = t.filter(pc.or_(pc.equal(t["src_mac"], mac), pc.equal(t["dst_mac"], mac)))
            if sel.num_rows == 0:
                continue
            sel = sel.slice(0, S42.MAX_PKTS)
            cols = [sel[c].to_pylist() for c in COLS]
            ev, tokens = device_events(list(zip(*cols)), mac)
            res[dev] = {"events": ev, "tokens_full": dict(tokens), "n_rows": int(sel.num_rows)}
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"session": session, "devices": res}, fh)
        os.replace(tmp, out)
        return {"session": session, "status": "ok", "n_dev": len(res)}
    except Exception as e:
        return {"session": session, "status": "error", "error": f"{type(e).__name__}: {e}"}


def merge():
    """Gop cache, kiem tra khop script 42, xay vocab va mang so nguyen."""
    prof = json.loads((F / "phase2_payload_profiles_v6.json").read_text(encoding="utf-8"))["records"]
    ref = {(r["device_id"], r["session_id"]): r for r in prof if r.get("device_id")}
    seqs, vocab_counter, mism, checked, missing = defaultdict(list), Counter(), [], 0, []
    for p in sorted(CACHE.glob("*.pkl")):
        d = pickle.loads(p.read_bytes())
        for dev, v in d["devices"].items():
            r = ref.get((dev, d["session"]))
            if r is None:
                missing.append((dev, d["session"]))
                continue
            checked += 1
            if v["tokens_full"] != r["tokens"] or v["n_rows"] != r["n_rows_total"]:
                mism.append((dev, d["session"]))
            seqs[dev].append((d["session"], v["events"]))
            for s, _, _ in v["events"]:
                vocab_counter[s] += 1
    vocab = ["<pad>"] + sorted(vocab_counter)
    idx = {s: i for i, s in enumerate(vocab)}
    out = {}
    for dev, lst in seqs.items():
        out[dev] = [{"session": ses, "sym": np.array([idx[s] for s, _, _ in ev], dtype=np.int16), "lenb": np.array([a for _, a, _ in ev], dtype=np.uint8),
                     "dtb": np.array([b for _, _, b in ev], dtype=np.uint8)} for ses, ev in lst]
    ref_pairs = set(ref)
    got_pairs = {(dev, s["session"]) for dev, ls in out.items() for s in ls}
    report = {"device_sessions_in_profiles": len(ref_pairs), "device_sessions_in_cache": len(got_pairs), "pairs_missing_from_cache": len(ref_pairs - got_pairs),
              "pairs_not_in_profiles": len(missing), "checked": checked, "token_count_mismatches": len(mism), "mismatch_examples": mism[:5],
              "n_devices": len(out), "vocab_size": len(vocab), "events_total": int(sum(len(s["sym"]) for ls in out.values() for s in ls))}
    with open(F / "phase2_sequences_v6.pkl", "wb") as fh:
        pickle.dump({"vocab": vocab, "devices": out, "cap_events": CAP_EVENTS}, fh)
    (F / "phase2_sequences_v6_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    CACHE.mkdir(exist_ok=True)
    if "--merge-only" not in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
        jobs, n_ids, n_cov = S42.select_jobs()
        if limit:
            jobs = jobs[:limit]
        print(f"{len(jobs)} files, {n_cov}/{n_ids} devices covered", flush=True)
        t0, errs = time.time(), []
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
                if r["status"] == "error":
                    errs.append(r)
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} files, {time.time() - t0:.0f}s, errors {len(errs)}", flush=True)
        print("errors:", errs[:5], flush=True)
        if limit:
            return
    rep = merge()
    assert rep["token_count_mismatches"] == 0, "so dem token khong khop script 42"
    assert rep["pairs_missing_from_cache"] == 0, "thieu cap thiet bi-session"
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
