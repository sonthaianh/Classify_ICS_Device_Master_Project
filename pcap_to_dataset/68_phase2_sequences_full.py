"""Phase 2 (DL) - chuoi su kien payload theo thiet bi, doc DAY DU moi session va moi goi (2026-11-11). Thay cho gioi han cua script 60 (giu nguyen, la pass 1).

Gio han cu cua script 60 (thua huong tu script 42): 3 session/thiet bi, file <= 400 MB, frames_sent >= 30, 20.000 goi dau moi session.
Script nay: doc MOI cap thiet bi-session (7.543 cap, 1.473 file) va MOI goi (streaming theo lo, an toan bo nho); su kien duoc luu giong script 60
(ky hieu vai tro:huong:thong diep, bin do dai, bin khoang cach thoi gian). GIU NGUYEN gioi han luu tru CAP_EVENTS = 2048 su kien DAU moi cap thiet bi-session
(de so sanh cong bang voi pass 1 va vi cua so CNN chi 256 su kien) - nhung dem token tren TOAN BO goi de doi chieu.
Kiem tra (--verify): (1) dem token + so dong khop script 67 cho moi cap thiet bi-session; (2) voi cap co trong pass 1, 2048 su kien dau (hoac it hon) GIONG HET pass 1.
Cache: Dataset/features/seq_full_cache_v6/<session>.pkl.  Dau ra: phase2_sequences_v6_full.pkl (+ _report.json); phase2_sequences_v6.pkl cu KHONG bi ghi de.
Chay: py -3.14 68_phase2_sequences_full.py [--workers 8] | --merge-only | --verify
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


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S60 = _load("s60", "60_phase2_sequences.py")
S67 = _load("s67", "67_phase2_payload_profiles_full.py")
S42 = S60.S42
F, DS = S60.F, S60.DS
CACHE = F / "seq_full_cache_v6"
OUT = F / "phase2_sequences_v6_full.pkl"
CAP_EVENTS, BATCH, CUT = S60.CAP_EVENTS, 100_000, 300
COLS = S60.COLS


def process_file(job):
    session, targets = job
    out = CACHE / f"{session}.pkl"
    if out.exists():
        return {"session": session, "status": "cached"}
    try:
        pf = pq.ParquetFile(DS / "raw_parquet" / f"{session}.parquet")
        macs = pa.array(list(targets))
        st = {m: {"ev": [], "tok": Counter(), "prev": None, "rows": 0} for m in targets}
        for batch in pf.iter_batches(batch_size=BATCH, columns=COLS):
            tb = pa.Table.from_batches([batch])
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
            ts, sm, dm = (tb[c].to_pylist() for c in ("timestamp", "src_mac", "dst_mac"))
            sp, dp, proto, plen = (tb[c].to_pylist() for c in ("src_port", "dst_port", "ip_proto", "payload_len"))
            pay = pay_col.to_pylist()
            for t, a, b, s_, d_, pr, pl, p in zip(ts, sm, dm, sp, dp, proto, plen, pay):
                for mac, sender_is_dev in ((a, True), (b, False)):
                    if mac not in st:
                        continue
                    if not sender_is_dev and a == b:   # src_mac == dst_mac: count once, as sent (script 42/60 semantics)
                        continue
                    S = st[mac]
                    S["rows"] += 1
                    if s_ is None or d_ is None:
                        continue
                    fam, role = S42.family_and_role(int(s_), int(d_), sender_is_dev)
                    direc = "tx" if sender_is_dev else "rx"
                    sym = f"{role}:{direc}:~{fam}"
                    if p:
                        sender_is_server = (role == "S") == sender_is_dev
                        tk = S42.msg_token(fam, p, sender_is_server, pr == 6)
                        if tk:
                            sym = f"{role}:{direc}:{tk}"
                            S["tok"][sym] += 1
                    if len(S["ev"]) < CAP_EVENTS:
                        S["ev"].append((sym, S60.len_bin(pl), S60.dt_bin(None if S["prev"] is None else max(t - S["prev"], 0.0))))
                    S["prev"] = t
        res = {targets[m]: {"events": S["ev"], "tokens_full": dict(S["tok"]), "n_rows": S["rows"]} for m, S in st.items() if S["rows"] > 0}
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"session": session, "devices": res}, fh)
        os.replace(tmp, out)
        return {"session": session, "status": "ok", "n_dev": len(res)}
    except Exception as e:
        return {"session": session, "status": "error", "error": f"{type(e).__name__}: {e}"}


def merge():
    ref = {(r["device_id"], r["session_id"]): r for r in json.loads((F / "phase2_payload_profiles_v6_full.json").read_text(encoding="utf-8"))["records"]}
    seqs, vc, mism, checked = defaultdict(list), Counter(), [], 0
    for p in sorted(CACHE.glob("*.pkl")):
        d = pickle.loads(p.read_bytes())
        for dev, v in d["devices"].items():
            r = ref.get((dev, d["session"]))
            checked += 1
            if r is None or v["tokens_full"] != r["tokens"] or v["n_rows"] != r["n_rows_total"]:
                mism.append((dev, d["session"]))
            seqs[dev].append((d["session"], v["events"]))
            for s, _, _ in v["events"]:
                vc[s] += 1
    vocab = ["<pad>"] + sorted(vc)
    idx = {s: i for i, s in enumerate(vocab)}
    out = {dev: [{"session": ses, "sym": np.array([idx[s] for s, _, _ in ev], dtype=np.int16), "lenb": np.array([a for _, a, _ in ev], dtype=np.uint8),
                  "dtb": np.array([b for _, _, b in ev], dtype=np.uint8)} for ses, ev in lst] for dev, lst in seqs.items()}
    got = {(dev, s["session"]) for dev, ls in out.items() for s in ls}
    rep = {"device_sessions_in_profiles_full": len(ref), "device_sessions_in_cache": len(got), "pairs_missing_from_cache": len(set(ref) - got), "checked": checked,
           "token_or_rowcount_mismatches_vs_script67": len(mism), "mismatch_examples": mism[:5], "n_devices": len(out), "vocab_size": len(vocab),
           "events_total": int(sum(len(s["sym"]) for ls in out.values() for s in ls)), "cap_events": CAP_EVENTS}
    with open(OUT, "wb") as fh:
        pickle.dump({"vocab": vocab, "devices": out, "cap_events": CAP_EVENTS}, fh)
    (F / "phase2_sequences_v6_full_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return rep


def verify():
    new = pickle.loads(OUT.read_bytes())
    old = pickle.loads((F / "phase2_sequences_v6.pkl").read_bytes())
    bad, n = [], 0
    for dev, lst in old["devices"].items():
        nm = {s["session"]: s for s in new["devices"].get(dev, [])}
        for s in lst:
            n += 1
            t = nm.get(s["session"])
            if t is None:
                bad.append(("missing", dev, s["session"]))
                continue
            a = [old["vocab"][i] for i in s["sym"]]
            b = [new["vocab"][i] for i in t["sym"]]
            if a != b or not np.array_equal(s["lenb"], t["lenb"]) or not np.array_equal(s["dtb"], t["dtb"]):
                bad.append(("events_differ", dev, s["session"], len(a), len(b)))
    print(f"compared {n} pass-1 device-sessions (events decoded to strings); mismatches {len(bad)}")
    for b in bad[:10]:
        print("  MISMATCH", b)
    print("devices old/new:", len(old["devices"]), len(new["devices"]), "| device-sessions old/new:", n, sum(len(v) for v in new["devices"].values()))
    print("VERIFY", "OK" if not bad else "FAILED")
    return 0 if not bad else 1


def main():
    if "--verify" in sys.argv:
        return verify()
    CACHE.mkdir(exist_ok=True)
    if "--merge-only" not in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        jobs, ids, n_pairs = S67.build_jobs()
        todo = [j for j in jobs if not (CACHE / f"{j[0]}.pkl").exists()]
        print(f"{len(ids)} devices, {n_pairs} pairs, {len(jobs)} files; {len(todo)} to do", flush=True)
        t0, errs = time.time(), []
        with Pool(workers, maxtasksperchild=10) as pool:
            for i, r in enumerate(pool.imap_unordered(process_file, todo), 1):
                if r["status"] == "error":
                    errs.append(r)
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} files, {time.time() - t0:.0f}s, errors {len(errs)}", flush=True)
        for e in errs[:10]:
            print("ERROR", e)
        if errs:
            return 1
    rep = merge()
    ok = rep["token_or_rowcount_mismatches_vs_script67"] == 0 and rep["pairs_missing_from_cache"] == 0
    print("DONE" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
