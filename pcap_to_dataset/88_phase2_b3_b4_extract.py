"""Phase 2: trich du lieu cho baseline B3 (Al Ghazo & Kumar 2019, phan cap theo tap dich) va B4 (Pospisil & Fujdiak 2024, entropy payload) (2026-11-13).
KHONG ghi de ket qua dong bang. Moi session co thiet bi co nhan giai doan 2 (144 thiet bi co ho so payload) doc 2 lan:
  Luot 1 (cot nho)   : do thi giao tiep unicast theo MAC (src_mac -> dst_mac), dem goi, danh dau canh theo cong OT (flag 1 = huong yeu cau: cong dich thuoc danh muc cong OT cua v6; 2 = huong phan hoi; 3 = hai dau; 0 = khong OT) -> B3
  Luot 2 (payload)   : toi da MAX_PKT goi mang payload (TCP/UDP, hai huong) dau tien cua moi thiet bi; moi goi: entropy Shannon (log2) cua <= 24 byte dau
                       va tong gia tri 24 byte dau (kich ban 3 cua Pospisil va Fujdiak); chia mau 15 phut (bin 900 s theo dau thoi gian), giu mau >= 20 goi,
                       moi mau -> 10 dac trung (do lech chuan, he so bien thien, 'skew' = trung binh - trung vi, khoang, IQR cho entropy va cho tong) -> B4
Cache: Dataset/features/b3b4_cache_v6/<session>.pkl.  Chay: py -3.14 88_phase2_b3_b4_extract.py [--workers 8]
"""
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "Dataset"
F = DS / "features"
CACHE = F / "b3b4_cache_v6"
MAX_PKT, BIN, MIN_PK, MAX_SAMPLES, NB = 20000, 900.0, 20, 40, 24


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_OT = None


def ot_ports():
    global _OT
    if _OT is None:
        import re
        text = (DS.parent / "utils/DataConfig/ot_ports.yaml").read_text(encoding="utf-8")
        ports = set()
        for line in text.splitlines():
            if line.startswith("port_to_protocol"):
                break
            m = re.match(r"^\s*-\s*(\d+)\b", line)
            if m:
                ports.add(int(m.group(1)))
        _OT = ports
    return _OT


def is_mc(m):
    return isinstance(m, str) and len(m) > 1 and m[1].lower() in "13579bdf"


def byte_stats(sl):
    """sl: list[bytes] (<= 24 byte dau) -> (entropy log2, tong) theo tung goi, vector hoa."""
    n = len(sl)
    M = np.full((n, NB), 256, dtype=np.int16)
    ln = np.zeros(n, dtype=np.int16)
    for i, b in enumerate(sl):
        k = min(len(b), NB)
        if k:
            M[i, :k] = np.frombuffer(b[:k], dtype=np.uint8)
        ln[i] = k
    valid = M < 256
    eq = (M[:, :, None] == M[:, None, :]) & valid[:, None, :]
    cnt = eq.sum(axis=2).astype(np.float64)
    nn_ = np.maximum(ln, 1)[:, None].astype(np.float64)
    ent = np.where(valid, (1.0 / nn_) * np.log2(nn_ / np.maximum(cnt, 1)), 0.0).sum(axis=1)
    sm = np.where(valid, M, 0).sum(axis=1)
    return ent, sm.astype(np.float64)


def five(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std()
    q1, q3 = np.percentile(x, [25, 75])
    return [sd, (sd / mu) if mu != 0 else 0.0, mu - np.median(x), x.max() - x.min(), q3 - q1]


def process(job):
    session, targets = job            # targets: {mac: device_id}
    out = CACHE / f"{session}.pkl"
    if out.exists():
        return {"session": session, "status": "cached"}
    try:
        path = DS / "raw_parquet" / f"{session}.parquet"
        otp = ot_ports()
        pf = pq.ParquetFile(path)
        edges = defaultdict(int)
        for batch in pf.iter_batches(batch_size=500000, columns=["src_mac", "dst_mac", "src_port", "dst_port"]):
            d = pa.Table.from_batches([batch]).to_pandas()
            d = d[d.src_mac.notna() & d.dst_mac.notna() & (d.src_mac != d.dst_mac)]
            if d.empty:
                continue
            mc = d.dst_mac.str.slice(1, 2).str.lower().isin(list("13579bdf"))
            d = d[~mc]
            sp_ot, dp_ot = d.src_port.isin(otp), d.dst_port.isin(otp)
            flag = np.where(dp_ot & ~sp_ot, 1, np.where(sp_ot & ~dp_ot, 2, np.where(sp_ot & dp_ot, 3, 0)))   # 1 = huong yeu cau (cong dich OT), 2 = huong phan hoi, 3 = hai dau OT, 0 = khong OT
            g = d.assign(flag=flag).groupby(["src_mac", "dst_mac", "flag"]).size()
            for (s, t, o), c in g.items():
                edges[(s, t, int(o))] += int(c)
        # luot 2: payload
        pf = pq.ParquetFile(path)
        macs = pa.array(list(targets))
        buf = {m: [] for m in targets}
        cnt = {m: 0 for m in targets}
        for batch in pf.iter_batches(batch_size=200000, columns=["timestamp", "src_mac", "dst_mac", "ip_proto", "payload_len", "app_payload"]):
            tb = pa.Table.from_batches([batch])
            keep = pc.and_(pc.greater(tb["payload_len"], 0), pc.is_in(tb["ip_proto"], value_set=pa.array([6, 17], type=tb["ip_proto"].type)))
            keep = pc.and_(keep, pc.or_(pc.is_in(tb["src_mac"], value_set=macs), pc.is_in(tb["dst_mac"], value_set=macs)))
            tb = tb.filter(keep)
            if tb.num_rows:
                pay = pc.binary_slice(tb["app_payload"], 0, NB).to_pylist()
                ts = tb["timestamp"].to_pylist()
                sm, dm = tb["src_mac"].to_pylist(), tb["dst_mac"].to_pylist()
                for t, a, b, p in zip(ts, sm, dm, pay):
                    for mac in {a, b}:
                        if mac in buf and cnt[mac] < MAX_PKT and p is not None:
                            buf[mac].append((t, bytes(p)))
                            cnt[mac] += 1
            if all(c >= MAX_PKT for c in cnt.values()):
                break
        samples = {}
        for mac, rows in buf.items():
            if not rows:
                continue
            ts = np.array([r[0] for r in rows])
            ent, sm = byte_stats([r[1] for r in rows])
            bins = np.floor(ts / BIN).astype(np.int64)
            res = []
            for b in np.unique(bins):
                m = bins == b
                if m.sum() >= MIN_PK:
                    res.append(five(ent[m]) + five(sm[m]) + [int(m.sum())])
                if len(res) >= MAX_SAMPLES:
                    break
            samples[targets[mac]] = {"samples": res, "n_pkts": int(len(rows))}
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"session": session, "edges": dict(edges), "samples": samples, "targets": targets}, fh)
        os.replace(tmp, out)
        return {"session": session, "status": "ok", "edges": len(edges)}
    except Exception as e:
        return {"session": session, "status": "error", "error": f"{type(e).__name__}: {e}"}


def build_jobs():
    recs = json.loads((F / "phase2_payload_profiles_v6_full.json").read_text(encoding="utf-8"))["records"]
    lab = pd.read_csv(F / "phase2_type_labels_4class_plus_v6_plus_all.csv")
    ls = set(lab.device_id)
    reg = pd.read_csv(DS / "metadata/devices.csv", dtype=str, keep_default_na=False).set_index("device_id")
    by = defaultdict(dict)
    for r in recs:
        if r["device_id"] in ls:
            by[r["session_id"]][reg.loc[r["device_id"], "mac"]] = r["device_id"]
    return sorted(by.items(), key=lambda kv: -(DS / "raw_parquet" / f"{kv[0]}.parquet").stat().st_size)


def main():
    CACHE.mkdir(exist_ok=True)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    jobs = build_jobs()
    todo = [j for j in jobs if not (CACHE / f"{j[0]}.pkl").exists()]
    print(f"{len(jobs)} sessions, {len(todo)} to do", flush=True)
    t0, errs = time.time(), []
    with Pool(workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(process, todo), 1):
            if r["status"] == "error":
                errs.append(r)
                print("ERROR", r, flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s errors {len(errs)}", flush=True)
    print("DONE-B3B4-EXTRACT" if not errs else "ERRORS")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
