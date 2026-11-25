"""Phase 1: tinh THEM dac trung moi (thoi gian giua cac goi, thong ke do dai goi, do thi giao tiep khong dinh danh) theo dung cua so 45 s cua v6 (2026-11-12).
KHONG ghi de du lieu dong bang; ket qua o Dataset/features/phase1_newfeat_v6/ va phase1_newfeat_v6.parquet.

Chi dung cot khong-payload cua raw_parquet (timestamp, MAC, IP, cong, ip_proto, packet_len): day la dac trung hanh vi.
Khong tinh entropy payload (noi dung, khong phai hanh vi; xem ke hoach WP4).
Loai goi trung bang khoa (MAC, IP, cong, giao thuc, timestamp, do dai) - xap xi chinh sach loai trung cua v6 (v6 co them md5 payload).
Dac trung (13), tinh tren goc nhin cua tung thiet bi trong moi cua so:
  Thoi gian (T)   iat_cv, iat_log_median_ms, iat_entropy, iat_burst, active_slot_ratio      (tren goi do thiet bi GUI; active_slot theo moi huong)
  Do dai (L)      len_cv, len_entropy, len_unique_ratio                                        (tren goi do thiet bi GUI)
  Do thi (G)      peer_count_log, fanout_ratio, reciprocity, top_peer_share, peer_entropy      (chi doi tac unicast; khong dung dia chi lam dac trung)
  py -3.14 84_phase1_new_features.py --workers 10 [--limit N]
"""
import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("s26", HERE / "26_build_behavior_dataset.py")
S26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S26)
ROOT = S26.ROOT
W = 45
FEATS = ["iat_cv", "iat_log_median_ms", "iat_entropy", "iat_burst", "active_slot_ratio",
         "len_cv", "len_entropy", "len_unique_ratio",
         "peer_count_log", "fanout_ratio", "reciprocity", "top_peer_share", "peer_entropy"]
GROUPS = {"T": FEATS[0:5], "L": FEATS[5:8], "G": FEATS[8:13]}
COLS = ["timestamp", "src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "packet_len"]
BINS = np.arange(-6.0, 2.01, 0.5)          # log10(giay)


def ent(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def group_features(ts, out, peer, plen, mcast_peer):
    """ts da sap xep tang; out=1 la goi do thiet bi gui."""
    r = dict.fromkeys(FEATS, np.nan)
    sent = out == 1
    ts_s = ts[sent]
    if len(ts_s) >= 4:
        iat = np.diff(ts_s)
        m = iat.mean()
        r["iat_cv"] = float(iat.std() / m) if m > 0 else np.nan
        med = float(np.median(iat))
        r["iat_log_median_ms"] = float(np.log1p(med * 1000.0))
        pos = np.log10(np.clip(iat, 1e-6, None))
        r["iat_entropy"] = ent(np.histogram(pos, bins=BINS)[0].astype(float)) / np.log2(len(BINS) - 1)
        r["iat_burst"] = float(np.log1p(np.percentile(iat, 90) / med)) if med > 0 else np.nan
    slots = np.unique(np.floor(ts).astype(np.int64))
    r["active_slot_ratio"] = float(len(slots) / W)
    pl_s = plen[sent].astype(float)
    if len(pl_s) >= 3:
        mu = pl_s.mean()
        r["len_cv"] = float(pl_s.std() / mu) if mu > 0 else np.nan
        _, c = np.unique(pl_s, return_counts=True)
        r["len_entropy"] = ent(c.astype(float))
        r["len_unique_ratio"] = float(len(c) / len(pl_s))
    uni = ~mcast_peer
    if uni.any():
        pe, ou = peer[uni], out[uni]
        sp, rp = set(pe[ou == 1].tolist()), set(pe[ou == 0].tolist())
        allp = sp | rp
        r["peer_count_log"] = float(np.log1p(len(allp)))
        r["fanout_ratio"] = float(len(sp) / (len(sp) + len(rp))) if (sp or rp) else np.nan
        r["reciprocity"] = float(len(sp & rp) / len(allp))
        _, c = np.unique(pe, return_counts=True)
        r["top_peer_share"] = float(c.max() / c.sum())
        r["peer_entropy"] = ent(c.astype(float))
    return r


def process(job):
    session, m2d, cache = job
    outp = Path(cache) / (session + ".newfeat.parquet")
    raw = ROOT / "Dataset/raw_parquet" / (session + ".parquet")
    st = raw.stat()
    fp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    meta = outp.with_suffix(".json")
    if outp.exists() and meta.exists() and json.loads(meta.read_text()).get("input") == fp:
        return session, None
    pf = pq.ParquetFile(raw)
    pa.set_cpu_count(2)
    wanted = pa.array(list(m2d))
    parts = []
    for batch in pf.iter_batches(batch_size=500000, columns=COLS):
        t = pa.Table.from_batches([batch])
        rel = pc.or_(pc.fill_null(pc.is_in(t["src_mac"], value_set=wanted), False), pc.fill_null(pc.is_in(t["dst_mac"], value_set=wanted), False))
        d = t.filter(rel).to_pandas()
        if len(d):
            parts.append(d)
    if not parts:
        raise ValueError("no scoped packets")
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d.timestamp)].reset_index(drop=True)
    n0 = len(d)
    # loai trung (xap xi): cung khoa tren goi IP
    code = {c: pd.factorize(d[c].fillna(""))[0].astype(np.int32) for c in ("src_mac", "dst_mac", "src_ip", "dst_ip")}
    keyf = pd.DataFrame({**code, "sp": d.src_port.fillna(-1).astype("int64"), "dp": d.dst_port.fillna(-1).astype("int64"),
                         "pr": d.ip_proto.fillna(-1).astype("int64"), "ts": d.timestamp, "pl": d.packet_len.fillna(-1).astype("int64")})
    ip = d.ip_proto.isin([6, 17]) & d.src_port.notna() & d.dst_port.notna() & d.src_ip.notna() & d.dst_ip.notna()
    dup = (keyf.duplicated(keep="first") & ip).to_numpy()
    del keyf, code, ip
    n_dup = int(dup.sum())
    d = d.loc[~dup, ["timestamp", "src_mac", "dst_mac", "packet_len"]].reset_index(drop=True)
    del dup
    macs = pd.concat([d.src_mac.fillna(""), d.dst_mac.fillna("")], ignore_index=True)
    codes, uniq = pd.factorize(macs)
    sm, dm = codes[:len(d)], codes[len(d):]
    dev_of = np.full(len(uniq), -1, dtype=np.int64)
    devs = sorted(set(m2d.values()))
    didx = {x: i for i, x in enumerate(devs)}
    for i, m in enumerate(uniq):
        if m in m2d:
            dev_of[i] = didx[m2d[m]]
    is_mc = np.array([(len(m) > 1 and m[1].lower() in "13579bdf") for m in uniq])
    ts, pl = d.timestamp.to_numpy(), d.packet_len.fillna(0).to_numpy()
    views = []
    s_ok = dev_of[sm] >= 0
    views.append((dev_of[sm][s_ok], np.ones(s_ok.sum(), dtype=np.int8), dm[s_ok], ts[s_ok], pl[s_ok]))
    d_ok = (dev_of[dm] >= 0) & (sm != dm)
    views.append((dev_of[dm][d_ok], np.zeros(d_ok.sum(), dtype=np.int8), sm[d_ok], ts[d_ok], pl[d_ok]))
    dv, ou, pe, tt, pp = [np.concatenate(z) for z in zip(*views)]
    win = np.floor(tt / W) * W
    order = np.lexsort((tt, win, dv))
    dv, ou, pe, tt, pp, win = dv[order], ou[order], pe[order], tt[order], pp[order], win[order]
    brk = np.flatnonzero((np.diff(dv) != 0) | (np.diff(win) != 0)) + 1
    starts, ends = np.r_[0, brk], np.r_[brk, len(dv)]
    mcp = is_mc[pe]
    rows = []
    for a, b in zip(starts, ends):
        r = group_features(tt[a:b], ou[a:b], pe[a:b], pp[a:b], mcp[a:b])
        r["device_id"], r["window_start"] = devs[dv[a]], float(win[a])
        rows.append(r)
    res = pd.DataFrame(rows)
    res["session_id"] = session
    res.to_parquet(outp.with_suffix(".tmp"), index=False)
    outp.with_suffix(".tmp").replace(outp)
    audit = {"input": fp, "scoped": n0, "dup_dropped": n_dup, "windows": len(res)}
    meta.write_text(json.dumps(audit))
    return session, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-mb", type=float, help="only captures up to this raw size (MB)")
    ap.add_argument("--min-mb", type=float, help="only captures above this raw size (MB)")
    ap.add_argument("--smallest", action="store_true", help="smoke: take the N smallest captures")
    a = ap.parse_args()
    reg = pd.read_csv(ROOT / "Dataset/metadata/devices.csv", dtype=str, keep_default_na=False).set_index("device_id")
    pairs = S26.behavior_pairs(ROOT / "Dataset/metadata/devices.csv")
    sessions = sorted(pairs.session_id.unique())
    if a.limit:
        if a.smallest:
            sessions = sorted(sessions, key=lambda s: (ROOT / "Dataset/raw_parquet" / (s + ".parquet")).stat().st_size)[:a.limit]
        else:
            sessions = sessions[:a.limit]
    szmb = lambda s: (ROOT / "Dataset/raw_parquet" / (s + ".parquet")).stat().st_size / 1e6
    if a.max_mb is not None:
        sessions = [s for s in sessions if szmb(s) <= a.max_mb]
    if a.min_mb is not None:
        sessions = [s for s in sessions if szmb(s) > a.min_mb]
    out = ROOT / "Dataset/features/phase1_newfeat_v6"
    cache = out / "captures"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(s, {reg.loc[dv, "mac"]: dv for dv in pairs.loc[pairs.session_id == s, "device_id"]}, str(cache)) for s in sessions]
    t0, fail = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                s, au = f.result()
                print(f"[{i}/{len(jobs)}] {s}: {'cached' if au is None else au}; {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                fail.append((futs[f], repr(e)))
                print(f"[{i}/{len(jobs)}] ERROR {futs[f]}: {e}", flush=True)
    if fail:
        (out / "failures.json").write_text(json.dumps(fail, indent=1))
        print("FAILURES", len(fail))
        sys.exit(1)
    if not a.limit and a.max_mb is None and a.min_mb is None:
        allf = pd.concat([pd.read_parquet(cache / (s + ".newfeat.parquet")) for s in sessions], ignore_index=True)
        allf.to_parquet(ROOT / "Dataset/features/phase1_newfeat_v6.parquet", index=False)
        print("DONE-NEWFEAT", len(allf), "windows", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
