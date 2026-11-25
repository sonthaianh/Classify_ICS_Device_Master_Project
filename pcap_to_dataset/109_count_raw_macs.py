"""Dem lai tu cac tep parquet THO (Dataset/raw_parquet, 2.230 tep): tong so goi, so MAC phan biet (src U dst), MAC unicast / multicast / broadcast (2026-11-22).
Chi doc hai cot src_mac, dst_mac; KHONG ghi de du lieu. Dau ra: Dataset/features/raw_mac_recount_20261122.json
Chay: py -3.14 109_count_raw_macs.py"""
import glob
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

D = Path(__file__).resolve().parent.parent / "Dataset"
OUT = D / "features/raw_mac_recount_20261122.json"


def main():
    fs = sorted(glob.glob(str(D / "raw_parquet/*.parquet")))
    src, dst = set(), set()
    rows = nulls = 0
    t0 = time.time()
    for k, f in enumerate(fs, 1):
        pf = pq.ParquetFile(f)
        rows += pf.metadata.num_rows
        t = pf.read(columns=["src_mac", "dst_mac"])
        for name, st in (("src_mac", src), ("dst_mac", dst)):
            col = t[name]
            nulls += int(col.null_count)
            u = pc.unique(col.combine_chunks() if hasattr(col, "combine_chunks") else col)
            st.update(x for x in u.to_pylist() if x is not None)
        if k % 100 == 0:
            print(k, len(fs), rows, len(src | dst), round(time.time() - t0), "s", flush=True)
    allm = src | dst

    def first_octet(m):
        try:
            return int(m.replace("-", ":").split(":")[0], 16)
        except Exception:
            return None

    bad = [m for m in allm if first_octet(m) is None]
    ok = [m for m in allm if first_octet(m) is not None]
    bcast = [m for m in ok if m.lower() == "ff:ff:ff:ff:ff:ff"]
    multi = [m for m in ok if first_octet(m) & 1 and m.lower() != "ff:ff:ff:ff:ff:ff"]
    uni = [m for m in ok if not first_octet(m) & 1]
    zero = [m for m in uni if m == "00:00:00:00:00:00"]
    res = {"n_files": len(fs), "n_packets": rows, "null_mac_cells": nulls, "distinct_src_mac": len(src), "distinct_dst_mac": len(dst), "distinct_mac_union": len(allm),
           "unparsable": len(bad), "broadcast": len(bcast), "multicast_nonbroadcast": len(multi), "unicast": len(uni), "unicast_zero_mac": len(zero),
           "src_only_unicast": len([m for m in uni if m in src and m not in dst]), "dst_only_unicast": len([m for m in uni if m in dst and m not in src]),
           "seconds": round(time.time() - t0)}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
