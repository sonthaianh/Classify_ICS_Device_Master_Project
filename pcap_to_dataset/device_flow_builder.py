#!/usr/bin/env python3
r"""
device_flow_builder.py

Buoc 3 trong lo trinh (guide muc 39 / skill muc 5): chuyen Flow Dataset
(bidirectional, tu flow_builder.py) thanh Device-centric Flow - moi flow
tao ra 2 device_flow record (1 cho initiator, 1 cho responder), dao dung
sent/received cho tung phia (guide muc 7.1-7.3).

device_id / peer_id CHUA duoc dien (de trong) - theo dung thu tu lo trinh
(skill muc 5: Device-centric Flow lam TRUOC Ground Truth), device_id se
duoc JOIN vao sau khi devices.csv ton tai (khoa join du kien: device_mac).
device_ip/device_mac van co san va duoc ghi day du.

application_protocol chua co (phu thuoc dissect theo flow, flow_builder.py
muc 1.4 - chua lam) - bo qua, giong flow_builder.py hien tai.

Vectorized (pd.concat 2 nua bang da doi ten cot, khong loop Python tung
dong) - de xu ly duoc flow.parquet hang trieu dong (vd file ABB tcpflood
part1-12, ~5 trieu flow/file).

Chay 1 file:
    "C:\Program Files\Python314\python.exe" Tools\pcap_to_dataset\device_flow_builder.py --file "Tools\Dataset\processed\flows\S0003.parquet"

Chay nhieu file (batch, toan bo hoac lay mau):
    "C:\Program Files\Python314\python.exe" Tools\pcap_to_dataset\device_flow_builder.py --dir "Tools\Dataset\processed\flows" --n 10

Tham so:
    --file    1 file flow.parquet dau vao (dung 1 trong 2: --file hoac --dir)
    --dir     thu muc chua nhieu file flow.parquet, xu ly hang loat
    --n       (chi voi --dir) chi lay mau ngau nhien N file, mac dinh: xu ly TAT CA
    --seed    seed cho --n (mac dinh 42)
    --out     thu muc luu device_flow.parquet (mac dinh: Tools/Dataset/processed/device_flows/)
    --overwrite  xu ly lai file da co output
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Cot chung, khong doi chieu, giu nguyen cho ca 2 device_flow record.
SHARED_COLS = [
    "flow_id", "flow_start", "flow_end", "duration",
    "transport_protocol", "initiator_confidence", "total_packets",
]

# Cot chung TUY CHON - chi co o mot so flow (vd l2_protocol chi co trong
# output cua 06_flow_builder_l2.py, khong co trong flow_builder.py TCP/UDP).
# Copy sang neu ton tai, KHONG bao loi neu thieu, de 1 script doc duoc ca 2
# loai flow.parquet lan lon trong cung thu muc processed/flows/.
OPTIONAL_SHARED_COLS = ["l2_protocol"]

# (cot_phia_initiator, cot_phia_responder) -> ten cot output sau khi dao.
# "sent" tu goc nhin cua device dang xet, "received" la chieu nguoc lai.
DIRECTIONAL_MAP = [
    ("forward_packets", "backward_packets", "packets_sent", "packets_received"),
    ("forward_bytes", "backward_bytes", "bytes_sent", "bytes_received"),
    ("forward_payload_bytes", "backward_payload_bytes", "payload_bytes_sent", "payload_bytes_received"),
    ("forward_payload_packets", "backward_payload_packets", "payload_packets_sent", "payload_packets_received"),
]


def build_device_flows(flows_df: pd.DataFrame) -> pd.DataFrame:
    """Moi dong flow -> 2 dong device_flow (initiator-side, responder-side).
    Vectorized: dung pd.concat 2 nua bang thay vi loop Python tung dong."""
    n = len(flows_df)

    initiator_side = pd.DataFrame({
        "device_ip": flows_df["initiator_ip"].values,
        "device_mac": flows_df["initiator_mac"].values,
        "peer_ip": flows_df["responder_ip"].values,
        "peer_mac": flows_df["responder_mac"].values,
        "local_port": flows_df["initiator_port"].values,
        "remote_port": flows_df["responder_port"].values,
        "direction": ["initiated"] * n,
    })
    responder_side = pd.DataFrame({
        "device_ip": flows_df["responder_ip"].values,
        "device_mac": flows_df["responder_mac"].values,
        "peer_ip": flows_df["initiator_ip"].values,
        "peer_mac": flows_df["initiator_mac"].values,
        "local_port": flows_df["responder_port"].values,
        "remote_port": flows_df["initiator_port"].values,
        "direction": ["responded"] * n,
    })

    for fwd_col, bwd_col, sent_col, received_col in DIRECTIONAL_MAP:
        # Initiator: forward = sent, backward = received.
        initiator_side[sent_col] = flows_df[fwd_col].values
        initiator_side[received_col] = flows_df[bwd_col].values
        # Responder: nguoc lai - backward (ve phia responder) = sent, forward = received.
        responder_side[sent_col] = flows_df[bwd_col].values
        responder_side[received_col] = flows_df[fwd_col].values

    for col in SHARED_COLS:
        initiator_side[col] = flows_df[col].values
        responder_side[col] = flows_df[col].values
    for col in OPTIONAL_SHARED_COLS:
        if col in flows_df.columns:
            initiator_side[col] = flows_df[col].values
            responder_side[col] = flows_df[col].values

    device_flows = pd.concat([initiator_side, responder_side], ignore_index=True)

    int_cols = [c for _, _, sent_c, recv_c in DIRECTIONAL_MAP for c in (sent_c, recv_c)]
    int_cols += ["local_port", "remote_port", "total_packets"]
    for col in int_cols:
        device_flows[col] = device_flows[col].astype("int64")

    return device_flows


def validate_inversion(flows_df: pd.DataFrame, device_flows_df: pd.DataFrame) -> list:
    """Unit test don gian (checklist muc 4): bytes_sent(A) == bytes_received(B)
    va nguoc lai, doi chieu voi so lieu flow goc. Tra ve danh sach loi (rong
    neu pass). Chi chay khi co it nhat 1 flow, dung index-aligned so sanh."""
    errors = []
    n = len(flows_df)
    initiator_rows = device_flows_df.iloc[:n].reset_index(drop=True)
    responder_rows = device_flows_df.iloc[n:].reset_index(drop=True)

    checks = [
        (initiator_rows["bytes_sent"], flows_df["forward_bytes"].reset_index(drop=True)),
        (initiator_rows["bytes_received"], flows_df["backward_bytes"].reset_index(drop=True)),
        (responder_rows["bytes_sent"], flows_df["backward_bytes"].reset_index(drop=True)),
        (responder_rows["bytes_received"], flows_df["forward_bytes"].reset_index(drop=True)),
        (initiator_rows["bytes_sent"], responder_rows["bytes_received"]),
        (initiator_rows["bytes_received"], responder_rows["bytes_sent"]),
    ]
    for a, b in checks:
        if not (a == b).all():
            errors.append("Phat hien dong khong khop khi doi chieu bytes_sent/bytes_received giua 2 phia.")
            break
    return errors


def process_file(in_path: Path, out_dir: Path, overwrite: bool = False) -> dict:
    result = {"file": in_path.name}
    out_path = out_dir / in_path.name
    if out_path.exists() and not overwrite:
        result["skipped_already_done"] = True
        return result

    try:
        flows_df = pq.ParquetFile(str(in_path)).read().to_pandas()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"khong doc duoc file: {exc}"
        return result

    result["n_flows_in"] = len(flows_df)
    if flows_df.empty:
        result["n_device_flows"] = 0
        result["note"] = "file flow rong"
        return result

    device_flows_df = build_device_flows(flows_df)
    errors = validate_inversion(flows_df, device_flows_df)
    if errors:
        result["error"] = "VALIDATION FAIL: " + "; ".join(errors)
        return result

    result["n_device_flows"] = len(device_flows_df)
    table = pa.Table.from_pandas(device_flows_df, preserve_index=False)
    pq.write_table(table, str(out_path))
    result["output"] = str(out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="1 file flow.parquet dau vao")
    parser.add_argument("--dir", help="thu muc chua nhieu file flow.parquet")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "Dataset" / "processed" / "device_flows"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.file and not args.dir:
        print("LOI: can truyen --file (1 file) hoac --dir (nhieu file)")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        in_path = Path(args.file)
        if not in_path.is_file():
            print(f"LOI: khong tim thay file {in_path}")
            return
        r = process_file(in_path, out_dir, overwrite=args.overwrite)
        if "error" in r:
            print(f"LOI: {r['error']}")
            return
        print(f"File: {in_path.name}")
        print(f"So flow dau vao: {r.get('n_flows_in', 0)}")
        print(f"So device_flow tao ra: {r.get('n_device_flows', 0)} (ky vong = 2 x so flow)")
        if "output" in r:
            print(f"Da luu vao: {r['output']}")
        print("Validation dao sent/received: PASS")
        return

    data_dir = Path(args.dir)
    all_files = sorted(data_dir.glob("*.parquet"))
    if args.n is not None and args.n < len(all_files):
        random.seed(args.seed)
        files = sorted(random.sample(all_files, args.n))
    else:
        files = all_files

    print(f"Xu ly {len(files)} file (tren tong so {len(all_files)} file trong {data_dir})...\n", flush=True)
    results = []
    for idx, f in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {f.name}", end=" ", flush=True)
        r = process_file(f, out_dir, overwrite=args.overwrite)
        if "error" in r:
            print(f"LOI: {r['error']}", flush=True)
        elif r.get("skipped_already_done"):
            print("da xu ly roi, bo qua", flush=True)
        else:
            print(f"-> {r.get('n_flows_in', 0)} flow -> {r.get('n_device_flows', 0)} device_flow", flush=True)
        results.append(r)

    newly_processed = [r for r in results if "error" not in r and not r.get("skipped_already_done")]
    n_errors = sum(1 for r in results if "error" in r)
    total_flows = sum(r.get("n_flows_in", 0) for r in newly_processed)
    total_device_flows = sum(r.get("n_device_flows", 0) for r in newly_processed)
    print("\n" + "=" * 70)
    print("TONG KET:")
    print(f"  So file xu ly lan nay: {len(newly_processed)} / {len(files)}")
    print(f"  So file loi: {n_errors}")
    if n_errors:
        for r in results:
            if "error" in r:
                print(f"    {r['file']}: {r['error']}")
    print(f"  Tong so flow dau vao: {total_flows}")
    print(f"  Tong so device_flow tao ra: {total_device_flows} (ky vong = 2 x {total_flows} = {2*total_flows})")


if __name__ == "__main__":
    main()
