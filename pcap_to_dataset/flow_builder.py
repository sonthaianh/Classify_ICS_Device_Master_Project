#!/usr/bin/env python3
r"""
flow_builder.py

Buoc 2 trong lo trinh (guide muc 39): xay Flow Dataset bidirectional tu
Packet Dataset. Da co: gop packet thanh flow 2 chieu, xac dinh dung
initiator/responder, dao dung sent/received, packet-size stats (mean/std)
va IAT stats (mean/std) theo chieu forward/backward (dung schema muc 6.2
guide). Con thieu: application_protocol (phu thuoc dissect theo flow, muc
1.4), VLAN (vlan_id da co san trong Packet Dataset nhung chua duoc dung de
tach flow), IP fragmentation.

Tu tinh is_duplicate_copy ngay trong script nay (dung lai chinh xac logic da
kiem chung o 01/02_check_duplicate_packets.py) va LOC RA truoc khi build flow
- dung chinh sach da chot, khong phai tuy chon.

Chi xu ly TCP/UDP (IPv4). Cac packet khac (ARP/ICMP/...) duoc dem va bao cao
ro trong summary, KHONG bi am tham bo qua.

Initiator/responder (dung guide muc 6.3):
    TCP: uu tien goi SYN THAT (SYN=1, ACK=0) - src cua goi do la initiator.
         Neu khong thay handshake trong file (capture bat dau giua chung),
         fallback: goi som nhat theo timestamp -> initiator, confidence thap hon.
    UDP: luon dung first-sender (goi som nhat), confidence thap hon TCP co
         handshake - KHONG bao gio coi ngang hang voi TCP co bang chung SYN that.

Chay 1 file:
    python pcap_to_dataset\flow_builder.py --file "Dataset\raw_parquet\S0003.parquet"

Chay nhieu file cung luc (batch, lay mau ngau nhien N file trong 1 thu muc):
    python pcap_to_dataset\flow_builder.py --dir "Dataset\raw_parquet" --n 10

Tham so:
    --file    1 file parquet Packet Dataset dau vao (dung 1 trong 2: --file hoac --dir)
    --dir     thu muc chua nhieu file parquet, xu ly hang loat
    --n       (chi dung voi --dir) chi lay mau ngau nhien N file, mac dinh: xu ly TAT CA file trong --dir
    --seed    seed cho --n (mac dinh 42, de tai lap duoc)
    --out     thu muc luu flow.parquet (mac dinh: Dataset/processed/flows/)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ep stdout/stderr dung UTF-8 (thay the ky tu khong encode duoc thay vi sap
# chuong trinh) - tren Windows, khi redirect ra file, Python mac dinh dung
# codepage console (vd cp1252) va se crash neu ten file co ky tu Unicode la.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SCHEMA_LEGACY = {
    "timestamp": "timestamp", "src_mac": "src_mac", "dst_mac": "dst_mac",
    "src_ip": "src_ip", "src_port": "src_port", "dst_ip": "dst_ip", "dst_port": "dst_port",
    "ip_proto": "ip_proto", "tcp_flags": "tcp_flags", "payload": "app_payload",
    "packet_len": "packet_len",
}
SCHEMA_BRONZE_NEW = {
    "timestamp": "capture_timestamp_epoch_ns", "src_mac": "src_mac", "dst_mac": "dst_mac",
    "src_ip": "src_ip", "src_port": "src_port", "dst_ip": "dst_ip", "dst_port": "dst_port",
    "ip_proto": "ip_proto", "tcp_flags": "tcp_flags", "payload": "payload",
    "packet_len": "packet_len",
}
KNOWN_SCHEMAS = [("legacy", SCHEMA_LEGACY), ("bronze_new", SCHEMA_BRONZE_NEW)]

TCP_PROTO = 6
UDP_PROTO = 17
SYN_FLAG = 0x02
ACK_FLAG = 0x10


def detect_schema(columns: list) -> tuple:
    for name, mapping in KNOWN_SCHEMAS:
        if all(c in columns for c in mapping.values()):
            return name, mapping
    return None, None


def build_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Ban vectorized (khong loop Python tung flow) - de xu ly duoc ca file
    hang chuc trieu dong. Logic giu NGUYEN nhu ban cu (da kiem chung tren
    file SEL): flow_key = cap endpoint da chuan hoa (khong phan biet chieu) +
    ip_proto; initiator uu tien SYN that (TCP), fallback first-packet;
    responder = dst cua chinh dong packet duoc chon lam initiator (vi 1
    packet luon co du src/dst dung cua no)."""
    df = df.reset_index(drop=True)

    # --- Flow key vectorized (thay the df.apply(flow_key, axis=1)) ---
    ep_a = df["src_ip"].astype(str) + ":" + df["src_port"].astype("int64").astype(str)
    ep_b = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype("int64").astype(str)
    a_first = ep_a < ep_b
    smaller = ep_a.where(a_first, ep_b)
    larger = ep_b.where(a_first, ep_a)
    ip_proto_int = df["ip_proto"].astype("int64")
    df["_flow_key"] = smaller + "|" + larger + "|" + ip_proto_int.astype(str)

    # --- Chon dong "initiator" cho moi flow: uu tien SYN that, fallback dong som nhat ---
    flags = df["tcp_flags"].fillna(0).astype("int64")
    is_pure_syn = (ip_proto_int == TCP_PROTO) & ((flags & SYN_FLAG) != 0) & ((flags & ACK_FLAG) == 0)
    df["_syn_priority"] = (~is_pure_syn).astype("int8")  # 0 = SYN that (uu tien), 1 = khong

    sorted_df = df.sort_values(["_flow_key", "_syn_priority", "timestamp"], kind="mergesort")
    chosen = sorted_df.drop_duplicates(subset="_flow_key", keep="first")

    chosen_confidence = pd.Series(
        index=chosen.index,
        data=pd.NA,
    )
    is_tcp_chosen = chosen["ip_proto"].astype("int64") == TCP_PROTO
    chosen_confidence = chosen_confidence.mask(
        (chosen["_syn_priority"] == 0), "tcp_syn_handshake"
    )
    chosen_confidence = chosen_confidence.mask(
        (chosen["_syn_priority"] != 0) & is_tcp_chosen, "first_packet_no_handshake_seen"
    )
    chosen_confidence = chosen_confidence.mask(
        (chosen["_syn_priority"] != 0) & ~is_tcp_chosen, "udp_first_sender_heuristic"
    )

    flow_meta = pd.DataFrame({
        "_flow_key": chosen["_flow_key"].values,
        "initiator_ip": chosen["src_ip"].values,
        "initiator_port": chosen["src_port"].astype("int64").values,
        "initiator_mac": chosen["src_mac"].values,
        "responder_ip": chosen["dst_ip"].values,
        "responder_port": chosen["dst_port"].astype("int64").values,
        "responder_mac": chosen["dst_mac"].values,
        "transport_protocol": np.where(is_tcp_chosen.values, "tcp", "udp"),
        "initiator_confidence": chosen_confidence.values,
    }).set_index("_flow_key")

    # --- Gan huong forward/backward cho moi packet (vectorized) ---
    init_lookup = flow_meta[["initiator_ip", "initiator_port"]].rename(
        columns={"initiator_ip": "_init_ip", "initiator_port": "_init_port"}
    )
    df = df.join(init_lookup, on="_flow_key")
    is_forward = (df["src_ip"] == df["_init_ip"]) & (df["src_port"].astype("int64") == df["_init_port"])
    df["_direction"] = np.where(is_forward, "forward", "backward")

    # --- Aggregate (groupby vectorized, khong loop Python) ---
    packet_counts = df.groupby(["_flow_key", "_direction"]).size().unstack(fill_value=0)
    packet_counts = packet_counts.reindex(columns=["forward", "backward"], fill_value=0)

    payload_sums = df.groupby(["_flow_key", "_direction"])["payload_len"].sum().unstack(fill_value=0)
    payload_sums = payload_sums.reindex(columns=["forward", "backward"], fill_value=0)

    # packet_len bao gom ca header (khac payload_len chi tinh phan payload) -
    # dung cho forward_bytes/backward_bytes theo dung schema muc 6.2 guide.
    byte_sums = df.groupby(["_flow_key", "_direction"])["packet_len"].sum().unstack(fill_value=0)
    byte_sums = byte_sums.reindex(columns=["forward", "backward"], fill_value=0)

    has_payload = df["payload_len"] > 0
    payload_pkt_counts = df[has_payload].groupby(["_flow_key", "_direction"]).size().unstack(fill_value=0)
    payload_pkt_counts = payload_pkt_counts.reindex(index=byte_sums.index, columns=["forward", "backward"], fill_value=0)

    # packet_len_mean/std theo chieu - dung ddof=0 (population std) de flow chi
    # co 1 packet/chieu tra ve 0.0 thay vi NaN (khong co bien thien quan sat duoc).
    len_grouped = df.groupby(["_flow_key", "_direction"])["packet_len"]
    len_mean = len_grouped.mean().unstack(fill_value=0.0).reindex(columns=["forward", "backward"], fill_value=0.0)
    len_std = len_grouped.std(ddof=0).unstack(fill_value=0.0).reindex(columns=["forward", "backward"], fill_value=0.0)

    # iat (inter-arrival time) theo chieu - sap xep theo timestamp trong tung
    # (flow, direction) roi lay diff. Packet dau tien cua moi chieu khong co
    # iat (NaN), pandas mean/std tu dong bo qua NaN (skipna=True mac dinh).
    # Flow co <2 packet o 1 chieu -> khong quan sat duoc iat nao -> fillna 0.0.
    iat_sorted = df.sort_values(["_flow_key", "_direction", "timestamp"], kind="mergesort")
    iat_sorted["_iat"] = iat_sorted.groupby(["_flow_key", "_direction"])["timestamp"].diff()
    iat_grouped = iat_sorted.groupby(["_flow_key", "_direction"])["_iat"]
    iat_mean = iat_grouped.mean().unstack(fill_value=0.0).reindex(columns=["forward", "backward"], fill_value=0.0).fillna(0.0)
    iat_std = iat_grouped.std(ddof=0).unstack(fill_value=0.0).reindex(columns=["forward", "backward"], fill_value=0.0).fillna(0.0)

    flow_time = df.groupby("_flow_key")["timestamp"].agg(flow_start="min", flow_end="max")
    flow_time["duration"] = flow_time["flow_end"] - flow_time["flow_start"]
    total_packets = df.groupby("_flow_key").size().rename("total_packets")

    flows_df = flow_meta.join([
        packet_counts.rename(columns={"forward": "forward_packets", "backward": "backward_packets"}),
        byte_sums.rename(columns={"forward": "forward_bytes", "backward": "backward_bytes"}),
        payload_sums.rename(columns={"forward": "forward_payload_bytes", "backward": "backward_payload_bytes"}),
        payload_pkt_counts.rename(columns={"forward": "forward_payload_packets", "backward": "backward_payload_packets"}),
        len_mean.rename(columns={"forward": "forward_packet_len_mean", "backward": "backward_packet_len_mean"}),
        len_std.rename(columns={"forward": "forward_packet_len_std", "backward": "backward_packet_len_std"}),
        iat_mean.rename(columns={"forward": "forward_iat_mean", "backward": "backward_iat_mean"}),
        iat_std.rename(columns={"forward": "forward_iat_std", "backward": "backward_iat_std"}),
        flow_time,
        total_packets,
    ]).reset_index()

    flows_df["flow_id"] = flows_df["_flow_key"].apply(lambda k: hashlib.md5(k.encode()).hexdigest()[:16])
    flows_df = flows_df.drop(columns=["_flow_key"])
    for col in ["forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
                "forward_payload_bytes", "backward_payload_bytes",
                "forward_payload_packets", "backward_payload_packets", "total_packets"]:
        flows_df[col] = flows_df[col].astype("int64")
    for col in ["forward_packet_len_mean", "forward_packet_len_std", "backward_packet_len_mean", "backward_packet_len_std",
                "forward_iat_mean", "forward_iat_std", "backward_iat_mean", "backward_iat_std"]:
        flows_df[col] = flows_df[col].astype("float64")

    return flows_df


def process_file(in_path: Path, out_dir: Path, verbose: bool = True, max_packets: int = 1_000_000, overwrite: bool = False) -> dict:
    """Xu ly 1 file Packet Dataset -> 1 file flow.parquet. Tra ve dict tom tat
    (dung de in bang tong ket khi chay batch nhieu file)."""
    result = {"file": in_path.name}

    try:
        pf = pq.ParquetFile(str(in_path))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"khong mo duoc file: {exc}"
        return result

    total_rows_in_file = pf.metadata.num_rows
    if max_packets is not None and total_rows_in_file > max_packets:
        result["skipped_too_large"] = True
        result["total_packets"] = total_rows_in_file
        stale_out_path = out_dir / in_path.name
        stale_warning = ""
        if stale_out_path.exists():
            stale_warning = (
                f" CANH BAO: {stale_out_path} van con TON TAI tu lan chay truoc (co the la "
                "schema/nguong cu) - file nay bi bo qua nen KHONG duoc ghi de, output cu se "
                "TIEP TUC duoc dung boi cac buoc sau neu khong tu don. Kiem tra thu cong."
            )
        result["error"] = (
            f"BO QUA (khong xu ly): {total_rows_in_file} packet > nguong an toan {max_packets}. "
            "File qua lon co the la flood/DDoS capture voi hang trieu flow rieng biet - "
            "loop hien tai chua toi uu cho truong hop nay. Dung --max-packets de nang nguong "
            "neu muon thu xu ly (co the rat cham)." + stale_warning
        )
        return result

    out_path = out_dir / in_path.name
    if out_path.exists() and not overwrite:
        result["skipped_already_done"] = True
        result["total_packets"] = total_rows_in_file
        if verbose:
            print(f"  [{in_path.name}] da xu ly roi (output ton tai), bo qua. Dung --overwrite de lam lai.", flush=True)
        return result

    schema_name, mapping = detect_schema(pf.schema_arrow.names)
    if mapping is None:
        result["error"] = f"khong nhan dien duoc schema (cot hien co: {pf.schema_arrow.names})"
        return result
    result["schema"] = schema_name

    if verbose:
        print(f"  Dang xu ly: {in_path.name} ({total_rows_in_file} packet)...", flush=True)

    df = pf.read(columns=list(mapping.values())).to_pandas()
    df = df.rename(columns={v: k for k, v in mapping.items()})
    total_rows = len(df)
    result["total_packets"] = total_rows

    if total_rows == 0:
        result["n_flows"] = 0
        result["note"] = "file rong"
        return result

    if verbose:
        print(f"    -> dang tinh duplicate fingerprint...", flush=True)

    # --- Buoc 1: gan co va loc packet trung lap that (chinh sach da chot) ---
    df["payload_len"] = df["payload"].apply(lambda b: len(b) if b else 0)
    df["payload_md5"] = df["payload"].apply(lambda b: hashlib.md5(b or b"").hexdigest())
    fingerprint_cols = ["src_ip", "src_port", "dst_ip", "dst_port", "ip_proto", "timestamp", "payload_md5"]
    df["is_duplicate_copy"] = df.duplicated(subset=fingerprint_cols, keep="first")

    n_dup = int(df["is_duplicate_copy"].sum())
    result["n_duplicate_excluded"] = n_dup
    df = df[~df["is_duplicate_copy"]].copy()

    # --- Buoc 2: chi giu TCP/UDP ---
    supported_mask = df["ip_proto"].isin([TCP_PROTO, UDP_PROTO])
    result["n_non_tcp_udp_skipped"] = int((~supported_mask).sum())
    df = df[supported_mask].copy()

    missing_port = df["src_port"].isna() | df["dst_port"].isna()
    result["n_missing_port_skipped"] = int(missing_port.sum())
    df = df[~missing_port].copy()

    if df.empty:
        result["n_flows"] = 0
        result["note"] = "khong con packet TCP/UDP hop le sau khi loc"
        return result

    if verbose:
        print(f"    -> dang gop {len(df)} packet thanh flow...", flush=True)

    # --- Buoc 3: build flow ---
    flows_df = build_flows(df)
    result["n_flows"] = len(flows_df)
    result["initiator_confidence_breakdown"] = flows_df["initiator_confidence"].value_counts().to_dict()
    result["transport_breakdown"] = flows_df["transport_protocol"].value_counts().to_dict()

    out_path = out_dir / in_path.name
    table = pa.Table.from_pandas(flows_df, preserve_index=False)
    pq.write_table(table, str(out_path))
    result["output"] = str(out_path)

    if verbose:
        print(f"    -> XONG: {total_rows} packet -> {len(flows_df)} flow "
              f"(loai {n_dup} trung lap, bo qua {result['n_non_tcp_udp_skipped']} non-TCP/UDP)", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="1 file parquet dau vao")
    parser.add_argument("--dir", help="thu muc chua nhieu file parquet, xu ly hang loat")
    parser.add_argument("--n", type=int, default=None, help="(chi voi --dir) chi lay mau ngau nhien N file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-packets", type=int, default=15_000_000,
        help="Bo qua (khong xu ly) file co nhieu hon N packet. Sau khi vectorized, "
             "file 56 trieu dong cung chi mat ~7 phut - nguong nay chi de tranh xu ly "
             "cac file flood/DDoS cuc lon (>=20 trieu dong trong corpus hien tai) khong "
             "phuc vu bai toan phan loai thiet bi. Dat 0 de khong gioi han. Mac dinh 15,000,000.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "Dataset" / "processed" / "flows"),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Xu ly lai ca file da co output roi (mac dinh: bo qua file da xu ly xong, de chay lai sau khi loi giua chung khong phai lam lai tu dau).",
    )
    parser.add_argument(
        "--dedup-manifest", action="append", default=None,
        help="(chi voi --dir) Duong dan 1 file manifest JSON do 04_check_duplicate_files.py "
             "sinh ra (co key 'duplicate_groups' voi 'redundant_copies'). Cac file ten nam "
             "trong 'redundant_copies' se BI BO QUA (khong build flow), tranh 2 ban sao cua "
             "cung 1 capture bi coi la 2 session doc lap (leakage sau nay). Truyen nhieu lan "
             "de gop nhieu manifest (vd nhieu batch khac nhau). KHONG xoa file redundant khoi "
             "dia, chi bo qua khi build flow.",
    )
    args = parser.parse_args()

    max_packets = None if args.max_packets == 0 else args.max_packets

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
        print(f"File: {in_path.name}")
        r = process_file(in_path, out_dir, verbose=False, max_packets=max_packets, overwrite=args.overwrite)
        if "error" in r:
            print(f"LOI: {r['error']}")
            return
        print(f"Schema: {r.get('schema')}")
        print(f"Tong so packet: {r['total_packets']}")
        print(f"Da loai {r.get('n_duplicate_excluded', 0)} packet trung lap")
        print(f"Bo qua {r.get('n_non_tcp_udp_skipped', 0)} packet khong phai TCP/UDP, "
              f"{r.get('n_missing_port_skipped', 0)} packet thieu port")
        print(f"\nDa build {r.get('n_flows', 0)} flow.")
        if "output" in r:
            print(f"Da luu vao: {r['output']}")
        print("\nTOM TAT initiator_confidence:")
        for k, v in r.get("initiator_confidence_breakdown", {}).items():
            print(f"  {k}: {v}")
        print("\nTOM TAT transport_protocol:")
        for k, v in r.get("transport_breakdown", {}).items():
            print(f"  {k}: {v}")
        return

    # --- Batch mode: --dir ---
    data_dir = Path(args.dir)
    all_files = sorted(data_dir.glob("*.parquet"))

    redundant_names: set = set()
    if args.dedup_manifest:
        for manifest_path in args.dedup_manifest:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            for group in manifest.get("duplicate_groups", []):
                redundant_names.update(group.get("redundant_copies", []))
        print(f"Da doc {len(args.dedup_manifest)} manifest, {len(redundant_names)} file se bi BO QUA vi la ban sao trung lap.\n", flush=True)

    if redundant_names:
        skipped_dup_files = [f for f in all_files if f.name in redundant_names]
        all_files = [f for f in all_files if f.name not in redundant_names]
    else:
        skipped_dup_files = []

    if args.n is not None and args.n < len(all_files):
        random.seed(args.seed)
        files = sorted(random.sample(all_files, args.n))
    else:
        files = all_files

    print(f"Xu ly {len(files)} file (tren tong so {len(all_files)} file trong {data_dir}"
          f"{f', da loai {len(skipped_dup_files)} file trung lap theo manifest' if skipped_dup_files else ''})...\n", flush=True)

    results = []
    for idx, f in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}]", end=" ", flush=True)
        r = process_file(f, out_dir, verbose=True, max_packets=max_packets, overwrite=args.overwrite)
        if "error" in r:
            print(f"  {f.name}: {r['error']}", flush=True)
        results.append(r)

    print("\n" + "=" * 70)
    print("TONG KET:")
    if skipped_dup_files:
        print(f"  So file bo qua vi la ban sao trung lap (theo dedup manifest, KHONG build flow): {len(skipped_dup_files)}")
    # CHI tinh tong tren cac file THAT SU vua xu ly trong lan chay nay (khong
    # phai loi cu, khong phai skip-vi-qua-lon, khong phai skip-vi-da-xong-tu-truoc)
    # - tranh bug cong nham gia tri mac dinh 0 cua file bi bo qua vao tong.
    newly_processed = [r for r in results if "error" not in r and not r.get("skipped_too_large") and not r.get("skipped_already_done")]
    total_flows_new = sum(r.get("n_flows", 0) for r in newly_processed)
    total_packets_new = sum(r.get("total_packets", 0) for r in newly_processed)
    total_dup_new = sum(r.get("n_duplicate_excluded", 0) for r in newly_processed)
    n_skipped_large = sum(1 for r in results if r.get("skipped_too_large"))
    n_skipped_done = sum(1 for r in results if r.get("skipped_already_done"))
    n_errors = sum(1 for r in results if "error" in r and not r.get("skipped_too_large"))
    print(f"  Tong so file xu ly thanh cong (ke ca da xu ly tu truoc): {len(results) - n_errors - n_skipped_large} / {len(results)}")
    print(f"  Trong do, da xu ly tu lan chay truoc (bo qua lan nay, KHONG tinh vao thong ke ben duoi): {n_skipped_done}")
    print(f"  So file THAT SU moi xu ly trong lan chay nay: {len(newly_processed)}")
    if newly_processed:
        n_new_zero_flow = sum(1 for r in newly_processed if r.get("n_flows", 0) == 0)
        print(f"    Trong do so file moi ma KHONG tao duoc flow nao (n_flows=0, dang ngo): {n_new_zero_flow} / {len(newly_processed)}")
        if n_new_zero_flow > 0:
            print("    Vi du 5 file dau tien co n_flows=0 (can kiem tra rieng):")
            for r in [r for r in newly_processed if r.get("n_flows", 0) == 0][:5]:
                print(f"      {r['file']}: total_packets={r.get('total_packets')}, note={r.get('note')}")
    print(f"  So file bo qua vi qua lon (> {max_packets or 'khong gioi han'} packet): {n_skipped_large}")
    if n_skipped_large:
        for r in results:
            if r.get("skipped_too_large"):
                print(f"    {r['file']}: {r['total_packets']} packet")
    print(f"  [Chi tinh tren file MOI xu ly lan nay] Tong so packet da doc: {total_packets_new}")
    print(f"  [Chi tinh tren file MOI xu ly lan nay] Tong so packet trung lap da loai: {total_dup_new}")
    print(f"  [Chi tinh tren file MOI xu ly lan nay] Tong so flow da build: {total_flows_new}")
    if n_errors:
        print(f"  So file loi khac (khong phai do qua lon): {n_errors}")
        for r in results:
            if "error" in r and not r.get("skipped_too_large"):
                print(f"    {r['file']}: {r['error']}")

    all_confidence: dict = {}
    for r in results:
        for k, v in r.get("initiator_confidence_breakdown", {}).items():
            all_confidence[k] = all_confidence.get(k, 0) + v
    print("\n  Tong hop initiator_confidence tren toan bo mau:")
    for k, v in sorted(all_confidence.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
