#!/usr/bin/env python3
r"""
pcap_to_parquet.py

Batch converter: PCAP/PCAPNG -> canonical Parquet (1 output file per input file).

Muc dich (theo Ke_hoach_Passive_IT_OT_Asset_Inventory_ML_DL.md, WP02):
  - Xu ly toan bo thu muc PCAP, moi file input xuat ra dung 1 file Parquet tuong ung.
  - Khong bao gio ghi/sua file PCAP goc (chi mo read-only).
  - Tinh SHA256 cua file PCAP goc TRUOC khi xu ly va ghi vao manifest -> dam bao
    truy vet duoc (integrity/provenance), phat hien neu file bi doi/hong sau nay.
  - Xu ly dung ca cac frame bi PHAN MANH IP (IP fragmentation): CHI frame co
    frag_offset == 0 moi duoc coi la co transport header (TCP/UDP/ICMP) that su;
    cac fragment tiep theo (frag_offset > 0) khong bao gio bi gan port/header gia.
    Day la fix truc tiep cho loi da phat hien o bo converter cu (raw_parquet/):
    bo cu gia dinh MOI packet co ip_proto=17 deu co 8-byte UDP header, kien den
    fragment tiep theo bi cat mat 8 byte du lieu that va bi gan port=0 thay vi null.
  - Doc/ghi theo kieu streaming (khong nap toan bo file lon vao RAM): doc PCAP
    packet-by-packet qua scapy RawPcapReader/RawPcapNgReader, ghi Parquet theo
    tung batch qua pyarrow.ParquetWriter.
  - Loi o 1 packet khong lam mat ket qua da parse hop le cua ca file: ghi canh bao
    vao cot parse_warning va tiep tuc, KHONG dung ca file. Loi khong the phuc hoi
    o muc file (vi du magic header sai) duoc bat rieng va ghi status="failed"
    trong manifest, khong lam sap toan bo batch job.

Yeu cau cai dat (tren may se chay script nay):
    pip install -r requirements.txt
    (scapy>=2.5, pyarrow>=14)

Vi du chay:
    python pcap_to_parquet.py --input D:\pcap_source --output D:\dataset_out
    python pcap_to_parquet.py --input D:\pcap_source\one_file.pcap --output D:\dataset_out
    python pcap_to_parquet.py --input D:\pcap_source --output D:\dataset_out --workers 4 --overwrite

Dau ra:
    <output>/<ten_file_goc>.parquet          1 file parquet / 1 file pcap input
    <output>/manifests/<ten_file_goc>.json   manifest rieng cho tung file (sha256, so packet, warning...)
    <output>/processing_summary.jsonl        1 dong / file, ghi append moi lan chay (audit toan batch)
    <output>/pcap_to_parquet.log             log chi tiet
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Thieu pyarrow. Chay: pip install pyarrow", file=sys.stderr)
    raise

try:
    import scapy
    from scapy.utils import RawPcapReader
    from scapy.layers.l2 import Ether, Dot1Q, ARP
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
except ImportError:
    print("Thieu scapy. Chay: pip install scapy", file=sys.stderr)
    raise


# --------------------------------------------------------------------------
# Canonical schema (bam theo muc 6.4 / 22.3 cua ke hoach: provenance, time,
# L2, L3 (co fragment info), L4, size (captured vs wire), application-guess,
# quality). Day la ban thao nhap dau tien cho lop "bronze" - chua co protocol
# transaction parsing sau (Modbus function code, S7comm operation...), phan
# do thuoc WP02 giai doan sau / WP06.
# --------------------------------------------------------------------------
SCHEMA_VERSION = "canonical_packet_v1_draft"
PARSER_NAME = "scapy"
PARSER_VERSION = getattr(scapy, "VERSION", "unknown")

CANONICAL_ARROW_SCHEMA = pa.schema([
    # --- Provenance ---
    ("source_pcap_file", pa.string()),
    ("source_pcap_sha256", pa.string()),
    ("schema_version", pa.string()),
    ("parser_name", pa.string()),
    ("parser_version", pa.string()),
    ("packet_index", pa.uint32()),

    # --- Time ---
    ("capture_timestamp_epoch_ns", pa.int64()),
    ("capture_timestamp_resolution", pa.string()),  # "microsecond" | "nanosecond" | "unknown"

    # --- Layer 2 ---
    ("link_type", pa.int32()),
    ("src_mac", pa.string()),
    ("dst_mac", pa.string()),
    ("dst_mac_is_broadcast", pa.bool_()),
    ("dst_mac_is_multicast", pa.bool_()),
    ("eth_type", pa.uint32()),
    ("vlan_stack", pa.list_(pa.uint16())),

    # --- Layer 3 ---
    ("ip_version", pa.uint8()),
    ("src_ip", pa.string()),
    ("dst_ip", pa.string()),
    ("ip_ttl_or_hoplimit", pa.uint8()),
    ("ip_proto", pa.uint8()),
    ("ip_identification", pa.uint32()),
    ("ip_frag_offset_bytes", pa.uint32()),
    ("ip_flags_mf", pa.bool_()),
    ("ip_flags_df", pa.bool_()),
    ("is_ip_fragment", pa.bool_()),
    ("has_transport_header", pa.bool_()),

    # --- Layer 4 ---
    ("transport_protocol", pa.string()),  # "tcp" | "udp" | "icmp" | "icmpv6" | null
    ("src_port", pa.uint16()),
    ("dst_port", pa.uint16()),
    ("tcp_flags", pa.uint8()),
    ("tcp_seq", pa.uint32()),
    ("tcp_ack", pa.uint32()),
    ("tcp_window", pa.uint16()),
    ("icmp_type", pa.uint8()),
    ("icmp_code", pa.uint8()),

    # --- Application (chi la heuristic theo well-known port, KHONG phai parse
    #     that su. Danh dau ro nguon de khong bi hieu nham la ground truth /
    #     rule co bang chung manh, dung tinh than muc 9.3 va 10.2 cua ke hoach) ---
    ("app_protocol_guess", pa.string()),
    ("app_protocol_guess_source", pa.string()),

    # --- Size ---
    ("wire_length", pa.uint32()),
    ("captured_length", pa.uint32()),
    ("is_truncated", pa.bool_()),

    # --- Payload (giu nguyen byte thuc te sau IP header, hoac sau Ethernet
    #     header neu khong co IP; KHONG gia dinh header lop tren ton tai) ---
    ("payload", pa.binary()),
    ("payload_len", pa.uint32()),

    # --- Quality ---
    ("parse_warning", pa.string()),
])


# Heuristic port -> protocol-guess map. Day CHI la goi y theo well-known port,
# khong phai bang chung xac dinh (mot service co the chay tren port khac).
WELL_KNOWN_OT_IT_PORTS: Dict[int, str] = {
    502: "modbus",
    102: "s7comm",
    20000: "dnp3",
    47808: "bacnet",
    44818: "ethernet_ip",
    2222: "ethernet_ip_io",
    4840: "opcua",
    789: "redlion",
    1962: "profinet_pnio_over_tcp",  # hiem, PROFINET RT thuong la L2 rieng
    18245: "ge_srtp",
    18246: "ge_srtp",
    80: "http",
    443: "https",
    53: "dns",
    67: "dhcp",
    68: "dhcp",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    161: "snmp",
    162: "snmp_trap",
    123: "ntp",
    445: "smb",
    3389: "rdp",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Doc file theo tung chunk (khong nap toan bo vao RAM) de tinh SHA256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def mac_is_broadcast(mac: Optional[str]) -> bool:
    return mac is not None and mac.lower() == "ff:ff:ff:ff:ff:ff"


def mac_is_multicast(mac: Optional[str]) -> bool:
    if not mac:
        return False
    try:
        first_octet = int(mac.split(":")[0], 16)
    except (ValueError, IndexError):
        return False
    return bool(first_octet & 0x01)


def extract_vlan_stack(eth_pkt) -> Tuple[List[int], Any]:
    """Di qua cac lop Dot1Q lien tiep, tra ve (danh sach VLAN id, layer con lai)."""
    vlan_ids: List[int] = []
    layer = eth_pkt.payload
    while layer is not None and layer.__class__.__name__ == "Dot1Q":
        vlan_ids.append(int(layer.vlan))
        layer = layer.payload
    return vlan_ids, layer


def guess_app_protocol(src_port: Optional[int], dst_port: Optional[int]) -> Optional[str]:
    for port in (dst_port, src_port):
        if port is not None and port in WELL_KNOWN_OT_IT_PORTS:
            return WELL_KNOWN_OT_IT_PORTS[port]
    return None


# --------------------------------------------------------------------------
# Raw packet iteration (streaming, ho tro ca classic pcap va pcapng)
# --------------------------------------------------------------------------

def iter_raw_packets(path: Path) -> Iterator[Tuple[int, bytes, int, int, int, str]]:
    """
    Yield (packet_index, raw_bytes, wire_length, captured_length,
           epoch_ns, timestamp_resolution) cho tung packet trong file.

    scapy.utils.RawPcapReader tu dong nhan dien classic pcap vs pcapng qua
    magic number va tra ve doi tuong reader tuong ung (RawPcapReader hoac
    RawPcapNgReader) voi metadata namedtuple khac nhau -> can xu ly ca 2 dang.
    """
    reader = RawPcapReader(str(path))
    try:
        for idx, (data, meta) in enumerate(reader):
            wirelen = int(getattr(meta, "wirelen"))
            caplen = int(getattr(meta, "caplen", len(data)))

            if hasattr(meta, "tsresol"):
                # pcapng: timestamp 64-bit chia (tshigh, tslow), don vi 1/tsresol giay.
                tsresol = int(meta.tsresol)
                ts_units = (int(meta.tshigh) << 32) | int(meta.tslow)
                seconds = ts_units // tsresol
                frac_units = ts_units % tsresol
                if tsresol > 0 and 1_000_000_000 % tsresol == 0:
                    frac_ns = frac_units * (1_000_000_000 // tsresol)
                else:
                    # tsresol khong chia het 1e9 (hiem) -> quy doi gan dung,
                    # danh dau resolution la "unknown" de khong tao do chinh xac gia.
                    frac_ns = int(round(frac_units * (1_000_000_000 / tsresol)))
                epoch_ns = seconds * 1_000_000_000 + frac_ns
                resolution = "nanosecond" if tsresol >= 1_000_000_000 else (
                    "microsecond" if tsresol == 1_000_000 else f"custom_{tsresol}_per_sec"
                )
            else:
                # classic pcap: sec + usec (gia dinh microsecond theo chuan classic pcap;
                # file *.pcapng duoc doc o nhanh tren, khong roi vao day).
                sec = int(getattr(meta, "sec"))
                usec = int(getattr(meta, "usec"))
                epoch_ns = sec * 1_000_000_000 + usec * 1_000
                resolution = "microsecond"

            yield idx, bytes(data), wirelen, caplen, epoch_ns, resolution
    finally:
        try:
            reader.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Per-packet parsing -> canonical record dict
# --------------------------------------------------------------------------

def parse_packet_record(
    packet_index: int,
    raw_bytes: bytes,
    wire_length: int,
    captured_length: int,
    epoch_ns: int,
    ts_resolution: str,
    source_pcap_file: str,
    source_pcap_sha256: str,
    link_type: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "source_pcap_file": source_pcap_file,
        "source_pcap_sha256": source_pcap_sha256,
        "schema_version": SCHEMA_VERSION,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "packet_index": packet_index,
        "capture_timestamp_epoch_ns": epoch_ns,
        "capture_timestamp_resolution": ts_resolution,
        "link_type": link_type,
        "src_mac": None,
        "dst_mac": None,
        "dst_mac_is_broadcast": False,
        "dst_mac_is_multicast": False,
        "eth_type": None,
        "vlan_stack": [],
        "ip_version": None,
        "src_ip": None,
        "dst_ip": None,
        "ip_ttl_or_hoplimit": None,
        "ip_proto": None,
        "ip_identification": None,
        "ip_frag_offset_bytes": None,
        "ip_flags_mf": None,
        "ip_flags_df": None,
        "is_ip_fragment": None,
        "has_transport_header": None,
        "transport_protocol": None,
        "src_port": None,
        "dst_port": None,
        "tcp_flags": None,
        "tcp_seq": None,
        "tcp_ack": None,
        "tcp_window": None,
        "icmp_type": None,
        "icmp_code": None,
        "app_protocol_guess": None,
        "app_protocol_guess_source": None,
        "wire_length": wire_length,
        "captured_length": captured_length,
        "is_truncated": captured_length < wire_length,
        "payload": b"",
        "payload_len": 0,
        "parse_warning": None,
    }

    if link_type != 1:  # 1 = DLT_EN10MB (Ethernet)
        # Khong ep parse Ethernet neu link type khac (vi du Linux cooked capture).
        # Van giu lai raw bytes lam payload de khong mat du lieu quan sat duoc.
        record["parse_warning"] = f"unsupported_link_type_{link_type}"
        record["payload"] = raw_bytes
        record["payload_len"] = len(raw_bytes)
        return record

    try:
        eth_pkt = Ether(raw_bytes)
    except Exception as exc:  # noqa: BLE001 - muon bat moi loi parse 1 packet
        record["parse_warning"] = f"ethernet_parse_error: {exc}"
        record["payload"] = raw_bytes
        record["payload_len"] = len(raw_bytes)
        return record

    record["src_mac"] = str(eth_pkt.src) if eth_pkt.src else None
    record["dst_mac"] = str(eth_pkt.dst) if eth_pkt.dst else None
    record["dst_mac_is_broadcast"] = mac_is_broadcast(record["dst_mac"])
    record["dst_mac_is_multicast"] = mac_is_multicast(record["dst_mac"])

    try:
        vlan_stack, inner_layer = extract_vlan_stack(eth_pkt)
        record["vlan_stack"] = vlan_stack
        # eth_type sau khi bo cac VLAN tag: lay type field cua layer ngay truoc inner_layer.
        # scapy luu san o eth_pkt.type cho truong hop khong VLAN; khi co VLAN,
        # lop Dot1Q cuoi cung co field .type la ethertype thuc su cua payload ben trong.
        if vlan_stack:
            last_dot1q = eth_pkt.payload
            for _ in range(len(vlan_stack) - 1):
                last_dot1q = last_dot1q.payload
            record["eth_type"] = int(last_dot1q.type)
        else:
            record["eth_type"] = int(eth_pkt.type)
    except Exception as exc:  # noqa: BLE001
        record["parse_warning"] = f"vlan_parse_error: {exc}"
        inner_layer = eth_pkt.payload

    ip_layer = None
    ip_version = None
    if inner_layer is not None:
        cls_name = inner_layer.__class__.__name__
        if cls_name == "IP":
            ip_layer = inner_layer
            ip_version = 4
        elif cls_name == "IPv6":
            ip_layer = inner_layer
            ip_version = 6

    if ip_layer is not None:
        record["ip_version"] = ip_version
        record["src_ip"] = str(ip_layer.src)
        record["dst_ip"] = str(ip_layer.dst)

        if ip_version == 4:
            record["ip_ttl_or_hoplimit"] = int(ip_layer.ttl)
            record["ip_proto"] = int(ip_layer.proto)
            record["ip_identification"] = int(ip_layer.id)
            frag_offset_bytes = int(ip_layer.frag) * 8
            record["ip_frag_offset_bytes"] = frag_offset_bytes
            try:
                mf = bool(ip_layer.flags.MF)
                df = bool(ip_layer.flags.DF)
            except Exception:  # noqa: BLE001
                mf, df = False, False
            record["ip_flags_mf"] = mf
            record["ip_flags_df"] = df
            record["is_ip_fragment"] = (frag_offset_bytes > 0) or mf
            # CHI frame co frag_offset == 0 moi mang transport header that su.
            # Day la fix cot loi cho loi da phat hien o bo converter cu:
            # KHONG duoc gia dinh moi packet cung ip_proto deu co du header L4.
            has_header = frag_offset_bytes == 0
            record["has_transport_header"] = has_header
            payload_bytes = bytes(ip_layer.payload)

            if has_header:
                sub_cls = ip_layer.payload.__class__.__name__
                if sub_cls == "TCP":
                    tcp = ip_layer.payload
                    record["transport_protocol"] = "tcp"
                    record["src_port"] = int(tcp.sport)
                    record["dst_port"] = int(tcp.dport)
                    record["tcp_flags"] = int(tcp.flags)
                    record["tcp_seq"] = int(tcp.seq)
                    record["tcp_ack"] = int(tcp.ack)
                    record["tcp_window"] = int(tcp.window)
                elif sub_cls == "UDP":
                    udp = ip_layer.payload
                    record["transport_protocol"] = "udp"
                    record["src_port"] = int(udp.sport)
                    record["dst_port"] = int(udp.dport)
                elif sub_cls == "ICMP":
                    icmp = ip_layer.payload
                    record["transport_protocol"] = "icmp"
                    record["icmp_type"] = int(icmp.type)
                    record["icmp_code"] = int(icmp.code)
                # ip_proto khac (vd GRE, ESP...) : giu ip_proto so, khong ep transport_protocol

            record["app_protocol_guess"] = guess_app_protocol(record["src_port"], record["dst_port"])
            if record["app_protocol_guess"] is not None:
                record["app_protocol_guess_source"] = "well_known_port_heuristic"

            record["payload"] = payload_bytes
            record["payload_len"] = len(payload_bytes)

        else:  # IPv6
            record["ip_ttl_or_hoplimit"] = int(ip_layer.hlim)
            record["ip_proto"] = int(ip_layer.nh)
            # IPv6 fragmentation dung Extension Header rieng (khong lam trong ban thao
            # dau tien nay) -> mac dinh coi la co transport header, danh dau ro trong
            # parse_warning neu that su la fragment de khong ngo nhan.
            record["has_transport_header"] = True
            record["is_ip_fragment"] = False
            payload_bytes = bytes(ip_layer.payload)
            sub_cls = ip_layer.payload.__class__.__name__
            if sub_cls == "TCP":
                tcp = ip_layer.payload
                record["transport_protocol"] = "tcp"
                record["src_port"] = int(tcp.sport)
                record["dst_port"] = int(tcp.dport)
                record["tcp_flags"] = int(tcp.flags)
                record["tcp_seq"] = int(tcp.seq)
                record["tcp_ack"] = int(tcp.ack)
                record["tcp_window"] = int(tcp.window)
            elif sub_cls == "UDP":
                udp = ip_layer.payload
                record["transport_protocol"] = "udp"
                record["src_port"] = int(udp.sport)
                record["dst_port"] = int(udp.dport)
            elif "ICMPv6" in sub_cls:
                record["transport_protocol"] = "icmpv6"

            record["app_protocol_guess"] = guess_app_protocol(record["src_port"], record["dst_port"])
            if record["app_protocol_guess"] is not None:
                record["app_protocol_guess_source"] = "well_known_port_heuristic"

            record["payload"] = payload_bytes
            record["payload_len"] = len(payload_bytes)
    else:
        # Non-IP frame (ARP, LLDP, PROFINET DCP raw L2, v.v.) - KHONG loai bo,
        # giu nguyen phan payload sau Ethernet/VLAN header.
        if inner_layer is not None:
            try:
                payload_bytes = bytes(inner_layer)
            except Exception:  # noqa: BLE001
                payload_bytes = b""
        else:
            payload_bytes = b""
        record["payload"] = payload_bytes
        record["payload_len"] = len(payload_bytes)

    return record


# --------------------------------------------------------------------------
# Xu ly 1 file pcap -> 1 file parquet + 1 manifest
# --------------------------------------------------------------------------

def process_one_file(
    pcap_path: Path,
    output_dir: Path,
    manifests_dir: Path,
    batch_size: int,
    overwrite: bool,
) -> Dict[str, Any]:
    started_at = time.time()
    output_path = output_dir / f"{pcap_path.stem}.parquet"
    manifest_path = manifests_dir / f"{pcap_path.stem}.json"

    manifest: Dict[str, Any] = {
        "source_pcap_path": str(pcap_path),
        "source_pcap_sha256": None,
        "source_pcap_size_bytes": None,
        "output_parquet_path": str(output_path),
        "schema_version": SCHEMA_VERSION,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "total_packets_written": 0,
        "packet_parse_warning_count": 0,
        "fragmented_packet_count": 0,
        "truncated_packet_count": 0,
        "non_ethernet_packet_count": 0,
        "processing_started_utc": None,
        "processing_finished_utc": None,
        "status": "pending",
        "error_message": None,
    }

    if output_path.exists() and manifest_path.exists() and not overwrite:
        return {"status": "skipped", "file": str(pcap_path), "reason": "already_processed"}

    try:
        manifest["source_pcap_size_bytes"] = pcap_path.stat().st_size
        manifest["source_pcap_sha256"] = sha256_of_file(pcap_path)
        manifest["processing_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))

        writer: Optional[pq.ParquetWriter] = None
        batch: List[Dict[str, Any]] = []

        def flush_batch() -> None:
            nonlocal writer, batch
            if not batch:
                return
            table = pa.Table.from_pylist(batch, schema=CANONICAL_ARROW_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), CANONICAL_ARROW_SCHEMA)
            writer.write_table(table)
            batch = []

        for idx, raw_bytes, wirelen, caplen, epoch_ns, resolution in iter_raw_packets(pcap_path):
            try:
                record = parse_packet_record(
                    packet_index=idx,
                    raw_bytes=raw_bytes,
                    wire_length=wirelen,
                    captured_length=caplen,
                    epoch_ns=epoch_ns,
                    ts_resolution=resolution,
                    source_pcap_file=pcap_path.name,
                    source_pcap_sha256=manifest["source_pcap_sha256"],
                    link_type=1,  # RawPcapReader/RawPcapNgReader nay chua expose linktype
                                  # dong nhat qua metadata -> gia dinh Ethernet (pho bien
                                  # cho SPAN capture); goi y cai tien: doc tu Interface
                                  # Description Block cho file pcapng nhieu interface.
                )
            except Exception as exc:  # noqa: BLE001 - 1 packet loi khong duoc lam sap ca file
                record = {
                    "source_pcap_file": pcap_path.name,
                    "source_pcap_sha256": manifest["source_pcap_sha256"],
                    "schema_version": SCHEMA_VERSION,
                    "parser_name": PARSER_NAME,
                    "parser_version": PARSER_VERSION,
                    "packet_index": idx,
                    "capture_timestamp_epoch_ns": epoch_ns,
                    "capture_timestamp_resolution": resolution,
                    "link_type": 1,
                    "wire_length": wirelen,
                    "captured_length": caplen,
                    "is_truncated": caplen < wirelen,
                    "payload": raw_bytes,
                    "payload_len": len(raw_bytes),
                    "parse_warning": f"unhandled_parse_exception: {exc}",
                }
                for field_name in CANONICAL_ARROW_SCHEMA.names:
                    record.setdefault(field_name, None)

            if record.get("parse_warning"):
                manifest["packet_parse_warning_count"] += 1
            if record.get("is_ip_fragment"):
                manifest["fragmented_packet_count"] += 1
            if record.get("is_truncated"):
                manifest["truncated_packet_count"] += 1
            if record.get("ip_version") is None and not (record.get("parse_warning") or "").startswith("unsupported_link_type"):
                pass  # co the la ARP/LLDP hop le, khong tinh la loi

            batch.append(record)
            manifest["total_packets_written"] += 1

            if len(batch) >= batch_size:
                flush_batch()

        flush_batch()
        if writer is None:
            # File rong (0 packet) - van tao 1 file parquet dung schema de nhat quan.
            writer = pq.ParquetWriter(str(output_path), CANONICAL_ARROW_SCHEMA)
        writer.close()

        manifest["status"] = (
            "succeeded_with_warnings" if manifest["packet_parse_warning_count"] > 0 else "succeeded"
        )

    except Exception as exc:  # noqa: BLE001 - loi o muc file (vd magic header sai, file hong)
        manifest["status"] = "failed"
        manifest["error_message"] = f"{exc}\n{traceback.format_exc()}"

    manifest["processing_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


# --------------------------------------------------------------------------
# CLI / batch driver
# --------------------------------------------------------------------------

PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}


def discover_pcap_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = [
        p for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in PCAP_EXTENSIONS
    ]
    return files


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pcap_to_parquet.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert PCAP/PCAPNG files to canonical Parquet.")
    parser.add_argument("--input", required=True, help="File pcap don le hoac thu muc chua nhieu file pcap.")
    parser.add_argument("--output", required=True, help="Thu muc dau ra (se tao neu chua co).")
    parser.add_argument("--batch-size", type=int, default=100_000, help="So packet moi lan flush ra Parquet (mac dinh 100000).")
    parser.add_argument("--workers", type=int, default=1, help="So tien trinh song song xu ly nhieu file (mac dinh 1 = tuan tu).")
    parser.add_argument("--overwrite", action="store_true", help="Xu ly lai ca file da co ket qua truoc do.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    manifests_dir = output_dir / "manifests"
    summary_path = output_dir / "processing_summary.jsonl"

    setup_logging(output_dir)
    log = logging.getLogger(__name__)

    if not input_path.exists():
        log.error("Duong dan input khong ton tai: %s", input_path)
        sys.exit(1)

    pcap_files = discover_pcap_files(input_path)
    log.info("Tim thay %d file pcap/pcapng can xu ly.", len(pcap_files))
    if not pcap_files:
        log.warning("Khong co file nao phu hop (%s) trong %s", PCAP_EXTENSIONS, input_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for pcap_path in pcap_files:
            log.info("Dang xu ly: %s", pcap_path)
            result = process_one_file(pcap_path, output_dir, manifests_dir, args.batch_size, args.overwrite)
            log.info("  -> %s (status=%s)", pcap_path.name, result.get("status"))
            results.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(
                    process_one_file, pcap_path, output_dir, manifests_dir, args.batch_size, args.overwrite
                ): pcap_path
                for pcap_path in pcap_files
            }
            for future in concurrent.futures.as_completed(future_to_path):
                pcap_path = future_to_path[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "failed", "file": str(pcap_path), "error_message": str(exc)}
                log.info("  -> %s (status=%s)", pcap_path.name, result.get("status"))
                results.append(result)

    with open(summary_path, "a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    status_counts: Dict[str, int] = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    log.info("Hoan tat. Tong hop trang thai: %s", status_counts)
    log.info("Chi tiet tung file: %s", summary_path)


if __name__ == "__main__":
    main()
