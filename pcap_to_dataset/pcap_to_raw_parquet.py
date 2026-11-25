#!/usr/bin/env python3
r"""
pcap_to_raw_parquet.py

Batch converter: PCAP/PCAPNG -> RAW Parquet, KHONG diễn giải protocol.
Đây là lớp "raw/" trong kiến trúc 4 lớp của ke hoach (muc 4.2, 6.3):
"raw/ PCAP bat bien + capture manifest" - tach biet hoan toan khoi lop
"bronze/" da parse (xem pcap_to_parquet.py, script rieng, KHONG bi thay the
boi file nay).

Nguyen tac thiet ke - "mapping 1:1, khong xu ly":
  - KHONG tach Ethernet/IP/TCP/UDP. Moi dong = 1 packet/record, chi chua
    dung nhung gi ban than file pcap da luu san: timestamp tho, do dai
    goc/do dai capture, va TOAN BO byte cua block/record (raw_record_bytes)
    dung y nguyen nhu tren dia - khong sua, khong cat, khong dien giai.
  - Cot "capture_timestamp_epoch_ns" la tien ich duy nhat duoc suy ra (quy
    doi thuan nghich, khong mat do chinh xac, tu chinh cac truong tho da
    luu cung dong) - khong dong nghia voi viec da "parse" du lieu goi tin.
  - Dam bao toan ven bang 2 co che doc lap:
      1) SHA256 cua file pcap goc, tinh TRUOC khi xu ly, luu vao manifest.
      2) Round-trip verify: sau khi convert, TAI TAO LAI chinh xac file
         pcap tu parquet + manifest (ghep lai dung thu tu goc), tinh SHA256
         cua ban tai tao, so voi SHA256 goc. Ket qua PASS/FAIL duoc ghi vao
         manifest - day la bang chung toan ven mang tinh chung minh duoc,
         khong chi la loi khang dinh.

Ho tro ca 2 dinh dang:
  - Classic pcap (magic 0xa1b2c3d4/0xd4c3b2a1 microsecond, 0xa1b23c4d/
    0x4d3cb2a1 nanosecond): global header 24 byte + lap (16-byte record
    header + du lieu).
  - PCAPNG: doc theo tung "block" tu mo ta (Block Type + Block Total
    Length + body + Block Total Length lap lai o cuoi - tu kiem tra duoc
    2 gia tri length co khop khong). Block la Enhanced/Simple/obsolete
    Packet Block duoc coi la "packet"; moi block khac (Section Header,
    Interface Description, Name Resolution, Interface Statistics...)
    duoc luu nguyen ven lam "structural block" trong manifest de tai tao
    dung vi tri khi ghep lai.

KHONG can scapy/tshark - script nay tu doc byte theo dung dac ta dinh dang
pcap/pcapng (chi dung struct cua Python chuan), giam phu thuoc khi mang
sang may khac chay.

Cai dat: pip install pyarrow  (xem requirements.txt)

Vi du chay:
    python pcap_to_raw_parquet.py --input D:\pcap_source --output D:\raw_dataset_out
    python pcap_to_raw_parquet.py --input D:\pcap_source --output D:\raw_dataset_out --workers 4 --overwrite
    python pcap_to_raw_parquet.py --input D:\pcap_source --output D:\raw_dataset_out --no-verify   # bo qua round-trip (nhanh hon)

Dau ra:
    <output>/<ten_file_goc>.parquet          1 file parquet / 1 file pcap input
    <output>/manifests/<ten_file_goc>.json   manifest rieng (sha256, structural blocks, ket qua round-trip verify...)
    <output>/processing_summary.jsonl        1 dong / file, ghi append moi lan chay
    <output>/pcap_to_raw_parquet.log         log chi tiet
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import struct
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


SCHEMA_VERSION = "raw_packet_v1"
PARSER_NAME = "pcap_to_raw_parquet_custom_walker"
PARSER_VERSION = "1.0.0"

# ---- Magic numbers / block types theo dac ta pcap va pcapng ----
PCAP_MAGIC_MICRO_LE = 0xA1B2C3D4
PCAP_MAGIC_MICRO_BE = 0xD4C3B2A1
PCAP_MAGIC_NANO_LE = 0xA1B23C4D
PCAP_MAGIC_NANO_BE = 0x4D3CB2A1
PCAPNG_SHB_TYPE = 0x0A0D0D0A
PCAPNG_BOM_LE = b"\x4d\x3c\x2b\x1a"  # doc theo little-endian ra dung 0x1A2B3C4D
PCAPNG_BOM_BE = b"\x1a\x2b\x3c\x4d"  # doc theo big-endian ra dung 0x1A2B3C4D

BLOCK_TYPE_IDB = 0x00000001
BLOCK_TYPE_OBSOLETE_PB = 0x00000002
BLOCK_TYPE_SPB = 0x00000003
BLOCK_TYPE_EPB = 0x00000006
PACKET_BLOCK_TYPES = {BLOCK_TYPE_OBSOLETE_PB, BLOCK_TYPE_SPB, BLOCK_TYPE_EPB}


RAW_ARROW_SCHEMA = pa.schema([
    ("source_pcap_file", pa.string()),
    ("source_pcap_sha256", pa.string()),
    ("schema_version", pa.string()),
    ("parser_name", pa.string()),
    ("parser_version", pa.string()),

    ("sequence_position", pa.uint32()),   # vi tri toan cuc trong file (tinh ca block phi-packet)
    ("packet_index", pa.uint32()),        # chi so packet (null neu day la block phi-packet - khong xay ra trong output vi ta chi ghi hang la packet)
    ("capture_format", pa.string()),      # "pcap_classic" | "pcapng"
    ("block_type_hex", pa.string()),      # chi co pcapng

    # Timestamp: 2 truong tho dung nhu file luu + 1 cot tien ich quy doi.
    ("ts_raw_field_1", pa.int64()),       # classic: ts_sec | pcapng: ts_high
    ("ts_raw_field_2", pa.int64()),       # classic: ts_usec_or_nsec | pcapng: ts_low
    ("ts_resolution_units_per_second", pa.int64()),
    ("capture_timestamp_epoch_ns", pa.int64()),

    ("orig_len", pa.uint32()),            # do dai that tren day (wire length)
    ("incl_len", pa.uint32()),            # do dai duoc ghi trong file (co the < orig_len neu bi snaplen cat)
    ("is_length_mismatch", pa.bool_()),   # orig_len != incl_len

    ("raw_record_bytes", pa.binary()),    # TOAN BO byte cua block/record, y nguyen nhu tren dia
    ("raw_record_len", pa.uint32()),

    ("integrity_note", pa.string()),      # canh bao neu co (vd block length header/footer khong khop)
])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_epoch_ns(sec: int, frac_units: int, units_per_second: int) -> Optional[int]:
    if units_per_second is None or units_per_second <= 0:
        return None
    if 1_000_000_000 % units_per_second == 0:
        frac_ns = frac_units * (1_000_000_000 // units_per_second)
    else:
        # do phan giai khong chia het 1e9 (hiem) -> quy doi gan dung, van
        # giu nguyen 2 truong tho o cot rieng de khong mat thong tin goc.
        frac_ns = int(round(frac_units * (1_000_000_000 / units_per_second)))
    return sec * 1_000_000_000 + frac_ns


def parse_idb_tsresol(idb_full_block_bytes: bytes, endian: str) -> int:
    """Doc option if_tsresol (code=9) trong Interface Description Block.
    Mac dinh 1_000_000 (microsecond) theo dung dac ta pcapng neu khong co
    option nay hoac parse that bai - khong bao gio nem loi ra ngoai."""
    try:
        body = idb_full_block_bytes[8:-4]  # bo type(4)+totallen(4) dau, totallen(4) cuoi
        pos = 8  # bo qua LinkType(2)+Reserved(2)+SnapLen(4) truoc options
        while pos + 4 <= len(body):
            opt_code, opt_len = struct.unpack(endian + "HH", body[pos:pos + 4])
            if opt_code == 0 and opt_len == 0:
                break
            val_start = pos + 4
            if opt_code == 9 and opt_len == 1 and val_start < len(body):
                b = body[val_start]
                return (2 ** (b & 0x7F)) if (b & 0x80) else (10 ** b)
            padded_len = (opt_len + 3) // 4 * 4
            pos = val_start + padded_len
    except Exception:  # noqa: BLE001
        pass
    return 1_000_000


# --------------------------------------------------------------------------
# Classic pcap walker
# --------------------------------------------------------------------------

def _sniff_classic_endian(magic_le_value: int) -> Optional[Tuple[str, bool]]:
    if magic_le_value == PCAP_MAGIC_MICRO_LE:
        return "<", False
    if magic_le_value == PCAP_MAGIC_MICRO_BE:
        return ">", False
    if magic_le_value == PCAP_MAGIC_NANO_LE:
        return "<", True
    if magic_le_value == PCAP_MAGIC_NANO_BE:
        return ">", True
    return None


def walk_classic_pcap(f, endian: str, ts_is_nsec: bool) -> Iterator[Dict[str, Any]]:
    seq = 0
    pkt_idx = 0
    resolution = 1_000_000_000 if ts_is_nsec else 1_000_000
    while True:
        header_bytes = f.read(16)
        if len(header_bytes) == 0:
            break
        if len(header_bytes) < 16:
            yield {
                "sequence_position": seq, "packet_index": pkt_idx, "block_type_hex": None,
                "ts_raw_field_1": None, "ts_raw_field_2": None,
                "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                "raw_record_bytes": header_bytes, "raw_record_len": len(header_bytes),
                "integrity_note": "truncated_record_header_at_eof",
            }
            break

        ts_sec, ts_frac, incl_len, orig_len = struct.unpack(endian + "IIII", header_bytes)
        data = f.read(incl_len)
        note = None
        if len(data) < incl_len:
            note = "truncated_packet_data_at_eof"

        record_bytes = header_bytes + data
        yield {
            "sequence_position": seq,
            "packet_index": pkt_idx,
            "block_type_hex": None,
            "ts_raw_field_1": ts_sec,
            "ts_raw_field_2": ts_frac,
            "ts_resolution_units_per_second": resolution,
            "capture_timestamp_epoch_ns": compute_epoch_ns(ts_sec, ts_frac, resolution),
            "orig_len": orig_len,
            "incl_len": incl_len,
            "is_length_mismatch": orig_len != incl_len,
            "raw_record_bytes": record_bytes,
            "raw_record_len": len(record_bytes),
            "integrity_note": note,
        }
        seq += 1
        pkt_idx += 1
        if note:
            break


# --------------------------------------------------------------------------
# PCAPNG walker
# --------------------------------------------------------------------------

def walk_pcapng(f) -> Iterator[Dict[str, Any]]:
    f.seek(0)
    seq = 0
    pkt_idx = 0
    endian: Optional[str] = None
    interface_tsresol: List[int] = []

    while True:
        type_bytes = f.read(4)
        if len(type_bytes) == 0:
            break
        if len(type_bytes) < 4:
            yield {
                "sequence_position": seq, "packet_index": None, "block_type_hex": None,
                "ts_raw_field_1": None, "ts_raw_field_2": None,
                "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                "raw_record_bytes": type_bytes, "raw_record_len": len(type_bytes),
                "integrity_note": "truncated_at_eof_reading_block_type",
            }
            break

        # Khong bao gio seek() lui lai de "doc lai" - moi byte da doc duoc
        # cong don thang vao header_consumed/full_block_bytes, tranh loi
        # doc chong/doc nham do quan ly con tro file thu cong.
        is_shb_candidate = type_bytes == struct.pack("<I", PCAPNG_SHB_TYPE)  # palindrome -> dung cho ca 2 endian

        if is_shb_candidate:
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                yield {
                    "sequence_position": seq, "packet_index": None, "block_type_hex": "0x0a0d0d0a",
                    "ts_raw_field_1": None, "ts_raw_field_2": None,
                    "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                    "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                    "raw_record_bytes": type_bytes + len_bytes, "raw_record_len": len(type_bytes) + len(len_bytes),
                    "integrity_note": "truncated_at_eof_reading_block_length",
                }
                break
            bom_bytes = f.read(4)
            if len(bom_bytes) < 4:
                yield {
                    "sequence_position": seq, "packet_index": None, "block_type_hex": "0x0a0d0d0a",
                    "ts_raw_field_1": None, "ts_raw_field_2": None,
                    "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                    "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                    "raw_record_bytes": type_bytes + len_bytes + bom_bytes,
                    "raw_record_len": len(type_bytes) + len(len_bytes) + len(bom_bytes),
                    "integrity_note": "truncated_at_eof_reading_byte_order_magic",
                }
                break
            if bom_bytes == PCAPNG_BOM_LE:
                endian = "<"
            elif bom_bytes == PCAPNG_BOM_BE:
                endian = ">"
            else:
                yield {
                    "sequence_position": seq, "packet_index": None, "block_type_hex": "0x0a0d0d0a",
                    "ts_raw_field_1": None, "ts_raw_field_2": None,
                    "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                    "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                    "raw_record_bytes": type_bytes + len_bytes + bom_bytes,
                    "raw_record_len": 12,
                    "integrity_note": "invalid_byte_order_magic_in_section_header_block",
                }
                break
            interface_tsresol = []  # section moi -> reset danh sach interface
            block_type_int = struct.unpack(endian + "I", type_bytes)[0]
            block_total_len = struct.unpack(endian + "I", len_bytes)[0]
            header_consumed = type_bytes + len_bytes + bom_bytes  # da doc 12 byte dau cua block
        else:
            if endian is None:
                yield {
                    "sequence_position": seq, "packet_index": None, "block_type_hex": None,
                    "ts_raw_field_1": None, "ts_raw_field_2": None,
                    "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                    "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                    "raw_record_bytes": type_bytes, "raw_record_len": len(type_bytes),
                    "integrity_note": "block_before_first_section_header_block",
                }
                break
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                yield {
                    "sequence_position": seq, "packet_index": None, "block_type_hex": None,
                    "ts_raw_field_1": None, "ts_raw_field_2": None,
                    "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                    "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                    "raw_record_bytes": type_bytes + len_bytes, "raw_record_len": len(type_bytes) + len(len_bytes),
                    "integrity_note": "truncated_at_eof_reading_block_length",
                }
                break
            block_type_int = struct.unpack(endian + "I", type_bytes)[0]
            block_total_len = struct.unpack(endian + "I", len_bytes)[0]
            header_consumed = type_bytes + len_bytes  # da doc 8 byte dau cua block

        if block_total_len < len(header_consumed) + 4:
            remaining = f.read()
            yield {
                "sequence_position": seq, "packet_index": None,
                "block_type_hex": f"0x{block_type_int:08x}",
                "ts_raw_field_1": None, "ts_raw_field_2": None,
                "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                "raw_record_bytes": header_consumed + remaining, "raw_record_len": len(header_consumed) + len(remaining),
                "integrity_note": f"invalid_block_total_length_{block_total_len}",
            }
            break

        rest_bytes = f.read(block_total_len - len(header_consumed))
        full_block_bytes = header_consumed + rest_bytes
        note = None
        if len(rest_bytes) < block_total_len - len(header_consumed):
            note = "truncated_block_body_at_eof"
            yield {
                "sequence_position": seq, "packet_index": None,
                "block_type_hex": f"0x{block_type_int:08x}",
                "ts_raw_field_1": None, "ts_raw_field_2": None,
                "ts_resolution_units_per_second": None, "capture_timestamp_epoch_ns": None,
                "orig_len": None, "incl_len": None, "is_length_mismatch": None,
                "raw_record_bytes": full_block_bytes, "raw_record_len": len(full_block_bytes),
                "integrity_note": note,
            }
            break

        trailing_len = struct.unpack(endian + "I", full_block_bytes[-4:])[0]
        if trailing_len != block_total_len:
            note = "block_length_header_footer_mismatch"

        record: Dict[str, Any] = {
            "sequence_position": seq,
            "packet_index": None,
            "block_type_hex": f"0x{block_type_int:08x}",
            "ts_raw_field_1": None,
            "ts_raw_field_2": None,
            "ts_resolution_units_per_second": None,
            "capture_timestamp_epoch_ns": None,
            "orig_len": None,
            "incl_len": None,
            "is_length_mismatch": None,
            "raw_record_bytes": full_block_bytes,
            "raw_record_len": len(full_block_bytes),
            "integrity_note": note,
        }

        if block_type_int == BLOCK_TYPE_IDB:
            interface_tsresol.append(parse_idb_tsresol(full_block_bytes, endian))

        elif block_type_int == BLOCK_TYPE_EPB:
            try:
                body = full_block_bytes[8:-4]
                iface_id = struct.unpack(endian + "I", body[0:4])[0]
                ts_high = struct.unpack(endian + "I", body[4:8])[0]
                ts_low = struct.unpack(endian + "I", body[8:12])[0]
                incl_len = struct.unpack(endian + "I", body[12:16])[0]
                orig_len = struct.unpack(endian + "I", body[16:20])[0]
                tsresol = interface_tsresol[iface_id] if 0 <= iface_id < len(interface_tsresol) else 1_000_000
                ts_units = (ts_high << 32) | ts_low
                seconds = ts_units // tsresol if tsresol > 0 else 0
                frac_units = ts_units % tsresol if tsresol > 0 else 0
                record.update({
                    "packet_index": pkt_idx,
                    "ts_raw_field_1": ts_high,
                    "ts_raw_field_2": ts_low,
                    "ts_resolution_units_per_second": tsresol,
                    "capture_timestamp_epoch_ns": compute_epoch_ns(seconds, frac_units, tsresol),
                    "orig_len": orig_len,
                    "incl_len": incl_len,
                    "is_length_mismatch": orig_len != incl_len,
                })
                pkt_idx += 1
            except Exception as exc:  # noqa: BLE001
                record["integrity_note"] = f"epb_field_parse_error: {exc}"
                record["packet_index"] = pkt_idx
                pkt_idx += 1

        elif block_type_int == BLOCK_TYPE_OBSOLETE_PB:
            try:
                body = full_block_bytes[8:-4]
                iface_id = struct.unpack(endian + "H", body[0:2])[0]
                incl_len = struct.unpack(endian + "I", body[4:8])[0]
                orig_len = struct.unpack(endian + "I", body[8:12])[0]
                record.update({
                    "packet_index": pkt_idx,
                    "incl_len": incl_len,
                    "orig_len": orig_len,
                    "is_length_mismatch": orig_len != incl_len,
                })
                pkt_idx += 1
            except Exception as exc:  # noqa: BLE001
                record["integrity_note"] = f"pb_field_parse_error: {exc}"
                record["packet_index"] = pkt_idx
                pkt_idx += 1

        elif block_type_int == BLOCK_TYPE_SPB:
            # Simple Packet Block: khong co timestamp/interface rieng, chi
            # co Original Packet Length. Giu nguyen raw_record_bytes (luon
            # dung), danh dau ro metadata bi han che.
            record["packet_index"] = pkt_idx
            record["integrity_note"] = (record["integrity_note"] or "") + ";simple_packet_block_limited_metadata"
            pkt_idx += 1

        yield record
        seq += 1


# --------------------------------------------------------------------------
# Dispatch: doc format, tach preamble/structural blocks rieng khoi packet rows
# --------------------------------------------------------------------------

def process_one_file(
    pcap_path: Path,
    output_dir: Path,
    manifests_dir: Path,
    batch_size: int,
    overwrite: bool,
    verify_round_trip: bool,
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
        "capture_format": None,
        "file_preamble_hex": None,          # classic pcap: 24-byte global header
        "structural_blocks": [],            # pcapng: cac block khong phai packet, giu nguyen vi tri
        "total_packet_rows": 0,
        "total_structural_blocks": 0,
        "integrity_warning_count": 0,
        "round_trip_verify": {"attempted": False},
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

        with open(pcap_path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                raise ValueError("file_too_short_to_sniff_format")
            magic_le_value = struct.unpack("<I", magic_bytes)[0]

            writer: Optional[pq.ParquetWriter] = None
            batch: List[Dict[str, Any]] = []

            def flush_batch() -> None:
                nonlocal writer, batch
                if not batch:
                    return
                for rec in batch:
                    rec.setdefault("source_pcap_file", pcap_path.name)
                    rec.setdefault("source_pcap_sha256", manifest["source_pcap_sha256"])
                    rec.setdefault("schema_version", SCHEMA_VERSION)
                    rec.setdefault("parser_name", PARSER_NAME)
                    rec.setdefault("parser_version", PARSER_VERSION)
                    rec.setdefault("capture_format", manifest["capture_format"])
                table = pa.Table.from_pylist(batch, schema=RAW_ARROW_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_path), RAW_ARROW_SCHEMA)
                writer.write_table(table)
                batch = []

            if magic_le_value == PCAPNG_SHB_TYPE:
                manifest["capture_format"] = "pcapng"
                f.seek(0)
                for rec in walk_pcapng(f):
                    if rec.get("integrity_note"):
                        manifest["integrity_warning_count"] += 1
                    if rec.get("packet_index") is not None:
                        batch.append(rec)
                        manifest["total_packet_rows"] += 1
                        if len(batch) >= batch_size:
                            flush_batch()
                    else:
                        manifest["structural_blocks"].append({
                            "sequence_position": rec["sequence_position"],
                            "block_type_hex": rec.get("block_type_hex"),
                            "raw_bytes_hex": rec["raw_record_bytes"].hex(),
                            "integrity_note": rec.get("integrity_note"),
                        })
                        manifest["total_structural_blocks"] += 1
            else:
                classic = _sniff_classic_endian(magic_le_value)
                if classic is None:
                    raise ValueError(f"unrecognized_magic_bytes: {magic_bytes.hex()}")
                endian, ts_is_nsec = classic
                f.seek(0)
                preamble_bytes = f.read(24)
                if len(preamble_bytes) < 24:
                    raise ValueError("file_too_short_for_classic_global_header")
                manifest["capture_format"] = "pcap_classic"
                manifest["file_preamble_hex"] = preamble_bytes.hex()
                for rec in walk_classic_pcap(f, endian, ts_is_nsec):
                    if rec.get("integrity_note"):
                        manifest["integrity_warning_count"] += 1
                    batch.append(rec)
                    manifest["total_packet_rows"] += 1
                    if len(batch) >= batch_size:
                        flush_batch()

            flush_batch()
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), RAW_ARROW_SCHEMA)
            writer.close()

        manifest["status"] = (
            "succeeded_with_warnings" if manifest["integrity_warning_count"] > 0 else "succeeded"
        )

        if verify_round_trip:
            manifest["round_trip_verify"] = reconstruct_and_verify(output_path, manifest)
            if manifest["round_trip_verify"].get("attempted") and not manifest["round_trip_verify"].get("matches_source"):
                manifest["status"] = "failed"
                manifest["error_message"] = "round_trip_verify_mismatch: du lieu tai tao khong khop SHA256 goc"

    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "failed"
        manifest["error_message"] = f"{exc}\n{traceback.format_exc()}"

    manifest["processing_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


# --------------------------------------------------------------------------
# Round-trip verify: tai tao lai file goc tu parquet + manifest, so SHA256
# --------------------------------------------------------------------------

def reconstruct_and_verify(parquet_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    hasher = hashlib.sha256()
    try:
        pf = pq.ParquetFile(str(parquet_path))
        if manifest["capture_format"] == "pcap_classic":
            hasher.update(bytes.fromhex(manifest["file_preamble_hex"]))
            for record_batch in pf.iter_batches(batch_size=10_000, columns=["sequence_position", "raw_record_bytes"]):
                seq_col = record_batch.column("sequence_position").to_pylist()
                bytes_col = record_batch.column("raw_record_bytes").to_pylist()
                for _, b in sorted(zip(seq_col, bytes_col), key=lambda x: x[0]):
                    hasher.update(b)
        else:
            structural_sorted = sorted(manifest["structural_blocks"], key=lambda b: b["sequence_position"])
            struct_iter = iter(structural_sorted)
            next_struct = next(struct_iter, None)
            for record_batch in pf.iter_batches(batch_size=10_000, columns=["sequence_position", "raw_record_bytes"]):
                seq_col = record_batch.column("sequence_position").to_pylist()
                bytes_col = record_batch.column("raw_record_bytes").to_pylist()
                for seq, b in zip(seq_col, bytes_col):
                    while next_struct is not None and next_struct["sequence_position"] < seq:
                        hasher.update(bytes.fromhex(next_struct["raw_bytes_hex"]))
                        next_struct = next(struct_iter, None)
                    hasher.update(b)
            while next_struct is not None:
                hasher.update(bytes.fromhex(next_struct["raw_bytes_hex"]))
                next_struct = next(struct_iter, None)

        reconstructed_sha256 = hasher.hexdigest()
        matches = reconstructed_sha256 == manifest["source_pcap_sha256"]
        return {
            "attempted": True,
            "reconstructed_sha256": reconstructed_sha256,
            "source_sha256": manifest["source_pcap_sha256"],
            "matches_source": matches,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "attempted": True,
            "matches_source": False,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


# --------------------------------------------------------------------------
# CLI / batch driver
# --------------------------------------------------------------------------

PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}


def discover_pcap_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return [
        p for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in PCAP_EXTENSIONS
    ]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pcap_to_raw_parquet.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert PCAP/PCAPNG to RAW Parquet (khong dien giai protocol).")
    parser.add_argument("--input", required=True, help="File pcap don le hoac thu muc chua nhieu file pcap.")
    parser.add_argument("--output", required=True, help="Thu muc dau ra (se tao neu chua co).")
    parser.add_argument("--batch-size", type=int, default=100_000, help="So packet moi lan flush ra Parquet.")
    parser.add_argument("--workers", type=int, default=1, help="So tien trinh song song xu ly nhieu file.")
    parser.add_argument("--overwrite", action="store_true", help="Xu ly lai ca file da co ket qua truoc do.")
    parser.add_argument("--no-verify", action="store_true", help="Bo qua buoc round-trip verify (nhanh hon, giam bang chung toan ven).")
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
    verify_round_trip = not args.no_verify

    results: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for pcap_path in pcap_files:
            log.info("Dang xu ly: %s", pcap_path)
            result = process_one_file(pcap_path, output_dir, manifests_dir, args.batch_size, args.overwrite, verify_round_trip)
            rt = result.get("round_trip_verify", {})
            log.info(
                "  -> %s (status=%s, round_trip_match=%s)",
                pcap_path.name, result.get("status"), rt.get("matches_source"),
            )
            results.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(
                    process_one_file, pcap_path, output_dir, manifests_dir, args.batch_size, args.overwrite, verify_round_trip
                ): pcap_path
                for pcap_path in pcap_files
            }
            for future in concurrent.futures.as_completed(future_to_path):
                pcap_path = future_to_path[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "failed", "file": str(pcap_path), "error_message": str(exc)}
                rt = result.get("round_trip_verify", {})
                log.info(
                    "  -> %s (status=%s, round_trip_match=%s)",
                    pcap_path.name, result.get("status"), rt.get("matches_source"),
                )
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
