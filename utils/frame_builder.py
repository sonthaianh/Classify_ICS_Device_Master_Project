#!/usr/bin/env python3
r"""
frame_builder.py

Tai tao (synthetic) Ethernet+IP+TCP/UDP frame tu 1 dong du lieu da tach lop san
(schema hien co cua raw_parquet/: src_mac, dst_mac, eth_type, src_ip, dst_ip,
ip_proto, ip_ttl, src_port, dst_port, tcp_flags, app_payload - xem audit trong
dpi_enrich.py). Dung chung cho ca webapp demo (Tools/webapp/app.py) va batch
pipeline (Tools/pcap_to_dataset/dpi_enrich.py) de khong lap logic.

VI SAO CAN TAI TAO FRAME (khong dissect thang tu app_payload):
    Dissector cua Wireshark (vd packet-mbtcp.c) can biet port TCP/UDP that su
    (bang dissector_table "tcp.port"/"udp.port") de biet ap dung dissector nao,
    va can 1 "khung" pcap hop le (Ethernet/IP/TCP) de tshark bat dau dissect.
    app_payload trong dataset hien tai CHI la byte sau TCP/UDP header (da xac
    minh: mot goi Modbus/TCP co app_payload bat dau thang bang MBAP header),
    nen phai boc lai header truoc khi dua vao tshark.

GIOI HAN CAN GHI RO (theo tinh than khong giau gia dinh cua ke hoach luan van):
    - Header L2/L3/L4 la TAI TAO, khong phai byte goc tren day (dataset hien
      tai khong luu raw_record_bytes, chi luu field da tach). Checksum khong
      duoc tinh.
    - TCP sequence number duoc tinh RIENG cho tung chieu cua tung flow
      (5-tuple), tang dan theo do dai app_payload da gui truoc do trong CUNG
      lan chay nay - de ho tro dung TCP desegmentation (packet-mbtcp.c dung
      tcp_dissect_pdus()) khi 1 PDU Modbus trai qua nhieu packet. Neu chi xu
      ly MOT PHAN cua 1 flow (vi du webapp demo chi lay N dong dau), seq co
      the khong khop lien tuc voi phan con lai bi cat - dissector se bao loi
      desegmentation cho packet giap ranh, khong anh huong cac packet khac.
    - Flag TCP (SYN/ACK/PSH/FIN...) duoc LAY THAT tu cot tcp_flags, khong bia.
    - VLAN tag (vlan_id) CHUA duoc chen lai trong ban v1 - frame luon la
      Ethernet II don gian (EtherType IPv4 truc tiep), ke ca khi dong goc co
      vlan_id. Ghi nhan day la gioi han bo sung sau.
    - Chi ho tro IPv4 + TCP/UDP trong ban v1 (ip_proto in {6, 17}). Cac dong
      khac (ARP, ICMP, IPv6, VLAN...) duoc BAO CAO rieng (skip_reason), khong
      bi am tham bo qua.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterable, List, Optional, Tuple

DLT_ETHERNET = 1
ETHERTYPE_IPV4 = 0x0800

SUPPORTED_IP_PROTO = {6, 17}  # TCP, UDP


class FlowSeqTracker:
    """Theo doi TCP sequence number tang dan rieng cho tung chieu cua tung
    flow (5-tuple), de nhieu packet cung 1 flow duoc dissect nhu 1 luong TCP
    lien tuc (ho tro desegmentation dung khi 1 PDU trai qua nhieu packet)."""

    def __init__(self) -> None:
        self._next_seq: Dict[Tuple[str, int, str, int, int], int] = {}

    def next_seq(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int,
                 ip_proto: int, payload_len: int) -> int:
        key = (src_ip, int(src_port), dst_ip, int(dst_port), int(ip_proto))
        seq = self._next_seq.get(key, 1)
        self._next_seq[key] = seq + payload_len
        return seq


def _mac_str_to_bytes(mac: Optional[str]) -> bytes:
    if not mac:
        return b"\x00\x00\x00\x00\x00\x00"
    try:
        return bytes(int(part, 16) for part in mac.split(":"))
    except (ValueError, AttributeError):
        return b"\x00\x00\x00\x00\x00\x00"


def _ip_str_to_bytes(ip: Optional[str]) -> Optional[bytes]:
    if not ip:
        return None
    try:
        parts = [int(p) for p in ip.split(".")]
        if len(parts) != 4 or any(p < 0 or p > 255 for p in parts):
            return None
        return bytes(parts)
    except (ValueError, AttributeError):
        return None


def build_frame_for_row(row: Dict[str, Any], seq_tracker: FlowSeqTracker) -> Tuple[Optional[bytes], Optional[str]]:
    """Tai tao 1 frame Ethernet+IP+TCP/UDP tu 1 dong dataset.

    Tra ve (frame_bytes, skip_reason). Neu khong the tai tao (protocol chua
    ho tro, thieu du lieu bat buoc...), frame_bytes = None va skip_reason
    ghi ro ly do (KHONG bao gio am tham bo qua ma khong ghi nhan)."""
    ip_proto = row.get("ip_proto")
    src_ip = row.get("src_ip")
    dst_ip = row.get("dst_ip")
    app_payload = row.get("app_payload") or b""

    if ip_proto is None or (isinstance(ip_proto, float) and ip_proto != ip_proto):  # NaN check
        return None, "missing_ip_proto"
    ip_proto = int(ip_proto)

    if ip_proto not in SUPPORTED_IP_PROTO:
        return None, f"unsupported_ip_proto_{ip_proto}_v1_only_handles_tcp_udp"

    src_ip_bytes = _ip_str_to_bytes(src_ip)
    dst_ip_bytes = _ip_str_to_bytes(dst_ip)
    if src_ip_bytes is None or dst_ip_bytes is None:
        return None, "missing_or_invalid_ipv4_address"

    src_port = row.get("src_port")
    dst_port = row.get("dst_port")
    if src_port is None or dst_port is None:
        return None, "missing_transport_port"
    src_port = int(src_port)
    dst_port = int(dst_port)

    src_mac = _mac_str_to_bytes(row.get("src_mac"))
    dst_mac = _mac_str_to_bytes(row.get("dst_mac"))
    eth_type = int(row.get("eth_type") or ETHERTYPE_IPV4)

    if ip_proto == 6:
        tcp_flags = int(row.get("tcp_flags") or 0)
        seq = seq_tracker.next_seq(src_ip, src_port, dst_ip, dst_port, ip_proto, len(app_payload))
        l4 = struct.pack(
            "!HHIIBBHHH",
            src_port, dst_port,
            seq, 0,               # seq (that, theo flow), ack (khong tai tao - khong bat buoc de dissect)
            (5 << 4),              # data offset: 20-byte header, khong option
            tcp_flags,
            int(row.get("tcp_window") or 64240),
            0,                     # checksum khong tinh (khong bat buoc de dissect)
            0,                     # urgent pointer
        ) + app_payload
    else:  # UDP
        l4 = struct.pack("!HHHH", src_port, dst_port, 8 + len(app_payload), 0) + app_payload

    ttl = int(row.get("ip_ttl") or 64)
    total_len = 20 + len(l4)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0,
        total_len,
        0,      # identification (khong tai tao - khong anh huong dissect Modbus/S7comm o tang tren)
        0,      # flags/fragment offset
        ttl,
        ip_proto,
        0,      # checksum khong tinh
        src_ip_bytes, dst_ip_bytes,
    )

    eth_hdr = dst_mac + src_mac + struct.pack("!H", eth_type)
    return eth_hdr + ip_hdr + l4, None


def build_frames_for_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[bytes], List[int], List[Dict[str, Any]]]:
    """Ap dung build_frame_for_row() cho nhieu dong, GIU DUNG THU TU dau vao
    (quan trong de bao toan conversation/transaction context khi dissect).

    Tra ve:
        frames            - list frame bytes da tai tao thanh cong (dung de
                             dua vao tshark_dpi.dissect_batch())
        included_positions - vi tri (index trong `rows`) tuong ung voi tung
                             phan tu trong `frames`, dung de gan ket qua
                             dissect lai dung ve dong goc.
        skipped           - list {"position": i, "reason": ...} cho cac dong
                             KHONG tao duoc frame (van duoc bao cao, khong bi
                             lam mat).
    """
    seq_tracker = FlowSeqTracker()
    frames: List[bytes] = []
    included_positions: List[int] = []
    skipped: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        frame, skip_reason = build_frame_for_row(row, seq_tracker)
        if frame is None:
            skipped.append({"position": i, "reason": skip_reason})
        else:
            frames.append(frame)
            included_positions.append(i)

    return frames, included_positions, skipped
