#!/usr/bin/env python3
r"""
tshark_dpi.py

Deep Packet Inspection engine dung chung, dua tren tshark (tshark.exe da cai san
tren may, ke thua toan bo dissector cua Wireshark - bao gom Modbus/TCP, S7comm,
DNP3, BACnet, EtherNet/IP, MMS, ...) thay vi tu viet lai parser cho tung protocol
(xem WP02 muc 6.4: "TShark cung cap co che liet ke protocol/field va xuat du lieu
de xay adapter nay").

Khong tu suy dien gia tri "function code" bang cach doan ten cot cho tung
protocol rieng le: dung tshark de dissect that su, roi doc lai cay field da
duoc dissector cua chinh Wireshark tao ra. Ket qua vi vay dang tin cay bang
dung Wireshark GUI de mo file, khong phai mot bo quy tac tu viet co the sai/
thieu.

Cac ham chinh:
    find_tshark()                  - tim duong dan tshark.exe tren may
    get_tshark_version()           - lay version string, ghi vao provenance
    build_multi_frame_pcap()       - dong goi nhieu frame (bytes) thanh 1 file
                                      pcap trong bo nho (khong ghi file tam),
                                      giu dung THU TU goc de bao toan ngu canh
                                      conversation/transaction khi dissect.
    run_tshark_json()              - chay tshark qua stdin, tra ve JSON (moi
                                      frame 1 object), khong dung file tam.
    parse_tshark_json()            - parse JSON tho cua tshark thanh cau truc
                                      de dung: protocol_stack, primary_protocol,
                                      function_code, all_fields.

QUAN TRONG - gioi han da biet (ghi ro theo tinh than "khong giau gia dinh" cua
ke hoach luan van, muc 6.6):
    - Neu frame duoc dung lai (build_multi_frame_pcap) tu payload da tach lop
      (khong phai raw_record_bytes nguyen ban), thi Ethernet/IP/TCP header la
      TAI TAO (synthetic), khong phai byte goc tren day. Checksum L3/L4 KHONG
      duoc tinh (khong can thiet de dissector hoat dong, tshark khong drop goi
      vi checksum sai - chi hien expert-info warning).
    - Modbus request/response transaction-matching (packet-mbtcp.c dung
      conversation + transaction_id/unit_id, KHONG dung TCP seq) van hoat dong
      dung ngay ca voi header tai tao, MIEN LA: (1) giu dung 5-tuple that,
      (2) tat ca packet cung 1 file duoc dissect trong CUNG 1 lan goi tshark,
      dung thu tu goc. Ham build_multi_frame_pcap() dam bao dieu (2).
    - TCP segment reassembly (PDU Modbus trai dai qua 2 segment) can TCP
      sequence number tang dung theo huong luong - xem ham
      build_synthetic_tcp_frame() trong dpi_enrich.py, noi seq duoc theo doi
      rieng cho tung (src_ip,src_port,dst_ip,dst_port).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Dinh vi tshark.exe
# --------------------------------------------------------------------------

_COMMON_WINDOWS_PATHS = [
    r"C:\Program Files\Wireshark\tshark.exe",
    r"C:\Program Files (x86)\Wireshark\tshark.exe",
]


class TsharkNotFoundError(RuntimeError):
    pass


def find_tshark(explicit_path: Optional[str] = None) -> str:
    """Tim duong dan tshark.exe. Uu tien: tham so truyen vao > bien moi truong
    TSHARK_PATH > PATH he thong > cac duong dan cai dat Windows pho bien."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("TSHARK_PATH")
    if env_path:
        candidates.append(env_path)
    which = shutil.which("tshark")
    if which:
        candidates.append(which)
    candidates.extend(_COMMON_WINDOWS_PATHS)

    for c in candidates:
        if c and Path(c).is_file():
            return str(c)

    raise TsharkNotFoundError(
        "Khong tim thay tshark.exe. Cai Wireshark (bao gom TShark) hoac dat "
        "bien moi truong TSHARK_PATH tro toi tshark.exe."
    )


def get_tshark_version(tshark_path: str) -> str:
    proc = subprocess.run([tshark_path, "-v"], capture_output=True, text=True, check=True)
    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    return first_line.strip()


# --------------------------------------------------------------------------
# Dong goi nhieu frame thanh 1 pcap trong bo nho (khong ghi file tam)
# --------------------------------------------------------------------------

DLT_ETHERNET = 1


def build_multi_frame_pcap(frames: List[bytes], linktype: int = DLT_ETHERNET) -> bytes:
    """Ghep danh sach frame bytes (moi phan tu = 1 frame hoan chinh, vi du
    Ethernet+IP+TCP+payload) thanh 1 file pcap classic hoan chinh trong bo nho,
    GIU DUNG THU TU trong list - thu tu nay quyet dinh thu tu frame.number khi
    tshark dissect, nen phai khop voi thu tu packet_idx goc trong dataset."""
    global_hdr = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,  # magic: native-endian, do phan giai microsecond
        2, 4,        # version major.minor
        0, 0,        # thiszone, sigfigs (khong dung)
        65535,       # snaplen
        linktype,
    )
    out = bytearray(global_hdr)
    for i, frame in enumerate(frames):
        # ts_sec/ts_usec danh gia tri tang dan gia (1 giay/frame) chi de co
        # thu tu don dieu hop le trong file pcap; timestamp THAT da co san o
        # cot rieng trong dataset (khong bi mat, chi khong dung lam ts cua
        # frame tai tao nay).
        rec_hdr = struct.pack("<IIII", i, 0, len(frame), len(frame))
        out += rec_hdr
        out += frame
    return bytes(out)


# --------------------------------------------------------------------------
# Chay tshark qua stdin (khong ghi file tam ra dia), doc JSON qua stdout
# --------------------------------------------------------------------------

def run_tshark_json(tshark_path: str, pcap_bytes: bytes, timeout: int = 300) -> List[Dict[str, Any]]:
    """Chay `tshark -r - -T json` voi pcap_bytes dua qua stdin. Tra ve list
    cac object JSON tho cua tshark (1 phan tu / frame), theo dung thu tu
    frame.number tang dan."""
    proc = subprocess.run(
        [tshark_path, "-r", "-", "-T", "json", "-n"],
        input=pcap_bytes,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"tshark exit code {proc.returncode}: {stderr[:2000]}")
    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    if not stdout_text.strip():
        return []
    return json.loads(stdout_text)


# --------------------------------------------------------------------------
# Parse JSON tho cua tshark thanh cau truc gon: protocol_stack, primary
# protocol, function-code-like field, toan bo field da flatten.
# --------------------------------------------------------------------------

# Cac lop "wrapper" khong duoc coi la "protocol ung dung chinh" khi chon
# primary_protocol tu frame.protocols (vd "eth:ethertype:ip:tcp:mbtcp:modbus"
# -> primary_protocol = "modbus", khong phai "tcp").
_WRAPPER_LAYERS = {
    "frame", "eth", "ethertype", "vlan", "ip", "ipv6", "tcp", "udp",
    "sll", "null", "raw", "cotp", "tpkt",
}

# Ten field "function/service/operation code" da biet cho tung protocol -
# tra cuu truc tiep, uu tien hon heuristic doan ten field. Bo sung dan khi
# gap protocol moi trong parser_capability_matrix.
KNOWN_FUNCTION_CODE_FIELDS: Dict[str, List[str]] = {
    "modbus": ["modbus.func_code"],
    "s7comm": ["s7comm.header.rosctr", "s7comm.param.func"],
    "s7comm-plus": ["s7comm-plus.data.opcode"],
    "dnp3": ["dnp3.al.func"],
    "cip": ["cip.cm.sc", "cip.service"],
    "enip": ["enip.command"],
    "bacapp": ["bacapp.confirmed_service", "bacapp.unconfirmed_service"],
    "iec104asdu": ["iec104.asdu.typeid"],
    "goose": ["goose.gocbRef"],
}

# Field danh dau "da khop transaction request<->response" - chi dien khi da
# xac nhan tan mat trong source dissector (khong doan). Modbus: xem
# packet-mbtcp.c dissect_modbus(), hf_modbus_request_frame = "modbus.request_frame".
KNOWN_TRANSACTION_LINK_FIELDS: Dict[str, str] = {
    "modbus": "modbus.request_frame",
}

# Heuristic du phong khi protocol chua co trong KNOWN_FUNCTION_CODE_FIELDS o tren -
# tim field co ten goi y la "function/service/operation/command code" trong
# chinh layer cua primary_protocol. Danh dau ro nguon = "heuristic_field_name_match"
# (do tin cay thap hon tra cuu KNOWN_FUNCTION_CODE_FIELDS o tren).
import re as _re  # noqa: E402
_FUNCTION_CODE_NAME_PATTERN = _re.compile(
    r"(func[_-]?code|function[_-]?code|service[_-]?code|op[_-]?code|rosctr|\.func$|\.service$|\.cmd$|\.command$)",
    _re.IGNORECASE,
)


def _primary_protocol_from_stack(protocol_stack: List[str]) -> Optional[str]:
    for proto in reversed(protocol_stack):
        if proto not in _WRAPPER_LAYERS:
            return proto
    return None


def parse_tshark_frame(raw_frame_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyen 1 object JSON tho cua tshark (`_source.layers`) thanh cau truc gon.

    Tra ve dict:
        frame_number, protocol_stack (list[str]), primary_protocol (str|None),
        layers (dict proto -> {field_name: value_str}),
        function_code_field, function_code_value, function_code_source,
        transaction_matched (bool)
    """
    layers = raw_frame_obj.get("_source", {}).get("layers", {})
    frame_layer = layers.get("frame", {})
    protocols_str = frame_layer.get("frame.protocols", "")
    protocol_stack = protocols_str.split(":") if protocols_str else []
    frame_number = frame_layer.get("frame.number")

    primary_protocol = _primary_protocol_from_stack(protocol_stack)

    # tshark -T json co the long field lap (vd "eth.dst_tree") thanh dict long
    # nhau; o day chi lay cac field co gia tri chuoi truc tiep (leaf), du cho
    # muc dich function-code + value extraction. Field group long duoc giu
    # nguyen trong "layers_raw" de khong mat thong tin cho nguoi muon dao sau.
    flat_layers: Dict[str, Dict[str, str]] = {}
    for proto, proto_fields in layers.items():
        flat_layers[proto] = _flatten_json_layer(proto_fields)

    function_code_field = None
    function_code_value = None
    function_code_source = None

    if primary_protocol:
        proto_fields = flat_layers.get(primary_protocol, {})
        for candidate in KNOWN_FUNCTION_CODE_FIELDS.get(primary_protocol, []):
            if candidate in proto_fields:
                function_code_field = candidate
                function_code_value = proto_fields[candidate]
                function_code_source = "known_field_lookup"
                break
        if function_code_field is None:
            for name, value in proto_fields.items():
                if _FUNCTION_CODE_NAME_PATTERN.search(name):
                    function_code_field = name
                    function_code_value = value
                    function_code_source = "heuristic_field_name_match"
                    break

    transaction_matched = False
    if primary_protocol in KNOWN_TRANSACTION_LINK_FIELDS:
        link_field = KNOWN_TRANSACTION_LINK_FIELDS[primary_protocol]
        transaction_matched = link_field in flat_layers.get(primary_protocol, {})

    return {
        "frame_number": int(frame_number) if frame_number is not None else None,
        "protocol_stack": protocol_stack,
        "primary_protocol": primary_protocol,
        "layers": flat_layers,
        "function_code_field": function_code_field,
        "function_code_value": function_code_value,
        "function_code_source": function_code_source,
        "transaction_matched": transaction_matched,
    }


def _flatten_json_layer(node: Any, prefix: str = "") -> Dict[str, str]:
    """tshark -T json bieu dien field long (subtree) bang key phu "<field>_tree".
    Ham nay duyet de­quy, chi giu lai cap {ten_field_wireshark: gia_tri_chuoi}
    cho moi field co gia tri chuoi/so truc tiep (bo qua cac key ho tro noi bo
    nhu "_ws.expert")."""
    result: Dict[str, str] = {}
    if not isinstance(node, dict):
        return result
    for key, value in node.items():
        if key.startswith("_ws.") or key.endswith("_tree"):
            if isinstance(value, dict):
                result.update(_flatten_json_layer(value))
            continue
        if isinstance(value, dict):
            result.update(_flatten_json_layer(value))
        elif isinstance(value, list):
            # Field lap lai nhieu lan (vd nhieu Coil Status) - luu dang chuoi
            # noi bang ";" de giu du lieu, khong lam no rong schema cot.
            result[key] = ";".join(str(v) for v in value)
        else:
            result[key] = value
    return result


def dissect_batch(tshark_path: str, frames: List[bytes], linktype: int = DLT_ETHERNET) -> List[Dict[str, Any]]:
    """Tien ich goi gon: dong goi frames -> chay tshark -> parse JSON, tra ve
    list cung do dai/thu tu voi `frames` dau vao (1-1 theo vi tri)."""
    if not frames:
        return []
    pcap_bytes = build_multi_frame_pcap(frames, linktype=linktype)
    raw_results = run_tshark_json(tshark_path, pcap_bytes)
    parsed = [parse_tshark_frame(r) for r in raw_results]
    if len(parsed) != len(frames):
        raise RuntimeError(
            f"So frame tshark tra ve ({len(parsed)}) khac so frame gui vao ({len(frames)}) - "
            "co the do 1 frame bi tshark tu dong bo qua (malformed)."
        )
    return parsed
