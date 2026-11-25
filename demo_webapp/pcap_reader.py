"""Minimal streaming PCAP/PCAPNG -> packet table with the SAME columns as Dataset/raw_parquet.

Columns: file_id, packet_idx, timestamp, src_mac, dst_mac, eth_type, vlan_id, src_ip, dst_ip, ip_proto,
ip_ttl, src_port, dst_port, tcp_flags, tcp_window, packet_len, payload_len, app_payload.
Only Ethernet (link type 1) is supported. IP fragments after the first carry no ports (null).
Written for the web app; the training raw_parquet came from an earlier converter, so a parity check against a
matching file is run by `parity_check.py` and its result is shown in the app's About box.
"""
from __future__ import annotations

import ipaddress
import struct
from pathlib import Path

import pandas as pd

COLUMNS = ["file_id", "packet_idx", "timestamp", "src_mac", "dst_mac", "eth_type", "vlan_id", "src_ip", "dst_ip",
           "ip_proto", "ip_ttl", "src_port", "dst_port", "tcp_flags", "tcp_window", "packet_len", "payload_len", "app_payload"]


class UnsupportedCapture(Exception):
    pass


def _mac(b):
    return ":".join(f"{x:02x}" for x in b)


def _iter_classic(f, magic):
    le = magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1")
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    e = "<" if le else ">"
    hdr = f.read(20)
    if len(hdr) < 20:
        return
    linktype = struct.unpack(e + "I", hdr[16:20])[0] & 0xFFFF
    if linktype != 1:
        raise UnsupportedCapture(f"link type {linktype} (only Ethernet=1 is supported)")
    div = 1e9 if nano else 1e6
    while True:
        rh = f.read(16)
        if len(rh) < 16:
            return
        sec, frac, caplen, wirelen = struct.unpack(e + "IIII", rh)
        data = f.read(caplen)
        if len(data) < caplen:
            return
        yield sec + frac / div, wirelen, data


def _iter_pcapng(f):
    f.seek(0)
    e = "<"
    ifaces = []  # (linktype, units_per_second)
    while True:
        bh = f.read(8)
        if len(bh) < 8:
            return
        btype = struct.unpack("<I", bh[:4])[0]
        if btype == 0x0A0D0D0A:  # section header: detect endianness
            rest = f.read(4)
            bom = rest
            e = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            total = struct.unpack(e + "I", bh[4:8])[0]
            f.read(total - 12)
            ifaces = []
            continue
        btype, total = struct.unpack(e + "II", bh)
        body = f.read(total - 8)
        if len(body) < total - 8:
            return
        body = body[:-4]
        if btype == 1:  # interface description
            linktype = struct.unpack(e + "H", body[:2])[0]
            res, opts = 1_000_000, body[8:]
            i = 0
            while i + 4 <= len(opts):
                code, ln = struct.unpack(e + "HH", opts[i:i + 4])
                if code == 0:
                    break
                if code == 9 and ln >= 1:
                    v = opts[i + 4]
                    res = 2 ** (v & 0x7F) if v & 0x80 else 10 ** (v & 0x7F)
                i += 4 + ((ln + 3) & ~3)
            ifaces.append((linktype, res))
        elif btype == 6:  # enhanced packet
            iid, th, tl, caplen, wirelen = struct.unpack(e + "IIIII", body[:20])
            linktype, res = ifaces[iid] if iid < len(ifaces) else (1, 1_000_000)
            if linktype != 1:
                raise UnsupportedCapture(f"link type {linktype} (only Ethernet=1 is supported)")
            yield ((th << 32) | tl) / res, wirelen, body[20:20 + caplen]
        elif btype == 3:  # simple packet (no timestamp)
            wirelen = struct.unpack(e + "I", body[:4])[0]
            yield 0.0, wirelen, body[4:]


def _walk_ipv6(p, off):
    nxt, frag = p[off + 6], False
    off += 40
    for _ in range(6):
        if nxt in (0, 43, 60):
            nxt, ln = p[off], 8 + p[off + 1] * 8
            off += ln
        elif nxt == 44:
            nxt = p[off]
            frag = (struct.unpack(">H", p[off + 2:off + 4])[0] >> 3) != 0
            off += 8
        else:
            break
    return nxt, off, frag


def _parse(p):
    if len(p) < 14:
        return None
    dst, src = _mac(p[0:6]), _mac(p[6:12])
    et, off, vlan = struct.unpack(">H", p[12:14])[0], 14, None
    while et in (0x8100, 0x88A8, 0x9100) and len(p) >= off + 4:
        v = struct.unpack(">H", p[off:off + 2])[0] & 0x0FFF
        vlan = v if vlan is None else vlan
        et = struct.unpack(">H", p[off + 2:off + 4])[0]
        off += 4
    row = dict(src_mac=src, dst_mac=dst, eth_type=et, vlan_id=vlan, src_ip=None, dst_ip=None, ip_proto=None, ip_ttl=None,
               src_port=None, dst_port=None, tcp_flags=None, tcp_window=None, payload_len=None, app_payload=None)
    ip_off, frag, l3_payload_off = None, False, None
    try:
        if et == 0x0800 and len(p) >= off + 20:
            ihl = (p[off] & 0x0F) * 4
            fr = struct.unpack(">H", p[off + 6:off + 8])[0]
            frag = (fr & 0x1FFF) != 0
            row.update(src_ip=str(ipaddress.IPv4Address(p[off + 12:off + 16])), dst_ip=str(ipaddress.IPv4Address(p[off + 16:off + 20])),
                       ip_proto=p[off + 9], ip_ttl=p[off + 8])
            ip_off = off + ihl
            l3_payload_off = ip_off
        elif et == 0x86DD and len(p) >= off + 40:
            nxt, tr, frag = _walk_ipv6(p, off)
            # earlier converter kept the first next-header value (0 = hop-by-hop) unless TCP/UDP follows
            row.update(src_ip=str(ipaddress.IPv6Address(p[off + 8:off + 24])), dst_ip=str(ipaddress.IPv6Address(p[off + 24:off + 40])),
                       ip_proto=nxt if nxt in (6, 17) else p[off + 6], ip_ttl=p[off + 7])
            ip_off = tr
            l3_payload_off = off + 40
        if ip_off is not None and not frag:
            proto = row["ip_proto"]
            if proto == 6 and len(p) >= ip_off + 20:
                doff = (p[ip_off + 12] >> 4) * 4
                row.update(src_port=struct.unpack(">H", p[ip_off:ip_off + 2])[0], dst_port=struct.unpack(">H", p[ip_off + 2:ip_off + 4])[0],
                           tcp_flags=p[ip_off + 13], tcp_window=struct.unpack(">H", p[ip_off + 14:ip_off + 16])[0])
                pay = p[ip_off + doff:]
                row.update(payload_len=len(pay), app_payload=bytes(pay))
            elif proto == 17 and len(p) >= ip_off + 8:
                row.update(src_port=struct.unpack(">H", p[ip_off:ip_off + 2])[0], dst_port=struct.unpack(">H", p[ip_off + 2:ip_off + 4])[0])
                pay = p[ip_off + 8:]
                row.update(payload_len=len(pay), app_payload=bytes(pay))
        if row["app_payload"] is None:  # parity with the earlier converter: ARP/LLDP/L2 = bytes after Ethernet(+VLAN) header,
            start = l3_payload_off if l3_payload_off is not None else off  # IPv6 non-TCP/UDP = after the fixed IPv6 header
            row.update(payload_len=len(p) - start, app_payload=bytes(p[start:]))
    except (struct.error, ValueError, IndexError):
        pass
    return row


def read_pcap(path, max_packets=None):
    """Return (DataFrame in raw_parquet column layout, info dict)."""
    path = Path(path)
    rows, truncated = [], False
    with path.open("rb") as f:
        magic = f.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            it = _iter_pcapng(f)
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
            it = _iter_classic(f, magic)
        else:
            raise UnsupportedCapture("not a pcap/pcapng file (bad magic)")
        for idx, (ts, wirelen, data) in enumerate(it):
            if max_packets and idx >= max_packets:
                truncated = True
                break
            r = _parse(data)
            if r is None:
                continue
            r.update(file_id=path.stem, packet_idx=idx, timestamp=ts, packet_len=wirelen)
            rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df, {"packets_read": len(df), "truncated": truncated, "max_packets": max_packets}
