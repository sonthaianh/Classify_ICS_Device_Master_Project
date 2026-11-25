#!/usr/bin/env python3
"""Build v6 (expanded behaviour scope) packet-exact windows and evidence; NEVER assign labels.

Copy of 23_build_review_dataset.py (v5 untouched). Only differences: scope = devices.csv
include_in_behavior_dataset=True (>=1 flow, not spoofed), device/session pairs taken from
Dataset/processed/device_flows; outputs are named v6 and never overwrite v5.

Scope = all device/session pairs of v4, merging __l2 sessions into their raw
capture. Adds packet-only windows for these devices, not previously excluded
devices. Streams all packets (no file size skip). Resumable per-capture cache.
Application evidence uses bounded, explicitly sampled synthetic IPv4 frames;
no claim of complete dissection, true device identity or operational function.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from feature_contract import TRAINING_FEATURES, EXPERIMENTAL_FEATURES

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
import frame_builder
import tshark_dpi

W = 45
VERSION = "v6_behavior_scope_20261104"
KEY = ["device_id", "session_id", "window_start"]
L2 = {0x88B8: "goose", 0x88BA: "sv", 0x8892: "profinet-rt", 0x88A4: "ethercat"}
FIELDS = ["frame.number", "frame.protocols", "_ws.malformed",
          "modbus.func_code", "s7comm.header.rosctr", "s7comm.param.func",
          "s7comm-plus.data.opcode", "dnp3.al.func", "enip.command", "cip.service",
          "bacapp.confirmed_service", "iec104.asdu.typeid",
          "s7comm.szl.001c.0001.name", "s7comm.szl.001c.0007.cputypname",
          "s7comm.szl.001c.0005.serialn", "enip.lir.product_name", "enip.lir.serial"]
COUNTS = ["packet_count", "frames_sent", "frames_received", "arp_frame_count",
          "icmp_frame_count", "nonip_frame_count", "unparsed_ip_frame_count", "bc_count"]


def json_list(values):
    return json.dumps(sorted({str(x) for x in values if pd.notna(x) and str(x) != ""}), ensure_ascii=False)


def port_catalog():
    text = (ROOT / "utils/DataConfig/ot_ports.yaml").read_text(encoding="utf-8")
    ports = set()
    for line in text.splitlines():
        if line.startswith("port_to_protocol"):
            break
        m = re.match(r"^\s*-\s*(\d+)\b", line)
        if m:
            ports.add(int(m.group(1)))
    return ports


def multicast(series):
    return series.fillna("").str.slice(1, 2).str.lower().isin(list("13579bdf"))


def packet_keys(d):
    """Same bidirectional conversation definition as v4, plus VLAN isolation.

    No idle timeout: reused 5-tuples remain one conversation within capture.
    L2 publish keys preserve source/destination order.
    """
    a = d.src_ip.fillna("") + ":" + d.src_port.fillna(-1).astype("int64").astype(str)
    b = d.dst_ip.fillna("") + ":" + d.dst_port.fillna(-1).astype("int64").astype(str)
    ip = d.ip_proto.isin([6, 17]) & d.src_port.notna() & d.dst_port.notna() & d.src_ip.notna() & d.dst_ip.notna()
    layer2 = d.eth_type.isin(L2)
    ka = a.where(a <= b, b)
    kb = b.where(a <= b, a)
    k = ka + "|" + kb + "|" + d.ip_proto.fillna(-1).astype("int64").astype(str)
    pub = multicast(d.dst_mac) | multicast(d.src_mac)
    la = d.src_mac.where((d.src_mac <= d.dst_mac) | pub, d.dst_mac)
    lb = d.dst_mac.where((d.src_mac <= d.dst_mac) | pub, d.src_mac)
    lk = la + "|" + lb + "|" + d.eth_type.fillna(0).astype("int64").astype(str)
    vlan = d.vlan_id.fillna(-1).astype("int64").astype(str)
    return (k.where(~layer2, lk) + "|vlan=" + vlan).where(ip | layer2), ip, layer2


def deduplicate(d, seen):
    """Match existing directed TCP/UDP duplicate policy, across batch boundaries.

    Two independent 64-bit hashes, not a single floating point hash. Raw frame
    counters intentionally include capture copies, matching v4 U1 semantics.
    """
    cols = ["src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "timestamp"]
    fp = d[cols].copy()
    fp["payload_md5"] = d.app_payload.map(lambda x: hashlib.md5(bytes(x or b"")).hexdigest())
    h1 = pd.util.hash_pandas_object(fp, index=False, hash_key="0123456789123456").to_numpy()
    h2 = pd.util.hash_pandas_object(fp, index=False, hash_key="fedcba9876543210").to_numpy()
    keep = []
    for a, b in zip(h1, h2):
        key = (int(a), int(b))
        keep.append(key not in seen)
        seen.add(key)
    return np.asarray(keep, dtype=bool)


def endpoint_view(d, mac_to_dev):
    views = []
    for outgoing in (True, False):
        own, peer = ("src", "dst") if outgoing else ("dst", "src")
        mask = d[f"{own}_mac"].isin(mac_to_dev)
        if not outgoing:
            mask &= d.src_mac != d.dst_mac  # count a captured self-frame once
        # Do not carry/copy large binary payloads through every endpoint group.
        v = d.loc[mask, [c for c in d.columns if c != "app_payload"]].copy()
        v["device_id"] = v[f"{own}_mac"].map(mac_to_dev)
        v["own_ip"] = v[f"{own}_ip"].fillna("")
        v["peer_ip"] = v[f"{peer}_ip"].fillna("")
        v["peer_mac"] = v[f"{peer}_mac"].fillna("")
        v["local_port"] = v[f"{own}_port"]
        v["remote_port"] = v[f"{peer}_port"]
        v["outgoing"] = int(outgoing)
        views.append(v)
    return pd.concat(views, ignore_index=True)


def empty_evidence():
    return {"ips": set(), "peer_ips": set(), "peer_macs": set(), "services": set(),
            "protocols": set(), "refs": set(), "messages": set(), "models": set(),
            "stations": set(), "identity_refs": set(), "transports": set()}


def decode_samples(samples, tshark, fields, evidence):
    if not samples:
        return {"status": "no_supported_payload_sample", "frames": 0}
    frames, owners = [], []
    tracker = frame_builder.FlowSeqTracker()
    for row in samples:
        frame, reason = frame_builder.build_frame_for_row(row, tracker)
        if frame is not None:
            frames.append(frame)
            owners.append(row)
    if not frames:
        return {"status": "no_reconstructable_ipv4_payload", "frames": 0}
    cmd = [tshark, "-n", "-r", "-", "-T", "fields", "-E", "separator=/t",
           "-E", "quote=d", "-E", "occurrence=a"]
    for field in fields:
        cmd.extend(["-e", field])
    proc = subprocess.run(cmd, input=tshark_dpi.build_multi_frame_pcap(frames), capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-1500:])
    count = 0
    for values in csv.reader(io.StringIO(proc.stdout.decode("utf-8", errors="replace")), delimiter="\t"):
        decoded = dict(zip(fields, values))
        number = int(decoded.get("frame.number") or 0)
        if not 1 <= number <= len(owners):
            raise ValueError("tshark frame/sample alignment failed")
        row = owners[number - 1]
        ref = f"raw_packet_idx={row['packet_idx']}"
        # Malformed synthetic dissection is not evidence of a valid protocol.
        if decoded.get("_ws.malformed"):
            continue
        protocols = set(decoded.get("frame.protocols", "").split(":")) - {"", "frame", "eth", "ethertype", "ip", "ipv6", "tcp", "udp", "data", "tpkt", "cotp"}
        for dev in row["_owners"]:
            e = evidence[dev]
            e["protocols"].update(protocols)
            if protocols:
                e["refs"].add(ref)
            for f, value in decoded.items():
                if not value or f in {"frame.number", "frame.protocols", "_ws.malformed"}:
                    continue
                if "szl.001c" in f or f.startswith("enip.lir"):
                    # Identity fields describe the sender of this message only;
                    # never copy a PLC identity to both sides of a conversation.
                    if dev not in row["_sender_owners"]:
                        continue
                    e["identity_refs"].add(ref + ";field=" + f)
                    if f.endswith("0001.name"):
                        e["stations"].add(value)
                    elif f.endswith("cputypname") or f.endswith("product_name"):
                        e["models"].add(value)
                else:
                    e["messages"].add(f + "=" + value)
        count += 1
    return {"status": "sampled_synthetic_ipv4_dissection", "frames": len(frames), "usable_frames": count,
            "limitation": "bounded payload samples; synthetic headers; missing TCP reassembly context; absence is not negative evidence"}


def build_capture(path, mac_to_dev, registry, ot_ports, tshark, fields, batch_size=250000, sample_cap=2000):
    """Return all windows and one evidence row per scoped device/capture."""
    evidence = defaultdict(empty_evidence)
    stats_parts, flow_parts = [], []
    seen, sample_counts, samples = set(), defaultdict(int), []
    arp_targets = defaultdict(set)
    window_ips = defaultdict(set)
    local_ports, service_ports = defaultdict(set), defaultdict(set)
    first_seen, syn_seen = {}, {}
    pf = pq.ParquetFile(path)
    required = ["timestamp", "src_mac", "dst_mac", "eth_type", "src_ip", "dst_ip", "ip_proto",
                "src_port", "dst_port", "tcp_flags", "packet_len", "app_payload", "packet_idx", "vlan_id"]
    missing = set(required) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Unsupported raw schema: {sorted(missing)}")
    columns = required + (["ip_ttl"] if "ip_ttl" in pf.schema_arrow.names else [])
    wanted = pa.array(list(mac_to_dev))
    invalid_timestamps = duplicates = selected_packets = 0
    for batch_index, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), 1):
        if batch_index % 10 == 0:
            print(f"  {path.stem}: scanning batch {batch_index}, selected={selected_packets:,}", flush=True)
        table = pa.Table.from_batches([batch])
        relevant = pc.or_(pc.fill_null(pc.is_in(table["src_mac"], value_set=wanted), False),
                          pc.fill_null(pc.is_in(table["dst_mac"], value_set=wanted), False))
        d = table.filter(relevant).to_pandas()
        if d.empty:
            continue
        invalid_timestamps += int((~np.isfinite(d.timestamp)).sum())
        d = d[np.isfinite(d.timestamp)].copy()
        if d.empty:
            continue
        d["window_start"] = np.floor(d.timestamp / W) * W
        d["flow_key"], ip, layer2 = packet_keys(d)
        d["flow_eligible"] = ip | layer2
        d["transport"] = np.select([d.ip_proto.eq(6), d.ip_proto.eq(17), layer2], ["tcp", "udp", "l2"], default="other")
        d["ot_service"] = d.src_port.isin(ot_ports) | d.dst_port.isin(ot_ports)
        d["is_syn"] = (d.tcp_flags.fillna(0).astype("int64") & 0x12) == 0x02
        d["is_synack"] = (d.tcp_flags.fillna(0).astype("int64") & 0x12) == 0x12
        d["bc"] = multicast(d.dst_mac).astype(int)
        d["arp"] = d.eth_type.eq(0x0806).astype(int)
        d["icmp"] = d.ip_proto.eq(1).astype(int)
        has_ip = d.src_ip.fillna("").ne("") | d.dst_ip.fillna("").ne("") | d.eth_type.isin([0x0800, 0x86DD])
        d["nonip"] = (~has_ip & d.eth_type.notna() & ~d.eth_type.isin([0x8100, 0x88A8])).astype(int)
        d["unparsed_ip"] = (has_ip & d.ip_proto.isna()).astype(int)
        d["keep_flow"] = True
        if ip.any():
            keep = deduplicate(d.loc[ip], seen)
            d.loc[ip, "keep_flow"] = keep
            duplicates += int((~keep).sum())
        selected_packets += len(d)
        arp_rows = d[d.arp.eq(1) & d.src_mac.isin(mac_to_dev)]
        for row in arp_rows[["src_mac", "window_start", "app_payload"]].itertuples(index=False):
            payload = bytes(row.app_payload or b"")
            if len(payload) >= 28 and payload[0:2] == b"\x00\x01" and payload[2:4] == b"\x08\x00" and payload[4:6] == b"\x06\x04" and payload[6:8] == b"\x00\x01":
                arp_targets[(mac_to_dev[row.src_mac], row.window_start)].add(".".join(str(b) for b in payload[24:28]))
        v = endpoint_view(d, mac_to_dev)
        v["packet_count"] = 1
        v["frames_sent"] = v.outgoing
        v["frames_received"] = 1 - v.outgoing
        for target, source in [("arp_frame_count", "arp"), ("icmp_frame_count", "icmp"),
                               ("nonip_frame_count", "nonip"), ("unparsed_ip_frame_count", "unparsed_ip"), ("bc_count", "bc")]:
            v[target] = v[source] * v.outgoing
        agg = {c: (c, "sum") for c in COUNTS}
        agg.update(first_observed=("timestamp", "min"), last_observed=("timestamp", "max"))
        stats_parts.append(v.groupby(["device_id", "window_start"], sort=False).agg(**agg).reset_index())
        addresses = v[["device_id", "window_start", "own_ip"]].drop_duplicates()
        for (dev, ws), q in addresses.groupby(["device_id", "window_start"], sort=False):
            window_ips[(dev, ws)].update(x for x in q.own_ip.unique() if x)
        meta = v[["device_id", "own_ip", "peer_ip", "peer_mac", "transport", "eth_type", "local_port", "remote_port"]].drop_duplicates()
        for dev, q in meta.groupby("device_id", sort=False):
            e = evidence[dev]
            e["ips"].update(q.own_ip.unique())
            e["peer_ips"].update(q.peer_ip.unique())
            e["peer_macs"].update(q.peer_mac.unique())
            e["transports"].update(q.transport.unique())
            for typ in q.eth_type.dropna().unique():
                if int(typ) in L2:
                    e["protocols"].add(L2[int(typ)])
            l2refs = v[v.device_id.eq(dev) & v.eth_type.isin(L2)].packet_idx.head(3)
            e["refs"].update(f"raw_packet_idx={int(i)};ethertype" for i in l2refs)
            for proto, lp, rp in q[["transport", "local_port", "remote_port"]].drop_duplicates().itertuples(index=False, name=None):
                if proto in ("tcp", "udp"):
                    e["services"].add(f"{proto}:local={lp:g},peer={rp:g}")
        f = v[v.flow_eligible & v.keep_flow].copy()
        if not f.empty:
            f["sent_pk"] = f.outgoing
            f["sent_by"] = f.packet_len * f.outgoing
            f["ot_pk"] = f.ot_service.astype(int)
            f["udp_pk"] = f.transport.eq("udp").astype(int)
            fa = f.groupby(["device_id", "window_start", "flow_key"], sort=False).agg(
                pk=("packet_len", "size"), by=("packet_len", "sum"), sent_pk=("sent_pk", "sum"),
                sent_by=("sent_by", "sum"), ot_pk=("ot_pk", "sum"), udp_pk=("udp_pk", "sum"),
                first=("timestamp", "min"), last=("timestamp", "max"),
                transport=("transport", "first"), local_port=("local_port", "first"), remote_port=("remote_port", "first")).reset_index()
            flow_parts.append(fa)
            distinct_ports = f[["device_id", "window_start", "local_port"]].drop_duplicates()
            for (dev, ws), q in distinct_ports.groupby(["device_id", "window_start"], sort=False):
                local_ports[(dev, ws)].update(q.local_port.dropna().astype(int).unique())
            for key, ts in d[d.flow_eligible & d.keep_flow].groupby("flow_key").timestamp.min().items():
                first_seen[key] = min(first_seen.get(key, ts), ts)
            handshake = d[(d.is_syn | d.is_synack) & d.flow_eligible & d.keep_flow]
            for row in handshake.sort_values("timestamp").itertuples():
                client = row.src_mac if row.is_syn else row.dst_mac
                prev = syn_seen.get(row.flow_key)
                if prev is None or row.timestamp < prev[0]:
                    syn_seen[row.flow_key] = (row.timestamp, client)
        if len(samples) >= sample_cap:
            continue
        candidates = d[d.flow_eligible & d.keep_flow & d.ip_proto.isin([6, 17]) & d.app_payload.map(lambda x: len(x or b"") > 0)]
        candidates = candidates[candidates.groupby(["flow_key", "src_mac"], sort=False).cumcount() < 2]
        for row in candidates.to_dict("records"):
            direction = (row["flow_key"], row["src_mac"])
            if sample_counts[direction] >= 2 or len(samples) >= sample_cap:
                continue
            sample_counts[direction] += 1
            row["_owners"] = list({mac_to_dev[m] for m in (row["src_mac"], row["dst_mac"]) if m in mac_to_dev})
            row["_sender_owners"] = [mac_to_dev[row["src_mac"]]] if row["src_mac"] in mac_to_dev else []
            samples.append(row)
    if not stats_parts:
        raise ValueError("No scoped packets in raw capture")
    stats = pd.concat(stats_parts, ignore_index=True).groupby(["device_id", "window_start"], sort=False).agg(
        **{c: (c, "sum") for c in COUNTS}, first_observed=("first_observed", "min"), last_observed=("last_observed", "max")).reset_index()
    stats["observed_seconds"] = stats.last_observed - stats.first_observed
    stats["frame_rate"] = stats.frames_sent / stats.observed_seconds.clip(lower=1)
    stats["bcast_frame_ratio"] = stats.bc_count / stats.frames_sent.replace(0, np.nan)
    stats["arp_target_unique"] = [len(arp_targets[(d, w)]) for d, w in zip(stats.device_id, stats.window_start)]
    stats["ip_count"] = [len(window_ips[(d, w)]) for d, w in zip(stats.device_id, stats.window_start)]
    if flow_parts:
        f = pd.concat(flow_parts, ignore_index=True)
        f = f.groupby(["device_id", "window_start", "flow_key"], sort=False).agg(
            **{c: (c, "sum") for c in ["pk", "by", "sent_pk", "sent_by", "ot_pk", "udp_pk"]},
            first=("first", "min"), last=("last", "max"), transport=("transport", "first"),
            local_port=("local_port", "first"), remote_port=("remote_port", "first")).reset_index()
        f["dur"] = f["last"] - f["first"]
        f["small"] = (f.pk <= 2).astype(int)
        f["oneway"] = ((f.sent_pk == 0) | (f.sent_pk == f.pk)).astype(int)
        f["udp"] = f.transport.eq("udp").astype(int)
        f["tcp"] = f.transport.eq("tcp").astype(int)
        f["carry"] = [int(first_seen[k] < ws) for k, ws in zip(f.flow_key, f.window_start)]
        role, syn = [], []
        dev_to_mac = {v: k for k, v in mac_to_dev.items()}
        for row in f.itertuples():
            h = syn_seen.get(row.flow_key)
            known_handshake = h is not None and h[0] < row.window_start + W
            # L2 is publish/subscribe or unknown, never infer server by port.
            if row.transport == "l2":
                role.append(np.nan); syn.append(0)
            elif known_handshake:
                role.append(float(dev_to_mac[row.device_id] != h[1])); syn.append(1)
            elif pd.notna(row.local_port) and pd.notna(row.remote_port) and row.local_port != row.remote_port and row.local_port > 0 and row.remote_port > 0:
                lp, rp = row.local_port, row.remote_port
                server = lp in ot_ports if ((lp in ot_ports) != (rp in ot_ports)) else lp < rp
                role.append(float(server)); syn.append(0)
            else:
                role.append(np.nan); syn.append(0)
            if pd.notna(role[-1]):
                service_ports[(row.device_id, row.window_start)].add(int(row.local_port if role[-1] else row.remote_port))
        f["srv"] = role
        f["syn"] = syn
        f["known"] = f.srv.notna().astype(int)
        f["tcp_nohand"] = ((f.tcp == 1) & (f.syn == 0)).astype(int)
        f["ot_flow"] = (f.ot_pk > 0).astype(int)
        a = f.groupby(["device_id", "window_start"], sort=False).agg(
            flows_active=("flow_key", "size"), pkt_alloc=("pk", "sum"), bytes_observed=("by", "sum"),
            sent_bytes=("sent_by", "sum"), udp=("udp", "sum"), small=("small", "sum"), oneway=("oneway", "sum"),
            flow_dur_median=("dur", "median"), first_flow_packet=("first", "min"), last_flow_packet=("last", "max"),
            server_role_ratio=("srv", "mean"), known=("known", "sum"), syn=("syn", "sum"),
            tcp=("tcp", "sum"), tcp_nohand=("tcp_nohand", "sum"), carry=("carry", "sum"),
            ot_flow=("ot_flow", "sum"), ot_pk=("ot_pk", "sum")).reset_index()
        a["pkts_per_flow"] = a.pkt_alloc / a.flows_active
        a["bytes_per_packet"] = a.bytes_observed / a.pkt_alloc
        a["exposure_seconds"] = a.last_flow_packet - a.first_flow_packet
        a["packet_rate"] = a.pkt_alloc / a.exposure_seconds.clip(lower=1)
        for target, col in [("udp_ratio", "udp"), ("small_flow_ratio", "small"), ("one_way_flow_ratio", "oneway"),
                            ("role_known_ratio", "known"), ("role_syn_ratio", "syn"), ("carryover_flow_ratio", "carry"), ("ot_flow_ratio", "ot_flow")]:
            a[target] = a[col] / a.flows_active
        a["sent_byte_ratio"] = a.sent_bytes / a.bytes_observed.replace(0, np.nan)
        a["started_before_capture_ratio"] = a.tcp_nohand / a.tcp.replace(0, np.nan)
        a["ot_pkt_share"] = a.ot_pk / a.pkt_alloc
        a["unique_local_port_count"] = [len(local_ports[(d, w)]) for d, w in zip(a.device_id, a.window_start)]
        a["svc_port_unique"] = [len(service_ports[(d, w)]) for d, w in zip(a.device_id, a.window_start)]
        used = ["device_id", "window_start", "flows_active", "pkt_alloc", "bytes_observed", "exposure_seconds"] + [c for c in TRAINING_FEATURES + EXPERIMENTAL_FEATURES if c in a]
        stats = stats.merge(a[list(dict.fromkeys(used))], on=["device_id", "window_start"], how="left", validate="one_to_one")
    for c in TRAINING_FEATURES + EXPERIMENTAL_FEATURES:
        if c not in stats:
            stats[c] = np.nan
    for c in ["flows_active", "pkt_alloc", "bytes_observed", "exposure_seconds"]:
        if c not in stats:
            stats[c] = 0.0
        stats[c] = stats[c].fillna(0)
    stats["has_flow_packets"] = stats.flows_active > 0
    stats["window_end"] = stats.window_start + W
    stats["session_id"] = path.stem
    stats["feature_version"] = VERSION
    stats["packet_allocation_method"] = "exact_packet_timestamp"
    decode = decode_samples(samples, tshark, fields, evidence)
    reviews = []
    for dev, q in stats.groupby("device_id", sort=False):
        e, r = evidence[dev], registry.loc[dev]
        row = {"device_id": dev, "session_id": path.stem, "capture_file": path.name,
               "device_mac": r["mac"], "device_ips": json_list(e["ips"]),
               "peer_ips": json_list(e["peer_ips"]), "peer_macs": json_list(e["peer_macs"]),
               "service_ports": json_list(e["services"]), "transports_seen": json_list(e["transports"]),
               "protocols_seen": json_list(e["protocols"]), "message_types": json_list(e["messages"]),
               "model": json_list(e["models"]), "station_name": json_list(e["stations"]),
               "identity_packet_refs": json_list(e["identity_refs"]), "vendor": r.get("vendor", ""),
               "vendor_source": "registry_oui_lookup_unverified", "evidence_source": str(path.relative_to(ROOT)),
               "evidence_packet_refs": json_list(e["refs"]), "protocol_detection_method": decode["status"] + ";l2_ethertype_when_present",
               "dissection_scope": "sampled_synthetic_ipv4_payload;not_full_capture_or_complete_reassembly",
               "label": "", "label_reason": "", "label_quality": "unlabeled", "review_status": "pending",
               "reviewer": "", "reviewed_at": "", "proposed_label": "",
               "ambiguity_note": "Port/OUI/transport role do not prove OT function; absent evidence does not prove non-OT.",
               "n_windows": len(q), "packet_count": int(q.packet_count.sum()),
               "observed_seconds": float(q.last_observed.max() - q.first_observed.min()),
               "first_observed": float(q.first_observed.min()), "last_observed": float(q.last_observed.max()),
               "role_evidence": "TCP SYN/SYN-ACK observed before window end; otherwise port heuristic; role not functional label",
               "role_hint": "unknown", "source_group": "", "source_group_status": "not_verified",
               "legacy_label": r.get("ground_truth_family", ""), "legacy_label_source": r.get("label_source", ""),
               "legacy_protocols_seen": r.get("protocols_seen", ""), "legacy_evidence_scope": "device_across_sessions_not_session_verified",
               "feature_version": VERSION}
        if flow_parts:
            dq = f[f.device_id == dev]
            roles = set(dq.srv.dropna())
            row["role_hint"] = "both" if roles == {0.0, 1.0} else "server_hint" if roles == {1.0} else "client_hint" if roles == {0.0} else "unknown"
            row["flow_count"] = int(dq.flow_key.nunique())
        else:
            row["flow_count"] = 0
        reviews.append(row)
    audit = {"raw_packets": pf.metadata.num_rows, "scoped_packets": selected_packets, "invalid_timestamps": invalid_timestamps,
             "duplicate_flow_packets_excluded": duplicates, "windows": len(stats), "review_rows": len(reviews),
             "dissection": decode, "sample_cap": sample_cap}
    return stats.drop(columns=["bc_count"]), pd.DataFrame(reviews), audit


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_WORKER_CONTEXT = None


def init_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)


def process_session(job):
    session, m2d = job
    ctx = _WORKER_CONTEXT
    cache, registry = ctx["cache"], ctx["registry"]
    wp, rp, ap = [cache / (session + suffix) for suffix in [".windows.parquet", ".review.parquet", ".audit.json"]]
    raw_path = ROOT / "Dataset/raw_parquet" / (session + ".parquet")
    stat = raw_path.stat()
    fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if wp.exists() and rp.exists() and ap.exists() and json.loads(ap.read_text(encoding="utf-8")).get("input") == fingerprint:
        return session, None
    windows, review, audit = build_capture(raw_path, m2d, registry, ctx["ot_ports"], ctx["tshark"], ctx["fields"], ctx["batch_size"], ctx["sample_cap"])
    if windows.duplicated(KEY).any():
        raise ValueError("Duplicate window keys")
    windows.to_parquet(wp.with_suffix(".tmp"), index=False)
    wp.with_suffix(".tmp").replace(wp)
    review.to_parquet(rp.with_suffix(".tmp"), index=False)
    rp.with_suffix(".tmp").replace(rp)
    audit["input"] = fingerprint
    ap.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return session, audit


def behavior_pairs(registry_path):
    """Device/session pairs for every in-scope MAC that has a device-flow row in that capture."""
    reg = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    mac_to_dev = dict(zip(reg.loc[reg.include_in_behavior_dataset.eq("True"), "mac"],
                          reg.loc[reg.include_in_behavior_dataset.eq("True"), "device_id"]))
    rows = []
    for p in sorted((ROOT / "Dataset/processed/device_flows").glob("*.parquet")):
        session = re.sub(r"__l2$", "", p.stem)
        for mac in pq.read_table(p, columns=["device_mac"]).column(0).unique().to_pylist():
            if mac in mac_to_dev:
                rows.append((mac_to_dev[mac], session))
    pairs = pd.DataFrame(rows, columns=["device_id", "session_id"]).drop_duplicates()
    return pairs.sort_values(["session_id", "device_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Smoke test only; writes to separate smoke directory")
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--sample-cap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2, help="Independent captures in parallel; bounded memory")
    parser.add_argument("--tshark", default="C:/Program Files/Wireshark/tshark.exe")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.sample_cap <= 0 or args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch size, sample cap and limit must be positive")
    registry_path = ROOT / "Dataset/metadata/devices.csv"
    pairs = behavior_pairs(registry_path)
    pairs_hash = hashlib.sha256(pairs.to_csv(index=False).encode()).hexdigest()
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False).set_index("device_id")
    if not registry.index.is_unique or registry.mac.duplicated().any():
        raise ValueError("Registry device IDs and MAC addresses must be unique")
    tshark = tshark_dpi.find_tshark(args.tshark)
    all_fields = subprocess.run([tshark, "-G", "fields"], capture_output=True, timeout=60, check=True).stdout.decode("utf-8", errors="replace")
    supported = {line.split("\t")[2] for line in all_fields.splitlines() if line.startswith("F\t") and len(line.split("\t")) > 2}
    fields = [f for f in FIELDS if f in supported]
    if "frame.number" not in fields or "frame.protocols" not in fields:
        raise RuntimeError("tshark does not expose required frame fields")
    out = ROOT / "Dataset/features" / ("v6_smoke" if args.limit else "v6_behavior_build")
    out.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "Dataset/features" if not args.limit else out
    existing_review = destination / "device_session_review_v6.csv"
    if existing_review.exists():
        old = pd.read_csv(existing_review, dtype=str, keep_default_na=False)
        edited = old.get("review_status", pd.Series("pending", index=old.index)).ne("pending")
        for field in ["label", "label_reason", "reviewer", "reviewed_at", "proposed_label"]:
            if field in old:
                edited |= old[field].str.strip().ne("")
        if edited.any():
            raise RuntimeError("Review file contains human work. Preserve it and use a new dataset version before rebuilding.")
    sessions = sorted(pairs.session_id.unique())
    if args.limit:
        sessions = sessions[:args.limit]
    ot_ports = port_catalog()
    run_config = {"version": VERSION, "script_sha256": sha256(Path(__file__)), "pairs_sha256": pairs_hash,
                  "registry_sha256": sha256(registry_path), "sample_cap": args.sample_cap,
                  "ot_ports": sorted(ot_ports), "fields": fields,
                  "tshark_version": tshark_dpi.get_tshark_version(tshark)}
    config_path = out / "build_config.json"
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / "captures" / config_hash
    cache.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    started = time.monotonic()
    failures = []
    context = dict(cache=cache, registry=registry, ot_ports=ot_ports, tshark=tshark,
                   fields=fields, batch_size=args.batch_size, sample_cap=args.sample_cap)
    jobs = [(s, {registry.loc[dev, "mac"]: dev for dev in pairs.loc[pairs.session_id == s, "device_id"]}) for s in sessions]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(context,)) as pool:
        futures = {pool.submit(process_session, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                session, audit = future.result()
            except Exception as exc:
                failures.append({"session": futures[future], "error": repr(exc)})
                print(f"[{i}/{len(sessions)}] ERROR {futures[future]}: {exc}", flush=True)
                continue
            detail = "cached" if audit is None else f"{audit['scoped_packets']:,} packets -> {audit['windows']:,} windows"
            print(f"[{i}/{len(sessions)}] {session}: {detail}; {time.monotonic()-started:.0f}s", flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} captures failed; no complete master published; see {out / 'failures.json'}")
    windows = pd.concat([pd.read_parquet(cache / (s + ".windows.parquet")) for s in sessions], ignore_index=True)
    review = pd.concat([pd.read_parquet(cache / (s + ".review.parquet")) for s in sessions], ignore_index=True)
    # Explicit allowlist means joined evidence cannot leak into training.
    evidence_cols = [c for c in review.columns if c not in {"packet_count", "observed_seconds", "first_observed", "last_observed", "feature_version"}]
    master = windows.merge(review[evidence_cols], on=["device_id", "session_id"], how="left", validate="many_to_one")
    if master.duplicated(KEY).any() or review.duplicated(["device_id", "session_id"]).any():
        raise ValueError("Duplicate dataset/review keys")
    expected_pairs = pairs[pairs.session_id.isin(sessions)]
    coverage = review.merge(expected_pairs, on=["device_id", "session_id"], how="outer", indicator=True)
    if coverage._merge.ne("both").any():
        raise ValueError("Device/session coverage differs from v6 scope (device_flows pairs); refusing to publish incomplete master")
    if master.label.fillna("").ne("").any():
        raise ValueError("Label assignment is outside this build")
    numeric = master[TRAINING_FEATURES].to_numpy(float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite training feature")
    if (master.flow_dur_median.dropna() > W + 1e-6).any():
        raise ValueError("Duration outside window")
    master_path = destination / "master_device_windows_v6_behavior.parquet"
    review_path = destination / "device_session_review_v6.csv"
    master.to_parquet(master_path.with_suffix(".tmp"), index=False)
    master_path.with_suffix(".tmp").replace(master_path)
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    # Diverse preview: one row per device/session first, rather than 100 consecutive windows.
    preview = master.drop_duplicates(["device_id", "session_id"]).head(100)
    preview.to_csv(destination / "preview_100_device_windows_v6.csv", index=False, encoding="utf-8-sig")
    column_roles = {c: "training_feature" if c in TRAINING_FEATURES else "experimental_not_training" if c in EXPERIMENTAL_FEATURES else "label_not_input" if c in {"label", "label_reason", "label_quality", "review_status", "reviewer", "reviewed_at"} else "metadata_not_training" for c in master.columns}
    manifest = {**run_config, "rows": len(master), "devices": master.device_id.nunique(), "sessions": master.session_id.nunique(),
                "review_rows": len(review), "columns": column_roles, "training_features": TRAINING_FEATURES,
                "labeling": "deferred; all label fields empty; legacy labels are metadata only",
                "scope": "devices.csv include_in_behavior_dataset=True (>=1 flow, not spoofed); pairs from processed/device_flows, __l2 merged into raw session; includes packet-only windows",
                "feature_semantics": "packet timestamps exact; per-window observed conversations (5tuple+VLAN, no idle timeout); duration=max(timestamp)-min(timestamp) within window; small/oneway within window; carryover=earlier observed packet in same conversation; roles as of window end",
                "raw_counter_policy": "raw frame counters include capture duplicates; flow features use directed packet duplicate policy",
                "missing_policy": "undefined ratios and no-flow features remain NaN; counts are zero only if observed absence",
                "limitations": ["same MAC across sources not verified as one physical asset", "protocol sampling is not full dissection", "synthetic frames may limit TCP protocol detection", "OT port list is heuristic; never a label", "IPv6 header evidence limited by legacy raw parser", "window rows from same conversation must not be randomly split"],
                "missing_rates": master[TRAINING_FEATURES + EXPERIMENTAL_FEATURES].isna().mean().to_dict(),
                "source_pair_coverage": len(review.merge(pairs, on=["device_id", "session_id"], how="inner")),
                "expected_source_pairs": len(pairs[pairs.session_id.isin(sessions)]),
                "output_sha256": sha256(master_path), "review_sha256": sha256(review_path)}
    (destination / "feature_manifest_v6.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {len(master):,} rows, {len(review):,} review pairs, {len(master.columns)} columns -> {master_path}", flush=True)


if __name__ == "__main__":
    main()
