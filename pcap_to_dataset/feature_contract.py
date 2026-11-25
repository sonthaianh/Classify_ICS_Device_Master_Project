"""Explicit feature allowlists. Evidence, identity and labels never enter X."""
from __future__ import annotations

import sys


def configure_console_utf8():
    """Shared CLI setup: Windows pipes may otherwise default to cp1252.

    Capture names contain Unicode. Console encoding must not interrupt a
    complete build; this does not change feature calculations or cache values.
    Notebook/captured streams without reconfigure are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


configure_console_utf8()

TRAINING_FEATURES = [
    "pkts_per_flow", "bytes_per_packet", "udp_ratio", "packet_rate",
    "small_flow_ratio", "one_way_flow_ratio", "flow_dur_median",
    "unique_local_port_count", "svc_port_unique", "frames_sent",
    "arp_frame_count", "arp_target_unique", "icmp_frame_count",
    "nonip_frame_count", "bcast_frame_ratio", "carryover_flow_ratio", "ip_count",
]
EXPERIMENTAL_FEATURES = [
    "server_role_ratio", "role_known_ratio", "role_syn_ratio", "sent_byte_ratio",
    "started_before_capture_ratio", "ot_flow_ratio", "ot_pkt_share", "frame_rate",
]


def select_training_features(frame, columns=None):
    """Fail closed: extra numeric evidence must never silently become features."""
    selected = list(TRAINING_FEATURES if columns is None else columns)
    forbidden = set(selected) - set(TRAINING_FEATURES)
    if forbidden:
        raise ValueError(f"Columns outside D17 allowlist: {sorted(forbidden)}")
    missing = set(selected) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing training features: {sorted(missing)}")
    return frame.loc[:, selected].copy()
