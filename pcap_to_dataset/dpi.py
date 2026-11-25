"""Deep Packet Inspection for port 102 (ISO-on-TCP).

Identifies the exact application protocol inside TPKT-encapsulated traffic:
    - S7comm      (Siemens classic)       — magic byte 0x32
    - S7comm-Plus (Siemens TIA Portal)    — magic byte 0x72
    - MMS         (IEC 61850)             — ISO Session SPDU types
    - COTP        (connection-oriented TP) — fallback

Port 102 protocol stack:
    TPKT (4 bytes) → COTP (variable) → Application
"""

import logging

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MMS_SESSION_SPDU_TYPES = {
    0x01,  # Give Tokens + Data Transfer (most common — carries MMS data)
    0x0D,  # Connect (CN) — MMS session setup
    0x0E,  # Accept (AC) — MMS session accepted
    0x09,  # Finish (FN) — MMS session end
    0x0A,  # Disconnect (DN)
    0x19,  # Abort (AB)
}

_SPDU_NAMES = {
    0x01: "DT (Data Transfer)",
    0x0D: "CN (Connect)",
    0x0E: "AC (Accept)",
    0x09: "FN (Finish)",
    0x0A: "DN (Disconnect)",
    0x19: "AB (Abort)",
}

_MMS_PDU_TYPES = {
    0xA0: "Confirmed-Request",
    0xA1: "Confirmed-Response",
    0xA2: "Confirmed-Error",
    0xA3: "Unconfirmed-PDU",
    0xA4: "Reject-PDU",
    0xA8: "Initiate-Request",
    0xA9: "Initiate-Response",
    0xAA: "Initiate-Error",
    0x8B: "Conclude-Request",
    0x8C: "Conclude-Response",
}


# ── Single-packet DPI ────────────────────────────────────────────────────────

def dpi_port102(payload) -> tuple:
    """Identify the exact protocol inside a port-102 (ISO-on-TCP) packet.

    Parameters
    ----------
    payload : bytes, bytearray, or hex str
        Application-layer payload from a TCP port 102 packet.

    Returns
    -------
    (protocol_name: str, evidence: str)
        Empty strings when payload is too short or not a TPKT frame.
    """
    if isinstance(payload, str):
        try:
            payload = bytes.fromhex(payload)
        except (ValueError, TypeError):
            return "", ""

    if payload is None or len(payload) < 4:
        return "", ""

    # Step 1: TPKT header — version must be 3
    if payload[0] != 0x03 or payload[1] != 0x00:
        return "", ""

    # Step 2: COTP header (starts at offset 4)
    if len(payload) < 5:
        return "TPKT", "TPKT header only, no COTP"

    cotp_length = payload[4]
    cotp_pdu_type = payload[5] & 0xF0 if len(payload) > 5 else 0

    # Application data starts after TPKT(4) + length-byte(1) + cotp_length
    app_offset = 4 + 1 + cotp_length

    if len(payload) <= app_offset:
        return "COTP", f"COTP PDU=0x{cotp_pdu_type:02X}, no app data"

    # Step 3: First byte after COTP identifies the application protocol
    first_byte = payload[app_offset]

    # ── S7comm (Siemens classic): magic 0x32 ──
    if first_byte == 0x32:
        rosctr = payload[app_offset + 1] if len(payload) > app_offset + 1 else None
        ev = "TPKT+COTP+S7comm (0x32)"
        if rosctr is not None:
            ev += f" ROSCTR=0x{rosctr:02X}"
        return "S7comm", ev

    # ── S7comm-Plus (Siemens TIA Portal): magic 0x72 ──
    if first_byte == 0x72:
        return "S7comm-Plus", "TPKT+COTP+S7commPlus (0x72)"

    # ── MMS (IEC 61850): ISO Session SPDU types ──
    if first_byte in _MMS_SESSION_SPDU_TYPES:
        spdu_name = _SPDU_NAMES.get(first_byte, f"0x{first_byte:02X}")

        mms_detail = ""
        for b in payload[app_offset:app_offset + 80]:
            if b in _MMS_PDU_TYPES:
                mms_detail = _MMS_PDU_TYPES[b]
                break

        ev = f"TPKT+COTP+Session SPDU={spdu_name}"
        if mms_detail:
            ev += f", MMS PDU={mms_detail}"
        return "MMS", ev

    # ── COTP only (unrecognized application byte) ──
    return "COTP", f"TPKT+COTP, unknown app byte=0x{first_byte:02X}"


# ── Batch DPI ─────────────────────────────────────────────────────────────────

def batch_dpi_port102(payloads, dst_ports, src_ports) -> list:
    """Apply DPI on port-102 packets in batch.

    Parameters
    ----------
    payloads  : list of bytes/None
    dst_ports : list of int/None
    src_ports : list of int/None

    Returns
    -------
    list[str] of protocol names (same length as inputs).
    Non-port-102 packets get empty string.
    """
    n = len(payloads)
    results = [""] * n
    hit = 0

    for i in range(n):
        dp = dst_ports[i] or 0
        sp = src_ports[i] or 0
        if dp == 102 or sp == 102:
            p = payloads[i]
            if p and len(p) > 0:
                proto, _ = dpi_port102(p)
                results[i] = proto
                if proto:
                    hit += 1

    if hit > 0:
        log.debug(f"  DPI port 102: {hit} packets identified")
    return results
