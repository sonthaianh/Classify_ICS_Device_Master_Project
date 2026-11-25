"""PCAP -> device windows (same builder as training, script 26) -> Phase 1 model -> per-device result.

Reuses `build_capture` of 26_build_behavior_dataset.py so features are computed exactly as in training.
Scope in the app: every unicast MAC seen in the capture that has at least one flow window (training scope:
>=1 flow); capture with extreme MAC counts (e.g. MAC flooding) is capped at MAX_DEVICES by packet count.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS / "pcap_to_dataset"))
sys.path.insert(0, str(TOOLS / "utils"))
sys.path.insert(0, str(HERE))
import pcap_reader  # noqa: E402

_spec = importlib.util.spec_from_file_location("behavior_builder", TOOLS / "pcap_to_dataset/26_build_behavior_dataset.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

MAX_PACKETS = 3_000_000
MAX_DEVICES = 2000
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
MODEL = joblib.load(HERE / "model/phase1_lightgbm.joblib")
META = json.loads((HERE / "model/model_meta.json").read_text(encoding="utf-8"))
FEATURES = META["features"]
TRAIN_MACS = set(META["train_macs"])
OT_PROTOCOLS = {"mbtcp", "modbus", "s7comm", "s7comm-plus", "enip", "cip", "cipcls", "cipm", "dnp3", "bacnet", "bacapp", "bvlc", "iec60870_104",
                "iec60870_asdu", "goose", "sv", "profinet-rt", "pn_io", "mms", "omron", "ethercat", "opcua", "hart_ip", "ff", "codesys", "fins",
                "srtp", "knxip", "mqtt"}
RX = re.compile(r"local=(\d+),peer=(\d+)")
_TSHARK = None
_FIELDS = None


def _tshark():
    global _TSHARK, _FIELDS
    if _TSHARK is None:
        _TSHARK = B.tshark_dpi.find_tshark("C:/Program Files/Wireshark/tshark.exe")
        allf = subprocess.run([_TSHARK, "-G", "fields"], capture_output=True, timeout=60, check=True).stdout.decode("utf-8", "replace")
        _FIELDS = [f for f in B.FIELDS if f in allf]
    return _TSHARK, _FIELDS


_OUI = None


def _oui_vendor(mac):
    """Vendor from the OUI table; unverified hint only (a MAC prefix does not prove the device)."""
    global _OUI
    try:
        if _OUI is None:
            import oui_lookup
            _OUI = oui_lookup.OuiLookup()
        r = _OUI.lookup(mac)
        return r.get("full_name") or r.get("short_name") or ""
    except Exception:
        return ""


def _is_unicast(mac):
    return bool(mac) and (int(mac.split(":")[0], 16) & 1) == 0 and mac != "00:00:00:00:00:00"


def rule_evidence(protocols, service_ports):
    prot = sorted(set(protocols) & OT_PROTOCOLS)
    pairs = RX.findall(service_ports)
    peer = sorted({int(b) for _, b in pairs} & B.PORTS_CACHE)
    local = sorted({int(a) for a, _ in pairs} & B.PORTS_CACHE)
    tier = "protocol" if prot else "peer_port" if peer else "local_port_only" if local else ""
    return {"rule_label": "OT_confirmed" if tier else "unknown", "rule_tier": tier, "ot_protocols": prot, "ot_peer_ports": peer[:12], "ot_local_ports": local[:12]}


def analyze(pcap_path, job_id, progress=lambda pct, msg: None):
    t0 = time.time()
    pcap_path = Path(pcap_path)
    progress(5, "Đọc file PCAP")
    df, info = pcap_reader.read_pcap(pcap_path, MAX_PACKETS)
    if df.empty:
        raise ValueError("Không đọc được gói tin Ethernet nào trong file.")
    ts = df.timestamp[df.timestamp > 0]
    capture = {"file": pcap_path.name, "packets": int(len(df)), "truncated": bool(info["truncated"]),
               "duration_s": round(float(ts.max() - ts.min()), 1) if len(ts) else 0.0}
    macs = pd.concat([df.src_mac, df.dst_mac])
    counts = macs[macs.map(_is_unicast)].value_counts()
    capped = len(counts) > MAX_DEVICES
    counts = counts.head(MAX_DEVICES)
    capture["unicast_macs"] = int(len(counts)); capture["devices_capped"] = bool(capped)
    registry = pd.DataFrame({"device_id": [f"D{i + 1:04d}" for i in range(len(counts))], "mac": counts.index, "vendor": [_oui_vendor(m) for m in counts.index],
                             "ground_truth_family": "", "label_source": "", "protocols_seen": ""}).set_index("device_id", drop=False)
    mac_to_dev = dict(zip(registry.mac, registry.device_id))
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    raw = job_dir / f"{job_id}.parquet"
    df.to_parquet(raw, index=False)
    progress(20, "Tính đặc trưng theo cửa sổ 45 giây")
    tshark, fields = _tshark()
    B.PORTS_CACHE = B.port_catalog()
    windows, review, audit = B.build_capture(raw, mac_to_dev, registry, B.PORTS_CACHE, tshark, fields, sample_cap=500)
    progress(75, "Chạy mô hình Phase 1")
    windows = windows[windows.device_id.isin(windows.loc[windows.has_flow_packets, "device_id"].unique())].copy()
    if windows.empty:
        raise ValueError("Không có thiết bị nào có flow (TCP/UDP hoặc L2 OT) trong file; mô hình cần ít nhất một flow.")
    windows["prob"] = MODEL.predict_proba(windows[FEATURES])[:, 1]
    rev = review.set_index("device_id")
    devices = []
    for dev, w in windows.groupby("device_id"):
        r = rev.loc[dev]
        prot = json.loads(r.protocols_seen)
        rule = rule_evidence(prot, r.service_ports)
        ports = pd.Series([int(b) for _, b in RX.findall(r.service_ports[:3_000_000])]).value_counts().head(6)
        w = w.sort_values("window_start")
        devices.append({"device_id": dev, "mac": r.device_mac, "vendor": r.vendor, "in_training": r.device_mac in TRAIN_MACS,
                        "ips": json.loads(r.device_ips)[:6], "n_peer_ips": len(json.loads(r.peer_ips)), "protocols": prot[:15],
                        "message_types": json.loads(r.message_types)[:10], "top_peer_ports": {str(k): int(v) for k, v in ports.items()},
                        "role_hint": r.role_hint, "n_windows": int(len(w)), "n_flow_windows": int(w.has_flow_packets.sum()), "packets": int(r.packet_count),
                        "flows": int(r.flow_count), "score": float(w.prob.mean()), "score_max": float(w.prob.max()), **rule,
                        "features_median": {c: (None if pd.isna(w[c].median()) else round(float(w[c].median()), 4)) for c in FEATURES},
                        "timeline": [{"t": round(float(a - w.window_start.min()), 0), "p": round(float(p), 4)} for a, p in zip(w.window_start, w.prob)][:400]})
    devices.sort(key=lambda d: -d["score"])
    overlap = sum(d["in_training"] for d in devices)
    notes = []
    phase2_info = None
    progress(88, "Dự đoán vai trò thiết bị (Phase 2)")
    try:
        import phase2
        phase2_info = phase2.annotate(df, devices, pcap_path.stem)
        n_pred = sum(1 for d in devices if d.get("phase2"))
        notes.append(f"Phase 2 (vai trò PLC/HMI/EWS/OTHER) chỉ chạy cho thiết bị có điểm Phase 1 từ {str(phase2.THRESHOLD).replace('.', ',')} trở lên: {n_pred} thiết bị được dự đoán. Độ chính xác cân bằng tham khảo {str(phase2_info['cv_reference']['balanced_acc']).replace('.', ',')} (4 lớp, ngẫu nhiên 0,25); HMI và EWS yếu hơn.")
    except Exception as e:  # Phase 1 van tra ket qua neu Phase 2 loi
        notes.append(f"Không chạy được Phase 2: {type(e).__name__}: {e}")
    if overlap:
        notes.append(f"{overlap}/{len(devices)} thiết bị có MAC nằm trong dữ liệu huấn luyện: điểm của các thiết bị này KHÔNG phải kết quả kiểm tra độc lập.")
    if capture["truncated"]:
        notes.append(f"File lớn: chỉ xử lý {MAX_PACKETS:,} gói đầu tiên.")
    if capped:
        notes.append(f"Có hơn {MAX_DEVICES} MAC unicast (nghi ngờ MAC flooding): chỉ giữ {MAX_DEVICES} MAC có nhiều gói nhất.")
    if capture["duration_s"] < 45:
        notes.append("Capture ngắn hơn 45 giây: mỗi thiết bị chỉ có rất ít cửa sổ, điểm kém ổn định.")
    low = sum(d["n_flow_windows"] < 3 for d in devices)
    if low:
        notes.append(f"{low} thiết bị có dưới 3 cửa sổ có flow: bằng chứng ít, nên đọc kết quả thận trọng.")
    result = {"capture": capture, "devices": devices, "notes": notes, "model": {k: META[k] for k in ("algorithm", "features", "default_threshold", "train_devices", "labels", "cv_reference_device_level")},
              "phase2_model": phase2_info, "elapsed_s": round(time.time() - t0, 1), "dissection": audit.get("dissection", {}).get("status", "")}
    (job_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    raw.unlink(missing_ok=True)
    progress(100, "Xong")
    return result
