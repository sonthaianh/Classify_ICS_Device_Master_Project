"""Phase 2 cho ung dung web: du doan vai tro (PLC / HMI / EWS / OTHER) cua thiet bi qua cong Phase 1, bang mo hinh huan luyen tren bo 186 nhan.

Dung lai ham cua script 69 (cung cong thuc ho so payload, vai tro ket noi, tac vu ky thuat nhu luc huan luyen) va mo hinh trong phase2_infer/model_186/
(huan luyen bang 99_phase2_infer_186.py --train). Khong sua, khong huan luyen lai gi o day.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
MODEL_DIR = TOOLS / "phase2_infer" / "model_186"
THRESHOLD = 0.5            # giong nguyen mau o muc 3.6.2: chi du doan vai tro cho thiet bi co diem Phase 1 >= 0,5
_S: dict = {}


def _state():
    if not _S:
        spec = importlib.util.spec_from_file_location("p69_webapp", TOOLS / "pcap_to_dataset" / "69_phase2_infer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _S.update(mod=mod, rf=joblib.load(MODEL_DIR / "phase2_rf.joblib"), meta=json.loads((MODEL_DIR / "phase2_meta.json").read_text(encoding="utf-8")))
    return _S


def info():
    """Thong tin mo hinh de hien thi trong giao dien (khong nap thu vien nang)."""
    meta = json.loads((MODEL_DIR / "phase2_meta.json").read_text(encoding="utf-8"))
    return {"n_train": meta["n_train"], "class_counts": meta["class_counts"], "cv_reference": meta["cv_reference"], "caveat": meta["caveat"], "threshold": THRESHOLD}


def annotate(df, devices, capture_name):
    """Them khoa 'phase2' va 'phase2_status' vao tung thiet bi trong `devices` (danh sach dict cua pipeline.analyze). Tra ve thong tin mo hinh."""
    s = _state()
    mod, rf, meta = s["mod"], s["rf"], s["meta"]
    cols, classes, train_macs = meta["columns"], list(rf.classes_), set(meta["train_macs"])
    todo = {d["mac"].lower(): d["device_id"] for d in devices if d["score"] >= THRESHOLD}
    recs, hs = mod.profile_from_df(df, todo, capture_name)
    for d in devices:
        dev, mac = d["device_id"], d["mac"].lower()
        d["phase2"], d["in_training_phase2"] = None, mac in train_macs
        if dev not in todo.values():
            d["phase2_status"] = "Không chạy: điểm Phase 1 dưới 0,5"
            continue
        if dev not in recs:
            d["phase2_status"] = "Không có gói payload TCP/UDP: không đủ bằng chứng"
            continue
        rec = recs[dev]
        beh = [(-1.0 if d["features_median"].get(c) is None else float(d["features_median"][c])) for c in mod.S58.B.FEATS]
        x = mod.feature_row(dev, rec, hs[dev], beh, cols)
        pr = rf.predict_proba(x.to_frame().T)[0]
        order = np.argsort(-pr)
        n_tok = int(sum(rec["tokens"].values()))
        a = mod.S43.aggregate_records([rec])[dev]
        role_v = mod.role_vector(a, *hs[dev])
        role_txt = "unknown" if role_v[0] < 0 else ("server" if role_v[0] >= 0.8 else "client" if role_v[0] <= 0.2 else "mixed")
        sub = df[((df.src_mac == mac) | (df.dst_mac == mac)) & (df.payload_len.fillna(0) > 0)]
        ts = sub.timestamp[sub.timestamp > 0]
        span = float(ts.max() - ts.min()) if len(ts) else 0.0
        evidence = "strong" if (n_tok >= 100 and span >= 45) else "weak" if n_tok > 0 else "no_tokens"
        margin = float(pr[order[0]] - pr[order[1]])
        d["phase2"] = {"pred": classes[order[0]], "prob": {c: round(float(p), 3) for c, p in zip(classes, pr)}, "second": classes[order[1]], "margin": round(margin, 3),
                       "uncertain": bool(margin < 0.10), "evidence": evidence, "span_s": round(span, 3), "n_payload_pkts": int(rec["n_pkts"]), "n_message_tokens": n_tok,
                       "port_rule_role": role_txt, "syn_sent_ot": int(hs[dev][0]), "synack_sent_ot": int(hs[dev][1])}
        d["phase2_status"] = "predicted"
    return info()
