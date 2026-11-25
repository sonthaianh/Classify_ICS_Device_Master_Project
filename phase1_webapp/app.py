"""Phase 1 + Phase 2 test web app: upload a PCAP/PCAPNG -> per-device OT score (Phase 1) and, for OT devices, role PLC/HMI/EWS/OTHER (Phase 2, model_186).

Run:  python Tools/phase1_webapp/app.py      then open http://127.0.0.1:5001
Local use only (binds to 127.0.0.1). Uploaded captures are deleted after analysis; only the result JSON is kept.
"""
from __future__ import annotations

import csv
import io
import json
import secrets
import threading
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
UPLOADS = HERE / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED = {".pcap", ".pcapng", ".cap"}
MAX_MB = 300

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()          # one analysis at a time (tshark + memory heavy)
_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        import pipeline as p  # heavy import (loads model + feature builder) only once
        _pipeline = p
    return _pipeline


def run_job(job_id: str, path: Path):
    job = JOBS[job_id]

    def progress(pct, msg):
        job.update(progress=pct, message=msg)

    try:
        job.update(status="waiting", message="Đang chờ tới lượt xử lý", progress=1)
        with LOCK:
            job.update(status="running")
            job["result"] = pipeline().analyze(path, job_id, progress)
            job["result"]["capture"]["file"] = job["name"]
            job["status"] = "done"
    except Exception as e:  # report to the UI, keep server alive
        job.update(status="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=4))
    finally:
        path.unlink(missing_ok=True)


@app.get("/")
def index():
    meta = json.loads((HERE / "model/model_meta.json").read_text(encoding="utf-8"))
    meta.pop("train_macs", None)
    try:
        import phase2
        p2 = phase2.info()
    except Exception:
        p2 = None
    return render_template("index.html", meta=meta, p2=p2, max_mb=MAX_MB)


@app.post("/api/upload")
def upload():
    f = request.files.get("pcap")
    if not f or not f.filename:
        return jsonify(error="Chưa chọn file"), 400
    name = secure_filename(f.filename)
    if Path(name).suffix.lower() not in ALLOWED:
        return jsonify(error="Chỉ nhận .pcap, .pcapng hoặc .cap"), 400
    job_id = secrets.token_hex(8)
    path = UPLOADS / f"{job_id}_{name}"
    f.save(path)
    JOBS[job_id] = {"status": "queued", "progress": 0, "message": "Đã nhận file", "name": name}
    threading.Thread(target=run_job, args=(job_id, path), daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/status/<job_id>")
def status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify(error="Không tìm thấy job"), 404
    return jsonify({k: j.get(k) for k in ("status", "progress", "message", "name")})


@app.get("/api/result/<job_id>")
def result(job_id):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        return jsonify(error="Chưa có kết quả"), 404
    return jsonify(j["result"])


@app.get("/api/export/<job_id>.csv")
def export(job_id):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        return "Chưa có kết quả", 404
    thr = float(request.args.get("threshold", j["result"]["model"]["default_threshold"]))
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["device_id", "mac", "vendor", "ips", "n_windows", "n_flow_windows", "score_mean", "score_max", "decision_at_threshold", "threshold",
                "rule_label", "rule_tier", "ot_protocols", "ot_peer_ports", "ot_local_ports", "in_training_data",
                "phase2_role", "phase2_uncertain", "phase2_second", "phase2_margin", "phase2_evidence", "phase2_prob_PLC", "phase2_prob_HMI", "phase2_prob_EWS", "phase2_prob_OTHER", "phase2_status"])
    for d in j["result"]["devices"]:
        w.writerow([d["device_id"], d["mac"], d["vendor"], ";".join(d["ips"]), d["n_windows"], d["n_flow_windows"], round(d["score"], 4), round(d["score_max"], 4),
                    "OT" if d["score"] >= thr else "unknown", thr, d["rule_label"], d["rule_tier"], ";".join(d["ot_protocols"]),
                    ";".join(map(str, d["ot_peer_ports"])), ";".join(map(str, d["ot_local_ports"])), d["in_training"],
                    *(lambda p: [p["pred"], p["uncertain"], p["second"], p["margin"], p["evidence"], *[p["prob"].get(k, "") for k in ("PLC", "HMI", "EWS", "OTHER")]] if p else [""] * 9)(d.get("phase2")),
                    d.get("phase2_status", "")])
    return Response("﻿" + out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=phase1_{job_id}.csv"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
