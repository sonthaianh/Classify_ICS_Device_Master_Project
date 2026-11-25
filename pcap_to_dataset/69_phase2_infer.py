"""Phase 2 tren MOT file PCAP: Phase 1 (OT?) -> cong -> ho so payload -> feature -> RandomForest 4 lop PLC / HMI / EWS / OTHER (2026-11-11).

Dung lai code da kiem chung: doc PCAP = phase1_webapp/pcap_reader (cung 18 cot voi raw_parquet); diem Phase 1 + 15 feature hanh vi = phase1_webapp/pipeline (script 26);
ho so payload = cung logic script 67 (family_and_role, msg_token cua script 42); feature payload/vai tro/tac vu = cung cong thuc script 58/57/45.
Mo hinh = RandomForest huan luyen tren TOAN BO thiet bi co nhan va co ho so (bo feature 'behaviour_payload_role_eng', 64 cot).

  py -3.14 Tools/pcap_to_dataset/69_phase2_infer.py --train                 # huan luyen mo hinh cuoi -> Tools/phase2_infer/model/
  py -3.14 Tools/pcap_to_dataset/69_phase2_infer.py <file.pcap> [--all]     # phan tich 1 file; --all: du doan Phase 2 ca thiet bi khong qua cong Phase 1
  py -3.14 Tools/pcap_to_dataset/69_phase2_infer.py --selftest              # kiem tra tuong thich feature: cong thuc ban moi == ma tran huan luyen

GIOI HAN (doc truoc khi tin ket qua): mo hinh chi dung 117 nhan (PLC 37 / HMI 11 / EWS 15 / OTHER 54), chua chuyen gia xac nhan; do chinh xac cheo ~0,58 (4 lop, ngau nhien 0,25),
HMI va EWS yeu (recall ~0,3-0,5); thiet bi khong co thong diep OT (token) thuong bi doan OTHER vi nhan OTHER phan lon la thiet bi khong noi giao thuc OT.
"""
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
APP = TOOLS / "phase1_webapp"
OUTDIR = TOOLS / "phase2_infer"
MODEL_DIR = OUTDIR / "model"
FEATURE_SET = "behaviour_payload_role_eng"
CUT, CAP_ROWS, MIN_PKTS_FOR_PRED = 300, 60000, 1
PHASE1_THRESHOLD = 0.5


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S42 = _load("s42", "42_phase2_payload_profiles.py")
S43 = _load("s43", "43_phase2_similarity_labels.py")
S45 = _load("s45", "45_phase2_role_confirm.py")
S57 = _load("s57", "57_phase2_eng_features.py")
S58 = _load("s58", "58_phase2_train4.py")
S67 = _load("s67", "67_phase2_payload_profiles_full.py")
F = S43.F
CELLS = [f"{f}|{r}" for f in S43.FAMS for r in ("S", "C")]
OT_FAMS = set(S45.OT_FAMS)
NON_OT = ("it_service", "unknown")


# ------------------------------------------------------------------ feature building (same formulas as script 58 / 57 / 45)
def profile_from_df(df, macs, capture):
    """In-memory version of script 67's process_file. macs: {mac: device_id}. Returns (records, handshake dict)."""
    t = df[(df.payload_len.fillna(0) > 0) & df.ip_proto.isin([6, 17]) & (df.src_mac.isin(macs) | df.dst_mac.isin(macs))]
    A = {m: {"fam_role": Counter(), "tokens": Counter(), "tx": Counter(), "rx": Counter(), "ps": set(), "pc": set(), "n": 0, "rows": 0} for m in macs}
    for a, b, sa, da, s_, d_, pr, pl, p in zip(t.src_mac, t.dst_mac, t.src_ip, t.dst_ip, t.src_port, t.dst_port, t.ip_proto, t.payload_len, t.app_payload):
        for mac, sender_is_dev in ((a, True), (b, False)):
            if mac not in A or (not sender_is_dev and a == b):
                continue
            acc = A[mac]
            acc["rows"] += 1
            if pd.isna(s_) or pd.isna(d_):
                continue
            fam, role = S42.family_and_role(int(s_), int(d_), sender_is_dev)
            acc["n"] += 1
            acc["fam_role"][f"{fam}|{role}"] += 1
            acc["tx" if sender_is_dev else "rx"][int(pl)] += 1
            (acc["pc"] if role == "S" else acc["ps"]).add(da if sender_is_dev else sa)
            if p:
                sender_is_server = (role == "S") == sender_is_dev
                tk = S42.msg_token(fam, bytes(p)[:CUT], sender_is_server, int(pr) == 6)
                if tk:
                    acc["tokens"][f"{role}:{'tx' if sender_is_dev else 'rx'}:{tk}"] += 1
    recs = {}
    for mac, dev in macs.items():
        acc = A[mac]
        if acc["rows"] == 0:
            continue
        recs[dev] = {"n_pkts": acc["n"], "fam_role": dict(acc["fam_role"]), "tokens": dict(acc["tokens"]), "tx_n": sum(acc["tx"].values()), "rx_n": sum(acc["rx"].values()),
                     "tx_len_med": S67.hist_percentile(acc["tx"], 50), "tx_len_p90": S67.hist_percentile(acc["tx"], 90),
                     "rx_len_med": S67.hist_percentile(acc["rx"], 50), "rx_len_p90": S67.hist_percentile(acc["rx"], 90),
                     "n_clients_of_device": len(acc["pc"]), "n_servers_contacted": len(acc["ps"]), "device_id": dev, "session_id": capture, "n_rows_total": acc["rows"], "truncated": False}
    # TCP handshakes on OT service ports (same rule as script 45: SYN sent = client, SYN-ACK sent = server; first CAP_ROWS per device)
    tcp = df[(df.ip_proto == 6) & df.tcp_flags.notna()]
    fl = tcp.tcp_flags.astype(int) & 0x12
    hs = {}
    for mac, dev in macs.items():
        s1 = tcp[(fl == 0x02) & (tcp.src_mac == mac)].head(CAP_ROWS)
        s2 = tcp[(fl == 0x12) & (tcp.src_mac == mac)].head(CAP_ROWS)
        syn = sum(1 for dp in s1.dst_port if not pd.isna(dp) and S45.fam_of(int(dp)) in OT_FAMS)
        ack = sum(1 for sp in s2.src_port if not pd.isna(sp) and S45.fam_of(int(sp)) in OT_FAMS)
        hs[dev] = (syn, ack)
    return recs, hs


def payload_vector(a):
    """Same formula as script 58 build(): 14 family|role shares + 7 numeric payload stats (a = aggregate from S43.aggregate_records)."""
    nn = lambda z: 0.0 if z is None or (isinstance(z, float) and np.isnan(z)) else float(np.log1p(z))
    tot = max(sum(a["fam_role"].values()), 1)
    v = [a["fam_role"].get(c, 0) / tot for c in CELLS]
    return v + [nn(a["tx_med"]), nn(a["tx_p90"]), nn(a["rx_med"]), nn(a["rx_p90"]), a["tx_share"], np.log1p(a["n_clients"]), np.log1p(a["n_servers"])]


def role_vector(a, syn, ack):
    ot = {k: v for k, v in a["fam_role"].items() if not k.startswith(NON_OT)}
    n = sum(ot.values())
    srv = round(sum(v for k, v in ot.items() if k.endswith("|S")) / n, 3) if n else np.nan
    n_hs = syn + ack
    cs = round(syn / n_hs, 3) if n_hs else np.nan
    f = lambda z: -1.0 if pd.isna(z) else float(z)
    return [f(srv), f(cs), float(np.log1p(syn)), float(np.log1p(ack))]


def eng_vector(dev, rec):
    tok, fam, tx, rx, srv = {dev: Counter(rec["tokens"])}, {dev: Counter(rec["fam_role"])}, {dev: rec["tx_n"]}, {dev: rec["rx_n"]}, {dev: rec["n_servers_contacted"]}
    e = S57.feats(dev, tok, fam, tx, rx, srv)
    return [0.0 if pd.isna(e.get(k)) else float(e[k]) for k in S58.ENG]


def feature_row(dev, rec, hs, beh15, columns):
    a = S43.aggregate_records([rec])[dev]
    v = list(beh15) + payload_vector(a) + role_vector(a, *hs) + eng_vector(dev, rec)
    assert len(v) == len(columns), (len(v), len(columns))
    return pd.Series(v, index=columns, dtype=float)


# ------------------------------------------------------------------ training
def train():
    lab, sets = S58.build()
    X, y = sets[FEATURE_SET], lab.type4
    m = lab.has_profile
    Xa, ya = X[m], y[m]
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xa, ya)
    macs = pd.read_csv(F / "device_labels_v6_rule.csv", encoding="utf-8-sig", dtype=str).set_index("device_id").device_mac
    cv = json.loads((F / "phase2_train4_v6.json").read_text(encoding="utf-8"))["all_labels"]["A_profiled_only"][FEATURE_SET]
    tk = json.loads((F / "phase2_train4_tokens_only_v6.json").read_text(encoding="utf-8"))["all_labels"].get(FEATURE_SET, {})
    meta = {"trained_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "feature_set": FEATURE_SET, "columns": list(X.columns), "classes": list(rf.classes_),
            "n_train": int(len(Xa)), "class_counts": ya.value_counts().to_dict(), "label_sources": lab[m].source.value_counts().to_dict(),
            "profile_file": os.environ.get("P2_PROFILE", "phase2_payload_profiles_v6.json"),
            "train_macs": sorted({macs[d].lower() for d in Xa.index}),
            "cv_reference": {"balanced_acc": cv["balanced_acc"], "balanced_acc_std": cv["balanced_acc_std"], "recall": cv["recall"], "n": int(len(Xa)), "chance": 0.25,
                             "tokens_only_subset": {"balanced_acc": tk.get("balanced_acc"), "recall": tk.get("recall"), "n": tk.get("n")} if isinstance(tk, dict) else None},
            "caveat": "labels not expert-verified; HMI 11 / EWS 15 samples; no-token devices are mostly OTHER in training"}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "phase2_rf.joblib")
    (MODEL_DIR / "phase2_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("trained", len(Xa), "devices;", meta["class_counts"], "| cv ref", meta["cv_reference"]["balanced_acc"])
    return 0


def selftest():
    """Feature parity: rebuild P / R / E vectors with THIS script's formulas from the aggregated training profiles and compare with script 58's matrix."""
    lab, sets = S58.build()
    X = sets[FEATURE_SET]
    cols = list(X.columns)
    agg = S43.load_profiles()
    prof = json.loads((F / os.environ.get("P2_PROFILE", "phase2_payload_profiles_v6.json")).read_text(encoding="utf-8"))["records"]
    by = {}
    for r in prof:
        by.setdefault(r["device_id"], []).append(r)
    role = pd.read_csv(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    bad, n = [], 0
    nb = len(S58.B.FEATS)
    for d in lab.index[lab.has_profile][:200]:
        a = agg[d]
        pv = payload_vector(a)
        ref_p = X.loc[d, cols[nb:nb + len(pv)]].to_numpy(float)
        if not np.allclose(pv, ref_p, atol=1e-9, equal_nan=True):
            bad.append(("payload", d))
        rv = role_vector(a, int(role.loc[d, "ot_syn_sent"]), int(role.loc[d, "ot_synack_sent"]))
        ref_r = X.loc[d, cols[nb + len(pv):nb + len(pv) + 4]].to_numpy(float)
        if not np.allclose(rv, ref_r, atol=1e-6):
            bad.append(("role", d, rv, ref_r.tolist()))
        tok, fam = Counter(), Counter()
        tx = rx = srv = 0
        for r in by[d]:
            tok.update(r["tokens"]); fam.update(r["fam_role"]); tx += r["tx_n"]; rx += r["rx_n"]; srv = max(srv, r["n_servers_contacted"])
        e = S57.feats(d, {d: tok}, {d: fam}, {d: tx}, {d: rx}, {d: srv})
        ev = [0.0 if pd.isna(e.get(k)) else float(e[k]) for k in S58.ENG]
        ref_e = X.loc[d, cols[-len(S58.ENG):]].to_numpy(float)
        if not np.allclose(ev, ref_e, atol=1e-9):
            bad.append(("eng", d, ev, ref_e.tolist()))
        n += 1
    print(f"selftest: {n} training devices compared (payload, role, eng vectors); mismatches {len(bad)}")
    for b in bad[:8]:
        print("  MISMATCH", b)
    print("SELFTEST", "OK" if not bad else "FAILED")
    return 0 if not bad else 1


# ------------------------------------------------------------------ inference
def analyze(path, force_all=False):
    t0 = time.time()
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(TOOLS / "pcap_to_dataset"))
    import pcap_reader  # noqa: E402
    import pipeline as P1  # noqa: E402  (loads Phase 1 model + feature builder)
    path = Path(path)
    rf = joblib.load(MODEL_DIR / "phase2_rf.joblib")
    meta = json.loads((MODEL_DIR / "phase2_meta.json").read_text(encoding="utf-8"))
    cols, classes, train_macs = meta["columns"], list(rf.classes_), set(meta["train_macs"])
    p1 = P1.analyze(path, "p2_" + path.stem[:24])
    df, _ = pcap_reader.read_pcap(path, P1.MAX_PACKETS)
    devs = {d["mac"].lower(): d for d in p1["devices"]}
    todo = {m: d["device_id"] for m, d in devs.items() if force_all or d["score"] >= PHASE1_THRESHOLD}
    recs, hs = profile_from_df(df, todo, path.stem)
    rows = []
    for mac, d in sorted(devs.items(), key=lambda kv: -kv[1]["score"]):
        dev = d["device_id"]
        row = {"device_id": dev, "mac": mac, "ips": d["ips"], "vendor_unverified": d["vendor"], "phase1_score": round(d["score"], 3), "phase1_ot": bool(d["score"] >= PHASE1_THRESHOLD),
               "in_training_phase1": d["in_training"], "in_training_phase2": mac in train_macs, "n_windows": d["n_windows"], "phase2": None}
        if dev not in todo.values():
            row["phase2_status"] = "skipped_phase1_non_OT"
        elif dev not in recs:
            row["phase2_status"] = "no_payload_packets (khong du bang chung)"
        else:
            rec = recs[dev]
            beh = [(-1.0 if d["features_median"].get(c) is None else float(d["features_median"][c])) for c in S58.B.FEATS]
            x = feature_row(dev, rec, hs[dev], beh, cols)
            pr = rf.predict_proba(x.to_frame().T)[0]
            order = np.argsort(-pr)
            n_tok = int(sum(rec["tokens"].values()))
            a = S43.aggregate_records([rec])[dev]
            role_v = role_vector(a, *hs[dev])
            role_txt = "unknown" if role_v[0] < 0 else ("server" if role_v[0] >= 0.8 else "client" if role_v[0] <= 0.2 else "mixed")
            sub = df[((df.src_mac == mac) | (df.dst_mac == mac)) & (df.payload_len.fillna(0) > 0)]
            ts = sub.timestamp[sub.timestamp > 0]
            span = float(ts.max() - ts.min()) if len(ts) else 0.0       # thoi gian thiet bi co payload trong capture nay (mo hinh huan luyen tren nhieu phut/session)
            evidence = "strong" if (n_tok >= 100 and span >= 45) else "weak(capture_ngan)" if n_tok > 0 else "no_OT_message_tokens"
            margin = float(pr[order[0]] - pr[order[1]])
            row.update(phase2={"pred": classes[order[0]], "prob": {c: round(float(p), 3) for c, p in zip(classes, pr)}, "second": classes[order[1]], "margin": round(margin, 3),
                               "uncertain": bool(margin < 0.10), "span_s": round(span, 3), "evidence": evidence, "n_payload_pkts": rec["n_pkts"], "n_message_tokens": n_tok, "ot_server_share": role_v[0] if role_v[0] >= 0 else None,
                               "role_port_rule": role_txt, "syn_sent_ot": hs[dev][0], "synack_sent_ot": hs[dev][1]}, phase2_status="predicted")
        rows.append(row)
    out = {"capture": {**p1["capture"], "file": path.name}, "phase1_threshold": PHASE1_THRESHOLD, "model": {"phase2": "RandomForest 4-class", "n_train": meta["n_train"], "class_counts": meta["class_counts"],
                                                                                                        "cv_reference": meta["cv_reference"], "caveat": meta["caveat"]},
           "notes": p1["notes"] + (["Phase 2 chi du doan cho thiet bi qua cong Phase 1 (diem >= 0,5)"] if not force_all else ["--all: Phase 2 du doan ca thiet bi khong qua cong Phase 1 (ngoai pham vi huan luyen)"]),
           "devices": rows, "elapsed_s": round(time.time() - t0, 1)}
    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / "output").mkdir(exist_ok=True)
    (OUTDIR / "output" / f"{path.stem}_phase2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def show(out):
    c = out["capture"]
    print(f"\nFile {c['file']}: {c['packets']:,} goi, {c['duration_s']} s, {c['unicast_macs']} MAC unicast | {out['elapsed_s']} s")
    for n in out["notes"]:
        print("  *", n)
    print(f"\n{'MAC':18} {'IP':16} {'P1':>5} {'P1?':4} {'Phase 2':8} {'conf':>5} {'2nd':6} {'bang chung':20} {'tok':>6} {'vaitro':7} {'trong train':11}")
    for r in out["devices"]:
        p = r["phase2"]
        ip = (r["ips"] or [""])[0]
        tr = "P1+P2" if r["in_training_phase1"] and r["in_training_phase2"] else "P1" if r["in_training_phase1"] else ""
        if p:
            print(f"{r['mac']:18} {ip:16} {r['phase1_score']:5.2f} {'OT' if r['phase1_ot'] else '-':4} {p['pred'] + ('?' if p['uncertain'] else ''):8} {p['prob'][p['pred']]:5.2f} {p['second']:6} {p['evidence']:20} {p['n_message_tokens']:6d} {p['role_port_rule']:7} {tr:11}")
        else:
            print(f"{r['mac']:18} {ip:16} {r['phase1_score']:5.2f} {'OT' if r['phase1_ot'] else '-':4} {r['phase2_status']:40} {tr}")
    print("\nDo chinh xac tham khao (cheo, 4 lop):", out["model"]["cv_reference"]["balanced_acc"], "| recall:", out["model"]["cv_reference"]["recall"], "| chance 0,25")


def main():
    if "--train" in sys.argv:
        return train()
    if "--selftest" in sys.argv:
        return selftest()
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    show(analyze(files[0], force_all="--all" in sys.argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
