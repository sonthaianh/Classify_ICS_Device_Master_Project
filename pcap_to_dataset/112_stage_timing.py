"""Nhom A4 cua so theo doi (2026-11-25): chi phi cua tung giai doan. KHONG huan luyen lai, KHONG ghi de ket qua dong bang.

(1) Khoi luong phai doc payload: so tep va dung luong (tep raw_parquet) khi CO cong loc giai doan 1 (chi thiet bi co diem >= 0,5 va cac thiet bi OT cua bang phong thi nghiem,
    dung pham vi cua script 67) va khi KHONG co cong loc (moi thiet bi cua giai doan 1).
(2) Thoi gian suy luan mo hinh tren du lieu da trich dac trung: LightGBM giai doan 1 tren moi cua so cua 3.923 thiet bi; Random Forest giai doan 2 tren vector 64 cot.
(3) Thoi gian tung buoc cua nguyen mau tren cac tep PCAP mau (Tools/Dataset/pcap): doc PCAP; giai doan 1 (dung cua so 45 giay + LightGBM); giai doan 2 (doc payload cua
    thiet bi qua cong, dung vector 64 cot, Random Forest). Moi tep lap 3 lan, lay trung vi. Them truong hop khong co cong loc (doc payload cua moi thiet bi).
Dau ra: stage_timing_a4.json.    Chay: py -3.14 112_stage_timing.py [--skip-pcap]
"""
import importlib.util
import json
import os
import platform
import sys
import time
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
F = TOOLS / "Dataset/features"
DS = TOOLS / "Dataset"
APP = TOOLS / "phase1_webapp"
THR = 0.5
REP = 3
GB = 1024 ** 3


def volume():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id"])
    pairs = x.drop_duplicates()
    nwin = x.groupby("device_id").size()
    p1 = pd.read_csv(F / "phase1_oof_scores_v6.csv", encoding="utf-8-sig").set_index("device_id").lightgbm
    pool = set(pd.read_csv(F / "phase2_candidate_pool_v6.csv", encoding="utf-8-sig", dtype=str).device_id)
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    scope = pool | set(lab[lab.gold_label == "OT"].device_id_v6)
    passed = set(p1.index[p1 >= THR])
    size = {}
    for s in pairs.session_id.unique():
        p = DS / "raw_parquet" / f"{s}.parquet"
        size[s] = p.stat().st_size if p.exists() else np.nan
    all_files = sorted(p.stat().st_size for p in (DS / "raw_parquet").glob("*.parquet"))

    def blk(ids):
        pr = pairs[pairs.device_id.isin(ids)]
        ss = pr.session_id.unique()
        return {"devices": int(len(set(ids) & set(pairs.device_id))), "device_capture_pairs": int(len(pr)), "files": int(len(ss)),
                "raw_parquet_GB": round(float(np.nansum([size[s] for s in ss])) / GB, 1), "raw_parquet_bytes": int(np.nansum([size[s] for s in ss])), "windows": int(nwin.reindex(list(ids)).fillna(0).sum())}
    out = {"all_raw_parquet_files": {"files": len(all_files), "GB": round(sum(all_files) / GB, 1)},
           "without_stage1_gate_all_stage1_devices": blk(set(p1.index)),
           "with_stage1_gate_score_ge_0.5": blk(passed),
           "with_stage1_gate_as_used_script67_pool_plus_lab_OT": blk(scope),
           "n_devices_stage1": int(len(p1)), "n_devices_score_ge_0.5": int(len(passed)), "missing_raw_files": int(sum(np.isnan(v) for v in size.values()))}
    a, b = out["without_stage1_gate_all_stage1_devices"], out["with_stage1_gate_as_used_script67_pool_plus_lab_OT"]
    out["reduction_with_gate"] = {"devices": round(1 - b["devices"] / a["devices"], 3), "files": round(1 - b["files"] / a["files"], 3), "GB": round(1 - b["raw_parquet_GB"] / a["raw_parquet_GB"], 3),
                                  "device_capture_pairs": round(1 - b["device_capture_pairs"] / a["device_capture_pairs"], 3)}
    return out


def med_time(fn, rep=REP):
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), r


def model_inference():
    model = joblib.load(APP / "model/phase1_lightgbm.joblib")
    meta = json.loads((APP / "model/model_meta.json").read_text(encoding="utf-8"))
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id"] + meta["features"])
    t1, p = med_time(lambda: model.predict_proba(x[meta["features"]])[:, 1])
    t1g, _ = med_time(lambda: pd.Series(p, index=x.index).groupby(x.device_id).mean())
    ndev = x.device_id.nunique()
    one = x[x.device_id == x.device_id.iloc[0]][meta["features"]].head(20)
    t1s, _ = med_time(lambda: model.predict_proba(one), rep=50)
    rf = joblib.load(TOOLS / "phase2_infer/model_186/phase2_rf.joblib")
    m2 = json.loads((TOOLS / "phase2_infer/model_186/phase2_meta.json").read_text(encoding="utf-8"))
    X = pd.DataFrame(np.random.default_rng(0).random((1123, len(m2["columns"]))), columns=m2["columns"])
    t2, _ = med_time(lambda: rf.predict_proba(X))
    t2s, _ = med_time(lambda: rf.predict_proba(X.iloc[[0]]), rep=50)
    return {"stage1_lightgbm": {"n_devices": int(ndev), "n_windows": int(len(x)), "n_features": len(meta["features"]), "batch_all_windows_s": round(t1 + t1g, 3),
                                "ms_per_device_batch": round(1000 * (t1 + t1g) / ndev, 3), "ms_per_window_batch": round(1000 * t1 / len(x), 5), "ms_single_device_20_windows": round(1000 * t1s, 2)},
            "stage2_random_forest": {"n_trees": int(rf.n_estimators), "n_features": len(m2["columns"]), "n_devices_batch": 1123, "batch_s": round(t2, 3), "ms_per_device_batch": round(1000 * t2 / 1123, 3),
                                     "ms_single_device": round(1000 * t2s, 2)}}


def prototype():
    spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.CUR["path"] = str(M.F / "phase2_type_labels_4class_186_devices.csv")
    src = (HERE / "69_phase2_infer.py").read_text(encoding="utf-8").replace('MODEL_DIR = OUTDIR / "model"', 'MODEL_DIR = OUTDIR / "model_186"')
    P69 = types.ModuleType("p69_186")
    P69.__file__ = str(HERE / "69_phase2_infer.py")
    exec(compile(src, str(HERE / "69_phase2_infer.py"), "exec"), P69.__dict__)
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(HERE))
    import pcap_reader
    import pipeline as P1
    rf = joblib.load(P69.MODEL_DIR / "phase2_rf.joblib")
    cols = json.loads((P69.MODEL_DIR / "phase2_meta.json").read_text(encoding="utf-8"))["columns"]
    FE = P69.S58.B.FEATS
    rows = []
    for path in sorted((DS / "pcap").glob("*.pcap")):
        rec = {"file": path.name, "size_MB": round(path.stat().st_size / 1e6, 2), "runs": []}
        for rep in range(REP):
            t = time.perf_counter()
            df, info = pcap_reader.read_pcap(path, P1.MAX_PACKETS)
            t_read = time.perf_counter() - t
            # ---- giai doan 1: dung cua so 45 giay + LightGBM (cac buoc cua pipeline.analyze)
            t = time.perf_counter()
            macs = pd.concat([df.src_mac, df.dst_mac])
            counts = macs[macs.map(P1._is_unicast)].value_counts().head(P1.MAX_DEVICES)
            registry = pd.DataFrame({"device_id": [f"D{i + 1:04d}" for i in range(len(counts))], "mac": counts.index, "vendor": [P1._oui_vendor(m) for m in counts.index],
                                     "ground_truth_family": "", "label_source": "", "protocols_seen": ""}).set_index("device_id", drop=False)
            mac_to_dev = dict(zip(registry.mac, registry.device_id))
            job = P1.WORK / "a4_timing"
            job.mkdir(parents=True, exist_ok=True)
            raw = job / "a4_timing.parquet"
            df.to_parquet(raw, index=False)
            tshark, fields = P1._tshark()
            P1.B.PORTS_CACHE = P1.B.port_catalog()
            windows, review, audit = P1.B.build_capture(raw, mac_to_dev, registry, P1.B.PORTS_CACHE, tshark, fields, sample_cap=500)
            windows = windows[windows.device_id.isin(windows.loc[windows.has_flow_packets, "device_id"].unique())].copy()
            t_feat = time.perf_counter() - t
            t = time.perf_counter()
            windows["prob"] = P1.MODEL.predict_proba(windows[P1.FEATURES])[:, 1]
            score = windows.groupby("device_id").prob.mean()
            t_lgbm = time.perf_counter() - t
            dev_mac = {d: m.lower() for m, d in mac_to_dev.items()}
            beh = windows.groupby("device_id")[FE].median()

            def stage2(devs):
                t_ = time.perf_counter()
                todo = {dev_mac[d]: d for d in devs}
                recs, hs = P69.profile_from_df(df, todo, path.stem)
                tp = time.perf_counter() - t_
                t_ = time.perf_counter()
                n = 0
                for d, r in recs.items():
                    b = [(-1.0 if pd.isna(v) else float(v)) for v in beh.loc[d]]
                    xr = P69.feature_row(d, r, hs[d], b, cols)
                    rf.predict_proba(xr.to_frame().T)
                    n += 1
                return tp, time.perf_counter() - t_, n
            passed = list(score.index[score >= THR])
            tp, tr, n2 = stage2(passed) if passed else (0.0, 0.0, 0)
            tp_all, tr_all, n2_all = stage2(list(score.index))
            rec["runs"].append({"t_read_pcap": t_read, "t_stage1_features": t_feat, "t_stage1_lightgbm": t_lgbm, "t_stage2_payload_profile": tp, "t_stage2_vector_and_rf": tr,
                                "t_stage2_nogate_payload_profile": tp_all, "t_stage2_nogate_vector_and_rf": tr_all})
            rec.update(packets=int(len(df)), devices_scored=int(len(score)), devices_passed_gate=int(len(passed)), devices_stage2_predicted=int(n2), devices_stage2_predicted_nogate=int(n2_all),
                       windows=int(len(windows)))
        md = {k: float(np.median([r[k] for r in rec["runs"]])) for k in rec["runs"][0]}
        rec["median_s"] = {k: round(v, 3) for k, v in md.items()}
        s1 = md["t_stage1_features"] + md["t_stage1_lightgbm"]
        s2 = md["t_stage2_payload_profile"] + md["t_stage2_vector_and_rf"]
        s2a = md["t_stage2_nogate_payload_profile"] + md["t_stage2_nogate_vector_and_rf"]
        rec["per_device_ms"] = {"read_pcap_per_scored_device": round(1000 * md["t_read_pcap"] / max(rec["devices_scored"], 1), 1), "stage1_per_scored_device": round(1000 * s1 / max(rec["devices_scored"], 1), 1),
                                "stage2_per_passed_device": round(1000 * s2 / max(rec["devices_passed_gate"], 1), 1) if rec["devices_passed_gate"] else None,
                                "stage2_nogate_per_scored_device": round(1000 * s2a / max(rec["devices_scored"], 1), 1)}
        rec["totals_s"] = {"read": round(md["t_read_pcap"], 3), "stage1": round(s1, 3), "stage2_with_gate": round(s2, 3), "stage2_without_gate": round(s2a, 3)}
        del rec["runs"]
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    tot = lambda k: float(sum(r["totals_s"][k] for r in rows))
    nd, npass = sum(r["devices_scored"] for r in rows), sum(r["devices_passed_gate"] for r in rows)
    summary = {"n_pcap": len(rows), "packets": int(sum(r["packets"] for r in rows)), "devices_scored": int(nd), "devices_passed_gate": int(npass),
               "ms_per_device_read_pcap": round(1000 * tot("read") / nd, 1), "ms_per_device_stage1": round(1000 * tot("stage1") / nd, 1),
               "ms_per_passed_device_stage2": round(1000 * tot("stage2_with_gate") / max(npass, 1), 1), "ms_per_device_stage2_without_gate": round(1000 * tot("stage2_without_gate") / nd, 1),
               "total_s": {k: round(tot(k), 2) for k in ("read", "stage1", "stage2_with_gate", "stage2_without_gate")}}
    return {"files": rows, "summary": summary}


def main():
    out = {"date": "2026-11-25", "machine": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": platform.python_version()},
           "note": "wall-clock, single process; model inference uses the library defaults for threads; offline payload extraction of the full data set (script 67, 6 worker processes) took 460 s for 1,473 files (log phase2_payload_full_run.log)"}
    if "--volume-only" in sys.argv:
        pth = F / "stage_timing_a4.json"
        old = json.loads(pth.read_text(encoding="utf-8"))
        old["payload_volume"] = volume()
        pth.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(old["payload_volume"], indent=1))
        return 0
    out["payload_volume"] = volume()
    print(json.dumps(out["payload_volume"], indent=1), flush=True)
    out["model_inference_on_extracted_features"] = model_inference()
    print(json.dumps(out["model_inference_on_extracted_features"], indent=1), flush=True)
    if "--skip-pcap" not in sys.argv:
        out["prototype_on_sample_pcap"] = prototype()
        print(json.dumps(out["prototype_on_sample_pcap"]["summary"], indent=1), flush=True)
    (F / "stage_timing_a4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-A4")


if __name__ == "__main__":
    sys.exit(main())
