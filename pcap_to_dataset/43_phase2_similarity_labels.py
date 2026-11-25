"""Phase 2: similarity of payload communication profiles to the lab reference devices (PLC / HMI / IO) -> pseudo-labels.

Seeds: gold_lab_inventory_v6.csv OT rows with a type (owner-provided table; MAC verified in packets).
Vector (cosine): block A = protocol-family x role share (sqrt), block B = message-type token share (sqrt, top-K tokens, device perspective
S/C + tx/rx), block C = payload size / direction / peer counts (z-scored).  Blocks are L2-normalised and weighted (WA, WB, WC).
Validation: leave-one-seed-out nearest-seed class accuracy (PLC vs HMI), then independent checks against sonthai-reviewed Phase-2 labels
(PLC annotated devices should resemble PLC seeds; IED/other devices should resemble no seed).
Outputs: phase2_similarity_scores_v6.csv (all profiled devices), phase2_similarity_validation_v6.json.
Pseudo-labels are NOT gold: they say 'behaves like seed X', nothing more.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F = HERE.parent / "Dataset/features"
WA, WB, WC, TOPK = 1.0, 1.0, 0.5, 80
FAMS = ["s7_iso", "modbus", "enip", "bacnet", "dnp3", "iec104", "profinet_ethercat", "omron_fins", "fox_niagara", "opcua", "mqtt", "codesys_ge_misc", "it_service", "unknown"]


def load_profiles():
    d = json.loads((F / os.environ.get("P2_PROFILE", "phase2_payload_profiles_v6.json")).read_text(encoding="utf-8"))
    return aggregate_records(d["records"])


def aggregate_records(records):
    """records: list of per device-session profile dicts -> per-device aggregate (used by load_profiles and by the PCAP inference script 69)."""
    per = defaultdict(list)
    for r in records:
        per[r["device_id"]].append(r)
    agg = {}
    for dev, rs in per.items():
        fr, tk = Counter(), Counter()
        for r in rs:
            fr.update(r["fam_role"])
            tk.update(r["tokens"])
        def wmean(key, wkey):
            v = [(r[key], r[wkey]) for r in rs if r[key] is not None and r[wkey]]
            return float(np.average([a for a, _ in v], weights=[w for _, w in v])) if v else np.nan
        tx_n, rx_n = sum(r["tx_n"] for r in rs), sum(r["rx_n"] for r in rs)
        agg[dev] = {"fam_role": fr, "tokens": tk, "n_pkts": sum(r["n_pkts"] for r in rs), "n_sessions": len(rs),
                    "tx_med": wmean("tx_len_med", "tx_n"), "tx_p90": wmean("tx_len_p90", "tx_n"), "rx_med": wmean("rx_len_med", "rx_n"),
                    "rx_p90": wmean("rx_len_p90", "rx_n"), "tx_share": tx_n / max(tx_n + rx_n, 1),
                    "n_clients": max(r["n_clients_of_device"] for r in rs), "n_servers": max(r["n_servers_contacted"] for r in rs)}
    return agg


def build_matrix(agg, min_pkts=30):
    devs = [d for d, a in agg.items() if a["n_pkts"] >= min_pkts]
    cells = [f"{f}|{r}" for f in FAMS for r in ("S", "C")]
    A = np.array([[agg[d]["fam_role"].get(c, 0) for c in cells] for d in devs], float)
    A = np.sqrt(A / np.maximum(A.sum(1, keepdims=True), 1))
    df_tok = Counter(t for d in devs for t in agg[d]["tokens"])
    toks = [t for t, _ in df_tok.most_common(TOPK)]
    Bm = np.array([[agg[d]["tokens"].get(t, 0) for t in toks] for d in devs], float)
    tot = np.array([sum(agg[d]["tokens"].values()) for d in devs], float)[:, None]
    Bm = np.sqrt(Bm / np.maximum(tot, 1))
    C = np.array([[np.log1p(agg[d]["tx_med"]) if not np.isnan(agg[d]["tx_med"]) else 0, np.log1p(agg[d]["tx_p90"]) if not np.isnan(agg[d]["tx_p90"]) else 0,
                   np.log1p(agg[d]["rx_med"]) if not np.isnan(agg[d]["rx_med"]) else 0, np.log1p(agg[d]["rx_p90"]) if not np.isnan(agg[d]["rx_p90"]) else 0,
                   agg[d]["tx_share"], np.log1p(agg[d]["n_clients"]), np.log1p(agg[d]["n_servers"])] for d in devs], float)
    C = (C - C.mean(0)) / np.maximum(C.std(0), 1e-6)
    norm = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    V = np.hstack([WA * norm(A), WB * norm(Bm), WC * norm(C)])
    V = norm(V)
    return devs, V, toks


def main():
    agg = load_profiles()
    devs, V, toks = build_matrix(agg)
    idx = {d: i for i, d in enumerate(devs)}
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"][["device_id_v6", "device_type", "ip", "model"]].rename(columns={"device_id_v6": "device_id"})
    seeds = lab[lab.device_id.isin(idx)].copy()
    S = V[[idx[d] for d in seeds.device_id]]
    cls = seeds.device_type.to_numpy()
    sim = V @ S.T  # all devices x seeds
    # --- leave-one-seed-out validation
    loo = []
    for j, d in enumerate(seeds.device_id):
        s = sim[idx[d]].copy()
        s[j] = -9
        k = int(np.argmax(s))
        per_cls = {c: float(np.max(np.where(cls == c, s, -9))) for c in set(cls) if np.any((cls == c) & (np.arange(len(cls)) != j))}
        loo.append({"device_id": d, "true": cls[j], "nearest_seed": seeds.device_id.iloc[k], "pred_1nn": cls[k], "sim_1nn": round(float(s[k]), 3),
                    "ok": bool(cls[j] == cls[k]), **{f"max_sim_{c}": round(v, 3) for c, v in per_cls.items()}})
    L = pd.DataFrame(loo)
    ph = L[L.true.isin(["PLC", "HMI"])]
    base = ph.true.value_counts().max() / len(ph)
    res = {"n_seeds_profiled": int(len(seeds)), "n_seeds_total": 22, "seed_types": seeds.device_type.value_counts().to_dict(),
           "loo_1nn_accuracy_all": round(float(L.ok.mean()), 3), "loo_1nn_accuracy_PLC_vs_HMI_rows": round(float(ph.ok.mean()), 3),
           "majority_baseline_PLC_vs_HMI": round(float(base), 3), "loo_detail": loo,
           "loo_mean_sim_correct": round(float(L[L.ok].sim_1nn.mean()), 3) if L.ok.any() else None,
           "loo_mean_sim_wrong": round(float(L[~L.ok].sim_1nn.mean()), 3) if (~L.ok).any() else None}
    # thresholds from LOO: accept a match if sim >= tau; tau_hi = median of correct 1-NN sims, tau_lo = min of correct
    tau_hi = float(L[L.ok].sim_1nn.median()) if L.ok.any() else 0.9
    tau_lo = float(L[L.ok].sim_1nn.min()) if L.ok.any() else 0.8
    res["tau_hi"], res["tau_lo"] = round(tau_hi, 3), round(tau_lo, 3)
    # --- scores for all devices (exclude self-match for seeds)
    rows = []
    for d in devs:
        s = sim[idx[d]].copy()
        if d in set(seeds.device_id):
            s[list(seeds.device_id).index(d)] = -9
        per = {c: float(np.max(np.where(cls == c, s, -9))) for c in sorted(set(cls))}
        best = max(per, key=per.get)
        others = [v for c, v in per.items() if c != best]
        margin = per[best] - (max(others) if others else -1)
        nn = seeds.device_id.iloc[int(np.argmax(s))]
        tier = "high" if per[best] >= tau_hi and margin >= 0.02 else ("medium" if per[best] >= tau_lo and margin >= 0.0 else "none")
        rows.append({"device_id": d, "n_pkts": agg[d]["n_pkts"], "n_sessions": agg[d]["n_sessions"], **{f"sim_{c}": round(v, 3) for c, v in per.items()},
                     "best_class": best, "margin": round(margin, 3), "nearest_seed": nn, "tier": tier,
                     "is_seed": d in set(seeds.device_id), "dominant_family_role": agg[d]["fam_role"].most_common(1)[0][0] if agg[d]["fam_role"] else ""})
    out = pd.DataFrame(rows)
    out["pseudo_label"] = np.where(out.tier == "none", "no_close_seed", out.best_class)
    out.to_csv(F / "phase2_similarity_scores_v6.csv", index=False, encoding="utf-8-sig")
    # --- independent checks vs sonthai-reviewed sheet
    ann = pd.read_csv(F / "phase2_type_labeling_sheet_v6_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_confidence.isin(["high", "medium"])].set_index("device_id").gold_type
    o = out[~out.is_seed].set_index("device_id")
    chk = {}
    for name, grp in ann.groupby(ann):
        ids = [i for i in grp.index if i in o.index]
        if ids:
            sub = o.loc[ids]
            chk[name] = {"n_profiled": len(ids), "median_best_sim": round(float(sub[[c for c in sub.columns if c.startswith("sim_")]].max(axis=1).median()), 3),
                         "pct_with_close_seed(tier!=none)": round(float((sub.tier != "none").mean()), 3),
                         "pseudo_label_counts": sub.pseudo_label.value_counts().to_dict()}
    res["vs_sonthai_review"] = chk
    res["pool_pseudo_label_counts_non_seed"] = o.pseudo_label.value_counts().to_dict()
    res["pool_tier_counts_non_seed"] = o.tier.value_counts().to_dict()
    (F / "phase2_similarity_validation_v6.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(L.to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k != "loo_detail"}, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
