"""Phase 2: role-based comparison with the lab reference devices (PLC / HMI / IO) -> coarse pseudo-labels with named analog seeds.

Why role: on the 22 seeds the share of OT-protocol payload traffic in which the device is the SERVER separates PLC from HMI
(AUC 0.90), while a blended cosine over protocol/message-type/size vectors (script 43) does not (LOO 1-NN = majority baseline).
Rule (coarse, interpretable):
  OT payload packets < MIN_OT            -> 'no_ot_payload'
  server share >= HI (0.8)               -> 'responder_like'  (PLC / IED / RTU / IO / gateway / simulator ... answers OT requests)
  server share <= LO (0.2)               -> 'initiator_like'  (HMI / SCADA / engineering WS / master / scanner ... sends OT requests)
  otherwise                              -> 'mixed'
Seed analogs: within the same dominant OT protocol family and role, the seed with the highest message-type cosine; a seed type is named
only when such a seed exists (families without a seed, e.g. IEC 104, DNP3, BACnet, MMS, get 'no_seed_analog').
Outputs: phase2_role_labels_v6.csv, phase2_role_validation_v6.json.  Pseudo-labels, not gold.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s43", HERE / "43_phase2_similarity_labels.py")
S43 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S43)
F = S43.F
MIN_OT, HI, LO = 30, 0.8, 0.2
NON_OT = ("it_service", "unknown")


def role_features(a):
    ot = {k: v for k, v in a["fam_role"].items() if not k.startswith(NON_OT)}
    n = sum(ot.values())
    srv = sum(v for k, v in ot.items() if k.endswith("|S"))
    fam = Counter()
    for k, v in ot.items():
        fam[k.split("|")[0]] += v
    return n, (srv / n if n else np.nan), (fam.most_common(1)[0][0] if fam else "")


def role_class(n, share):
    if n < MIN_OT:
        return "no_ot_payload"
    return "responder_like" if share >= HI else ("initiator_like" if share <= LO else "mixed")


def token_vec(a, toks):
    v = np.array([a["tokens"].get(t, 0) for t in toks], float)
    t = v.sum()
    v = np.sqrt(v / t) if t else v
    return v / max(np.linalg.norm(v), 1e-9)


def main():
    agg = S43.load_profiles()
    lab = pd.read_csv(F / "gold_lab_inventory_v6.csv", encoding="utf-8-sig")
    lab = lab[lab.gold_label == "OT"].set_index("device_id_v6")
    toks = [t for t, _ in Counter(t for a in agg.values() for t in a["tokens"]).most_common(120)]
    rows = []
    for d, a in agg.items():
        n, share, fam = role_features(a)
        rows.append({"device_id": d, "ot_pkts": n, "ot_server_share": round(share, 3) if n else np.nan, "dominant_ot_family": fam,
                     "role_class": role_class(n, share), "n_sessions": a["n_sessions"]})
    df = pd.DataFrame(rows).set_index("device_id")
    seeds = df.loc[[d for d in lab.index if d in df.index]].join(lab[["device_type", "ip", "model"]])
    # --- seed validation: role rule vs lab type
    s = seeds[seeds.device_type.isin(["PLC", "HMI"]) & (seeds.ot_pkts >= MIN_OT)]
    y = (s.device_type == "PLC")
    val = {"seeds_with_ot_payload": int(len(s)), "auc_server_share_PLC_vs_HMI": round(float(roc_auc_score(y, s.ot_server_share)), 3),
           "rule_hi_lo_table": pd.crosstab(s.device_type, s.role_class).to_dict(),
           "all_seed_rows": seeds[["device_type", "ip", "ot_pkts", "ot_server_share", "dominant_ot_family", "role_class"]].reset_index().to_dict("records")}
    sw = seeds[seeds.device_type.isin(["PLC", "HMI"]) & (seeds.ot_pkts >= MIN_OT)]
    pred = np.where(sw.ot_server_share >= 0.5, "PLC", "HMI")
    val["rule_0.5_accuracy_PLC_vs_HMI"] = round(float((pred == sw.device_type).mean()), 3)
    val["rule_0.5_errors"] = sw[pred != sw.device_type][["device_type", "ip", "model", "ot_server_share"]].reset_index().to_dict("records")
    # --- analog seeds: nearest seed in same (dominant family, role) by message-type cosine
    seed_vec = {d: token_vec(agg[d], toks) for d in seeds.index}
    out = []
    for d, r in df.iterrows():
        analog, sim, a_type, label = "", np.nan, "", r.role_class
        if r.role_class in ("responder_like", "initiator_like"):
            cands = [sd for sd in seeds.index if sd != d and seeds.loc[sd, "dominant_ot_family"] == r.dominant_ot_family and seeds.loc[sd, "role_class"] == r.role_class]
            if cands:
                v = token_vec(agg[d], toks)
                sims = {sd: float(v @ seed_vec[sd]) for sd in cands}
                analog = max(sims, key=sims.get)
                sim = round(sims[analog], 3)
                a_type = seeds.loc[analog, "device_type"]
                label = f"{r.role_class}|{a_type}_analog"
            else:
                label = f"{r.role_class}|no_seed_analog"
        out.append({"device_id": d, "analog_seed": analog, "analog_type": a_type, "analog_token_cosine": sim, "pseudo_label": label})
    res_df = df.join(pd.DataFrame(out).set_index("device_id"))
    res_df["is_seed"] = res_df.index.isin(seeds.index)
    # coarse, honest label: role only (type-level PLC/IED/IO cannot be told apart by role; see validation)
    res_df["coarse_label"] = res_df.role_class.map({"responder_like": "field_device_like(answers OT requests)", "initiator_like": "supervisory_or_engineering_like(sends OT requests)",
                                                    "mixed": "mixed_role", "no_ot_payload": "no_ot_payload(cannot judge)"})
    res_df["evidence_strength"] = np.where(res_df.ot_pkts >= 300, "strong", np.where(res_df.ot_pkts >= MIN_OT, "weak", "none"))
    res_df.reset_index().to_csv(F / "phase2_role_labels_v6.csv", index=False, encoding="utf-8-sig")
    # --- independent check against sonthai-reviewed sheet
    ann = pd.read_csv(F / "phase2_type_labeling_sheet_v6_reviewed_sonthai.csv", encoding="utf-8-sig", dtype=str).fillna("")
    ann = ann[ann.gold_confidence.isin(["high", "medium"])].set_index("device_id").gold_type
    ns = res_df[~res_df.is_seed]
    chk = {}
    for name in ann.unique():
        ids = [i for i in ann[ann == name].index if i in ns.index]
        sub = ns.loc[ids]
        chk[name] = {"n_profiled": len(ids), "role_class_counts": sub.role_class.value_counts().to_dict(), "pseudo_label_counts": sub.pseudo_label.value_counts().to_dict()}
    val["vs_sonthai_review"] = chk
    val["pool_non_seed_role_class"] = ns.role_class.value_counts().to_dict()
    val["pool_non_seed_pseudo_label"] = ns.pseudo_label.value_counts().to_dict()
    val["thresholds"] = {"MIN_OT": MIN_OT, "HI": HI, "LO": LO}
    (F / "phase2_role_validation_v6.json").write_text(json.dumps(val, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in val.items() if k != "all_seed_rows"}, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
