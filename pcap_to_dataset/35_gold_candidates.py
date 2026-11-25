"""Select candidate devices for the independent gold set and write a BLIND annotation sheet.

Nothing is labelled here. The annotation sheet hides the rule label, evidence tier, flood flag and the model
score; they are written to a separate key file that must not be opened while annotating.
Stratified sample (seeded) so hard cases are included, not only easy ones.
Gold basis must be independent of the rule (scenario description, vendor/model knowledge, Wireshark function
inspection); `protocol_port_only` basis is allowed to be recorded but is excluded from independent evaluation.
"""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cmp", HERE / "34_compare_lightgbm_xgboost.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
F = L.A.F
SEED = 20261109
PLAN = [("protocol_nonflood", 35), ("protocol_flood", 10), ("peer_port_nonflood", 30), ("peer_port_flood", 8),
        ("local_port_only_nonflood", 30), ("local_port_only_flood", 8), ("unknown_high_score_nonflood", 30),
        ("unknown_flood_like", 10), ("unknown_random_nonflood", 59)]
RX = re.compile(r"(tcp|udp):local=(\d+),peer=(\d+)")


def stratum(r):
    t, fl = r.evidence_tier, r.flood_like
    if r.label == "OT_confirmed":
        return f"{t}_{'flood' if fl else 'nonflood'}".replace("local_port_only_", "local_port_only_")
    return "unknown_flood_like" if fl else "unknown_rest_nonflood"


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet")
    dev = pd.read_csv(F / "device_folds_v6.csv", dtype={"evidence_tier": str}, keep_default_na=False)
    dev["flood_like"] = dev.flood_like.astype(str).eq("True")
    x["evidence_tier"] = x.evidence_tier.fillna("")
    dev["evidence_tier"] = dev.evidence_tier.fillna("")
    _, score = L.run("lightgbm", x, dev, False)
    dev["oof_score"] = score
    dev["stratum"] = dev.apply(stratum, axis=1)
    rng = np.random.default_rng(SEED)
    chosen = []
    unk = dev[dev.stratum == "unknown_rest_nonflood"]
    pools = {name: dev[dev.stratum == name] for name in dict(PLAN) if name in set(dev.stratum)}
    pools["unknown_high_score_nonflood"] = unk.sort_values("oof_score", ascending=False).head(150)
    rest = unk[~unk.device_id.isin(pools["unknown_high_score_nonflood"].device_id)]
    pools["unknown_random_nonflood"] = rest
    used = set()
    for name, n in PLAN:
        pool = pools.get(name, dev.iloc[0:0])
        pool = pool[~pool.device_id.isin(used)]
        take = pool.iloc[rng.permutation(len(pool))[:n]]
        used |= set(take.device_id)
        chosen.append(take.assign(plan_stratum=name))
    g = pd.concat(chosen).reset_index(drop=True)
    ids = set(g.device_id)
    # evidence from per-capture review caches (only the selected devices)
    cache = next((F / "v6_behavior_build/captures").glob("*/"))
    info = {d: {"sessions": Counter(), "vendor": "", "mac": "", "protocols": set(), "peer_ports": Counter(), "refs": [], "files": []} for d in ids}
    cols = ["device_id", "session_id", "capture_file", "device_mac", "vendor", "protocols_seen", "service_ports", "evidence_packet_refs", "packet_count"]
    for f in sorted(cache.glob("*.review.parquet")):
        t = pq.read_table(f, columns=cols).to_pandas()
        t = t[t.device_id.isin(ids)]
        for r in t.itertuples(index=False):
            i = info[r.device_id]
            i["sessions"][r.session_id] += int(r.packet_count)
            i["vendor"], i["mac"] = r.vendor, r.device_mac
            i["protocols"].update(json.loads(r.protocols_seen))
            if len(i["refs"]) < 3:
                i["refs"] += json.loads(r.evidence_packet_refs)[:3 - len(i["refs"])]
            if len(r.service_ports) < 3_000_000:
                for _, _, peer in RX.findall(r.service_ports):
                    i["peer_ports"][int(peer)] += 1
    beh = x.groupby("device_id").agg(median_pkts_per_flow=("pkts_per_flow", "median"), median_packet_rate=("packet_rate", "median"),
                                      median_udp_ratio=("udp_ratio", "median"), max_svc_port_unique=("svc_port_unique", "max"),
                                      max_unique_local_ports=("unique_local_port_count", "max")).round(2)
    rows, key = [], []
    for r in g.itertuples(index=False):
        i = info[r.device_id]
        top = i["sessions"].most_common(3)
        rows.append({"device_id": r.device_id, "device_mac": i["mac"], "vendor_oui_unverified": i["vendor"], "n_sessions": r.n_sessions, "n_windows": r.n_windows,
                     "top_sessions": " | ".join(f"{s} ({n} pkts)" for s, n in top), "raw_packet_idx_examples": ";".join(i["refs"]),
                     **beh.loc[r.device_id].to_dict(),
                     "protocols_seen_all(basis of rule - use with care)": ";".join(sorted(i["protocols"])),
                     "top_peer_ports(basis of rule - use with care)": ";".join(f"{p}x{n}" for p, n in i["peer_ports"].most_common(8)),
                     "gold_label": "", "gold_basis": "", "gold_confidence": "", "notes": ""})
        key.append({"device_id": r.device_id, "plan_stratum": r.plan_stratum, "rule_label": r.label, "evidence_tier": r.evidence_tier,
                    "flood_like": r.flood_like, "oof_lightgbm_score": round(float(r.oof_score), 4), "fold_device": r.fold_device,
                    "primary_source_family": r.primary_source_family})
    sheet = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffled: order carries no information
    sheet.to_csv(F / "gold_candidates_v6_blind.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(F / "gold_candidates_v6_key_DO_NOT_OPEN_WHILE_ANNOTATING.csv", index=False, encoding="utf-8-sig")
    print(len(sheet), "devices"); print(pd.DataFrame(key).plan_stratum.value_counts().to_string())
    print(pd.DataFrame(key).primary_source_family.value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
