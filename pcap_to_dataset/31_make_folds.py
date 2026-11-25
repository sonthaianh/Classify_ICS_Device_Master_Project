"""Device-level fold table for Phase 1 (no training, no data modified).

fold_device    : 5 folds, one fold per device, stratified by label x evidence_tier (main result).
fold_component : 5 folds, devices linked through shared sessions stay together (strict check).
                 The giant component cannot be split, so it is its own fold; the rest are balanced.
primary_source_family : family (draft, unreviewed) with most windows of the device; use for
                 leave-one-source-out. n_source_families shows devices that span several.
Windows are never split: every window of a device inherits that device's fold.
"""
from pathlib import Path
import collections

import numpy as np
import pandas as pd

F = Path(__file__).resolve().parent.parent / "Dataset/features"
SEED, NF = 20261108, 5


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "label", "evidence_tier", "flood_like"])
    src = pd.read_csv(F / "session_source_map_draft.csv", usecols=["session_id", "source_family"])
    x = x.merge(src, on="session_id", how="left")
    dev = x.groupby("device_id").agg(label=("label", "first"), evidence_tier=("evidence_tier", "first"), flood_like=("flood_like", "first"),
                                     n_windows=("session_id", "size"), n_sessions=("session_id", "nunique")).reset_index()
    # primary family = most windows; ties broken by name for determinism
    fam = x.groupby(["device_id", "source_family"]).size().rename("w").reset_index().sort_values(["device_id", "w", "source_family"], ascending=[True, False, True])
    dev = dev.merge(fam.drop_duplicates("device_id")[["device_id", "source_family"]].rename(columns={"source_family": "primary_source_family"}), on="device_id")
    dev["n_source_families"] = dev.device_id.map(x.groupby("device_id").source_family.nunique())
    # connected components over shared sessions
    par = {}

    def find(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for d, s in x.drop_duplicates(["device_id", "session_id"])[["device_id", "session_id"]].itertuples(index=False):
        par[find("D" + d)] = find("S" + s)
    root = {d: find("D" + d) for d in dev.device_id}
    ids = {r: i for i, r in enumerate(sorted(set(root.values())))}
    dev["component_id"] = dev.device_id.map(lambda d: ids[root[d]])
    rng = np.random.default_rng(SEED)
    # --- fold_device: stratified round-robin inside each label x tier stratum
    dev["fold_device"] = -1
    for _, g in dev.groupby(["label", "evidence_tier"]):
        idx = rng.permutation(g.index.to_numpy())
        dev.loc[idx, "fold_device"] = np.arange(len(idx)) % NF
    # --- fold_component: largest component alone, others greedy by OT count then size
    comp = dev.groupby("component_id").agg(n=("device_id", "size"), ot=("label", lambda s: int((s == "OT_confirmed").sum()))).reset_index().sort_values("n", ascending=False)
    load, assign = np.zeros((NF, 2)), {}
    first = comp.iloc[0].component_id
    assign[first] = 0
    load[0] += comp.iloc[0][["n", "ot"]].to_numpy(float)
    rest = comp.iloc[1:].assign(_r=rng.random(len(comp) - 1)).sort_values(["n", "_r"], ascending=[False, True])
    for c, n, ot in zip(rest.component_id, rest.n, rest.ot):
        f = 1 + int(np.argmin(load[1:, 0]))
        assign[c] = f
        load[f] += (n, ot)
    dev["fold_component"] = dev.component_id.map(assign)
    out = F / "device_folds_v6.csv"
    dev.to_csv(out, index=False, encoding="utf-8-sig")
    # ---- checks
    assert dev.device_id.is_unique and len(dev) == 3923 and (dev.fold_device >= 0).all()
    pairs = x.drop_duplicates(["device_id", "session_id"]).merge(dev[["device_id", "fold_component", "fold_device"]], on="device_id")
    assert pairs.groupby("component_id" if "component_id" in pairs else "session_id").fold_component.nunique().max() == 1
    print("devices", len(dev), "| components", dev.component_id.nunique())
    for col in ("fold_device", "fold_component"):
        t = dev.groupby(col).agg(devices=("device_id", "size"), OT=("label", lambda s: int((s == "OT_confirmed").sum())), windows=("n_windows", "sum"))
        t["OT%"] = (t.OT / t.devices * 100).round(1)
        print(col); print(t.to_string())
    s = pairs.groupby("session_id").fold_device.nunique()
    print("sessions whose devices span >1 fold_device:", int((s > 1).sum()), "of", len(s), "(expected: many; level-1 does not separate sessions)")
    print("OT tier counts per fold_device:"); print(pd.crosstab(dev.fold_device, dev.evidence_tier.replace("", "unknown")).to_string())


if __name__ == "__main__":
    main()
