"""Verification of the full identity extraction (script 52) and of the label table built from it (scripts 47/48).

Checks (each PASS/FAIL, exit code 1 on any FAIL):
  completeness : every expected capture has a cache file; no error; merged record count consistent
  superset     : every identity item found by the first pass (script 46) is still present (same device, session) in the full pass
  coverage     : every device-session pair of the scope is represented by a record
  labels       : pass-1 vs full-pass label table: unchanged / added / changed / lost, conflicts (two classes with vote >= 2), exclusions
  review list  : every NEW PLC/HMI label with its evidence (printed for manual review)
Compares phase2_type_labels_v6_pass1.csv (saved before the re-run) with phase2_type_labels_v6.csv.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s52", HERE / "52_phase2_identity_full.py")
S52 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S52)
F = S52.F
ok_all = True


def chk(cond, msg):
    global ok_all
    print(("PASS " if cond else "FAIL ") + msg)
    ok_all &= bool(cond)


def main():
    jobs, n_ids, n_pairs = S52.build_jobs()
    full = json.loads((F / "phase2_identity_raw_v6_full.json").read_text(encoding="utf-8"))
    chk(full["n_files_missing"] == 0 and full["n_files_expected"] == len(jobs), f"all {len(jobs)} captures have cache files (missing {full['n_files_missing']})")
    # per-file error / coverage / counters from the cache
    n_rec, pairs_seen, bad_cache, scanned0 = 0, set(), [], 0
    for s, targets in jobs:
        d = json.loads((S52.CACHE / f"{s}.json").read_text(encoding="utf-8"))
        if d.get("session") != s:
            bad_cache.append(s)
        for r in d["records"]:
            n_rec += 1
            pairs_seen.add((r["device_id"], r["session_id"]))
            if r["n_device_payload_pkts"] > 0 and r["n_ascii_scanned"] == 0:
                scanned0 += 1
    chk(not bad_cache, "cache files carry the right session id")
    chk(n_rec == n_pairs and len(pairs_seen) == n_pairs, f"one record per device-session pair ({n_rec} records, {n_pairs} pairs)")
    chk(scanned0 == 0, "every device-session with payload had its first packets scanned")
    # superset of first pass
    old = json.loads((F / "phase2_identity_raw_v6.json").read_text(encoding="utf-8"))["records"]
    new = {(r["device_id"], r["session_id"]): {(a, b, c) for a, b, c, n in r["items"]} for r in full["records"]}
    oldmap = {(r["device_id"], r["session_id"]): {(a, b, c) for a, b, c, n in r["items"]} for r in old}
    lost = [(k, list(v - new.get(k, set()))[:2]) for k, v in oldmap.items() if v - new.get(k, set())]
    chk(not lost, f"full pass contains every item of the first pass (lost {len(lost)})")
    if lost:
        print("   e.g.", lost[:3])
    n_old_items = sum(len(r["items"]) for r in old)
    n_new_items = sum(len(v) for v in new.values())
    print(f"INFO items: first pass {n_old_items} -> full pass {n_new_items}; devices with items {len({r['device_id'] for r in old})} -> {len({k[0] for k in new})}")
    # labels
    a = pd.read_csv(F / "phase2_type_labels_v6_pass1.csv", encoding="utf-8-sig").set_index("device_id")
    b = pd.read_csv(F / "phase2_type_labels_v6.csv", encoding="utf-8-sig").set_index("device_id")
    idt = pd.read_csv(F / "phase2_identity_types_v6.csv", encoding="utf-8-sig", dtype=str).fillna("").set_index("device_id")
    same = [d for d in a.index if d in b.index and a.type[d] == b.type[d]]
    changed = [d for d in a.index if d in b.index and a.type[d] != b.type[d]]
    lostl = [d for d in a.index if d not in b.index]
    added = [d for d in b.index if d not in a.index]
    print(f"INFO labels: pass1 {len(a)} -> full {len(b)} | unchanged {len(same)} | changed {len(changed)} | lost {len(lostl)} | added {len(added)}")
    chk(all(d in idt.index and idt.identity_type[d] == "undecided" for d in lostl),
        f"previously labelled devices disappear only by the margin rule (now 'undecided'): {lostl}")
    for d in changed:
        print(f"   CHANGED {d}: {a.type[d]} ({a.source[d]}) -> {b.type[d]} ({b.source[d]}) | {str(b.detail.get(d, ''))[:100]}")
    # lab seeds must stay as given by the owner
    seeds = b[b.source == "lab_inventory"]
    chk(len(seeds) == 22 and (seeds.type == a[a.source == "lab_inventory"].type.reindex(seeds.index)).all(), "22 lab seeds unchanged")
    # conflicts: two different classes with vote >= 2 in the voting string
    conf = []
    for d, r in idt.iterrows():
        v = [x.split(":") for x in r.votes_all_classes.split(";") if ":" in x]
        strong = [c for c, sc in v if float(sc) >= 2]
        if len(strong) >= 2:
            conf.append((d, r.votes_all_classes, r.identity_type))
    print(f"INFO devices with two classes voted >= 2: {len(conf)}")
    for c in conf[:15]:
        print("   conflict", c)
    # new PLC/HMI labels for manual review
    new_ph = [d for d in added if b.type[d] in ("PLC", "HMI")]
    print(f"\nREVIEW: {len(new_ph)} NEW PLC/HMI labels")
    pd.set_option("display.width", 250)
    for d in new_ph:
        print(f"  {d} {b.type[d]:4s} {b.confidence[d]:6s} role={b.tier1_role[d]:11s} src={idt.strong_sources.get(d, '')!s:14s} | {str(b.detail[d])[:140]}")
    print("VERIFY", "OK" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
