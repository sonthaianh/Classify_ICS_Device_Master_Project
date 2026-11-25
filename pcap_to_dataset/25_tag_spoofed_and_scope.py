"""Step 1 of the expanded behaviour dataset (2026-11-04).

Adds two columns to devices.csv WITHOUT touching include_in_training:
  spoofed_identity            True  -> MAC seen only in the flood session S1822
  include_in_behavior_dataset True  -> unicast MAC with >=1 device-flow row and not spoofed

Criterion requested by the user: any device that has a flow with another device;
vendor/MAC identity does not matter. Labels of newly included devices stay unknown.
Idempotent: re-running recomputes the two columns; a backup is written once.
"""
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "Dataset/metadata/devices.csv"
BACKUP = ROOT / "Dataset/metadata/devices_backup_20261104_before_scope.csv"
FLOOD_SESSION = "S1822"


def main():
    if not BACKUP.exists():
        shutil.copy2(CSV, BACKUP)
    d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    sessions = d["session_ids"].str.split(";").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    only_flood = sessions.apply(lambda s: s == {FLOOD_SESSION})
    rows = pd.to_numeric(d["n_device_flow_rows"], errors="coerce").fillna(0)
    first = d["mac"].str.split(":").str[0].apply(lambda h: int(h, 16))
    unicast = (first & 1) == 0
    d["spoofed_identity"] = only_flood.map({True: "True", False: "False"})
    d["include_in_behavior_dataset"] = ((rows >= 1) & unicast & ~only_flood).map({True: "True", False: "False"})
    d.to_csv(CSV, index=False)
    inc = d["include_in_behavior_dataset"] == "True"
    old = d["include_in_training"].str.lower() == "true"
    print("rows", len(d), "| spoofed", int(only_flood.sum()), "| non-unicast", int((~unicast).sum()))
    print("include_in_behavior_dataset:", int(inc.sum()), "| old include_in_training:", int(old.sum()))
    print("old subset of new:", bool((old <= inc).all()), "| new devices added:", int((inc & ~old).sum()))


if __name__ == "__main__":
    main()
