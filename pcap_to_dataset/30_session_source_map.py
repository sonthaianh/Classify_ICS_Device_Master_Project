"""Draft session -> source_family map, inferred ONLY from session names (no provenance file exists).

Heuristic, to be reviewed by the dataset owner. Rules are ordered; first match wins.
Sessions that match no rule stay `unassigned` (never guessed). Does not touch any dataset.
Output: Dataset/features/session_source_map_draft.csv
"""
from pathlib import Path
import re

import pandas as pd

F = Path(__file__).resolve().parent.parent / "Dataset/features"
RULES = [  # (family, regex on the name without its numeric prefix, case-insensitive)
    ("anonymized_hourly", r"anonymized"),
    ("eth2dump_attack_lab", r"^eth\d+dump"),
    ("abb_attack_lab", r"^abb_.*_\d{4}"),
    ("schneider_attack_lab", r"^schneider_.*_\d{4}"),
    ("siemens_attack_lab", r"^(seimens|siemens)_.*_\d{4}"),
    ("sel_quickset", r"(^|_)sel_"),
    ("switch_series_siw_swi", r"(^s\d+-siw|swi\d+|^a\d+[ -]?swi|^g\d+swi|^ap\d+r)"),
    ("4sics", r"4sics"),
    ("cve_synthetic", r"^cve-.*_synt"),
    ("etherwan", r"^etherwan"),
    ("setup_b_series", r"^setup-b"),
    ("booktcpset", r"^booktcpset"),
]


def family(name):
    base = re.sub(r"^\d+_", "", name)
    for fam, rx in RULES:
        if re.search(rx, base, re.IGNORECASE):
            return fam, rx
    return "unassigned", ""


def main():
    x = pd.read_parquet(F / "master_device_windows_v6_fe3.parquet", columns=["device_id", "session_id", "label"])
    pairs = x.drop_duplicates(["device_id", "session_id"])
    sess = pairs.groupby("session_id").agg(n_devices=("device_id", "size"), n_ot_devices=("label", lambda s: int((s == "OT_confirmed").sum()))).reset_index()
    rows = x.groupby("session_id").size().rename("n_windows").reset_index()
    sess = sess.merge(rows, on="session_id")
    sess[["source_family", "rule"]] = sess.session_id.apply(lambda s: pd.Series(family(s)))
    sess["basis"] = "name_pattern_heuristic_unreviewed"
    sess.to_csv(F / "session_source_map_draft.csv", index=False, encoding="utf-8-sig")
    g = sess.groupby("source_family").agg(sessions=("session_id", "size"), device_session_pairs=("n_devices", "sum"), ot_pairs=("n_ot_devices", "sum"), windows=("n_windows", "sum")).sort_values("windows", ascending=False)
    g["windows%"] = (g.windows / g.windows.sum() * 100).round(1)
    print(g.to_string())
    print(sess[sess.source_family == "unassigned"].session_id.sample(40, random_state=0).tolist())


if __name__ == "__main__":
    main()
