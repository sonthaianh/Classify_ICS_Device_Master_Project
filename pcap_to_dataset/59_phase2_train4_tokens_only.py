"""Phase 2 - bo sung cho script 58: danh gia 4 lop CHI tren thiet bi co token thong diep (n~65), noi feature "tac vu ky thuat" khong bi
confound do thieu du lieu (thiet bi co ho so nhung khong co token bi dien 0, phan lon la OTHER). Dung lai ham cua script 58.
Chay: py -3.14 Tools/pcap_to_dataset/59_phase2_train4_tokens_only.py     Dau ra: phase2_train4_tokens_only_v6.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("s58", HERE / "58_phase2_train4.py")
S58 = importlib.util.module_from_spec(s)
s.loader.exec_module(S58)
F, CLASSES = S58.F, S58.CLASSES


def perm_only():
    """Them kiem tra hoan vi nhan (100 lan) cho payload_role_eng tren tap nghiem co token; ghi them vao JSON da co."""
    import numpy as np
    lab, sets = S58.build()
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    path = F / "phase2_train4_tokens_only_v6.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261110)
    for strict in (False, True):
        sub = lab[tok & (~lab.owner_approved if strict else True)]
        y, g = sub.type4.to_numpy(), sub.group.to_numpy()
        tag = "strict" if strict else "all_labels"
        obs = out[tag]["payload_role_eng"]["balanced_acc"]
        null = [S58.cv(sets["payload_role_eng"].loc[sub.index], y, g, seeds=1, n_est=100, shuffle_y=rng.permutation(y))[0]["balanced_acc"] for _ in range(100)]
        out[tag]["permutation_payload_role_eng"] = {"observed": obs, "null_mean": round(float(np.mean(null)), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                                                    "p_value": round(float((1 + sum(n >= obs for n in null)) / 101), 3), "n_perm": 100}
        print(tag, out[tag]["permutation_payload_role_eng"], flush=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-PERM")


def main():
    if "--perm" in sys.argv:
        return perm_only()
    lab, sets = S58.build()
    eng = pd.read_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig").set_index("device_id")
    tok = pd.Series({d: (d in eng.index and int(eng.loc[d, "has_tokens"]) == 1) for d in lab.index})
    out = {"status": "EXPLORATORY - only devices with message-type tokens; no absence confound for eng features", "classes": CLASSES}
    for strict in (False, True):
        sub = lab[tok & (~lab.owner_approved if strict else True)]
        y, g = sub.type4.to_numpy(), sub.group.to_numpy()
        block = {"n": int(len(y)), "class_counts": pd.Series(y).value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
                 "majority_baseline_acc": round(float(pd.Series(y).value_counts().max() / len(y)), 3)}
        for ks in ("behaviour", "payload_role", "payload_role_eng", "behaviour_payload_role_eng"):
            res, _ = S58.cv(sets[ks].loc[sub.index], y, g)
            block[ks] = res
            print("strict" if strict else "all", ks, res["balanced_acc"], res["recall"], flush=True)
        out["strict" if strict else "all_labels"] = block
    assert out["all_labels"]["n"] == int(tok.sum())
    (F / "phase2_train4_tokens_only_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
