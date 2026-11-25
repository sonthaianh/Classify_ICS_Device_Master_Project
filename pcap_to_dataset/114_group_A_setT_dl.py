"""Nhom A (2026-11-25): MLP 64 cot, CNN 1D tu dau (5 hat) va Transformer tu dau (3 hat) tren TAP T cua bo nhan doc lap (cho Bang 4.5). KHONG ghi de ket qua dong bang.
Chay: py -3.14 114_group_A_setT_dl.py   -> phase2_groupA_setT_dl_indep160.json"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "--set", "indep160"]
spec = importlib.util.spec_from_file_location("g110", HERE / "110_group_A_independent.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
lab, sets, A, T = G.base()
T = T[T.index.isin(G.S62.DEV)]
y, g, devs = T.type4.to_numpy(), T.group.to_numpy(), list(T.index)
Xn = np.nan_to_num(sets[G.KS].loc[T.index].to_numpy(dtype=float), nan=0.0)
out, t0 = {"n": len(y)}, time.time()
p = G.generic(lambda tr, te, seed, yy: G.S61.mlp_fit_predict(Xn[tr], yy[tr], Xn[te], seed), y, g, 5, "MLP64-T")
out["MLP64"] = G.ms([balanced_accuracy_score(y, q) for q in p])
p = G.generic(lambda tr, te, seed, yy: G.S62.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None), y, g, 5, "CNN1D-T")
out["CNN1D"] = G.ms([balanced_accuracy_score(y, q) for q in p])
(G.F / "phase2_groupA_setT_dl_indep160.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
S71 = G.M._load("s71", "71_phase2_dl_transformer.py")
p = G.generic(lambda tr, te, seed, yy: S71.ft_predict([devs[i] for i in tr], yy[tr], [devs[i] for i in te], seed, None), y, g, 3, "TF-T")
out["Transformer"] = G.ms([balanced_accuracy_score(y, q) for q in p])
(G.F / "phase2_groupA_setT_dl_indep160.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out), "DONE-SETT", round(time.time() - t0), "s")
