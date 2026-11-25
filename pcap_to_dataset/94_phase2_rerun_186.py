"""Phase 2: chay lai cac script dong bang (58, 59, 61, 66) tren bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang.

Cach lam: dung script 80 de chuyen huong pd.read_csv (file nhan cu -> file 186), roi chay nguyen ham cua script goc.
Moi file dau ra ghi vao Dataset/features duoc doi ten them hau to _186 (vi du phase2_train4_v6.json -> phase2_train4_v6_186.json);
file doc lai trong cung tien trinh cung duoc doi sang ten _186 neu da ghi.
Chay (tung phan):
  py -3.14 94_phase2_rerun_186.py --rf      # script 58: bo dac trung x tap A/B/C x strict, hoan vi nhan, tam quan trong   -> phase2_train4_v6_186.json, preds _186.csv
  py -3.14 94_phase2_rerun_186.py --tok     # script 59: chi thiet bi co token (+ --perm)                                  -> phase2_train4_tokens_only_v6_186.json
  py -3.14 94_phase2_rerun_186.py --mlp     # script 61: MLP 64 cot, 5 hat                                                   -> phase2_dl_mlp_v6_186.json
  py -3.14 94_phase2_rerun_186.py --b25     # script 83: B2 (OUI) + B5 (1-NN khop mo)                                       -> phase2_wp2_baselines_186.json
  py -3.14 94_phase2_rerun_186.py --b34     # script 89: B3 + B4                                                              -> phase2_wp2b_b3_b4_186.json
  py -3.14 94_phase2_rerun_186.py --e2e     # script 66: lan truyen loi giai doan 1 -> 2 (can --rf truoc)                    -> endtoend_phase1_phase2_v6_186.*
"""
import importlib.util
import io
import sys
import types
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
F, CUR = M.F, M.CUR
CUR["path"] = str(F / "phase2_type_labels_4class_186_devices.csv")
WRITTEN = set()


def _rename(p):
    p = Path(p)
    try:
        inside = p.resolve().parent == F.resolve()
    except OSError:
        inside = False
    if inside and "186" not in p.name:
        q = p.with_name(p.stem + "_186" + p.suffix)
        return q
    return p


_wt = Path.write_text


def write_text(self, *a, **k):
    q = _rename(self)
    if q != self:
        WRITTEN.add(self.name)
    return _wt(q, *a, **k)


Path.write_text = write_text
_to_csv = pd.DataFrame.to_csv


def to_csv(self, path_or_buf=None, *a, **k):
    if isinstance(path_or_buf, (str, Path)):
        q = _rename(path_or_buf)
        if q != Path(path_or_buf):
            WRITTEN.add(Path(path_or_buf).name)
        path_or_buf = q
    return _to_csv(self, path_or_buf, *a, **k)


pd.DataFrame.to_csv = to_csv
_prev_read = pd.read_csv   # da duoc script 80 va chuyen huong nhan


def read_csv(p, *a, **k):
    if isinstance(p, (str, Path)) and Path(p).name in WRITTEN:
        p = _rename(p)
    return _prev_read(p, *a, **k)


pd.read_csv = read_csv


def run(fname, patch=None, argv=None, entry="main"):
    src = (HERE / fname).read_text(encoding="utf-8")
    for a, b in (patch or []):
        assert a in src, (fname, a)
        src = src.replace(a, b)
    mod = types.ModuleType("_run_" + fname[:2])
    mod.__file__ = str(HERE / fname)
    old = sys.argv
    sys.argv = [fname] + (argv or [])
    try:
        exec(compile(src, str(HERE / fname), "exec"), mod.__dict__)   # chay nhu script; __name__ khac "__main__" nen goi main() ben duoi
        r = getattr(mod, entry)()
    finally:
        sys.argv = old
    return r


if __name__ == "__main__":
    if "--rf" in sys.argv:
        run("58_phase2_train4.py")
    if "--tok" in sys.argv:
        run("59_phase2_train4_tokens_only.py", argv=["--perm"] if "--perm" in sys.argv else [],
            patch=[('path = F / "phase2_train4_tokens_only_v6.json"', 'path = F / "phase2_train4_tokens_only_v6_186.json"')] if "--perm" in sys.argv else None)
    if "--mlp" in sys.argv:
        run("61_phase2_dl_mlp.py")
    if "--e2e" in sys.argv:
        WRITTEN.add("phase2_train4_preds_v6.csv")   # doc du doan _186 do --rf ghi, khong doc file dong bang
        run("66_endtoend_phase1_phase2.py", patch=[("assert len(d) == 122 and", "assert len(d) == len(lab) and")])
    if "--b25" in sys.argv:    # B2 (OUI) va B5 (1-NN khop mo), script 83
        run("83_phase2_baselines_rle.py", entry="baselines")
    if "--b34" in sys.argv:    # B3 va B4, script 89
        run("89_phase2_b3_b4_eval.py")
    if not any(a in sys.argv for a in ("--rf", "--tok", "--mlp", "--e2e", "--b25", "--b34")):
        print(__doc__)
