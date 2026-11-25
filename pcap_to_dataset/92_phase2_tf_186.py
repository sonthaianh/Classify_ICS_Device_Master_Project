"""Phase 2: Transformer tren bo 186 nhan (2026-11-20). Boc script 80 de dung duong dan nhan 186, KHONG ghi de ket qua dong bang.

Chay (cham, ~2 phut/fold):
  py -3.14 92_phase2_tf_186.py --labels x186 --variant pretrained --seeds 3
  py -3.14 92_phase2_tf_186.py --labels x186 --variant scratch --seeds 3
Dau ra: phase2_retrain_plus_all_tf_x186_<variant>.json
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

_orig_label_file = M.label_file


def label_file(name):
    if name == "x186":
        return M.F / "phase2_type_labels_4class_186_devices.csv"
    if name == "x186_noLow":
        return M.F / "phase2_type_labels_4class_186_devices_x186_noLow.csv"
    return _orig_label_file(name)


M.label_file = label_file

if __name__ == "__main__":
    M.tf()
