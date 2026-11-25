"""Nguyen mau giai doan 2 tren MOT file PCAP, huan luyen tren BO NHAN DOC LAP (160 nhan, 139 thiet bi co ho so payload) (2026-11-25).
KHONG ghi de mo hinh 186 nhan (model_186) va mo hinh 117 nhan (model).

Ly do: ket qua chinh cua luan van (0,538) tinh tren bo nhan doc lap, nen nguyen mau cung dung mo hinh huan luyen tren dung bo nhan nay,
khong dung 26 nhan gan tu chinh hanh vi ma dac trung do. Boc script 69 giong script 99:
  doi file nhan sang phase2_type_labels_4class_186_devices_indep160.csv (qua script 80), ghi mo hinh vao phase2_infer/model_indep160/,
  lay ket qua tham khao (do chinh xac can bang, do phu tung lop) tu phase2_groupA_indep160.json (script 110),
  ghi ket qua phan tich vao phase2_infer/output/<ten>_phase2_indep160.json.
  py -3.14 116_phase2_infer_indep160.py --train
  py -3.14 116_phase2_infer_indep160.py --selftest
  py -3.14 116_phase2_infer_indep160.py <file.pcap> [--all]
"""
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CUR["path"] = str(M.F / "phase2_type_labels_4class_186_devices_indep160.csv")

REPL = [
    ('MODEL_DIR = OUTDIR / "model"', 'MODEL_DIR = OUTDIR / "model_indep160"'),
    ('cv = json.loads((F / "phase2_train4_v6.json").read_text(encoding="utf-8"))["all_labels"]["A_profiled_only"][FEATURE_SET]',
     'cv = (lambda m: {"balanced_acc": m["mean"], "balanced_acc_std": m["sd"], "recall": m["recall"]})(json.loads((F / "phase2_groupA_indep160.json").read_text(encoding="utf-8"))["report_set_A"]["models"]["RF64"])'),
    ('tk = json.loads((F / "phase2_train4_tokens_only_v6.json").read_text(encoding="utf-8"))["all_labels"].get(FEATURE_SET, {})', 'tk = {}'),
    ('f"{path.stem}_phase2.json"', 'f"{path.stem}_phase2_indep160.json"'),
    ('"caveat": "labels not expert-verified; HMI 11 / EWS 15 samples; no-token devices are mostly OTHER in training"',
     '"caveat": "labels not expert-verified; independent label set (160 labels, 139 with payload profile); HMI 14 / EWS 10 samples; the 26 labels derived from session behaviour are excluded"'),
]


def main():
    src = (HERE / "69_phase2_infer.py").read_text(encoding="utf-8")
    for a, b in REPL:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    mod = types.ModuleType("p69_indep160")
    mod.__file__ = str(HERE / "69_phase2_infer.py")
    exec(compile(src, str(HERE / "69_phase2_infer.py"), "exec"), mod.__dict__)
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
