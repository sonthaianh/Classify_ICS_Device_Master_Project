"""Nguyen mau Phase 2 tren MOT file PCAP, huan luyen lai tren bo 186 nhan (2026-11-20). KHONG ghi de mo hinh 117 nhan.

Boc script 69: doi file nhan sang phase2_type_labels_4class_186_devices.csv (qua script 80), ghi mo hinh vao phase2_infer/model_186/,
doc ket qua CV tu cac file *_186 va ghi ket qua phan tich vao phase2_infer/output/<ten>_phase2_186.json.
  py -3.14 99_phase2_infer_186.py --train
  py -3.14 99_phase2_infer_186.py --selftest
  py -3.14 99_phase2_infer_186.py <file.pcap> [--all]
"""
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m80", HERE / "80_phase2_retrain_plus_all.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CUR["path"] = str(M.F / "phase2_type_labels_4class_186_devices.csv")

REPL = [
    ('MODEL_DIR = OUTDIR / "model"', 'MODEL_DIR = OUTDIR / "model_186"'),
    ('"phase2_train4_v6.json"', '"phase2_train4_v6_186.json"'),
    ('"phase2_train4_tokens_only_v6.json"', '"phase2_train4_tokens_only_v6_186.json"'),
    ('f"{path.stem}_phase2.json"', 'f"{path.stem}_phase2_186.json"'),
    ('"caveat": "labels not expert-verified; HMI 11 / EWS 15 samples; no-token devices are mostly OTHER in training"',
     '"caveat": "labels not expert-verified; 186-label set (165 with payload profile); HMI 18 / EWS 20 samples; 21 reviewed labels derived from the same session evidence; 16 customer-site gateways are always predicted PLC", "label_file": "phase2_type_labels_4class_186_devices.csv"'),
]


def main():
    src = (HERE / "69_phase2_infer.py").read_text(encoding="utf-8")
    for a, b in REPL:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    mod = types.ModuleType("p69_186")
    mod.__file__ = str(HERE / "69_phase2_infer.py")
    exec(compile(src, str(HERE / "69_phase2_infer.py"), "exec"), mod.__dict__)
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
