"""Phase 2: MO RONG bo doc thong diep (OPC UA, MQTT, PROFINET DCE/RPC, Mitsubishi MELSEC/MC 3E) (2026-11-12). KHONG ghi de ket qua dong bang.

Quy tac: ham msg_token goc cua script 42 duoc goi truoc; chi khi no tra ve None (hien la ky hieu "~<ho>") moi dung ham mo rong. Vi vay moi ky hieu da co GIU NGUYEN;
cac su kien truoc day la "~opcua", "~mqtt", "~profinet_ethercat", "~codesys_ge_misc", "~unknown" co the doi thanh ky hieu cu the.
  OPC UA   (ho opcua, TCP): HEL/ACK/ERR/RHE; OPN/CLO; MSG -> id dich vu neu doc duoc o che do khong ma hoa (tap id dich vu da biet), nguoc lai "msg_enc"; chunk tiep theo "cont"
  MQTT     (ho mqtt): loai goi dieu khien 1..14 (+ QoS cho PUBLISH)
  PNIO RPC (ho profinet_ethercat, UDP, DCE/RPC 80 byte dau): loai goi RPC + opnum
  MELSEC   (ho codesys_ge_misc va unknown): khung 3E nhi phan (0x5000 yeu cau / 0xD000 phan hoi) voi truong do dai khop; yeu cau -> ma lenh, phan hoi -> ma ket thuc
Chi doc lai cac session co thiet bi dung cac ho nay; cac session con lai dung lai cache cu. Kiem tra: so su kien va ky hieu khong doi o moi vi tri tru cac ky hieu moi.
Dau ra: phase2_sequences_v6_ext.pkl, phase2_sequences_v6_ext_report.json; cache Dataset/features/seq_full_cache_v6_ext/.
Chay: py -3.14 85_phase2_extended_reader.py [--workers 8] | --merge-only
"""
import importlib.util
import json
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S68 = _load("s68", "68_phase2_sequences_full.py")
S42, S67, F = S68.S42, S68.S67, S68.F
CACHE_NEW = F / "seq_full_cache_v6_ext"
OUT = F / "phase2_sequences_v6_ext.pkl"
PREFIX = ("opcua_", "mqtt_", "pnio_", "melsec_")
UA_SERVICES = {397, 422, 425, 428, 431, 461, 464, 467, 470, 473, 476, 527, 530, 533, 536, 554, 557, 631, 634, 664, 667, 673, 676, 712, 715,
               751, 754, 763, 766, 769, 772, 781, 784, 787, 790, 793, 796, 799, 802, 826, 829, 832, 835, 847, 850}
_orig_msg_token = S42.msg_token


def ext_token(fam, b, sender_is_server, tcp):
    n = len(b)
    try:
        if fam == "opcua" and tcp and n >= 8:
            mt = bytes(b[:3])
            if mt in (b"HEL", b"ACK", b"ERR", b"RHE"):
                return "opcua_" + mt.decode().lower()
            if mt in (b"OPN", b"MSG", b"CLO") and b[3] in (0x46, 0x43, 0x41):   # F, C, A
                if b[3] == 0x43:
                    return f"opcua_{mt.decode().lower()}_cont"
                if mt == b"MSG":
                    if n >= 28 and b[24] == 0x01:
                        sid = b[26] | (b[27] << 8)
                        if sid in UA_SERVICES:
                            return f"opcua_msg_{sid}"
                    return "opcua_msg_enc"
                return "opcua_" + mt.decode().lower()
        if fam == "mqtt" and n >= 2 and tcp:
            t = b[0] >> 4
            if 1 <= t <= 14:
                return f"mqtt_t{t}" + (f"q{(b[0] >> 1) & 3}" if t == 3 else "")
        if fam == "profinet_ethercat" and not tcp and n >= 80 and b[0] == 4:
            op = (b[68] | (b[69] << 8)) if (b[4] & 0x10) else ((b[68] << 8) | b[69])
            return f"pnio_rpc_pt{b[1]}_op{op}"
        if fam in ("codesys_ge_misc", "unknown") and n >= 11 and b[1] == 0x00 and b[0] in (0x50, 0xD0):
            ln = b[7] | (b[8] << 8)
            if ln == n - 9:
                if b[0] == 0x50 and n >= 13:
                    return f"melsec_req_c{b[11] | (b[12] << 8):04x}"
                if b[0] == 0xD0:
                    return f"melsec_resp_e{b[9] | (b[10] << 8):04x}"
    except IndexError:
        return None
    return None


def msg_token_ext(fam, b, sender_is_server, tcp):
    t = _orig_msg_token(fam, b, sender_is_server, tcp)
    return t if t else ext_token(fam, b, sender_is_server, tcp)


S42.msg_token = msg_token_ext
S68.CACHE = CACHE_NEW


def work(job):
    return S68.process_file(job)


def affected_sessions():
    recs = json.loads((F / "phase2_payload_profiles_v6_full.json").read_text(encoding="utf-8"))["records"]
    want = {"opcua", "mqtt", "profinet_ethercat", "codesys_ge_misc"}
    return {r["session_id"] for r in recs if any(k.split("|")[0] in want for k in r["fam_role"])}


def merge():
    old = pickle.loads((F / "phase2_sequences_v6_full.pkl").read_bytes())
    new_cache = {p.stem: pickle.loads(p.read_bytes()) for p in CACHE_NEW.glob("*.pkl")}
    changed_pairs = sym_changed = events = 0
    tok_counts = Counter()
    bad = []
    per_dev = {}
    for dev, lst in old["devices"].items():
        out = []
        for s in lst:
            nc = new_cache.get(s["session"])
            nv = nc["devices"].get(dev) if nc else None
            if nv is None:
                out.append((s["session"], [old["vocab"][i] for i in s["sym"]], s["lenb"], s["dtb"]))
                continue
            ev = nv["events"]
            osyms = [old["vocab"][i] for i in s["sym"]]
            nsyms = [e[0] for e in ev]
            if len(ev) != len(osyms) or [e[1] for e in ev] != list(s["lenb"]) or [e[2] for e in ev] != list(s["dtb"]):
                bad.append((dev, s["session"], len(ev), len(osyms)))
                out.append((s["session"], osyms, s["lenb"], s["dtb"]))
                continue
            diff = [(a, b) for a, b in zip(osyms, nsyms) if a != b]
            changed_pairs += int(bool(diff))
            for a, b in diff:
                if not a.split(":")[-1].startswith("~") or not b.split(":")[-1].startswith(PREFIX):
                    bad.append((dev, s["session"], a, b))
                tok_counts[b.split(":")[-1]] += 1
            sym_changed += len(diff)
            if diff:
                per_dev.setdefault(dev, 0)
                per_dev[dev] += len(diff)
            out.append((s["session"], nsyms, s["lenb"], s["dtb"]))
        events += sum(len(o[1]) for o in out)
        old["devices"][dev] = out
    vc = Counter(sym for lst in old["devices"].values() for _, syms, _, _ in lst for sym in syms)
    vocab = ["<pad>"] + sorted(vc)
    idx = {s: i for i, s in enumerate(vocab)}
    devs = {dev: [{"session": ses, "sym": np.array([idx[x] for x in syms], dtype=np.int16), "lenb": np.asarray(lb, dtype=np.uint8), "dtb": np.asarray(db, dtype=np.uint8)}
                  for ses, syms, lb, db in lst] for dev, lst in old["devices"].items()}
    with open(OUT, "wb") as fh:
        pickle.dump({"vocab": vocab, "devices": devs, "cap_events": old["cap_events"]}, fh)
    rep = {"sessions_reprocessed": len(new_cache), "device_sessions_changed": changed_pairs, "events_total": events, "events_symbol_changed": sym_changed,
           "devices_with_changed_events": len(per_dev), "vocab_old": len(old["vocab"]), "vocab_new": len(vocab), "inconsistencies": len(bad), "inconsistency_examples": bad[:8],
           "new_token_counts": dict(tok_counts.most_common(60)), "events_changed_by_device": dict(sorted(per_dev.items(), key=lambda kv: -kv[1])[:30])}
    (F / "phase2_sequences_v6_ext_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("events_changed_by_device",)}, indent=2, ensure_ascii=False)[:3000])
    print("DONE-EXT" if not bad else "CHECK-FAILED")


def main():
    CACHE_NEW.mkdir(exist_ok=True)
    if "--merge-only" not in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        jobs, ids, n_pairs = S67.build_jobs()
        aff = affected_sessions()
        todo = [j for j in jobs if j[0] in aff and not (CACHE_NEW / f"{j[0]}.pkl").exists()]
        print(f"{len(aff)} affected sessions; {len(todo)} to do", flush=True)
        t0, errs = time.time(), []
        with Pool(workers, maxtasksperchild=5) as pool:
            for i, r in enumerate(pool.imap_unordered(work, todo), 1):
                if r["status"] == "error":
                    errs.append(r)
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} files, {time.time() - t0:.0f}s, errors {len(errs)}", flush=True)
        for e in errs[:10]:
            print("ERROR", e)
        if errs:
            return 1
    merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
