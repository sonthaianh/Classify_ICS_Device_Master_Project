"""Kiem chung msg_token() (script 42) voi bo phan tich cua tshark (2026-11-12). Chi doc du lieu; ghi file rieng.

Cach lam: lay mau goi payload tu raw_parquet theo 7 ho giao thuc (cong mac dinh), dung lai goi bang scapy (Ether/IP/TCP|UDP/Raw) thanh mot PCAP tam,
cho tshark phan tich, roi so ma chuc nang/loai thong diep do tshark doc voi ky hieu cua msg_token(). Bao cao theo giao thuc:
  n_sampled        so goi lay mau (payload du dai, khong bi cat)
  agree            msg_token va tshark cung ket luan (cung ma chuc nang)
  disagree         ca hai deu doc duoc nhung khac ma
  token_only       msg_token doc duoc, tshark khong nhan ra giao thuc
  tshark_only      tshark doc duoc ma msg_token tra None (bo sot)
  neither          ca hai khong doc duoc
Han che: goi dung lai khong co bat tay TCP that; PCAP tam khong co phan manh/ghep lai; chi cong mac dinh; chi kiem ma chuc nang, khong kiem truong khac.
  py -3.14 82_validate_msg_token_tshark.py [--per-family 400] [--max-files 400]
"""
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "Dataset"
F = DS / "features"
TSHARK = r"C:\Program Files\Wireshark\tshark.exe"


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S42 = _load("s42", "42_phase2_payload_profiles.py")
FAMS = {"s7_iso": ([102], "tcp"), "modbus": ([502], "tcp"), "enip": ([44818, 2222], "any"), "bacnet": ([47808], "udp"), "dnp3": ([20000], "tcp"),
        "iec104": ([2404], "tcp"), "omron_fins": ([9600], "udp")}
TS_FIELDS = ["s7comm.header.rosctr", "s7comm.param.func", "modbus.func_code", "modbus.exception_code", "enip.command", "bacapp.type", "bacapp.confirmed_service",
             "bacapp.unconfirmed_service", "dnp3.al.func", "iec60870_104.type", "iec60870_asdu.typeid", "omron.command"]


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def sample_packets(per_family, max_files):
    random.seed(20261112)
    files = sorted((DS / "raw_parquet").glob("*.parquet"))
    random.shuffle(files)
    got = defaultdict(list)
    cols = ["src_mac", "dst_mac", "src_ip", "dst_ip", "ip_proto", "src_port", "dst_port", "payload_len", "app_payload", "ip_ttl"]
    seen_files = 0
    for p in files:
        if all(len(v) >= per_family for v in got.values()) and len(got) == len(FAMS):
            break
        if p.stat().st_size > 80e6:
            continue
        seen_files += 1
        if seen_files > max_files:
            break
        try:
            t = pq.read_table(p, columns=cols)
        except Exception:
            continue
        t = t.filter(pc.and_(pc.greater(t["payload_len"], 0), pc.is_in(t["ip_proto"], value_set=pa.array([6, 17], type=t["ip_proto"].type))))
        if t.num_rows == 0:
            continue
        d = t.to_pydict()
        for i in range(t.num_rows):
            sp, dp, pr, pay = d["src_port"][i], d["dst_port"][i], d["ip_proto"][i], d["app_payload"][i]
            if sp is None or dp is None or not pay or d["src_ip"][i] is None or ":" in (d["src_ip"][i] or ""):
                continue
            if len(pay) < d["payload_len"][i]:
                continue                                           # payload bi cat: khong dung de so sanh
            for fam, (ports, tp) in FAMS.items():
                if len(got[fam]) >= per_family:
                    continue
                if tp == "tcp" and pr != 6 or tp == "udp" and pr != 17:
                    continue
                if sp in ports or dp in ports:
                    if fam == "s7_iso" and not (pay[0] == 3 and len(pay) > 1 and pay[1] == 0):
                        continue                                   # S7 tren cong 102: chi lay goi co chu ky TPKT (version 3, reserved 0), de khong lay mau toan goi rac/quet
                    got[fam].append((d["src_mac"][i], d["dst_mac"][i], d["src_ip"][i], d["dst_ip"][i], pr, sp, dp, bytes(pay), d["ip_ttl"][i] or 64, p.name))
                    break
    return got, seen_files


def build_pcap(rows, path):
    from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap
    pkts = []
    for k, (sm, dm, sip, dip, pr, sp, dp, pay, ttl, _) in enumerate(rows):
        l4 = TCP(sport=sp, dport=dp, flags="PA", seq=1000 + k * 10000, ack=1) if pr == 6 else UDP(sport=sp, dport=dp)
        pkts.append(Ether(src=sm or "00:00:00:00:00:01", dst=dm or "00:00:00:00:00:02") / IP(src=sip, dst=dip, ttl=int(ttl)) / l4 / Raw(load=pay))
    wrpcap(str(path), pkts)


def run_tshark(pcap):
    cmd = [TSHARK, "-r", str(pcap), "-o", "tcp.desegment_tcp_streams:FALSE", "-o", "tcp.analyze_sequence_numbers:FALSE", "-T", "fields", "-E", "separator=|", "-E", "occurrence=f", "-e", "frame.number"]
    for f in TS_FIELDS:
        cmd += ["-e", f]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    res = {}
    for line in out:
        parts = line.split("|")
        if len(parts) == 1 + len(TS_FIELDS):
            res[int(parts[0])] = dict(zip(TS_FIELDS, parts[1:]))
    return res


def to_int(s):
    try:
        return int(s, 0)
    except (ValueError, TypeError):
        return None


def compare(fam, token, ts, tcp):
    """Tra ve (ket_qua, chi_tiet): ket_qua in agree/disagree/token_only/tshark_only/neither."""
    g = lambda k: to_int(ts.get(k, ""))
    if fam == "s7_iso":
        ros, fn = g("s7comm.header.rosctr"), g("s7comm.param.func")
        t_has = ros is not None
        if token is None:
            return ("tshark_only" if t_has else "neither"), ""
        if token.startswith("s7_job_fc") or token.startswith("s7_ackdata_fc"):
            exp_ros = 1 if token.startswith("s7_job") else 3
            tf = int(token[-2:], 16)
            if not t_has:
                return "token_only", token
            return ("agree" if ros == exp_ros and fn == tf else "disagree"), f"{token} vs ros={ros} fn={fn}"
        if token.startswith("s7_rosctr"):
            return ("agree" if t_has and ros == int(token[len("s7_rosctr"):]) else ("token_only" if not t_has else "disagree")), token
        return ("agree" if not t_has else "disagree"), token                 # iso_cr, iso_cc, s7plus...: tshark khong co s7comm
    if fam == "modbus":
        fc, exc = g("modbus.func_code"), ts.get("modbus.exception_code", "")
        if token is None:
            return ("tshark_only" if fc is not None else "neither"), ""
        if fc is None:
            return "token_only", token
        if token.endswith("_exc"):
            return ("agree" if exc != "" else "disagree"), token
        return ("agree" if fc == int(token[-2:], 16) else "disagree"), f"{token} vs fc={fc}"
    if fam == "enip":
        cmd = g("enip.command")
        if token is None:
            return ("tshark_only" if cmd is not None else "neither"), ""
        if token.startswith("enip_cmd"):
            exp = int(token[8:12], 16)
            return ("token_only" if cmd is None else ("agree" if cmd == exp else "disagree")), f"{token} vs cmd={cmd}"
        return ("agree" if True else "disagree"), token                      # enip_udp_io / enip_udp_other: khong so sanh ma
    if fam == "bacnet":
        cs, us, bt = g("bacapp.confirmed_service"), g("bacapp.unconfirmed_service"), g("bacapp.type")
        if token is None:
            return ("tshark_only" if bt is not None else "neither"), ""
        if token.startswith("bac_creq_svc"):
            return ("agree" if cs == int(token[-2:], 16) else ("token_only" if bt is None else "disagree")), f"{token} vs svc={cs}"
        if token.startswith("bac_ureq_svc"):
            return ("agree" if us == int(token[-2:], 16) else ("token_only" if bt is None else "disagree")), f"{token} vs svc={us}"
        return ("agree" if True else "disagree"), token                      # nlm, apdu khac: khong so sanh ma
    if fam == "dnp3":
        fn = g("dnp3.al.func")
        if token is None:
            return ("tshark_only" if fn is not None else "neither"), ""
        if token.startswith("dnp3_") and "_fc" in token:
            return ("token_only" if fn is None else ("agree" if fn == int(token[-2:], 16) else "disagree")), f"{token} vs fn={fn}"
        return "agree", token
    if fam == "iec104":
        ty, tid = g("iec60870_104.type"), g("iec60870_asdu.typeid")
        if token is None:
            return ("tshark_only" if ty is not None else "neither"), ""
        if token.startswith("iec104_I_t"):
            return ("token_only" if tid is None else ("agree" if tid == int(token[-2:], 16) else "disagree")), f"{token} vs typeid={tid}"
        want = {"iec104_S": 1, "iec104_U": 3}.get(token)
        return ("token_only" if ty is None else ("agree" if want is None or ty == want else "disagree")), f"{token} vs type={ty}"
    if fam == "omron_fins":
        cm = g("omron.command")
        if token is None:
            return ("tshark_only" if cm is not None else "neither"), ""
        exp = int(token.split("_")[1], 16)
        return ("token_only" if cm is None else ("agree" if cm == exp else "disagree")), f"{token} vs cmd={cm}"
    return "neither", ""


def main():
    per, mf = arg("--per-family", 400), arg("--max-files", 400)
    got, nfiles = sample_packets(per, mf)
    rows, owner = [], []
    for fam, lst in got.items():
        for r in lst:
            rows.append(r)
            owner.append(fam)
    tmp = Path(tempfile.mkdtemp(prefix="msgtok_")) / "sample.pcap"
    build_pcap(rows, tmp)
    ts = run_tshark(tmp)
    res = {fam: Counter() for fam in FAMS}
    examples = defaultdict(list)
    tok_counter = defaultdict(Counter)
    for k, (fam, r) in enumerate(zip(owner, rows)):
        sm, dm, sip, dip, pr, sp, dp, pay, ttl, fname = r
        f2, role = S42.family_and_role(int(sp), int(dp), True)
        tok = S42.msg_token(fam, pay[:300], role == "S", pr == 6)
        out, detail = compare(fam, tok, ts.get(k + 1, {}), pr == 6)
        res[fam][out] += 1
        res[fam]["n_sampled"] += 1
        tok_counter[fam][("none" if tok is None else "token") + "/" + out] += 1
        if out in ("disagree", "tshark_only") and len(examples[fam]) < 5:
            examples[fam].append({"result": out, "detail": detail, "file": fname, "payload_head": pay[:16].hex(" ")})
    report = {"files_scanned": nfiles, "per_family_target": per, "tshark": subprocess.run([TSHARK, "-v"], capture_output=True, text=True).stdout.splitlines()[0],
              "families": {}}
    for fam, c in res.items():
        n, a, d, to, so, ne = c["n_sampled"], c["agree"], c["disagree"], c["token_only"], c["tshark_only"], c["neither"]
        both = a + d
        report["families"][fam] = {"n_sampled": n, "agree": a, "disagree": d, "token_only": to, "tshark_only": so, "neither": ne,
                                   "agree_rate_when_both_read": round(a / both, 3) if both else None,
                                   "missed_by_token_rate": round(so / max(so + both, 1), 3) if (so + both) else None, "examples": examples.get(fam, [])}
        print(fam, dict(c))
    (F / "msg_token_tshark_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE-WP5")


if __name__ == "__main__":
    main()
