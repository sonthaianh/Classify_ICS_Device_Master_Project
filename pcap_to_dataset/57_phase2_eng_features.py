"""Phase 2 - feature "tac vu ky thuat" de tach EWS khoi HMI (2026-11-10). KHONG doc lai PCAP: dung token thong diep da trich o script 42.

Token trong phase2_payload_profiles_v6.json co dang  <vai tro thiet bi S|C>:<tx|rx>:<ten thong diep>  (vi du C:tx:s7_job_fc1a).
Feature moi (cong tren toi da 3 session / thiet bi):
  s7_prog_n      : thong diep S7 tai/dua chuong trinh (download/upload block 1a-1f, PI service 28, PLC stop 29) khi thiet bi la CLIENT
  s7_userdata_n  : S7 userdata (rosctr 7: danh sach block, doc SZL/chan doan) khi la client
  mb_diag_n      : Modbus chan doan/nhan dang/vendor (fc 07,08,0b,0c,11,14-17,2b,5a, >=0x41) khi la client
  fins_ctrl_n    : FINS dieu khien/cau hinh/file/debug (khac 01xx doc-ghi du lieu va 0000) khi la client
  enip_disc_n    : EtherNet/IP ListIdentity/ListServices/NOP (0063,0064,0001) khi la client
  eng_task_n / eng_task_share / has_download_upload / eng_any
  c_tx_distinct  : so loai thong diep client da gui (bo token thiet lap ket noi)
  n_client_fams  : so ho giao thuc OT thiet bi dong vai client
  log_servers    : log1p(so may chu thiet bi lien lac, lay max theo session)
Danh gia: EWS vs HMI tren thiet bi co ho so payload. LUU Y quan trong:
  * 5 nhan EWS duyet tay (owner_approved_session_evidence) duoc chon tu TEN SESSION mo ta chinh nhung tac vu nay -> khong doc lap; bao cao rieng.
  * feature S7 co the chi phan anh "dung giao thuc S7": co bang phan tich ho giao thuc chinh va AUC trong rieng nhom S7.
Dau ra: phase2_eng_features_v6.csv, phase2_eng_features_eval_v6.json
Chay: py -3.14 Tools/pcap_to_dataset/57_phase2_eng_features.py
"""
import json
import os
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

F = Path(__file__).resolve().parents[1] / "Dataset" / "features"
RNG = np.random.default_rng(20261110)

SETUP = ("iso_cr", "iso_cc", "s7_job_fcf0", "s7_ackdata_fcf0", "enip_cmd0065", "enip_cmd0066", "iso_data_other(mms?)")
S7_PROG = re.compile(r"^s7_(job|ackdata)_fc(1a|1b|1c|1d|1e|1f|28|29)$")
MB = re.compile(r"^mb_req_fc([0-9a-f]{2})$")
FINS = re.compile(r"^fins_([0-9a-f]{4})_req$")
ENIP = re.compile(r"^enip_cmd([0-9a-f]{4})_req$")
MB_DIAG = {0x07, 0x08, 0x0B, 0x0C, 0x11, 0x14, 0x15, 0x16, 0x17, 0x2B, 0x5A}


def device_tokens():
    recs = json.loads((F / os.environ.get("P2_PROFILE", "phase2_payload_profiles_v6.json")).read_text(encoding="utf-8"))["records"]
    tok, sess, fam, tx, rx, srv = defaultdict(Counter), defaultdict(set), defaultdict(Counter), Counter(), Counter(), Counter()
    for r in recs:
        d = r["device_id"]
        sess[d].add(r["session_id"])
        for k, v in r["tokens"].items():
            tok[d][k] += v
        for k, v in r["fam_role"].items():
            fam[d][k] += v
        tx[d] += r.get("tx_n", 0)
        rx[d] += r.get("rx_n", 0)
        srv[d] = max(srv[d], r.get("n_servers_contacted", 0))
    return tok, sess, fam, tx, rx, srv


def feats(d, tok, fam, tx, rx, srv):
    c = Counter()
    ctx_total, ctx_names, call_total = 0, set(), 0
    for k, v in tok[d].items():
        role, direc, name = k.split(":", 2)
        if role != "C":
            continue
        call_total += v          # moi thong diep phia client (ca hai chieu) - mau so cua eng_task_share
        if direc == "tx":
            ctx_total += v
            if not name.startswith(SETUP):
                ctx_names.add(name)
        if S7_PROG.match(name):
            c["s7_prog_n"] += v
        if name == "s7_rosctr7" and direc == "tx":
            c["s7_userdata_n"] += v
        if direc != "tx":
            continue
        m = MB.match(name)
        if m:
            fc = int(m.group(1), 16)
            if fc in MB_DIAG or fc >= 0x41:
                c["mb_diag_n"] += v
        m = FINS.match(name)
        if m and not m.group(1).startswith("01") and m.group(1) != "0000":
            c["fins_ctrl_n"] += v
        m = ENIP.match(name)
        if m and m.group(1) in ("0063", "0064", "0001"):
            c["enip_disc_n"] += v
    eng = sum(c[k] for k in ("s7_prog_n", "s7_userdata_n", "mb_diag_n", "fins_ctrl_n", "enip_disc_n"))
    cf = {k.split("|")[0] for k, v in fam[d].items() if k.endswith("|C") and v > 0 and not k.startswith(("it_service", "unknown"))}
    t = tx[d] + rx[d]
    out = {k: c[k] for k in ("s7_prog_n", "s7_userdata_n", "mb_diag_n", "fins_ctrl_n", "enip_disc_n")}
    out.update(eng_task_n=eng, eng_task_share=eng / max(call_total, 1), has_download_upload=int(c["s7_prog_n"] > 0), eng_any=int(eng > 0),
               c_tx_distinct=len(ctx_names), n_client_fams=len(cf), log_servers=float(np.log1p(srv[d])), tx_share=(tx[d] / t if t else np.nan),
               c_tx_total=ctx_total)
    return out


def fisher(a, b, c, d):
    """Fisher exact 2-sided cho bang [[a,b],[c,d]] (khong phu thuoc scipy)."""
    n1, n2, k = a + b, c + d, a + c
    tot = n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(tot, k)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, k - n2), min(n1, k) + 1) if pr(x) <= p0 * (1 + 1e-9))


def auc_ci(y, s, n=500):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y)) < 2:
        return None
    base = roc_auc_score(y, s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        bs.append(roc_auc_score(y[i], s[i]))
    return round(float(base), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    tok, sess, fam, tx, rx, srv = device_tokens()
    devs = sorted(sess)
    X = pd.DataFrame({d: feats(d, tok, fam, tx, rx, srv) for d in devs}).T
    X.index.name = "device_id"
    X["has_tokens"] = [int(bool(tok[d])) for d in X.index]
    roles = pd.read_csv(F / "phase2_role_tier1_v6.csv", encoding="utf-8-sig").set_index("device_id")
    X["dominant_family"] = roles.dominant_ot_family.reindex(X.index)
    X.to_csv(F / "phase2_eng_features_v6.csv", encoding="utf-8-sig")

    lab = pd.read_csv(F / "phase2_type_labels_4class_plus_v6.csv", encoding="utf-8-sig").set_index("device_id")
    both = lab[lab.type4.isin(["EWS", "HMI"])].join(X, how="left")
    prof = both[both.has_tokens.fillna(0).astype(int) == 1].copy()
    prof["owner_approved"] = (prof.source == "owner_approved_session_evidence")
    FEATS = ["s7_prog_n", "s7_userdata_n", "mb_diag_n", "fins_ctrl_n", "enip_disc_n", "eng_task_n", "eng_task_share", "c_tx_distinct", "n_client_fams", "log_servers", "tx_share"]
    res = {"coverage": {"EWS_labelled": int((lab.type4 == "EWS").sum()), "EWS_with_tokens": int(((prof.type4 == "EWS")).sum()),
                         "EWS_identity_with_tokens": int(((prof.type4 == "EWS") & ~prof.owner_approved).sum()),
                         "HMI_labelled": int((lab.type4 == "HMI").sum()), "HMI_with_tokens": int((prof.type4 == "HMI").sum())}}
    res["dominant_family_table"] = (prof.assign(grp=np.where(prof.type4 == "HMI", "HMI", np.where(prof.owner_approved, "EWS_owner", "EWS_identity")))
                                    .groupby(["grp", prof.dominant_family.fillna("none")]).size().unstack(0, fill_value=0).to_dict())

    def evaluate(sub, name):
        y = (sub.type4 == "EWS").astype(int).values
        out = {"n_EWS": int(y.sum()), "n_HMI": int((1 - y).sum()), "auc": {}, "binary": {}}
        for f in FEATS:
            s = sub[f].astype(float).fillna(0).values
            out["auc"][f] = auc_ci(y, s)
        for f in ("has_download_upload", "eng_any"):
            v = sub[f].astype(int).values
            a, b, c, d = int(((v == 1) & (y == 1)).sum()), int(((v == 1) & (y == 0)).sum()), int(((v == 0) & (y == 1)).sum()), int(((v == 0) & (y == 0)).sum())
            out["binary"][f] = {"EWS_yes": a, "HMI_yes": b, "EWS_no": c, "HMI_no": d, "fisher_p": round(fisher(a, b, c, d), 4)}
        res[name] = out
        return out

    evaluate(prof[~prof.owner_approved], "EWS_identity_vs_HMI")          # chinh: nhan EWS tu danh tinh (doc lap voi tac vu)
    evaluate(prof, "EWS_all_vs_HMI")                                      # co 5 ca duyet tay (khong doc lap)
    s7 = prof[prof.dominant_family.isin(["s7_iso"])]
    evaluate(s7, "within_S7_family_only")                                 # kiem soat nhieu giao thuc
    res["note"] = ("n rat nho; ly giai la thang tham do, khong phai ket qua. 5 EWS duyet tay duoc chon tu ten session mo ta cung tac vu nen khong doc lap. "
                   "Feature S7 co the chi phan anh giao thuc S7: xem within_S7_family_only.")

    # ---- tu kiem ----
    assert X.index.is_unique and len(X) == len(devs)
    assert res["coverage"]["EWS_labelled"] == 15 and res["coverage"]["HMI_labelled"] == 11
    assert (X[["s7_prog_n", "eng_task_n"]] >= 0).all().all() and ((X.eng_task_share >= 0) & (X.eng_task_share <= 1.0 + 1e-9)).all()
    # kiem tay 1 thiet bi: DEV_3229 (STEP7 PC, EWS duyet tay) phai co s7_prog_n > 0
    chk = {}
    for dd in ("DEV_3229", "DEV_0160", "DEV_0235"):
        chk[dd] = None if dd not in X.index else int(X.loc[dd, "s7_prog_n"])
    res["spot_check_s7_prog_n_owner_approved"] = chk
    (F / "phase2_eng_features_eval_v6.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    sys.exit(main())
