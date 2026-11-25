# Chỉ mục script theo giai đoạn

Mọi script nằm phẳng trong `pcap_to_dataset/` và giữ số thứ tự gốc, vì các script gọi nhau theo tên tệp và tìm dữ liệu theo đường dẫn tương đối (`../Dataset/features`, `../utils`, `../demo_webapp`, `../phase2_infer`). Vì vậy không có thư mục riêng cho từng giai đoạn; bảng dưới cho biết script nào thuộc giai đoạn nào. Cột mô tả là dòng đầu của docstring.

Chạy từ thư mục gốc của repo: `python pcap_to_dataset/<tên script>.py`.

| Nhóm | Số script |
|---|---|
| [0. Chuyển PCAP thành bảng gói tin và dựng flow](#0-chuyển-pcap-thành-bảng-gói-tin-và-dựng-flow) | 7 |
| [1. Dựng bộ dữ liệu hành vi theo thiết bị và cửa sổ 45 giây (phiên bản v6)](#1-dựng-bộ-dữ-liệu-hành-vi-theo-thiết-bị-và-cửa-sổ-45-giây-phiên-bản-v6) | 8 |
| [2. Giai đoạn 1: thiết bị OT hay không-OT](#2-giai-đoạn-1-thiết-bị-ot-hay-khôngot) | 18 |
| [3. Giai đoạn 2: nhãn vai trò](#3-giai-đoạn-2-nhãn-vai-trò) | 17 |
| [4. Giai đoạn 2: đặc trưng, mô hình và phương pháp cơ sở](#4-giai-đoạn-2-đặc-trưng-mô-hình-và-phương-pháp-cơ-sở) | 26 |
| [5. Hai giai đoạn nối nhau: đánh giá đầu–cuối và nguyên mẫu trên một tệp PCAP](#5-hai-giai-đoạn-nối-nhau-đánh-giá-đầucuối-và-nguyên-mẫu-trên-một-tệp-pcap) | 4 |
| [6. Kết quả chính trên bộ nhãn độc lập và các kiểm tra bổ sung](#6-kết-quả-chính-trên-bộ-nhãn-độc-lập-và-các-kiểm-tra-bổ-sung) | 11 |

## 0. Chuyển PCAP thành bảng gói tin và dựng flow

| Script | Mục đích |
|---|---|
| `device_flow_builder.py` |  |
| `dpi.py` | Deep Packet Inspection for port 102 (ISO-on-TCP). |
| `feature_contract.py` | Explicit feature allowlists. Evidence, identity and labels never enter X. |
| `flow_builder.py` |  |
| `pcap_to_parquet.py` |  |
| `pcap_to_raw_parquet.py` |  |
| `test_review_dataset.py` | Semantic regression tests for packet windows, evidence and ML separation. |

## 1. Dựng bộ dữ liệu hành vi theo thiết bị và cửa sổ 45 giây (phiên bản v6)

| Script | Mục đích |
|---|---|
| `23_build_review_dataset.py` | Build v5 packet-exact windows and session evidence; NEVER assign labels. |
| `25_tag_spoofed_and_scope.py` | Step 1 of the expanded behaviour dataset (2026-11-04). |
| `26_build_behavior_dataset.py` | Build v6 (expanded behaviour scope) packet-exact windows and evidence; NEVER assign labels. |
| `27_validate_behavior_dataset.py` | Validate complete v6 (expanded behaviour scope) artifacts; no model fit, no labels. |
| `28_finalize_behavior_dataset.py` | Finalize v6 from the per-capture cache written by 26_build_behavior_dataset.py. |
| `29_make_fe3_view.py` | FE-3 derived view of v6 (v6 itself is never modified). |
| `30_session_source_map.py` | Draft session -> source_family map, inferred ONLY from session names (no provenance file exists). |
| `31_make_folds.py` | Device-level fold table for Phase 1 (no training, no data modified). |

## 2. Giai đoạn 1: thiết bị OT hay không-OT

| Script | Mục đích |
|---|---|
| `32_fe4_ablation.py` | FE-4 ablation on the device-level folds (diagnostic, not the final model). |
| `34_compare_lightgbm_xgboost.py` | Phase 1: LightGBM vs XGBoost under one identical protocol (15 provisional features). |
| `35_gold_candidates.py` | Select candidate devices for the independent gold set and write a BLIND annotation sheet. |
| `36_phase1_gold_eval.py` | Phase 1: out-of-fold scores, independent-gold evaluation and threshold table (reproducible; replaces scratch scripts). |
| `37_phase1_explain_k_ablation.py` | Phase 1: K comparison, drop-one feature ablation, global explanation, metadata-only baseline. |
| `38_phase1_freeze.py` | Phase 1 freeze (FE-5, PROVISIONAL): hash data/labels/folds/gold/features/params so results can be reproduced and audited. |
| `39_phase1_scanner_strict_obstime.py` | Phase 1 follow-ups (2026-11-09): |
| `63_phase1_dl_vs_lgbm.py` | Phase 1 - so sanh deep learning voi LightGBM duoi CUNG giao thuc (2026-11-10, chu de tai yeu cau them). THAM DO, khong doi mo hinh Phase 1 da dong bang. |
| `64_phase1_device_agg_gbm.py` | Phase 1 - doi chung bo sung cho script 63: LightGBM tren feature GOP THEO THIET BI (khong phai DL), 2026-11-10. |
| `65_phase1_agg_candidate_eval.py` | Phase 1 - ung vien chinh: LightGBM tren feature GOP THEO THIET BI (mean/std/min/max cua <= K=20 cua so dau), 2026-11-11. |
| `71_phase1_synthetic_devices.py` | Phase 1: sinh them thiet bi tong hop (synthetic) va thu nghiem xem co giup mo hinh khong (2026-11-12). |
| `72_phase1_learning_curve.py` | Phase 1: duong cong hoc tren du lieu THAT (khong dung du lieu sinh) (2026-11-12). |
| `81_phase1_baselines_pu.py` | Phase 1: baseline cong trinh truoc + hoc voi nhan duong va chua gan nhan (PU) + nhan hop nhat (2026-11-12). KHONG ghi de ket qua dong bang. |
| `84_phase1_new_features.py` | Phase 1: tinh THEM dac trung moi (thoi gian giua cac goi, thong ke do dai goi, do thi giao tiep khong dinh danh) theo dung cua so 45 s cua v6 (2026-11-12). |
| `87_phase1_newfeat_ablation.py` | Phase 1: danh gia dac trung moi (script 84) bang ablation theo nhom, giao thuc DONG BANG (2026-11-12). KHONG ghi de ket qua dong bang. |
| `109_count_raw_macs.py` | Dem lai tu cac tep parquet THO (Dataset/raw_parquet, 2.230 tep): tong so goi, so MAC phan biet (src U dst), MAC unicast / multicast / broadcast (2026-11-22). |
| `111_phase1_13feat.py` | Nhom A2 cua so theo doi (2026-11-25): giai doan 1 voi 13 dac trung (bo 2 dac trung dem cong svc_port_unique, unique_local_port_count). KHONG ghi de ket qua dong bang. |
| `115_recheck_appendix_G.py` | Kiem tra lai hai cho so o Phu luc G cua luan van (2026-11-25). KHONG ghi de ket qua dong bang; chi ghi appendix_G_recheck.json. |

## 3. Giai đoạn 2: nhãn vai trò

| Script | Mục đích |
|---|---|
| `40_phase2_feasibility.py` | Phase 2 step 1 (feasibility, NO training): candidate pool + type-labelling sheet. |
| `42_phase2_payload_profiles.py` | Phase 2: payload communication profiles per device (protocol family x role, message-type tokens, payload sizes, peers). |
| `43_phase2_similarity_labels.py` | Phase 2: similarity of payload communication profiles to the lab reference devices (PLC / HMI / IO) -> pseudo-labels. |
| `44_phase2_role_labels.py` | Phase 2: role-based comparison with the lab reference devices (PLC / HMI / IO) -> coarse pseudo-labels with named analog seeds. |
| `45_phase2_role_confirm.py` | Phase 2 tier 1: confirm client/server role of each device with TCP handshakes, and compare with the port-ownership rule (script 44). |
| `46_phase2_identity_extract.py` | Phase 2 tier 2: extract device IDENTITY evidence from payloads sent BY the device (independent of behaviour features). |
| `47_phase2_identity_types.py` | Phase 2 tier 2: map payload identity evidence (script 46) to device TYPE with explicit, auditable rules; validate on the lab seeds. |
| `48_phase2_type_label_table.py` | Phase 2 tier 2: consolidated type-label table + readiness check (no training). |
| `52_phase2_identity_full.py` | Phase 2: identity extraction over EVERY session of the in-scope devices (careful, resumable, memory-safe version of script 46). |
| `53_phase2_identity_verify.py` | Verification of the full identity extraction (script 52) and of the label table built from it (scripts 47/48). |
| `54_phase2_hostname_extract.py` | Phase 2: hostname evidence from name-service packets sent by the in-scope devices (all sessions, streaming, resumable). |
| `55_phase2_taxonomy4.py` | Phase 2 - thu nho taxonomy con 4 lop: PLC / HMI / EWS / OTHER (2026-11-10, theo yeu cau chu de tai). |
| `56_phase2_add_other_from_gold.py` | Phase 2 - them vi du OTHER (scanner / may IT) tu gold non-OT ma Phase 1 gan nham OT (2026-11-10, chu de tai dong y). |
| `75_inventory_match.py` | Doi chieu bang tai san cua mot nha may (CSV import-asset) voi du lieu v6 (2026-11-12). Chi doc; ghi 1 file dau ra. |
| `77_roster_labels_4sics.py` | Gan nhan thiet bi 4SICS GeekLounge (151020-151022) tu danh sach thiet bi cong bo (Netresec PCAP4SICS) (2026-11-12). |
| `78_merge_roster_labels.py` | Hop nhat nhan tu danh sach 4SICS vao BAN GOLD MOI (khong ghi de file dong bang) (2026-11-12). |
| `79_apply_inventory_labels.py` | Ap quyet dinh cua chu de tai (2026-11-12) de gan nhan Phase 1 / Phase 2 cho thiet bi khop bang tai san nha may (script 75) va danh sach 4SICS (script 77). |

## 4. Giai đoạn 2: đặc trưng, mô hình và phương pháp cơ sở

| Script | Mục đích |
|---|---|
| `57_phase2_eng_features.py` | Phase 2 - feature "tac vu ky thuat" de tach EWS khoi HMI (2026-11-10). KHONG doc lai PCAP: dung token thong diep da trich o script 42. |
| `58_phase2_train4.py` | Phase 2 - huan luyen THU mo hinh cay 4 lop PLC / HMI / EWS / OTHER (2026-11-10). THAM DO, khong phai ket qua cuoi. |
| `59_phase2_train4_tokens_only.py` | Phase 2 - bo sung cho script 58: danh gia 4 lop CHI tren thiet bi co token thong diep (n~65), noi feature "tac vu ky thuat" khong bi |
| `60_phase2_sequences.py` | Phase 2 (huong deep learning) - trich CHUOI su kien payload theo thiet bi (2026-11-10). |
| `61_phase2_dl_mlp.py` | Phase 2 - mo hinh deep learning DOI CHUNG: MLP tren cac bo feature cua script 58/59 (2026-11-10). THAM DO. |
| `62_phase2_dl_seq.py` | Phase 2 - mo hinh deep learning tren CHUOI su kien payload: 1D-CNN + tien huan luyen khong nhan (2026-11-10). THAM DO. |
| `67_phase2_payload_profiles_full.py` | Phase 2: payload profiles over EVERY session and EVERY packet of the in-scope devices (full re-read, 2026-11-11). |
| `68_phase2_sequences_full.py` | Phase 2 (DL) - chuoi su kien payload theo thiet bi, doc DAY DU moi session va moi goi (2026-11-11). Thay cho gioi han cua script 60 (giu nguyen, la pass 1). |
| `71_phase2_dl_transformer.py` | Phase 2 - Transformer tren chuoi su kien payload (de cuong hua hen Transformer-based), 2026-11-11. THAM DO, cung giao thuc voi CNN (script 62). |
| `72_phase2_baselines_ci.py` | Luan van, Chuong 4: co so doi chieu (baseline), khoang tin cay bootstrap va so sanh ghep cap (2026-11-11/12). KHONG doi mo hinh/nhan. |
| `80_phase2_retrain_plus_all.py` | Phase 2: huan luyen lai tren bo nhan MOI (165 nhan = 122 cu + 6 tu 4SICS + 37 tu bang tai san/4SICS) (2026-11-12). KHONG ghi de ket qua dong bang. |
| `82_validate_msg_token_tshark.py` | Kiem chung msg_token() (script 42) voi bo phan tich cua tshark (2026-11-12). Chi doc du lieu; ghi file rieng. |
| `83_phase2_baselines_rle.py` | Phase 2: baseline cong trinh truoc (OUI; khop mo chuoi) va thu nen lap chuoi su kien cho CNN 1D (2026-11-12). KHONG ghi de ket qua dong bang. |
| `85_phase2_extended_reader.py` | Phase 2: MO RONG bo doc thong diep (OPC UA, MQTT, PROFINET DCE/RPC, Mitsubishi MELSEC/MC 3E) (2026-11-12). KHONG ghi de ket qua dong bang. |
| `86_phase2_transformer_rle.py` | Phase 2: nen lap chuoi su kien cho TRANSFORMER (2026-11-12). KHONG ghi de ket qua dong bang. |
| `88_phase2_b3_b4_extract.py` | Phase 2: trich du lieu cho baseline B3 (Al Ghazo & Kumar 2019, phan cap theo tap dich) va B4 (Pospisil & Fujdiak 2024, entropy payload) (2026-11-13). |
| `89_phase2_b3_b4_eval.py` | Phase 2: danh gia baseline B3 (Al Ghazo & Kumar 2019) va B4 (Pospisil & Fujdiak 2024) tren CUNG tap A (117 thiet bi co ho so), cung chi so (2026-11-13). |
| `90_phase2_vendor_and_merge.py` | Phase 2: thu nghiem (1) them dac trung hang (OUI) cho phan loai vai tro, (2) gop SCADA va DCS vao EWS (2026-11-20). |
| `91_phase2_train_186.py` | Phase 2: huan luyen thu (RF + MLP) tren bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang. |
| `92_phase2_tf_186.py` | Phase 2: Transformer tren bo 186 nhan (2026-11-20). Boc script 80 de dung duong dan nhan 186, KHONG ghi de ket qua dong bang. |
| `93_phase2_full_186.py` | Phase 2: chay lai phan con thieu cua quy trinh dong bang tren bo 186 nhan (2026-11-19). KHONG ghi de ket qua dong bang. |
| `94_phase2_rerun_186.py` | Phase 2: chay lai cac script dong bang (58, 59, 61, 66) tren bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang. |
| `95_phase2_errors_186.py` | Phase 2: phan tich loi theo nguon nhan, bang chung vai tro va loai con tren bo 186 nhan (Phu luc G.5; 2026-11-20). |
| `96_phase2_tf_T_186.py` | Phase 2: Transformer tren TAP T (thiet bi co token OT) cua bo 186 nhan (2026-11-20). Boc script 80/92, KHONG ghi de ket qua dong bang. |
| `97_phase2_tf_ci_186.py` | Phase 2: them Transformer (huan luyen tu dau, hat 0) vao bang khoang tin cay bootstrap cua bo 186 nhan (2026-11-20). KHONG ghi de ket qua dong bang. |
| `98_phase2_mlp_T64_186.py` | Phase 2: MLP 64 cot tren TAP T (thiet bi co token OT) cua bo 186 nhan, ca nhan day du va nhan nghiem (2026-11-20). KHONG ghi de ket qua dong bang. |

## 5. Hai giai đoạn nối nhau: đánh giá đầu–cuối và nguyên mẫu trên một tệp PCAP

| Script | Mục đích |
|---|---|
| `66_endtoend_phase1_phase2.py` | Phase 1 -> Phase 2 error propagation (RQ5), 2026-11-11. Chi dung ket qua da co, KHONG huan luyen lai. |
| `69_phase2_infer.py` | Phase 2 tren MOT file PCAP: Phase 1 (OT?) -> cong -> ho so payload -> feature -> RandomForest 4 lop PLC / HMI / EWS / OTHER (2026-11-11). |
| `99_phase2_infer_186.py` | Nguyen mau Phase 2 tren MOT file PCAP, huan luyen lai tren bo 186 nhan (2026-11-20). KHONG ghi de mo hinh 117 nhan. |
| `112_stage_timing.py` | Nhom A4 cua so theo doi (2026-11-25): chi phi cua tung giai doan. KHONG huan luyen lai, KHONG ghi de ket qua dong bang. |

## 6. Kết quả chính trên bộ nhãn độc lập và các kiểm tra bổ sung

| Script | Mục đích |
|---|---|
| `101_phase2_fold_audit.py` | Phase 2: kiem tra cach chia fold va cot trung tren bo 186 nhan (2026-11-22). KHONG ghi de ket qua dong bang. |
| `102_build_manifest_186.py` | Manifest nguon - capture - session - thiet bi cho bo 186 nhan (2026-11-22). Chi doc du lieu co san, khong ghi de ket qua. |
| `103_group_c_checks.py` | Nhom C theo nhan xet cua Thay (2026-11-22). Chi dung du lieu co san; KHONG ghi de ket qua dong bang. |
| `104_label_evidence_dossier.py` | Ho so bang chung cho 9 nhan DCS va 21 nhan ra soat tu bang chung session (2026-11-22). Chi doc du lieu co san. |
| `105_group_c_rerun.py` | Nhom C (bao cao review toan bo, 2026-11-22): cac phan tich can chay them tren bo 186 nhan. KHONG ghi de ket qua dong bang. |
| `106_tf_paired.py` | So sanh ghep cap Random Forest 64 cot - Transformer tu dau (3 hat) tren bo 186 nhan (2026-11-22). Doc du doan tung hat da luu: |
| `107_phase2_fold_counts.py` | Dem so thiet bi huan luyen / kiem tra theo lop o tung fold cua giai doan 2 (2026-11-22). KHONG huan luyen, KHONG ghi de ket qua dong bang. |
| `108_phase2_perclass_models.py` | Bang precision / do phu / F1 theo lop va ma tran nham lan cho RF64, MLP64, CNN 1D, Transformer (tu dau) tren tap A (2026-11-22). |
| `110_group_A_independent.py` | Nhom A cua so theo doi (2026-11-25): chay lai giai doan 2 va baseline tren BO NHAN DOC LAP. KHONG ghi de ket qua dong bang. |
| `113_group_A_extra.py` | Nhom A (2026-11-25), phan bo sung cho muc 4.3 tren bo nhan doc lap: bo dac trung, doi chung chi vai tro ket noi, hoan vi nhan, phan tich loi hat 0, |
| `114_group_A_setT_dl.py` | Nhom A (2026-11-25): MLP 64 cot, CNN 1D tu dau (5 hat) va Transformer tu dau (3 hat) tren TAP T cua bo nhan doc lap (cho Bang 4.5). KHONG ghi de ket qua dong bang. |

## Script ứng với kết quả chính của luận văn

| Nội dung | Script | Tệp kết quả trong `Dataset/features/` |
|---|---|---|
| Giai đoạn 1: LightGBM so với XGBoost | `34_compare_lightgbm_xgboost.py` | `phase1_lgbm_vs_xgb_results.json` |
| Giai đoạn 1: đánh giá trên tập gold, bảng ngưỡng | `36_phase1_gold_eval.py`, `39_phase1_scanner_strict_obstime.py` | `phase1_gold_eval_v6.json`, `phase1_threshold_union_v6.csv` |
| Giai đoạn 1: chia chặt theo thành phần liên thông | `39_phase1_scanner_strict_obstime.py` | `phase1_followups_v6.json`, `phase1_strict_split_v6.csv` |
| Giai đoạn 1: bỏ hai đặc trưng đếm cổng (13 đặc trưng) | `111_phase1_13feat.py` | `phase1_13feat_eval.json` |
| Giai đoạn 1: phương pháp cơ sở | `81_phase1_baselines_pu.py` | `phase1_wp3_baselines_pu.json` |
| Giai đoạn 2 trên bộ nhãn độc lập: Random Forest, MLP, CNN 1D, Transformer, luật vai trò, phương pháp cơ sở, chia theo thành phần liên thông | `110_group_A_independent.py` | `phase2_groupA_indep160.json` |
| Giai đoạn 2: bộ đặc trưng, hoán vị nhãn | `113_group_A_extra.py` | `phase2_groupA_extra_indep160.json` |
| Giai đoạn 2: mô hình học sâu trên tập T | `114_group_A_setT_dl.py` | `phase2_groupA_setT_dl_indep160.json` |
| Đầu–cuối hai giai đoạn và đối chứng một giai đoạn | `110_group_A_independent.py` | `phase2_groupA_indep160.json` (khóa `end_to_end`) |
| Chi phí từng giai đoạn | `112_stage_timing.py` | `stage_timing_a4.json` |
| Số thiết bị theo fold | `107_phase2_fold_counts.py` | `phase2_fold_counts_186.json` |
| Kiểm tra chia fold, kiểm tra lồng, độ nhạy nhãn DCS (bộ đủ 186 nhãn) | `101_phase2_fold_audit.py`, `103_group_c_checks.py`, `105_group_c_rerun.py` | `phase2_fold_audit_186.json`, `phase2_groupc_extra_186.json`, `phase2_groupc_rerun_186.json` |
| Nguyên mẫu hai giai đoạn trên một tệp PCAP | `99_phase2_infer_186.py`, `69_phase2_infer.py`, thư mục `demo_webapp/` | `phase2_infer/output/` |
