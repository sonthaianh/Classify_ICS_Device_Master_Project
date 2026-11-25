# Phase 1 OT Detector (web app)

Tải file PCAP/PCAPNG lên, nhận xác suất OT cho từng thiết bị (mô hình Phase 1, cửa sổ 45 giây).

## Chạy
```
"C:\Program Files\Python314\python.exe" Tools\phase1_webapp\app.py
```
Mở http://127.0.0.1:5001 . Cần: flask, lightgbm, pandas, pyarrow, joblib, Wireshark (tshark) ở `C:\Program Files\Wireshark`.

## Cấu trúc
- `pcap_reader.py` — đọc PCAP/PCAPNG, cùng 18 cột với `Dataset/raw_parquet` (đã kiểm tra khớp 100% trên 2 file Siemens).
- `pipeline.py` — dùng đúng `build_capture` của script 26 (cùng đặc trưng với huấn luyện), chạy mô hình, thêm bằng chứng luật giao thức/cổng để đối chiếu.
- `train_model.py` — huấn luyện lại mô hình (LightGBM, 15 đặc trưng) → `model/phase1_lightgbm.joblib`, `model/model_meta.json`.
- `app.py`, `templates/index.html` — giao diện Flask.

## Giới hạn
- Chỉ Ethernet (link type 1). Tối đa 300 MB và 3.000.000 gói/file; tối đa 2.000 MAC (chống MAC flooding).
- Mô hình huấn luyện trên toàn bộ dữ liệu v6 với nhãn tham chiếu gán theo giao thức và cổng; MAC trùng dữ liệu huấn luyện được đánh dấu, điểm của chúng không phải kiểm tra độc lập.
- Mỗi thiết bị cần ít nhất một cửa sổ có flow; capture ngắn (<45 s) cho điểm kém ổn định.
- Ngưỡng mặc định 0,5 chưa được chọn theo tập gold (xem checklist).

## Phase 2 (vai trò thiết bị: PLC / HMI / EWS / OTHER)
Từ 2026-11-20 ứng dụng web chạy cả Phase 2 cho thiết bị có điểm Phase 1 từ 0,5 trở lên, bằng mô hình Random Forest huấn luyện trên bộ 186 nhãn (`Tools/phase2_infer/model_186/`, 165 thiết bị có hồ sơ payload, độ chính xác cân bằng tham khảo 0,641). Bảng kết quả có thêm cột "Vai trò (Phase 2)"; bấm vào một thiết bị để xem bốn xác suất, mức bằng chứng và các giới hạn. CSV xuất ra có thêm các cột `phase2_*`.
- `phase2.py` — nạp mô hình và dùng lại hàm của `69_phase2_infer.py` (cùng công thức đặc trưng như lúc huấn luyện); lỗi ở Phase 2 không làm mất kết quả Phase 1 (có ghi chú trong giao diện).
- Dấu "?" ở vai trò nghĩa là hai lớp đầu cách nhau dưới 0,10. Xác suất chưa hiệu chỉnh; HMI và EWS yếu hơn PLC; gateway nối tiếp thường bị dự đoán thành PLC.
- Huấn luyện lại bằng `py -3.14 Tools\pcap_to_dataset\99_phase2_infer_186.py --train`. Không chạy `69_phase2_infer.py --train` để ghi vào `model_186` (script đó đọc bộ 117 nhãn).
- Dòng lệnh không cần web: `py -3.14 Tools\pcap_to_dataset\99_phase2_infer_186.py <file.pcap> [--all]`. Mô hình cũ (117 nhãn) vẫn ở `Tools/phase2_infer/model/`.
