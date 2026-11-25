# Demo toàn quy trình: từ một tệp PCAP đến bảng vai trò thiết bị

Nguyên mẫu chạy cả hai giai đoạn trên một tệp PCAP:

1. Đọc tệp PCAP, dựng flow và cửa sổ 45 giây cho từng thiết bị (địa chỉ MAC).
2. **Giai đoạn 1:** LightGBM chấm điểm OT cho từng thiết bị từ 15 đặc trưng hành vi.
3. Thiết bị có điểm từ 0,5 trở lên đi tiếp; các thiết bị còn lại dừng ở nhóm không-OT.
4. **Giai đoạn 2:** dựng hồ sơ payload của thiết bị, rồi Random Forest dự đoán vai trò PLC, HMI, EWS hoặc OTHER kèm xác suất và mức bằng chứng.

Có hai cách chạy: giao diện web và dòng lệnh. Cả hai dùng chung mã và mô hình.

## Cần có

- Python 3.14 và các thư viện trong `requirements.txt`.
- Wireshark (cần chương trình `tshark`); trên Windows, nguyên mẫu tìm ở `C:/Program Files/Wireshark/tshark.exe`.
- Các tệp mô hình: `demo_webapp/model/phase1_lightgbm.joblib`, `demo_webapp/model/model_meta.json`, `phase2_infer/model_186/phase2_rf.joblib`, `phase2_infer/model_186/phase2_meta.json`.

## Cách 1: giao diện web

```bash
python demo_webapp/app.py
```

Mở <http://127.0.0.1:5001>, tải một tệp PCAP hoặc PCAPNG lên. Trang kết quả cho từng thiết bị:

- điểm giai đoạn 1 và kết luận OT hay không-OT;
- vai trò dự đoán ở giai đoạn 2, bốn xác suất, lớp đứng thứ hai và khoảng cách giữa hai lớp đầu (dấu "?" khi khoảng cách dưới 0,10);
- mức bằng chứng của vai trò kết nối.

Cột "trong dữ liệu huấn luyện" luôn trống ở bản công khai, vì danh sách địa chỉ MAC huấn luyện trong tệp mô hình đã được ẩn danh.

Kết quả tải về được ở dạng CSV.

## Cách 2: dòng lệnh

```bash
python pcap_to_dataset/99_phase2_infer_186.py <tệp.pcap>          # chỉ thiết bị qua ngưỡng giai đoạn 1
python pcap_to_dataset/99_phase2_infer_186.py <tệp.pcap> --all    # mọi thiết bị
```

Kết quả được in ra màn hình và ghi vào `phase2_infer/output/<tên tệp>_phase2_186.json`.

## Huấn luyện lại mô hình của nguyên mẫu

```bash
python demo_webapp/train_model.py                       # giai đoạn 1: LightGBM trên 15 đặc trưng hành vi
python pcap_to_dataset/99_phase2_infer_186.py --train   # giai đoạn 2: Random Forest trên 64 đặc trưng
```

Hai lệnh này cần dữ liệu trong `Dataset/features/`.

## Giới hạn

- Chỉ đọc lưu lượng Ethernet; tối đa 300 MB, 3.000.000 gói và 2.000 địa chỉ MAC mỗi tệp.
- Capture ngắn hơn 45 giây cho điểm kém ổn định; mỗi thiết bị cần ít nhất một cửa sổ có flow.
- Xác suất ở giai đoạn 2 chưa được hiệu chỉnh; lớp HMI và EWS yếu hơn lớp PLC.
- Nguyên mẫu chạy ngoại tuyến trên tệp PCAP, chưa nối vào một NIDS đang vận hành.
