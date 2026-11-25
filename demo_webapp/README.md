# demo_webapp — nguyên mẫu web chạy cả hai giai đoạn

Tải một tệp PCAP hoặc PCAPNG lên; ứng dụng chấm điểm OT cho từng thiết bị (giai đoạn 1), rồi dự đoán vai trò PLC, HMI, EWS hoặc OTHER cho các thiết bị có điểm từ 0,5 trở lên (giai đoạn 2).

```bash
python demo_webapp/app.py
```

Mở <http://127.0.0.1:5001>. Hướng dẫn đầy đủ, gồm cách chạy bằng dòng lệnh, ở [docs/DEMO.md](../docs/DEMO.md).

| Tệp | Vai trò |
|---|---|
| `app.py`, `templates/index.html` | Giao diện Flask: tải tệp, bảng kết quả, xuất CSV |
| `pcap_reader.py` | Đọc PCAP/PCAPNG thành bảng gói tin, cùng 18 cột với dữ liệu huấn luyện |
| `pipeline.py` | **Giai đoạn 1:** dựng cửa sổ và 15 đặc trưng hành vi bằng đúng hàm của `pcap_to_dataset/26_build_behavior_dataset.py`, rồi chấm điểm bằng LightGBM |
| `phase2.py` | **Giai đoạn 2:** dựng hồ sơ payload và 64 đặc trưng bằng hàm của `pcap_to_dataset/69_phase2_infer.py`, rồi dự đoán vai trò bằng Random Forest trong `phase2_infer/model_186/` |
| `train_model.py` | Huấn luyện lại mô hình giai đoạn 1 và ghi vào `model/` |
| `model/` | Mô hình LightGBM của giai đoạn 1 và siêu dữ liệu |

Lỗi ở giai đoạn 2 không làm mất kết quả của giai đoạn 1: thiết bị vẫn có điểm OT, cột vai trò ghi trạng thái lỗi.
