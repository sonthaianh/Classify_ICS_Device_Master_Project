# Dữ liệu trong repo

Thư mục `Dataset/features/` chứa 175 tệp: bảng đặc trưng, nhãn, cách chia fold và các tệp kết quả mà script đọc hoặc ghi. Dữ liệu gói tin thô (2.230 tệp PCAP, khoảng 116 GB ở dạng bảng) không nằm trong repo.

## Đã ẩn danh như thế nào

- **Thiết bị:** mỗi thiết bị mang mã `DEV_xxxx`. Cột địa chỉ MAC chỉ còn ba byte đầu (mã nhà sản xuất), ba byte sau là `00:00:00`.
- **Địa chỉ IP:** mọi địa chỉ IPv4 trong dữ liệu được thay bằng `0.0.0.0`.
- **Tên capture:** mỗi capture mang mã `S` cộng bốn chữ số (ví dụ `S0062`). Thứ tự sắp xếp của mã giống thứ tự của tên gốc, nên cách chia fold và kết quả không đổi.
- **Cột đã xóa giá trị:** tên capture tiêu biểu, ví dụ chỉ số gói tin, model và mô tả thiết bị, chuỗi danh tính trích từ payload, tên máy, hãng suy từ OUI ở các phiếu rà soát.
- **Không đưa vào repo:** bảng gói tin thô, bảng tài sản của cơ sở công nghiệp và các tệp ghép với bảng này, chuỗi danh tính và tên máy trích từ payload, bảng kê capture, các phiếu Excel, tệp log, các phiên bản dữ liệu cũ (v1 đến v5) và dữ liệu tổng hợp.

Script `109`, `26`–`28`, `42`, `45`, `46`, `52`, `54`, `67`, `68`, `84`, `85`, `88` đọc bảng gói tin thô nên không chạy lại được chỉ với dữ liệu trong repo. Các script huấn luyện và đánh giá (ví dụ `34`, `36`, `39`, `107`, `110`, `111`, `113`, `114`) chạy được từ các bảng đã công bố.

## Tệp chính

| Tệp | Nội dung |
|---|---|
| `master_device_windows_v6_fe3.parquet` | 768.927 cửa sổ 45 giây của 3.923 thiết bị với đặc trưng hành vi của giai đoạn 1 |
| `device_labels_v6_rule.csv` | Nhãn tham chiếu của giai đoạn 1 theo luật giao thức và cổng |
| `device_folds_v6.csv` | Cách chia fold theo thiết bị và theo thành phần liên thông |
| `gold_union_v6.csv` | Tập gold của giai đoạn 1 |
| `phase2_payload_profiles_v6_full.json` | Hồ sơ payload theo thiết bị và capture |
| `phase2_sequences_v6_full.pkl` | Chuỗi sự kiện payload cho CNN 1D và Transformer |
| `phase2_eng_features_v6.csv`, `phase2_role_tier1_v6.csv` | Đặc trưng tác vụ kỹ thuật và vai trò kết nối |
| `phase2_type_labels_4class_186_devices.csv` | 186 nhãn vai trò PLC, HMI, EWS, OTHER |
| `phase2_type_labels_4class_186_devices_indep160.csv` | 160 nhãn độc lập với đặc trưng, dùng cho kết quả chính |
| `phase2_groupA_indep160.json` | Kết quả chính của giai đoạn 2 trên bộ nhãn độc lập |
