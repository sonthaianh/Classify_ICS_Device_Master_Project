# Classify ICS Device — nhận diện vai trò thiết bị ICS từ lưu lượng mạng thụ động

Mã nguồn và dữ liệu đã ẩn danh của luận văn thạc sĩ "Nghiên cứu cải tiến phương pháp nhận diện thiết bị ICS dựa trên học kết hợp phục vụ hệ thống NIDS" (tác giả: sonthai).

Phương pháp gồm hai giai đoạn, chỉ dùng lưu lượng mạng thụ động:

1. **Giai đoạn 1** phân biệt thiết bị công nghiệp (OT) với thiết bị không công nghiệp bằng LightGBM trên 15 đặc trưng hành vi, tính theo cửa sổ 45 giây.
2. **Giai đoạn 2** xác định vai trò PLC, HMI, EWS hoặc OTHER bằng Random Forest trên 64 đặc trưng lấy từ tiêu đề thông điệp của bảy giao thức công nghiệp và hành vi giao tiếp. MLP, CNN 1D và Transformer được dùng để so sánh.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `pcap_to_dataset/` | 90 script của toàn bộ quy trình: chuyển PCAP, dựng dữ liệu, gán nhãn, huấn luyện, đánh giá. Xem [docs/PIPELINE.md](docs/PIPELINE.md) |
| `utils/` | Tra cứu OUI, dựng khung gói tin, danh mục cổng và giao thức OT/IT |
| `phase1_webapp/` | Nguyên mẫu web: tải một tệp PCAP, chấm điểm giai đoạn 1 và dự đoán vai trò ở giai đoạn 2 |
| `phase2_infer/` | Mô hình Random Forest của giai đoạn 2 dùng cho nguyên mẫu |
| `Dataset/features/` | Bảng đặc trưng, nhãn, cách chia fold và tệp kết quả, đã ẩn danh. Xem [docs/DATA.md](docs/DATA.md) |
| `docs/` | Chỉ mục script, mô tả dữ liệu, danh sách nguồn PCAP công khai |

Tên các thư mục gốc được giữ như trong dự án vì script tìm nhau và tìm dữ liệu theo đường dẫn tương đối.

## Cài đặt

Dự án chạy với Python 3.14.

```bash
pip install -r requirements.txt
```

## Chạy lại kết quả

Chạy từ thư mục gốc của repo. Ví dụ:

```bash
python pcap_to_dataset/107_phase2_fold_counts.py      # số thiết bị theo fold, không huấn luyện
python pcap_to_dataset/36_phase1_gold_eval.py         # giai đoạn 1 trên tập gold
python pcap_to_dataset/110_group_A_independent.py     # giai đoạn 2 trên bộ nhãn độc lập
```

Bảng ghép script với kết quả của luận văn nằm ở cuối [docs/PIPELINE.md](docs/PIPELINE.md).

Lưu ý khi so số: kết quả chia theo thiết bị và theo capture đầu tiên lặp lại đúng từng chữ số (ví dụ `101_phase2_fold_audit.py` cho Random Forest 0,641 trên bộ đủ 186 nhãn). Kết quả chia theo thành phần liên thông dao động khoảng 0,01 đến 0,02 giữa các lần chạy, vì mã nhóm phụ thuộc thứ tự duyệt tập hợp của Python; đặt biến môi trường `PYTHONHASHSEED` để cố định.

## Nguyên mẫu

```bash
python phase1_webapp/app.py
```

Mở địa chỉ mà ứng dụng in ra, rồi tải một tệp PCAP lên.

## Dữ liệu

Repo không chứa dữ liệu gói tin thô. Các capture của phòng thí nghiệm và bảng tài sản của cơ sở công nghiệp là dữ liệu nội bộ nên không được công bố. Người đọc có thể tải dữ liệu công khai từ các nguồn liệt kê ở [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

Các bảng trong `Dataset/features/` đã được ẩn danh: thiết bị mang mã `DEV_xxxx`, địa chỉ MAC chỉ còn mã nhà sản xuất, địa chỉ IP và tên capture đã được thay thế. Chi tiết ở [docs/DATA.md](docs/DATA.md).
