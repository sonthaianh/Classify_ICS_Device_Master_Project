# Classify ICS Device — nhận diện vai trò thiết bị ICS từ lưu lượng mạng thụ động

Mã nguồn và dữ liệu đã ẩn danh của luận văn thạc sĩ "Cải tiến mô hình nhận diện vai trò thiết bị ICS qua fingerprinting thụ động bằng học kết hợp phục vụ NIDS" (tác giả: sonthai).

Phương pháp gồm hai giai đoạn nối tiếp, chỉ dùng lưu lượng mạng thụ động:

1. **Giai đoạn 1** phân biệt thiết bị công nghiệp (OT) với thiết bị không công nghiệp bằng LightGBM trên 15 đặc trưng hành vi, tính theo cửa sổ 45 giây.
2. **Giai đoạn 2** xác định vai trò PLC, HMI, EWS hoặc OTHER của thiết bị OT bằng Random Forest trên 64 đặc trưng lấy từ tiêu đề thông điệp của bảy giao thức công nghiệp và hành vi giao tiếp. MLP, CNN 1D và Transformer được dùng để so sánh.

## Tìm gì ở đâu

| Bạn cần | Ở đâu |
|---|---|
| **Giai đoạn 1** (OT hay không-OT) | 18 script trong `pcap_to_dataset/`, tên có `phase1` hoặc số 32–39; mô hình trong `demo_webapp/model/` |
| **Giai đoạn 2** (vai trò PLC, HMI, EWS, OTHER) | 43 script trong `pcap_to_dataset/`, tên có `phase2`; mô hình trong `phase2_infer/` |
| **Hai giai đoạn nối nhau** | Đánh giá đầu–cuối: `pcap_to_dataset/110_group_A_independent.py`, `66_endtoend_phase1_phase2.py` |
| **Demo toàn quy trình** trên một tệp PCAP | Web: `demo_webapp/`; dòng lệnh: `pcap_to_dataset/99_phase2_infer_186.py`. Xem [docs/DEMO.md](docs/DEMO.md) |
| Script nào thuộc giai đoạn nào | [docs/PIPELINE.md](docs/PIPELINE.md) (cũng hiện khi mở thư mục `pcap_to_dataset/`) |
| Dữ liệu và cách ẩn danh | [docs/DATA.md](docs/DATA.md) |
| Nguồn PCAP công khai | [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) |

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `pcap_to_dataset/` | 91 script của cả hai giai đoạn: chuyển PCAP, dựng dữ liệu, gán nhãn, huấn luyện, đánh giá |
| `demo_webapp/` | Nguyên mẫu web chạy **cả hai giai đoạn** trên một tệp PCAP |
| `phase2_infer/` | Mô hình Random Forest của giai đoạn 2 dùng cho nguyên mẫu |
| `utils/` | Tra cứu OUI, dựng khung gói tin, danh mục cổng và giao thức OT/IT |
| `Dataset/features/` | Bảng đặc trưng, nhãn, cách chia fold và tệp kết quả, đã ẩn danh |
| `docs/` | Chỉ mục script theo giai đoạn, hướng dẫn demo, mô tả dữ liệu, nguồn PCAP công khai |

Script nằm phẳng trong một thư mục và giữ số thứ tự gốc, vì chúng gọi nhau theo tên tệp và tìm dữ liệu theo đường dẫn tương đối. Số thứ tự cho biết trình tự làm việc; bảng ở [docs/PIPELINE.md](docs/PIPELINE.md) xếp chúng theo giai đoạn.

## Cài đặt

Dự án chạy với Python 3.14. Nguyên mẫu cần thêm Wireshark (chương trình `tshark`).

```bash
pip install -r requirements.txt
```

## Demo toàn quy trình

```bash
python demo_webapp/app.py                                    # giao diện web, mở http://127.0.0.1:5001
python pcap_to_dataset/99_phase2_infer_186.py <tệp.pcap>     # dòng lệnh
```

Cả hai chạy giai đoạn 1 rồi giai đoạn 2 trên tệp PCAP đưa vào. Chi tiết ở [docs/DEMO.md](docs/DEMO.md).

## Chạy lại kết quả của luận văn

Chạy từ thư mục gốc của repo. Ví dụ:

```bash
python pcap_to_dataset/36_phase1_gold_eval.py                               # giai đoạn 1 trên tập gold
python pcap_to_dataset/110_group_A_independent.py --set indep160 --tab      # giai đoạn 2 trên bộ nhãn độc lập, kèm đầu–cuối
python pcap_to_dataset/107_phase2_fold_counts.py                            # số thiết bị theo fold, không huấn luyện
```

Bảng ghép script với kết quả của luận văn nằm ở cuối [docs/PIPELINE.md](docs/PIPELINE.md).

Lưu ý khi so số: kết quả chia theo thiết bị và theo capture đầu tiên lặp lại đúng từng chữ số. Kết quả chia theo thành phần liên thông ở script 110 là trung bình của 12 lần đánh số lại thành phần, nên trung bình có thể lệch khoảng 0,002 giữa các lần chạy; đặt biến môi trường `PYTHONHASHSEED` để cố định.

## Dữ liệu

Repo không chứa dữ liệu gói tin thô. Các capture của phòng thí nghiệm và bảng tài sản của cơ sở công nghiệp là dữ liệu nội bộ nên không được công bố. Người đọc có thể tải dữ liệu công khai từ các nguồn liệt kê ở [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

Các bảng trong `Dataset/features/` đã được ẩn danh: thiết bị mang mã `DEV_xxxx`, địa chỉ MAC chỉ còn mã nhà sản xuất, địa chỉ IP và tên capture đã được thay thế. Chi tiết ở [docs/DATA.md](docs/DATA.md).
