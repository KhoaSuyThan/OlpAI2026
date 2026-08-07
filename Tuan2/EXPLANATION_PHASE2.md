# Tài Liệu Giải Thích Tổng Quan Giai Đoạn 2: Neural Machine Translation & Attention (Góc Nhìn Kỹ Sư Phần Mềm)

Tài liệu này hệ thống hóa toàn bộ kiến thức và pipeline thực hành trong **Tuần 2 / Giai Đoạn 2 (Neural Machine Translation - NMT)** dưới góc nhìn **Kỹ sư Phần mềm (Software Engineer)**.

---

## 1. Bảng Quy Đổi Thuật Ngữ AI $\rightarrow$ Khái Niệm Phần Mềm

| Thuật ngữ AI / Machine Learning | Khái niệm Kỹ thuật Phần mềm tương đương | Giải thích chi tiết |
| :--- | :--- | :--- |
| **Encoder-Decoder Architecture** | **Request Serialization & Deserialization Pipeline** | Encoder biến đổi mảng đầu vào thành định dạng nén (Buffer/Payload), Decoder giải mã Buffer đó thành kết quả đầu ra. |
| **Context Vector (No Attention)** | **Fixed-size Payload Bottleneck** | Giống như việc nén một file log dung lượng lớn vào một struct có kích thước cố định $128$ bytes $\rightarrow$ Dễ làm mất thông tin quan trọng. |
| **Bahdanau Attention Mechanism** | **Dynamic Key-Value Lookup / Database Index Search** | Tại mỗi bước sinh dữ liệu, Decoder thực hiện truy vấn (Query) tới toàn bộ bản ghi (Keys/Values) ở Encoder để trích xuất đúng thông tin cần thiết. |
| **Padding Mask (`source_mask`)** | **Null / Sentinel Value Filtering** | Đảm bảo các phần tử vô nghĩa (`<PAD>`) không tham gia vào phép tính trọng số, tương tự như việc bỏ qua `NULL` trong câu lệnh SQL `WHERE id IS NOT NULL`. |
| **Teacher Forcing** | **Mock Data Injection / Golden Reference Debugging** | Trong lúc huấn luyện (Unit Test), bơm dữ liệu chuẩn (Golden Data) vào bước tiếp theo để tránh trôi luồng xử lý do lỗi tích lũy. |
| **Greedy Decoding** | **First-Fit Search / Greedy Algorithm** | Chọn kết quả có xác suất cao nhất tại thời điểm hiện tại mà không quay lui (Backtracking). |
| **SacreBLEU Metric** | **Automated Integration Test Assertion Score** | Đo lường mức độ trùng khớp giữa chuỗi kết quả sinh ra và chuỗi kỳ vọng (Expected Output) theo chuẩn quốc tế. |

---

## 2. Phân Tích Kiến Trúc Hệ Thống & So Sánh Luồng Dữ Liệu

### 2.1. Luồng dữ liệu không Attention (Notebook 01)
```
Input Tokens ──> [Embedding] ──> [Encoder LSTM] ──> (h_n, c_n) [Bottleneck Buffer] ──> [Decoder LSTM] ──> Output Tokens
```
* **Đặc điểm**: Đơn giản, xử lý tuần tự nhẹ nhàng, nhưng bị suy giảm hiệu năng nghiêm trọng khi chuỗi đầu vào dài.

### 2.2. Luồng dữ liệu có Bahdanau Attention (Notebook 02)
```
Input Tokens ──> [Embedding] ──> [Encoder LSTM] ──> All Hidden States H = [h_1, h_2, ..., h_S]
                                                            │
                                  ┌─────────────────────────┘
                                  ▼
[Decoder Hidden s_{t-1}] ──> [Attention Score & Softmax + Masking] ──> Dynamic Context c_t ──> [Decoder Step t] ──> Output Token
```
* **Đặc điểm**: Loại bỏ hoàn toàn bottleneck, tự động điều hướng luồng thông tin (Information Routing), tăng khả năng giải thích hệ thống (Traceability/Explainability) thông qua Attention Matrix.

---

## 3. Các Điểm Cần Chú Ý Trong Kỳ Thi OlpAI 2026

1. **Chống Rò Rỉ Dữ Liệu (Data Leakage Control)**:
   - Vocabulary chỉ được `.fit()` / tạo từ tập `Train`.
   - Các câu xuất hiện ở tập Validation/Test trùng với Train phải được lọc sạch trước khi huấn luyện.
2. **Cố Định Seed & Tính Tái Lập (Reproducibility)**:
   - Luôn chạy `seed_everything(42)` đầu notebook.
3. **Xử Lý Padding Chuẩn Xác**:
   - Luôn sử dụng `ignore_index=pad_id` trong CrossEntropyLoss.
   - Luôn áp dụng mask $-\infty$ trước khi tính Softmax cho Attention weights để tránh học nhiễu từ các vị trí `<PAD>`.

---
