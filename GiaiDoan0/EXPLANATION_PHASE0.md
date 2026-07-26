# 📘 HƯỚNG DẪN & GIẢI THÍCH CHI TIẾT GIAI ĐOẠN 0 (PYTORCH FUNDAMENTALS)

Tài liệu này giải thích chi tiết toàn bộ các bài tập thực hành PyTorch trong thư mục `GiaiDoan0/` một cách đơn giản và dễ hiểu nhất!

---

## 📓 BÀI TẬP FILE 01: `01_PyTorch_Pipeline_Exercises.ipynb`

Thư mục này chứa 3 bài tập nhỏ liên hoàn về quy trình làm bài thi phân loại nhị phân:

### 🔹 Bài Tập 1: Phân Loại Nhị Phân (`Binary Classification`)
* **Đề bài:** Giả lập 200 mẫu thông tin (10 cột đặc trưng). Cột đầu tiên bị thiếu (NaN) $10\%$ dữ liệu. Mục tiêu là dự đoán nhãn `0` hoặc `1`.
* **Cách làm:**
  1. Điền giá trị khuyết bằng trung vị `median`: `df['feat_0'].fillna(df['feat_0'].median())`.
  2. Tạo `BinaryDataset` chuyển đổi dữ liệu sang `torch.float32`.
  3. Xây dựng mạng `BinaryClassifier` gồm các lớp `Linear` -> `ReLU` -> `Linear` -> `Sigmoid` (đưa xác suất về $[0, 1]$).
  4. Huấn luyện 15 Epochs dùng hàm phạt `nn.BCELoss()` và bộ tối ưu `Adam`.

### 🔹 Bài Tập 2: Mô Hình Hồi Quy (`Regression Model`)
* **Đề bài:** Giả lập 150 mẫu với 4 đặc trưng. Mục tiêu là dự đoán một con số thực liên tục (không phải nhãn 0/1).
* **Cách làm:**
  1. Xây dựng mạng `Regressor` với lớp cuối cùng **không có hàm kích hoạt** (vì giá trị dự đoán có thể là âm hoặc dương bất kỳ).
  2. Dùng hàm phạt chênh lệch bình phương `nn.MSELoss()` và bộ tối ưu `SGD`.

### 🔹 Bài Tập 3: Chạy Dự Đoán Tập Test & Xuất File CSV Nộp Bài
* **Đề bài:** Cho 50 mẫu dữ liệu tập Test (chỉ có ID từ 1001 đến 1050, không có nhãn). Yêu cầu chạy dự đoán và xuất file `submission_ex3.csv`.
* **Cách làm:**
  1. Chuyển mô hình sang chế độ đánh giá `model.eval()`.
  2. Tắt tính toán đạo hàm `with torch.no_grad():` để tiết kiệm bộ nhớ và tăng tốc.
  3. Chuyển xác suất đầu ra thành nhãn $0$ hoặc $1$ (`outputs >= 0.5`).
  4. Tạo hàm `create_submission()` lưu thành file `.csv` với tham số bắt buộc `index=False` và `encoding='utf-8'` chuẩn quy chế OlpAI.

---

## 📓 BÀI TẬP FILE 02: `02_PyTorch_Pipeline_Exercises.ipynb`

### 🔹 Phân Loại Đa Lớp Về Đánh Giá Tập Validation (`Multi-Class Classification & Validation`)
* **Đề bài:** Giả lập 300 mẫu với 8 đặc trưng và 3 nhãn phân loại (`0`, `1`, `2`). Yêu cầu chia tập Train/Validation và đánh giá điểm Accuracy qua từng Epoch.
* **Cách làm:**
  1. Chia $80\%$ dữ liệu làm tập Train và $20\%$ làm tập Validation với `train_test_split(..., stratify=y)`.
  2. Tạo mạng `MultiClassClassifier` với lớp cuối có **3 nodes** đại diện cho 3 lớp (Không dùng Softmax ở cuối).
  3. Dùng hàm phạt `nn.CrossEntropyLoss()` (tự động áp dụng Log-Softmax bên trong).
  4. Trong vòng lặp Epoch, dùng `torch.max(preds, dim=1)` để lấy ra lớp có xác suất cao nhất và tính điểm `Val Acc (%)` thực tế.
