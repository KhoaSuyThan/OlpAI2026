# 📙 HƯỚNG DẪN & GIẢI THÍCH CHI TIẾT GIAI ĐOẠN 1 (TABULAR MACHINE LEARNING)

Tài liệu này giải thích chi tiết từng bài tập nhỏ trong từng file notebook thuộc thư mục `GiaiDoan1_Tabular_Data/` một cách trực quan, đơn giản nhất!

---

## 📓 BÀI TẬP FILE 01: `01_Tabular_ML_Pipeline.ipynb`

Thư mục này chứa 2 bài tập quy trình làm việc chuẩn cho dữ liệu bảng:

### 🔹 Bài Tập 1: Tiền Xử Lý & Feature Engineering (`Preprocessing Pipeline`)
* **Đề bài:** Tạo 500 mẫu dữ liệu giả lập có cả biến số (`num_feat_1`, `num_feat_2`), biến chữ (`cat_feat_1`) và có giá trị bị khuyết `NaN`.
* **Cách làm:**
  1. Điền giá trị khuyết `NaN` bằng trung vị `median()`.
  2. Thiết kế `ColumnTransformer` gồm:
     - `StandardScaler()`: Chuẩn hóa cột số về phân phối chuẩn ($\mu=0, \sigma=1$).
     - `OneHotEncoder(sparse_output=False)`: Biến cột chữ 'Category_A', 'Category_B', 'Category_C' thành các cột số $0$ và $1$.
  3. Ép kiểu mảng dữ liệu về `np.array` chuẩn để sẵn sàng đưa vào mô hình.

### 🔹 Bài Tập 2: Stratified K-Fold CV với XGBoost & LightGBM
* **Đề bài:** Áp dụng kỹ thuật chia nếp gấp đánh giá Out-Of-Fold (OOF) để đo độ chính xác của 2 thuật toán cây hàng đầu: XGBoost và LightGBM.
* **Cách làm:**
  1. Dùng `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` chia dữ liệu thành 5 phần đồng đều tỷ lệ nhãn target.
  2. Huấn luyện **XGBoost** (`xgb.XGBClassifier`) trên 4 phần, dự đoán xác suất phần còn lại.
  3. Huấn luyện **LightGBM** (`lgb.LGBMClassifier`) tương tự.
  4. Lấy trung bình dự đoán của 2 mô hình (Ensemble) và tính điểm `OOF Accuracy`.

---

## 📓 BÀI TẬP FILE 02: `02_CatBoost_and_Encoding.ipynb`

Notebook này chứa 3 bài tập nhỏ thực hành làm chủ CatBoost và kỹ thuật Blending:

### 🔹 Bài Tập 1: Giả Lập Dữ Liệu Chứa Thuộc Tính Phân Loại
* **Đề bài:** Tạo 600 mẫu dữ liệu khách hàng với 2 cột thuộc tính chữ: `city` (Hà Nội, HCM, Đà Nẵng, Cần Thơ) và `job` (Kỹ sư, Bác sĩ, Giáo viên...).
* **Cách làm:** Chuyển đổi định dạng dữ liệu cột chữ sang kiểu String `df[col].astype(str)` để chuẩn bị truyền trực tiếp vào CatBoost.

### 🔹 Bài Tập 2: Huấn Luyện CatBoost với Native `cat_features`
* **Đề bài:** Huấn luyện mô hình CatBoost không qua bước One-Hot Encoding thủ công.
* **Cách làm:** 
  1. Khai báo `model_cb = CatBoostClassifier(cat_features=['city', 'job'], ...)`.
  2. CatBoost tự động áp dụng thuật toán Target Statistics mã hóa chữ tối ưu trực tiếp trong lúc học.
  3. Đánh giá điểm OOF Accuracy & F1-Score trên 5-Fold CV.

### 🔹 Bài Tập 3: So Sánh 3 Mô Hình & Kỹ Thuật Weighted Blending
* **Đề bài:** So sánh hiệu năng của cả 3 mô hình (XGBoost, LightGBM, CatBoost) và tạo mô hình tổ hợp tốt nhất.
* **Cách làm:**
  1. Dùng `pd.get_dummies` tạo mã hóa One-Hot để huấn luyện độc lập **XGBoost** và **LightGBM**.
  2. Kết hợp kết quả xác suất dự đoán của 3 mô hình theo công thức trọng số:
     $$\text{OOF}_{\text{Blend}} = 0.4 \times \text{OOF}_{\text{CatBoost}} + 0.3 \times \text{OOF}_{\text{XGBoost}} + 0.3 \times \text{OOF}_{\text{LightGBM}}$$
  3. Tính điểm `Weighted Ensemble Blend OOF Accuracy` để chọn ra phương án nộp bài tối ưu nhất.
