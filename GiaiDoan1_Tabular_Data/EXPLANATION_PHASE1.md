# 📙 HƯỚNG DẪN & GIẢI THÍCH CHI TIẾT GIAI ĐOẠN 1 (TABULAR MACHINE LEARNING)
> 💡 **Dành riêng cho Lập trình viên Phần mềm (Software Engineers):** Tài liệu này dịch các thuật toán Machine Learning trên dữ liệu bảng (XGBoost, LightGBM, CatBoost, Optuna) sang các khái niệm Lập trình, Cấu trúc dữ liệu & Thiết kế hệ thống (If-Else Rules, Middleware Data Transformation, Auto-tuning Configuration, Strategy Pattern).

---

## 💻 TỪ KHÓA BẢN CHẤT: TREE-BASED MACHINE LEARNING VS PHẦN MỀM

| Khái niệm AI / Tabular ML | Tương đương trong Lập trình Phần mềm | Giải thích bản chất |
| :--- | :--- | :--- |
| **Decision Tree (Cây quyết định)** | Bộ quy tắc `IF-ELSE` lồng nhau | Mô hình tự động tạo ra một cây điều kiện dạng `if (tuoi > 30 && thunhap < 10tr) return Nợ_Xấu;`. |
| **Gradient Boosting (XGBoost/LightGBM/CatBoost)** | Thuật toán tối ưu chuỗi lặp (`Iterative Fix`) | Tạo ra cây thứ 1 $\rightarrow$ Kiểm tra cây 1 sai ở đâu $\rightarrow$ Tạo cây thứ 2 chuyên sửa lỗi của cây 1 $\rightarrow$ Ghép 100 cây lại thành bộ quy tắc cực kỳ chính xác. |
| **One-Hot Encoding** | Chuyển Enum / String thành mảng Flag Bolean | Biến biến chữ `'Thành phố'` (Hà Nội, HCM) thành các cột số `is_HaNoi = 1`, `is_HCM = 0`. |
| **ColumnTransformer** | Middleware Data Transformation Pipeline | Chuỗi xử lý làm sạch, chuẩn hóa dữ liệu tự động trước khi đưa vào Controller xử lý logic. |
| **Stratified K-Fold CV** | Kịch bản Unit Test / Integration Test đa dạng ( Cross-Validation ) | Chia dữ liệu làm 5 phần, luân phiên dùng 4 phần làm Code và 1 phần làm Test Case để kiểm tra xem mô hình có bị bug Overfitting trên bất kỳ tập dữ liệu nào không. |
| **Optuna** | Framework Auto-Tuning Config tự động | Tự động chạy thử các bộ cấu hình (learning rate, depth,...) để tìm ra file `.json` cấu hình tối ưu nhất cho hệ thống. |
| **Ensemble / Blending** | Pattern `Strategy` / `Voting System` | Gọi 3 API mô hình khác nhau (CatBoost, XGBoost, LightGBM), sau đó lấy trung bình cộng dự đoán để đảm bảo tính an toàn & chính xác tuyệt đối. |

---

## 📓 BÀI TẬP FILE 01: `01_Tabular_ML_Pipeline.ipynb`

### 🏢 Bối cảnh thực tế (Real-world Scenario):
Bạn xây dựng tính năng **Đánh Giá Rủi Ro Tín Dụng Nợ Xấu (Credit Scoring)** cho một Ngân hàng điện tử. Hệ thống nhận vào thông tin khách hàng gồm cả số (Thu nhập, Điểm tín dụng) và chữ (Loại công việc, Tình trạng hôn nhân). Dữ liệu thực tế thường xuyên bị thiếu thông tin (`NaN`) do khách hàng bỏ qua bước nhập liệu.

### 🔹 Bài Tập 1: Tiền Xử Lý Dữ Liệu Tự Động (`ColumnTransformer Pipeline`)

#### 🛠️ Các bước xử lý trong Code:
1. **Xử lý giá trị NULL (Imputation)**:
   - *Code*: `df['num_feat_1'].fillna(df['num_feat_1'].median())`
   - *Ý nghĩa phần mềm*: Tránh lỗi `NullPointerException`. Dùng giá trị trung vị của toàn bộ hệ thống để điền vào các khoảng trống dữ liệu.
2. **Thiết kế Middleware Chuẩn Hóa Dữ Liệu (`ColumnTransformer`)**:
   - *StandardScaler*: Đưa các số tiền từ vài triệu đến hàng tỷ về cùng một quy mô phân phối chuẩn ($\mu=0, \sigma=1$), tránh việc cột tiền lương quá lớn làm "lấn áp" các cột số dư tài khoản.
   - *OneHotEncoder*: Mã hóa cột chữ như `'Category_A', 'Category_B'` thành mảng Flag $0/1$. `sparse_output=False` ép kết quả về mảng NumPy Dense dễ thao tác.
   - *Output*: Mảng NumPy sẵn sàng nạp vào các thuật toán Machine Learning.

---

### 🔹 Bài Tập 2: Đánh Giá Chống Overfitting Với Stratified K-Fold CV & Tree Models

#### 🛠️ Các bước xử lý trong Code:
1. **Chia nếp gấp Unit Test (`StratifiedKFold`)**:
   - `n_splits=5`: Chia toàn bộ 500 khách hàng thành 5 phần bằng nhau ($100$ khách hàng/phần).
   - `shuffle=True, random_state=2026`: Xáo trộn dữ liệu cố định seed để kết quả kiểm thử có thể tái lập ($100\%$ Reproducible).
2. **Huấn luyện XGBoost & LightGBM**:
   - Trong mỗi Fold (Lần test): Huấn luyện mô hình trên 4 phần ($400$ mẫu) và dự đoán xác suất `predict_proba` trên 1 phần còn lại ($100$ mẫu).
   - Lưu trữ dự đoán ra mảng `oof_preds` (Out-Of-Fold Predictions).
3. **Tính điểm OOF Accuracy**:
   - Đánh giá chất lượng của mô hình trên toàn bộ tập dữ liệu khi mô hình chưa từng được nhìn thấy tập Validation trong lúc học.

---

## 📓 BÀI TẬP FILE 02: `02_CatBoost_and_Encoding.ipynb`

### 🏢 Bối cảnh thực tế (Real-world Scenario):
Hệ thống viễn thông muốn dự đoán **Khách Hàng Hủy Hợp Đồng Dịch Vụ (Telecom Churn Analysis)**. Dữ liệu chứa rất nhiều cột định dạng chữ như Thành phố (`HaNoi, HCM, DaNang, CanTho`) và Nghề nghiệp (`Engineer, Teacher, Doctor,...`).

### 🔹 Bài Tập 1 & 2: Làm Chủ CatBoost Với Native Categorical Features

#### 🛠️ Các bước xử lý trong Code:
1. **Ép kiểu dữ liệu chuỗi (String Formatting)**:
   - *Code*: `df[col] = df[col].astype(str)`
   - *Lý do*: Đảm bảo các cột thuộc tính phân loại có kiểu dữ liệu chuỗi chuẩn để CatBoost nhận diện.
2. **Truyền trực tiếp danh sách cột chữ (`cat_features`)**:
   - *Code*: `model_cb = CatBoostClassifier(cat_features=['city', 'job'], ...)`
   - *Ưu điểm vượt trội của CatBoost*: **Không cần thực hiện One-Hot Encoding thủ công**! CatBoost tự động sử dụng thuật toán Target Statistics mã hóa các giá trị chữ trong lúc học, tránh làm phình to dung lượng RAM và tăng tốc độ xử lý gấp nhiều lần.

---

### 🔹 Bài Tập 3: So Sánh 3 Mô Hình & Kết Hợp Trọng Số (`Weighted Blending`)

#### 🛠️ Các bước xử lý trong Code:
1. **Mã hóa Dummy Variables cho XGBoost & LightGBM**:
   - Vì XGBoost và LightGBM yêu cầu đầu vào là số hoàn toàn, ta dùng `pd.get_dummies()` để tự động tạo mảng One-Hot cho các cột chữ.
2. **Công thức Blending Trọng Số (Strategy Pattern)**:
   $$\text{Probability}_{\text{Final}} = 0.4 \times P_{\text{CatBoost}} + 0.3 \times P_{\text{XGBoost}} + 0.3 \times P_{\text{LightGBM}}$$
   - *Tư duy phần mềm*: CatBoost xử lý biến chữ tốt nhất nên nhận $40\%$ trọng số quyết định. XGBoost và LightGBM nhận mỗi bên $30\%$. Việc lấy trung bình có trọng số giúp loại bỏ các lỗi dự đoán sai lệch cá biệt của từng mô hình.

---

## 📓 BÀI TẬP FILE 03: `03_Optuna_Submission_and_Feature_Importance.ipynb`

### 🏢 Bối cảnh thực tế (Real-world Scenario):
Hệ thống **Định Giá Bất Động Sản & Dự Báo Mua Nhà**. Dữ liệu thực tế bị dính nhiều thông tin rác / ngoại lệ cực đoan (Ví dụ: Căn hộ nhập nhầm giá $1000$ tỷ hoặc $-99$ tỷ). Bạn phải làm sạch dữ liệu nhiễu, dùng AI tự tìm tham số tối ưu và xuất file nộp bài chuẩn quy cách.

### 🔹 Bài Tập 1: Handling Outliers (Clipping IQR), Label Encoding & Feature Importance

#### 🛠️ Các bước xử lý trong Code:
1. **Xử lý giá trị ngoại lệ cực đoan (Outlier Clipping)**:
   - *Thuật toán*: Tính khoảng Tứ phân vị $IQR = Q_3 - Q_1$. Ngưỡng hợp lệ nằm trong khoảng $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.
   - *Code*: `df['feature_outliers'].clip(lower=lower_bound, upper=upper_bound)`
   - *Tư duy phần mềm*: Đưa các giá trị dị biệt vượt khung về đúng biên giới hợp lý, tránh việc mô hình bị "sốc" do dữ liệu nhiễu.
2. **Phân tích độ quan trọng của thuộc tính (`Feature Importance`)**:
   - *Code*: `importances = model.get_feature_importance()`
   - *Ứng dụng thực tế*: Giúp kỹ sư phần mềm biết cột dữ liệu nào thực sự có giá trị (ví dụ: Vị trí nhà, Diện tích) và cột nào là dữ liệu rác để chủ động loại bỏ khỏi CSDL, giúp tiết kiệm bộ nhớ và chi phí lưu trữ.

---

### 🔹 Bài Tập 2: Tự Động Tinh Chỉnh Siêu Tham Số Với Optuna (`Auto-Tuning Config`)

#### 🛠️ Các bước xử lý trong Code:
1. **Định nghĩa hàm mục tiêu (`Objective Function`)**:
   - Giống như việc bạn viết một hàm Test tự động nhận vào các giá trị Config ngẫu nhiên (`learning_rate`, `depth`, `l2_leaf_reg`) và trả về điểm chất lượng F1-Score.
2. **Optuna Optimization Loop**:
   - *Code*: `study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=15)`
   - *Cách hoạt động*: Optuna thử 15 bộ cấu hình khác nhau, sử dụng thuật toán Bayesian Optimization để đoán xem bộ tham số tiếp theo nào có khả năng mang lại điểm cao nhất.
   - *Kết quả*: Xuất ra bộ tham số tốt nhất `study.best_params`.

---

### 🔹 Bài Tập 3: Quy Trình Inference Dự Đoán Tập Test & Export Submission

#### 🛠️ Các bước xử lý trong Code:
1. **Đồng bộ hóa Pipeline trên Tập Test**:
   - Tập Test (200 mẫu khách hàng mới) bắt buộc phải trải qua chính xác các bước biến đổi dữ liệu (Clipping, LabelEncoder transform) y hệt như tập Train.
2. **K-Fold Ensemble Inference**:
   - Tích lũy xác suất dự đoán của cả 5 mô hình (từ 5 Folds) trên tập Test:
     $$\text{Test Probability} = \frac{1}{5} \sum_{i=1}^{5} \left( 0.6 \times P_{\text{CatBoost}, i} + 0.4 \times P_{\text{LightGBM}, i} \right)$$
3. **Xuất File Nộp Bài `submission_phase1.csv`**:
   - Áp ngưỡng $0.5$ để đưa xác suất về nhãn nhị phân ($0$ hoặc $1$).
   - Sử dụng `to_csv('submission_phase1.csv', index=False, encoding='utf-8')` để đảm bảo file nộp bài đạt $100\%$ quy chuẩn kỹ thuật của kỳ thi.
