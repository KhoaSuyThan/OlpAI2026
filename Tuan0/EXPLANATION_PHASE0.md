# 📘 HƯỚNG DẪN & GIẢI THÍCH CHI TIẾT GIAI ĐOẠN 0 (PYTORCH FUNDAMENTALS)
> 💡 **Dành riêng cho Lập trình viên Phần mềm (Software Engineers):** Tài liệu này dịch toàn bộ khái niệm AI sang các thuật ngữ Lập trình & Phần mềm quen thuộc (Object-Oriented Programming, Data Mapping, Loops, Read-Only States, Unit Testing Concept) để bạn dễ tiếp thu nhất!

---

## 💻 TỪ KHÓA BẢN CHẤT: AI VS LẬP TRÌNH TRUYỀN THỐNG

| Khái niệm AI / PyTorch | Tương đương trong Lập trình Phần mềm | Giải thích bản chất |
| :--- | :--- | :--- |
| **Tensor** | Mảng đa chiều (`Array` / `Matrix`) | Tương tự Mảng N chiều trong C#/Java/Python, nhưng có khả năng tính toán song song trên GPU và tự lưu vết đạo hàm để tối ưu (Auto-grad). |
| **Model (`nn.Module`)** | Hàm biến đổi dữ liệu (`Blackbox Function`) | Nhận đầu vào là Mảng số $X$, tính toán theo công thức chứa các trọng số (parameters) để trả về đầu ra $Y$. |
| **Epoch** | Vòng lặp duyệt dữ liệu (`For-Loop`) | 1 Epoch = 1 Lượt hệ thống duyệt qua toàn bộ danh sách dữ liệu đầu vào để học. |
| **Batch Size** | Kích thước phân trang (`Paging Size`) | Chia danh sách lớn thành từng trang nhỏ (ví dụ 32 dòng/trang) để xử lý cho đỡ ngập bộ nhớ RAM. |
| **Dataset & DataLoader** | Pattern `Iterator` / `Queue Generator` | Lớp chịu trách nhiệm đọc dữ liệu từ DB/File, mã hóa thành Tensor và tạo luồng cấp dữ liệu cho ứng dụng. |
| **`model.train()`** | Bật trạng thái Mutable (`Write Mode`) | Mô hình cho phép cập nhật, ghi đè trọng số khi học. |
| **`model.eval()`** | Bật trạng thái Read-Only (`Read Mode`) | Khóa mô hình, chỉ dùng để đọc/dự đoán, không làm thay đổi trọng số nội bộ. |

---

## 📓 BÀI TẬP FILE 01: `01_PyTorch_Pipeline_Exercises.ipynb`

### 🔹 Bài Tập 1: Bài Toán Phát Hiện Giao Dịch Gian Lận Thẻ Tín Dụng (`Binary Classification`)

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Bạn làm tại một Ngân hàng, cần xây dựng một Service kiểm tra tự động xem một giao dịch quẹt thẻ tín dụng là **Hợp lệ (Nhãn 0)** hay **Gian lận (Nhãn 1)** dựa trên 10 thông số (Số tiền, Vị trí địa lý, Tần suất giao dịch,...). Dữ liệu gửi về bị thiếu $10\%$ thông tin ở cột đầu tiên do lỗi mạng.

#### 🛠️ Các bước xử lý trong Code:
1. **Tiền xử lý & Sửa lỗi dữ liệu thiếu (Data Cleaning)**:
   - *Logic phần mềm*: Giống như việc xử lý `NULL` / `Undefined` trong CSDL.
   - *Code*: `df['feat_0'].fillna(df['feat_0'].median(), inplace=True)`  
   - *Giải thích*: Tìm giá trị nằm ở giữa danh sách (`median`) để lấp vào các ô bị `NULL`, giúp ứng dụng không bị crash khi tính toán.

2. **Đóng gói dữ liệu (`CustomDataset` & `DataLoader`)**:
   - *Logic phần mềm*: Viết một lớp kế thừa từ `torch.utils.data.Dataset` giống như triển khai Interface `IEnumerable` trong C# hoặc `Iterable` trong Java.
   - *Code*:
     ```python
     class BinaryDataset(Dataset):
         def __getitem__(self, idx):
             return self.X[idx], self.y[idx] # Trả về Tuple (Dữ liệu vào, Nhãn thật)
     ```
   - *DataLoader*: Giống như một `Queue` đẩy dữ liệu theo từng Batch (trang) và có cơ chế `shuffle=True` (xáo trộn dữ liệu ngẫu nhiên mỗi Epoch để mô hình học khách quan).

3. **Thiết kế Kiến trúc Mô hình (`BinaryClassifier`)**:
   - *Logic phần mềm*: Dựng một Class gồm các tầng xử lý dữ liệu liên tiếp (`nn.Linear` là phép nhân ma trận $Y = X \cdot W + b$).
   - *Sơ đồ dòng chảy dữ liệu*:  
     `Input (10 số) -> Linear(10 -> 16) -> ReLU() -> Linear(16 -> 1) -> Sigmoid() -> Output (Xác suất 0.0 - 1.0)`
   - *Hàm Sigmoid*: Đóng vai trò nén mọi giá trị đầu ra (dù âm hay dương cực lớn) về khoảng $[0.0, 1.0]$, đại diện cho % nguy cơ gian lận.

4. **Vòng lặp Huấn luyện (Training Loop - 5 bước chuẩn)**:
   ```python
   # Trong mỗi Batch dữ liệu:
   optimizer.zero_grad()           # 1. Clear bộ nhớ cache đạo hàm cũ (Reset state)
   outputs = model(inputs)         # 2. Call API mô hình để lấy dự đoán (Forward pass)
   loss = criterion(outputs, y)    # 3. Tính độ sai lệch/lỗi so với đáp án thật (Loss Value)
   loss.backward()                 # 4. Tính toán mức độ đóng góp lỗi của từng biến (Backpropagation)
   optimizer.step()                # 5. Cập nhật nhẹ trọng số để lần sau bớt sai (Weight Update)
   ```

---

### 🔹 Bài Tập 2: Bài Toán Dự Báo Doanh Thu Cửa Hàng (`Regression Model`)

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Chuỗi cửa hàng bán lẻ cần một API dự đoán **Doanh thu bằng tiền (USD)** trong tháng tới dựa trên các chỉ số như Diện tích mặt bằng, Số lượng nhân viên, Chi phí Marketing,... (Đầu ra là một con số thực liên tục bất kỳ, ví dụ $15,450.50\$$, không phải nhãn đúng/sai $0/1$).

#### 🛠️ Các bước xử lý trong Code:
1. **Kiến trúc mạng `Regressor`**:
   - Khác với bài phân loại, tầng cuối cùng **KHÔNG DÙNG hàm kích hoạt (`Sigmoid`)**, để đầu ra tự do nhận mọi giá trị số thực $(-\infty, +\infty)$.
2. **Hàm phạt chênh lệch `nn.MSELoss()` (Mean Squared Error)**:
   - *Công thức*: $\text{Loss} = (\text{Giá trị thực tế} - \text{Giá trị dự đoán})^2$.
   - *Ý nghĩa phần mềm*: Tính bình phương độ lệch tiền mặt. Dự đoán lệch càng xa thì số tiền phạt càng bị nhân lên gấp bội.

---

### 🔹 Bài Tập 3: Quy Trình Dự Đoán Tập Test & Export CSV Nộp Bài (`Inference & Export`)

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Sau khi huấn luyện xong, phòng Kiểm toán gửi cho bạn file `test.csv` gồm 50 giao dịch mới (chỉ có ID, chưa biết gian lận hay không). Bạn phải chạy mô hình và xuất kết quả ra file `submission_ex3.csv` theo đúng định dạng nhà trường/công ty yêu cầu.

#### 🛠️ Các bước xử lý trong Code:
1. **Chuyển mô hình sang chế độ Read-Only**:
   ```python
   model.eval() # Khóa các lớp đặc thù như Dropout / BatchNorm
   with torch.no_grad(): # Tắt bộ nhớ lưu trữ đạo hàm, giúp giảm 50% RAM và tăng tốc xử lý gấp 2 lần
       outputs = model(X_test)
   ```
2. **Chuyển xác suất thành nhãn Quyết định nhị phân**:
   - Nếu xác suất $\ge 0.5 \rightarrow$ Đánh dấu `1` (Gian lận).
   - Nếu xác suất $< 0.5 \rightarrow$ Đánh dấu `0` (Hợp lệ).
3. **Đóng gói file nộp bài chuẩn quy cách phần mềm**:
   - Sử dụng `pandas.DataFrame.to_csv('submission_ex3.csv', index=False, encoding='utf-8')`.
   - `index=False`: Bỏ cột STT tự động của pandas để không bị lệch định dạng file yêu cầu.

---

## 📓 BÀI TẬP FILE 02: `02_PyTorch_Pipeline_Exercises.ipynb`

### 🔹 Bài Tập: Phân Loại Mức Độ Ưu Tiên Thẻ Hỗ Trợ Kỹ Thuật (`Multi-Class Classification & Validation`)

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Hệ thống Customer Support tự động phân loại các Ticket gửi về thành 3 mức độ:
- `Nhãn 0`: Ưu tiên Thấp (Hỏi đáp thông thường)
- `Nhãn 1`: Ưu tiên Trung bình (Lỗi giao diện/tính năng nhẹ)
- `Nhãn 2`: Ưu tiên Khẩn cấp (Sập Server / Mất dữ liệu)

#### 🛠️ Các bước xử lý trong Code:
1. **Chia dữ liệu Train / Validation (`80 / 20`)**:
   - *Logic phần mềm*: Giống như việc bạn chia dữ liệu thành 2 tập: Tập dùng để Code/Fix bug (Train) và tập dùng để QA/Tester chạy Unit Test độc lập (Validation) nhằm đảm bảo mô hình không bị "học vẹt" (Overfitting).
   - *Tham số `stratify=y`*: Đảm bảo tỷ lệ 3 loại ticket ở cả 2 tập Train và Validation là y hệt nhau (tránh trường hợp tập Test toàn ticket loại 2).
2. **Kiến trúc mạng `MultiClassClassifier`**:
   - Đầu ra có **3 nodes** (tương ứng với 3 lớp nhãn $0, 1, 2$).
3. **Hàm phạt `nn.CrossEntropyLoss()`**:
   - Tự động kết hợp hàm Softmax bên trong để quy đổi 3 đầu ra thành tổng xác suất bằng $100\%$ và phạt nặng lớp dự đoán sai.
4. **Trích xuất nhãn dự đoán cao nhất (`torch.max`)**:
   - *Code*: `_, predicted_classes = torch.max(outputs, dim=1)`
   - *Ý nghĩa*: Lấy ra chỉ số (Index $0, 1$ hoặc $2$) của Node có điểm số cao nhất làm kết quả dự đoán cuối cùng.
