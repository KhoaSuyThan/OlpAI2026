# 📘 HƯỚNG DẪN & GIẢI THÍCH CHI TIẾT GIAI ĐOẠN 2 (NEURAL MACHINE TRANSLATION & ATTENTION)
> 💡 **Dành riêng cho Lập trình viên Phần mềm (Software Engineers):** Tài liệu này dịch toàn bộ khái niệm AI về Dịch máy Chuỗi-sang-Chuỗi (Seq2Seq) và Các Cơ chế Attention sang các thuật ngữ Lập trình & Kiến trúc Phần mềm quen thuộc (Serialization, Payload Bottleneck, Dynamic Key-Value Lookup, Mock Data Injection, Automated Testing) để bạn dễ dàng nắm bắt nhất!

---

## 💻 TỪ KHÓA BẢN CHẤT: AI VS LẬP TRÌNH TRUYỀN THỐNG

| Khái niệm AI / PyTorch | Tương đương trong Lập trình Phần mềm | Giải thích bản chất |
| :--- | :--- | :--- |
| **Seq2Seq (Sequence-to-Sequence)** | **Stream Serialization & Deserialization Pipeline** | Mô hình nhận một chuỗi biến đổi (dữ liệu vào) nén thành Buffer và giải mã Buffer đó ra một chuỗi kết quả mới (dữ liệu ra). |
| **Context Vector (No Attention)** | **Fixed-size Payload Bottleneck** | Ép toàn bộ câu nguồn dài bất kỳ vào một Struct cố định $512$ floats $\rightarrow$ Gây mất mát dữ liệu nghiêm trọng ở câu dài. |
| **Bahdanau Attention (Additive)** | **Pre-Step Dynamic Key-Value Lookup** | Decoder thực hiện truy vấn (Query) tới toàn bộ bản ghi Encoder *trước* khi đưa vào bước LSTM tiếp theo, sử dụng $s_{t-1}$ và phép cộng $\tanh(W_s s_{t-1} + W_h h_j)$. |
| **Luong Attention (Multiplicative)** | **Post-Step Fast Matrix Product Lookup** | Decoder đi qua bước LSTM trước để tạo query $h_t$, sau đó thực hiện nhân ma trận $h_t^T W_a h_j$ cho tốc độ tính toán GPU cực nhanh. |
| **Padding Mask (`source_mask`)** | **Null / Sentinel Value Filtering** | Đảm bảo các phần tử vô nghĩa (`<PAD>`) nhận điểm $-\infty$ (hoặc `min_float`) để Softmax ép giá trị chú ý về đúng $0.0$. |
| **Teacher Forcing** | **Mock Data Injection / Golden Reference Debugging** | Trong lúc huấn luyện (Unit Test), bơm dữ liệu chuẩn (Golden Data) vào bước tiếp theo để tránh trôi luồng xử lý do lỗi tích lũy. |
| **Dynamic Padding (`make_collate_fn`)** | **Dynamic Batch Allocation** | Chỉ đệm token `<PAD>` đến độ dài lớn nhất trong chính Batch đó (thay vì cố định `MAX_LEN`), tiết kiệm VRAM và nhân ma trận gấp 2-3 lần. |
| **`pack_padded_sequence`** | **Skip-Null Optimization** | Bỏ qua toàn bộ các vị trí `<PAD>` trong phép tính LSTM của Encoder, tối ưu GPU. |
| **Greedy Decoding** | **First-Fit Search / Greedy Algorithm** | Chọn token có xác suất cao nhất ($\arg\max$) tại thời điểm hiện tại mà không quay lui (Backtracking). |
| **SacreBLEU Metric** | **Automated Integration Test Assertion Score** | Đo lường mức độ trùng khớp giữa chuỗi kết quả sinh ra và chuỗi kỳ vọng (Expected Output) theo chuẩn quốc tế (`tokenize='none'`). |

---

## 📓 BÀI TẬP FILE 01: `01_en_vi_seq2seq_no_attention.ipynb`

### 🔹 Bài Toán Dịch Máy Anh ↔ Việt Bằng Seq2Seq LSTM (Không Attention)

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Bạn làm tại một công ty công nghệ đa quốc gia, cần xây dựng Service dịch tự động phụ đề phim hai chiều giữa tiếng Anh và tiếng Việt. Dữ liệu huấn luyện là tập câu song song IWSLT'15 ($133.317$ cặp câu). Yêu cầu dịch tự động chính xác và không bị rò rỉ dữ liệu giữa môi trường dev và sản phẩm.

#### 🛠️ Các bước xử lý trong Code:

1. **Tiền xử lý & Làm sạch dữ liệu song song (Parallel Data Cleaning)**:
   - *Logic phần mềm*: Chuẩn hóa chuỗi UTF-8 về NFC, loại bỏ HTML entities (`&amp;` $\rightarrow$ `&`).
   - *Lọc dữ liệu*: Loại bỏ câu rỗng, câu quá dài ($> 50$ từ) và cặp câu có tỷ lệ độ dài lệch quá 3 lần ($\frac{\text{len}(src)}{\text{len}(tgt)} > 3.0$ hoặc $< 0.33$).

2. **Chống Rò Rỉ Dữ Liệu (Data Leakage Prevention)**:
   - *Logic phần mềm*: Đảm bảo dữ liệu test không bị lộ vào tập train (tương tự không lộ đáp án test vào code logic).
   - *Code*: Hàm `remove_overlap()` loại bỏ toàn bộ cặp câu trong tập Validation/Test nếu chuỗi nguồn hoặc đích đã từng xuất hiện trong tập Train.
   - *Vocabulary*: Chỉ tạo từ tập **Train** (`Vocabulary.from_dataframe(train_df)`).

3. **Đóng gói Dữ liệu & Dynamic Padding (`ParallelTranslationDataset` & `collate_fn`)**:
   - *Code*:
     ```python
     def make_collate_fn(source_pad_id, target_pad_id):
         def collate_fn(batch):
             # Tìm max length trong BATCH hiện tại
             max_src = max(len(x['source']) for x in batch)
             max_tgt = max(len(x['target']) for x in batch)
             # Pad vừa đủ max length của batch đó
             ...
     ```

4. **Kiến trúc Mô hình Seq2Seq Không Attention (`Seq2SeqLSTM`)**:
   - **Encoder (`EncoderLSTM`)**: Nhận câu nguồn `[B, S]`, đi qua `nn.Embedding` và `nn.LSTM`, xuất ra trạng thái ẩn cuối cùng `(h_n, c_n)` có kích thước `[layers, B, H]`.
   - **Context Vector Bottleneck**: Toàn bộ nội dung câu nguồn $S$ từ bị ép vào vector cố định `(h_n, c_n)`. Khi câu dài $> 25$ từ, thông tin đầu câu bị quá tải và quên dần.
   - **Decoder (`DecoderLSTM`)**: Nhận `(h_n, c_n)` làm khởi tạo, giải mã từng bước tự hồi quy:
     $$\text{Input: } y_{t-1} \longrightarrow \text{LSTM} \longrightarrow \text{Linear} \longrightarrow \text{Logits [B, V_{tgt}]}$$

5. **Kỹ thuật Huấn luyện & Greedy Decoding**:
   - **Teacher Forcing ($p_{tf} = 0.5$)**: Với xác suất 50%, dùng từ đúng trong nhãn ($y_{t}^{true}$) làm đầu vào cho bước $t+1$.
   - **Hàm Mất Mát**: `nn.CrossEntropyLoss(ignore_index=target_pad_id)` không tính loss cho token `<PAD>`.
   - **Greedy Decoding**: Khi inference (`model.eval()`), bắt đầu bằng `<SOS>`, tại mỗi bước chọn `argmax(logits)` cho đến khi gặp `<EOS>`.

---

## 📓 BÀI TẬP FILE 02: `02_en_vi_seq2seq_bahdanau_attention.ipynb`

### 🔹 Bài Toán Dịch Máy Nâng Cấp Với Bahdanau Additive Attention

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Mô hình ở File 01 dịch rất tốt câu ngắn ($\le 15$ từ), nhưng khi dịch các đoạn văn dài ($\ge 30$ từ), câu dịch bị mất ý nghiêm trọng do nút thắt Bottleneck. Bạn cần nâng cấp hệ thống với **Bahdanau Attention** để Decoder có thể "tập trung nhìn lại" đúng vị trí từ nguồn cần dịch tại mỗi thời điểm.

#### 🛠️ Các bước xử lý trong Code:

1. **Công Thức Toán Học Bahdanau Additive Attention**:
   - Điểm Alignment Score:
     $$e_{t, j} = v_a^T \tanh(W_s s_{t-1} + W_h h_j + b_a)$$
   - Trong đó $h_j$ là tất cả các trạng thái ẩn Encoder `[B, S, H]`, $s_{t-1}$ là trạng thái Decoder bước trước `[B, H]`.

2. **Kỹ thuật Padding Masking ($-\infty$ fill)**:
   - *Logic phần mềm*: Loại bỏ nhiễu từ các ô đệm `<PAD>`.
   - *Code*:
     ```python
     scores = scores.masked_fill(~source_mask, -1e9)
     attention_weights = F.softmax(scores, dim=1)
     ```
   - *Kết quả*: Vì $\exp(-1e9) \approx 0$, trọng số $\alpha_{t, j}$ tại các vị trí `<PAD>` bằng chính xác $0.0$.

3. **Kiến trúc PyTorch (`BahdanauAttention` & `BahdanauDecoderLSTM`)**:
   - `BahdanauAttention`: Tạo Context Vector động $c_t = \sum \alpha_{t, j} h_j$ có shape `[B, 1, H]`.
   - `BahdanauDecoderLSTM`: Nối $c_t$ với embedding từ đầu vào $y_{t-1}$ đưa qua LSTM. Sau đó nối 3 vector `[output; context; embedded]` chiếu qua `Linear` để dự đoán từ tiếp theo.

4. **Kiểm Tra Masking & Length Benchmark**:
   - **`verify_attention_padding_mask()`**: Xác nhận tổng trọng số chú ý tại mọi vị trí đệm `<PAD>` bằng $0.0000$.
   - **Length Benchmark Subsets**: Đánh giá SacreBLEU trên 3 nhóm độ dài ($\le 15$, $16-29$, $\ge 30$). Kết quả cho thấy Bahdanau Attention giữ vững phong độ ở câu dài $\ge 30$ từ.
   - **Attention Heatmap (`plot_attention_heatmap`)**: Vẽ ma trận căn chỉnh từ giữa ngôn ngữ nguồn và đích.

---

## 📓 BÀI TẬP FILE 03: `03_en_vi_seq2seq_luong_attention.ipynb`

### 🔹 Bài Toán Tối Ưu Tốc Độ & Hiệu Năng Với Luong Multiplicative Attention

#### 🏢 Bối cảnh thực tế (Real-world Scenario):
Hệ thống dịch máy của bạn cần phục vụ hàng triệu request mỗi ngày. Mặc dù Bahdanau Attention dịch chính xác, phép tính cộng $\tanh(W_s s_{t-1} + W_h h_j)$ tốn khá nhiều thời gian nhân ma trận trên GPU. Bạn tiến hành nâng cấp sang **Luong Multiplicative General Attention** để tối ưu hóa tốc độ xử lý GPU mà vẫn giữ điểm BLEU cao.

#### 🛠️ Các bước xử lý trong Code:

1. **Công Thức Toán Học Luong Multiplicative General Attention**:
   - Điểm General Score:
     $$e_{t, j} = h_t^T W_a h_j$$
   - Phép nhân ma trận trực tiếp $h_t^T W_a h_j$ được thực hiện cực nhanh bằng toán tử Batch Matrix Multiplication (`torch.bmm`).

2. **Khác biệt về Thời Điểm Tính Attention (Post-LSTM)**:
   - *Bahdanau*: Tính Attention **TRƯỚC** khi qua LSTM (dùng $s_{t-1}$).
   - *Luong*: Tính Attention **SAU** khi qua LSTM (dùng trạng thái ẩn hiện tại $h_t$ làm Query).

3. **Kỹ Thuật `pack_padded_sequence` Trong Encoder**:
   - *Code*:
     ```python
     packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
     packed_outputs, (hidden, cell) = self.lstm(packed)
     encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=source.size(1))
     ```
   - *Tác dụng*: Giúp GPU bỏ qua hoàn toàn các bước tính toán lãng phí trên token `<PAD>` trong Encoder.

4. **Kiến trúc PyTorch (`LuongGeneralAttention` & `LuongDecoderLSTM`)**:
   - `LuongDecoderLSTM`: Cho từ qua LSTM trước để lấy $h_t$, tính $c_t$ qua `LuongGeneralAttention`, sau đó tạo vector Attentional Hidden State:
     $$\tilde{h}_t = \tanh(W_c [h_t; c_t])$$
     $$\text{Logits} = \text{Output\_Projection}(\tilde{h}_t)$$
   - **Khởi tạo trọng số Xavier Uniform (`initialize_model`)**: Khởi tạo trọng số ma trận chuẩn hóa Xavier Uniform và ép trọng số embedding của token `<PAD>` về đúng $0$.

---

## 📊 BẢNG SO SÁNH TỔNG HỢP TOÀN DIỆN 3 MÔ HÌNH TUẦN 2

| Tiêu chí | Notebook 01 (No Attention) | Notebook 02 (Bahdanau Attention) | Notebook 03 (Luong Attention) |
| :--- | :--- | :--- | :--- |
| **Loại Score Function** | Không có | **Additive**: $v_a^T \tanh(W_s s_{t-1} + W_h h_j)$ | **Multiplicative (General)**: $h_t^T W_a h_j$ |
| **Thời điểm tính Attention** | N/A | **Trước** khi đi qua Decoder LSTM | **Sau** khi đi qua Decoder LSTM |
| **Query Vector** | N/A | Trạng thái bước trước $s_{t-1}$ | Trạng thái hiện tại $h_t$ |
| **Tốc độ ma trận GPU** | Nhanh ($O(S+T)$) | Trung bình (có phép cộng $\tanh$) | Rất nhanh (nhân ma trận `bmm` tối ưu) |
| **Encoder Optimization** | Standard Padding LSTM | Standard Padding LSTM | Tích hợp `pack_padded_sequence` |
| **Dịch câu dài ($\ge 30$)** | Suy giảm điểm BLEU nghiêm trọng | Rất tốt (Duy trì ngữ cảnh) | Rất tốt (Duy trì ngữ cảnh & chính xác) |
| **Khả năng hiển thị Heatmap** | Hộp đen (Black Box) | Hiển thị ma trận căn chỉnh từ | Hiển thị ma trận căn chỉnh từ |

---

## 🎯 CHECKLIST CẦN NHỚ CHO KỲ THI OLPAI 2026

1. **Cố định seed**: Luôn gọi `seed_everything(42)` ngay cell đầu tiên.
2. **Chống rò rỉ dữ liệu**: `Vocabulary` chỉ tạo từ tập Train, tuyệt đối không fit trên Val/Test. Lọc sạch overlap dữ liệu.
3. **Dynamic Padding**: Sử dụng `collate_fn` đệm động theo batch để tiết kiệm bộ nhớ GPU.
4. **Masking `<PAD>`**: Trong Attention phải luôn điền $-\infty$ (hoặc `min_float`) vào vị trí `<PAD>` trước Softmax. Dùng `ignore_index=pad_id` trong `CrossEntropyLoss`.
5. **Evaluation Metric**: Sử dụng `sacrebleu` với `tokenize='none'` chuẩn quốc tế.

---
