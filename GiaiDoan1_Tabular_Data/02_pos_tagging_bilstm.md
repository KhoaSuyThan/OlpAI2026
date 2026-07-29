# Hướng dẫn & Phân tích Bài tập 02: Part-of-Speech (POS) Tagging với Bidirectional LSTM (BiLSTM)

Tài liệu này phân tích chi tiết toàn bộ luồng xử lý (pipeline), kiến trúc mô hình và kết quả thực nghiệm trong notebook `02_pos_tagging_bilstm.ipynb`. Bài tập tập trung vào bài toán **Gắn nhãn từ loại (Part-of-Speech Tagging - POS Tagging)** trên tập dữ liệu Kaggle NER Dataset, so sánh hiệu năng giữa **Unidirectional LSTM** và **Bidirectional LSTM (BiLSTM)**.

---

## 1. Tổng quan Bài toán & Mục tiêu Bài học

### 1.1. Bài toán Gắn nhãn từ loại (POS Tagging / Many-to-Many)
* **Đầu vào (Input):** Chuỗi các từ (tokens) trong một câu hoàn chỉnh $X = (x_1, x_2, \dots, x_T)$.
* **Đầu ra (Output):** Chuỗi các nhãn từ loại tương ứng $Y = (y_1, y_2, \dots, y_T)$ có độ dài **đúng bằng** độ dài câu.
* **Tập nhãn POS (42 nhãn):** Bao gồm các nhãn ngữ pháp chuẩn như `NN` (Noun), `NNP` (Proper Noun), `VB` (Verb), `VBD` (Verb Past), `JJ` (Adjective), `DT` (Determiner), `IN` (Preposition), v.v.
* **Kiến trúc NLP cơ bản:** Phân loại chuỗi Nhiều-sang-Nhiều (Sequence Labelling / Token-level Classification).

```
Token 1 (Thousands)     --> [ BiLSTM Cell ] --> Logits_1 --> POS: NNS
Token 2 (of)            --> [ BiLSTM Cell ] --> Logits_2 --> POS: IN
Token 3 (demonstrators) --> [ BiLSTM Cell ] --> Logits_3 --> POS: NNS
...
Token T (.)             --> [ BiLSTM Cell ] --> Logits_T --> POS: .
```

---

## 2. Quy trình Xử lý Dữ liệu (Data Pipeline & Preprocessing)

Pipeline dữ liệu được xây dựng chặt chẽ để đảm bảo không bị rò rỉ dữ liệu (Data Leakage) giữa các tập Train/Validation/Test.

```mermaid
graph TD
    A[Tải dữ liệu Kaggle NER Dataset] --> B[Forward-fill sentence_id & Clean text]
    B --> C[Gộp dòng theo CÂU HOÀN CHỈNH - groupby]
    C --> D[Chia Train / Validation / Test theo CÂU]
    D --> E[Tạo Word Vocab (Train) & POS Mapping (Corpus)]
    E --> F[PyTorch Dataset & Simultaneous Pad Batch]
    F --> G[DataLoader nạp đồng thời Word IDs & POS IDs]
```

### 2.1. Tiền xử lý Dữ liệu thô & Forward-Fill
Dữ liệu thô có cấu trúc dạng bảng 1 dòng/1 token, trong đó `sentence_id` chỉ xuất hiện ở token đầu tiên của câu.
* **Làm đầy `sentence_id`:** Sử dụng `selected_df["sentence_id"].ffill()` để gán ID câu cho tất cả các token thuộc câu đó.
* **Chuẩn hóa Unicode:** Áp dụng chuẩn hóa Unicode NFC (`unicodedata.normalize("NFC", value)`) và loại bỏ khoảng trắng thừa.

### 2.2. Nhóm dữ liệu theo Câu hoàn chỉnh (`groupby`)
* Các dòng token được nhóm lại theo `sentence_id` thành danh sách `tokens` và danh sách `pos_tags`.
* **Tổng số câu hợp lệ:** **47,959 câu** (độ dài trung bình ~21.8 token/câu, câu dài nhất là 104 token).

### 2.3. Chia tập Dữ liệu theo CÂU (Sentence-Level Split)
* **Quy tắc quan trọng:** Phải phân chia ngẫu nhiên trên danh sách **CÂU HOÀN CHỈNH** (Sentence DataFrames), không được chia trên cấp độ dòng token.
* **Tỷ lệ phân chia:**
  * **Train Set:** 33,571 câu (70%)
  * **Validation Set:** 7,194 câu (15%)
  * **Test Set:** 7,194 câu (15%)
* **Đảm bảo tính giao bằng 0 (Disjoint Test Sets):** `train_ids`, `val_ids`, `test_ids` hoàn toàn tách biệt, ngăn ngừa việc token của cùng 1 câu nằm ở cả 2 tập.

### 2.4. Xây dựng Từ vựng & Mapping
* **Word Vocabulary (30,157 từ):** Xây dựng duy nhất từ `train_df`. Chứa 2 token đặc biệt: `<PAD>` (ID `0`), `<UNK>` (ID `1`).
* **POS Vocabulary (43 nhãn bao gồm `<PAD>`):** Xây dựng trên toàn bộ corpus để giữ ánh xạ ID cố định (từ ID `1` đến `42` cho các nhãn POS thực tế, ID `0` dành riêng cho `<PAD>`).

### 2.5. Simultaneous Dynamic Padding (`pad_batch`)
Hàm `pad_batch` thực hiện đệm (padding) đồng thời cho cả **Word Tensor** và **Target POS Tensor** tới độ dài cực đại của câu dài nhất trong Batch (`max_length`):
* `word_batch` có shape `(batch_size, max_length)` điền `WORD_PAD_ID=0`.
* `pos_batch` có shape `(batch_size, max_length)` điền `POS_PAD_ID=0`.
* Trả về thêm Tensor `lengths` ghi lại độ dài thực tế của từng câu để phục vụ `pack_padded_sequence`.

---

## 3. Kiến trúc Mô hình: Unidirectional LSTM vs BiLSTM (`LSTMPOSTagger`)

Mô hình PyTorch chuyển đổi linh hoạt giữa LSTM 1 chiều và BiLSTM qua tham số `bidirectional`:

```
          [ Backward LSTM Layer ]  <--- (Đọc từ phải sang trái)
                     │
Word IDs ---> [ Embedding ] ---> [ Forward LSTM Layer ]  ---> (Đọc từ trái sang phải)
                     │                      │
                     └──────────┬───────────┘
                                ▼
                     [ Concat Hidden States ] (Shape: [B, T, 2 * hidden_dim])
                                │
                                ▼
                     [ Linear Classifier Head ] (2 * hidden_dim -> 43 classes)
                                │
                                ▼
                     [ Logits per Token ] (Shape: [batch_size, max_length, 43])
```

### 3.1. Điểm khác biệt Cốt lõi giữa LSTM và BiLSTM
1. **Unidirectional LSTM (`bidirectional=False`):**
   * Tại vị trí token thứ $t$, mô hình chỉ nhận được thông tin từ quá khứ $x_1, x_2, \dots, x_t$.
   * Dimension của hidden state: `hidden_dim`.
2. **Bidirectional LSTM (`bidirectional=True`):**
   * Bao gồm 2 lớp LSTM chạy song song:
     * **Forward LSTM ($\vec{h}_t$):** Đọc chuỗi từ đầu đến cuối ($x_1 \to x_T$).
     * **Backward LSTM ($\overleftarrow{h}_t$):** Đọc chuỗi từ cuối về đầu ($x_T \to x_1$).
   * State đầu ra tại vị trí $t$ là sự kết hợp (concatenation) của cả 2 chiều: $h_t = [\vec{h}_t ; \overleftarrow{h}_t]$.
   * Dimension của hidden state đầu ra: `hidden_dim * 2`.

### 3.2. Tại sao BiLSTM lý tưởng cho bài toán POS Tagging?
Trong cú pháp ngôn ngữ (Syntax), nhãn từ loại của một từ phụ thuộc rất lớn vào **từ đứng sau nó**:
* *Ví dụ 1:* Từ *"bank"* trong *"river **bank**"* (Danh từ - NN) vs *"**bank** account"* (Tính từ/Danh từ bổ nghĩa).
* *Ví dụ 2:* Từ *"reading"* trong *"I am **reading**"* (Động từ - VBG) vs *"The **reading** material"* (Tính từ/Danh từ bổ nghĩa - JJ/NN).
* LSTM 1 chiều khi đứng tại từ *"bank"* hay *"reading"* chưa biết từ đứng ngay sau nó là gì. BiLSTM nhờ có lớp Backward LSTM đã "nhìn thấy trước" ngữ cảnh tương lai, giúp đưa ra dự đoán chính xác vượt trội.

---

## 4. Kỹ thuật Tính Loss & Metric Loại trừ Padding

### 4.1. `CrossEntropyLoss(ignore_index=POS_PAD_ID)`
* Cài đặt `ignore_index = 0` giúp PyTorch tự động bỏ qua hoàn toàn các token `<PAD>` khi tính toán hàm mất mát (Loss). Gradient sẽ không bị ảnh hưởng bởi các vị trí padding.

### 4.2. Hàm lọc Padding (`flatten_without_padding`)
Để tính các chỉ số đánh giá thực tế, ta sử dụng mask loại bỏ các vị trí padding:
```python
mask = targets.ne(POS_PAD_ID)
y_true = targets[mask].cpu().numpy()
y_pred = predictions[mask].cpu().numpy()
```

### 4.3. Các Chỉ số Đánh giá Cấp độ Token (Token-level Metrics)
* **Token Accuracy:** Tỷ lệ từ loại được dự đoán đúng trên tổng số token thực tế.
* **Macro Precision, Macro Recall, Macro F1:** TÍnh trung bình trên 42 nhãn POS thực tế (từ ID `1` đến `42`), không tính nhãn `<PAD>`.

---

## 5. Kết quả Thực nghiệm & Phân tích Đánh giá

### 5.1. Bảng So sánh Kết quả Đánh giá trên Tập Test (Test Set Evaluation)

Dưới đây là bảng kết quả thực tế thu được sau khi huấn luyện cả 2 mô hình trong notebook với cùng seed và hyperparameters:

| Mô hình (Model) | Test Loss | Token Accuracy | Macro Precision | Macro Recall | **Macro F1-score** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unidirectional LSTM** | 0.1219 | 96.35% | 87.37% | 87.23% | **87.22%** |
| **Bidirectional LSTM (BiLSTM)** | **0.0953** | **97.26%** | **94.08%** | **91.90%** | **92.49%** |
| **Mức độ Cải thiện ($\Delta$)** | **-0.0266** | **+0.91%** | **+6.71%** | **+4.67%** | **+5.27%** |

### 5.2. Phân tích Chi tiết Báo cáo Phân loại (Classification Report)
* **Token Accuracy cao (96.35% vs 97.26%):** Do các từ loại phổ biến (`NN`, `NNP`, `IN`, `DT`, `.`, `,`) chiếm số lượng rất lớn và cả 2 mô hình đều nhận diện rất tốt (>98%).
* **Macro F1 tăng vượt bậc từ 87.22% lên 92.49% (+5.27%):** Sự khác biệt lớn nhất nằm ở các nhãn từ loại hiếm hoặc phụ thuộc mạnh vào ngữ cảnh phía sau:
  * Nhãn `:` (Colon/Punctuation): LSTM đạt F1 = 59.57% $\to$ BiLSTM đạt **83.53%** (+23.96%).
  * Nhãn `EX` (Existential there): LSTM đạt F1 = 91.75% $\to$ BiLSTM đạt **96.00%** (+4.25%).
  * Nhãn `RBR` (Adverb, comparative): LSTM đạt F1 = 66.42% $\to$ BiLSTM đạt **80.00%** (+13.58%).
  * Nhãn `WDT` (Wh-determiner): LSTM đạt F1 = 88.08% $\to$ BiLSTM đạt **96.23%** (+8.15%).

---

## 6. Đáp án Bài tập & Hướng phát triển Cải tiến

#### **Bài tập 1: Tại sao BiLSTM lại đạt hiệu năng cao hơn hẳn LSTM 1 chiều cho bài toán POS Tagging?**
* **Trả lời:** POS Tagging là bài toán gán nhãn phụ thuộc ngữ cảnh hai chiều. Một từ loại bị chi phối bởi cả từ đứng trước (ngữ pháp quá khứ) và từ đứng sau (ngữ pháp tương lai). BiLSTM kết hợp vector trạng thái của cả Forward LSTM và Backward LSTM tại mỗi vị trí token, cung cấp bức tranh ngữ cảnh toàn diện 360 độ cho lớp phân loại Linear.

#### **Bài tập 2: Tại sao phải tính Macro F1 bên cạnh Token Accuracy?**
* **Trả lời:** Trong dữ liệu POS Tagging, tần suất các nhãn bị lệch rất lớn (Class Imbalance). Các nhãn như `NN` (145k token), `NNP` (131k token) chiếm số lượng áp đảo, trong khi các nhãn như `PDT` (147 token), `WP$` (99 token), `UH` (24 token) rất hiếm. 
* Nếu chỉ nhìn vào Token Accuracy, một mô hình đoán đúng các lớp phổ biến sẽ có accuracy cao nhưng lại bỏ sót các từ loại quan trọng khác. Macro F1 tính trung bình F1 của từng lớp riêng biệt, phản ánh chính xác khả năng nhận diện các từ loại hiếm.

#### **Bài tập 3: Các phương pháp nâng cấp mô hình POS Tagging trong thực tế?**
1. **BiLSTM-CRF (Conditional Random Field):** Thay lớp Linear head bằng lớp CRF. Lớp CRF giúp học mối quan hệ chuyển giao giữa các nhãn kề nhau (ví dụ: sau Tính từ `JJ` thường là Danh từ `NN`, không bao giờ là Động từ chia thì `VBD` trực tiếp).
2. **Character-level Embeddings (BiLSTM char-level / CNN char-level):** Trích xuất thông tin hình thái từ (Prefix/Suffix như *-ing*, *-ed*, *-tion*, *-ly*) giúp dự đoán chính xác POS tag của các từ chưa từng xuất hiện trong từ vựng (Out-of-Vocabulary - OOV).
3. **Pretrained Language Models (BERT / RoBERTa):** Sử dụng các mô hình Transformer đã qua tiền huấn luyện để trích xuất Contextual Embeddings cho từng token.
