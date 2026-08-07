# Hướng Dẫn & Giải Thích Chi Tiết Từng Cell: Notebook 01 - Seq2Seq LSTM (Không Attention)

Tài liệu này phân tích toàn bộ **23 phần (48 Cells)** trong notebook `01_en_vi_seq2seq_no_attention.ipynb`. Mọi đoạn mã PyTorch, công thức toán học, hình dạng Tensor (Tensor Shapes) và logic xử lý dữ liệu đều được giải thích sâu.

---

## 📍 PHẦN 1: TỔNG QUAN KIẾN TRÚC & TENSOR SHAPES

Mô hình Seq2Seq cổ điển (Sutskever et al., 2014) gồm 2 thành phần chính: **Encoder** và **Decoder**.

```
Source Sequence: [x_1, x_2, ..., x_S] 
   │
   ▼
[Encoder LSTM] ──> Final Hidden State (h_n, c_n) [Context Vector - Bottleneck]
                                                 │
                                                 ▼
[Decoder LSTM] <── Start Token <SOS> ────────────┘
   │
   ▼
Target Sequence: [y_1, y_2, ..., y_T]
```

### Bảng Phân Tích Hình Dạng Tensor (Tensor Shape Flow):
| Bước | Ký hiệu | Hình dạng Tensor (Shape) | Giải thích |
| :--- | :--- | :--- | :--- |
| **Source Input** | `source` | `[B, S]` | `B`: Batch size, `S`: Source sequence length |
| **Encoder Embed** | `embedded` | `[B, S, E_src]` | `E_src`: Source embedding dimension |
| **Encoder Output** | `outputs` | `[B, S, H]` | `H`: Hidden dimension |
| **Context Vector** | `(h_n, c_n)` | `[num_layers, B, H]` | Trạng thái ẩn & ô cuối cùng của Encoder |
| **Decoder Input** | `input_token` | `[B]` (hoặc `[B, 1]`) | Token tại bước $t$ (bắt đầu bằng `<SOS>`) |
| **Decoder Logits** | `logits` | `[B, V_tgt]` | `V_tgt`: Target vocabulary size |

---

## 📍 PHẦN 2: GIẢI THÍCH CHI TIẾT TỪNG MỤC CELL (CELL-BY-CELL BREAKDOWN)

### Mục 1 & 2: Cài Đặt Thư Viện & Imports (Cells 1 – 4)
- **Cell 1–2**: Tự động kiểm tra và cài đặt `sacrebleu` (thư viện chuẩn đo lường điểm BLEU quốc tế) và `underthesea` / `pyvi` (tách từ tiếng Việt).
- **Cell 3–4**: Import các module cốt lõi:
  - `torch`, `torch.nn`, `torch.optim`, `Dataset`, `DataLoader`.
  - `sacrebleu.metrics.BLEU`: Đo lường điểm BLEU không phụ thuộc tokenizer nội cục (`tokenize='none'`).

---

### Mục 3 & 4: Cấu Hình Thí Nghiệm & Seed (Cells 5 – 8)
- **Cell 5–6 (Hyperparameters)**:
  - `QUICK_RUN = False`: Đặt `False` để chạy toàn bộ dataset (cho kết quả BLEU cao). Khi `True`, chỉ lấy $10.000$ cặp câu để test nhanh code.
  - `MAX_SENTENCE_TOKENS = 50`: Giới hạn độ dài tối đa của câu. Câu dài hơn sẽ bị lọc bỏ để giữ ổn định VRAM.
  - `MAX_VOCAB_SIZE = 12_000`: Giới hạn kích thước từ vựng tối đa cho mỗi ngôn ngữ.
  - `EMBEDDING_DIM = 256`, `HIDDEN_DIM = 512`, `NUM_LAYERS = 2`, `DROPOUT = 0.3`.
  - `TEACHER_FORCING_RATIO = 0.5`: Xác suất 50% dùng từ thật làm đầu vào Decoder trong khi huấn luyện.
- **Cell 7–8 (Seed Reproducibility)**:
  - Hàm `set_seed(42)` cố định seed cho `random`, `numpy`, `torch.manual_seed()` và `torch.cuda.manual_seed_all()`, đồng thời bật `torch.backends.cudnn.deterministic = True`.

---

### Mục 5 & 6: Nạp Dữ Liệu & Kiểm Tra Tệp Song Song (Cells 9 – 13)
- **Cell 9–10**: Nạp dataset IWSLT'15 (English-Vietnamese Parallel Corpus). Tự động tìm kiếm file địa phương hoặc tải từ mirror uy tín.
- **Cell 11–13**: Đếm số dòng (`count_lines`) và kiểm tra đọc UTF-8. Xác nhận tập Train có $133.317$ cặp câu, Validation có $1.553$ cặp câu và Test có $1.268$ cặp câu.

---

### Mục 7 – 10: Xử Lý Dữ Liệu & Chống Rò Rỉ (Data Leakage) (Cells 14 – 21)
- **Cell 14–15 (Normalization & Tokenization)**:
  - `normalize_text()`: Chuẩn hóa NFC, unescape ký tự HTML (`&amp;` $\rightarrow$ `&`), chuyển chữ thường (`lowercase=True`).
  - `tokenize_english()` / `tokenize_vietnamese()`: Tách từ tiếng Anh bằng Regex / NLTK và tiếng Việt bằng PyVi / Regex.
- **Cell 16–17 (Làm sạch cặp câu)**:
  - Loại bỏ các cặp câu bị rỗng sau khi tách từ.
  - Lọc câu có độ dài $> 50$ tokens.
  - Lọc các cặp câu có tỷ lệ độ dài bất thường: $\frac{\text{len}(src)}{\text{len}(tgt)} > 3.0$ hoặc $< 0.33$.
- **Cell 18–19 (Chống Rò Rỉ Dữ Liệu - Data Leakage Prevention)**:
  - Hàm `remove_overlap()` kiểm tra và xóa toàn bộ các cặp câu trong tập Validation/Test nếu chuỗi nguồn hoặc chuỗi đích đã xuất hiện trong tập Train.
- **Cell 20–21**: Thống kê số lượng câu và tỷ lệ trung bình độ dài sau khi làm sạch.

---

### Mục 11: Tạo Từ Vựng (Vocabulary) Chỉ Từ Tập Train (Cells 22 – 24)
- **Cell 23 (`class Vocabulary`)**:
  - Xây dựng bảng từ vựng với các token đặc biệt bắt buộc:
    - `<PAD>` (id = 0): Token đệm cho câu ngắn.
    - `<UNK>` (id = 1): Token cho từ không xuất hiện trong từ vựng.
    - `<SOS>` (id = 2): Token bắt đầu câu (Start of Sentence).
    - `<EOS>` (id = 3): Token kết thúc câu (End of Sentence).
  - Chỉ đếm tần suất từ trên tập **Train** và giữ lại `MAX_VOCAB_SIZE - 4` từ phổ biến nhất.
- **Cell 24**: Kiểm tra tỷ lệ từ xuất hiện dạng `<UNK>` trên tập Validation và Test.

---

### Mục 12 & 13: PyTorch Dataset, Collate Function & DataLoader (Cells 25 – 28)
- **Cell 26 (`ParallelTranslationDataset` & `make_collate_fn`)**:
  - `ParallelTranslationDataset`: Chuyển chuỗi từ thành danh sách token ID, tự động chèn `<SOS>` ở đầu và `<EOS>` ở cuối câu đích.
  - `make_collate_fn()` **(Dynamic Padding)**:
    - Không đệm cố định đến `MAX_SENTENCE_TOKENS`.
    - Tìm độ dài lớn nhất `max_src_len` và `max_tgt_len` **trong chính batch đó**, dùng `torch.nn.utils.rnn.pad_sequence` để đệm token `<PAD>`.
    - Kỹ thuật này giúp giảm bớt VRAM tiêu tốn và tăng tốc độ nhân ma trận GPU gấp 2-3 lần.
- **Cell 27–28**: Kiểm tra trực tiếp 1 batch đầu ra từ `DataLoader` để xác nhận hình dạng tensor `[B, S]` chính xác.

---

### Mục 14: Mã Nguồn Các Lớp Mô Hình (Encoder, Decoder, Seq2Seq) (Cells 29 – 30)

#### 1. Lớp `EncoderLSTM`
```python
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, source_lengths):
        # source: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(source)) # [batch_size, seq_len, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden, cell: [num_layers, batch_size, hidden_dim]
        return outputs, hidden, cell
```

#### 2. Lớp `DecoderLSTM`
```python
class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch_size] -> unsqueeze thành [batch_size, 1]
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token)) # [batch_size, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1)) # [batch_size, output_dim]
        return prediction, hidden, cell
```

#### 3. Lớp `Seq2SeqLSTM`
- Kết nối Encoder và Decoder.
- Trong hàm `forward()`, duyệt qua từng bước thời gian $t = 1 \dots T_{target}-1$.
- Áp dụng Teacher Forcing:
  ```python
  teacher_force = random.random() < teacher_forcing_ratio
  top1 = prediction.argmax(1)
  input_token = target[:, t] if teacher_force else top1
  ```

---

### Mục 15 & 16: Vòng Lặp Huấn Luyện & Train Hai Chiều (Cells 31 – 34)
- **Cell 31–32**:
  - Optimizer: `optim.Adam(model.parameters(), lr=1e-3)`.
  - Loss Function: `nn.CrossEntropyLoss(ignore_index=target_pad_id)`.
  - Gradient Clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` chống bùng nổ gradient (Gradient Explosion).
- **Cell 33–34**: Chạy huấn luyện cả 2 chiều:
  1. English $\rightarrow$ Vietnamese
  2. Vietnamese $\rightarrow$ English

---

### Mục 17 & 18: Đồ Thị Loss & Greedy Decoding (Cells 35 – 38)
- **Cell 35–36**: Trực quan hóa đường cong Train Loss và Validation Loss.
- **Cell 37–38 (`greedy_decode_batch`)**:
  - Sử dụng `@torch.inference_mode()` tối ưu tốc độ.
  - decoder bắt đầu bằng `<SOS>`. Tại mỗi bước, chọn `next_token = logits.argmax(dim=-1)`.
  - Dừng giải mã khi tất cả các câu trong batch đều gặp `<EOS>` hoặc đạt `MAX_DECODE_LENGTH`.

---

### Mục 19 – 23: Đánh Giá SacreBLEU & Dịch Thử (Cells 39 – 47)
- **Cell 39–40**: Tính toán điểm SacreBLEU chính thức trên tập Test.
- **Cell 41–42**: Lựa chọn 10 câu test hoàn toàn không xuất hiện trong tập Train (`unseen_test_mask`), in ra câu gốc (Source), câu chuẩn (Reference) và câu mô hình dịch (Hypothesis).
- **Cell 43–44**: Hàm `translate_sentence()` cho phép người dùng nhập một câu văn bản bất kỳ để dịch thử.
- **Cell 45–46**: Dịch thử danh sách câu tự tạo ngoài thực tế.
- **Cell 47**: Kết luận bài học và tổng kết điểm BLEU.

---
