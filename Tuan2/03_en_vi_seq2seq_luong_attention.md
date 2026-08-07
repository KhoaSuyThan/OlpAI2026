# Hướng Dẫn & Giải Thích Chi Tiết Từng Cell: Notebook 03 - Seq2Seq với Luong Attention (Bài 03.3)

Tài liệu này phân tích toàn bộ **27 phần (55 Cells)** trong notebook `03_en_vi_seq2seq_luong_attention.ipynb`. Bài tập hướng dẫn xây dựng mô hình Dịch máy hai chiều Anh ↔ Việt với **Cơ chế Chú ý Nhân (Luong Multiplicative / General Attention Mechanism - Luong et al., 2015)**.

---

## 📍 PHẦN 1: CƠ CHẾ TOÁN HỌC & TENSOR SHAPE FLOW CỦA LUONG ATTENTION

Cơ chế Luong Attention (General Score) có 2 điểm khác biệt cốt lõi so với Bahdanau Attention:

1. **Thời điểm tính toán**: Luong Attention được tính toán **SAU** khi trạng thái ẩn Decoder $h_t$ hiện tại được sinh ra bởi LSTM (thay vì trước khi qua LSTM như Bahdanau).
2. **Hàm Score Multiplicative**: Luong dùng phép nhân ma trận $h_t^T W_a h_j$ nhanh hơn phép cộng Additive $\tanh(W_s s_{t-1} + W_h h_j)$.

```
Decoder Step t Input y_{t-1} ──> [Decoder LSTM] ──> Current Query h_t [B, H]
                                                         │
Encoder Outputs H: [B, S, H] ────────────────────────────┼──> Query Projection W_a * h_t
          │                                              │
          ▼                                              ▼
   Scores e_{t, j} = bmm( H, W_a * h_t )  ───────────────┘  [B, S]
          │
  Masking (<PAD> -> min_float)
          │
       Softmax
          ▼
  Attention Weights α_t [B, S]
          │
  Context Vector c_t = bmm( α_t, H ) [B, H]
          │
Attentional Hidden h~_t = tanh( W_c * [h_t; c_t] ) [B, H]
          │
Logits = Linear( h~_t ) [B, V_tgt]
```

### Bảng Phân Tích Hình Dạng Tensor (Tensor Shape Flow):
| Tên Tensor trong Code | Ký hiệu Toán | Hình dạng Tensor (Shape) | Giải thích |
| :--- | :--- | :--- | :--- |
| `encoder_outputs` | $H$ | `[B, S, H]` | Trạng thái ẩn của Encoder (đã qua `pack_padded_sequence`) |
| `query` | $h_t$ | `[B, H]` | Trạng thái ẩn Decoder hiện tại bước $t$ |
| `projected_query` | $W_a h_t$ | `[B, H, 1]` | Query đã qua chiếu tuyến tính `query_projection` |
| `scores` | $e_{t, j}$ | `[B, S]` | Điểm Multiplicative score $h_j^T W_a h_t$ |
| `attention_weights` | $\alpha_{t, j}$ | `[B, S]` | Trọng số chú ý đã chuẩn hóa bằng Softmax |
| `context` | $c_t$ | `[B, H]` | Vector ngữ cảnh động tại bước $t$ |
| `attentional_hidden`| $\tilde{h}_t$ | `[B, H]` | Trạng thái kết hợp $\tanh(W_c [h_t; c_t])$ |

---

## 📍 PHẦN 2: GIẢI THÍCH CHI TIẾT TỪNG MỤC CELL (CELL-BY-CELL BREAKDOWN)

### Mục 1 – 13: Tiền Xử Lý Dữ Liệu & Dynamic Padding (Cells 1 – 28)
- **Cells 1–8**: Khai báo thư viện (`torch`, `sacrebleu`, `pyvi`), cố định seed `set_seed(42)`, đặt cấu hình thí nghiệm `QUICK_RUN = False`.
- **Cells 9–13**: Đọc tập dữ liệu IWSLT'15, kiểm tra số dòng UTF-8 ($133.317$ câu Train, $1.553$ câu Val, $1.268$ câu Test).
- **Cells 14–21**: Chuẩn hóa Unicode NFC, unescape HTML, tokenize Anh/Việt, lọc các câu $> 50$ tokens hoặc có tỷ lệ độ dài bất thường, loại bỏ triệt để rò rỉ (overlap) giữa Train và Val/Test.
- **Cells 22–24**: Tạo lớp `Vocabulary` chỉ fit từ tập Train.
- **Cells 25–28**: Khai báo `ParallelTranslationDataset` và `make_collate_fn()` thực hiện Dynamic Padding theo batch.

---

### Mục 14: Mã Nguồn Các Lớp PyTorch Với Luong Attention (Cells 29 – 30)

#### 1. Lớp `EncoderLSTM` (Tích hợp `pack_padded_sequence`)
```python
class EncoderLSTM(nn.Module):
    def forward(self, source: torch.Tensor, source_lengths: torch.Tensor):
        embedded = self.dropout(self.embedding(source))
        # BỎ QUA TÍNH TOÁN PADDING TOKEN BANG PACKED SEQUENCE
        packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=source.size(1))
        return encoder_outputs, hidden, cell
```

#### 2. Lớp `LuongGeneralAttention` (Multiplicative General Score)
```python
class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query: torch.Tensor, encoder_outputs: torch.Tensor, source_mask: torch.Tensor):
        # query: [B, H] -> projected_query: [B, H, 1]
        projected_query = self.query_projection(query).unsqueeze(2)
        # TÍNH MULTIPLICATIVE SCORE: bmm([B, S, H], [B, H, 1]) -> [B, S]
        scores = torch.bmm(encoder_outputs, projected_query).squeeze(2)
        
        # PADDING MASKING: Gán giá trị min_float vào vị trí <PAD>
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=1)
        
        # CONTEXT VECTOR: [B, 1, S] * [B, S, H] -> [B, H]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights
```

#### 3. Lớp `LuongDecoderLSTM` (Attentional Vector $\tilde{h}_t$)
```python
class LuongDecoderLSTM(nn.Module):
    def forward(self, input_token, hidden, cell, encoder_outputs, source_mask):
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        # 1. Đi qua LSTM trước
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        query = output.squeeze(1)
        
        # 2. Tính Attention dựa trên current query h_t
        context, attention_weights = self.attention(query, encoder_outputs, source_mask)
        
        # 3. Kết hợp query h_t và context c_t qua tanh layer
        attentional_hidden = torch.tanh(self.context_projection(torch.cat([query, context], dim=1)))
        
        # 4. Dự đoán Logits
        logits = self.output_projection(self.dropout(attentional_hidden))
        return logits, hidden, cell, attention_weights
```

#### 4. Khởi tạo Trọng số `initialize_model()` (Xavier Uniform)
```python
def initialize_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.zeros_(parameter)
    # Cố định embedding weight của <PAD> token bằng 0
    with torch.no_grad():
        model.encoder.embedding.weight[model.encoder.embedding.padding_idx].zero_()
        model.decoder.embedding.weight[model.decoder.embedding.padding_idx].zero_()
```

---

### Mục 15 & 16: Training Loop & Huấn Luyện Hai Chiều (Cells 31 – 34)
- **Cell 31–32**: Huấn luyện với Optimizer Adam (`lr=1e-3`), CrossEntropyLoss (`ignore_index=target_pad_id`) và Gradient Clipping (`max_norm=1.0`).
- **Cell 33–34**: Chạy huấn luyện 2 chiều English $\rightarrow$ Vietnamese và Vietnamese $\rightarrow$ English.

---

### Mục 17 & 18: Đồ Thị Loss & Greedy Decoding (Cells 35 – 38)
- **Cell 37–38 (`greedy_decode_batch`)**: Giải mã Greedy không Teacher Forcing, tự động thu thập ma trận trọng số Luong Attention `[B, T, S]` để trực quan hóa.

---

### Mục 19 – 21: SacreBLEU & Benchmarks So Sánh 3 Mô Hình (Cells 39 – 44)
- **Cell 39–40**: Đánh giá điểm SacreBLEU chính thức.
- **Cell 41–42 (`subset_sacrebleu`)**: Đánh giá điểm BLEU theo 3 nhóm độ dài câu ($\le 15$, $16-29$, $\ge 30$).
- **Cell 43–44**: Đọc file CSV benchmark và hiển thị bảng so sánh 3 mô hình:
  1. `Seq2Seq LSTM` (No Attention)
  2. `Seq2Seq LSTM + Bahdanau` (Additive Attention)
  3. `Seq2Seq LSTM + Luong` (Multiplicative General Attention)

---

### Mục 22 – 27: Dịch Thử, Visualizing Heatmap & Kết Luận (Cells 45 – 54)
- **Cell 49–50**: `verify_attention_padding_mask()` xác nhận tổng chú ý tại `<PAD>` bằng $0.0$, và `plot_attention_heatmap()` vẽ biểu đồ nhiệt Luong Attention.
- **Cell 51–52**: Dịch câu thực tế do người dùng nhập.
- **Cell 53–54**: Kết luận bài học và gợi ý mở rộng dataset.

---
