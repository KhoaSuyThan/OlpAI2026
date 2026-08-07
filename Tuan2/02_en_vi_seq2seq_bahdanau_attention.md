# Hướng Dẫn & Giải Thích Chi Tiết Từng Cell: Notebook 02 - Seq2Seq với Bahdanau Attention

Tài liệu này phân tích toàn bộ **27 phần (55 Cells)** trong notebook `02_en_vi_seq2seq_bahdanau_attention.ipynb`. Mọi công thức toán học, đoạn mã PyTorch, hình dạng Tensor (Tensor Shapes) và kỹ thuật Masking đều được giải thích sâu.

---

## 📍 PHẦN 1: CƠ CHẾ TOÁN HỌC & TENSOR SHAPE FLOW CỦA BAHDANAU ATTENTION

Trong kiến trúc Bahdanau Additive Attention (Bahdanau et al., 2014):

```
Encoder Outputs H: [B, S, H_enc]        Decoder Hidden s_{t-1}: [B, H_dec]
         │                                        │
         ├─── W_h (Linear) ─────────── W_s (Linear) ──┤
         │        [B, S, H_att]           [B, 1, H_att] │
         ▼                                             ▼
     tanh( W_h * H + W_s * s_{t-1} )  ──> Energy [B, S, H_att]
                                              │
                                           v_a (Linear)
                                              ▼
                                         Scores [B, S]
                                              │
                                   Masking (<PAD> -> -1e9)
                                              │
                                           Softmax
                                              ▼
                                   Attention Weights α_t [B, 1, S]
                                              │
                              bmm( α_t, H )  ─┘
                                              ▼
                                Context Vector c_t [B, 1, H_enc]
```

### Bảng Phân Tích Hình Dạng Tensor (Tensor Shape Flow):
| Tên Tensor trong Code | Ký hiệu Toán | Hình dạng Tensor (Shape) | Giải thích |
| :--- | :--- | :--- | :--- |
| `encoder_outputs` | $H$ | `[B, S, H_enc]` | Tất cả trạng thái ẩn của Encoder cho $S$ từ nguồn |
| `decoder_hidden` | $s_{t-1}$ | `[B, H_dec]` | Trạng thái ẩn của Decoder tại bước trước $t-1$ |
| `source_mask` | $M$ | `[B, S]` | Tensor boolean: `True` tại vị trí từ thật, `False` tại `<PAD>` |
| `scores` | $e_{t, j}$ | `[B, S]` | Điểm chưa chuẩn hóa của từng vị trí nguồn |
| `attention_weights`| $\alpha_{t, j}$ | `[B, 1, S]` | Trọng số chú ý đã chuẩn hóa bằng Softmax (tổng bằng $1.0$) |
| `context` | $c_t$ | `[B, 1, H_enc]` | Vector ngữ cảnh động được tính cho bước $t$ |

---

## 📍 PHẦN 2: GIẢI THÍCH CHI TIẾT TỪNG MỤC CELL (CELL-BY-CELL BREAKDOWN)

### Mục 1 – 13: Tiền Xử Lý Dữ Liệu & Dynamic Padding (Cells 1 – 28)
*(Giữ nguyên tính nhất quán và quy chuẩn làm sạch của Notebook 01)*
- **Cells 1–8**: Khai báo thư viện, thiết lập `QUICK_RUN`, cố định seed với `set_seed(42)`.
- **Cells 9–13**: Tải tập dữ liệu song song IWSLT'15, đếm số dòng UTF-8.
- **Cells 14–21**: Chuẩn hóa Unicode NFC, unescape HTML, tách từ tiếng Anh/Việt, lọc các câu $> 50$ tokens hoặc tỷ lệ độ dài bất thường, lọc loại bỏ hoàn toàn câu rò rỉ (overlap) giữa Train và Val/Test.
- **Cells 22–24**: Tạo từ vựng `Vocabulary` riêng cho từng ngôn ngữ chỉ từ tập **Train**.
- **Cells 25–28**: Khai báo `ParallelTranslationDataset` và hàm `make_collate_fn` thực hiện **Dynamic Padding**.

---

### Mục 14: Mã Nguồn Các Lớp PyTorch Với Attention (Cells 29 – 30)

#### 1. Lớp `BahdanauAttention` (Cơ chế Chú ý Cộng)
```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False) # Projection cho Encoder outputs
        self.W_s = nn.Linear(hidden_dim, attn_dim, bias=False) # Projection cho Decoder hidden
        self.v_a = nn.Linear(attn_dim, 1, bias=False)          # Vector trọng số cuối v_a

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, source_mask: torch.Tensor):
        # decoder_hidden: [B, H] -> unsqueeze(1) thành [B, 1, H]
        # encoder_outputs: [B, S, H]
        score_h = self.W_h(encoder_outputs)  # [B, S, attn_dim]
        score_s = self.W_s(decoder_hidden).unsqueeze(1)  # [B, 1, attn_dim]

        # TÍNH TOÁN SCORE: tanh(W_h * H + W_s * s_{t-1})
        energy = torch.tanh(score_h + score_s)  # BroadCast thành [B, S, attn_dim]
        scores = self.v_a(energy).squeeze(2)    # [B, S]

        # PADDING MASKING: Gán -1e9 vào vị trí <PAD> trước Softmax
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask, -1e9)

        # SOFTMAX: Alpha = exp(scores) / sum(exp(scores))
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # [B, 1, S]

        # CONTEXT VECTOR: c_t = bmm(Alpha, Encoder_Outputs)
        context = torch.bmm(attention_weights, encoder_outputs)    # [B, 1, H]
        return context, attention_weights.squeeze(1)
```

#### 2. Lớp `BahdanauDecoderLSTM`
- Kết hợp embedding từ đầu vào bước trước $y_{t-1}$ và vector ngữ cảnh động $c_t$ làm đầu vào cho LSTM:
  ```python
  lstm_input = torch.cat([embedded, context], dim=2) # [B, 1, embed_dim + hidden_dim]
  output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
  ```
- Chiếu ra từ vựng bằng cách ghép cả 3 vector: `[output; context; embedded]`:
  ```python
  prediction = self.fc_out(torch.cat([output, context, embedded], dim=2).squeeze(1))
  ```

#### 3. Lớp `Seq2SeqBahdanau`
- Tích hợp `EncoderLSTM`, `BahdanauDecoderLSTM` và `BahdanauAttention`.
- Tạo `source_mask = source.ne(source_pad_id)` truyền vào cho Attention ở từng bước giải mã.

---

### Mục 15 & 16: Training Loop & Huấn Luyện Hai Chiều (Cells 31 – 34)
- **Cell 31–32**: Huấn luyện với CrossEntropyLoss (`ignore_index=target_pad_id`) và Gradient Clipping (`max_norm=1.0`).
- **Cell 33–34**: Chạy huấn luyện 2 chiều En $\rightarrow$ Vi và Vi $\rightarrow$ En.

---

### Mục 17 & 18: Đồ Thị Loss & Greedy Decoding Có Attention Weights (Cells 35 – 38)
- **Cell 37–38 (`greedy_decode_batch`)**:
  - Không chỉ trả về các token ID dự đoán `predicted_ids`, hàm còn thu thập và trả về ma trận trọng số Attention `[B, T, S]` tại mọi bước thời gian giải mã.

---

### Mục 19 & 20: Đánh Giá SacreBLEU & Length Benchmark (Cells 39 – 42)
- **Cell 39–40**: Tính toán điểm SacreBLEU tổng thể trên tập Test.
- **Cell 41–42 (`subset_sacrebleu`)**:
  - Chia câu test thành 3 nhóm độ dài:
    1. Ngắn: $\le 15$ tokens
    2. Trung bình: $16 - 29$ tokens
    3. Dài: $\ge 30$ tokens
  - Điểm BLEU cho câu dài ($\ge 30$) đạt mức ấn tượng nhờ Attention không bị trôi ngữ cảnh.

---

### Mục 21: So Sánh Các Benchmark Đã Chạy (Cells 43 – 44)
- Tự động đọc file CSV trong thư mục `benchmarks/` và vẽ bảng so sánh đối chiếu giữa:
  - Notebook 01: Baseline (No Attention)
  - Notebook 02: Bahdanau Attention

---

### Mục 22 – 25: Trực Quan Hóa Attention Heatmap & Kiểm Tra Masking (Cells 45 – 52)
- **Cell 49–50 (`verify_attention_padding_mask` & `plot_attention_heatmap`)**:
  - `verify_attention_padding_mask()`: Kiểm tra tự động xác nhận tổng trọng số chú ý tại mọi vị trí đệm `<PAD>` là $0.0000$.
  - `plot_attention_heatmap()`: Vẽ ma trận heatmap giữa câu nguồn (trục X) và câu đích (trục Y). Độ sáng của pixel thể hiện mức độ chú ý của mô hình khi sinh từ.

---

### Mục 26 & 27: Mở Rộng Việt – Trung & Kết Luận (Cells 53 – 54)
- **Cell 53**: Thảo luận khả năng mở rộng sang các cặp ngôn ngữ khác (Việt – Trung với dataset `ngocdang83/tran-vi-teacher`).
- **Cell 54**: Tổng kết ưu điểm vượt trội của Bahdanau Attention trong bài toán Dịch máy hai chiều.

---
