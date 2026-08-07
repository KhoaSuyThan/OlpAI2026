# Changelog - OlpAI 2026

Tất cả những thay đổi quan trọng của dự án sẽ được ghi nhận tại đây.

---

---
## [Unreleased] - 2026-08-07

### Added
- Khởi tạo bộ 3 tài liệu giải thích chi tiết và hệ thống hóa kiến thức cho Tuần 2 (Neural Machine Translation & Attention Mechanisms):
  - `Tuan2/01_en_vi_seq2seq_no_attention.md`: Giải thích chi tiết kiến trúc Seq2Seq Encoder-Decoder LSTM không Attention, quy trình làm sạch dữ liệu IWSLT'15, dynamic padding, Teacher Forcing và Greedy Decoding.
  - `Tuan2/02_en_vi_seq2seq_bahdanau_attention.md`: Giải thích cơ chế Bahdanau Additive Attention, công thức toán học, kỹ thuật Padding Masking, trực quan hóa ma trận Attention Alignment và Length Benchmark.
  - `Tuan2/EXPLANATION_PHASE2.md`: Tài liệu tổng hợp Giai đoạn 2 theo góc nhìn Kỹ sư Phần mềm (Software Engineer), quy đổi thuật ngữ AI sang khái niệm Phần mềm, phân tích luồng dữ liệu và các lưu ý trọng tâm cho kỳ thi OlpAI.

---
## [Unreleased] - 2026-07-30

### Added
- Khởi tạo tài liệu Lộ trình Ôn tập & Hệ thống hóa Kiến thức Buổi 02 (`GiaiDoan1_Tabular_Data/LO_TRINH_ON_TAP_BUOI_02.md`): Phân tích lý thuyết toán học (RNN, GRU, LSTM, BiRNN), ma trận ánh xạ 4 kiến trúc NLP với 2 notebook bài tập thực hành (`01_text_classification_rnn.ipynb` & `02_pos_tagging_bilstm.ipynb`), lộ trình học 4 bước và bộ 6 câu hỏi tự đánh giá.

## [Unreleased] - 2026-07-29

### Added
- Phân tích chi tiết và khởi tạo 2 tài liệu hướng dẫn sư phạm chuyên sâu cho mảng NLP Recurrent Neural Networks:
  - `GiaiDoan1_Tabular_Data/01_text_classification_rnn.md`: Giải thích toàn bộ luồng Sentiment Analysis tiếng Việt, kỹ thuật `pack_padded_sequence`, so sánh Vanilla RNN, GRU, LSTM và đáp án chi tiết các câu hỏi thực hành.
  - `GiaiDoan1_Tabular_Data/02_pos_tagging_bilstm.md`: Giải thích bài toán POS Tagging (Sequence Labelling), quy trình gom câu không rò rỉ dữ liệu, cơ chế đệm đồng thời, so sánh Unidirectional LSTM vs BiLSTM và giải thích lý do BiLSTM đạt F1-Score vượt trội (92.49% vs 87.22%).

  
## [Unreleased] - 2026-07-27

### Added
- Viết lại toàn bộ 2 file tài liệu giải thích `GiaiDoan0/EXPLANATION_PHASE0.md` và `GiaiDoan1_Tabular_Data/EXPLANATION_PHASE1.md` theo góc nhìn Kỹ sư Phần mềm (Software Engineers): Bảng quy đổi thuật ngữ AI sang khái niệm Phần mềm, gắn ngữ cảnh bài toán doanh nghiệp thực tế (Bank Fraud Detection, Telecom Churn, Credit Scoring, Property Valuation) và phân tích từng bước xử lý dữ liệu.
- Bổ sung notebook `GiaiDoan1_Tabular_Data/03_Optuna_Submission_and_Feature_Importance.ipynb` hoàn thiện Giai đoạn 1 với các kỹ thuật: Handling Outliers (IQR Clipping), Label Encoding, Feature Importance Analysis, Optuna Hyperparameter Tuning và luồng K-Fold Ensemble Inference.
- Tạo thành công file nộp bài mẫu `GiaiDoan1_Tabular_Data/submission_phase1.csv` đúng quy chuẩn `index=False` và `encoding='utf-8'`.
- Cập nhật tài liệu hướng dẫn và giải thích chi tiết notebook 03 trong `GiaiDoan1_Tabular_Data/EXPLANATION_PHASE1.md`.

---

## [Unreleased] - 2026-07-26

### Added
- Tạo Kế hoạch thực hành Giai đoạn 1 (`phase1_implementation_plan.md`) chi tiết.
- Xây dựng bài tập thực hành `GiaiDoan1_Tabular_Data/02_CatBoost_and_Encoding.ipynb` hướng dẫn sử dụng Native Categorical Features của CatBoost và So sánh điểm Out-Of-Fold giữa 3 mô hình XGBoost, LightGBM, CatBoost.
- Thêm tài liệu hướng dẫn & giải thích từng bài tập nhỏ cho Giai đoạn 0 (`GiaiDoan0/EXPLANATION_PHASE0.md`) và Giai đoạn 1 (`GiaiDoan1_Tabular_Data/EXPLANATION_PHASE1.md`).

---

## [0.1.0] - 2026-07-25

### Added
- Khởi tạo môi trường phát triển và cài đặt hoàn tất các gói thư viện ML/DL: `xgboost`, `lightgbm`, `catboost`, `optuna`, `timm`, `transformers`, `albumentations`.

### Changed
- Cấu hình môi trường Conda mặc định (`olpai2026`) cho VS Code workspace và Pyrefly Language Analyzer (`pyrefly.toml`).
- Tối ưu signature của `__getitem__` trong `BinaryDataset` để phù hợp với chuẩn interface của PyTorch.

### Refactored
- Đổi tên thư mục bài tập cơ bản từ `Tuan1` sang `GiaiDoan0` để đồng bộ với cấu trúc Lộ trình học trong `README.md`.

---
