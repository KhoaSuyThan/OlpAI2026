# Changelog - OlpAI 2026

Tất cả những thay đổi quan trọng của dự án sẽ được ghi nhận tại đây.

---
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
