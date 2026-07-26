# Changelog - OlpAI 2026

Tất cả những thay đổi quan trọng của dự án sẽ được ghi nhận tại đây.

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
