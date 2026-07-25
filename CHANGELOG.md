# Changelog - OlpAI 2026

Tất cả những thay đổi quan trọng của dự án sẽ được ghi nhận tại đây.

---

## [Unreleased] - 2026-07-25

### Added
- Khởi tạo môi trường phát triển và cài đặt hoàn tất các gói thư viện ML/DL: `xgboost`, `lightgbm`, `catboost`, `optuna`, `timm`, `transformers`, `albumentations`.
- Tạo thư mục `GiaiDoan1_Tabular_Data` chuẩn bị cho chuỗi bài tập Machine Learning dữ liệu bảng.

### Changed
- Cấu hình môi trường Conda mặc định (`olpai2026`) cho VS Code workspace và Pyrefly Language Analyzer (`pyrefly.toml`).
- Tối ưu signature của `__getitem__` trong `BinaryDataset` để phù hợp với chuẩn interface của PyTorch.

### Refactored
- Đổi tên thư mục bài tập cơ bản từ `Tuan1` sang `GiaiDoan0` để đồng bộ với cấu trúc Lộ trình học trong `README.md`.

---
