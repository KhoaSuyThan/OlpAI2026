# 🚀 OlpAI 2026 - Lộ Trình Tự Học & Template Mã Nguồn AI

Repository này lưu trữ toàn bộ bài tập thực hành, pipeline thí nghiệm và bộ template code chuẩn bị cho **Kỳ thi Olympic AI (OlpAI 2026)**.
> ⚠️ **LƯU Ý / DISCLAIMER:**  
> Đây là **nhật ký học tập và lộ trình tự học cá nhân** được biên soạn dựa trên AI nhằm mục đích tự ôn luyện, thử nghiệm thuật toán và lưu trữ các đoạn mã mẫu (template). Đây **KHÔNG** phải là quy trình chuẩn mực chính thức, giáo trình chuẩn hay quy định bắt buộc của bất kỳ tổ chức/kỳ thi nào. Các phương pháp và kiến thức trong này được tổng hợp và tùy biến theo tốc độ học cá nhân.

---

## 📌 Tổng Quan Lộ Trình Tự Học Chi Tiết (Self-Study Roadmap)

> 💡 **Lưu ý quan trọng cho kỳ thi:**
> 1. **Reproducibility:** Tất cả các bài làm bắt buộc phải cố định Random Seed (`seed_everything`).
> 2. **Submission Format:** File nộp bài `.csv` luôn phải kiểm tra kỹ 2 thông số: `index=False` và `encoding='utf-8'`.
> 3. **Pipeline tách biệt:** Luôn phân chia rõ ràng giữa phase **Train** (`model.train()`) và phase **Validation / Inference** (`model.eval()`, `with torch.no_grad():`).

---

### 📍 Giai Đoạn 0: Setup Môi Trường & PyTorch Fundamentals (ĐÃ HOÀN THÀNH)
* **Kiến thức cốt lõi:**
  * Khởi tạo môi trường Anaconda `olpai2026` (Python 3.10), quản lý Kernel trên VS Code Jupyter.
  * Thao tác cơ bản với PyTorch Tensor & Chuyển đổi linh hoạt giữa Tensor $\leftrightarrow$ NumPy.
  * Xử lý dữ liệu bảng (Tabular Data) bị khuyết (`NaN`) bằng `mean` / `median` với `pandas`.
* **Kỹ thuật bắt buộc:**
  * Viết hàm `seed_everything()` cố định seed cho `random`, `np.random`, `torch`.
  * Xây dựng `CustomDataset` và `DataLoader` (Batching, Shuffling).
  * Viết vòng lặp huấn luyện (`Training Loop`) chuẩn 5 bước PyTorch cho bài toán Binary Classification, Multi-class Classification & Regression.
  * Đánh giá mô hình trên tập Validation (Loss & Accuracy).
  * Viết hàm `create_submission()` đóng gói file nộp bài chuẩn quy chế.

---

### 📍 Giai Đoạn 1: Machine Learning Cho Dữ Liệu Bảng (Tabular Data)
* **Thời gian dự kiến:** 1 - 2 tuần
* **Kiến thức cốt lõi:**
  * **Feature Engineering:** One-Hot Encoding, Label Encoding, Scaling (StandardScaler, MinMaxScaler), handling outliers.
  * **Mô hình Tree-based:** Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost.
  * **K-Fold Cross Validation:** Stratified K-Fold để đánh giá mô hình khách quan và chống Overfitting.
  * **Hyperparameter Tuning:** Optuna, GridSearch, RandomSearch.
* **Sản phẩm cần đạt:**
  * Xây dựng pipeline XGBoost/CatBoost chuẩn có K-Fold CV và tạo dự đoán Ensemble (bình quân trọng số) xuất ra file `submission.csv`.

---

### 📍 Giai Đoạn 2: Computer Vision (CV) - Xử Lý Ảnh
* **Thời gian dự kiến:** 2 - 3 tuần
* **Kiến thức cốt lõi:**
  * **Image Processing & Augmentation:** Albumentations, torchvision.transforms (Resize, Normalization, Random Crop, Flip, Rotation).
  * **CNN Architectures:** ResNet (ResNet18/34/50), EfficientNet, ConvNeXt.
  * **Transfer Learning:** Sử dụng pretrained weights từ `timm` (PyTorch Image Models), Fine-tuning & Feature Extraction.
  * **Các bài toán chính:** Image Classification, Object Detection cơ bản (YOLO/Faster R-CNN concept), Semantic Segmentation cơ bản.
* **Sản phẩm cần đạt:**
  * Viết Custom Image Dataset nạp ảnh từ thư mục, huấn luyện mô hình ResNet/EfficientNet phân loại ảnh với `timm`.

---

### 📍 Giai Đoạn 3: Natural Language Processing (NLP) - Xử Lý Ngôn Ngữ Tự Nhiên
* **Thời gian dự kiến:** 2 - 3 tuần
* **Kiến thức cốt lõi:**
  * **Text Preprocessing:** Tokenization, Cleaning, Stopwords, Lowercasing (Tiếng Việt & Tiếng Anh).
  * **Transformers & HuggingFace:** Thư viện `transformers`, `datasets`, `evaluate`.
  * **Pretrained Models:** PhoBERT, viBERT (cho tiếng Việt), RoBERTa, BERT.
  * **Các bài toán chính:** Text Classification (Phân loại cảm xúc/chủ đề), Named Entity Recognition (NER), Question Answering.
* **Sản phẩm cần đạt:**
  * Fine-tune mô hình PhoBERT với HuggingFace `Trainer` hoặc PyTorch Native DataLoader để phân loại văn bản.

---

### 📍 Giai Đoạn 4: Kỹ Thuật Nâng Cao & Tối Ưu Điểm Số (Advanced & Ensembling)
* **Thời gian dự kiến:** 1 - 2 tuần trước kỳ thi
* **Kiến thức cốt lõi:**
  * **Model Ensembling:** Blending, Stacking, Voting Classifier.
  * **Post-processing:** Threshold tuning (tối ưu F1-score/Accuracy bằng cách chọn ngưỡng xác suất phù hợp).
  * **Optimization:** Learning Rate Schedulers (CosineAnnealing, ReduceLROnPlateau), Mixed Precision Training (`torch.cuda.amp` giúp tăng tốc & tiết kiệm VRAM).
  * **Viết Báo Cáo Kỹ Thuật:** Chuẩn bị sẵn template báo cáo phương pháp luận, bảng so sánh thực nghiệm (Experiments Table).

---
