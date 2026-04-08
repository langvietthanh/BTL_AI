# Nhiệm vụ Đồ án Phân loại Spam

- [/] **1. Khởi tạo môi trường & Cài đặt thư viện**
  - [ ] Tạo file `requirements.txt`.
  - [ ] Cài đặt các thư viện: `scikit-learn`, `pandas`, `nltk`, `streamlit`, `matplotlib`, `seaborn`.

- [ ] **2. Thu thập và Xử lý Dữ liệu**
  - [ ] Tự động tải tập dữ liệu SMS Spam Collection (UCI).
  - [ ] Viết hàm tiền xử lý văn bản (xóa dấu câu, viết thường, tokenize, stop words).

- [ ] **3. Huấn luyện Mô hình (Model Training)**
  - [ ] Trích xuất đặc trưng với TF-IDF.
  - [ ] Huấn luyện thuật toán Naive Bayes.
  - [ ] Đánh giá mô hình: In ra Accuracy, Precision, Recall, F1.
  - [ ] Lưu trữ mô hình đã huấn luyện (để dùng cho Web App).
  
- [ ] **4. Xây dựng Giao diện Web (Streamlit App)**
  - [ ] Tạo `app.py` nhận đầu vào từ giao diện.
  - [ ] Tích hợp mô hình đã lưu vào ứng dụng.
  - [ ] Trả về kết quả Spam/Ham trực quan.
