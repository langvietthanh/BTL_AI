# Kế hoạch Đồ án: Phân loại Spam Email / Tin nhắn (Spam Detection)

Dự án này rất phù hợp cho môn **Nhập môn AI / Học máy (Machine Learning)**. Việc phân loại xem một email hay tin nhắn là "Spam" (Thư rác) hay "Ham" (Thư hợp lệ) là bài toán phân loại văn bản kinh điển (Text Classification), đủ đơn giản để bắt đầu nhưng cũng có nhiều kỹ thuật mở rộng để ăn điểm cao.

Dưới đây là kế hoạch từng bước chi tiết để bạn triển khai dự án này một cách chuyên nghiệp.

## 1. Mục tiêu (Goal)
- Xây dựng một hệ thống AI có khả năng nhận danh bản text (email/SMS) và dự đoán nó là Thư rác hay Thư bình thường.
- So sánh hiệu năng của các thuật toán cơ bản, làm quen với quy trình chuẩn của NLP (Xử lý ngôn ngữ tự nhiên) trong AI.

## 2. Công nghệ Cần dùng (Tech Stack)
- **Ngôn ngữ:** Python
- **Thư viện AI / ML:** `scikit-learn` (để tạo mô hình nhận diện), `nltk` hoặc `spacy` (để xử lý ngôn ngữ tự nhiên).
- **Thư viện thao tác dữ liệu:** `pandas`, `numpy`.
- **Thư viện hiển thị biểu đồ:** `matplotlib`, `seaborn` (vẽ biểu diễn từ vựng, confusion matrix).
- **(Tuỳ chọn ăn điểm cao) Giao diện Demo:** `Streamlit` hoặc `Gradio` (rất dễ làm, chỉ mất 20 dòng code để tạo web demo).

---

## 3. Các Giai đoạn Thực hiện (Phases)

### Giai đoạn 1: Thu thập Dữ liệu (Dataset)
- **Hành động:** Tìm và tải dữ liệu chuẩn đã được phân nhãn sẵn.
- **Nguồn khuyên dùng:**
  - **SMS Spam Collection (Kaggle):** Tập dữ liệu rất phổ biến gồm ~5.500 tin nhắn SMS đã được dán nhãn Spam/Ham. Rất nhẹ và dễ huấn luyện nhanh.
  - **Enron Email Dataset:** Tập dữ liệu email thực tế, danh tiếng và phức tạp hơn, phù hợp nếu muốn làm dự án lớn và sâu hơn về email thay vì SMS.
- **Đầu ra mong muốn:** File `dataset.csv` chứa 2 cột chính: `text` (nội dung) và `label` (nhãn spam/ham).

### Giai đoạn 2: Tiền xử lý Dữ liệu Ngôn ngữ (Data Preprocessing - Rất Quan Trọng)
*Văn bản thô không thể đưa trực tiếp vào mô hình, ta phải chuyển nó thành số.*
1. **Làm sạch (Cleaning):** Chuyển tất cả chữ về in thường, xóa các dấu câu (`.,!?`), loại bỏ URL và ký tự đặc biệt, số điện thoại.
2. **Tokenization:** Tách câu thành các từ rời rạc.
3. **Stop words removal:** Bỏ các từ nối, từ thông dụng không mang nhiều ý nghĩa nhận diện (như *and, or, the, is...*).
4. **Lemmatization / Stemming:** Đưa từ về nguyên thể (ví dụ: *running* -> *run*, *cars* -> *car*).
5. **Trích xuất đặc trưng (Feature Extraction - Vector hóa):**
   - **TF-IDF (Term Frequency-Inverse Document Frequency):** Kỹ thuật chuyển từ ngữ thành dạng số phổ biến và tốt nhất cho bài toán này ở mức độ cơ bản.

### Giai đoạn 3: Huấn luyện Mô hình (Model Training)
Vì là môn Nhập môn AI, bạn nên dùng các mô hình Machine Learning cổ điển để làm Baseline (biết kết quả cơ bản).
- **Thuật toán thứ 1 (Bắt buộc - Baseline): Naive Bayes (MultinomialNB).** Đây là thuật toán tiêu chuẩn và nổi tiếng nhất cho bài toán chống spam nhờ tính hiệu quả về tốc độ dựa trên xác suất, có kết quả rất ấn tượng ngay từ đầu.
- **Thuật toán thứ 2 (Nâng cao để so sánh): Support Vector Machine (SVM) hoặc Logistic Regression.** Trọng số của các mẫu spam đôi khi phức tạp và SVM có thể tìm được ranh giới phi tuyến tốt hơn Naive Bayes.
- **Chia tập dữ liệu (Train/Test Split):** 80% dữ liệu để huấn luyện (Train), 20% dữ liệu ẩn đi để kiểm tra (Test).

### Giai đoạn 4: Đánh giá Mô hình (Evaluation)
Bài toán Spam thì Accuracy (Độ chính xác tổng) không nói lên tất cả nếu số lượng nhãn Spam ít hơn mức bình thường (Imbalanced Data). Cần đánh giá qua:
- **Precision (Độ chuẩn xác):** *Rất quan trọng cho bài toán Email*. Nếu AI nói thư này là Spam, tỉ lệ nó là Spam thật là bao nhiêu? Bạn không được phép nhận diện nhầm Thư người yêu gửi thành Thư rác và xoá mất.
- **Recall (Độ bao phủ):** Tỉ lệ nhận diện được số thư Spam trong toàn bộ Thư rác thực tế có. 
- **F1-Score:** Trung bình hài hoà cân bằng giữa Precision và Recall.
- **Báo cáo:** Vẽ Confusion Matrix ra báo cáo.

### Giai đoạn 5 (Tuỳ chọn ăn điểm): Demo UI bằng Giao diện Web
- Tạo một file app `app.py` bằng `Streamlit`.
- Lưu model đã train bằng `joblib` hoặc `pickle`.
- Khởi động app: Giao diện sẽ có 1 khung Text box, người dùng (để cho thầy cô) gõ hoặc chép tin nhắn vào -> Bấm "Predict" -> Màn hình in ra chữ Thư rác đỏ chót hoặc Thư thường màu xanh.

---

## 4. Dự kiến Lịch trình (Timeline)
Nếu bạn có 3 tuần đến 1 tháng để làm:
- **Tuần 1:** Cài đặt môi trường, tải dữ liệu, khám phá dữ liệu (EDA - xem phân phối các từ thường gặp trong Spam vs. Ham).
- **Tuần 2:** Viết các hàm Pipeline làm sạch ngôn ngữ (Preprocessing + TF-IDF). Chạy thử thuật toán Naive Bayes.
- **Tuần 3:** Tinh chỉnh model, chạy so sánh nghiệm với Logistic Regression, trích xuất hình ảnh báo cáo (Confusion matrix, Precision/Recall charts) vào file báo cáo Word/Powerpoint.
- **Tuần 4 (Nếu dư thời gian):** Code bản Web UI Streamlit, tập dượt thuyết trình.

---

> [!TIP]
> **Điểm nhấn ăn điểm cộng trong bài thuyết trình:**
> Hãy nhấn mạnh vào **Trade-off giữa Precision và Recall**. Giải thích cho giảng viên là: *"Tụi em chấp nhận để lọt một vài email spam (Recall thấp một chút), thay vì nhận diện sai hoàn toàn thư xin việc thành thư rác (Precision cao ưu tiên) vì trải nghiệm khách hàng sẽ rất tệ nếu mất thư quan trọng."* Điều này chứng tỏ bạn thực sự hiểu ý nghĩa thực tiễn chứ không chỉ là chạy code.

## Câu hỏi mở (User Review Required)
Bạn có muốn tôi tiến hành thực thi và viết code các phần mềm dựa trên kế hoạch này không? 
1. Bạn đã có sẵn Dataset chưa, hay muốn tôi code một script tự tải Dataset ví dụ (như Kaggle SMS Spam)?
2. Bạn muốn xuất ra dạng Jupyter Notebook (`.ipynb`) - Rất tiện nộp bài làm report, hay là code dạng Script Python thông thường (`.py`)?
3. Bạn có muốn làm phần giao diện web Demo (như Streamlit) luôn không?
