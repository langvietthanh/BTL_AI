# Chi Tiết Bước 3: Đào Tạo Chuyên Gia (Huấn luyện Multinomial Naive Bayes)

Trong bước này, hệ thống sẽ tiến hành "học" từ kho dữ liệu đã được tiền xử lý và chuyển đổi thành ma trận số học (TF-IDF). Thuật toán được sử dụng để đóng vai trò "chuyên gia phân loại" là **Multinomial Naive Bayes (MultinomialNB)**. 

## 1. Tại sao lại chọn Multinomial Naive Bayes?
- **Phù hợp với dữ liệu văn bản:** Dữ liệu text thường có tính rời rạc và số chiều rất lớn (hàng ngàn từ vựng). MultinomialNB được thiết kế đặc biệt để xử lý việc đếm tần suất xuất hiện và tỷ lệ của các từ, điều này cực kỳ tương thích với ma trận TF-IDF.
- **Tốc độ cực nhanh:** Có thể huấn luyện trên tập dữ liệu khổng lồ với thời gian tính bằng giây, trong khi kết quả lại ngang ngửa với các mạng nơ-ron phức tạp khi áp dụng trên bài toán phân loại nhị phân (Spam/Ham).

## 2. Hoạt động cốt lõi của thuật toán (Định lý Bayes)
Bản chất của thuật toán dựa trên **Định lý xác suất Bayes**, cụ thể là theo công thức tổng quát:
`P(A|B) = [P(B|A) * P(A)] / P(B)`

Trong đó, áp dụng vào dự án:
- **`A`** là nhãn của bức thư (Spam hoặc Ham).
- **`B`** là những từ ngữ xuất hiện trong bức thư đó (VD: "trúng_thưởng", "iphone",...).
- Hệ thống sẽ tính xác suất: Tỷ lệ để bức thư là Spam khi biết nó chứa những từ ngữ này là bao nhiêu?

Quá trình "học" (Fit model) chính là việc máy đi lưu trữ 2 loại thống kê:
1. **Xác suất tiên nghiệm (Prior Probability):** Tỷ lệ thư Spam và Ham trên toàn bộ tập dữ liệu (VD: 30% Spam, 70% Ham).
2. **Xác suất có điều kiện (Likelihood):** Tỷ lệ xuất hiện của từng từ cụ thể trong tập Spam là bao nhiêu, trong tập Ham là bao nhiêu (VD: Từ "Click_ngay" xuất hiện chủ yếu trong tập Spam, gần như không có ở Ham).

## 3. Quá trình xuất bản "Bộ Não" (Pickle Export)
Sau khi tiêu thụ xong ma trận đặc trưng ở Bước 2, kết quả là mô hình mang trong mình những chỉ số thống kê xác suất gắn liền với từng từ vựng. 

Để ứng dụng này có thể đưa lên Website ở Bước 4 mà không phải "học" lại từ đầu mỗi khi người dùng tải trang, máy sẽ "đóng băng" trạng thái thông minh này và xuất ra dưới dạng **File mã hóa nhị phân (.pkl)**:
- `spam_model.pkl`: Bản đồ tỷ lệ cược và log xác suất của Naive Bayes đã học xong.
- `tfidf_vectorizer.pkl`: Cuốn từ điển ánh xạ từ chữ sang ma trận (bắt buộc phải xuất ra song hành để dùng chiếu xạ cho các input văn bản mới).
