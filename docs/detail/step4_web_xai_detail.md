# Chi Tiết Bước 4: Giao Diện Web Và Cơ Chế XAI Minh Bạch (Explainable AI)

Sau khi đào tạo xong mô hình, hệ thống cần một cách tương tác thân thiện với người dùng cuối, đồng thời chứng minh cho Hội đồng thi thấy AI hoạt động như thế nào và dựa trên cơ sở gì mà không phải là một chiếc "Hộp đen" (Black box).

## 1. Streamlit - Framework Trình Diễn Web
- Giao diện được xây dựng bằng **Streamlit**, một framework mã nguồn mở của Python giúp xây dựng cấu trúc Data Apps nhanh chóng mà không cần code nhiều HTML/CSS/JS.
- Hệ thống ngay lập tức nạp 2 file `spam_model.pkl` và `tfidf_vectorizer.pkl` lên RAM của server.
- **Tiền xử lý Real-time:** Bất kỳ câu từ nào người dùng nhập vào Form đều trải qua một quy trình y hệt lúc train. Chúng bị `clean_text()` gọt rửa, sau đó được `vectorizer.transform()` biến thành ma trận để đưa vào cho Naive Bayes dự đoán.

## 2. Điểm Nhấn Cốt Lõi: Custom XAI (Giải thích AI Minh Bạch)
Trong học máy truyền thống, AI chỉ nhả ra lệnh "Đây là Spam (99%)". Nhưng Giảng viên sẽ chất vấn: *Vì sao AI lại dám khẳng định như vậy? Nó dựa vào từ nào để phán xét?* 

Cơ chế XAI phân giải thuật toán của dự án sinh ra để giải quyết bài toán này:
1. **Giải phẫu thuật toán Naive Bayes:** Bóc tách trọng số bên trong bộ não. Thuật toán có thuộc tính `feature_log_prob_` lưu trữ giá trị log-xác suất của từng từ ở dạng mảng 2 chiều (Lớp Spam và Lớp Ham).
2. **Đo lường độ lệch tiêu chuẩn (Diff):** 
   - Với mỗi từ vựng trong tin nhắn mà người dùng vừa nhập, ta đối chiếu log_prob trong lớp Spam trừ đi log_prob trong lớp Ham: `Diff = Log_prob(Spam) - Log_prob(Ham)`.
   - Nếu `Diff` lớn hơn một Ngưỡng chặn (Threshold) do ta tùy ý thiết lập (Ví dụ `Diff > 0.5` hoặc `1.0`), ta kết luận từ khóa đó đã cấu thành **động cơ chính** kéo bức thư về nhãn Spam.
3. **HTML Rendering trực quan:** Trên giao diện Streamlit, ta dùng hàm `st.markdown(..., unsafe_allow_html=True)` để chủ động nhúng code CSS vào văn bản. Những từ khóa "vượt ngưỡng" sẽ bị bôi viền đỏ hoặc tô đậm (`<mark>`). Giúp người trải nghiệm nhìn bằng mắt thường cũng hiểu được logic phán đoán của AI.

## 3. Nhận Diện Vĩ Mô - Tạo Mây Từ Khóa (Word Cloud)
Ngoài kiểm tra cá nhân từng câu chữ, hệ thống cung cấp góc nhìn toàn cảnh về toàn bộ bộ dữ liệu rác thông qua Word Cloud.
- Hàm vẽ biểu đồ sẽ xuất ra các từ vựng xuất hiện lặp lại một cách nguy hiểm nhất.
- Những chữ như "Miễn_phí", "Vay", "Trúng_thưởng", v.v., sẽ được in rất to trên màn hình tùy theo chỉ số của nó ở trong mô hình thuật toán.
- **Tác dụng:** Cho người đọc nắm bắt Data Insight (Sự thật ngầm hiểu) của một tập dữ liệu hàng nghìn dòng chỉ chưa mất đến 5 giây. Đề cao kỹ năng tư duy làm Data Science của nhóm thao tác.
