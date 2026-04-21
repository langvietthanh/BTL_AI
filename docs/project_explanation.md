# Thuyết minh Luồng Hoạt Động Cốt Lõi Của Dự Án AI (Project Explanation)

Tài liệu này được biên soạn để hỗ trợ đọc hiểu nhanh cơ chế hoạt động của đồ án Nhận diện Thư Rác (Spam Classification). Phục vụ cho việc viết báo cáo, làm Slide và bảo vệ (defense) trước hội đồng Giảng viên.

Dự án hoạt động như một cỗ máy 4 bước, tuân thủ nguyên lý biến ngôn ngữ tự nhiên thành Toán học phân tích.

---

## Bước 1: Vét Dữ Liệu và Gọt Rửa Dữ Liệu (Tiền xử lý Data)
**Vị trí Code:** `data/data_preprocessing.py`
- Ban đầu, ta có một đống tin nhắn hỗn tạp từ Anh tới Việt (nằm trong thư mục `dataset/`). Những tin nhắn này lộn xộn, sai hoa thường, chứa nhiều kí tự đặc biệt (!, @, #, ?), khoảng trắng thừa.
- Hàm `clean_text()` đóng vai trò như một bộ "gọt chữ": Nó nhằn hết mọi dấu câu ra, ép tất cả các từ thành dạng chữ in thường, và bóp nghẹt các khoảng trắng thừa.
- **Mục tiêu:** Giúp AI không bị hoang mang. Với AI, chữ `Free!!!`, `FrEe`, và `free` vốn là 3 chữ khác nhau. Việc lọc sạch chuẩn hóa về đúng định dạng gốc `free` giúp tiết kiệm siêu cấp bộ nhớ và tối ưu độ chuẩn xác.

## Bước 2: Ép Kiểu Chữ Của Trái Đất Thành Số Của Máy Bước (Giải thuật TF-IDF)
**Vị trí Code:** Đoạn đầu của `models/model_training.py`
Đây là bước cực kỳ đắt giá. Máy phát triển ML không bao giờ "đọc" String, chúng chỉ tính Toán các ma trận.
- Thuật toán `TfidfVectorizer` được sử dụng như một nhà Biên dịch viên. Nó sẽ rà soát toàn bộ từ vựng trong hơn 6.000 tin nhắn để lập ra một cuốn **Từ Điển Toán Học** riêng biệt.
- Nó chấm điểm từ vựng dựa trên cơ chế: **TF** (Tần suất một từ xuất hiện trong câu) nhân với **IDF** (Độ quý hiếm của từ đó ở các câu khác). Một từ chuyên dụng như "trúng_thưởng" mà chỉ hay tập trung ở bên nhánh văn bản Spam, nó sẽ được đẩy thông số điểm lên mức báo động đỏ.
- **Kỹ thuật xử lý từ ghép Tiếng Việt (N-gram):** Thay vì chỉ cắt chéo từng chữ cái lẻ tẻ, nhóm gài thuật toán bằng tham số `ngram_range=(1,2)`. Việc nhóm đi đôi các cặp từ giúp nó hình thành khái niệm định nghĩa cụm từ ghép, giúp AI hiểu "vay tiền" đe dọa cao hơn chữ "vay" đứng độc lập một mình.

## Bước 3: Đào Tạo Chuyên Gia (Huấn luyện Multinomial Naive Bayes)
**Vị trí Code:** Nửa sau của `models/model_training.py`
- Khi mọi câu chữ được chuyển thể thành các Ma Trận Số Học thập phân, chúng ta giao nộp nó cho **Thuật toán Multinomial Naive Bayes (Xác suất Vượt mức)**.
- Bản chất của MultinomialNB hoạt động dựa trên **Định lý Thống kê Bayes**: Nó rút gọn lại kinh nghiệm là *"Nếu 100 lần nhắc đến chữ iphone thì có 99 bức là lừa đảo, vậy bản chất chữ iphone hàm chứa tỷ lệ Spam ác liệt (+X điểm Spam)"*. 
- Quá trình này ngốn rất nhiều sức mạnh tính toán để quy tập tất cả đặc trưng của mọi chữ cái. Sau khi tính xong, chương trình đóng băng và xuất khẩu (export) ra 2 file não bộ không thể chỉnh sửa: `spam_model.pkl` (Bảng thống kê tỷ lệ cá cược) và `tfidf_vectorizer.pkl` (Từ điển chiếu bóng).

## Bước 4: Khởi Động Giao Diện Web Và Cơ Chế XAI Minh Bạch (Explainable AI)
**Vị trí Code:** `app.py`
- Giao diện đập vào mắt người dùng, được sinh ra từ thư viện `Streamlit`.
- Khi người dùng nhập bất kì tin nhắn nào vào hộp thoại, File mã nguồn trực tiếp lôi **Từ Điển** và **Não bộ** bảng điểm (2 bản .pkl) lên hoạt động trên RAM thay vì phải train lại từ đầu. Giúp kết quả trả về trong tích tắc (<0.05 giây).
- **Cơ chế bóc tách Minhh Bạch (Custom XAI - Đỉnh rổ của dự án)**: Giảng viên sẽ luôn thắc mắc là liệu AI học lỏm học vẹt hay thực sự phân loại có cơ sở. Web chứa hệ thống vòng lặp phân tích ngược nội tâm của Bảng Điểm. Tiến hành kiểm tra trọng số (Log_prob) của Nhãn Spam trừ đi Nhãn Ham. Khi điểm độ lệch (diff) > 1.0 (Nặng mùi rác), web sẽ tự động render thẻ bôi viền `HTML Đỏ Đậm` dội vào chữ đấy ngay lập tức. Giúp thuyết phục tuyệt đối Hội Tỷ lệ chấm thi.
- Thêm nữa, thông qua sức mạnh nội tâm đó, tính năng rút lõi các Tín Hiệu Spam Nhạy cảm nhất còn được bơm ra chân Web bằng thuật toán tạo Mây Từ Khoá **(Word Cloud)** để mọi người đắm chìm vào.

---

> 💡 **Tóm gọn 1 câu cực chất để chốt sale với Hội đồng thi:**
> *"Dự án của nhóm chúng em hoạt động dựa trên cơ chế chuẩn hóa dữ liệu đa ngôn ngữ thành các ma trận tần suất mức TF-IDF có nhận diện từ kép (N-gram), sau đó sử dụng thống kê định lý Bayes nhằm suy luận ranh giới tập hợp kết hợp với cơ chế cốt lõi Minh bạch XAI tự diễn giải điểm số của từng từ dưới dạng HTML Rendering để chẩn đoán Thư Lừa Đảo một cách có căn cứ rõ ràng nhất."*
