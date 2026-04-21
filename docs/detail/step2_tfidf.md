# Đi sâu vào Bước 2: Biến Chữ Thành Số (Thuật toán TF-IDF)

Sau khi Bước 1 đã gọt rửa xong đống văn bản thô, chúng ta gặp một rào cản chí mạng: **Thuật toán Trí tuệ Nhân tạo là một cỗ máy làm Toán**. Một ma trận tính toán không bao giờ biết chữ cái "A" hay chữ cái "B" là gì. Nó mù chữ hoàn toàn!

Nhiệm vụ của Bước 2 là đập nát toàn bộ các chữ cái ra và ép kiểu chúng về một hệ thống các con số (Vector hóa - Vectorization).

---

## 1. Bản chất của TF-IDF là gì?
Nhóm chúng ta đã sử dụng hàm `TfidfVectorizer` của thư viện Toán học `scikit-learn`. TF-IDF là chữ viết tắt của 2 thành phần độc lập gộp lại:

*   **TF (Term Frequency - Tần suất xuất hiện của từ):** 
    Hiểu đơn giản, nếu chữ "tiền" xuất hiện 5 lần trong một tin nhắn ngắn, hàm TF sẽ cộng cho nó một số điểm bự. Hàm này tin rằng chữ nào lặp lại nhiều tức là chữ đó là Linh hồn, là trọng tâm của đoạn văn.

*   **IDF (Inverse Document Frequency - Nghịch đảo Tần suất Tập tài liệu):** 
    Nhưng, có một kẽ hở! Các chữ dạng như "thì, là, mà, ở, the, is, a..." lặp lại vô vàn trong mọi tin nhắn. Nếu chỉ dùng mỗi TF, chữ "là" sẽ có điểm cao vót, đánh lừa Máy học. 
    Để trị tận gốc bệnh này, IDF ra đời. Hàm IDF đi soi từ vựng đó trong *tồn bộ 6.100 tin nhắn dự án*. Nếu chữ đó xuất hiện ở mọi nơi, mọi nhà, IDF sẽ **trừ điểm thẳng tay**, kéo trọng lượng chữ đó về sát mức 0 (coi như rác, không mang giá trị nhận diện). Ngược lại, nếu một chữ vô cùng sắc bén như "trúng_thưởng" chỉ lẻ tẻ xuất hiện ở bên nhánh Spam, điểm IDF của khu vực đó sẽ bùng nổ lên rất vĩ đại.

> **Tóm gọn (TF-IDF = TF nhân với IDF):** Nó hoạt động như một cỗ máy lọc vàng. Tôn vinh nhấc bổng những từ đắt giá tạo nên danh tính của tin nhắn, và dìm chết vùi dập khối từ ngữ dư thừa phổ thông. Một cách xử lý chữ nghĩa ở Đẳng cấp Cao!

## 2. Giải mã các Tham Số (Parameters) ngầm định cực xịn
Lướt qua dòng lệnh trong file `models/model_training.py`, giảng viên sẽ thấy nhóm gọi hàm:
```python
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
```
Thoạt nhìn thì ngắn, nhưng đẳng cấp Kỹ sư nằm ở lõi ngoặc đơn này. Khi Báo cáo bạn phải xoáy mạnh vào 2 thông số này:

### A. Siêu giới hạn `max_features=8000`
Khi Vector hóa (Ép kiểu Toán) 6.100 tin nhắn, tổng lượng từ vựng độc nhất có thể lên tới 50.000 từ. Nếu bắt máy tính vẽ một cái biểu đồ hệ tọa độ 50.000 chiều (Dimensions), máy tính sẽ đối mặt lời nguyền "Curse of Dimensionality" (Lời nguyền Đa chiều), dẫn đến nổ RAM hoặc chạy chậm như rùa bò. 
*   Bằng việc chèn `max_features`, chúng ta nói với AI: *"Hãy chặt đứt cây, chỉ lấy đúng 8.000 Từ khóa mạnh mẽ, xịn xò nhất, có điểm TF-IDF tàn khốc nhất. Phần từ vựng lai nhai còn lại vứt bỏ để đẩy tốc độ Web lên mức Ánh sáng."*

### B. Vị cứu tinh của Tiếng Việt - `ngram_range=(1, 2)`
Bình thường TF-IDF có chế độ `(1,1)` (Unigram). Tức là mỗi chữ là 1 đơn vị. (Ví dụ: "Hôm - nay - em - vay - tiền" -> Máy sẽ hiểu đây là 5 từ tách biệt). Rất ổn với Tiếng Anh.
**Nhưng với Tiếng Việt thì bị Phá sản!** Tiếng Việt là tiếng Kép.
Chữ `vay` (Có thể là "vay mượn" hay "vay cánh") và chữ `tiền` (Có thể là "tiền bạc" hay "mặt tiền"). 
*   Khi ta thêm tham số `(1, 2)`, ta ép thuật toán đẻ ra thêm khái niệm **Bi-gram (Bắt cặp từ liền kề)**. Lúc này cuốn Từ điển của AI sẽ nhét nguyên cụm từ `"vay tiền"` vào thành một khái niệm Toán học duy nhất! Không bị tách rách rời. 
*   Đây chính là Nguồn Gốc Sức Mạnh khiến con AI hiện tại của dự án hiểu cực kì sâu sắc các sắc thái lừa đảo của Tiếng Việt (như: "hack like", "khóa thẻ", "nhận quà").

---
Ghim ngay đoạn văn này vào đầu, và bạn sẵn sàng làm gỏi mọi câu hỏi khó nhằn thuộc về Lý thuyết Xử lý Ngôn ngữ Tự nhiên (NLP)!
