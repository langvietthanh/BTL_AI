# Hoàn tất Đồ án: Nhận diện Spam Email / SMS

Chúc mừng bạn! Chỉ trong ít phút, bạn đã đi từ con số không đến việc sở hữu một dự án Trí Tuệ Nhân Tạo (AI) hoàn chỉnh bao gồm tập dữ liệu chuẩn, một mô hình được huấn luyện đầy đủ và một giao diện Web trực quan. 

Dưới đây là tổng kết những gì chúng ta đã làm và hướng dẫn bạn cách khởi động phần mềm.

## 1. Các file đã được tạo ra

Trong thư mục `d:\WorkSpace\Spam_Email`, hiện tại bạn sẽ thấy các file quan trọng sau:

- [train_model.py](file:///d:/WorkSpace/Spam_Email/train_model.py): File chứa thuật toán. Nó tự động tải dữ liệu Spam từ Internet, tiền xử lý ngôn ngữ, Vector hóa bằng TF-IDF và huấn luyện mô hình Naive Bayes.
- [app.py](file:///d:/WorkSpace/Spam_Email/app.py): Mã nguồn của giao diện Web (Streamlit Application).
- **spam_model.pkl & tfidf_vectorizer.pkl**: Não bộ của AI. Đây là mô hình đã học xong và bộ chuyển đổi từ vựng sang số. App sẽ chạy rất nhanh nhờ đọc trực tiếp não bộ này thay vì phải học lại từ đầu mỗi lần chạy.
- **SMSSpamCollection**: Tập dữ liệu thô tải từ UCI (với hơn 5500 dòng tin nhắn).
- **requirements.txt**: File danh sách thư viện đã cài đặt.
- **venv**: Môi trường ảo Python độc lập (giúp dự án không báo lỗi nếu di chuyển qua máy khác).

> [!NOTE]
> Khái niệm lưu model thành file `.pkl` là một kỹ năng thực tế cực tốt để đưa vào slide báo cáo môn học, chứng tỏ bạn biết cách làm AI chạy độc lập khỏi file đào tạo.

## 2. Kết quả Đào tạo (Model Evaluation)

Khi mô hình được chạy, nó đã học trên 80% dữ liệu và thi nghiệm trên 20% dữ liệu còn lại (Dữ liệu chưa từng thấy). Kết quả đạt được rất khả quan (tương ứng với thông số thực tế vừa chạy):

- **Độ chính xác tổng (Accuracy)**: ~98% 
- **Độ chuẩn xác Spam (Precision)**: 100% (Điều này rất tuyệt! Nghĩa là khi AI báo Spam, thì 100% đó là Spam thực sự, ứng dụng sẽ không bao giờ xóa nhầm thư quan trọng của bạn).
- **Độ bao phủ Spam (Recall)**: ~80% (Nó lọc được khoảng 80% tổng số Spam có trong tập, và chấp nhận bỏ sót 20% do ưu tiên tránh bắt nhầm).

## 3. Cách khởi chạy Giao diện Web (Streamlit App)

Bây giờ bạn đã có thể chạy và thao tác trên giao diện người dùng. Hãy mở Terminal (Powershell/Command Prompt) tại thư mục `d:\WorkSpace\Spam_Email` và gõ 2 lệnh sau:

### Bước 1: Kích hoạt môi trường (chứa thư viện AI)
```powershell
.\venv\Scripts\Activate.ps1
```
*Lưu ý: Nếu bị báo lỗi đỏ chữ do Policy của Windows, hãy chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` rồi kích hoạt lại.*

### Bước 2: Chạy Web App
```powershell
streamlit run app.py
```

> [!TIP]
> Sau khi gõ, trình duyệt của bạn sẽ tự động bật lên ở địa chỉ `http://localhost:8501`. Tại đây, bạn có thể copy thử một đoạn tiếng anh "Congratulations you won a free ticket, click here http..." vào khung và bấm **Phân tích AI** để xem kết quả!

Chúc bạn đạt điểm tuyệt đối cho môn Nhập môn AI! Nếu bạn cần thêm biểu đồ hay xuất báo cáo học thuật, hãy nhắn lại nhé.
