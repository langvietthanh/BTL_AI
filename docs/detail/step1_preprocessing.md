# Đi sâu vào Bước 1: Tiền Xử Lý Dữ Liệu (Data Preprocessing)

Tại sao lại gọi Data Engineer (Kỹ sư dữ liệu) là "kẻ làm sạch bữa ăn của AI"? Đây là báo cáo giải trình chuyên sâu dành cho Bước 1 để đối đáp khi bị giảng viên chất vấn.

---

## 1. Bản chất của Tập Dữ Liệu Thô (Raw Data)
Trong thế giới máy học, dữ liệu thô thường được gọi tên là **Garbage (Rác)**. Nếu bạn cho Máy tính ăn rác, kết quả nó nhả ra cũng chỉ là rác *(Nguyên lý Garbage In - Garbage Out)*. 
- Mẫu dữ liệu SMS Spam ban đầu có gì nguy hiểm? 
  - Khách hàng có thói quen viết hoa viết thường lộn xộn: `FREE`, `Free!`, `fRee`. Đám này khiến một chữ gốc bị băm vằn thành 3 chữ khác nhau, làm loãng tính toán học máy.
  - Chữ chứa rất nhiều ký tự lạ: `$1000`, `www.win.com`, `:)`. 
  - Đa cấu trúc: Tiếng Anh thì phân cách bằng phím Tab (file dạng TSV), Tiếng Việt thì mình tự sinh ra bằng phím Phẩy (file dạng CSV).

## 2. Giải mã đoạn mã (Source Code) cực quan trọng: Hàm gọt rửa `clean_text`
Nằm trong file `data/data_preprocessing.py`, hàm này được ứng dụng **Regex (Cỗ máy tìm kiếm dựa trên Biểu thức Chính quy)**.

```python
import re

def clean_text(text):
    text = text.lower()  # Ép in thường toàn bộ chữ
    text = re.sub(r'\W', ' ', text)  # Đóng đinh và xóa sạch kí hiệu lạ
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ các khoảng trắng liền kề
    return text
```

### Tại sao chúng em lại sử dụng 3 dòng lệnh này? Đâu là ý đồ ẩn sâu?

*   **`text.lower()` (Đồng nhất hóa - Homogenization):** Giải quyết triệt để vấn đề phân mảnh tính từ. Đưa mọi bức thư về chung một mặt phẳng là chữ thường.
*   **`re.sub(r'\W', ' ', text)` (Cắt gọt nhiễu âm - Noise Reduction):** Kí hiệu rỗng `\W` (W hoa) trong Biểu thức chính quy đại diện cho "Bất cứ thứ gì KHÔNG PHẢI LÀ CHỮ CÁI hoặc SỐ". Khi lệnh này chạy, các dấu phẩy, dấu chấm, ký hiệu tiền đô la `$` hay link website `.` đều bị chặt chém và đúc thành 1 dấu cách. 
    * *Ví dụ: `100$!` biến thành `100  `*
    * Hành động này giúp mô hình triệt tiêu hiệu ứng nhiễu loạn của chấm câu.
*   **`re.sub(r'\s+', ' ', text).strip()` (Chuẩn hóa Độ dài chiều ngang):** Do lệnh cắt gọt thứ 2 sẽ thường để lại cả tá dấu cách (space) khi cắt đứt kí tự. Dòng lệnh này nén những chùm 5-6 dấu cách đó dồn lại còn 1 dấu cách duy nhất `\s+`. Hàm `.strip()` đánh bay các dấu cách kẹt hai đầu đoạn văn.

## 3. Quá trình Trộn (Concat Data) theo cơ chế Song Ngữ

Vì tập dữ liệu Spam gốc bằng Tiếng Anh vô cùng đồ sộ (hơn 5.500 tin nhắn), nếu mình quăng 30 tin nhắn Tiếng Việt vào thì 30 tin nhắn đó sẽ bị nuốt chửng bởi 5.500 tin tiếng Anh, sức nặng (Weight) của các chữ `"vay tiền"` sẽ coi như = 0 vì độ loãng quá ít.

Nhờ thư viện **Pandas**, chúng ta giải quyết được bằng 1 tiểu xảo: Khai báo kịch bản 30 Tin Nhắn Tiếng Việt giả lập, đưa nó vào file cấu hình và **nhân nó lên 20 lần** (Data Augmentation Form) sinh ra 600 tin nhắn. Cuối cùng thực hiện lệnh `pd.concat([df, df_vn])` để ráp 2 kho ngôn ngữ lồng vào nhau. 

**Kết quả cuối cùng:** Dữ liệu hoàn toàn sạch sẽ, cấu trúc đồng quy, chuẩn hóa chiều ngang và cân bằng Cán Cân 2 Ngôn Ngữ với quy mô trên 6.100 đoạn thoại. Đó là một thành quả không hề tồi trước khi ta bón cho Máy tính.
