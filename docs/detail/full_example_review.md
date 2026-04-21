# BÍ KÍP ÔN THI: BẢN ĐỒ HUẤN LUYỆN AI TỪ A ĐẾN Z BẰNG VÍ DỤ TRỰC QUAN

Tài liệu này chứa một "Vũ trụ thu nhỏ" của cả đồ án, giúp bạn hình dung dòng chảy của hệ thống theo từng bước một cách trần trụi nhất, đặc biệt có **kèm theo chi tiết tính toán công thức Toán học điểm 10** để bạn vững vàng phản biện.

---

## BƯỚC 1: TIỀN XỬ LÝ (GỌT RỬA DỮ LIỆU)

**Kịch Bản Nhập Liệu (Hệ thống có 3 bức thư, N = 3):**
* Thư 1 (Nhãn Bình Thường - Ham): `"TÔI ĐI HỌC!!"`
* Thư 2 (Nhãn Bình Thường - Ham): `"Tôi đi... LÀM.. @#"`
* Thư 3 (Nhãn Spam): `"**TÔI Trúng_thưởng** !!!!"`

Hàm `clean_text()` băm nát và gọt sạch chúng về danh sách tinh khiết:
1. `tôi đi học` (Ham) - Số từ: 3
2. `tôi đi làm` (Ham) - Số từ: 3
3. `tôi trúng_thưởng` (Spam) - Số từ: 2

---

## BƯỚC 2: TF-IDF ("BIÊN DỊCH VIÊN" ÉP CHỮ THÀNH ĐIỂM)

TF-IDF không đoán ngẫu nhiên, nó dựa vào 2 công thức Toán học vô cùng chặt chẽ để diệt từ rác và tôn vinh từ khóa:

### 1. Phân Tích Chữ `"tôi"` (Từ phổ thông vô giá trị)
* **TF (Tần suất trong câu - Term Frequency):** Xem chữ đó xuất hiện với mật độ bao nhiêu trong bức thư. 
  👉 Ở Thư 3, có 2 chữ, chữ `"tôi"` chiếm 1 chữ -> `TF(tôi) = 1/2 = 0.5`.
* **IDF (Độ hiếm - Inverse Document Frequency):** Công thức: `Log10 (Tổng thư / Số thư chứa chữ đó)`. 
  Cả vũ trụ có 3 thư, chữ `"tôi"` xuất hiện ở cả 3.
  👉 `IDF(tôi) = Log10 (3/3) = Log10(1) = 0`.
* **🔥 Chốt sổ TF-IDF (Nhân lại):** `TF * IDF = 0.5 * 0 = 0 Điểm`.
*(Chữ "tôi" chính thức bị phế võ công, điểm bằng 0 tuyệt đối!)*

### 2. Phân Tích Chữ `"trúng_thưởng"` (Từ khóa tử thần)
* **TF (Tần suất):** Ở Thư 3, nó chiếm 1 vị trí trong câu có 2 chữ -> `TF(trúng_thưởng) = 1/2 = 0.5`.
* **IDF (Độ hiếm):** Cả vũ trụ có 3 thư, chỉ đúng 1 thư có nó.
  👉 `IDF(trúng_thưởng) = Log10 (3/1) = Log10(3) ≈ 0.477`.
* **🔥 Chốt sổ TF-IDF (Nhân lại):** `TF * IDF = 0.5 * 0.477 = 0.2385 Điểm`.

**Kết Quả Bước 2:** TF-IDF đã sản xuất ra một **"Ma trận Trọng Số"**. Trong dòng của Thư Spam 3, khối u "trúng_thưởng" nhô lên với `0.2385 điểm`, che lấp hoàn toàn chữ "tôi" khuyết tật mang `0 điểm`.

> **👀 BẢNG MÔ PHỎNG TRỰC QUAN QUÁ TRÌNH VECTOR HÓA BƯỚC 2:**
> Thay vì đem nguyên cẩu thả câu chữ đi tính toán, máy tính chỉ làm việc với cái mảng dãy số này. Từ vựng biến thành Tiêu đề Cột phân tích, còn câu văn biến thành mảng số liệu trọng lượng. Nhờ TF-IDF, chữ nào quá phổ thông vô giá trị (như chữ "tôi") bị trừng phạt liệt về số `0`, chữ nào đắt giá độc quyền bị đẩy số lên cực đỏ.
> 
> | Dữ Liệu Văn Bản Nhập Vào | [tôi] | [đi] | [học] | [làm] | [trúng_thưởng] | 🚀 Thành phẩm (Dãy Vector Toán Học) |
> |:---|:-:|:-:|:-:|:-:|:-:|:---|
> | **Thư 1:** `"tôi đi học"` | `0` | `0.3` | `0.3` | `0` | `0` | 👉 `[0, 0.3, 0.3, 0, 0]` |
> | **Thư 2:** `"tôi đi làm"` | `0` | `0.3` | `0` | `0.3` | `0` | 👉 `[0, 0.3, 0, 0.3, 0]` |
> | **Thư 3 (Spam):** `"tôi trúng_thưởng"` | **`0`** | `0` | `0` | `0` | **`0.238`** | 👉 `[0, 0, 0, 0, 0.238]` |
> 
> *(Lưu ý: Cái mảng dãy số li ti sinh ra từ Thư số 3 kia mới chính là thứ máy tính ném thẳng vào bộ não Naive Bayes ở Bước 3)*

---

## BƯỚC 3: NAIVE BAYES ("HỘI ĐỒNG" HỌC THUỘC MA TRẬN)

Thuật toán `MultinomialNB` sử dụng công thức toán học nội tại để nuốt Ma trận Trọng số ở Bước 2. Công thức cốt lõi là:
`P(Từ_Khóa | Lớp) = (Tổng điểm TF-IDF của Từ_Khóa đó trong nhóm + Alpha) / (Tổng ngân lượng TF-IDF của toàn nhóm + Alpha * Tổng số từ vựng)`
*(Với `Alpha = 1` là cơ chế Laplace Smoothing để tránh tử số bằng 0. Tổng số từ vựng từ điển là 5: tôi, đi, học, làm, trúng_thưởng).*

Bây giờ chúng ta tính Xác suất Ưu tiên (Sổ kinh nghiệm) cho nhóm Spam và Ham:

**1. Tỷ lệ Tiên nghiệm (Tỷ lệ đụng độ cơ bản):**
Có 3 bức thư, 1 Spam và 2 Ham.
👉 `P(Spam) = 1/3 ≈ 0.33`
👉 `P(Ham) = 2/3 ≈ 0.67`

**2. Học từ Lãnh địa Spam (Tổng ngân lượng TF-IDF của nhóm Spam = 0.238):**
* **Điểm của chữ `"trúng_thưởng"`:**
  👉 `P(trúng_thưởng | Spam) = (0.238 + 1) / (0.238 + 1 * 5) = 1.238 / 5.238 ≈ 0.236`.
* **Điểm của chữ `"tôi"`:** Do TF-IDF chấm nó 0 điểm.
  👉 `P(tôi | Spam) = (0 + 1) / (0.238 + 1 * 5) = 1 / 5.238 ≈ 0.190`.

**3. Học từ Lãnh địa Bình thường - Ham (Tổng ngân lượng TF-IDF nhóm Ham = `0.3 + 0.3 + 0.3 + 0.3` = 1.2):**
* **Điểm của chữ `"trúng_thưởng"`:** Không hề xuất hiện trong tập Ham (điểm 0).
  👉 `P(trúng_thưởng | Ham) = (0 + 1) / (1.2 + 1 * 5) = 1 / 6.2 ≈ 0.161`.
* **Điểm của chữ `"tôi"`:**
  👉 `P(tôi | Ham) = (0 + 1) / (1.2 + 1 * 5) = 1 / 6.2 ≈ 0.161`.

*(So sánh nhanh: Rõ ràng `P(trúng_thưởng | Spam) = 0.236` LỚN HƠN RẤT NHIỀU so với `P(trúng_thưởng | Ham) = 0.161` trong bản ghi nhớ).*
  
> **🔥 LÝ DO BẮT BUỘC SỬ DỤNG LAPLACE SMOOTHING (CÂU HỎI ĂN ĐIỂM):**
> Nếu hệ thống đếm ra số `0`, xác suất `P(trúng_thưởng | Ham)` sẽ chết cứng ở mức `0`. 
> Toán học quy định: Một chuỗi phép nhân xác suất liên tiếp `(A * B * C...)` mà xui xẻo có một phần tử bằng `0` thì **TẤT CẢ sẽ sụp đổ bằng 0 hết**. 
> Ví dụ thư có chữ "tôi", "đi", "ngủ" đếu mang xác suất thuộc Ham rất cao, nhưng lỡ lọt một chữ "trúng_thưởng" lạ hoắc (0 điểm). Rập khuôn nhân lại, Xác suất thuộc về Ham của cả câu đó bị đánh sập thành 0. Hoàn toàn vô lý chỉ vì 1 chữ cái!
> 👉 **Giải pháp Laplace Smoothing (Làm mịn +1):** Thuật toán tự động tịnh tiến, **cộng thêm siêu nhỏ (thường là +1 hoặc +Alpha)** vào tử số của tất cả mọi từ. Chữ "trúng_thưởng" từ chỗ 0 lần (vô danh) sẽ được vớt vát thành 1 lượng siêu bé. Nó tạo ra một tỷ lệ mỏng manh dạt dẹo như hạt cát nhưng **TUYỆT ĐỐI KHÔNG BẰNG 0**. Thao tác này cứu sống toàn bộ cấu trúc định lý Bayes không bị sập chuỗi biến số!

  👉 Chốt lại nhờ Smoothing: `P(trúng_thưởng | Ham) = Điểm Mặc định Laplace (Bé như hạt cát nhưng sống dai vì lớn hơn 0)`.

---

## BƯỚC 4: THAO TÁC ĐÓNG BĂNG TRI THỨC (XUẤT PICKLE DÙNG 1 LẦN)

Sau khi Naive Bayes tính xong bảng Xác suất và TF-IDF chốt xong bảng Ma trận, nhóm lập trình gọi lệnh Export xuất ra ổ cứng:
1. `tfidf_vectorizer.pkl`: Từ Điển tra mã Toán Học của riêng kho dữ liệu.
2. `spam_model.pkl`: Sổ Kinh Nghiệm tích trữ bộ Xác suất đã tính sẵn.
**Lý do:** Mọi vất vả tính toán phân số (P) ở trên sẽ chỉ CHẠY 1 LẦN DUY NHẤT.

---

## BƯỚC 5: GIAO DIỆN KIỂM SOÁT THỰC CHIẾN (TÁI SỬ DỤNG - WEB & XAI)

Giờ đến lúc thầy giáo dùng màn hình Web App Streamlit để chất vấn hệ thống.
Thầy tự gõ thử: **"Tôi trúng_thưởng"** xem hệ thống làm ăn ra sao.

Web nhấc file `.pkl` lên vứt vào RAM ngốn chưa tới 0.05 giây. Hệ thống đưa nạn nhân đi qua cửa ải phán quyết:

**Luật Cộng Logarit Tránh Tràn Số (Toán Học Bí Mật của thuật toán):**
Thay vì nhân, Máy tính sẽ kiểm tra bằng Hàm Tổng Logarit:
* **Tính tổng điểm Đội SPAM:** 
  `Log(P(Spam))` + `Log(P(tôi | Spam))` + `Log(P(trúng_thưởng | Spam))`
* **Tính tổng điểm Đội HAM:**
  `Log(P(Ham))` + `Log(P(tôi | Ham))` + `Log(P(trúng_thưởng | Ham))`

**Diễn biến thực tế bên trong Chip Vi Xử Lý:**
1. Rà qua chữ `"tôi"`, hệ thống thấy Điểm ánh xạ TF-IDF = 0. Máy gạt thẳng phăng chữ này sang một bên, nó đóng đinh không liên quan gì đến ván cược này.
2. Rà qua chữ `"trúng_thưởng"`: 
   * Đội Spam khoái chí vì Log xác suất chữ này trong kho của phe mình là siêu Dương, chỉ số `Log(P|Spam)` ở đỉnh chóp.
   * Đội Ham lụi bại vì Log xác suất chữ này bị kéo lùi do màng Smoothing, chỉ số `Log(P|Ham)` thụt lùi xuống đáy đại dương.
=> TỔNG KẾT BÙ TRỪ: Nhóm SPAM thắng áp đảo tuyệt đối vì có tổng số điểm cao hơn (lớn hơn). Đánh luôn dấu **THƯ RÁC (SPAM)**.

**Lưới trừng phạt bằng XAI (Sự lợi hại của code Web):**
Hàm tự thiết kế trong file `app.py` sẽ làm việc giải trình thay mồm con người. Nó lấy Dao trừ đi hai trọng số: 
`Độ Lệch (Diff) = Cột Log_prob(Spam) trừ đi Cột Log_prob(Ham) của chữ trúng_thưởng`. 
* 👉 Nó phát hiện Độ Lệch này trồi lên một cục (Dương khổng lồ). Nóng máu, trình Web vẩy ngay thẻ bôi màu `<html> Đỏ Đậm` nhúng vào thẳng chữ `"trúng_thưởng"` chiếu lên màn hình.

Thầy giáo nhìn thấy viền đỏ lòm đúng từ khóa chết người, và nghe xong bạn kể lưu loát câu chuyện đi từ Log(1) = 0 ở bước 2 đến màn bù trừ Log ở Bước 5. Không còn gì để nghi ngờ, bạn chắc chắn am hiểu code. Điểm tuyệt đối thuộc về nhóm bạn!
