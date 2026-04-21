import pandas as pd

# Tập dữ liệu mẫu thực tiễn: Những tin nhắn mà sinh viên Việt Nam hay gặp
vi_data = [
    # ---- Dữ liệu SPAM ----
    ('spam', 'Chúc mừng bạn đã trúng thưởng 1 chiếc iPhone 15 Pro Max, xin hãy truy cập link để điền thông tin nhận thưởng.'),
    ('spam', 'Cho vay tín chấp không cần thế chấp, giải ngân nhanh trong 5 phút, liên hệ Zalo 09xx'),
    ('spam', 'Tuyển nhân viên làm thêm tại nhà, việc nhẹ lương cao, thu nhập 500k-1 triệu/ngày, inbox để biết thêm chi tiết.'),
    ('spam', 'Quay số trúng thưởng xe máy Honda SH. Soạn tin nhắn gửi tổng đài để nhận phần thưởng.'),
    ('spam', 'Xổ số lô đề chuẩn, dự đoán kết quả chính xác 99%, soi cầu vip, cam kết trúng thưởng.'),
    ('spam', 'Dịch vụ tăng like, hack follow, mua sub youtube giá rẻ uy tín nhất.'),
    ('spam', 'Mời bạn tham gia khóa học làm giàu nhanh chóng, đầu tư sinh lời 300% mỗi tháng.'),
    ('spam', 'Vay tiền online lãi suất 0% cho lần vay đầu tiên, thủ tục chỉ cần CMND.'),
    ('spam', 'Tặng thẻ cào 500k cho khách hàng click vào đường link sau đây.'),
    ('spam', 'Làm quen kết bạn bốn phương, các em gái xinh đêm nay cô đơn gọi zalo ngay nha.'),
    ('spam', 'KM HOT: Giảm 50% toàn bộ gói cước Data 4G tốc độ cao, đăng ký ngay hôm nay.'),
    ('spam', 'Sở hữu ngay siêu phẩm căn hộ cao cấp chỉ với 1 tỷ đồng, mặt tiền đắc địa trung tâm thành phố.'),
    ('spam', 'Nhận ngay giftcode 1 triệu VNĐ khi đăng ký tài khoản game bài đổi thưởng, nhà cái uy tín Châu Á.'),
    ('spam', 'Thông báo khóa tài khoản ngân hàng, vui lòng truy cập đường dẫn www.bank-verify-vn.com để xác minh.'),
    ('spam', 'Bán sim giá rẻ, sim số đẹp phong thủy, lộc phát tài, gọi hotline để mua ngay.'),

    # ---- Dữ liệu HAM ----
    ('ham', 'Hôm nay trưa ăn gì vậy mọi người? Đi ăn bún chả không?'),
    ('ham', 'Trưởng phòng ơi, em đã gửi file tiến độ báo cáo tuần này qua email, anh kiểm tra nhé.'),
    ('ham', 'Nhớ ghé siêu thị mua cho mẹ chai mật ong nha con.'),
    ('ham', 'Tối thứ Bảy này hẹn nhóm mình ở quán cafe góc phố lúc 8h tối nha.'),
    ('ham', 'Cuối tuần này bạn có rảnh không, đi xem phim với mình nhé.'),
    ('ham', 'Alo nãy gọi không được, lúc nào rảnh gọi lại nhé đang bận họp.'),
    ('ham', 'Em ơi chuẩn bị phòng họp cho khách hàng lúc 2h chiều giúp chị, dọn nước non tử tế.'),
    ('ham', 'Cô giáo thông báo lớp mình mai nghỉ học, sẽ học bù vào chiều thứ Sáu nhé. Các bạn chú ý lịch.'),
    ('ham', 'Tiền nhà tháng này 3 triệu nhé, chuyển khoản vào số cũ cho bác.'),
    ('ham', 'Anh ơi kiểm tra lại giúp em đơn hàng mã số VNC1234, khách báo chưa nhận được.'),
    ('ham', 'Chiều nay về sớm ghé đón con học võ giùm em nha, em bận làm nốt báo cáo.'),
    ('ham', 'Sinh nhật sếp mua bánh kem gì cho hợp lý mấy đứa ơi?'),
    ('ham', 'Thảo luận môn Nhập môn AI chiều nay ở thư viện nha, ai đến được comment.'),
    ('ham', 'Oke bro, tối nay tao qua đón lúc 7 rưỡi nha.'),
    ('ham', 'Dạ thầy ơi cho em xin phép nộp bài trễ 1 ngày vì bị sốt xuất huyết ạ.')
]

# Nhân bản dữ liệu giả lập lên để làm cho trọng số Tiếng Việt có "sức nặng" cân bằng với bộ 5.500 tin Tiếng Anh
augmented_data = vi_data * 20  # Nhân 20 lần = 600 tin nhắn Tiếng Việt

import os
df_vi = pd.DataFrame(augmented_data, columns=['label', 'message'])
output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "vietnamese_data.csv")
df_vi.to_csv(output_path, index=False)
print(f"Đã tạo tập dữ liệu tiếng việt với {len(df_vi)} dòng tại {output_path}")
