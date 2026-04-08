import re
import urllib.request
import zipfile
import os

def download_data(zip_url, output_zip, txt_output_name):
    """Hàm hỗ trợ tải dữ liệu nếu chưa có."""
    if not os.path.exists(txt_output_name):
        print(f"Đang tải {txt_output_name}...")
        urllib.request.urlretrieve(zip_url, output_zip)
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Tải dữ liệu hoàn tất!")
    else:
        print(f"Dữ liệu {txt_output_name} đã sẵn sàng.")

def clean_text(text):
    """
    Module chung để làm sạch văn bản (Tiền xử lý).
    Bất kỳ chuỗi văn bản nào cũng đi qua đây để lọc dấu câu, viết thường.
    """
    text = text.lower()  # in thường
    text = re.sub(r'\W', ' ', text)  # loại bỏ dấu câu, ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()  # xoá khoảng trắng thừa dư
    return text
