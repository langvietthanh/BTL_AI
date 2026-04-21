import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data.data_preprocessing import clean_text

def train_model():
    print("="*50)
    print("MÔ-ĐUN: ĐÀO TẠO MÔ HÌNH (CHO ML ENGINEER)")
    print("="*50)

    # 1. Tải tập dữ liệu
    txt_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "SMSSpamCollection")
    if not os.path.exists(txt_path):
        print(f"Không tìm thấy dữ liệu tại: {txt_path}. Hãy chắc chắn bạn đã tải tập dữ liệu.")
        return

    # 2. Xử lý dữ liệu
    df = pd.read_csv(txt_path, sep='\t', header=None, names=['label', 'message'])
    print(f"Tổng số tin nhắn tiếng Anh: {len(df)}")
    
    # --- MỞ RỘNG TIẾNG VIỆT ---
    vn_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "vietnamese_data.csv")
    if os.path.exists(vn_path):
        df_vn = pd.read_csv(vn_path)
        df = pd.concat([df, df_vn], ignore_index=True)
        print(f"Đã trộn thêm dữ liệu Tiếng Việt. Tổng dataset: {len(df)}")
    

    df['cleaned_msg'] = df['message'].apply(clean_text)

    # 3. Vector hóa TF-IDF (Ép Chữ thành Ma trận Toán học)
    print("Đang phân tích...")
    
    # CHIẾN LƯỢC CHIA ĐỂ TRỊ: Cắt dữ liệu thành 2 mảng (80% để Mô hình Học, 20% giấu đi để Thi Kiểm Tra)
    # Input: df['cleaned_msg'] (cột chứa văn bản dạch) và df['label'] (cột chứa đáp án Spam/Ham tương ứng)
    # Output: X_train (Văn bản học), X_test (Văn bản thi), y_train (Đáp án học), y_test (Đáp án thi)
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_msg'], df['label'], test_size=0.2, random_state=42)
    
    # KHỞI TẠO TỪ ĐIỂN: Gom tối đa 8000 từ quyền lực nhất. Bắt cả từ đơn (1) và từ ghép 2 chữ (2) như "việc_nhẹ", "lương_cao"
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    
    # DÀNH CHO BÀI HỌC (X_TRAIN): Chạy lệnh `.fit_transform()`
    # Input: 80% Lượng chữ viết văn bản để học.
    # Nhiệm vụ: Fit (Tạo cuốn Từ điển bắt 8000 từ) + Transform (Ép dãy chữ đó thành bảng điểm Matrix của riêng nó)
    # Output: X_train_tfidf (Cuộn Ma trận Toán học chứa rặt điểm số của phe học)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # DÀNH CHO BÀI KIỂM TRA (X_TEST): Chỉ chạy lệnh `.transform()`
    # Input: 20% Chữ viết văn bản dùng làm đề thi.
    # Nhiệm vụ: Cấm học từ mới (Không Fit). Bắt buộc phải ép kiểu dựa trên cuốn Từ Điển đã tạo ở trên (Chỉ Transform).
    # Output: X_test_tfidf (Cuộn Ma trận điểm số chuẩn bị đem đi chấm thi)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Train Model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 5. Đánh giá
    y_pred = model.predict(X_test_tfidf)
    print("Đánh giá mô hình:")
    print(f"- Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"- Precision: {precision_score(y_test, y_pred, pos_label='spam')*100:.2f}%")
    print(f"- Recall   : {recall_score(y_test, y_pred, pos_label='spam')*100:.2f}%")

    # 6. Lưu mô hình (Thành viên 2 xuất file đưa Thành viên 3)
    models_dir = os.path.dirname(__file__)
    joblib.dump(model, os.path.join(models_dir, "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print("\n-> Đã lưu thành công `spam_model.pkl` và `tfidf_vectorizer.pkl` vào thư mục models/ !")

if __name__ == "__main__":
    train_model()
