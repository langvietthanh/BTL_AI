import sys
import os
# Cấu hình đường dẫn để có thể import thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# IMPORT hàm của Thành viên 1 ở thư mục data
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
    
    # GỌI HÀM CỦA THÀNH VIÊN 1
    df['cleaned_msg'] = df['message'].apply(clean_text)

    # 3. Vector hóa TF-IDF
    print("Đang phân tích...")
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_msg'], df['label'], test_size=0.2, random_state=42)
    # Phải bắt được từ ghép (ví dụ: "vay_tiền", "trúng_thưởng"), nên dùng ngram_range=(1, 2)
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
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
