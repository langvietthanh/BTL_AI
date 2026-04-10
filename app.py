import streamlit as st
import joblib
import os

# 1. IMPORT HÀM TỪ THÀNH VIÊN DATA ENGINEER
from data.data_preprocessing import clean_text

# 2. KHỞI TẠO ĐƯỜNG DẪN TỚI NÃO BỘ AI (THÀNH VIÊN ML ENGINEER)
MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Load mô hình và bộ vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Chưa tìm thấy mô hình. Hãy nhắc \u201cThành viên 2\u201d chạy file `models/model_training.py` trước để huấn luyện AI.")
    st.stop()

# Xây dựng giao diện Web (UI)
st.set_page_config(page_title="Spam Classifier (Nhóm 3 Người)", page_icon="🚫", layout="wide")

# Khởi tạo danh sách lịch sử nếu chưa có
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🚫 Hệ Thống Nhận Diện Spam AI")
st.markdown("---")

# Tạo 2 cột: Cột trái (phân tích) chiếm tỷ lệ 40%, cột phải (lịch sử) chiếm tỷ lệ 60%
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("Kiểm tra tin nhắn / Email")
    user_input = st.text_area("Nhập nội dung vào bên dưới để kiểm tra:", height=150, placeholder="Vd: Chúc mừng bạn đã trúng thưởng iPhone 15, hãy click vào link sau...")

    if st.button("Phân tích"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản để dự đoán.")
        else:
            # Tiền xử lý (Sử dụng code Tái chế từ Thành viên 1)
            cleaned_msg = clean_text(user_input)
            
            # Biến đổi text thành số (Vector Hóa)
            vectorized_msg = vectorizer.transform([cleaned_msg])
            
            # Nhận diện Spam/Ham (Sử dụng Model từ Thành viên 2)
            prediction = model.predict(vectorized_msg)[0]
            
            # Tính phần trăm tự tin (Probability)
            proba = model.predict_proba(vectorized_msg)[0]
            confidence = max(proba) * 100
            
            # Hiển thị kết quả ra màn hình
            st.markdown("---")
            st.subheader("Kết quả dự đoán:")
            
            if prediction == "spam":
                st.error(f"🚨 **ĐÂY LÀ THƯ RÁC (SPAM)**")
                st.write(f"Độ tin cậy của AI: **{confidence:.2f}%**")
            else:
                st.success(f"✅ **ĐÂY LÀ THƯ BÌNH THƯỜNG (HAM)**")
                st.write(f"Độ tin cậy của AI: **{confidence:.2f}%**")
            
            # Lưu kết quả vào lịch sử
            st.session_state.history.append({
                "Nội dung": user_input,
                "Phân loại": "SPAM 🚨" if prediction == "spam" else "HAM ✅",
                "Độ tin cậy (%)": round(confidence, 2)
            })

with col2:
    # -----------------
    # BẢNG LICH SỬ
    # -----------------
    st.subheader("📊 Lịch sử phân tích")

    if st.session_state.history:
        import pandas as pd
        # Hiển thị tin mới phân tích lên đầu
        df_history = pd.DataFrame(st.session_state.history)[::-1]
        
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("Chưa có tin nhắn nào được phân tích.")
