import streamlit as st
import streamlit.components.v1 as components
import joblib
import os
from wordcloud import WordCloud

# 1. IMPORT HÀM TỪ THÀNH VIÊN DATA ENGINEER
from data.data_preprocessing import clean_text

# 2. KHỞI TẠO ĐƯỜNG DẪN TỚI NÃO BỘ AI
MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Load mô hình và bộ vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Chưa tìm thấy mô hình")
    st.stop()

# Xây dựng giao diện
st.set_page_config(
    page_title="Spam Classifier", 
    page_icon="🚫", 
    layout="wide"
)

# Khởi tạo danh sách lịch sử nếu chưa có
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Hệ Thống Nhận Diện Spam AI")
st.markdown("---")

# Tạo 2 cột: Cột trái (phân tích) chiếm tỷ lệ 40%, cột phải (lịch sử) chiếm tỷ lệ 60%
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("Kiểm tra tin nhắn / Email")
    user_input = st.text_area("Nhập nội dung vào bên dưới để kiểm tra:", height=150, placeholder="Vd: Chúc mừng bạn đã trúng thưởng iPhone 15, hãy click vào link sau...")

    # --- TÍNH NĂNG IMPORT FILE .TXT ---
    st.markdown("**Hoặc tải lên file văn bản (.txt) để kiểm tra:**")
    uploaded_file = st.file_uploader(
        "Chọn file .txt",
        type=["txt"],
        help="Upload file văn bản .txt — nội dung trong file sẽ được đọc và kiểm tra tự động.",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            file_content = uploaded_file.read().decode("latin-1")

        st.info(f"📄 Đã đọc file **{uploaded_file.name}** ({len(file_content)} ký tự). Nội dung file sẽ được dùng để phân tích.")
        with st.expander("👁 Xem trước nội dung file"):
            st.text(file_content[:2000] + ("..." if len(file_content) > 2000 else ""))

        # Ghi đè nội dung từ file vào user_input
        user_input = file_content

    if st.button("Phân tích"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản để dự đoán.")
        else:
            # Tiền xử lý 
            cleaned_msg = clean_text(user_input)
            
            # Biến đổi text thành số (Vector Hóa)
            vectorized_msg = vectorizer.transform([cleaned_msg])
            
            # Nhận diện Spam/Ham
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
            
            # --- TÍNH NĂNG GIẢI THÍCH AI (XAI) ---
            st.markdown("---")
            st.subheader("Giải phẫu AI (Explainable AI)")
            st.write("Dưới đây là căn cứ đưa ra quyết định của AI. Màu đỏ (hoặc xanh lá) thể hiện các ký tự mang tính ảnh hưởng mạnh đến quyết định.")
            
            try:
                import html
                vocab = vectorizer.vocabulary_
                html_output = ["<div style='line-height: 1.8; font-size: 16px; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #ffffff; color: #1a1a1a;'>"]
                
                # Thuật toán Tự làm (Custom XAI) tính toán trọng số trực tiếp không cần viện ngoài
                for word in user_input.split():
                    clean_w = clean_text(word)
                    
                    if clean_w in vocab:
                        idx = vocab[clean_w]
                        spam_prob = model.feature_log_prob_[1][idx]
                        ham_prob = model.feature_log_prob_[0][idx]
                        diff = spam_prob - ham_prob
                        
                        if diff > 1.0: # Rất Spam (Bôi đỏ đậm)
                            html_output.append(f'<span style="background-color: #ff4d4d; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold;" title="Điểm Spam: {diff:.2f}">{html.escape(word)}</span>')
                        elif diff < -1.0: # Rất an toàn (Bôi xanh lá)
                            html_output.append(f'<span style="background-color: #28a745; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold;" title="Điểm An toàn: {diff:.2f}">{html.escape(word)}</span>')
                        elif diff > 0.0: # Hơi nghi ngờ (Đỏ nhạt)
                            html_output.append(f'<span style="background-color: #ffaa44; color: #ffffff; padding: 2px 6px; border-radius: 4px;" title="Điểm nghi ngờ: {diff:.2f}">{html.escape(word)}</span>')
                        else:
                            html_output.append(f'<span style="color: #1a1a1a;">{html.escape(word)}</span>')
                    else:
                        html_output.append(f'<span style="color: #555555;">{html.escape(word)}</span>')
                        
                html_output.append("</div>")
                
                # Trình chiếu HTML lên giao diện
                st.markdown(" ".join(html_output), unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Tính năng giải phẫu AI có lỗi nhỏ: {e}")
            
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
    st.subheader("Lịch sử phân tích")

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

    # -----------------
    # ĐÁM MÂY TỪ KHÓA
    # -----------------
    st.markdown("---")
    st.subheader("Đám mây từ khóa (Spam Word Cloud)")
    st.write("Bản đồ nhiệt biểu diễn các từ khóa nguy hiểm nhất mà AI vừa học:")
    
    try:
        # Rút trích từ điển và tần suất Spam từ não bộ mô hình Naive Bayes
        feature_names = vectorizer.get_feature_names_out()
        spam_counts = model.feature_count_[1] 
        word_freq = dict(zip(feature_names, spam_counts))
        
        # Vẽ đám mây
        wc = WordCloud(width=800, height=450, background_color='white', colormap='Reds', max_words=100)
        wc.generate_from_frequencies(word_freq)
        
        st.image(wc.to_array(), use_container_width=True)
    except Exception as e:
        st.warning(f"Lỗi vẽ mây: {e}")


