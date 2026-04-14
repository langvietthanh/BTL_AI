# Ứng Dụng Phân Loại Tin Nhắn Rác (Spam SMS Classification)

Dự án này là một ứng dụng web giúp nhận diện tin nhắn rác (spam) hoặc tin nhắn bình thường (ham) bằng các mô hình học máy. Giao diện được xây dựng bằng Python với thư viện Streamlit.

## 🚀 Hướng Dẫn Tải File Và Cài Đặt Dự Án Cho Bạn Bè (Quick Start Guide)

Để tải project này về và chạy trên máy tính cá nhân, bạn chỉ cần làm theo 4 bước cực kỳ đơn giản dưới đây:

### Bước 1: Tải mã nguồn dự án

Bạn có thể tải code về máy bằng 1 trong 2 cách sau:
- **Cách 1 - Dùng Git:** Mở Terminal / Command Prompt hoặc Git Bash rồi chạy lệnh sau:
  ```bash
  git clone https://github.com/langvietthanh/BTL_AI.git
  cd BTL_AI
  ```
- **Cách 2 - Không dùng Git:** Bấm vào nút màu xanh lá cây `<> Code` trên Github, chọn **Download ZIP**, sau đó giải nén file `.zip` vừa tải về, và mở thư mục đã giải nén ra trong Terminal/Command Prompt.

### Bước 2: Tạo môi trường ảo (Nên làm)

Thiết lập môi trường ảo để các thư viện của project này không bị xung đột với các project Python khác.

- **Trên Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **Trên macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện cần thiết

Để ứng dụng có thể chạy, bạn cần cài đặt các thư viện cơ bản (như `streamlit`, `scikit-learn`, `pandas`,...). Hãy chạy lệnh dưới đây:
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy Ứng Dụng lên máy

Sau khi đã cài đặt xong, bạn gõ lệnh dưới đây và nhấn Enter để mở ứng dụng ở trên trình duyệt:
```bash
streamlit run app.py
```

Lúc này, trình duyệt sẽ tự động mở lên một trang web với đường dẫn là `http://localhost:8501`. Bây giờ bạn đã có thể bắt đầu sử dụng dự án!

---

## 📜 Thông Tin Tập Dữ Liệu (SMS Spam Collection v.1 Dataset)

1. DESCRIPTION
--------------

The SMS Spam Collection v.1 (hereafter the corpus) is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. 

1.1. Compilation
----------------

This corpus has been collected from free or free for research sources at the Web:

- A collection of between 425 SMS spam messages extracted manually from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext Web site is: http://www.grumbletext.co.uk/
- A list of 450 SMS ham messages collected from Caroline Tag's PhD Theses available at http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf
- A subset of 3,375 SMS ham messages of the NUS SMS Corpus (NSC), which is a corpus of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: http://www.comp.nus.edu.sg/~rpnlpir/downloads/corpora/smsCorpus/
- The amount of 1,002 SMS ham messages and 322 spam messages extracted from the SMS Spam Corpus v.0.1 Big created by Jos Mara Gmez Hidalgo and public available at: http://www.esp.uem.es/jmgomez/smsspamcorpus/

1.2. Statistics
---------------

There is one collection:

- The SMS Spam Collection v.1 (text file: smsspamcollection) has a total of 4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages.

1.3. Format
-----------

The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text. Here are some examples:

ham   What you doing?how are you?
ham   Ok lar... Joking wif u oni...
ham   dun say so early hor... U c already then say...
ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*
ham   Siva is in hostel aha:-.
ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
spam   FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
spam   Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
spam   URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU

Note: messages are not chronologically sorted.

2. USAGE
--------

We offer a comprehensive study of this corpus in the following paper that is under review. This work presents a number of statistics, studies and baseline results for several machine learning methods.

[1] Almeida, T.A., Gmez Hidalgo, J.M., Yamakami, A. Contributions to the study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11), Mountain View, CA, USA, 2011. (Under review)

3. ABOUT
--------

The corpus has been collected by Tiago Agostinho de Almeida (http://www.dt.fee.unicamp.br/~tiago) and Jos Mara Gmez Hidalgo (http://www.esp.uem.es/jmgomez).

4. LICENSE/DISCLAIMER
---------------------

The SMS Spam Collection v.1 is provided for free and with no limitations excepting...
