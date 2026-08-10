import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import time

# إعداد الصفحة
st.set_page_config(page_title="Robot Safety Scanner", page_icon="🤖", layout="centered")

# CSS لإضافة لمسة جمالية
st.markdown("""
    <style>
    .stApp {transition: background 0.5s ease;}
    .robot-title {text-align: center; font-size: 50px;}
    </style>
    """, unsafe_allow_html=True)

# ... (أبقي تعريف texts, labels, train_model من كودك السابق هنا) ...

st.markdown("<h1 class='robot-title'>🤖 روبوت الحماية الذكي</h1>", unsafe_allow_html=True)
st.write("---")

# استخدام session_state للتحكم في تعبير الروبوت
if 'robot_status' not in st.session_state:
    st.session_state.robot_status = "🤖"

col_r1, col_r2 = st.columns([1, 3])
with col_r1:
    st.markdown(f"<h1 style='font-size: 80px;'>{st.session_state.robot_status}</h1>", unsafe_allow_html=True)
with col_r2:
    user_message = st.text_area("✍️ ضع الرسالة هنا وسأقوم بتحليلها:", height=100, key="msg_input")

col_a, col_b = st.columns(2)
check_btn = col_a.button("🚀 افحص الرسالة الآن")
if col_b.button("🗑️ مسح"):
    st.session_state.robot_status = "🤖"
    st.rerun()

if check_btn:
    if not user_message.strip():
        st.warning("الروبوت يحتاج لرسالة أولاً! 🤖")
    else:
        # تأثير التفكير
        with st.spinner('الروبوت يقوم بالتحليل المعقد...'):
            time.sleep(1.5) # وقت ممتع لعملية التحليل
            
            msg_vector = vectorizer.transform([user_message])
            prediction = model.predict(msg_vector)[0]
            
            if prediction == "Suspicious Message":
                st.session_state.robot_status = "😱"
                st.error("🚨 خطر! الروبوت اكتشف رسالة مشبوهة!")
                st.markdown("### نصيحة الروبوت: لا تلمس هذا الرابط نهائياً! 🛑")
            else:
                st.session_state.robot_status = "😎"
                st.success("✅ الروبوت يوافق: الرسالة آمنة تماماً!")
                st.balloons()
                st.markdown("### نصيحة الروبوت: يمكنك الرد بأمان. استمتع! 🎈")
        
        # تحديث الصفحة لإظهار تعبير الروبوت الجديد
        st.rerun()
