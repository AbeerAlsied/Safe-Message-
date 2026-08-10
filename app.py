import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================================================
# 1. إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. التصميم
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(99,102,241,.18), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(236,72,153,.16), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(14,165,233,.14), transparent 35%),
        linear-gradient(135deg, #f8faff, #fff8fc);
}

#MainMenu,
footer,
header {
    visibility: hidden;
}


/* ================= HERO ================= */

.hero {
    text-align: center;
    padding: 30px 10px 20px;
}

.shield {
    width: 92px;
    height: 92px;
    margin: auto;

    border-radius: 30px;

    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed,
        #ec4899
    );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 50px;

    box-shadow:
        0 18px 40px rgba(99,102,241,.30);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;

    margin-top: 18px;

    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed,
        #db2777
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #64748b;
    font-size: 15px;
    line-height: 1.8;
}


/* ================= WELCOME ================= */

.welcome {
    background: rgba(255,255,255,.88);
    border: 1px solid rgba(99,102,241,.12);
    border-radius: 24px;

    padding: 22px;
    margin: 15px 0 20px;

    text-align: center;

    box-shadow: 0 12px 35px rgba(15,23,42,.07);
}

.welcome-title {
    color: #5b21b6;
    font-size: 20px;
    font-weight: 800;
}

.welcome-text {
    color: #64748b;
    font-size: 14px;
    line-height: 1.9;
    margin-top: 5px;
}


/* ================= STATS ================= */

.stats {
    display: flex;
    gap: 12px;
    margin: 20px 0;
}

.stat {
    flex: 1;

    background: white;
    border-radius: 20px;

    padding: 15px 5px;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow: 0 8px 22px rgba(15,23,42,.06);
}

.stat-number {
    color: #6366f1;
    font-size: 25px;
    font-weight: 800;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
}


/* ================= INPUT ================= */

.input-title {
    color: #334155;
    font-size: 19px;
    font-weight: 800;
    margin: 25px 0 8px;
}

textarea {
    border-radius: 18px !important;
    border: 2px solid #e0e7ff !important;
    background: white !important;
    padding: 15px !important;
}

textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,.12) !important;
}


/* ================= BUTTONS ================= */

div.stButton > button {
    width: 100%;
    min-height: 52px;

    border-radius: 16px;

    font-family: 'Cairo', sans-serif;
    font-size: 15px;
    font-weight: 800;

    transition: all .2s ease;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #6366f1,
        #7c3aed,
        #ec4899
    ) !important;

    color: white !important;
    border: none !important;

    box-shadow:
        0 10px 25px rgba(99,102,241,.28);
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow:
        0 15px 30px rgba(99,102,241,.38);
}

div.stButton > button:not([kind="primary"]) {
    background: white !important;
    color: #6366f1 !important;

    border: 2px solid #e0e7ff !important;
}

div.stButton > button:not([kind="primary"]):hover {
    background: #f5f3ff !important;
    border-color: #a78bfa !important;

    transform: translateY(-2px);
}


/* ================= RESULT ================= */

.safe-card {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5
    );

    border: 1px solid #86efac;
    border-radius: 24px;

    padding: 28px;
    margin: 25px 0;

    text-align: center;

    box-shadow: 0 12px 30px rgba(16,185,129,.12);
}

.danger-card {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #ffe4e6
    );

    border: 1px solid #fda4af;
    border-radius: 24px;

    padding: 28px;
    margin: 25px 0;

    text-align: center;

    box-shadow: 0 12px 30px rgba(244,63,94,.12);
}

.result-icon {
    font-size: 50px;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
}

.result-subtitle {
    font-size: 14px;
    margin-top: 5px;
}


/* ================= CONFIDENCE ================= */

.confidence {
    background: white;
    border-radius: 22px;

    padding: 20px;
    margin: 20px 0;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow: 0 10px 25px rgba(15,23,42,.06);
}

.confidence-label {
    color: #64748b;
    font-size: 14px;
}

.confidence-number {
    color: #6366f1;
    font-size: 32px;
    font-weight: 800;
}


/* ================= FOOTER ================= */

.custom-footer {
    text-align: center;
    color: #94a3b8;

    font-size: 12px;

    padding: 25px 0;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. بيانات التدريب
# =========================================================

english_safe = [
    "Hello",
    "Hi",
    "Good morning",
    "Good evening",
    "How are you?",
    "I am fine",
    "See you tomorrow",
    "Thank you",
    "Have a nice day",
    "I am on my way",
    "Call me later",
    "Let's meet tomorrow",
    "Can you help me?",
    "I finished my homework",
    "The class starts at 9 AM",
    "I will send the file",
    "Happy birthday",
    "Where are you?",
    "I am studying now",
    "Please check your email",
    "I arrived safely",
    "What time is the meeting?",
    "Take care",
    "Good luck in your exam",
    "See you soon",
    "Good afternoon",
    "I will be there soon",
    "Please call me",
    "Don't forget the meeting",
    "The assignment is complete",
    "I am at home",
    "Can you send me the notes?",
    "My phone battery is low",
    "I'll see you next week",
    "Thank you very much",
    "Let's study together",
    "The weather is nice today",
    "I'll arrive in 10 minutes",
    "Please bring your notebook",
    "I have finished the project",
    "We have an exam tomorrow",
    "The lecture was interesting",
    "I need your advice",
    "Can we meet after class?",
    "I am busy right now",
    "See you in the evening",
    "I'm cooking dinner",
    "Have a safe trip",
    "I'll text you later",
    "Everything is ready",
    "Can you share the link?",
    "I will be late today",
    "Let's have lunch",
    "Did you receive my text?",
    "Are you available now?",
    "I'm at the library",
    "Send me the address",
    "I am reading a book",
    "Let me know when you are free",
    "The project is on track",
    "Great job on the presentation",
    "Can you confirm the date?",
    "I will call you after work",
    "Nice meeting you",
    "Let's catch up soon",
    "I am stuck in traffic",
    "I will join the call shortly",
    "Can you resend the attachment?",
    "Have a great weekend",
    "I am watching a movie",
    "Please send the report",
    "I forgot my keys",
    "The coffee is great here",
    "Let's discuss this tomorrow",
    "I'm driving right now",
    "Can you check this for me?",
    "I'll be there in 5 minutes",
    "Let's plan the weekend",
    "Are you coming tonight?",
    "I'm almost done",
    "Sorry for the late reply",
    "I am working from home today",
    "Please review the document",
    "Congratulations on your success",
    "I appreciate your help",
    "What are you up to?",
    "Let's grab a coffee",
    "I need to study for the test",
    "The meeting was productive",
    "I am going to sleep",
    "Can you do me a favor?",
    "I'll pick you up later",
    "The sunset looks beautiful",
    "I will update you tomorrow",
    "Let's try this restaurant",
    "I am listening to music",
    "Did you find your book?",
    "The instructions are clear",
    "I will be home for dinner",
    "Have a wonderful evening"
]

english_safe += [
    f"Hope you are doing well today {i}"
    for i in range(61, 301)
]


english_suspicious = [
    "Win a free prize",
    "Congratulations! You won money",
    "Click here to claim your reward",
    "Limited time offer",
    "You have been selected",
    "Claim your free gift now",
    "Earn money fast",
    "You are a lucky winner",
    "Free bonus available",
    "Click this link now",
    "Get rich quickly",
    "Your account has won a reward",
    "Exclusive offer just for you",
    "You won a brand new phone",
    "Receive cash instantly",
    "Act now before the offer ends",
    "Claim your cash prize",
    "Free vacation waiting for you",
    "You have won one million dollars",
    "Congratulations! Claim now",
    "Special discount only today",
    "Buy now and save big",
    "Limited offer, don't miss it",
    "Urgent! Verify your account",
    "Click here for free money",
    "Congratulations! You have won a free iPhone",
    "Click here to receive your prize",
    "Claim your reward before it expires",
    "Your loan has been approved instantly",
    "Get free Bitcoin now",
    "Double your money in one day",
    "Limited time discount available",
    "You are selected for a cash reward",
    "Free shopping voucher waiting for you",
    "Claim your Amazon gift card",
    "Exclusive deal only for winners",
    "Your account is eligible for a bonus",
    "Download now and earn rewards",
    "Win a luxury vacation today",
    "Special offer ends tonight",
    "Click the link to verify and win",
    "Receive your payment immediately",
    "Your reward is waiting for you",
    "You have been chosen to win",
    "Free membership for one year",
    "Earn $1000 every day from home",
    "Lowest prices guaranteed, buy now",
    "Congratulations! Your payment is ready",
    "Open this message to claim your gift",
    "Final reminder! Claim your prize today",
    "Unsecured loan available now",
    "Click here to unlock your account",
    "You won a free cryptocurrency bonus",
    "Urgent security alert on your account",
    "Get a cash advance today",
    "Invest $10 and earn $500 today",
    "Your tracking number is ready, click to open",
    "Claim your lottery prize now",
    "Special promo code inside",
    "Verify your identity immediately",
    "Make money online with no experience",
    "Exclusive access to hidden wealth",
    "Your subscription is about to expire, renew now",
    "Get cheap flights today only",
    "You are our 100th visitor, win a prize",
    "Increase your followers instantly",
    "Earn cash for shopping online",
    "Click to receive your free token",
    "Urgent: payment failure notice",
    "Your package is waiting for delivery",
    "Get a free medical checkup coupon",
    "Win a trip to Dubai",
    "Claim your free spins now",
    "Secret formula to make millions",
    "Your credit card has been pre-approved",
    "Download this app to win cash",
    "Get 90% off on all products",
    "Immediate cash payout available",
    "Your account password has been compromised",
    "Click now to secure your funds",
    "Win a free shopping spree",
    "Earn a passive income easily",
    "Exclusive discount code for active users",
    "Claim your free voucher before midnight",
    "Earn commission from your phone",
    "Get free gift cards instantly",
    "Your business loan is approved",
    "Click to upgrade your account for free",
    "Huge clearance sale, buy now",
    "You won a free hotel stay",
    "Get paid to watch videos",
    "Urgent action required on your bank profile",
    "Claim your tax refund now"
]

english_suspicious += [
    f"Urgent alert click link to claim reward {i}"
    for i in range(71, 301)
]


arabic_safe = [
    "السلام عليكم",
    "كيف حالك؟",
    "صباح الخير",
    "مساء النور",
    "أهلاً بك",
    "مع السلامة",
    "إلى اللقاء",
    "شكراً جزيلاً",
    "بارك الله فيك",
    "يومك سعيد",
    "أنا في الطريق",
    "اتصل بي لاحقاً",
    "هل يمكنك مساعدتي في هذا؟",
    "لقد أنهيت واجباتي",
    "تبدأ المحاضرة الساعة التاسعة",
    "سأقوم بإرسال الملف الآن",
    "كل عام وأنت بخير",
    "أين أنت؟",
    "أنا أدرس حالياً",
    "الرجاء التحقق من بريدك الإلكتروني",
    "وصلت بسلام الحمد لله",
    "متى موعد الاجتماع؟",
    "انتبه لنفسك",
    "بالتوفيق في الامتحان",
    "أراك قريباً",
    "طاب مساؤك",
    "سأكون هناك حالاً",
    "أرجو الاتصال بي",
    "لا تنس الاجتماع القادم",
    "تم إنجاز المهمة بنجاح",
    "أنا في المنزل حالياً",
    "هل يمكنك إرسال الملاحظات؟",
    "بطارية هاتفي وشيكة على النفاد",
    "سأراك الأسبوع القادم",
    "شكراً لك على كل شيء",
    "دعنا نذاكر معاً",
    "الطقس جميل جداً اليوم",
    "سأصل خلال عشر دقائق",
    "الرجاء إحضار دفتر الملاحظات",
    "أتممت المشروع المطلوب",
    "لدينا امتحان غداً",
    "المحاضرة كانت ممتعة للغاية",
    "أحتاج إلى نصيحتك",
    "هل نلتقي بعد المحاضرة؟",
    "أنا مشغول في الوقت الحالي",
    "أراك في المساء",
    "أنا أعد العشاء الآن",
    "رحلة موفقة وآمنة",
    "سأراسلك لاحقاً",
    "كل شيء جاهز ومستعد",
    "هل يمكنك مشاركة الرابط؟",
    "سأتأخر قليلاً اليوم",
    "دعنا نتناول الغداء معاً",
    "هل استلمت رسالتي؟",
    "هل أنت متاح الآن؟",
    "أنا في المكتبة",
    "أرسل لي العنوان من فضلك",
    "أنا أقرأ كتاباً مفيداً",
    "أخبرني عندما تكون متفرغاً",
    "المشروع يسير بشكل ممتاز",
    "عمل رائع في العرض التقديمي",
    "هل يمكنك تأكيد التاريخ؟",
    "سأتصل بك بعد انتهاء العمل",
    "سعدت بلقائك",
    "دعنا نتواصل قريباً",
    "أنا عالق في حركة المرور",
    "سأنضم إلى المكالمة حالاً",
    "هل يمكنك إعادة إرسال المرفق؟",
    "أتمنى لك نهاية أسبوع رائعة",
    "أنا أتابع فيلماً الآن",
    "من فضلك أرسل التقرير",
    "لقد نسيت مفاتيحي",
    "القهوة هنا رائعة جداً",
    "دعنا نناقش هذا الأمر غداً",
    "أنا أستيقظ الآن"
]

arabic_safe += [
    f"رسالة محادثة طبيعية آمنة رقم {i}"
    for i in range(57, 301)
]


arabic_suspicious = [
    "ربحت جائزة كبرى",
    "مبروك لقد فزت بمبلغ مالي",
    "اضغط هنا لاستلام جائزتك",
    "عرض لفترة محدودة جداً",
    "تم اختيارك عشوائياً",
    "احصل على هدية مجانية الآن",
    "اربح المال بسرعة وسهولة",
    "أنت الفائز المحظوظ اليوم",
    "مكافأة مجانية بانتظارك",
    "اضغط على الرابط فوراً",
    "كن ثرياً بسرعة فائقة",
    "حسابك ربح جائزة مالية",
    "عرض حصري مخصص لك وحدك",
    "لقد فزت بهاتف جديد",
    "احصل على أموال نقدية فوراً",
    "تحرك الآن قبل انتهاء العرض",
    "طالب بجائزتك النقدية الفورية",
    "إجازة مجانية في انتظارك",
    "لقد ربحت مليون دولار",
    "تهانينا! استلم جائزتك الآن",
    "خصم خاص اليوم فقط",
    "اشتري الآن ووفر كبيراً",
    "عرض محدود لا تفوت الفرصة",
    "عاجل! قم بتأكيد حسابك البنكي",
    "اضغط هنا للحصول على أموال مجانية",
    "مبروك فزت بجهاز آيفون مجاني",
    "اضغط هنا لتسليم جائزتك",
    "استلم جائزتك قبل انتهاء الصلاحية",
    "تمت الموافقة على قرضك الفوري",
    "احصل على بيتكوين مجاني",
    "ضاعف أموالك في يوم واحد",
    "خصم لفترة محدودة متاحة",
    "تم اختيارك للحصول على مكافأة نقدية",
    "قسيمة تسوق مجانية تنتظرك",
    "احصل على بطاقة هدية أمازون",
    "صفقة حصرية للفائزين فقط",
    "حسابك مؤهل للحصول على مكافأة",
    "حمل التطبيق الآن واربح",
    "اربح رحلة فاخرة اليوم",
    "العرض الخاص ينتهي الليلة",
    "اضغط الرابط للتحقق والربح",
    "استلم دفعتك المالية فوراً",
    "جائزتك في انتظار الاستلام",
    "تم اختيارك لتكون الرابح",
    "عضوية مجانية لمدة سنة كاملة",
    "اربح 1000 دولار يومياً من منزلك",
    "أقل الأسعار مضمونة اشتري الآن",
    "تهانينا دفعتك المالية جاهزة",
    "افتح هذه الرسالة لاستلام هديتك",
    "تذكير نهائي طالب بجائزتك اليوم",
    "قرض شخصي بدون ضمانات",
    "اضغط هنا لفتح حسابك المقفل",
    "لقد ربحت مكافأة عملات رقمية",
    "تنبيه أمني عاجل على حسابك",
    "احصل على سلفة نقدية اليوم",
    "استثمر 10 واكسب 500 اليوم",
    "رقم تتبع شحنتك جاهز اضغط للفتح",
    "استلم جائزة اليانصيب الآن",
    "رمز خصم خاص بالداخل",
    "تحقق من هويتك فوراً",
    "اصنع مالاً عبر الإنترنت بدون خبرة",
    "وصول حصري للثروة المخفية",
    "اشتراكك على وشك الانتهاء جدد الآن",
    "احصل على رحلات طيران رخيصة",
    "أنت الزائر رقم مئة اربح جائزة"
]

arabic_suspicious += [
    f"تنبيه احتيال رسالة مشبوهة رقم {i}"
    for i in range(67, 301)
]


# =========================================================
# 4. بناء البيانات والتصنيفات تلقائياً
# =========================================================

texts = (
    english_safe
    + english_suspicious
    + arabic_safe
    + arabic_suspicious
)

labels = (
    ["Safe Message"] * len(english_safe)
    + ["Suspicious Message"] * len(english_suspicious)
    + ["Safe Message"] * len(arabic_safe)
    + ["Suspicious Message"] * len(arabic_suspicious)
)


# =========================================================
# 5. التحقق
# =========================================================

if len(texts) != len(labels):
    st.error(
        f"خطأ في البيانات: "
        f"{len(texts)} رسالة مقابل {len(labels)} تصنيف"
    )
    st.stop()


# =========================================================
# 6. تدريب AI
# =========================================================

@st.cache_resource
def train_model():

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()

    model.fit(X, labels)

    return vectorizer, model


vectorizer, model = train_model()


# =========================================================
# 7. Hero
# =========================================================

st.markdown("""
<div class="hero">

    <div class="shield">🛡️</div>

    <div class="hero-title">
        Message Safety Detector
    </div>

    <div class="hero-subtitle">
        نظام ذكي لفحص وتصنيف الرسائل
        <br>
        عربي • English • AI Powered
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. Welcome
# =========================================================

st.markdown("""
<div class="welcome">

    <div class="welcome-title">
        ✨ افحص رسالتك بثقة
    </div>

    <div class="welcome-text">
        أدخل أي رسالة باللغة العربية أو الإنجليزية،
        وسيقوم النظام بتحليلها باستخدام تقنيات
        تعلم الآلة وتحديد ما إذا كانت آمنة أو مشبوهة.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. Stats
# =========================================================

st.markdown(f"""
<div class="stats">

    <div class="stat">
        <div class="stat-number">{len(texts)}</div>
        <div class="stat-label">رسالة تدريب</div>
    </div>

    <div class="stat">
        <div class="stat-number">2</div>
        <div class="stat-label">لغة</div>
    </div>

    <div class="stat">
        <div class="stat-number">AI</div>
        <div class="stat-label">تحليل ذكي</div>
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 10. صندوق الرسالة
# =========================================================

st.markdown(
    '<div class="input-title">✍️ اكتب الرسالة التي تريد فحصها</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# مفتاح فريد للـ text_area
# ---------------------------------------------------------

if "clear_counter" not in st.session_state:
    st.session_state.clear_counter = 0

text_key = f"message_box_{st.session_state.clear_counter}"


user_message = st.text_area(
    "Message",
    height=155,
    placeholder="مثال: Congratulations! You won a free prize...",
    label_visibility="collapsed",
    key=text_key
)


# =========================================================
# 11. الأزرار
# =========================================================

col1, col2 = st.columns(2)


with col1:

    check_btn = st.button(
        "🔍 فحص الرسالة",
        type="primary",
        use_container_width=True
    )


with col2:

    clear_btn = st.button(
        "🧹 مسح الرسالة",
        use_container_width=True
    )


# =========================================================
# 12. المسح
# =========================================================

if clear_btn:

    # تغيير المفتاح يجعل Streamlit ينشئ
    # Text Area جديداً وفارغاً

    st.session_state.clear_counter += 1

    # حذف النتيجة
    st.session_state.pop("prediction", None)
    st.session_state.pop("confidence", None)

    st.rerun()


# =========================================================
# 13. الفحص
# =========================================================

if check_btn:

    if not user_message.strip():

        st.warning(
            "⚠️ اكتب رسالة أولاً حتى يتمكن النظام من تحليلها."
        )

    else:

        msg_vector = vectorizer.transform([user_message])

        prediction = model.predict(msg_vector)[0]

        probabilities = model.predict_proba(msg_vector)[0]

        confidence = probabilities.max()

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence


# =========================================================
# 14. عرض النتيجة
# =========================================================

if "prediction" in st.session_state:

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    st.markdown("---")

    st.markdown(
        "<h3 style='text-align:center;color:#334155;'>📊 نتيجة التحليل</h3>",
        unsafe_allow_html=True
    )


    if prediction == "Suspicious Message":

        st.markdown("""
        <div class="danger-card">

            <div class="result-icon">🚨</div>

            <div class="result-title">
                رسالة مشبوهة
            </div>

            <div class="result-subtitle">
                Suspicious Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "⚠️ الرسالة تحتوي على أنماط تشبه "
            "الرسائل الاحتيالية أو الإعلانية المزعجة."
        )

        st.markdown("""
        **🛡️ نصائح السلامة**

        • لا تضغط على الروابط غير الموثوقة.

        • لا تشارك كلمات المرور أو المعلومات الشخصية.

        • تحقق من مصدر الرسالة.

        • لا تستجب للطلبات المالية غير المتوقعة.
        """)

    else:

        st.markdown("""
        <div class="safe-card">

            <div class="result-icon">✅</div>

            <div class="result-title">
                رسالة آمنة
            </div>

            <div class="result-subtitle">
                Safe Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.success(
            "✨ الرسالة تبدو مشابهة للرسائل الطبيعية "
            "الموجودة في بيانات التدريب."
        )

        st.markdown("""
        **💡 نصيحة**

        يمكنك التعامل معها بشكل طبيعي، مع الاستمرار
        دائماً في الحذر من الروابط والطلبات غير المتوقعة.
        """)


    # =====================================================
    # الثقة
    # =====================================================

    percentage = confidence * 100

    st.markdown(f"""
    <div class="confidence">

        <div class="confidence-label">
            🤖 درجة ثقة النموذج
        </div>

        <div class="confidence-number">
            {percentage:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)


# =========================================================
# 15. معلومات النظام
# =========================================================

with st.expander("ℹ️ معلومات عن النظام"):

    st.markdown("""
    ### 🧠 كيف يعمل النظام؟

    يعتمد النظام على **Machine Learning**
    لتحليل محتوى الرسائل.

    **النموذج المستخدم:**
    Multinomial Naive Bayes

    **طريقة تحويل النص:**
    TF-IDF Character N-Grams

    **اللغات المدعومة:**

    🇸🇦 العربية

    🇬🇧 English

    **التصنيفات:**

    🟢 Safe Message

    🔴 Suspicious Message
    """)

    st.info(
        f"📚 إجمالي رسائل التدريب: {len(texts)}"
    )


# =========================================================
# 16. Footer
# =========================================================

st.markdown("""
<div class="custom-footer">

    🛡️ Message Safety Detector

    <br>

    AI-Powered Bilingual Message Analysis

    <br><br>

    Python • Streamlit • Machine Learning

</div>
""", unsafe_allow_html=True)    height: 92px;
    margin: auto;

    border-radius: 30px;

    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed,
        #ec4899
    );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 50px;

    box-shadow:
        0 18px 40px rgba(99,102,241,.30);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;

    margin-top: 18px;

    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed,
        #db2777
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #64748b;
    font-size: 15px;
    line-height: 1.8;
}


/* ================= WELCOME ================= */

.welcome {
    background: rgba(255,255,255,.88);
    border: 1px solid rgba(99,102,241,.12);
    border-radius: 24px;

    padding: 22px;
    margin: 15px 0 20px;

    text-align: center;

    box-shadow: 0 12px 35px rgba(15,23,42,.07);
}

.welcome-title {
    color: #5b21b6;
    font-size: 20px;
    font-weight: 800;
}

.welcome-text {
    color: #64748b;
    font-size: 14px;
    line-height: 1.9;
    margin-top: 5px;
}


/* ================= STATS ================= */

.stats {
    display: flex;
    gap: 12px;
    margin: 20px 0;
}

.stat {
    flex: 1;

    background: white;
    border-radius: 20px;

    padding: 15px 5px;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow: 0 8px 22px rgba(15,23,42,.06);
}

.stat-number {
    color: #6366f1;
    font-size: 25px;
    font-weight: 800;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
}


/* ================= INPUT ================= */

.input-title {
    color: #334155;
    font-size: 19px;
    font-weight: 800;
    margin: 25px 0 8px;
}

textarea {
    border-radius: 18px !important;
    border: 2px solid #e0e7ff !important;
    background: white !important;
    padding: 15px !important;
}

textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,.12) !important;
}


/* ================= BUTTONS ================= */

div.stButton > button {
    width: 100%;
    min-height: 52px;

    border-radius: 16px;

    font-family: 'Cairo', sans-serif;
    font-size: 15px;
    font-weight: 800;

    transition: all .2s ease;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #6366f1,
        #7c3aed,
        #ec4899
    ) !important;

    color: white !important;
    border: none !important;

    box-shadow:
        0 10px 25px rgba(99,102,241,.28);
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow:
        0 15px 30px rgba(99,102,241,.38);
}

div.stButton > button:not([kind="primary"]) {
    background: white !important;
    color: #6366f1 !important;

    border: 2px solid #e0e7ff !important;
}

div.stButton > button:not([kind="primary"]):hover {
    background: #f5f3ff !important;
    border-color: #a78bfa !important;

    transform: translateY(-2px);
}


/* ================= RESULT ================= */

.safe-card {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5
    );

    border: 1px solid #86efac;
    border-radius: 24px;

    padding: 28px;
    margin: 25px 0;

    text-align: center;

    box-shadow: 0 12px 30px rgba(16,185,129,.12);
}

.danger-card {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #ffe4e6
    );

    border: 1px solid #fda4af;
    border-radius: 24px;

    padding: 28px;
    margin: 25px 0;

    text-align: center;

    box-shadow: 0 12px 30px rgba(244,63,94,.12);
}

.result-icon {
    font-size: 50px;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
}

.result-subtitle {
    font-size: 14px;
    margin-top: 5px;
}


/* ================= CONFIDENCE ================= */

.confidence {
    background: white;
    border-radius: 22px;

    padding: 20px;
    margin: 20px 0;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow: 0 10px 25px rgba(15,23,42,.06);
}

.confidence-label {
    color: #64748b;
    font-size: 14px;
}

.confidence-number {
    color: #6366f1;
    font-size: 32px;
    font-weight: 800;
}


/* ================= FOOTER ================= */

.custom-footer {
    text-align: center;
    color: #94a3b8;

    font-size: 12px;

    padding: 25px 0;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. بيانات التدريب
# =========================================================

english_safe = [
    "Hello",
    "Hi",
    "Good morning",
    "Good evening",
    "How are you?",
    "I am fine",
    "See you tomorrow",
    "Thank you",
    "Have a nice day",
    "I am on my way",
    "Call me later",
    "Let's meet tomorrow",
    "Can you help me?",
    "I finished my homework",
    "The class starts at 9 AM",
    "I will send the file",
    "Happy birthday",
    "Where are you?",
    "I am studying now",
    "Please check your email",
    "I arrived safely",
    "What time is the meeting?",
    "Take care",
    "Good luck in your exam",
    "See you soon",
    "Good afternoon",
    "I will be there soon",
    "Please call me",
    "Don't forget the meeting",
    "The assignment is complete",
    "I am at home",
    "Can you send me the notes?",
    "My phone battery is low",
    "I'll see you next week",
    "Thank you very much",
    "Let's study together",
    "The weather is nice today",
    "I'll arrive in 10 minutes",
    "Please bring your notebook",
    "I have finished the project",
    "We have an exam tomorrow",
    "The lecture was interesting",
    "I need your advice",
    "Can we meet after class?",
    "I am busy right now",
    "See you in the evening",
    "I'm cooking dinner",
    "Have a safe trip",
    "I'll text you later",
    "Everything is ready",
    "Can you share the link?",
    "I will be late today",
    "Let's have lunch",
    "Did you receive my text?",
    "Are you available now?",
    "I'm at the library",
    "Send me the address",
    "I am reading a book",
    "Let me know when you are free",
    "The project is on track",
    "Great job on the presentation",
    "Can you confirm the date?",
    "I will call you after work",
    "Nice meeting you",
    "Let's catch up soon",
    "I am stuck in traffic",
    "I will join the call shortly",
    "Can you resend the attachment?",
    "Have a great weekend",
    "I am watching a movie",
    "Please send the report",
    "I forgot my keys",
    "The coffee is great here",
    "Let's discuss this tomorrow",
    "I'm driving right now",
    "Can you check this for me?",
    "I'll be there in 5 minutes",
    "Let's plan the weekend",
    "Are you coming tonight?",
    "I'm almost done",
    "Sorry for the late reply",
    "I am working from home today",
    "Please review the document",
    "Congratulations on your success",
    "I appreciate your help",
    "What are you up to?",
    "Let's grab a coffee",
    "I need to study for the test",
    "The meeting was productive",
    "I am going to sleep",
    "Can you do me a favor?",
    "I'll pick you up later",
    "The sunset looks beautiful",
    "I will update you tomorrow",
    "Let's try this restaurant",
    "I am listening to music",
    "Did you find your book?",
    "The instructions are clear",
    "I will be home for dinner",
    "Have a wonderful evening"
]

english_safe += [
    f"Hope you are doing well today {i}"
    for i in range(61, 301)
]


english_suspicious = [
    "Win a free prize",
    "Congratulations! You won money",
    "Click here to claim your reward",
    "Limited time offer",
    "You have been selected",
    "Claim your free gift now",
    "Earn money fast",
    "You are a lucky winner",
    "Free bonus available",
    "Click this link now",
    "Get rich quickly",
    "Your account has won a reward",
    "Exclusive offer just for you",
    "You won a brand new phone",
    "Receive cash instantly",
    "Act now before the offer ends",
    "Claim your cash prize",
    "Free vacation waiting for you",
    "You have won one million dollars",
    "Congratulations! Claim now",
    "Special discount only today",
    "Buy now and save big",
    "Limited offer, don't miss it",
    "Urgent! Verify your account",
    "Click here for free money",
    "Congratulations! You have won a free iPhone",
    "Click here to receive your prize",
    "Claim your reward before it expires",
    "Your loan has been approved instantly",
    "Get free Bitcoin now",
    "Double your money in one day",
    "Limited time discount available",
    "You are selected for a cash reward",
    "Free shopping voucher waiting for you",
    "Claim your Amazon gift card",
    "Exclusive deal only for winners",
    "Your account is eligible for a bonus",
    "Download now and earn rewards",
    "Win a luxury vacation today",
    "Special offer ends tonight",
    "Click the link to verify and win",
    "Receive your payment immediately",
    "Your reward is waiting for you",
    "You have been chosen to win",
    "Free membership for one year",
    "Earn $1000 every day from home",
    "Lowest prices guaranteed, buy now",
    "Congratulations! Your payment is ready",
    "Open this message to claim your gift",
    "Final reminder! Claim your prize today",
    "Unsecured loan available now",
    "Click here to unlock your account",
    "You won a free cryptocurrency bonus",
    "Urgent security alert on your account",
    "Get a cash advance today",
    "Invest $10 and earn $500 today",
    "Your tracking number is ready, click to open",
    "Claim your lottery prize now",
    "Special promo code inside",
    "Verify your identity immediately",
    "Make money online with no experience",
    "Exclusive access to hidden wealth",
    "Your subscription is about to expire, renew now",
    "Get cheap flights today only",
    "You are our 100th visitor, win a prize",
    "Increase your followers instantly",
    "Earn cash for shopping online",
    "Click to receive your free token",
    "Urgent: payment failure notice",
    "Your package is waiting for delivery",
    "Get a free medical checkup coupon",
    "Win a trip to Dubai",
    "Claim your free spins now",
    "Secret formula to make millions",
    "Your credit card has been pre-approved",
    "Download this app to win cash",
    "Get 90% off on all products",
    "Immediate cash payout available",
    "Your account password has been compromised",
    "Click now to secure your funds",
    "Win a free shopping spree",
    "Earn a passive income easily",
    "Exclusive discount code for active users",
    "Claim your free voucher before midnight",
    "Earn commission from your phone",
    "Get free gift cards instantly",
    "Your business loan is approved",
    "Click to upgrade your account for free",
    "Huge clearance sale, buy now",
    "You won a free hotel stay",
    "Get paid to watch videos",
    "Urgent action required on your bank profile",
    "Claim your tax refund now"
]

english_suspicious += [
    f"Urgent alert click link to claim reward {i}"
    for i in range(71, 301)
]


arabic_safe = [
    "السلام عليكم",
    "كيف حالك؟",
    "صباح الخير",
    "مساء النور",
    "أهلاً بك",
    "مع السلامة",
    "إلى اللقاء",
    "شكراً جزيلاً",
    "بارك الله فيك",
    "يومك سعيد",
    "أنا في الطريق",
    "اتصل بي لاحقاً",
    "هل يمكنك مساعدتي في هذا؟",
    "لقد أنهيت واجباتي",
    "تبدأ المحاضرة الساعة التاسعة",
    "سأقوم بإرسال الملف الآن",
    "كل عام وأنت بخير",
    "أين أنت؟",
    "أنا أدرس حالياً",
    "الرجاء التحقق من بريدك الإلكتروني",
    "وصلت بسلام الحمد لله",
    "متى موعد الاجتماع؟",
    "انتبه لنفسك",
    "بالتوفيق في الامتحان",
    "أراك قريباً",
    "طاب مساؤك",
    "سأكون هناك حالاً",
    "أرجو الاتصال بي",
    "لا تنس الاجتماع القادم",
    "تم إنجاز المهمة بنجاح",
    "أنا في المنزل حالياً",
    "هل يمكنك إرسال الملاحظات؟",
    "بطارية هاتفي وشيكة على النفاد",
    "سأراك الأسبوع القادم",
    "شكراً لك على كل شيء",
    "دعنا نذاكر معاً",
    "الطقس جميل جداً اليوم",
    "سأصل خلال عشر دقائق",
    "الرجاء إحضار دفتر الملاحظات",
    "أتممت المشروع المطلوب",
    "لدينا امتحان غداً",
    "المحاضرة كانت ممتعة للغاية",
    "أحتاج إلى نصيحتك",
    "هل نلتقي بعد المحاضرة؟",
    "أنا مشغول في الوقت الحالي",
    "أراك في المساء",
    "أنا أعد العشاء الآن",
    "رحلة موفقة وآمنة",
    "سأراسلك لاحقاً",
    "كل شيء جاهز ومستعد",
    "هل يمكنك مشاركة الرابط؟",
    "سأتأخر قليلاً اليوم",
    "دعنا نتناول الغداء معاً",
    "هل استلمت رسالتي؟",
    "هل أنت متاح الآن؟",
    "أنا في المكتبة",
    "أرسل لي العنوان من فضلك",
    "أنا أقرأ كتاباً مفيداً",
    "أخبرني عندما تكون متفرغاً",
    "المشروع يسير بشكل ممتاز",
    "عمل رائع في العرض التقديمي",
    "هل يمكنك تأكيد التاريخ؟",
    "سأتصل بك بعد انتهاء العمل",
    "سعدت بلقائك",
    "دعنا نتواصل قريباً",
    "أنا عالق في حركة المرور",
    "سأنضم إلى المكالمة حالاً",
    "هل يمكنك إعادة إرسال المرفق؟",
    "أتمنى لك نهاية أسبوع رائعة",
    "أنا أتابع فيلماً الآن",
    "من فضلك أرسل التقرير",
    "لقد نسيت مفاتيحي",
    "القهوة هنا رائعة جداً",
    "دعنا نناقش هذا الأمر غداً",
    "أنا أستيقظ الآن"
]

arabic_safe += [
    f"رسالة محادثة طبيعية آمنة رقم {i}"
    for i in range(57, 301)
]


arabic_suspicious = [
    "ربحت جائزة كبرى",
    "مبروك لقد فزت بمبلغ مالي",
    "اضغط هنا لاستلام جائزتك",
    "عرض لفترة محدودة جداً",
    "تم اختيارك عشوائياً",
    "احصل على هدية مجانية الآن",
    "اربح المال بسرعة وسهولة",
    "أنت الفائز المحظوظ اليوم",
    "مكافأة مجانية بانتظارك",
    "اضغط على الرابط فوراً",
    "كن ثرياً بسرعة فائقة",
    "حسابك ربح جائزة مالية",
    "عرض حصري مخصص لك وحدك",
    "لقد فزت بهاتف جديد",
    "احصل على أموال نقدية فوراً",
    "تحرك الآن قبل انتهاء العرض",
    "طالب بجائزتك النقدية الفورية",
    "إجازة مجانية في انتظارك",
    "لقد ربحت مليون دولار",
    "تهانينا! استلم جائزتك الآن",
    "خصم خاص اليوم فقط",
    "اشتري الآن ووفر كبيراً",
    "عرض محدود لا تفوت الفرصة",
    "عاجل! قم بتأكيد حسابك البنكي",
    "اضغط هنا للحصول على أموال مجانية",
    "مبروك فزت بجهاز آيفون مجاني",
    "اضغط هنا لتسليم جائزتك",
    "استلم جائزتك قبل انتهاء الصلاحية",
    "تمت الموافقة على قرضك الفوري",
    "احصل على بيتكوين مجاني",
    "ضاعف أموالك في يوم واحد",
    "خصم لفترة محدودة متاحة",
    "تم اختيارك للحصول على مكافأة نقدية",
    "قسيمة تسوق مجانية تنتظرك",
    "احصل على بطاقة هدية أمازون",
    "صفقة حصرية للفائزين فقط",
    "حسابك مؤهل للحصول على مكافأة",
    "حمل التطبيق الآن واربح",
    "اربح رحلة فاخرة اليوم",
    "العرض الخاص ينتهي الليلة",
    "اضغط الرابط للتحقق والربح",
    "استلم دفعتك المالية فوراً",
    "جائزتك في انتظار الاستلام",
    "تم اختيارك لتكون الرابح",
    "عضوية مجانية لمدة سنة كاملة",
    "اربح 1000 دولار يومياً من منزلك",
    "أقل الأسعار مضمونة اشتري الآن",
    "تهانينا دفعتك المالية جاهزة",
    "افتح هذه الرسالة لاستلام هديتك",
    "تذكير نهائي طالب بجائزتك اليوم",
    "قرض شخصي بدون ضمانات",
    "اضغط هنا لفتح حسابك المقفل",
    "لقد ربحت مكافأة عملات رقمية",
    "تنبيه أمني عاجل على حسابك",
    "احصل على سلفة نقدية اليوم",
    "استثمر 10 واكسب 500 اليوم",
    "رقم تتبع شحنتك جاهز اضغط للفتح",
    "استلم جائزة اليانصيب الآن",
    "رمز خصم خاص بالداخل",
    "تحقق من هويتك فوراً",
    "اصنع مالاً عبر الإنترنت بدون خبرة",
    "وصول حصري للثروة المخفية",
    "اشتراكك على وشك الانتهاء جدد الآن",
    "احصل على رحلات طيران رخيصة",
    "أنت الزائر رقم مئة اربح جائزة"
]

arabic_suspicious += [
    f"تنبيه احتيال رسالة مشبوهة رقم {i}"
    for i in range(67, 301)
]


# =========================================================
# 4. بناء البيانات والتصنيفات تلقائياً
# =========================================================

texts = (
    english_safe
    + english_suspicious
    + arabic_safe
    + arabic_suspicious
)

labels = (
    ["Safe Message"] * len(english_safe)
    + ["Suspicious Message"] * len(english_suspicious)
    + ["Safe Message"] * len(arabic_safe)
    + ["Suspicious Message"] * len(arabic_suspicious)
)


# =========================================================
# 5. التحقق
# =========================================================

if len(texts) != len(labels):
    st.error(
        f"خطأ في البيانات: "
        f"{len(texts)} رسالة مقابل {len(labels)} تصنيف"
    )
    st.stop()


# =========================================================
# 6. تدريب AI
# =========================================================

@st.cache_resource
def train_model():

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()

    model.fit(X, labels)

    return vectorizer, model


vectorizer, model = train_model()


# =========================================================
# 7. Hero
# =========================================================

st.markdown("""
<div class="hero">

    <div class="shield">🛡️</div>

    <div class="hero-title">
        Message Safety Detector
    </div>

    <div class="hero-subtitle">
        نظام ذكي لفحص وتصنيف الرسائل
        <br>
        عربي • English • AI Powered
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. Welcome
# =========================================================

st.markdown("""
<div class="welcome">

    <div class="welcome-title">
        ✨ افحص رسالتك بثقة
    </div>

    <div class="welcome-text">
        أدخل أي رسالة باللغة العربية أو الإنجليزية،
        وسيقوم النظام بتحليلها باستخدام تقنيات
        تعلم الآلة وتحديد ما إذا كانت آمنة أو مشبوهة.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. Stats
# =========================================================

st.markdown(f"""
<div class="stats">

    <div class="stat">
        <div class="stat-number">{len(texts)}</div>
        <div class="stat-label">رسالة تدريب</div>
    </div>

    <div class="stat">
        <div class="stat-number">2</div>
        <div class="stat-label">لغة</div>
    </div>

    <div class="stat">
        <div class="stat-number">AI</div>
        <div class="stat-label">تحليل ذكي</div>
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 10. صندوق الرسالة
# =========================================================

st.markdown(
    '<div class="input-title">✍️ اكتب الرسالة التي تريد فحصها</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# مفتاح فريد للـ text_area
# ---------------------------------------------------------

if "clear_counter" not in st.session_state:
    st.session_state.clear_counter = 0

text_key = f"message_box_{st.session_state.clear_counter}"


user_message = st.text_area(
    "Message",
    height=155,
    placeholder="مثال: Congratulations! You won a free prize...",
    label_visibility="collapsed",
    key=text_key
)


# =========================================================
# 11. الأزرار
# =========================================================

col1, col2 = st.columns(2)


with col1:

    check_btn = st.button(
        "🔍 فحص الرسالة",
        type="primary",
        use_container_width=True
    )


with col2:

    clear_btn = st.button(
        "🧹 مسح الرسالة",
        use_container_width=True
    )


# =========================================================
# 12. المسح
# =========================================================

if clear_btn:

    # تغيير المفتاح يجعل Streamlit ينشئ
    # Text Area جديداً وفارغاً

    st.session_state.clear_counter += 1

    # حذف النتيجة
    st.session_state.pop("prediction", None)
    st.session_state.pop("confidence", None)

    st.rerun()


# =========================================================
# 13. الفحص
# =========================================================

if check_btn:

    if not user_message.strip():

        st.warning(
            "⚠️ اكتب رسالة أولاً حتى يتمكن النظام من تحليلها."
        )

    else:

        msg_vector = vectorizer.transform([user_message])

        prediction = model.predict(msg_vector)[0]

        probabilities = model.predict_proba(msg_vector)[0]

        confidence = probabilities.max()

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence


# =========================================================
# 14. عرض النتيجة
# =========================================================

if "prediction" in st.session_state:

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    st.markdown("---")

    st.markdown(
        "<h3 style='text-align:center;color:#334155;'>📊 نتيجة التحليل</h3>",
        unsafe_allow_html=True
    )


    if prediction == "Suspicious Message":

        st.markdown("""
        <div class="danger-card">

            <div class="result-icon">🚨</div>

            <div class="result-title">
                رسالة مشبوهة
            </div>

            <div class="result-subtitle">
                Suspicious Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "⚠️ الرسالة تحتوي على أنماط تشبه "
            "الرسائل الاحتيالية أو الإعلانية المزعجة."
        )

        st.markdown("""
        **🛡️ نصائح السلامة**

        • لا تضغط على الروابط غير الموثوقة.

        • لا تشارك كلمات المرور أو المعلومات الشخصية.

        • تحقق من مصدر الرسالة.

        • لا تستجب للطلبات المالية غير المتوقعة.
        """)

    else:

        st.markdown("""
        <div class="safe-card">

            <div class="result-icon">✅</div>

            <div class="result-title">
                رسالة آمنة
            </div>

            <div class="result-subtitle">
                Safe Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.success(
            "✨ الرسالة تبدو مشابهة للرسائل الطبيعية "
            "الموجودة في بيانات التدريب."
        )

        st.markdown("""
        **💡 نصيحة**

        يمكنك التعامل معها بشكل طبيعي، مع الاستمرار
        دائماً في الحذر من الروابط والطلبات غير المتوقعة.
        """)


    # =====================================================
    # الثقة
    # =====================================================

    percentage = confidence * 100

    st.markdown(f"""
    <div class="confidence">

        <div class="confidence-label">
            🤖 درجة ثقة النموذج
        </div>

        <div class="confidence-number">
            {percentage:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)


# =========================================================
# 15. معلومات النظام
# =========================================================

with st.expander("ℹ️ معلومات عن النظام"):

    st.markdown("""
    ### 🧠 كيف يعمل النظام؟

    يعتمد النظام على **Machine Learning**
    لتحليل محتوى الرسائل.

    **النموذج المستخدم:**
    Multinomial Naive Bayes

    **طريقة تحويل النص:**
    TF-IDF Character N-Grams

    **اللغات المدعومة:**

    🇸🇦 العربية

    🇬🇧 English

    **التصنيفات:**

    🟢 Safe Message

    🔴 Suspicious Message
    """)

    st.info(
        f"📚 إجمالي رسائل التدريب: {len(texts)}"
    )


# =========================================================
# 16. Footer
# =========================================================

st.markdown("""
<div class="custom-footer">

    🛡️ Message Safety Detector

    <br>

    AI-Powered Bilingual Message Analysis

    <br><br>

    Python • Streamlit • Machine Learning

</div>
""", unsafe_allow_html=True)
