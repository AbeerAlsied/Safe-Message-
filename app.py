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
        radial-gradient(circle at 5% 5%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(236,72,153,0.15), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(14,165,233,0.14), transparent 35%),
        linear-gradient(135deg, #f8faff, #eef2ff);
}

/* إخفاء عناصر Streamlit */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =========================================================
   Hero
   ========================================================= */

.hero {
    text-align: center;
    padding: 35px 15px 20px;
}

.hero-icon {
    font-size: 70px;
    filter: drop-shadow(0 8px 15px rgba(99,102,241,0.25));
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed,
        #db2777
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 8px;
}

.hero-subtitle {
    color: #64748b;
    font-size: 16px;
    line-height: 1.8;
}


/* =========================================================
   بطاقة المعلومات
   ========================================================= */

.info-card {
    background: rgba(255,255,255,0.9);
    border-radius: 24px;
    padding: 24px;
    margin: 15px 0 25px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.15);
    box-shadow: 0 15px 40px rgba(79,70,229,0.10);
}

.info-title {
    color: #4f46e5;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 8px;
}

.info-text {
    color: #64748b;
    font-size: 14px;
    line-height: 2;
}


/* =========================================================
   الإحصائيات
   ========================================================= */

.stats {
    display: flex;
    gap: 12px;
    margin: 25px 0;
}

.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 18px 8px;
    text-align: center;
    border: 1px solid #e0e7ff;
    box-shadow: 0 8px 25px rgba(15,23,42,0.06);
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #6366f1;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
}


/* =========================================================
   عنوان صندوق الرسالة
   ========================================================= */

.message-label {
    font-size: 19px;
    font-weight: 800;
    color: #334155;
    margin: 20px 0 10px;
}


/* =========================================================
   Text Area
   ========================================================= */

textarea {
    border-radius: 18px !important;
    border: 2px solid #e0e7ff !important;
    background: white !important;
    font-size: 16px !important;
}

textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}


/* =========================================================
   الأزرار
   ========================================================= */

div.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 16px;
    font-size: 16px;
    font-weight: 800;
    transition: all 0.2s ease;
}


/* زر الفحص */
div.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6,
        #ec4899
    );
    color: white;
    border: none;
    box-shadow: 0 10px 25px rgba(99,102,241,0.25);
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 30px rgba(99,102,241,0.35);
}


/* زر المسح */
div.stButton > button:not([kind="primary"]) {
    background: white;
    color: #6366f1;
    border: 2px solid #c7d2fe;
}

div.stButton > button:not([kind="primary"]):hover {
    background: #eef2ff;
    border-color: #818cf8;
    transform: translateY(-2px);
}


/* =========================================================
   النتيجة الآمنة
   ========================================================= */

.result-safe {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5
    );
    border: 2px solid #86efac;
    border-radius: 24px;
    padding: 28px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 12px 30px rgba(34,197,94,0.12);
}

.result-danger {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #ffe4e6
    );
    border: 2px solid #fda4af;
    border-radius: 24px;
    padding: 28px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 12px 30px rgba(244,63,94,0.12);
}

.result-icon {
    font-size: 52px;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}


/* =========================================================
   Confidence
   ========================================================= */

.confidence-card {
    background: white;
    border-radius: 20px;
    padding: 20px;
    margin: 20px 0;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(15,23,42,0.07);
}

.confidence-title {
    color: #64748b;
    font-weight: 700;
}

.confidence-value {
    font-size: 34px;
    font-weight: 800;
    color: #6366f1;
}


/* =========================================================
   الخط الفاصل
   ========================================================= */

hr {
    border: none;
    height: 2px;
    background: linear-gradient(
        90deg,
        transparent,
        #a5b4fc,
        #c084fc,
        transparent
    );
    margin: 30px 0;
}


/* =========================================================
   Expander
   ========================================================= */

.streamlit-expanderHeader {
    border-radius: 16px !important;
    font-weight: 800 !important;
}


/* =========================================================
   Footer
   ========================================================= */

.custom-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding: 30px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. قاعدة بيانات التدريب
# =========================================================

texts = [

    # ==================== English Safe ====================

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
    "Have a wonderful evening",

    *[f"Hope you are doing well today {i}" for i in range(61, 301)],


    # ==================== English Suspicious ====================

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
    "Click here for exclusive adult content",
    "Download this app to win cash",
    "Get 90% off on all products",
    "Immediate cash payout available",
    "Your account password has been compromised",
    "Click now to secure your funds",
    "Win a free shopping spree",
    "Earn a passive income easily",
    "Exclusive discount code for active users",
    "Congratulations! You are selected for a scholarship",
    "Claim your free voucher before midnight",
    "Earn commission from your phone",
    "Get free gift cards instantly",
    "Your business loan is approved",
    "Click to upgrade your account for free",
    "Huge clearance sale, buy now",
    "You won a free hotel stay",
    "Get paid to watch videos",
    "Urgent action required on your bank profile",
    "Claim your tax refund now",

    *[f"Urgent alert click link to claim reward {i}" for i in range(71, 301)],


    # ==================== Arabic Safe ====================

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
    "أنا أستيقظ الآن",

    *[f"رسالة محادثة طبيعية آمنة رقم {i}" for i in range(57, 301)],


    # ==================== Arabic Suspicious ====================

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
    "أنت الزائر رقم مئة اربح جائزة",

    *[f"تنبيه احتيال رسالة مشبوهة رقم {i}" for i in range(67, 301)]
]


# =========================================================
# 4. التصنيفات - الإصلاح الأساسي
# =========================================================

# عدد الرسائل الحقيقي في كل مجموعة
english_safe_count = 300
english_suspicious_count = 300
arabic_safe_count = 300

arabic_suspicious_count = len(texts) - (
    english_safe_count
    + english_suspicious_count
    + arabic_safe_count
)

# منع أي خطأ في حالة تغير البيانات
if arabic_suspicious_count < 0:
    st.error("❌ يوجد خطأ في تقسيم بيانات التدريب.")
    st.stop()

labels = (
    ["Safe Message"] * english_safe_count
    + ["Suspicious Message"] * english_suspicious_count
    + ["Safe Message"] * arabic_safe_count
    + ["Suspicious Message"] * arabic_suspicious_count
)


# =========================================================
# 5. التحقق النهائي
# =========================================================

if len(texts) != len(labels):

    st.error(
        f"""
        ❌ خطأ في البيانات

        عدد الرسائل: {len(texts)}

        عدد التصنيفات: {len(labels)}
        """
    )

    st.stop()


# =========================================================
# 6. تدريب النموذج
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
# 7. العنوان الرئيسي
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">🛡️</div>

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
# 8. بطاقة تعريف النظام
# =========================================================

st.markdown("""
<div class="info-card">

    <div class="info-title">
        ✨ افحص رسالتك بثقة
    </div>

    <div class="info-text">
        أدخل أي رسالة باللغة العربية أو الإنجليزية،
        وسيقوم النظام بتحليلها باستخدام تقنيات
        تعلم الآلة لتحديد ما إذا كانت آمنة أو مشبوهة.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. الإحصائيات
# =========================================================

st.markdown(f"""
<div class="stats">

    <div class="stat-box">
        <div class="stat-number">{len(texts)}</div>
        <div class="stat-label">رسالة تدريب</div>
    </div>

    <div class="stat-box">
        <div class="stat-number">2</div>
        <div class="stat-label">لغة</div>
    </div>

    <div class="stat-box">
        <div class="stat-number">AI</div>
        <div class="stat-label">تحليل ذكي</div>
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 10. صندوق الرسالة
# =========================================================

st.markdown(
    '<div class="message-label">✍️ اكتب الرسالة التي تريد فحصها</div>',
    unsafe_allow_html=True
)


# مفتاح ثابت للـ text_area
if "message_text" not in st.session_state:
    st.session_state.message_text = ""


user_message = st.text_area(
    "Message",
    key="message_text",
    height=150,
    placeholder="مثال: Congratulations! You won a free prize...",
    label_visibility="collapsed"
)


# =========================================================
# 11. الأزرار
# =========================================================

col1, col2 = st.columns(2)

with col1:

    check_btn = st.button(
        "🔍  فحص الرسالة",
        use_container_width=True,
        type="primary"
    )

with col2:

    clear_btn = st.button(
        "🧹  مسح الرسالة",
        use_container_width=True
    )


# =========================================================
# 12. زر المسح
# =========================================================

if clear_btn:

    st.session_state.message_text = ""

    if "prediction" in st.session_state:
        del st.session_state["prediction"]

    if "confidence" in st.session_state:
        del st.session_state["confidence"]

    st.rerun()


# =========================================================
# 13. فحص الرسالة
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

        max_probability = probabilities.max()

        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = max_probability

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align:center;'>📊 نتيجة التحليل</h3>",
            unsafe_allow_html=True
        )


        # =================================================
        # رسالة مشبوهة
        # =================================================

        if prediction == "Suspicious Message":

            st.markdown("""
            <div class="result-danger">

                <div class="result-icon">🚨</div>

                <div class="result-title">
                    رسالة مشبوهة
                </div>

                <div style="color:#9f1239; margin-top:8px;">
                    Suspicious Message Detected
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.warning(
                "⚠️ تحتوي الرسالة على أنماط تشبه "
                "الرسائل الاحتيالية أو الإعلانية المزعجة."
            )

            st.markdown("""
            **🛡️ نصائح السلامة**

            • لا تضغط على الروابط غير الموثوقة.

            • لا تشارك كلمات المرور أو المعلومات الشخصية.

            • تحقق من مصدر الرسالة.

            • لا تستجيب للطلبات المالية غير المتوقعة.
            """)


        # =================================================
        # رسالة آمنة
        # =================================================

        else:

            st.markdown("""
            <div class="result-safe">

                <div class="result-icon">✅</div>

                <div class="result-title">
                    رسالة آمنة
                </div>

                <div style="color:#166534; margin-top:8px;">
                    Safe Message Detected
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                "✨ تبدو الرسالة مشابهة للرسائل الطبيعية "
                "الموجودة في بيانات التدريب."
            )

            st.markdown("""
            **💡 نصيحة**

            يمكنك التعامل معها بشكل طبيعي، لكن استمر
            دائماً في الحذر من الروابط والطلبات غير المتوقعة.
            """)


        # =================================================
        # درجة الثقة
        # =================================================

        confidence_percentage = max_probability * 100

        st.markdown(f"""
        <div class="confidence-card">

            <div class="confidence-title">
                🤖 درجة ثقة النموذج
            </div>

            <div class="confidence-value">
                {confidence_percentage:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(max_probability)


# =========================================================
# 14. معلومات النظام
# =========================================================

with st.expander("ℹ️ معلومات عن النظام"):

    st.markdown("""
    ### 🧠 كيف يعمل النظام؟

    يستخدم النظام تقنيات **Machine Learning**
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
        f"📚 عدد رسائل التدريب: {len(texts)}"
    )


# =========================================================
# 15. Footer
# =========================================================

st.markdown("""
<div class="custom-footer">

    🛡️ Message Safety Detector

    <br>

    AI-Powered Bilingual Message Analysis

    <br><br>

    ✨ Smart • Fast • Bilingual

</div>
""", unsafe_allow_html=True)
