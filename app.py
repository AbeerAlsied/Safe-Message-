import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. PREMIUM DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.stApp {

    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(124,58,237,.20),
            transparent 28%
        ),

        radial-gradient(
            circle at 100% 0%,
            rgba(236,72,153,.18),
            transparent 28%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(14,165,233,.16),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #f8faff 0%,
            #eef2ff 45%,
            #faf5ff 100%
        );

    color: #172033;
}


/* ---------------------------------------------------------
   HIDE STREAMLIT DEFAULT ELEMENTS
--------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ---------------------------------------------------------
   MAIN WIDTH
--------------------------------------------------------- */

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HERO
========================================================= */

.hero {

    position: relative;
    overflow: hidden;

    padding: 42px 25px 38px;

    margin-bottom: 25px;

    border-radius: 32px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed 48%,
            #db2777
        );

    box-shadow:
        0 25px 60px rgba(79,70,229,.25);

    color: white;
}


.hero::before {

    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    border-radius: 50%;

    background: rgba(255,255,255,.10);

    top: -120px;
    left: -80px;
}


.hero::after {

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    border-radius: 50%;

    background: rgba(255,255,255,.08);

    bottom: -160px;
    right: -100px;
}


.hero-icon {

    position: relative;

    font-size: 70px;

    filter:
        drop-shadow(
            0 10px 15px rgba(0,0,0,.20)
        );
}


.hero-title {

    position: relative;

    font-size: 42px;

    font-weight: 900;

    margin-top: 5px;

    letter-spacing: -.5px;
}


.hero-subtitle {

    position: relative;

    font-size: 16px;

    opacity: .92;

    line-height: 1.9;

    margin-top: 8px;
}


.hero-badge {

    position: relative;

    display: inline-block;

    margin-top: 18px;

    padding: 7px 18px;

    border-radius: 50px;

    background: rgba(255,255,255,.15);

    border: 1px solid rgba(255,255,255,.25);

    font-size: 12px;

    font-weight: 700;

    backdrop-filter: blur(10px);
}


/* =========================================================
   STATISTICS
========================================================= */

.stats-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 14px;

    margin: 22px 0;
}


.stat-card {

    background: rgba(255,255,255,.78);

    border: 1px solid rgba(255,255,255,.8);

    border-radius: 22px;

    padding: 20px 10px;

    text-align: center;

    box-shadow:
        0 12px 35px rgba(30,41,59,.07);

    backdrop-filter: blur(15px);

    transition: .25s;
}


.stat-card:hover {

    transform:
        translateY(-5px);

    box-shadow:
        0 18px 40px rgba(79,70,229,.13);
}


.stat-icon {

    font-size: 27px;
}


.stat-number {

    font-size: 25px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #9333ea
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.stat-label {

    color: #64748b;

    font-size: 11px;

    font-weight: 600;
}


/* =========================================================
   SECTION TITLE
========================================================= */

.section-title {

    font-size: 23px;

    font-weight: 900;

    color: #1e293b;

    margin-top: 30px;

    margin-bottom: 5px;
}


.section-subtitle {

    color: #64748b;

    font-size: 13px;

    margin-bottom: 15px;
}


/* =========================================================
   MESSAGE CARD
========================================================= */

.message-card {

    background: rgba(255,255,255,.82);

    border-radius: 26px;

    padding: 24px;

    border: 1px solid rgba(255,255,255,.9);

    box-shadow:
        0 18px 50px rgba(30,41,59,.08);

    backdrop-filter: blur(18px);
}


/* =========================================================
   TEXT AREA
========================================================= */

textarea {

    border-radius: 18px !important;

    border:
        2px solid #e2e8f0 !important;

    background:
        rgba(248,250,252,.9) !important;

    color:
        #172033 !important;

    font-size:
        15px !important;

    padding:
        16px !important;

    transition:
        .25s !important;
}


textarea:focus {

    border:
        2px solid #8b5cf6 !important;

    box-shadow:
        0 0 0 4px
        rgba(139,92,246,.10) !important;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {

    min-height: 55px !important;

    border-radius: 17px !important;

    font-size: 15px !important;

    font-weight: 800 !important;

    border: none !important;

    transition: .25s !important;
}


.stButton > button:hover {

    transform:
        translateY(-3px) !important;

    box-shadow:
        0 12px 30px
        rgba(79,70,229,.18) !important;
}


/* =========================================================
   RESULT
========================================================= */

.result-card {

    padding: 32px 20px;

    margin-top: 25px;

    border-radius: 28px;

    text-align: center;

    box-shadow:
        0 18px 50px rgba(30,41,59,.08);
}


.result-safe {

    background:
        linear-gradient(
            135deg,
            #ecfdf5,
            #d1fae5
        );

    border:
        1px solid #86efac;
}


.result-danger {

    background:
        linear-gradient(
            135deg,
            #fff1f2,
            #ffe4e6
        );

    border:
        1px solid #fda4af;
}


.result-icon {

    font-size: 62px;
}


.result-title {

    font-size: 29px;

    font-weight: 900;

    margin-top: 8px;
}


.result-subtitle {

    font-size: 14px;

    margin-top: 5px;
}


/* =========================================================
   CONFIDENCE
========================================================= */

.confidence-card {

    background:
        rgba(255,255,255,.90);

    border-radius:
        22px;

    padding:
        22px;

    margin-top:
        18px;

    text-align:
        center;

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 12px 35px
        rgba(30,41,59,.06);
}


.confidence-title {

    color:
        #64748b;

    font-size:
        13px;

    font-weight:
        700;
}


.confidence-value {

    font-size:
        42px;

    font-weight:
        900;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #9333ea,
            #db2777
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


/* =========================================================
   INFO BOX
========================================================= */

.info-box {

    background:
        rgba(255,255,255,.80);

    border:
        1px solid #e2e8f0;

    border-radius:
        20px;

    padding:
        20px;

    margin-top:
        15px;

    color:
        #475569;

    line-height:
        1.9;

    box-shadow:
        0 8px 25px
        rgba(30,41,59,.05);
}


/* =========================================================
   SECURITY BOX
========================================================= */

.security-box {

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

    border:
        1px solid #c7d2fe;

    border-radius:
        22px;

    padding:
        22px;

    margin-top:
        18px;

    color:
        #3730a3;

    line-height:
        2;
}


/* =========================================================
   DIVIDER
========================================================= */

hr {

    border:
        none;

    height:
        1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            #c4b5fd,
            transparent
        );

    margin:
        30px 0;
}


/* =========================================================
   EXPANDER
========================================================= */

[data-testid="stExpander"] {

    border:
        1px solid #e2e8f0 !important;

    border-radius:
        18px !important;

    background:
        rgba(255,255,255,.75) !important;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {

    text-align:
        center;

    margin-top:
        45px;

    padding:
        25px;

    color:
        #94a3b8;

    font-size:
        12px;

    border-top:
        1px solid #e2e8f0;
}


/* =========================================================
   MOBILE
========================================================= */

@media (max-width: 700px) {

    .hero-title {
        font-size: 30px;
    }

    .hero-icon {
        font-size: 55px;
    }

    .stats-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .hero {
        padding:
            32px 15px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. TRAINING DATA
# =========================================================

texts = [

    # =====================================================
    # ENGLISH SAFE - 300
    # =====================================================

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

    *[
        f"Hope you are doing well today {i}"
        for i in range(61, 301)
    ],


    # =====================================================
    # ENGLISH SUSPICIOUS - 300
    # =====================================================

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
    "Click here for exclusive content",
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

    *[
        f"Urgent alert click link to claim reward {i}"
        for i in range(71, 301)
    ],


    # =====================================================
    # ARABIC SAFE - 300
    # =====================================================

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

    *[
        f"رسالة محادثة طبيعية آمنة رقم {i}"
        for i in range(57, 301)
    ],


    # =====================================================
    # ARABIC SUSPICIOUS - 300
    # =====================================================

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
    "احصل على بطاقة هدية",
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

    *[
        f"تنبيه احتيال رسالة مشبوهة رقم {i}"
        for i in range(67, 301)
    ]
]


# =========================================================
# 4. LABELS
# =========================================================

english_safe_count = 300
english_suspicious_count = 300
arabic_safe_count = 300
arabic_suspicious_count = 300

labels = (
    ["Safe Message"] * english_safe_count
    +
    ["Suspicious Message"] * english_suspicious_count
    +
    ["Safe Message"] * arabic_safe_count
    +
    ["Suspicious Message"] * arabic_suspicious_count
)


# =========================================================
# 5. DATA VALIDATION
# =========================================================

if len(texts) != len(labels):

    st.error(
        f"""
        خطأ في البيانات:

        عدد الرسائل = {len(texts)}

        عدد التصنيفات = {len(labels)}
        """
    )

    st.stop()


# =========================================================
# 6. TRAIN AI MODEL
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
# 7. HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">
        🛡️
    </div>

    <div class="hero-title">
        Message Safety Detector
    </div>

    <div class="hero-subtitle">

        نظام ذكي لفحص وتصنيف الرسائل

        <br>

        Intelligent Bilingual Message Security System

    </div>

    <div class="hero-badge">

        ✨ AI POWERED • TF-IDF • NAIVE BAYES

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. STATISTICS
# =========================================================

st.markdown(f"""
<div class="stats-grid">

    <div class="stat-card">

        <div class="stat-icon">
            📚
        </div>

        <div class="stat-number">
            {len(texts)}
        </div>

        <div class="stat-label">
            رسائل التدريب
        </div>

    </div>


    <div class="stat-card">

        <div class="stat-icon">
            🌐
        </div>

        <div class="stat-number">
            02
        </div>

        <div class="stat-label">
            اللغات
        </div>

    </div>


    <div class="stat-card">

        <div class="stat-icon">
            🧠
        </div>

        <div class="stat-number">
            AI
        </div>

        <div class="stat-label">
            تحليل ذكي
        </div>

    </div>


    <div class="stat-card">

        <div class="stat-icon">
            🛡️
        </div>

        <div class="stat-number">
            24/7
        </div>

        <div class="stat-label">
            فحص الرسائل
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. INTRODUCTION
# =========================================================

st.markdown("""
<div class="info-box">

    <h3 style="color:#4f46e5;">
        ✨ افحص رسالتك بثقة
    </h3>

    أدخل أي رسالة باللغة العربية أو الإنجليزية،
    وسيقوم نظام الذكاء الاصطناعي بتحليل محتواها
    وتصنيفها إلى رسالة آمنة أو رسالة مشبوهة.

</div>
""", unsafe_allow_html=True)


# =========================================================
# 10. INPUT
# =========================================================

st.markdown("""
<div class="section-title">
    🔍 فحص رسالة جديدة
</div>

<div class="section-subtitle">
    اكتب الرسالة التي تريد تحليلها ثم اضغط على زر الفحص.
</div>
""", unsafe_allow_html=True)


if "message" not in st.session_state:

    st.session_state.message = ""


if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "confidence" not in st.session_state:

    st.session_state.confidence = None


st.markdown(
    '<div class="message-card">',
    unsafe_allow_html=True
)


user_message = st.text_area(
    "Message",
    value=st.session_state.message,
    height=170,
    placeholder=(
        "✍️ اكتب الرسالة هنا...\n\n"
        "مثال:\n"
        "Congratulations! You won a free prize."
    ),
    label_visibility="collapsed"
)


st.session_state.message = user_message


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 11. BUTTONS
# =========================================================

st.write("")


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
# 12. CLEAR BUTTON
# =========================================================

if clear_btn:

    st.session_state.message = ""

    st.session_state.prediction = None

    st.session_state.confidence = None

    st.rerun()


# =========================================================
# 13. ANALYZE
# =========================================================

if check_btn:

    message = st.session_state.message.strip()

    if not message:

        st.warning(
            "⚠️ الرجاء كتابة رسالة أولاً حتى يتمكن النظام من فحصها."
        )

    else:

        msg_vector = vectorizer.transform([message])

        prediction = model.predict(msg_vector)[0]

        probabilities = model.predict_proba(msg_vector)[0]

        confidence = probabilities.max()

        st.session_state.prediction = prediction

        st.session_state.confidence = confidence


# =========================================================
# 14. RESULT
# =========================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    confidence = st.session_state.confidence

    confidence_percentage = confidence * 100


    st.markdown("<hr>", unsafe_allow_html=True)


    st.markdown("""
    <div class="section-title">
        📊 نتيجة التحليل
    </div>

    <div class="section-subtitle">
        AI Analysis Result
    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # SUSPICIOUS
    # =====================================================

    if prediction == "Suspicious Message":

        st.markdown("""
        <div class="result-card result-danger">

            <div class="result-icon">
                🚨
            </div>

            <div class="result-title">
                رسالة مشبوهة
            </div>

            <div class="result-subtitle">
                Suspicious Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div class="security-box">

            <b>⚠️ لماذا ظهرت هذه النتيجة؟</b>

            <br>

            اكتشف نموذج الذكاء الاصطناعي أنماطاً
            في الرسالة تشبه الرسائل المشبوهة
            الموجودة في بيانات التدريب.

            <br><br>

            <b>🛡️ نصائح الأمان:</b>

            <br>

            🔸 لا تضغط على الروابط غير الموثوقة.

            <br>

            🔸 لا تشارك كلمات المرور.

            <br>

            🔸 لا ترسل معلوماتك الشخصية.

            <br>

            🔸 تحقق من مصدر الرسالة.

            <br>

            🔸 انتبه للطلبات المالية غير المتوقعة.

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # SAFE
    # =====================================================

    else:

        st.markdown("""
        <div class="result-card result-safe">

            <div class="result-icon">
                🟢
            </div>

            <div class="result-title">
                رسالة آمنة
            </div>

            <div class="result-subtitle">
                Safe Message Detected
            </div>

        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div class="info-box">

            <b style="color:#15803d;">
                ✨ لماذا ظهرت هذه النتيجة؟
            </b>

            <br>

            الرسالة تشبه أنماط المحادثات الطبيعية
            الموجودة في بيانات التدريب.

            <br><br>

            <b>💡 ملاحظة:</b>

            <br>

            التصنيف الآمن لا يعني أن الرسالة موثوقة
            بشكل مطلق، لذلك يُفضل دائماً الحذر
            من الروابط والطلبات غير المتوقعة.

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.markdown(f"""
    <div class="confidence-card">

        <div class="confidence-title">

            🤖 AI CONFIDENCE SCORE

        </div>

        <div class="confidence-value">

            {confidence_percentage:.2f}%

        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
        ">

            درجة ثقة النموذج التقريبية

        </div>

    </div>
    """, unsafe_allow_html=True)


    st.progress(
        confidence,
        text=f"AI Confidence: {confidence_percentage:.2f}%"
    )


# =========================================================
# 15. HOW IT WORKS
# =========================================================

st.markdown("""
<div class="section-title">
    ⚙️ كيف يعمل النظام؟
</div>

<div class="section-subtitle">
    How the AI system works
</div>
""", unsafe_allow_html=True)


step1, step2, step3 = st.columns(3)


with step1:

    st.markdown("""
    <div class="info-box">

        <h3 style="color:#4f46e5;">
            01
        </h3>

        ✍️ <b>إدخال الرسالة</b>

        <br><br>

        يقوم المستخدم بإدخال الرسالة
        باللغة العربية أو الإنجليزية.

    </div>
    """, unsafe_allow_html=True)


with step2:

    st.markdown("""
    <div class="info-box">

        <h3 style="color:#7c3aed;">
            02
        </h3>

        🧠 <b>تحليل النص</b>

        <br><br>

        يتم تحويل النص إلى تمثيل رقمي
        باستخدام TF-IDF Character N-Grams.

    </div>
    """, unsafe_allow_html=True)


with step3:

    st.markdown("""
    <div class="info-box">

        <h3 style="color:#db2777;">
            03
        </h3>

        🛡️ <b>التصنيف</b>

        <br><br>

        يقوم Multinomial Naive Bayes
        بتحديد نوع الرسالة.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 16. SECURITY TIPS
# =========================================================

st.markdown("""
<div class="section-title">
    🔐 نصائح للحماية من الرسائل المشبوهة
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="security-box">

    🛡️ <b>احمِ معلوماتك الشخصية</b>

    <br>

    لا تشارك كلمات المرور أو البيانات الحساسة
    مع الرسائل غير الموثوقة.

    <br><br>

    🔗 <b>تحقق من الروابط</b>

    <br>

    لا تضغط على أي رابط مجهول قبل التأكد
    من مصدره.

    <br><br>

    💰 <b>انتبه للجوائز والأموال</b>

    <br>

    الرسائل التي تعدك بأموال أو جوائز
    بشكل غير متوقع تحتاج إلى الحذر.

    <br><br>

    🚨 <b>انتبه للرسائل العاجلة</b>

    <br>

    الضغط عليك لاتخاذ قرار سريع قد يكون
    علامة على رسالة مشبوهة.

</div>
""", unsafe_allow_html=True)


# =========================================================
# 17. SYSTEM INFORMATION
# =========================================================

with st.expander("🧠 معلومات تقنية عن النظام"):

    st.markdown(f"""
    ### 🤖 Artificial Intelligence

    **Algorithm**

    `Multinomial Naive Bayes`

    **Text Representation**

    `TF-IDF Character N-Grams`

    **N-Gram Range**

    `(2, 5)`

    **Training Messages**

    `{len(texts)}`

    **Languages**

    🇸🇦 العربية + 🇬🇧 English

    **Classes**

    🟢 Safe Message

    🔴 Suspicious Message
    """)


# =========================================================
# 18. FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    🛡️ <b>Message Safety Detector</b>

    <br><br>

    Intelligent Bilingual Message Security System

    <br>

    Powered by Machine Learning • TF-IDF • Naive Bayes

    <br><br>

    © 2026 AI Graduation Project

</div>
""", unsafe_allow_html=True)
