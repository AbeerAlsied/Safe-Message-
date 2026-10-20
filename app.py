import re
import ipaddress
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>
@import url(
'https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap'
);

:root {
    --primary: #6C63FF;
    --secondary: #8B5CF6;
    --background: #F5F7FC;
    --card: #FFFFFF;
    --text: #20243A;
    --muted: #70758A;
    --border: #E5E8F2;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Cairo', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

.hero {
    background: linear-gradient(
        135deg,
        #5146E5 0%,
        #7867F7 55%,
        #9A7BFA 100%
    );
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(81,70,229,0.18);
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero p {
    font-size: 16px;
    line-height: 1.9;
    margin-bottom: 0;
    color: #F0EEFF;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    margin: 25px 0 12px 0;
    color: #262849;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid var(--border);
    margin-bottom: 15px;
}

.info-card h3 {
    font-size: 18px;
    margin-bottom: 8px;
}

.info-card p {
    color: var(--muted);
    line-height: 1.8;
}

div.stButton > button {
    border-radius: 12px;
    min-height: 45px;
    font-weight: 700;
    border: 1px solid #DDDDF7;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: var(--primary);
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 16px;
}

textarea {
    border-radius: 14px !important;
}

.footer {
    text-align: center;
    color: #777B90;
    font-size: 13px;
    padding: 25px 0;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 768px) {
    .hero {
        padding: 23px;
    }

    .hero h1 {
        font-size: 25px;
    }

    .hero p {
        font-size: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================
# TRANSLATIONS
# =========================================

TRANSLATIONS = {
    "العربية": {
        "title": "كاشف الرسائل الذكي",
        "subtitle": "اكتشفي مؤشرات الاحتيال قبل التفاعل مع الرسائل.",
        "dashboard": "لوحة الفحص",
        "history": "سجل الفحوصات",
        "about": "عن المشروع",
        "input_title": "أدخلي الرسالة المراد فحصها",
        "placeholder": "الصقي الرسالة هنا...",
        "check": "تحليل الرسالة",
        "clear": "مسح النص",
        "empty": "الرجاء إدخال رسالة أولاً.",
        "result": "نتيجة التحليل",
        "score": "درجة الخطورة",
        "level": "مستوى الخطورة",
        "links": "الروابط المكتشفة",
        "reasons": "مؤشرات الاشتباه",
        "advice": "التوصية",
        "high": "مرتفعة",
        "medium": "متوسطة",
        "low": "منخفضة حسب المؤشرات",
        "total": "إجمالي الفحوصات",
        "high_count": "خطورة مرتفعة",
        "medium_count": "خطورة متوسطة",
        "low_count": "خطورة منخفضة",
        "download": "تنزيل السجل CSV",
        "clear_history": "مسح السجل",
        "no_history": "لا توجد فحوصات مسجلة بعد.",
        "no_reasons": "لم تظهر مؤشرات مطابقة للقواعد الحالية.",
        "no_links": "لم يتم اكتشاف روابط.",
        "warning": "هذا النظام أداة مساعدة، ولا يضمن اكتشاف جميع الرسائل الاحتيالية.",
        "safe_note": "عدم ظهور مؤشرات لا يعني أن الرسالة آمنة بالتأكيد.",
        "footer": "Message Safety Detector | مشروع توعوي للأمن الرقمي"
    },
    "English": {
        "title": "Smart Message Safety Detector",
        "subtitle": "Identify suspicious message patterns before responding.",
        "dashboard": "Scanner",
        "history": "Scan History",
        "about": "About",
        "input_title": "Enter a message to analyze",
        "placeholder": "Paste your message here...",
        "check": "Analyze Message",
        "clear": "Clear Text",
        "empty": "Please enter a message first.",
        "result": "Analysis Result",
        "score": "Risk Score",
        "level": "Risk Level",
        "links": "Detected Links",
        "reasons": "Suspicious Indicators",
        "advice": "Recommendation",
        "high": "High",
        "medium": "Medium",
        "low": "Low based on detected indicators",
        "total": "Total Scans",
        "high_count": "High Risk",
        "medium_count": "Medium Risk",
        "low_count": "Low Risk",
        "download": "Download CSV History",
        "clear_history": "Clear History",
        "no_history": "No scans recorded yet.",
        "no_reasons": "No indicators matched the current rules.",
        "no_links": "No links detected.",
        "warning": "This tool is an assistant and cannot detect every fraudulent message.",
        "safe_note": "No detected indicators does not prove that a message is safe.",
        "footer": "Message Safety Detector | Digital Safety Awareness Project"
    }
}


# =========================================
# SESSION STATE
# =========================================

if "language" not in st.session_state:
    st.session_state.language = "العربية"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_message" not in st.session_state:
    st.session_state.last_message = ""


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.markdown("## 🛡️ Message Safety")

    st.caption("Digital Security Assistant")

    st.divider()

    language = st.selectbox(
        "Language / اللغة",
        ["العربية", "English"],
        index=0 if st.session_state.language == "العربية" else 1
    )

    st.session_state.language = language

    t = TRANSLATIONS[language]

    page = st.radio(
        "Navigation",
        [
            t["dashboard"],
            t["history"],
            t["about"]
        ]
    )

    st.divider()

    st.markdown("### 🔐 نصائح أمنية")

    if language == "العربية":
        st.write("• لا تشاركي رمز التحقق مع الآخرين.")
        st.write("• تحققي من عنوان الموقع قبل إدخال بياناتك.")
        st.write("• لا تدفعي رسوماً للحصول على جائزة غير مؤكدة.")
    else:
        st.write("• Never share verification codes.")
        st.write("• Check website addresses carefully.")
        st.write("• Be cautious of unexpected prize fees.")


# =========================================
# TRAINING DATA
# =========================================

SAFE_MESSAGES = [
    "Your class starts tomorrow morning.",
    "Thank you for helping with the project.",
    "The meeting has been moved to Monday.",
    "Please send me the homework.",
    "Your order has been delivered.",
    "The university published the exam schedule.",
    "Can we meet at the library today?",
    "Your appointment is confirmed.",
    "Happy birthday and best wishes.",
    "The report is ready for review.",
    "مرحبا كيف حالك اليوم؟",
    "موعد المحاضرة بكرة الساعة عشرة.",
    "شكرا ليك على المساعدة.",
    "تم تغيير موعد الاجتماع.",
    "ممكن ترسلي لي الواجب؟",
    "طلبك وصل بنجاح.",
    "الجامعة أعلنت جدول الامتحانات.",
    "نتلاقى في المكتبة العصر؟",
    "تم تأكيد موعدك لبكرة.",
    "كل سنة وأنت طيب وبخير.",
    "أرسلت ليك التقرير للمراجعة.",
    "يرجى مراجعة الجدول الدراسي.",
    "سأرسل الملفات بعد المحاضرة.",
    "The team will discuss the project tomorrow.",
    "Please confirm your meeting attendance.",
    "I will call you after work.",
    "شكرا على رسالتك.",
    "بكرة عندنا اجتماع في الجامعة.",
    "وصلني الملف وسأراجعه.",
    "تم استلام طلب التسجيل."
]

SUSPICIOUS_MESSAGES = [
    "Congratulations you won a million dollars pay a fee now.",
    "Your bank account is blocked send your password immediately.",
    "Claim your prize before it expires.",
    "You inherited a fortune send your bank details.",
    "Send your OTP code to verify your account.",
    "Pay a processing fee to receive compensation.",
    "Urgent your account will be closed unless you act now.",
    "You won a lottery you never entered.",
    "Send your credit card details to receive your reward.",
    "Click this link to claim free money immediately.",
    "مبروك ربحت مليون دولار أرسل رسوم الاستلام الآن.",
    "حسابك البنكي اتقفل أرسل كلمة المرور فوراً.",
    "ارسل رمز التحقق لاستلام الجائزة.",
    "عندك تعويض مالي كبير ادفع رسوم المعالجة.",
    "اربح جائزة ضخمة اضغط الرابط الآن.",
    "سيتم إغلاق حسابك إذا لم ترسل بياناتك فوراً.",
    "أرسل رقم بطاقتك البنكية لاستلام المبلغ.",
    "تم اختيارك للفوز بجائزة لم تشارك فيها.",
    "أرسل رمز OTP لتأكيد الحساب.",
    "فرصة استثمار مضمونة وأرباح كبيرة بدون مخاطر.",
    "ادفع رسوم الشحن حتى تستلم جائزتك.",
    "أرسل بياناتك الشخصية لاستلام التعويض الدولي.",
    "Your account will be suspended verify your password now.",
    "Pay a fee before receiving your inheritance.",
    "Send your bank details to receive a large reward.",
    "You have won a prize pay the delivery charge.",
    "ارسل بياناتك السرية لتأكيد استلام المبلغ.",
    "مبروك تم اختيارك لجائزة مالية كبيرة.",
    "ادفع رسوم التسجيل لتحصل على التعويض.",
    "أرسل كلمة المرور ورمز التأكيد الآن."
]


# =========================================
# MACHINE LEARNING MODEL
# =========================================

@st.cache_resource
def train_model():

    texts = SAFE_MESSAGES + SUSPICIOUS_MESSAGES

    labels = (
        ["Safe"] * len(SAFE_MESSAGES)
        + ["Suspicious"] * len(SUSPICIOUS_MESSAGES)
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.3)
        )
    ])

    pipeline.fit(texts, labels)

    return pipeline


model = train_model()


# =========================================
# FRAUD DETECTION RULES
# =========================================

RULES = [
    (
        "طلب معلومات سرية / Secret information request",
        35,
        r"password|passcode|otp|verification code|credit card details|"
        r"bank details|card number|كلمة المرور|الرقم السري|"
        r"رمز التحقق|رمز التأكيد|بيانات البطاقة|بياناتك البنكية|رقم البطاقة"
    ),
    (
        "رسوم قبل استلام الأموال / Advance fee",
        30,
        r"processing fee|advance fee|pay.{0,30}fee|transfer fee|"
        r"release fee|shipping fee|رسوم المعالجة|رسوم الاستلام|"
        r"ادفع.{0,30}رسوم|دفع.{0,30}رسوم|رسوم التحويل|رسوم الشحن"
    ),
    (
        "جائزة أو تعويض مالي / Prize or compensation",
        20,
        r"you won|winner|lottery|claim your prize|inheritance|"
        r"compensation|million dollars|ربحت|فزت|جائزة|تعويض مالي|"
        r"ورث|مليون دولار|استلام المبلغ"
    ),
    (
        "استعجال غير معتاد / Unusual urgency",
        15,
        r"urgent|immediately|act now|limited time|account will be closed|"
        r"account suspended|within 24 hours|فوراً|حالاً|عاجل|بسرعة|"
        r"سيتم إغلاق حسابك|خلال 24 ساعة"
    ),
    (
        "وعود بأرباح مضمونة / Guaranteed profits",
        20,
        r"guaranteed profit|risk.free investment|double your money|"
        r"أرباح مضمونة|بدون مخاطر|ضاعف أموالك|ربح مضمون"
    ),
    (
        "ذكر جهة رسمية يحتاج للتحقق / Official institution reference",
        10,
        r"world bank|united nations|official grant|government compensation|"
        r"البنك الدولي|الأمم المتحدة|تعويض حكومي|منحة رسمية"
    )
]


# =========================================
# URL INSPECTION
# =========================================

def extract_urls(message):

    return re.findall(
        r"""(?:https?://|www\.)[^\s<>"']+""",
        message,
        flags=re.IGNORECASE
    )


def inspect_url(url):

    findings = []

    url = url.rstrip(".,!?;:،؛)")

    normalized = (
        url
        if url.lower().startswith(("http://", "https://"))
        else "http://" + url
    )

    try:

        parsed = urlparse(normalized)
        host = parsed.hostname or ""

        if not host:
            return ["تعذر تحليل الرابط"]

        if parsed.username or parsed.password:
            findings.append("بيانات دخول داخل عنوان الرابط")

        if (
            host.lower().startswith("xn--")
            or ".xn--" in host.lower()
        ):
            findings.append("اسم نطاق بترميز قد يخفي بعض الحروف")

        try:
            ipaddress.ip_address(host)
            findings.append("الرابط يستخدم عنوان IP")
        except ValueError:
            pass

        shorteners = {
            "bit.ly", "tinyurl.com", "t.co",
            "is.gd", "cutt.ly", "rb.gy"
        }

        if host.lower() in shorteners:
            findings.append("الرابط مختصر")

        if parsed.scheme.lower() == "http":
            findings.append("الرابط لا يستخدم HTTPS")

    except Exception:
        findings.append("تعذر تحليل الرابط")

    return findings


# =========================================
# MESSAGE ANALYSIS
# =========================================

def analyze_message(message):

    findings = []
    score = 0

    for name, weight, pattern in RULES:

        if re.search(pattern, message, re.IGNORECASE):

            findings.append(name)
            score += weight

    urls = extract_urls(message)

    for url in urls:

        url_findings = inspect_url(url)

        if url_findings:

            score += 20

            for finding in url_findings:
                findings.append(f"{url}: {finding}")

    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    classes = list(model.classes_)

    suspicious_probability = float(
        probabilities[classes.index("Suspicious")]
    )

    if (
        prediction == "Suspicious"
        and suspicious_probability >= 0.80
    ):

        score +=
