import re
import ipaddress
from urllib.parse import urlparse
from datetime import datetime

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. SESSION STATE
# =========================================================

if "language" not in st.session_state:
    st.session_state.language = "ar"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


def clear_message():
    """Clear the message before the text widget is rendered again."""
    st.session_state.message_input = ""
    st.session_state.last_result = None


# =========================================================
# 3. TRANSLATIONS
# =========================================================

TEXT = {
    "ar": {
        "title": "🛡️ نظام فحص وتصنيف الرسائل الذكي",
        "subtitle": (
            "افحص الرسائل واكتشف المؤشرات التي قد تدل على "
            "الاحتيال أو الرسائل المشبوهة."
        ),
        "language": "اللغة / Language",
        "sidebar_title": "القائمة الرئيسية",
        "scanner": "🔍 فحص رسالة",
        "history": "📋 سجل الفحوصات",
        "about": "ℹ️ حول النظام",
        "enter_message": "اكتبي الرسالة التي تريدين فحصها:",
        "placeholder": "الصقي الرسالة هنا...",
        "scan": "🔍 فحص الرسالة",
        "clear": "🧹 مسح النص",
        "empty": "الرجاء كتابة رسالة أولًا.",
        "safe": "رسالة منخفضة المؤشرات المشبوهة",
        "suspicious": "رسالة مشبوهة",
        "low": "منخفض",
        "medium": "متوسط",
        "high": "مرتفع",
        "risk": "مستوى الخطورة",
        "score": "درجة الخطورة",
        "classification": "نتيجة التصنيف",
        "reasons": "مؤشرات الفحص",
        "no_reasons": "لم تُكتشف مؤشرات مشبوهة واضحة من القواعد الحالية.",
        "url_details": "نتائج فحص الروابط",
        "no_urls": "لم يتم العثور على روابط في الرسالة.",
        "history_title": "📋 سجل الفحوصات",
        "no_history": "لا توجد فحوصات مسجلة في هذه الجلسة.",
        "download": "📥 تنزيل السجل بصيغة CSV",
        "clear_history": "🗑️ مسح سجل الفحوصات",
        "about_title": "ℹ️ حول النظام",
        "about_text": (
            "نظام تجريبي يستخدم قواعد لاكتشاف مؤشرات الاحتيال، "
            "بالإضافة إلى نموذج تعلم آلي لتصنيف الرسائل."
        ),
        "limitations": "ملاحظات مهمة",
        "limitation_text": (
            "قد يخطئ النظام في تصنيف بعض الرسائل. "
            "النتيجة ليست إثباتًا قاطعًا على أن الرسالة آمنة أو احتيالية. "
            "لا ترسلي كلمات المرور أو رموز التحقق أو المعلومات الشخصية "
            "الحساسة أثناء التجربة."
        ),
        "date": "التاريخ والوقت",
        "message": "الرسالة",
        "result": "التصنيف",
        "risk_level": "مستوى الخطورة",
        "url_safe": "لم تظهر مؤشرات واضحة على خطورة الرابط من الفحوصات البسيطة.",
        "url_suspicious": "الرابط يحتوي على مؤشر يستحق الحذر.",
        "url_ip": "الرابط يستخدم عنوان IP بدلًا من اسم نطاق معتاد.",
        "url_short": "الرابط يستخدم خدمة اختصار روابط.",
        "url_http": "الرابط لا يستخدم اتصال HTTPS.",
        "url_punycode": "اسم النطاق يحتوي على ترميز Punycode.",
        "url_invalid": "تعذر تحليل الرابط بصورة صحيحة.",
        "rule_urgent": "توجد عبارات تحث على التصرف بسرعة أو تهدد بإغلاق الحساب.",
        "rule_secret": "تطلب الرسالة كلمة مرور أو رمز تحقق أو بيانات سرية.",
        "rule_money": "تطلب الرسالة دفع رسوم أو تحويل أموال مقدمًا.",
        "rule_prize": "توجد عبارات عن جائزة أو تعويض أو مبلغ مالي غير متوقع.",
        "rule_personal": "تطلب الرسالة بيانات شخصية أو مالية حساسة.",
        "rule_profit": "توجد وعود بأرباح مضمونة أو عوائد غير واقعية.",
        "rule_authority": "تستخدم الرسالة اسم مؤسسة رسمية أو دولية ضمن سياق يحتاج إلى التحقق.",
        "footer": "Message Safety Detector | نظام تجريبي للتوعية الرقمية",
        "model_suspicious": "نموذج التعلم الآلي صنّف الرسالة على أنها مشبوهة.",
        "model_safe": "نموذج التعلم الآلي لم يصنّف الرسالة على أنها مشبوهة.",
        "probability": "درجة النموذج للرسائل المشبوهة",
        "notice": "تنبيه: لا تعتمدي على نتيجة النظام وحدها لاتخاذ قرار أمني.",
        "history_saved": "تم تسجيل نتيجة الفحص في سجل الجلسة.",
        "history_cleared": "تم مسح سجل الفحوصات.",
        "export_empty": "لا توجد بيانات لتنزيلها.",
    },

    "en": {
        "title": "🛡️ Message Safety Detector",
        "subtitle": (
            "Inspect messages for indicators that may suggest "
            "fraud or suspicious activity."
        ),
        "language": "Language / اللغة",
        "sidebar_title": "Main Menu",
        "scanner": "🔍 Message Scanner",
        "history": "📋 Scan History",
        "about": "ℹ️ About",
        "enter_message": "Enter the message you want to inspect:",
        "placeholder": "Paste your message here...",
        "scan": "🔍 Scan Message",
        "clear": "🧹 Clear Text",
        "empty": "Please enter a message first.",
        "safe": "Low suspicious indicators",
        "suspicious": "Suspicious Message",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "risk": "Risk Level",
        "score": "Risk Score",
        "classification": "Classification",
        "reasons": "Inspection Indicators",
        "no_reasons": "No clear suspicious indicators were detected by the current rules.",
        "url_details": "URL Inspection Results",
        "no_urls": "No URLs were found in the message.",
        "history_title": "📋 Scan History",
        "no_history": "No scans have been recorded in this session.",
        "download": "📥 Download History as CSV",
        "clear_history": "🗑️ Clear Scan History",
        "about_title": "ℹ️ About the System",
        "about_text": (
            "This experimental system combines rule-based fraud indicators "
            "with a machine-learning model for message classification."
        ),
        "limitations": "Important Notes",
        "limitation_text": (
            "The system may misclassify some messages. A result is not "
            "definitive proof that a message is safe or fraudulent. "
            "Do not enter passwords, verification codes, or sensitive "
            "personal information while testing."
        ),
        "date": "Date and Time",
        "message": "Message",
        "result": "Classification",
        "risk_level": "Risk Level",
        "url_safe": "No obvious risk indicators were found by these basic URL checks.",
        "url_suspicious": "The URL contains an indicator that deserves caution.",
        "url_ip": "The URL uses an IP address instead of a typical domain name.",
        "url_short": "The URL uses a URL-shortening service.",
        "url_http": "The URL does not use HTTPS.",
        "url_punycode": "The domain contains Punycode encoding.",
        "url_invalid": "The URL could not be parsed correctly.",
        "rule_urgent": "The message urges immediate action or threatens account closure.",
        "rule_secret": "The message requests a password, verification code, or secret information.",
        "rule_money": "The message requests an upfront fee or money transfer.",
        "rule_prize": "The message mentions an unexpected prize, compensation, or money.",
        "rule_personal": "The message requests sensitive personal or financial information.",
        "rule_profit": "The message promises guaranteed profits or unrealistic returns.",
        "rule_authority": "The message invokes an official or international organization in a context that needs verification.",
        "footer": "Message Safety Detector | Experimental digital-awareness tool",
        "model_suspicious": "The machine-learning model classified the message as suspicious.",
        "model_safe": "The machine-learning model did not classify the message as suspicious.",
        "probability": "Model score for suspicious messages",
        "notice": "Notice: Do not rely on this system alone for security decisions.",
        "history_saved": "The scan result was recorded in the session history.",
        "history_cleared": "Scan history cleared.",
        "export_empty": "There is no data to download.",
    }
}


def tr(key):
    return TEXT[st.session_state.language][key]


# =========================================================
# 4. TRAINING DATA
# =========================================================

SAFE_MESSAGES = [
    "Hello, how are you today?",
    "Thank you for your help yesterday.",
    "The meeting is scheduled for tomorrow morning.",
    "Your assignment has been received.",
    "Please find the document attached.",
    "I will call you when I arrive.",
    "Your appointment is confirmed.",
    "The lecture starts at ten o'clock.",
    "Can you send me the class notes?",
    "Your order has been shipped.",
    "مرحبا كيف حالك اليوم؟",
    "شكرا ليك على المساعدة.",
    "المحاضرة بكرة الساعة عشرة صباحا.",
    "تم استلام الواجب بنجاح.",
    "سوف أتصل بك عندما أصل.",
    "موعد الاجتماع غدا صباحا.",
    "هل يمكنك إرسال ملخص المحاضرة؟",
    "تم تأكيد موعدك.",
    "سأرسل الملف بعد قليل.",
    "أتمنى لك يومًا سعيدًا.",
    "تم استلام طلبك بنجاح.",
    "سنلتقي في الجامعة غدا.",
    "أرسل لي جدول المحاضرات من فضلك.",
    "وصلت إلى المنزل بسلام.",
    "شكرا على رسالتك."
]

SUSPICIOUS_MESSAGES = [
    "Congratulations you won a prize click here to claim now",
    "Send your password and verification code to unlock your account",
    "Pay an upfront processing fee to receive your compensation",
    "Urgent your account will be closed verify your details immediately",
    "You are guaranteed huge profits invest now",
    "Send your bank details to receive your reward",
    "You have been selected to receive free money pay delivery fees",
    "Your account is blocked click this link immediately",
    "Claim your lottery prize by paying a small fee today",
    "Provide your OTP to prevent account suspension",
    "مبروك ربحت جائزة اضغط هنا لاستلام المبلغ الآن",
    "أرسل كلمة المرور ورمز التحقق لتفعيل حسابك",
    "ادفع رسوم المعالجة مقدما لاستلام التعويض المالي",
    "عاجل سيتم إغلاق حسابك إذا لم تؤكد بياناتك الآن",
    "أرباح مضمونة ومضاعفة استثمر أموالك اليوم",
    "أرسل بيانات حسابك البنكي لاستلام الجائزة",
    "تم اختيارك للحصول على أموال مجانية ادفع رسوم التوصيل",
    "حسابك محظور اضغط على الرابط لاستعادته فورا",
    "ارسل رمز OTP حتى لا يتم إيقاف حسابك",
    "تعويض مالي كبير ينتظرك أرسل بياناتك الشخصية",
    "فرصة استثمار مضمونة لا يمكن أن تخسر أموالك",
    "لقد ربحت اليانصيب أرسل رسوم الشحن لاستلام جائزتك",
    "أرسل رقم البطاقة البنكية ورمز التحقق الآن",
    "اربح آلاف الدولارات خلال يوم واحد بدون أي مخاطرة",
    "آخر فرصة لاستلام المنحة أرسل بياناتك السرية"
]


# =========================================================
# 5. MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def train_model():
    messages = SAFE_MESSAGES + SUSPICIOUS_MESSAGES

    labels = (
        ["Safe"] * len(SAFE_MESSAGES)
        + ["Suspicious"] * len(SUSPICIOUS_MESSAGES)
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.3)
        )
    ])

    model.fit(messages, labels)

    return model


model = train_model()


# =========================================================
# 6. SUSPICIOUS MESSAGE RULES
# =========================================================

RULES = {
    "urgent": {
        "patterns": [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bact now\b",
            r"\blast chance\b",
            r"\baccount will be closed\b",
            r"\baccount is blocked\b",
            r"\bverify now\b",
            r"عاجل",
            r"فورا",
            r"فوراً",
            r"حالاً",
            r"الآن",
            r"اغلاق حسابك",
            r"إغلاق حسابك",
            r"إيقاف حسابك",
            r"آخر فرصة",
            r"اكد بياناتك",
            r"أكد بياناتك"
        ],
        "weight": 15,
        "translation": "rule_urgent"
    },

    "secret": {
        "patterns": [
            r"\bpassword\b",
            r"\bpasscode\b",
            r"\botp\b",
            r"\bverification code\b",
            r"\bsecurity code\b",
            r"\bpin code\b",
            r"كلمة المرور",
            r"رمز التحقق",
            r"رمز التأكيد",
            r"الرقم السري",
            r"الرقم السرى",
            r"كود التحقق",
            r"كود التأكيد"
        ],
        "weight": 25,
        "translation": "rule_secret"
    },

    "money": {
        "patterns": [
            r"\bprocessing fee\b",
            r"\badvance fee\b",
            r"\bpay a fee\b",
            r"\btransfer money\b",
            r"\bwire transfer\b",
            r"\bdelivery fee\b",
            r"رسوم المعالجة",
            r"رسوم مقدما",
            r"رسوم مقدماً",
            r"ادفع رسوم",
            r"دفع رسوم",
            r"تحويل أموال",
            r"حول مبلغ",
            r"حوّل مبلغ",
            r"رسوم الشحن",
            r"رسوم التوصيل"
        ],
        "weight": 25,
        "translation": "rule_money"
    },

    "prize": {
        "patterns": [
            r"\byou won\b",
            r"\byou have won\b",
            r"\bclaim your prize\b",
            r"\bfree money\b",
            r"\blottery\b",
            r"\breward\b",
            r"\bcompensation\b",
            r"\bgrant money\b",
            r"ربحت",
            r"جائزة",
            r"تعويض مالي",
            r"تعويض",
            r"منحة مالية",
            r"أموال مجانية",
            r"مبلغ مالي",
            r"مبروك",
            r"اليانصيب"
        ],
        "weight": 15,
        "translation": "rule_prize"
    },

    "personal": {
        "patterns": [
            r"\bbank details\b",
            r"\bcard number\b",
            r"\bpersonal details\b",
            r"\bidentity number\b",
            r"\bsend your details\b",
            r"\baccount information\b",
            r"بياناتك الشخصية",
            r"بيانات الحساب",
            r"بياناتك البنكية",
            r"رقم البطاقة",
            r"رقم الحساب البنكي",
            r"معلوماتك الشخصية",
            r"أرسل بياناتك",
            r"ارسل بياناتك",
            r"بياناتك السرية"
        ],
        "weight": 20,
        "translation": "rule_personal"
    },

    "profit": {
        "patterns": [
            r"\bguaranteed profit\b",
            r"\bguaranteed returns\b",
            r"\bno risk\b",
            r"\bdouble your money\b",
            r"\bhuge profits\b",
            r"\bget rich quick\b",
            r"أرباح مضمونة",
            r"ربح مضمون",
            r"بدون مخاطر",
            r"بدون أي مخاطرة",
            r"ضاعف أموالك",
            r"مضاعفة أموالك",
            r"استثمار مضمون",
            r"ثراء سريع"
        ],
        "weight": 20,
        "translation": "rule_profit"
    },

    "authority": {
        "patterns": [
            r"\bworld bank\b",
            r"\bunited nations\b",
            r"\bun\b",
            r"\bgovernment compensation\b",
            r"\bofficial grant\b",
            r"البنك الدولي",
            r"الأمم المتحدة",
            r"تعويض حكومي",
            r"منحة حكومية",
            r"منظمة دولية"
        ],
        "weight": 5,
        "translation": "rule_authority"
    }
}


# =========================================================
# 7. URL INSPECTION
# =========================================================

URL_PATTERN = re.compile(
    r"""(?i)\b(?:https?://|www\.)[^\s<>"']+"""
)

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
    "rb.gy",
    "ow.ly",
    "shorturl.at",
    "buff.ly"
}


def inspect_url(raw_url):
    language = st.session_state.language

    url = raw_url.rstrip(".,!?;:)]}")

    if url.lower().startswith("www."):
        url = "http://" + url

    try:
        parsed = urlparse(url)

        hostname = parsed.hostname

        if not hostname:
            return [tr("url_invalid")]

        findings = []

        try:
            ipaddress.ip_address(hostname)
            findings.append(tr("url_ip"))
        except ValueError:
            pass

        if hostname.lower() in SHORTENERS:
            findings.append(tr("url_short"))

        if parsed.scheme.lower() != "https":
            findings.append(tr("url_http"))

        if "xn--" in hostname.lower():
            findings.append(tr("url_punycode"))

        if not findings:
            findings.append(tr("url_safe"))

        return findings

    except Exception:
        return [tr("url_invalid")]


# =========================================================
# 8. MESSAGE ANALYSIS
# =========================================================

def analyze_message(message):
    message = message.strip()

    if not message:
        return None

    suspicious_probability = 0.0
    prediction = "Safe"

    try:
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)

        if "Suspicious" in classes:
            suspicious_index = classes.index("Suspicious")
            suspicious_probability = float(
                probabilities[suspicious_index]
            )

        prediction = model.predict([message])[0]

    except Exception:
        suspicious_probability = 0.0
        prediction = "Safe"

    findings = []
    score = 0

    # Apply rule-based checks.
    for rule in RULES.values():
        matched = any(
            re.search(pattern, message, re.IGNORECASE)
            for pattern in rule["patterns"]
        )

        if matched:
            findings.append(tr(rule["translation"]))
            score += rule["weight"]

    # Inspect URLs.
    urls = URL_PATTERN.findall(message)
    url_results = []

    for url in urls:
        results = inspect_url(url)

        url_results.append({
            "url": url,
            "findings": results
        })

        for finding in results:
            if finding == tr("url_ip"):
                score += 20
            elif finding == tr("url_short"):
                score += 10
            elif finding == tr("url_http"):
                score += 10
            elif finding == tr("url_punycode"):
                score += 20

    # Include a limited contribution from the ML model.
    if prediction == "Suspicious" and suspicious_probability >= 0.80:
        score += 10
    elif prediction == "Suspicious" and suspicious_probability >= 0.60:
        score += 5

    score = min(score, 100)

    if score >= 60:
        risk_level = "high"
        classification = "Suspicious"
    elif score >= 30:
        risk_level = "medium"
        classification = "Suspicious"
    else:
        risk_level = "low"

        # Avoid calling every message safe based on a low rule score alone.
        if prediction == "Suspicious" and suspicious_probability >= 0.70:
            classification = "Suspicious"
        else:
            classification = "Safe"

    return {
        "message": message,
        "classification": classification,
        "score": score,
        "risk_level": risk_level,
        "findings": findings,
        "urls": url_results,
        "model_probability": suspicious_probability,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# =========================================================
# 9. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }

    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        opacity: 0.8;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 2rem;
        padding: 1rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 2.7rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 10. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.title("🛡️ Message Safety")

    language_choice = st.radio(
        tr("language"),
        options=["العربية", "English"],
        index=0 if st.session_state.language == "ar" else 1
    )

    new_language = "ar" if language_choice == "العربية" else "en"

    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.rerun()

    st.divider()

    st.subheader(tr("sidebar_title"))

    page = st.radio(
        "Navigation",
        options=[
            tr("scanner"),
            tr("history"),
            tr("about")
        ],
        label_visibility="collapsed"
    )


# =========================================================
# 11. PAGE HEADER
# =========================================================

st.markdown(
    f'<div class="main-title">{tr("title")}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">{tr("subtitle")}</div>',
    unsafe_allow_html=True
)


# =========================================================
# 12. SCANNER PAGE
# =========================================================

if page == tr("scanner"):

    st.subheader(tr("enter_message"))

    message = st.text_area(
        tr("placeholder"),
        height=200,
        key="message_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        scan_clicked = st.button(
            tr("scan"),
            use_container_width=True
        )

    with col2:
        st.button(
            tr("clear"),
            use_container_width=True,
            on_click=clear_message
        )

    if scan_clicked:
        if not message.strip():
            st.warning(tr("empty"))

        else:
            result = analyze_message(message)

            st.session_state.last_result = result

            st.session_state.history.insert(
                0,
                {
                    "Date and Time": result["timestamp"],
                    "Message": result["message"],
                    "Classification": result["classification"],
                    "Risk Score": result["score"],
                    "Risk Level": result["risk_level"].title()
                }
            )

            st.success(tr("history_saved"))

    result = st.session_state.last_result

    if result is not None:

        st.divider()
        st.subheader(tr("classification"))

        if result["classification"] == "Suspicious":
            st.error("⚠️ " + tr("suspicious"))
        else:
            st.success("✅ " + tr("safe"))

        if result["risk_level"] == "high":
            level_text = tr("high")
        elif result["risk_level"] == "medium":
            level_text = tr("medium")
        else:
            level_text = tr("low")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                tr("score"),
                f'{result["score"]}/100'
            )

        with col2:
            st.metric(
                tr("risk"),
                level_text
            )

        st.progress(result["score"] / 100)

        st.subheader(tr("reasons"))

        if result["findings"]:
            for finding in result["findings"]:
                st.warning("• " + finding)
        else:
            st.info(tr("no_reasons"))

        st.subheader(tr("url_details"))

        if result["urls"]:
            for url_item in result["urls"]:
                st.write("🔗", url_item["url"])

                for finding in url_item["findings"]:
                    st.write("•", finding)
        else:
            st.info(tr("no_urls"))

        st.caption(
            f'{tr("probability")}: '
            f'{result["model_probability"] * 100:.1f}%'
        )

        st.info(tr("notice"))


# =========================================================
# 13. HISTORY PAGE
# =========================================================

elif page == tr("history"):

    st.subheader(tr("history_title"))

    if not st.session_state.history:
        st.info(tr("no_history"))

    else:
        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label=tr("download"),
            data=csv_data,
            file_name="message_scan_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            tr("clear_history"),
            use_container_width=True
        ):
            st.session_state.history = []
            st.rerun()


# =========================================================
# 14. ABOUT PAGE
# =========================================================

elif page == tr("about"):

    st.subheader(tr("about_title"))

    st.write(tr("about_text"))

    st.markdown("### " + tr("limitations"))

    st.warning(tr("limitation_text"))

    st.markdown("### How it works")

    if st.session_state.language == "ar":
        st.markdown(
            """
            1. **تحليل النص:** يفحص الرسالة بحثًا عن عبارات قد ترتبط بالاحتيال.
            2. **تحليل الروابط:** يتحقق من بعض العلامات التي تستحق الحذر.
            3. **التعلم الآلي:** يستخدم نموذجًا تجريبيًا لتصنيف الرسالة.
            4. **درجة الخطورة:** يجمع مؤشرات الفحص في درجة من 0 إلى 100.
            5. **سجل الفحوصات:** يعرض النتائج التي تم تسجيلها خلال الجلسة الحالية.
            """
        )
    else:
        st.markdown(
            """
            1. **Text analysis:** Checks for phrases that may be associated with fraud.
            2. **URL analysis:** Looks for basic indicators that deserve caution.
            3. **Machine learning:** Uses an experimental model to classify the message.
            4. **Risk score:** Combines indicators into a score from 0 to 100.
            5. **Scan history:** Displays results recorded during the current session.
            """
        )

    st.caption(tr("notice"))


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    f'<div class="footer">{tr("footer")}</div>',
    unsafe_allow_html=True
)
