import re
import ipaddress
from urllib.parse import urlparse
from datetime import datetime
import os

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Message & Phishing Shield",
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
    st.session_state.message_input = ""
    st.session_state.last_result = None


# =========================================================
# 3. TRANSLATIONS (AR / EN)
# =========================================================

TEXT = {
    "ar": {
        "title": "🛡️ الدرع الذكي لكشف الاحتيال والرسائل الخطرة",
        "subtitle": "نظام سيبراني متقدم مدعوم بالذكاء الاصطناعي لتحليل وحماية الرسائل والروابط.",
        "language": "اللغة / Language",
        "sidebar_title": "لوحة التحكم",
        "scanner": "🔍 فحص رسالة ذكي",
        "history": "📋 سجل الفحوصات الأمنية",
        "about": "ℹ️ حول النظام المتقدم",
        "enter_message": "أدخل أو الصق النص المراد فحص أمانه:",
        "placeholder": "مثال: تم حظر حسابك، اضغط على الرابط لتأكيد بياناتك وبطاقتك...",
        "scan": "🚀 بدء الفحص الشامل",
        "clear": "🧹 مسح النص",
        "empty": "الرجاء إدخال نص سليم لإجراء الفحص.",
        "safe": "الرسالة آمنة ومنخفضة المؤشرات المشبوهة",
        "suspicious": "تحذير: رسالة احتيالية أو مشبوهة!",
        "low": "منخفض",
        "medium": "متوسط",
        "high": "عالي الخطورة",
        "risk": "مستوى التهديد",
        "score": "درجة الخطورة الأمنية",
        "classification": "نتيجة التقييم النهائي",
        "reasons": "مؤشرات وقواعد الفحص المكتشفة",
        "no_reasons": "لم يتم رصد كلمات مفتاحية مشبوهة بناءً على القواعد الحالية.",
        "url_details": "فحص وتحليل الروابط المرفقة",
        "no_urls": "لا توجد روابط أو مواقع إلكترونية ضمن النص.",
        "history_title": "📋 سجل العمليات والفحوصات",
        "no_history": "لا توجد عمليات فحص مسجلة في هذه الجلسة حتى الآن.",
        "download": "📥 تصدير السجل بصيغة CSV",
        "clear_history": "🗑️ تفريغ السجل بالكامل",
        "about_title": "ℹ️ حول المنظومة الأمنية",
        "about_text": "نظام ذكي يدمج بين خوارزميات تعلم الآلة وتحليل الروابط المتقدم (Heuristics & NLP) لرصد محاولات التصيد والرسائل الاحتيالية بدقة عالية.",
        "limitations": "تنبيه قانوني وأمني",
        "limitation_text": "هذا النظام أداة مساعدة للتوعية الرقمية، ولا يعتبر ضمانة مطلقة. تجنب تماماً مشاركة كلمات المرور، الرموز السرية (OTP)، أو البيانات البنكية.",
        "date": "التاريخ والوقت",
        "message": "النص المفحوص",
        "result": "التصنيف",
        "risk_level": "مستوى الخطورة",
        "url_safe": "الرابط يبدو طبيعياً ولا توجد مؤشرات واضحة للخطورة.",
        "url_suspicious": "الرابط يحتوي على نمط يستدعي الحذر الشديد.",
        "url_ip": "الرابط يستعمل عنوان IP مباشر بدلاً من نطاق رسمي موثوق.",
        "url_short": "الرابط يعتمد على خدمات اختصار الروابط (تخفي لهوية الوجهة).",
        "url_http": "الرابط غير آمن ولا يدعم بروتوكول التشفير HTTPS.",
        "url_punycode": "الرابط يستخدم ترميز Punycode (غالباً لمحاكاة نطاقات شهيرة).",
        "url_invalid": "تعذر تحليل هيكل الرابط بصورة سليمة.",
        "rule_urgent": "رصد عبارات تضغط للتعجل الشديد أو التهديد بالإغلاق.",
        "rule_secret": "المحتوى يطلب بيانات سرية، كلمات مرور، أو رموز تحقق (OTP).",
        "rule_money": "المحتوى يطلب تحويل أموال أو دفع رسوم مقدماً.",
        "rule_prize": "وعود زائفة بجوائز ضخمة أو تعويضات مالية غير منطقية.",
        "rule_personal": "طلب مكشوف لبيانات الهوية أو تفاصيل البطاقات البنكية.",
        "rule_profit": "وعود بأرباح استثمارية خيالية ومضمونة بدون مخاطر.",
        "rule_authority": "انتحال صفة جهات رسمية أو حكومية أو دولية.",
        "footer": "Advanced Message Safety & Phishing Shield | الإصدار الاحترافي",
        "model_suspicious": "نموذج الذكاء الاصطناعي صنف المحتوى كاحتيالي.",
        "model_safe": "نموذج الذكاء الاصطناعي لم يرصد أنماط احتيال واضحة.",
        "probability": "مستوى ثقة نموذج الذكاء الاصطناعي",
        "notice": "تنويه: حافظ دائماً على سريَّة معلوماتك المصرفية ولا تستجب للرسائل المجهولة.",
        "history_saved": "تم تسجيل نتيجة الفحص بنجاح في سجل الجلسة.",
        "history_cleared": "تم مسح السجل بنجاح.",
    },
    "en": {
        "title": "🛡️ AI Phishing & Message Safety Shield",
        "subtitle": "Advanced cybersecurity system powered by AI to inspect messages and URLs.",
        "language": "Language / اللغة",
        "sidebar_title": "Control Panel",
        "scanner": "🔍 Smart Message Scanner",
        "history": "📋 Security Scan History",
        "about": "ℹ️ About the System",
        "enter_message": "Enter or paste text to inspect:",
        "placeholder": "Example: Your account is blocked, click the link to verify your card...",
        "scan": "🚀 Run Comprehensive Scan",
        "clear": "🧹 Clear Text",
        "empty": "Please enter valid text to scan.",
        "safe": "Message is safe with low suspicious indicators",
        "suspicious": "Warning: Fraudulent or Suspicious Message!",
        "low": "Low",
        "medium": "Medium",
        "high": "High Risk",
        "risk": "Threat Level",
        "score": "Security Risk Score",
        "classification": "Final Classification",
        "reasons": "Detected Indicators & Rules",
        "no_reasons": "No suspicious keywords detected by current rules.",
        "url_details": "URL Inspection Results",
        "no_urls": "No URLs found in the text.",
        "history_title": "📋 Scan History Logs",
        "no_history": "No scans recorded in this session yet.",
        "download": "📥 Export History as CSV",
        "clear_history": "🗑️ Clear History",
        "about_title": "ℹ️ About the Security Shield",
        "about_text": "An intelligent system combining machine learning and heuristic URL analysis to detect phishing and fraud attempts accurately.",
        "limitations": "Disclaimer",
        "limitation_text": "This system is a digital awareness tool. Never share passwords, OTP codes, or banking details.",
        "date": "Date & Time",
        "message": "Message",
        "result": "Classification",
        "risk_level": "Risk Level",
        "url_safe": "URL looks normal with no obvious risk indicators.",
        "url_suspicious": "URL contains indicators requiring caution.",
        "url_ip": "URL uses a raw IP address instead of a domain.",
        "url_short": "URL uses a shortening service (masks true destination).",
        "url_http": "URL does not use secure HTTPS protocol.",
        "url_punycode": "URL contains Punycode encoding (often used in spoofing).",
        "url_invalid": "URL could not be parsed properly.",
        "rule_urgent": "Contains urgent action phrases or account closure threats.",
        "rule_secret": "Requests passwords, secrets, or OTP verification codes.",
        "rule_money": "Requests upfront fees or money transfers.",
        "rule_prize": "Mentions unrealistic prizes, lotteries, or rewards.",
        "rule_personal": "Requests personal data or banking information.",
        "rule_profit": "Promises guaranteed unrealistic investment returns.",
        "rule_authority": "Invokes official or international organizations.",
        "footer": "Advanced Message Safety Shield | Professional Edition",
        "model_suspicious": "AI model classified the text as suspicious.",
        "model_safe": "AI model did not classify the text as suspicious.",
        "probability": "AI Model Confidence Score",
        "notice": "Notice: Keep your financial information secure and ignore unknown requests.",
        "history_saved": "Scan result recorded in session history.",
        "history_cleared": "History cleared.",
    }
}


def tr(key):
    return TEXT[st.session_state.language][key]


# =========================================================
# 4. TRAINING DATA
# =========================================================

SAFE_MESSAGES = [
    "Hello, how are you today?", "Thank you for your help yesterday.",
    "The meeting is scheduled for tomorrow morning.", "Your assignment has been received.",
    "Please find the document attached.", "I will call you when I arrive.",
    "Your appointment is confirmed.", "The lecture starts at ten o'clock.",
    "Can you send me the class notes?", "Your order has been shipped.",
    "مرحبا كيف حالك اليوم؟", "شكرا ليك على المساعدة.",
    "المحاضرة بكرة الساعة عشرة صباحا.", "تم استلام الواجب بنجاح.",
    "سوف أتصل بك عندما أصل.", "موعد الاجتماع غدا صباحا.",
    "هل يمكنك إرسال ملخص المحاضرة؟", "تم تأكيد موعدك.",
    "سأرسل الملف بعد قليل.", "أتمنى لك يومًا سعيدًا.",
    "تم استلام طلبك بنجاح.", "سنلتقي في الجامعة غدا.",
    "أرسل لي جدول المحاضرات من فضلك.", "وصلت إلى المنزل بسلام.",
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
# 5. MACHINE LEARNING MODEL & CACHING
# =========================================================

MODEL_PATH = "phishing_model.joblib"

@st.cache_resource
def get_trained_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    messages = SAFE_MESSAGES + SUSPICIOUS_MESSAGES
    labels = ["Safe"] * len(SAFE_MESSAGES) + ["Suspicious"] * len(SUSPICIOUS_MESSAGES)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=0.3))
    ])
    model.fit(messages, labels)
    
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass

    return model

model = get_trained_model()


# =========================================================
# 6. RULES CONFIGURATION
# =========================================================

RULES = {
    "urgent": {
        "patterns": [r"\burgent\b", r"\bimmediately\b", r"\bact now\b", r"\blast chance\b", r"\baccount will be closed\b", r"\baccount is blocked\b", r"\bverify now\b", r"عاجل", r"فورا", r"فوراً", r"حالاً", r"الآن", r"اغلاق حسابك", r"إغلاق حسابك", r"إيقاف حسابك", r"آخر فرصة", r"اكد بياناتك", r"أكد بياناتك"],
        "weight": 15,
        "translation": "rule_urgent"
    },
    "secret": {
        "patterns": [r"\bpassword\b", r"\bpasscode\b", r"\botp\b", r"\bverification code\b", r"\bsecurity code\b", r"\bpin code\b", r"كلمة المرور", r"رمز التحقق", r"رمز التأكيد", r"الرقم السري", r"الرقم السرى", r"كود التحقق", r"كود التأكيد"],
        "weight": 25,
        "translation": "rule_secret"
    },
    "money": {
        "patterns": [r"\bprocessing fee\b", r"\badvance fee\b", r"\bpay a fee\b", r"\btransfer money\b", r"\bwire transfer\b", r"\bdelivery fee\b", r"رسوم المعالجة", r"رسوم مقدما", r"رسوم مقدماً", r"ادفع رسوم", r"دفع رسوم", r"تحويل أموال", r"حول مبلغ", r"حوّل مبلغ", r"رسوم الشحن", r"رسوم التوصيل"],
        "weight": 25,
        "translation": "rule_money"
    },
    "prize": {
        "patterns": [r"\byou won\b", r"\byou have won\b", r"\bclaim your prize\b", r"\bfree money\b", r"\blottery\b", r"\breward\b", r"\bcompensation\b", r"\bgrant money\b", r"ربحت", r"جائزة", r"تعويض مالي", r"تعويض", r"منحة مالية", r"أموال مجانية", r"مبلغ مالي", r"مبروك", r"اليانصيب"],
        "weight": 15,
        "translation": "rule_prize"
    },
    "personal": {
        "patterns": [r"\bbank details\b", r"\bcard number\b", r"\bpersonal details\b", r"\bidentity number\b", r"\bsend your details\b", r"\baccount information\b", r"بياناتك الشخصية", r"بيانات الحساب", r"بياناتك البنكية", r"رقم البطاقة", r"رقم الحساب البنكي", r"معلوماتك الشخصية", r"أرسل بياناتك", r"ارسل بياناتك", r"بياناتك السرية"],
        "weight": 20,
        "translation": "rule_personal"
    },
    "profit": {
        "patterns": [r"\bguaranteed profit\b", r"\bguaranteed returns\b", r"\bno risk\b", r"\bdouble your money\b", r"\bhuge profits\b", r"\bget rich quick\b", r"أرباح مضمونة", r"ربح مضمون", r"بدون مخاطر", r"بدون أي مخاطرة", r"ضاعف أموالك", r"مضاعفة أموالك", r"استثمار مضمون", r"ثراء سريع"],
        "weight": 20,
        "translation": "rule_profit"
    },
    "authority": {
        "patterns": [r"\bworld bank\b", r"\bunited nations\b", r"\bun\b", r"\bgovernment compensation\b", r"\bofficial grant\b", r"البنك الدولي", r"الأمم المتحدة", r"تعويض حكومي", r"منحة حكومية", r"منظمة دولية"],
        "weight": 5,
        "translation": "rule_authority"
    }
}


# =========================================================
# 7. URL INSPECTION
# =========================================================

URL_PATTERN = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"']+""")

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at", "buff.ly"
}


def inspect_url(raw_url):
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
# 8. ANALYSIS ENGINE
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
            suspicious_probability = float(probabilities[suspicious_index])
        prediction = model.predict([message])[0]
    except Exception:
        suspicious_probability = 0.0
        prediction = "Safe"

    findings = []
    score = 0

    for rule in RULES.values():
        matched = any(re.search(pattern, message, re.IGNORECASE) for pattern in rule["patterns"])
        if matched:
            findings.append(tr(rule["translation"]))
            score += rule["weight"]

    urls = URL_PATTERN.findall(message)
    url_results = []
    for url in urls:
        results = inspect_url(url)
        url_results.append({"url": url, "findings": results})
        for finding in results:
            if finding == tr("url_ip"):
                score += 20
            elif finding == tr("url_short"):
                score += 10
            elif finding == tr("url_http"):
                score += 10
            elif finding == tr("url_punycode"):
                score += 20

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
# 9. PROFESSIONAL STYLING (UI/UX)
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #ff4b4b, #ffa41b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        opacity: 0.85;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        transition: 0.3s;
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(128,128,128,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 10. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.title("🛡️ Cyber Shield")
    
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
        options=[tr("scanner"), tr("history"), tr("about")],
        label_visibility="collapsed"
    )


# =========================================================
# 11. PAGE HEADER
# =========================================================

st.markdown(f'<div class="main-title">{tr("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{tr("subtitle")}</div>', unsafe_allow_html=True)


# =========================================================
# 12. SCANNER PAGE
# =========================================================

if page == tr("scanner"):
    st.subheader(tr("enter_message"))
    message = st.text_area(tr("placeholder"), height=180, key="message_input")

    col1, col2 = st.columns(2)
    with col1:
        scan_clicked = st.button(tr("scan"), use_container_width=True)
    with col2:
        st.button(tr("clear"), use_container_width=True, on_click=clear_message)

    if scan_clicked:
        if not message.strip():
            st.warning(tr("empty"))
        else:
            result = analyze_message(message)
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
                "Date and Time": result["timestamp"],
                "Message": result["message"],
                "Classification": result["classification"],
                "Risk Score": result["score"],
                "Risk Level": result["risk_level"].title()
            })
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
            st.metric(tr("score"), f'{result["score"]}/100')
        with col2:
            st.metric(tr("risk"), level_text)

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

        st.caption(f'{tr("probability")}: {result["model_probability"] * 100:.1f}%')
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
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=tr("download"),
            data=csv_data,
            file_name="security_scan_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(tr("clear_history"), use_container_width=True):
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
    st.caption(tr("notice"))


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")
st.markdown(f'<div class="footer">{tr("footer")}</div>', unsafe_allow_html=True)
