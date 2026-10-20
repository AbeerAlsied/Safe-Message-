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
# 1. PAGE CONFIGURATION & THEME
# =========================================================

st.set_page_config(
    page_title="CyberShield AI - Advanced Phishing & Fraud Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. SESSION STATE MANAGEMENT
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
# 3. ADVANCED LOCALIZED TEXTS (AR / EN)
# =========================================================

TEXT = {
    "ar": {
        "title": "🛡️ CyberShield AI: المنظومة الذكية لكشف الاحتيال والهندسة الاجتماعية",
        "subtitle": "تحليل أمني سيبراني متطور يجمع بين خوارزميات التعلم الآلي العميق وفحص الروابط الاستباقي.",
        "language": "اللغة / Language",
        "sidebar_title": "لوحة القيادة الأمنية",
        "scanner": "🔍 الفحص والتحليل الذكي",
        "history": "📋 السجل والأرشيف السيبراني",
        "about": "ℹ️ حول المنظومة التقنية",
        "enter_message": "أدخل النص أو الرسالة المراد فحصها أمنياً:",
        "placeholder": "ألصق النص هنا (رسائل واتساب، بريد إلكتروني، روابط مشبوهة)...",
        "scan": "🚀 إطلاق الفحص السيبراني المعمق",
        "clear": "🧹 مسح الحقل",
        "empty": "الرجاء إدخال نص صالح للتحليل.",
        "safe": "المحتوى يبدو سليماً ولا توجد مؤشرات خطر ظاهرة",
        "suspicious": "تحذير أمني خطير: المحتوى يظهر مؤشرات احتيال وتصيد!",
        "low": "منخفض الأهمية",
        "medium": "متوسط الخطورة",
        "high": "حرج / عالي الخطورة جداً",
        "risk": "مستوى التهديد الأمني",
        "score": "معامل التهديد الرقمي",
        "classification": "التقييم السيبراني النهائي",
        "reasons": "مؤشرات الهندسة الاجتماعية والقواعد المرصودة",
        "no_reasons": "لم يتم رصد أي عبارات أو أنماط احتيالية معروفة.",
        "url_details": "فحص وبصمة الروابط الرقمية",
        "no_urls": "لا توجد روابط خارجية مرفقة في النص.",
        "history_title": "📋 الأرشيف السيبراني للجلسة الحالية",
        "no_history": "لا توجد عمليات فحص مسجلة حتى الآن.",
        "download": "📥 تصدير التقرير والنتائج (CSV)",
        "clear_history": "🗑️ تفريغ الأرشيف بالكامل",
        "about_title": "ℹ️ الهندسة المعمارية للنظام",
        "about_text": "يعتمد النظام على دمج نماذج التصنيف الاحتمالي (Multinomial Naive Bayes مع تحليل الحروف النحوية N-grams) مع محرك قواعد استدلالي يرصد أساليب التلاعب النفسي والهندسة الاجتماعية في الرسائل.",
        "limitations": "إخلاء المسؤولية والتوجيهات الأمنية",
        "limitation_text": "هذه المنظومة أداة تحليل استباقية مساعدة. احرص دائماً على عدم مشاركة بياناتك الحساسة، كلمات المرور، أو رموز المصادقة الثنائية (OTP) مهما كانت الظروف.",
        "date": "التاريخ والوقت",
        "message": "النص المفحوص",
        "result": "التصنيف",
        "risk_level": "مستوى الخطورة",
        "url_safe": "الرابط يخلو من مؤشرات الاحتيال المباشرة.",
        "url_suspicious": "الرابط يحمل أنماطاً تتطلب الحذر الشديد.",
        "url_ip": "الرابط يستخدم عنوان IP مباشر (أسلوب شائع في إخفاء الهوية).",
        "url_short": "الرابط يستعمل خدمات الاختصار (تُستخدم لإخفاء الوجهة الحقيقية).",
        "url_http": "الرابط غير مشفر ويستخدم بروتوكول HTTP المكشوف.",
        "url_punycode": "الرابط يحتوي على رموز Punycode (محاكاة نطاقات شهيرة/Spoofing).",
        "url_invalid": "هيكل الرابط غير صالح أو تالف.",
        "rule_urgent": "استخدام أسلوب الضغط النفسي والتعجل المفرط أو التهديد بالإغلاق.",
        "rule_secret": "طلب معلومات سرية للغاية مثل كلمات المرور أو رموز التحقق.",
        "rule_money": "طلب مدفوعات مالية مسبقة أو رسوم شحن ومعالجة وهمية.",
        "rule_prize": "إغراءات بجوائز مالية ضخمة أو تعويضات وهمية غير منطقية.",
        "rule_personal": "طلب استباقي لبيانات الهوية أو تفاصيل الحسابات المصرفية.",
        "rule_profit": "وعود زائفة بأرباح استثمارية فلكية خالية من المخاطر.",
        "rule_authority": "انتحال صفة مؤسسات رسمية، بنوك، أو هيئات دولية.",
        "footer": "CyberShield AI Engine | نسخة الأمن الرقمي المتقدمة",
        "model_suspicious": "نموذج التعلم الآلي رجح أن النص ذو طابع احتيالي.",
        "model_safe": "نموذج التعلم الآلي اعتبر النص طبيعياً.",
        "probability": "مؤشر ثقة محرك الذكاء الاصطناعي",
        "notice": "تنبيه سيبراني: المؤسسات الرسمية لا تطلب أبداً الرموز السرية أو بيانات البطاقات عبر الرسائل.",
        "history_saved": "تم أرشفة نتيجة الفحص بنجاح.",
        "history_cleared": "تم تفريغ السجل بالكامل.",
    },
    "en": {
        "title": "🛡️ CyberShield AI: Advanced Phishing & Social Engineering Detector",
        "subtitle": "Cutting-edge cybersecurity analysis combining deep machine learning and heuristic URL inspection.",
        "language": "Language / اللغة",
        "sidebar_title": "Security Control Panel",
        "scanner": "🔍 Smart Scanning Engine",
        "history": "📋 Cyber History & Logs",
        "about": "ℹ️ System Architecture",
        "enter_message": "Enter text or message to inspect securely:",
        "placeholder": "Paste message here (WhatsApp, Email, suspicious links)...",
        "scan": "🚀 Launch Deep Cyber Scan",
        "clear": "🧹 Clear Input",
        "empty": "Please enter valid text for analysis.",
        "safe": "Content appears clean with no active threat indicators",
        "suspicious": "High Security Warning: Phishing or Fraud indicators detected!",
        "low": "Low Risk",
        "medium": "Medium Risk",
        "high": "Critical / High Risk",
        "risk": "Threat Level",
        "score": "Digital Threat Score",
        "classification": "Final Cyber Assessment",
        "reasons": "Detected Social Engineering Rules & Indicators",
        "no_reasons": "No suspicious keywords or patterns matched.",
        "url_details": "URL Fingerprint & Inspection",
        "no_urls": "No external URLs found in the text.",
        "history_title": "📋 Current Session Security Logs",
        "no_history": "No scan logs recorded yet.",
        "download": "📥 Export Security Report (CSV)",
        "clear_history": "🗑️ Clear All Logs",
        "about_title": "ℹ️ System Architecture",
        "about_text": "Combines probabilistic text classification (Multinomial Naive Bayes with character N-grams) with a heuristic inference rule engine targeting social engineering vectors.",
        "limitations": "Disclaimer & Security Guidelines",
        "limitation_text": "This system is a proactive analytical assistant. Never disclose sensitive data, passwords, or OTP codes under any circumstances.",
        "date": "Date & Time",
        "message": "Inspected Text",
        "result": "Classification",
        "risk_level": "Risk Level",
        "url_safe": "URL has no direct threat indicators.",
        "url_suspicious": "URL contains suspicious structural patterns.",
        "url_ip": "URL uses a raw IP address (common in obfuscation).",
        "url_short": "URL uses a shortening service (masks true destination).",
        "url_http": "URL is unencrypted and uses vulnerable HTTP protocol.",
        "url_punycode": "URL contains Punycode (potential domain spoofing).",
        "url_invalid": "URL structure is invalid or corrupted.",
        "rule_urgent": "Leverages urgency pressure tactics or account closure threats.",
        "rule_secret": "Requests critical secrets like passwords or OTP codes.",
        "rule_money": "Demands upfront money transfers or fake processing fees.",
        "rule_prize": "Unrealistic lottery or financial compensation baits.",
        "rule_personal": "Direct harvesting request for ID or banking credentials.",
        "rule_profit": "Fake guaranteed investment returns with zero risk.",
        "rule_authority": "Impersonation of official institutions or international bodies.",
        "footer": "CyberShield AI Engine | Advanced Digital Security Edition",
        "model_suspicious": "AI model classified text as malicious/fraudulent.",
        "model_safe": "AI model classified text as benign.",
        "probability": "AI Model Confidence Index",
        "notice": "Cyber Notice: Official entities never request passwords or card credentials via messages.",
        "history_saved": "Scan result archived successfully.",
        "history_cleared": "Logs cleared successfully.",
    }
}


def tr(key):
    return TEXT[st.session_state.language][key]


# =========================================================
# 4. COMPREHENSIVE TRAINING DATASET (EXPANDED)
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
    "شكرا على رسالتك.", "يرجى مراجعة الجدول الدراسي المرفق.",
    "تم تحديث النظام بنجاح، شكرا لتعاونكم."
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
    "آخر فرصة لاستلام المنحة أرسل بياناتك السرية",
    "تنبيه أمني: يرجى تحديث بياناتك البنكية عبر الرابط التالي تفادياً لإيقاف البطاقة"
]


# =========================================================
# 5. ML MODEL PIPELINE & CACHING PERSISTENCE
# =========================================================

MODEL_FILE = "cybershield_model.joblib"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    messages = SAFE_MESSAGES + SUSPICIOUS_MESSAGES
    labels = ["Safe"] * len(SAFE_MESSAGES) + ["Suspicious"] * len(SUSPICIOUS_MESSAGES)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=0.25))
    ])
    model.fit(messages, labels)

    try:
        joblib.dump(model, MODEL_FILE)
    except Exception:
        pass

    return model

model = load_or_train_model()


# =========================================================
# 6. HEURISTIC RULES ENGINE (SOCIAL ENGINEERING)
# =========================================================

RULES = {
    "urgent": {
        "patterns": [
            r"\burgent\b", r"\bimmediately\b", r"\bact now\b", r"\blast chance\b",
            r"\baccount will be closed\b", r"\baccount is blocked\b", r"\bverify now\b",
            r"عاجل", r"فورا", r"فوراً", r"حالاً", r"الآن", r"اغلاق حسابك",
            r"إغلاق حسابك", r"إيقاف حسابك", r"آخر فرصة", r"اكد بياناتك", r"أكد بياناتك", r"تنبيه أمني"
        ],
        "weight": 15,
        "translation": "rule_urgent"
    },
    "secret": {
        "patterns": [
            r"\bpassword\b", r"\bpasscode\b", r"\botp\b", r"\bverification code\b",
            r"\bsecurity code\b", r"\bpin code\b", r"كلمة المرور", r"رمز التحقق",
            r"رمز التأكيد", r"الرقم السري", r"الرقم السرى", r"كود التحقق", r"كود التأكيد"
        ],
        "weight": 25,
        "translation": "rule_secret"
    },
    "money": {
        "patterns": [
            r"\bprocessing fee\b", r"\badvance fee\b", r"\bpay a fee\b", r"\btransfer money\b",
            r"\bwire transfer\b", r"\bdelivery fee\b", r"رسوم المعالجة", r"رسوم مقدما",
            r"رسوم مقدماً", r"ادفع رسوم", r"دفع رسوم", r"تحويل أموال", r"حول مبلغ",
            r"حوّل مبلغ", r"رسوم الشحن", r"رسوم التوصيل"
        ],
        "weight": 25,
        "translation": "rule_money"
    },
    "prize": {
        "patterns": [
            r"\byou won\b", r"\byou have won\b", r"\bclaim your prize\b", r"\bfree money\b",
            r"\blottery\b", r"\breward\b", r"\bcompensation\b", r"\bgrant money\b",
            r"ربحت", r"جائزة", r"تعويض مالي", r"تعويض", r"منحة مالية", r"أموال مجانية",
            r"مبلغ مالي", r"مبروك", r"اليانصيب"
        ],
        "weight": 15,
        "translation": "rule_prize"
    },
    "personal": {
        "patterns": [
            r"\bbank details\b", r"\bcard number\b", r"\bpersonal details\b", r"\bidentity number\b",
            r"\bsend your details\b", r"\baccount information\b", r"بياناتك الشخصية",
            r"بيانات الحساب", r"بياناتك البنكية", r"رقم البطاقة", r"رقم الحساب البنكي",
            r"معلوماتك الشخصية", r"أرسل بياناتك", r"ارسل بياناتك", r"بياناتك السرية"
        ],
        "weight": 20,
        "translation": "rule_personal"
    },
    "profit": {
        "patterns": [
            r"\bguaranteed profit\b", r"\bguaranteed returns\b", r"\bno risk\b", r"\bdouble your money\b",
            r"\bhuge profits\b", r"\bget rich quick\b", r"أرباح مضمونة", r"ربح مضمون",
            r"بدون مخاطر", r"بدون أي مخاطرة", r"ضاعف أموالك", r"مضاعفة أموالك",
            r"استثمار مضمون", r"ثراء سريع"
        ],
        "weight": 20,
        "translation": "rule_profit"
    },
    "authority": {
        "patterns": [
            r"\bworld bank\b", r"\bunited nations\b", r"\bun\b", r"\bgovernment compensation\b",
            r"\bofficial grant\b", r"البنك الدولي", r"الأمم المتحدة", r"تعويض حكومي",
            r"منحة حكومية", r"منظمة دولية"
        ],
        "weight": 5,
        "translation": "rule_authority"
    }
}


# =========================================================
# 7. ADVANCED URL FINGERPRINTING & INSPECTION
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
# 8. CORE ANALYSIS ENGINE (INTEGRATED RISK MATRIX)
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
            idx = classes.index("Suspicious")
            suspicious_probability = float(probabilities[idx])
        prediction = model.predict([message])[0]
    except Exception:
        suspicious_probability = 0.0
        prediction = "Safe"

    findings = []
    score = 0

    # Rule checks
    for rule in RULES.values():
        matched = any(re.search(p, message, re.IGNORECASE) for p in rule["patterns"])
        if matched:
            findings.append(tr(rule["translation"]))
            score += rule["weight"]

    # URL checks
    urls = URL_PATTERN.findall(message)
    url_results = []
    for u in urls:
        res = inspect_url(u)
        url_results.append({"url": u, "findings": res})
        for f in res:
            if f == tr("url_ip"):
                score += 20
            elif f == tr("url_short"):
                score += 10
            elif f == tr("url_http"):
                score += 10
            elif f == tr("url_punycode"):
                score += 20

    # AI Model weight integration
    if prediction == "Suspicious" and suspicious_probability >= 0.75:
        score += 15
    elif prediction == "Suspicious" and suspicious_probability >= 0.50:
        score += 8

    score = min(score, 100)

    # Risk level classification matrix
    if score >= 55:
        risk_level = "high"
        classification = "Suspicious"
    elif score >= 25:
        risk_level = "medium"
        classification = "Suspicious"
    else:
        risk_level = "low"
        if prediction == "Suspicious" and suspicious_probability >= 0.65:
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
# 9. ADVANCED CYBERSTYLING & UI COMPONENTS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    .subtitle {
        text-align: center;
        opacity: 0.8;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 10. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.title("🛡️ CyberShield")
    
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
# 12. SCANNER PAGE INTERFACE
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
# 13. HISTORY & LOGS PAGE
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
            file_name="cybershield_audit_logs.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(tr("clear_history"), use_container_width=True):
            st.session_state.history = []
            st.rerun()


# =========================================================
# 14. ABOUT ARCHITECTURE PAGE
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
