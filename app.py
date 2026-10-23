import re
import ipaddress
from urllib.parse import urlparse
from datetime import datetime
from io import BytesIO

import pandas as pd
import qrcode
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

=====================================================

1. PAGE CONFIGURATION

=====================================================

APP_URL = "https://safetydetecto.streamlit.app/"

st.set_page_config(
page_title="CyberShield AI | Smart Security",
page_icon="🛡️",
layout="wide",
initial_sidebar_state="expanded"
)

=====================================================

2. SESSION STATE

=====================================================

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

=====================================================

3. TRANSLATIONS

=====================================================

TEXT = {
"ar": {
"brand": "CYBERSHIELD AI",
"tagline": "حماية رقمية أكثر ذكاءً",
"home": "الرئيسية",
"scanner": "مختبر الفحص",
"history": "سجل الفحوصات",
"about": "حول المشروع",
"hero_badge": "منظومة تحليل الرسائل الذكية",
"hero_title": "اكتشف الرسائل المشبوهة",
"hero_title2": "قبل أن تثق بها",
"hero_desc": "نظام تجريبي يستخدم التعلم الآلي وقواعد تحليلية لفحص الرسائل والروابط ورصد مؤشرات الاحتيال المحتملة.",
"start": "ابدأ الفحص الآن ↓",
"learn": "تعرّف على المشروع",
"feature1": "تحليل الرسائل",
"feature1_desc": "تصنيف أولي للرسائل باستخدام التعلم الآلي.",
"feature2": "فحص الروابط",
"feature2_desc": "رصد بعض الأنماط التي تستحق الحذر.",
"feature3": "مؤشرات الخطورة",
"feature3_desc": "توضيح العلامات التي رصدها النظام.",
"lab_title": "مختبر تحليل الرسائل",
"lab_desc": "أدخلي رسالة لتحليلها واكتشاف المؤشرات المشبوهة.",
"input_label": "الرسالة المراد فحصها",
"placeholder": "الصقي هنا رسالة بريد إلكتروني أو واتساب أو رابطاً...",
"scan": "🔍 تحليل الرسالة",
"clear": "مسح الحقل",
"empty": "الرجاء إدخال رسالة أولاً.",
"examples": "جرّبي مثالاً",
"example_safe": "رسالة عادية",
"example_suspicious": "رسالة مشبوهة",
"result_title": "نتيجة التحليل",
"safe": "لا توجد مؤشرات كافية لتصنيف الرسالة كمشبوهة",
"suspicious": "تم رصد مؤشرات تستدعي الحذر",
"score": "مؤشر الخطورة",
"risk": "مستوى الخطورة التقديري",
"low": "منخفض",
"medium": "متوسط",
"high": "مرتفع",
"reasons": "المؤشرات المكتشفة",
"no_reasons": "لم ترصد القواعد الحالية مؤشرات معروفة.",
"urls": "تحليل الروابط",
"no_urls": "لم يُعثر على روابط مطابقة لأنماط الفحص.",
"url_ip": "يستخدم عنوان IP مباشرة بدلاً من اسم نطاق.",
"url_short": "يستخدم خدمة اختصار روابط.",
"url_http": "يستخدم HTTP غير المشفر.",
"url_puny": "يحتوي اسم النطاق على Punycode.",
"url_ok": "لم تُكتشف الأنماط المحددة في قواعد فحص الرابط.",
"url_invalid": "تعذر تحليل بنية الرابط.",
"rule_urgent": "لغة استعجال أو تهديد بإيقاف الحساب.",
"rule_secret": "طلب كلمة مرور أو رمز تحقق سري.",
"rule_money": "طلب رسوم مسبقة أو دفعات مالية.",
"rule_prize": "وعد بجائزة أو تعويض أو أموال مجانية.",
"rule_personal": "طلب بيانات شخصية أو مصرفية.",
"rule_profit": "وعود بأرباح مضمونة أو استثمار بلا مخاطر.",
"rule_authority": "ذكر جهة رسمية في سياق قد يستدعي التحقق.",
"notice": "تنبيه: لا ترسلي كلمات المرور أو رموز OTP أو بيانات البطاقات لأي جهة عبر الرسائل.",
"history_title": "سجل الفحوصات",
"history_empty": "لم تُسجل أي فحوصات في هذه الجلسة بعد.",
"download": "📥 تنزيل السجل بصيغة CSV",
"clear_history": "مسح السجل",
"about_title": "عن CyberShield AI",
"about_desc": "مشروع توعوي وتجريبي يساعد على توضيح كيفية استخدام التعلم الآلي والقواعد التحليلية لرصد مؤشرات محتملة للتصيد والاحتيال الإلكتروني.",
"how_title": "كيف يعمل النظام؟",
"step1": "تحليل النص باستخدام نموذج تصنيف آلي.",
"step2": "فحص كلمات وأنماط مرتبطة بالاستعجال والجوائز والبيانات السرية.",
"step3": "فحص بعض خصائص الروابط الموجودة في الرسالة.",
"limitations": "حدود النظام",
"limitations_desc": "النموذج مدرّب على مجموعة أمثلة صغيرة. قد يخطئ في تصنيف الرسائل، ولا يضمن اكتشاف جميع عمليات الاحتيال. نتيجة الفحص ليست حكماً نهائياً على أمان الرسالة أو الرابط.",
"qr_title": "جرّبي المشروع من هاتفك",
"qr_desc": "امسحي رمز QR لفتح التطبيق.",
"footer": "CyberShield AI | مشروع توعوي للأمن الرقمي",
"confidence": "احتمال التصنيف الذي قدّره النموذج",
"date": "التاريخ والوقت",
"message": "الرسالة",
"classification": "التصنيف",
"risk_level": "مستوى الخطورة",
"ai_suspicious": "صُنفت الرسالة كمشبوهة بواسطة النموذج.",
"ai_safe": "صُنفت الرسالة كعادية بواسطة النموذج.",
"language": "اللغة",
"about_button": "اكتشف كيف يعمل النظام",
"sample_safe": "مرحباً، موعد الاجتماع غداً الساعة العاشرة. شكراً لمساعدتك.",
"sample_bad": "عاجل! مبروك ربحت جائزة مالية كبيرة. أرسل كلمة المرور ورمز التحقق وادفع رسوم المعالجة لاستلام المبلغ.",
"url_check": "ملاحظة: فحص الرابط يعتمد على أنماط محددة ولا يتحقق من محتوى الموقع الفعلي.",
},
"en": {
"brand": "CYBERSHIELD AI",
"tagline": "Smarter Digital Protection",
"home": "Home",
"scanner": "Message Scanner",
"history": "Scan History",
"about": "About",
"hero_badge": "INTELLIGENT MESSAGE ANALYSIS",
"hero_title": "Detect suspicious messages",
"hero_title2": "before you trust them",
"hero_desc": "An experimental tool combining machine learning and heuristic rules to identify potential fraud indicators in messages and links.",
"start": "Start Scanning ↓",
"learn": "Explore the Project",
"feature1": "Message Analysis",
"feature1_desc": "Preliminary message classification using machine learning.",
"feature2": "URL Inspection",
"feature2_desc": "Identify selected URL patterns that deserve caution.",
"feature3": "Risk Indicators",
"feature3_desc": "Understand which warning signs were detected.",
"lab_title": "Message Analysis Lab",
"lab_desc": "Enter a message to inspect it for suspicious indicators.",
"input_label": "Message to inspect",
"placeholder": "Paste an email, WhatsApp message, or URL here...",
"scan": "🔍 Analyze Message",
"clear": "Clear input",
"empty": "Please enter a message first.",
"examples": "Try an example",
"example_safe": "Normal message",
"example_suspicious": "Suspicious message",
"result_title": "Analysis Result",
"safe": "Not enough indicators were found to classify this message as suspicious",
"suspicious": "Potential warning indicators detected",
"score": "Risk Indicator",
"risk": "Estimated Risk Level",
"low": "Low",
"medium": "Medium",
"high": "High",
"reasons": "Detected Indicators",
"no_reasons": "No indicators matched the current rules.",
"urls": "URL Inspection",
"no_urls": "No URLs matching the inspection pattern were found.",
"url_ip": "Uses a raw IP address instead of a domain name.",
"url_short": "Uses a URL-shortening service.",
"url_http": "Uses unencrypted HTTP.",
"url_puny": "The hostname contains Punycode.",
"url_ok": "No patterns covered by the current URL rules were detected.",
"url_invalid": "Could not parse the URL structure.",
"rule_urgent": "Urgency or account-suspension pressure.",
"rule_secret": "Requests a password or secret verification code.",
"rule_money": "Requests advance fees or payments.",
"rule_prize": "Promises prizes, compensation, or free money.",
"rule_personal": "Requests personal or banking details.",
"rule_profit": "Promises guaranteed returns or risk-free investment.",
"rule_authority": "Mentions an official organization in a context that may need verification.",
"notice": "Never share passwords, OTP codes, or card details through messages.",
"history_title": "Scan History",
"history_empty": "No scans have been recorded in this session.",
"download": "📥 Download CSV report",
"clear_history": "Clear history",
"about_title": "About CyberShield AI",
"about_desc": "An educational prototype demonstrating how machine learning and heuristic rules can identify potential phishing and fraud indicators.",
"how_title": "How does it work?",
"step1": "Analyze text using a machine-learning classifier.",
"step2": "Check for patterns related to urgency, prizes, and secret information.",
"step3": "Inspect selected structural properties of URLs in the message.",
"limitations": "Limitations",
"limitations_desc": "The model uses a small example dataset and may misclassify messages. It cannot guarantee detection of every scam. A scan is not a definitive judgment about the safety of a message or URL.",
"qr_title": "Try the project on your phone",
"qr_desc": "Scan the QR code to open the application.",
"footer": "CyberShield AI | Digital Security Awareness Project",
"confidence": "Model-estimated suspicious classification probability",
"date": "Date & Time",
"message": "Message",
"classification": "Classification",
"risk_level": "Risk Level",
"ai_suspicious": "The model classified this message as suspicious.",
"ai_safe": "The model classified this message as normal.",
"language": "Language",
"about_button": "Explore how it works",
"sample_safe": "Hello, the meeting is tomorrow at ten. Thank you for your help.",
"sample_bad": "Urgent! Congratulations, you won a large cash prize. Send your password and verification code and pay a processing fee to receive the money.",
"url_check": "URL checks use selected patterns and do not inspect the actual website content.",
}
}

def tr(key):
return TEXT[st.session_state.language][key]

=====================================================

4. TRAINING DATA

=====================================================

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
"أتمنى لك يوما سعيدا.",
"تم استلام طلبك بنجاح.",
"سنلتقي في الجامعة غدا.",
"أرسل لي جدول المحاضرات من فضلك.",
"وصلت إلى المنزل بسلام.",
"شكرا على رسالتك.",
"يرجى مراجعة الجدول الدراسي المرفق.",
"تم تحديث النظام بنجاح، شكرا لتعاونكم.",
"I will send the report this afternoon.",
"Your class begins at nine tomorrow.",
"The documents are ready for review.",
"هل وصلت إلى الجامعة؟",
"سأرسل لك الملف بعد المحاضرة.",
"نتقابل بكرة إن شاء الله.",
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
"تنبيه أمني يرجى تحديث بياناتك البنكية تفاديا لإيقاف البطاقة",
"Your bank account is locked, send your PIN immediately",
"You have won free money, pay the release fee now",
"Send your identity number to claim your compensation",
"Guaranteed investment returns with no risk, invest today",
"عاجل أرسل الرقم السري لاستعادة حسابك البنكي",
"مبروك تم اختيارك لاستلام منحة مالية ادفع رسوم المعالجة",
"أرسل رقم البطاقة البنكية لاستلام التعويض",
]

=====================================================

5. MODEL

=====================================================

@st.cache_resource
def load_model():
messages = SAFE_MESSAGES + SUSPICIOUS_MESSAGES

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
            min_df=1,
            sublinear_tf=True
        )
    ),
    ("classifier", MultinomialNB(alpha=0.25))
])

pipeline.fit(messages, labels)
return pipeline

model = load_model()

=====================================================

6. HEURISTIC RULES

=====================================================

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
r"إغلاق حسابك",
r"إيقاف حسابك",
r"آخر فرصة",
r"أكد بياناتك",
r"اكد بياناتك",
r"تنبيه أمني",
],
"weight": 15,
"translation": "rule_urgent",
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
],
"weight": 25,
"translation": "rule_secret",
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
r"رسوم التوصيل",
],
"weight": 25,
"translation": "rule_money",
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
r"اليانصيب",
],
"weight": 15,
"translation": "rule_prize",
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
r"بياناتك السرية",
],
"weight": 20,
"translation": "rule_personal",
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
r"ثراء سريع",
],
"weight": 20,
"translation": "rule_profit",
},
"authority": {
"patterns": [
r"\bworld bank\b",
r"\bworld health organization\b",
r"\bunited nations\b",
r"\bgovernment compensation\b",
r"\bofficial grant\b",
r"البنك الدولي",
r"الأمم المتحدة",
r"تعويض حكومي",
r"منحة حكومية",
r"منظمة دولية",
],
"weight": 5,
"translation": "rule_authority",
},
}

=====================================================

7. URL INSPECTION

=====================================================

URL_PATTERN = re.compile(
r"""(?i)\b(?:https?://|www.)[^\s<>"']+"""
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
"buff.ly",
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
        findings.append(tr("url_puny"))

    if not findings:
        findings.append(tr("url_ok"))

    return findings

except Exception:
    return [tr("url_invalid")]

=====================================================

8. MESSAGE ANALYSIS

=====================================================

def analyze_message(message):
message = message.strip()

if not message:
    return None

prediction = model.predict([message])[0]

probabilities = model.predict_proba([message])[0]
classes = list(model.classes_)

suspicious_probability = float(
    probabilities[classes.index("Suspicious")]
)

findings = []
score = 0

for rule in RULES.values():
    matched = any(
        re.search(pattern, message, re.IGNORECASE)
        for pattern in rule["patterns"]
    )

    if matched:
        findings.append(tr(rule["translation"]))
        score += rule["weight"]

urls = URL_PATTERN.findall(message)
url_results = []

for url in urls:
    results = inspect_url(url)

    url_results.append({
        "url": url,
        "findings": results,
    })

    for result in results:
        if result == tr("url_ip"):
            score += 20
        elif result == tr("url_short"):
            score += 10
        elif result == tr("url_http"):
            score += 10
        elif result == tr("url_puny"):
            score += 20

if prediction == "Suspicious":
    if suspicious_probability >= 0.75:
        score += 15
    elif suspicious_probability >= 0.50:
        score += 8

score = min(score, 100)

if score >= 55:
    risk_level = "high"
    classification = "Suspicious"
elif score >= 25:
    risk_level = "medium"
    classification = "Suspicious"
else:
    risk_level = "low"

    if (
        prediction == "Suspicious"
        and suspicious_probability >= 0.65
    ):
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
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

=====================================================

9. QR CODE

=====================================================

@st.cache_data
def generate_qr_code(url):
qr = qrcode.QRCode(
version=1,
box_size=8,
border=3,
)

qr.add_data(url)
qr.make(fit=True)

image = qr.make_image(
    fill_color="#17152F",
    back_color="white",
)

buffer = BytesIO()
image.save(buffer, format="PNG")

return buffer.getvalue()

=====================================================

10. FUTURISTIC UI

=====================================================

st.markdown(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

:root {
    --background: #0b0a18;
    --panel: #151329;
    --purple: #9b72ff;
    --cyan: #59e1ff;
    --text: #f4f2ff;
    --muted: #b5b0d0;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #292052 0%, transparent 40%),
        radial-gradient(ellipse at 95% 25%, #17254b 0%, transparent 32%),
        #0b0a18;
    color: var(--text);
    font-family: 'Cairo', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17132d, #0e1021);
    border-right: 1px solid rgba(155,114,255,.22);
}

.hero {
    padding: 2.7rem 1.8rem;
    border: 1px solid rgba(155,114,255,.35);
    border-radius: 26px;
    text-align: center;
    background: linear-gradient(
        135deg,
        rgba(83,56,151,.30),
        rgba(19,29,59,.65)
    );
    box-shadow: 0 15px 55px rgba(0,0,0,.22);
    margin: 1rem 0 2rem 0;
}

.badge {
    display: inline-block;
    padding: .4rem .9rem;
    border-radius: 25px;
    border: 1px solid rgba(89,225,255,.4);
    color: #8ceaff;
    background: rgba(89,225,255,.07);
    font-size: .85rem;
    margin-bottom: 1rem;
}

.hero h1 {
    font-size: clamp(2rem, 5vw, 3.3rem);
    line-height: 1.4;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #bda6ff, #59e1ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #c5c0df;
    max-width: 750px;
    margin: auto;
    line-height: 2;
    font-size: 1.05rem;
}

.section-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 1.5rem;
    margin-bottom: .3rem;
}

.section-desc {
    color: #b5b0d0;
    line-height: 1.9;
    margin-bottom: 1.4rem;
}

.feature-card {
    padding: 1.25rem;
    border: 1px solid rgba(155,114,255,.25);
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        rgba(34,29,65,.88),
        rgba(17,23,44,.92)
    );
    min-height: 165px;
    margin-bottom: .8rem;
}

.feature-card h3 {
    color: #c8b6ff;
    font-size: 1.12rem;
}

.feature-card p {
    color: #c4c0da;
    line-height: 1.8;
    font-size: .95rem;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    min-height: 2.8rem;
    font-weight: 700;
    border: 1px solid rgba(155,114,255,.55);
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #59e1ff;
    color: #ffffff;
    box-shadow: 0 0 18px rgba(89,225,255,.13);
}

.stTextArea textarea {
    background: #111025;
    border: 1px solid #48406f;
    border-radius: 14px;
    color: #ffffff;
    line-height: 1.8;
}

.stTextArea textarea:focus {
    border-color: #9b72ff;
    box-shadow: 0 0 0 1px #9b72ff;
}

div[data-testid="stMetric"] {
    background: rgba(31,27,59,.8);
    border: 1px solid rgba(155,114,255,.25);
    padding: 1rem;
    border-radius: 15px;
}

.result-card {
    border-radius: 18px;
    border: 1px solid rgba(155,114,255,.32);
    padding: 1.3rem;
    background: rgba(23,20,46,.8);
    margin: .8rem 0 1.2rem 0;
    line-height: 1.9;
}

.footer {
    text-align: center;
    color: #aaa4ca;
    padding: 2rem .5rem .5rem .5rem;
    border-top: 1px solid rgba(155,114,255,.2);
    margin-top: 3rem;
    line-height: 2;
}

h1, h2, h3 {
    color: #f4f2ff;
}

[data-testid="stCaptionContainer"] {
    color: #b5b0d0;
}

</style>
""",
unsafe_allow_html=True,

)

=====================================================

11. SIDEBAR

=====================================================

with st.sidebar:
st.markdown(
"""
<div style="text-align:center;padding:12px 0 20px 0">
<div style="font-size:3rem">🛡️</div>
<div style="font-size:1.25rem;font-weight:800;color:#c6b1ff">
CYBERSHIELD
</div>
<div style="font-size:.8rem;color:#a9a3ca;letter-spacing:2px">
ARTIFICIAL INTELLIGENCE
</div>
</div>
""",
unsafe_allow_html=True,
)

language_choice = st.radio(
    "اللغة / Language",
    ["العربية", "English"],
    index=0 if st.session_state.language == "ar" else 1,
)

selected_language = (
    "ar" if language_choice == "العربية" else "en"
)

if selected_language != st.session_state.language:
    st.session_state.language = selected_language
    st.rerun()

st.divider()

page = st.radio(
    tr("tagline"),
    [
        tr("home"),
        tr("scanner"),
        tr("history"),
        tr("about"),
    ],
    label_visibility="collapsed",
)

st.divider()

st.caption("AI • TEXT ANALYSIS • URL INSPECTION")
st.caption("Educational prototype")

=====================================================

12. HOME PAGE

=====================================================

if page == tr("home"):

st.markdown(
    f"""
    <div class="hero">
        <div class="badge">✦ {tr("hero_badge")}</div>
        <h1>{tr("hero_title")}<br>{tr("hero_title2")}</h1>
        <p>{tr("hero_desc")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    if st.button(
        tr("start"),
        use_container_width=True,
        type="primary",
    ):
        st.session_state.page_request = tr("scanner")
        st.rerun()

with col2:
    if st.button(
        tr("learn"),
        use_container_width=True,
    ):
        st.session_state.page_request = tr("about")
        st.rerun()

# Apply requested navigation on the next run.
if "page_request" in st.session_state:
    requested = st.session_state.pop("page_request")

    if requested == tr("scanner"):
        st.session_state.last_navigation = "scanner"
    elif requested == tr("about"):
        st.session_state.last_navigation = "about"

st.markdown(
    f'<div class="section-title">{tr("features_title") if "features_title" in TEXT[st.session_state.language] else tr("feature1") + " • " + tr("feature2") + " • " + tr("feature3")}</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="feature-card">
            <div style="font-size:2rem">🧠</div>
            <h3>{tr("feature1")}</h3>
            <p>{tr("feature1_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="feature-card">
            <div style="font-size:2rem">🔗</div>
            <h3>{tr("feature2")}</h3>
            <p>{tr("feature2_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="feature-card">
            <div style="font-size:2rem">📊</div>
            <h3>{tr("feature3")}</h3>
            <p>{tr("feature3_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="section-title">{tr("qr_title")}</div>',
    unsafe_allow_html=True,
)

qr_col1, qr_col2 = st.columns([1, 2])

with qr_col1:
    st.image(
        generate_qr_code(APP_URL),
        caption=tr("qr_desc"),
        use_container_width=True,
    )

with qr_col2:
    st.markdown(
        f"""
        <div class="feature-card">
            <h3>🌐 {tr("brand")}</h3>
            <p>{tr("hero_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button(
        tr("start"),
        APP_URL,
        use_container_width=True,
    )

=====================================================

13. SCANNER PAGE

=====================================================

elif page == tr("scanner"):

st.markdown(
    f'<div class="section-title">{tr("lab_title")}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="section-desc">{tr("lab_desc")}</div>',
    unsafe_allow_html=True,
)

st.caption(tr("input_label"))

st.text_area(
    tr("placeholder"),
    height=180,
    key="message_input",
)

example_col1, example_col2 = st.columns(2)

with example_col1:
    if st.button(
        "🟢 " + tr("example_safe"),
        use_container_width=True,
    ):
        st.session_state.message_input = tr("sample_safe")
        st.rerun()

with example_col2:
    if st.button(
        "🟠 " + tr("example_suspicious"),
        use_container_width=True,
    ):
        st.session_state.message_input = tr("sample_bad")
        st.rerun()

st.write("")

scan_col, clear_col = st.columns(2)

with scan_col:
    scan_clicked = st.button(
        tr("scan"),
        type="primary",
        use_container_width=True,
    )

with clear_col:
    st.button(
        tr("clear"),
        on_click=clear_message,
        use_container_width=True,
    )

if scan_clicked:

    message = st.session_state.message_input.strip()

    if not message:
        st.warning(tr("empty"))

    else:
        result = analyze_message(message)

        st.session_state.last_result = result

        st.session_state.history.insert(
            0,
            {
                tr("date"): result["timestamp"],
                tr("message"): result["message"],
                tr("classification"): result["classification"],
                tr("score"): result["score"],
                tr("risk_level"): tr(result["risk_level"]),
            },
        )

result = st.session_state.last_result

if result is not None:

    st.divider()

    st.markdown(
        f'<div class="section-title">{tr("result_title")}</div>',
        unsafe_allow_html=True,
    )

    if result["classification"] == "Suspicious":
        st.error("⚠️ " + tr("suspicious"))
        st.markdown(
            f'<div class="result-card">🚨 {tr("ai_suspicious")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ " + tr("safe"))
        st.markdown(
            f'<div class="result-card">🛡️ {tr("ai_safe")}</div>',
            unsafe_allow_html=True,
        )

    risk_translation = tr(result["risk_level"])

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            tr("score"),
            f'{result["score"]}/100',
        )

    with metric2:
        st.metric(
            tr("risk"),
            risk_translation,
        )

    st.progress(result["score"] / 100)

    st.caption(
        f'{tr("confidence")}: '
        f'{result["model_probability"] * 100:.1f}%'
    )

    st.markdown(
        f'<div class="section-title">{tr("reasons")}</div>',
        unsafe_allow_html=True,
    )

    if result["findings"]:
        for finding in result["findings"]:
            st.warning("• " + finding)
    else:
        st.info(tr("no_reasons"))

    st.markdown(
        f'<div class="section-title">{tr("urls")}</div>',
        unsafe_allow_html=True,
    )

    if result["urls"]:

        for url_item in result["urls"]:

            st.code(url_item["url"], language=None)

            for finding in url_item["findings"]:
                st.write("•", finding)

    else:
        st.info(tr("no_urls"))

    st.info(tr("notice"))
    st.caption(tr("url_check"))

=====================================================

14. HISTORY PAGE

=====================================================

elif page == tr("history"):

st.markdown(
    f'<div class="section-title">{tr("history_title")}</div>',
    unsafe_allow_html=True,
)

if not st.session_state.history:

    st.info(tr("history_empty"))

else:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label=tr("download"),
        data=csv_data,
        file_name="cybershield_scan_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button(
        tr("clear_history"),
        use_container_width=True,
    ):
        st.session_state.history = []
        st.rerun()

=====================================================

15. ABOUT PAGE

=====================================================

elif page == tr("about"):

st.markdown(
    f'<div class="section-title">{tr("about_title")}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="feature-card"><p>{tr("about_desc")}</p></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="section-title">{tr("how_title")}</div>',
    unsafe_allow_html=True,
)

st.markdown("1. " + tr("step1"))
st.markdown("2. " + tr("step2"))
st.markdown("3. " + tr("step3"))

st.markdown(
    f'<div class="section-title">{tr("limitations")}</div>',
    unsafe_allow_html=True,
)

st.warning(tr("limitations_desc"))

st.markdown(
    f'<div class="section-title">{tr("qr_title")}</div>',
    unsafe_allow_html=True,
)

st.image(
    generate_qr_code(APP_URL),
    width=220,
    caption=tr("qr_desc"),
)

st.link_button(
    "🌐 " + tr("brand"),
    APP_URL,
)

=====================================================

16. FOOTER

=====================================================

st.markdown(
f"""
<div class="footer">
<div style="font-size:1.25rem;font-weight:800;color:#c6b1ff">
🛡️ CYBERSHIELD AI
</div>
{tr("footer")}
<br>
<span style="font-size:.8rem">
{tr("limitations_desc")}
</span>
</div>
""",
unsafe_allow_html=True,
)
