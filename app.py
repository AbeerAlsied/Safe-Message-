import re
import ipaddress
from urllib.parse import urlparse
from datetime import datetime
from io import BytesIO
import time

import requests

import pandas as pd
import qrcode
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

APP_URL = "https://safetydetecto.streamlit.app/"

st.set_page_config(
    page_title="CyberShield AI | Smart Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 2. SESSION STATE
# =====================================================

if "language" not in st.session_state:
    st.session_state.language = "ar"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "message_input" not in st.session_state:
    st.session_state.message_input = ""


def clear_message():
    st.session_state.message_input = ""
    st.session_state.last_result = None


# =====================================================
# 3. TRANSLATIONS
# =====================================================

TEXT = {
    "ar": {
        "brand": "CYBERSHIELD AI",
        "tagline": "حماية رقمية أكثر ذكاءً",
        "home": "الرئيسية",
        "scanner": "مختبر الفحص",
        "history": "سجل الفحوصات",
        "about": "حول المشروع",
        "hero_badge": "منظومة تحليل الرسائل الذكية",
        "hero_title": "اكتشفي الرسائل المشبوهة",
        "hero_title2": "قبل أن تثقي بها",
        "hero_desc": "نظام تجريبي يجمع بين التعلم الآلي والقواعد التحليلية لرصد مؤشرات الاحتيال والتصيد الإلكتروني.",
        "start": "ابدئي الفحص الآن ↓",
        "learn": "تعرّفي على المشروع",
        "feature1": "تحليل الرسائل",
        "feature1_desc": "تصنيف أولي للرسائل باستخدام التعلم الآلي.",
        "feature2": "فحص الروابط",
        "feature2_desc": "اكتشاف بعض أنماط الروابط التي تستحق الحذر.",
        "feature3": "مؤشرات الخطورة",
        "feature3_desc": "عرض العلامات التي اكتشفها النظام.",
        "lab_title": "مختبر تحليل الرسائل",
        "lab_desc": "أدخلي رسالة لتحليلها ورصد المؤشرات المشبوهة.",
        "input_label": "الرسالة المراد فحصها",
        "placeholder": "الصقي هنا رسالة بريد إلكتروني أو واتساب أو رابطاً...",
        "scan": "🔍 تحليل الرسالة",
        "clear": "مسح الحقل",
        "empty": "الرجاء إدخال رسالة أولاً.",
        "examples": "جرّبي مثالاً",
        "example_safe": "رسالة عادية",
        "example_suspicious": "رسالة مشبوهة",
        "result_title": "نتيجة التحليل",
        "safe": "لم تُكتشف مؤشرات كافية لتصنيف الرسالة كمشبوهة.",
        "suspicious": "تم رصد مؤشرات تستدعي الحذر.",
        "score": "مؤشر الخطورة",
        "risk": "مستوى الخطورة التقديري",
        "low": "منخفض",
        "medium": "متوسط",
        "high": "مرتفع",
        "reasons": "المؤشرات المكتشفة",
        "no_reasons": "لم ترصد القواعد الحالية مؤشرات معروفة.",
        "urls": "تحليل الروابط",
        "no_urls": "لم يُعثر على روابط مطابقة لأنماط الفحص.",
        "url_ip": "يستخدم عنوان IP مباشرة بدلاً من اسم النطاق.",
        "url_short": "يستخدم خدمة اختصار روابط.",
        "url_http": "يستخدم HTTP غير المشفر.",
        "url_puny": "يحتوي اسم النطاق على Punycode.",
        "url_suspicious_domain": "اسم النطاق غير مألوف أو يحتاج إلى تحقق إضافي.",
        "url_ok": "لم تُكتشف الأنماط المحددة في قواعد فحص الرابط.",
        "url_invalid": "تعذر تحليل بنية الرابط.",
        "rule_urgent": "لغة استعجال أو ضغط لاتخاذ إجراء سريع.",
        "rule_secret": "طلب كلمة مرور أو رمز تحقق سري.",
        "rule_money": "طلب رسوم مسبقة أو دفعات مالية.",
        "rule_prize": "وعد بجائزة أو تعويض أو أموال مجانية.",
        "rule_personal": "طلب بيانات شخصية أو مصرفية.",
        "rule_profit": "وعود بأرباح مضمونة أو استثمار بلا مخاطر.",
        "rule_authority": "ذكر جهة رسمية؛ يلزم التحقق من مصدر الرسالة.",
        "rule_link_click": "حثّ على الضغط على رابط للحصول على منفعة أو تجنب مشكلة.",
        "notice": "تنبيه: لا تشاركي كلمات المرور أو رموز OTP أو بيانات البطاقات عبر الرسائل.",
        "history_title": "سجل الفحوصات",
        "history_empty": "لم تُسجل أي فحوصات في هذه الجلسة بعد.",
        "download": "📥 تنزيل السجل بصيغة CSV",
        "clear_history": "مسح السجل",
        "about_title": "عن CyberShield AI",
        "about_desc": "مشروع توعوي وتجريبي يستخدم التعلم الآلي وقواعد تحليلية لرصد مؤشرات التصيد والاحتيال الإلكتروني.",
        "how_title": "كيف يعمل النظام؟",
        "step1": "تحليل النص باستخدام نموذج تصنيف آلي.",
        "step2": "فحص مؤشرات مثل الاستعجال والجوائز والبيانات السرية.",
        "step3": "فحص بعض الخصائص البنيوية للروابط.",
        "limitations": "حدود النظام",
        "limitations_desc": "النموذج مدرّب على مجموعة أمثلة صغيرة وقد يخطئ. لا يفحص محتوى المواقع فعلياً ولا يضمن اكتشاف كل الاحتيالات. اعتبري النتيجة مؤشراً أولياً وليس حكماً نهائياً.",
        "qr_title": "جرّبي المشروع من هاتفك",
        "qr_desc": "امسحي رمز QR لفتح التطبيق.",
        "footer": "CyberShield AI | مشروع توعوي للأمن الرقمي",
        "confidence": "احتمال الاشتباه حسب النموذج",
        "date": "التاريخ والوقت",
        "message": "الرسالة",
        "classification": "التصنيف",
        "risk_level": "مستوى الخطورة",
        "ai_suspicious": "رجّح النموذج أن الرسالة مشبوهة.",
        "ai_safe": "رجّح النموذج أن الرسالة عادية.",
        "language": "اللغة",
        "sample_safe": "مرحباً، موعد الاجتماع غداً الساعة العاشرة. شكراً لمساعدتك.",
        "sample_bad": "عاجل! مبروك ربحت جائزة مالية كبيرة. أرسل كلمة المرور ورمز التحقق وادفع رسوم المعالجة لاستلام المبلغ.",
        "url_check": "فحص الرابط هنا بنيوي فقط؛ لا يفتح الموقع ولا يؤكد أنه آمن أو ضار.",
        "footer_note": "أداة توعوية تجريبية وليست نظام حماية مضموناً.",
        "result_caution": "النتيجة تقديرية. تحققي من المصدر الرسمي عبر تطبيقه أو موقعه المعروف، ولا تستخدمي الرابط المشبوه نفسه للتحقق.",
        "domain_note": "اسم النطاق وحده لا يثبت أن الرابط ضار أو آمن.",
        "matched_terms": "العبارات التي ساهمت في التنبيه",
        "external_title": "فحص إضافي عبر VirusTotal",
        "external_consent": "أوافق على إرسال الروابط الموجودة في الرسالة إلى VirusTotal لفحصها (قد تُشارك الروابط مع خدمة خارجية).",
        "external_not_checked": "لم يتم فحص الروابط عبر VirusTotal. يمكنك تفعيل الخيار أعلاه إذا رغبت.",
        "external_no_urls": "لا توجد روابط لإرسالها إلى VirusTotal.",
        "external_pending": "لم تكتمل نتيجة الفحص بعد. جرّب لاحقاً.",
        "external_malicious": "اكتشافات ضارة",
        "external_suspicious": "اكتشافات مشبوهة",
        "external_harmless": "اكتشافات سليمة",
        "external_undetected": "لم تكتشفها المحركات",
        "external_error": "تعذر إكمال فحص VirusTotal:",
        "external_done": "نتائج VirusTotal لا تضمن أن الرابط آمن؛ هي مؤشر إضافي فقط.",
        "external_wait": "جارٍ إرسال الروابط وفحصها عبر VirusTotal...",
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
        "hero_desc": "An experimental tool combining machine learning and heuristic rules to identify potential fraud and phishing indicators.",
        "start": "Start Scanning ↓",
        "learn": "Explore the Project",
        "feature1": "Message Analysis",
        "feature1_desc": "Preliminary classification using machine learning.",
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
        "safe": "Not enough indicators were found to classify this message as suspicious.",
        "suspicious": "Potential warning indicators detected.",
        "score": "Risk indicator",
        "risk": "Estimated risk level",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "reasons": "Detected indicators",
        "no_reasons": "No indicators matched the current rules.",
        "urls": "URL inspection",
        "no_urls": "No URLs matching the inspection pattern were found.",
        "url_ip": "Uses a raw IP address instead of a domain name.",
        "url_short": "Uses a URL-shortening service.",
        "url_http": "Uses unencrypted HTTP.",
        "url_puny": "The hostname contains Punycode.",
        "url_suspicious_domain": "The domain is unfamiliar or needs further verification.",
        "url_ok": "No patterns covered by the current URL rules were detected.",
        "url_invalid": "Could not parse the URL structure.",
        "rule_urgent": "Urgency or pressure to act quickly.",
        "rule_secret": "Requests a password or secret verification code.",
        "rule_money": "Requests advance fees or payments.",
        "rule_prize": "Promises prizes, compensation, or free money.",
        "rule_personal": "Requests personal or banking details.",
        "rule_profit": "Promises guaranteed returns or risk-free investment.",
        "rule_authority": "Mentions an official organization; verify the source.",
        "rule_link_click": "Encourages clicking a link to get a benefit or avoid a problem.",
        "notice": "Never share passwords, OTP codes, or card details through messages.",
        "history_title": "Scan History",
        "history_empty": "No scans have been recorded in this session.",
        "download": "📥 Download CSV report",
        "clear_history": "Clear history",
        "about_title": "About CyberShield AI",
        "about_desc": "An educational prototype using machine learning and heuristic rules to identify potential phishing and fraud indicators.",
        "how_title": "How does it work?",
        "step1": "Analyze text using a machine-learning classifier.",
        "step2": "Check indicators such as urgency, prizes, and secret information.",
        "step3": "Inspect selected structural properties of URLs.",
        "limitations": "Limitations",
        "limitations_desc": "The model uses a small example dataset and may misclassify messages. It does not actually inspect website content and cannot guarantee detection of every scam. Treat results as preliminary indicators, not a final verdict.",
        "qr_title": "Try the project on your phone",
        "qr_desc": "Scan the QR code to open the application.",
        "footer": "CyberShield AI | Digital Security Awareness Project",
        "confidence": "Model-estimated suspicious probability",
        "date": "Date & Time",
        "message": "Message",
        "classification": "Classification",
        "risk_level": "Risk Level",
        "ai_suspicious": "The model leans toward suspicious.",
        "ai_safe": "The model leans toward normal.",
        "language": "Language",
        "sample_safe": "Hello, the meeting is tomorrow at ten. Thank you for your help.",
        "sample_bad": "Urgent! Congratulations, you won a large cash prize. Send your password and verification code and pay a processing fee to receive the money.",
        "url_check": "URL inspection here is structural only; it does not open the website or confirm it is safe or malicious.",
        "footer_note": "An educational prototype, not a guaranteed protection system.",
        "result_caution": "Results are estimates. Verify through the organization's known official app or website, not through the suspicious link.",
        "domain_note": "A domain name alone does not prove that a URL is malicious or safe.",
        "matched_terms": "Phrases that contributed to the warning",
        "external_title": "Additional VirusTotal check",
        "external_consent": "I agree to send URLs found in this message to VirusTotal for analysis (URLs may be shared with an external service).",
        "external_not_checked": "URLs were not checked with VirusTotal. Enable the option above if you want to.",
        "external_no_urls": "There are no URLs to submit to VirusTotal.",
        "external_pending": "The scan has not completed yet. Please try again later.",
        "external_malicious": "Malicious detections",
        "external_suspicious": "Suspicious detections",
        "external_harmless": "Harmless detections",
        "external_undetected": "Undetected engines",
        "external_error": "VirusTotal scan could not be completed:",
        "external_done": "VirusTotal results do not guarantee a URL is safe; treat them as an additional signal only.",
        "external_wait": "Submitting URLs to VirusTotal for analysis...",
    },
}


def tr(key):
    return TEXT[st.session_state.language][key]


# =====================================================
# 4. TRAINING DATA
# =====================================================

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
    "I will send the report this afternoon.",
    "Your class begins at nine tomorrow.",
    "The documents are ready for review.",
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
    "هل وصلت إلى الجامعة؟",
    "سأرسل لك الملف بعد المحاضرة.",
    "نتقابل بكرة إن شاء الله.",
    "Your teacher shared the slides for next week's lecture.",
    "The library will close at five today.",
    "Please review the meeting notes when you have time.",
    "شكراً، سأراجع المستند وأرد عليك غداً.",
    "تم تغيير موعد المحاضرة إلى يوم الاثنين.",
    "يمكنك استلام الشهادة من مكتب التسجيل.",
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
    "Your bank account is locked, send your PIN immediately",
    "Send your identity number to claim your compensation",
    "Guaranteed investment returns with no risk, invest today",
    "Urgent! You have won a cash prize. Pay a processing fee.",
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
    "عاجل أرسل الرقم السري لاستعادة حسابك البنكي",
    "مبروك تم اختيارك لاستلام منحة مالية ادفع رسوم المعالجة",
    "أرسل رقم البطاقة البنكية لاستلام التعويض",
    "The government is giving everyone free internet data. Click this strange link to claim it now!",
    "A public official approved a huge cash grant. Enter your personal details at this link.",
    "You have been selected for compensation. Pay a delivery fee to receive your money.",
    "تهانينا، إنترنت مجاني للجميع، اضغطي على الرابط الغريب لاستلامه فوراً.",
    "وافق مسؤول حكومي على تعويض كبير، أدخلي بياناتك الشخصية في الرابط.",
    "تم اختيارك لتعويض مالي، ادفعي رسوم التوصيل لاستلام المبلغ.",
]


# =====================================================
# 5. MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():
    messages = SAFE_MESSAGES + SUSPICIOUS_MESSAGES
    labels = ["Safe"] * len(SAFE_MESSAGES) + ["Suspicious"] * len(SUSPICIOUS_MESSAGES)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=0.25)),
    ])
    pipeline.fit(messages, labels)
    return pipeline


model = load_model()


# =====================================================
# 6. HEURISTIC RULES
# =====================================================

RULES = {
    "urgent": {
        "patterns": [
            r"\burgent\b", r"\bimmediately\b", r"\bact now\b",
            r"\blast chance\b", r"\baccount will be closed\b",
            r"\baccount is blocked\b", r"\bverify now\b",
            r"عاجل", r"فورا", r"فوراً", r"حالاً", r"الآن",
            r"إغلاق حسابك", r"إيقاف حسابك", r"آخر فرصة",
            r"أكد بياناتك", r"اكد بياناتك", r"تنبيه أمني",
            r"اضغط.{0,20}الآن", r"اضغطي.{0,20}فوراً",
            r"استلم.{0,20}فوراً", r"استلمي.{0,20}فوراً",
        ],
        "weight": 15, "translation": "rule_urgent",
    },
    "secret": {
        "patterns": [
            r"\bpassword\b", r"\bpasscode\b", r"\botp\b",
            r"\bverification code\b", r"\bsecurity code\b",
            r"\bpin code\b", r"كلمة المرور", r"رمز التحقق",
            r"رمز التأكيد", r"الرقم السري", r"الرقم السرى",
            r"كود التحقق",
        ],
        "weight": 25, "translation": "rule_secret",
    },
    "money": {
        "patterns": [
            r"\bprocessing fee\b", r"\badvance fee\b",
            r"\bpay a fee\b", r"\btransfer money\b",
            r"\bwire transfer\b", r"\bdelivery fee\b",
            r"رسوم المعالجة", r"رسوم مقدما", r"رسوم مقدماً",
            r"ادفع رسوم", r"دفع رسوم", r"تحويل أموال",
            r"حول مبلغ", r"حوّل مبلغ", r"رسوم الشحن",
            r"رسوم التوصيل", r"ادفعي رسوم", r"رسوم مسبقة",
        ],
        "weight": 25, "translation": "rule_money",
    },
    "prize": {
        "patterns": [
            r"\byou won\b", r"\byou have won\b",
            r"\bclaim your prize\b", r"\bfree money\b",
            r"\blottery\b", r"\breward\b", r"\bcompensation\b",
            r"\bgrant money\b", r"ربحت", r"جائزة", r"تعويض مالي",
            r"تعويض", r"منحة مالية", r"أموال مجانية", r"مبلغ مالي",
            r"مبروك", r"اليانصيب", r"إنترنت مجاني",
            r"انترنت مجاني", r"بيانات مجانية", r"جيجابايت مجانية",
            r"مجانا للجميع", r"مجانية للجميع",
        ],
        "weight": 15, "translation": "rule_prize",
    },
    "personal": {
        "patterns": [
            r"\bbank details\b", r"\bcard number\b",
            r"\bpersonal details\b", r"\bidentity number\b",
            r"\bsend your details\b", r"\baccount information\b",
            r"بياناتك الشخصية", r"بيانات الحساب", r"بياناتك البنكية",
            r"رقم البطاقة", r"رقم الحساب البنكي", r"معلوماتك الشخصية",
            r"أرسل بياناتك", r"ارسل بياناتك", r"بياناتك السرية",
            r"أدخلي بياناتك", r"ادخلي بياناتك",
        ],
        "weight": 20, "translation": "rule_personal",
    },
    "profit": {
        "patterns": [
            r"\bguaranteed profit\b", r"\bguaranteed returns\b",
            r"\bno risk\b", r"\bdouble your money\b",
            r"\bhuge profits\b", r"\bget rich quick\b",
            r"أرباح مضمونة", r"ربح مضمون", r"بدون مخاطر",
            r"بدون أي مخاطرة", r"ضاعف أموالك", r"مضاعفة أموالك",
            r"استثمار مضمون", r"ثراء سريع",
        ],
        "weight": 20, "translation": "rule_profit",
    },
    "authority": {
        "patterns": [
            r"\bworld bank\b", r"\bworld health organization\b",
            r"\bunited nations\b", r"\bgovernment compensation\b",
            r"\bofficial grant\b", r"البنك الدولي", r"الأمم المتحدة",
            r"تعويض حكومي", r"منحة حكومية", r"منظمة دولية",
            r"رئيس مجلس السيادة", r"الحكومة السودانية",
            r"الحكومة تقدم", r"بمناسبة المولد النبوي",
        ],
        "weight": 5, "translation": "rule_authority",
    },
    "link_click": {
        "patterns": [
            r"\bclick (?:here|the link|below)\b",
            r"\bclaim (?:it|now|your)\b",
            r"اضغط.{0,25}الرابط", r"اضغطي.{0,25}الرابط",
            r"الرابط أدناه", r"الرابط ادناه",
            r"احصل.{0,20}الإنترنت المجاني",
            r"احصلي.{0,20}الإنترنت المجاني",
        ],
        "weight": 15, "translation": "rule_link_click",
    },
}


# =====================================================
# 7. URL INSPECTION
# =====================================================

URL_PATTERN = re.compile(
    r"""(?i)(?:https?://|www\.)[^\s<>"'<>]+"""
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly",
    "rb.gy", "ow.ly", "shorturl.at", "buff.ly",
}

# These domains are known examples of commonly used legitimate providers.
# A domain not listed here is NOT automatically malicious.
COMMON_DOMAINS = {
    "google.com", "www.google.com", "youtube.com", "www.youtube.com",
    "microsoft.com", "www.microsoft.com", "apple.com", "www.apple.com",
    "wikipedia.org", "www.wikipedia.org", "github.com", "www.github.com",
    "facebook.com", "www.facebook.com", "whatsapp.com", "www.whatsapp.com",
    "instagram.com", "www.instagram.com", "gov.sd", "www.gov.sd",
}


def extract_urls(message):
    found = URL_PATTERN.findall(message)
    cleaned = []
    for url in found:
        url = url.rstrip(".,!?;:)]}،؛؟\"'")
        if url and url not in cleaned:
            cleaned.append(url)
    return cleaned


def inspect_url(raw_url):
    url = raw_url.strip().rstrip(".,!?;:)]}،؛؟\"'")
    if url.lower().startswith("www."):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return [tr("url_invalid")]

        hostname = hostname.lower().strip(".")
        findings = []

        try:
            ipaddress.ip_address(hostname)
            findings.append(tr("url_ip"))
        except ValueError:
            pass

        if hostname in SHORTENERS:
            findings.append(tr("url_short"))

        if parsed.scheme.lower() != "https":
            findings.append(tr("url_http"))

        if "xn--" in hostname:
            findings.append(tr("url_puny"))

        if not findings:
            findings.append(tr("url_ok"))

        return findings
    except (ValueError, TypeError):
        return [tr("url_invalid")]


# =====================================================
# 7B. OPTIONAL VIRUSTOTAL URL CHECK
# =====================================================

VT_BASE_URL = "https://www.virustotal.com/api/v3"


def scan_url_with_virustotal(url):
    """Submit one URL to VirusTotal and briefly wait for its analysis."""
    try:
        api_key = st.secrets.get("VT_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return {"url": url, "error": "VT_API_KEY is missing from Streamlit Secrets."}

    headers = {"x-apikey": api_key}
    try:
        response = requests.post(
            f"{VT_BASE_URL}/urls",
            headers=headers,
            data={"url": url},
            timeout=20,
        )
        if response.status_code == 429:
            return {"url": url, "error": "VirusTotal API rate limit reached. Try again later."}
        response.raise_for_status()
        analysis_id = response.json().get("data", {}).get("id")
        if not analysis_id:
            return {"url": url, "error": "VirusTotal did not return an analysis ID."}

        for _ in range(6):
            time.sleep(2)
            result_response = requests.get(
                f"{VT_BASE_URL}/analyses/{analysis_id}",
                headers=headers,
                timeout=20,
            )
            if result_response.status_code == 429:
                return {"url": url, "error": "VirusTotal API rate limit reached. Try again later."}
            result_response.raise_for_status()
            data = result_response.json().get("data", {}).get("attributes", {})
            if data.get("status") == "completed":
                stats = data.get("stats", {})
                return {
                    "url": url,
                    "malicious": int(stats.get("malicious", 0)),
                    "suspicious": int(stats.get("suspicious", 0)),
                    "harmless": int(stats.get("harmless", 0)),
                    "undetected": int(stats.get("undetected", 0)),
                    "status": "completed",
                }
        return {"url": url, "status": "pending"}
    except requests.RequestException as exc:
        return {"url": url, "error": str(exc)}
    except (ValueError, TypeError, KeyError) as exc:
        return {"url": url, "error": str(exc)}


def scan_message_urls(urls):
    results = []
    for url in urls:
        results.append(scan_url_with_virustotal(url))
    return results


# =====================================================
# 8. MESSAGE ANALYSIS
# =====================================================

def analyze_message(message):
    message = message.strip()
    if not message:
        return None

    # Model output is a supporting signal, not a verdict.
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)
    suspicious_probability = float(probabilities[classes.index("Suspicious")])

    lower_message = message.lower()
    findings = []
    matched_rules = []
    matched_phrases = []
    score = 0

    for rule_name, rule in RULES.items():
        hits = []
        for pattern in rule["patterns"]:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                hits.append(match.group(0))

        if hits:
            matched_rules.append(rule_name)
            findings.append(tr(rule["translation"]))
            # Show up to two matched phrases for transparency.
            for phrase in hits[:2]:
                if phrase not in matched_phrases:
                    matched_phrases.append(phrase)
            score += rule["weight"]

    urls = extract_urls(message)
    url_results = []
    strong_url_signal = False

    for url in urls:
        results = inspect_url(url)
        url_results.append({"url": url, "findings": results})

        for item in results:
            if item == tr("url_ip"):
                score += 20
                strong_url_signal = True
            elif item == tr("url_short"):
                score += 10
            elif item == tr("url_http"):
                score += 5
            elif item == tr("url_puny"):
                score += 20
                strong_url_signal = True

    # The model contributes only a modest amount, since its training set is small.
    if prediction == "Suspicious":
        if suspicious_probability >= 0.80:
            score += 10
        elif suspicious_probability >= 0.60:
            score += 5

    score = min(score, 100)

    # Avoid declaring a message suspicious based only on a single broad keyword.
    strong_rule_groups = {"secret", "money", "personal", "link_click"}
    has_strong_combination = (
        len(set(matched_rules).intersection(strong_rule_groups)) >= 2
    )
    has_multiple_signals = len(matched_rules) >= 2
    has_high_score = score >= 45

    if score >= 60 or (has_strong_combination and score >= 35) or (strong_url_signal and score >= 35):
        classification = "Suspicious"
        risk_level = "high" if score >= 60 or strong_url_signal else "medium"
    elif score >= 30 or (has_multiple_signals and score >= 25):
        classification = "Suspicious"
        risk_level = "medium"
    elif prediction == "Suspicious" and suspicious_probability >= 0.80 and score >= 15:
        classification = "Suspicious"
        risk_level = "medium"
    else:
        classification = "Safe"
        risk_level = "low" if score < 15 else "medium"

    # Avoid repeated identical reason strings.
    findings = list(dict.fromkeys(findings))

    return {
        "message": message,
        "classification": classification,
        "score": score,
        "risk_level": risk_level,
        "findings": findings,
        "matched_phrases": matched_phrases,
        "urls": url_results,
        "model_probability": suspicious_probability,
        "model_prediction": prediction,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "external_results": [],
    }


# =====================================================
# 9. QR CODE
# =====================================================

@st.cache_data
def generate_qr_code(url):
    qr = qrcode.QRCode(version=1, box_size=8, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="#17152F", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================
# 10. FUTURISTIC CSS + RTL
# =====================================================

CSS_STYLE = """
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

html:lang(ar) .stApp,
html:lang(ar) [data-testid="stSidebar"],
html:lang(ar) .stMarkdown,
html:lang(ar) label,
html:lang(ar) [data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
}

html:lang(ar) .stTextArea textarea {
    direction: rtl;
    text-align: right;
}

html:lang(ar) input,
html:lang(ar) textarea {
    text-align: right;
}

.hero {
    padding: 2.7rem 1.8rem;
    border: 1px solid rgba(155,114,255,.35);
    border-radius: 26px;
    text-align: center;
    background: linear-gradient(135deg, rgba(83,56,151,.30), rgba(19,29,59,.65));
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
    line-height: 1.5;
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
    color: #d8ceff;
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
    background: linear-gradient(145deg, rgba(34,29,65,.88), rgba(17,23,44,.92));
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
    line-height: 1.9;
}

.small-note {
    color: #aaa4ca;
    font-size: .88rem;
    line-height: 1.8;
}

.stAlert {
    border-radius: 12px;
}

h1, h2, h3, p, label {
    font-family: 'Cairo', sans-serif;
}

a {
    color: #8ceaff !important;
}

@media (max-width: 700px) {
    .hero { padding: 1.8rem 1rem; }
    .hero h1 { font-size: 2rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
"""


st.markdown(CSS_STYLE, unsafe_allow_html=True)


# =====================================================
# 11. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">🛡️</div>
            <h2 style="color:#c8b6ff; margin-bottom:0;">CYBERSHIELD</h2>
            <p style="color:#59e1ff; margin-top:0;">AI SECURITY LAB</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    language_choice = st.radio(
        tr("language"),
        ["العربية", "English"],
        index=0 if st.session_state.language == "ar" else 1,
        horizontal=True,
        key="language_selector",
    )
    new_language = "ar" if language_choice == "العربية" else "en"
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.rerun()

    st.markdown("---")
    page_options = {
        tr("home"): "Home",
        tr("scanner"): "Scanner",
        tr("history"): "History",
        tr("about"): "About",
    }
    labels = list(page_options.keys())
    current_label = next(
        (label for label, page_name in page_options.items()
         if page_name == st.session_state.page),
        labels[0],
    )
    selected_label = st.radio(
        "Navigation", labels,
        index=labels.index(current_label),
        label_visibility="collapsed",
    )
    st.session_state.page = page_options[selected_label]

    st.markdown("---")
    st.caption("🟣 Machine Learning")
    st.caption("🔵 URL Pattern Inspection")
    st.caption("🛡️ Digital Awareness")


# =====================================================
# 12. REUSABLE ANALYSIS RESULT
# =====================================================

def display_result(result):
    if not result:
        return

    st.markdown(f'<div class="section-title">{tr("result_title")}</div>', unsafe_allow_html=True)

    if result["classification"] == "Suspicious":
        st.error("⚠️ " + tr("suspicious"))
    else:
        st.success("✅ " + tr("safe"))

    col1, col2, col3 = st.columns(3)
    col1.metric(tr("score"), f'{result["score"]}/100')
    col2.metric(tr("risk"), tr(result["risk_level"]))
    col3.metric(tr("confidence"), f'{result["model_probability"] * 100:.1f}%')

    st.progress(result["score"] / 100)

    st.markdown(f"### 🚩 {tr('reasons')}")
    if result["findings"]:
        for finding in result["findings"]:
            st.markdown(f"- ⚠️ {finding}")
    else:
        st.info(tr("no_reasons"))

    if result.get("matched_phrases"):
        st.markdown(f"**{tr('matched_terms')}:**")
        st.code(" • ".join(result["matched_phrases"]), language=None)

    st.markdown(f"### 🔗 {tr('urls')}")
    if result["urls"]:
        for item in result["urls"]:
            st.markdown(f"**{item['url']}**")
            for finding in item["findings"]:
                st.markdown(f"- {finding}")
    else:
        st.info(tr("no_urls"))

    st.caption(tr("url_check"))
    st.caption(tr("domain_note"))
    st.warning(tr("notice"))
    st.info(tr("result_caution"))

    external_results = result.get("external_results", [])
    if external_results:
        st.markdown(f"### 🌐 {tr('external_title')}")
        for external in external_results:
            st.markdown(f"**{external.get('url', '')}**")
            if external.get("error"):
                st.warning(f"{tr('external_error')} {external['error']}")
            elif external.get("status") == "pending":
                st.info(tr("external_pending"))
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(tr("external_malicious"), external.get("malicious", 0))
                m2.metric(tr("external_suspicious"), external.get("suspicious", 0))
                m3.metric(tr("external_harmless"), external.get("harmless", 0))
                m4.metric(tr("external_undetected"), external.get("undetected", 0))
                if external.get("malicious", 0) > 0 or external.get("suspicious", 0) > 0:
                    st.error(tr("suspicious"))
                else:
                    st.info(tr("external_done"))


# =====================================================
# 13. HOME PAGE
# =====================================================

if st.session_state.page == "Home":
    st.markdown(
        f"""
        <div class="hero">
            <div class="badge">🛡️ {tr("hero_badge")}</div>
            <h1>{tr("hero_title")}<br>{tr("hero_title2")}</h1>
            <p>{tr("hero_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"### ✨ {tr('brand')}")
        st.markdown(f"#### {tr('tagline')}")
        st.write(tr("hero_desc"))
        if st.button(tr("start"), use_container_width=True):
            st.session_state.page = "Scanner"
            st.rerun()
        if st.button(tr("learn"), use_container_width=True):
            st.session_state.page = "About"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="feature-card" style="text-align:center;">
                <div style="font-size:4rem;">🛡️</div>
                <h3>AI + CYBER SECURITY</h3>
                <p>Machine learning, message analysis, URL inspection and digital awareness.</p>
                <div style="font-size:1.6rem;">🟣　🔵　✨</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f'<div class="section-title">{tr("about_title")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    for col, icon, title_key, desc_key in [
        (c1, "🔍", "feature1", "feature1_desc"),
        (c2, "🔗", "feature2", "feature2_desc"),
        (c3, "🚨", "feature3", "feature3_desc"),
    ]:
        with col:
            st.markdown(
                f'<div class="feature-card"><h3>{icon} {tr(title_key)}</h3><p>{tr(desc_key)}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    qr_col, info_col = st.columns([1, 2])
    with qr_col:
        st.image(generate_qr_code(APP_URL), caption="CyberShield AI", width=210)
    with info_col:
        st.markdown(f"### 📱 {tr('qr_title')}")
        st.write(tr("qr_desc"))
        st.code(APP_URL, language=None)
        st.markdown(f'<p class="small-note">{tr("footer_note")}</p>', unsafe_allow_html=True)


# =====================================================
# 14. SCANNER PAGE
# =====================================================

elif st.session_state.page == "Scanner":
    st.markdown(f"# 🔬 {tr('lab_title')}")
    st.markdown(f'<p class="section-desc">{tr("lab_desc")}</p>', unsafe_allow_html=True)
    st.markdown(f"**{tr('examples')}**")

    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("✅ " + tr("example_safe"), use_container_width=True):
            st.session_state.message_input = tr("sample_safe")
            st.session_state.last_result = None
            st.rerun()
    with example_col2:
        if st.button("⚠️ " + tr("example_suspicious"), use_container_width=True):
            st.session_state.message_input = tr("sample_bad")
            st.session_state.last_result = None
            st.rerun()

    st.text_area(
        tr("input_label"),
        key="message_input",
        height=220,
        placeholder=tr("placeholder"),
    )

    external_scan_consent = st.checkbox(tr("external_consent"), key="external_scan_consent")

    scan_col, clear_col = st.columns(2)
    with scan_col:
        scan_clicked = st.button(tr("scan"), use_container_width=True, type="primary")
    with clear_col:
        st.button(tr("clear"), use_container_width=True, on_click=clear_message)

    if scan_clicked:
        message = st.session_state.message_input.strip()
        if not message:
            st.warning(tr("empty"))
        else:
            result = analyze_message(message)
            urls_to_scan = [item["url"] for item in result.get("urls", [])]
            if external_scan_consent:
                if urls_to_scan:
                    with st.spinner(tr("external_wait")):
                        result["external_results"] = scan_message_urls(urls_to_scan)
                else:
                    st.info(tr("external_no_urls"))
            else:
                st.caption(tr("external_not_checked"))
            st.session_state.last_result = result
            st.session_state.history.insert(
                0,
                {
                    tr("date"): result["timestamp"],
                    tr("message"): result["message"],
                    tr("classification"): result["classification"],
                    tr("risk_level"): tr(result["risk_level"]),
                    tr("score"): result["score"],
                },
            )

    if st.session_state.last_result:
        display_result(st.session_state.last_result)


# =====================================================
# 15. HISTORY PAGE
# =====================================================

elif st.session_state.page == "History":
    st.markdown(f"# 📊 {tr('history_title')}")
    if not st.session_state.history:
        st.info(tr("history_empty"))
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=tr("download"),
            data=csv_data,
            file_name="cybershield_scan_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button(tr("clear_history"), use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# =====================================================
# 16. ABOUT PAGE
# =====================================================

elif st.session_state.page == "About":
    st.markdown(f"# 🛡️ {tr('about_title')}")
    st.markdown(f'<p class="section-desc">{tr("about_desc")}</p>', unsafe_allow_html=True)
    st.markdown(f"## ⚙️ {tr('how_title')}")
    st.markdown(f"1. {tr('step1')}")
    st.markdown(f"2. {tr('step2')}")
    st.markdown(f"3. {tr('step3')}")
    st.markdown("---")
    st.markdown(f"## ⚠️ {tr('limitations')}")
    st.warning(tr("limitations_desc"))
    st.markdown("---")
    st.markdown(f"## 📱 {tr('qr_title')}")
    st.write(tr("qr_desc"))
    qr_col, url_col = st.columns([1, 2])
    with qr_col:
        st.image(generate_qr_code(APP_URL), width=210)
    with url_col:
        st.code(APP_URL, language=None)
        st.markdown(f'<p class="small-note">{tr("footer_note")}</p>', unsafe_allow_html=True)


# =====================================================
# 17. FOOTER
# =====================================================

st.markdown(
    f"""
    <div class="footer">
        <strong>{tr("footer")}</strong><br>
        <span>{tr("footer_note")}</span><br>
        <span>Built with Python, Streamlit and Scikit-learn</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 
