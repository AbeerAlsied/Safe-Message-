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
        "hero_title": "اكتشف الرسائل المشبوهة",
        "hero_title2": "قبل أن تثق بها",
        "hero_desc": "نظام تجريبي يجمع بين التعلم الآلي والقواعد التحليلية لرصد مؤشرات الاحتيال والتصيد الإلكتروني.",
        "start": "ابدأ الفحص الآن ↓",
        "learn": "تعرّف على المشروع",
        "feature1": "تحليل الرسائل",
        "feature1_desc": "تصنيف أولي للرسائل باستخدام التعلم الآلي.",
        "feature2": "فحص الروابط",
        "feature2_desc": "اكتشاف بعض أنماط الروابط التي تستحق الحذر.",
        "feature3": "مؤشرات الخطورة",
        "feature3_desc": "عرض العلامات التي اكتشفها النظام.",
        "lab_title": "مختبر تحليل الرسائل",
        "lab_desc": "أدخل رسالة لتحليلها ورصد المؤشرات المشبوهة.",
        "input_label": "الرسالة المراد فحصها",
        "placeholder": "الصق هنا رسالة بريد إلكتروني أو واتساب أو رابطاً...",
        "scan": "🔍 تحليل الرسالة",
        "clear": "مسح الحقل",
        "empty": "الرجاء إدخال رسالة أولاً.",
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
        "notice": "تنبيه: لا تشارك كلمات المرور أو رموز OTP أو بيانات البطاقات عبر الرسائل.",
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
        "limitations_desc": "النموذج مدرّب على مجموعة أمثلة صغيرة وقد يخطئ. لا يفحص محتوى المواقع فعلياً ولا يضمن اكتشاف كل الاحتيالات. اعتبر النتيجة مؤشراً أولياً وليس حكماً نهائياً.",
        "qr_title": "جرّب المشروع من هاتفك",
        "qr_desc": "امسح رمز QR لفتح التطبيق.",
        "footer": "CyberShield AI | مشروع توعوي للأمن الرقمي",
        "confidence": "احتمال الاشتباه حسب النموذج",
        "date": "التاريخ والوقت",
        "message": "الرسالة",
        "classification": "التصنيف",
        "risk_level": "مستوى الخطورة",
        "ai_suspicious": "رجّح النموذج أن الرسالة مشبوهة.",
        "ai_safe": "رجّح النموذج أن الرسالة عادية.",
        "language": "اللغة",
        "url_check": "فحص الرابط هنا بنيوي فقط؛ لا يفتح الموقع ولا يؤكد أنه آمن أو ضار.",
        "footer_note": "أداة توعوية تجريبية وليست نظام حماية مضموناً.",
        "result_caution": "النتيجة تقديرية. تحقق من المصدر الرسمي عبر تطبيقه أو موقعه المعروف، ولا تستخدم الرابط المشبوه نفسه للتحقق.",
        "domain_note": "اسم النطاق وحده لا يثبت أن الرابط ضار أو آمن.",
        "matched_terms": "العبارات التي ساهمت في التنبيه",
        "external_title": "فحص إضافي عبر VirusTotal",
        "external_consent": "أوافق على إرسال الروابط الموجودة في الرسالة إلى VirusTotal لفحصها (قد تُشارك الروابط مع خدمة خارجية).",
        "external_not_checked": "لم يتم فحص الروابط عبر VirusTotal. يمكنك تفعيل الخيار أعلاه إذا رغبت.",
        "external_no_urls": "لا توجد روابط لإرسالها إلى VirusTotal.",
        "external_pending": "لم تكتمل نتيجة الفحص بعد. جرب لاحقاً.",
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
        "external_malicious": "Mal
