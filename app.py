import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import qrcode
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="centered",
)

# -------------------- Neutral Arabic / English interface --------------------
TEXT = {
    "ar": {
        "app_name": "CyberShield AI",
        "tagline": "فحص ذكي للرسائل والروابط المشبوهة",
        "language": "لغة الواجهة",
        "home": "الرئيسية",
        "scanner": "فحص الرسائل",
        "history": "السجل",
        "about": "حول التطبيق",
        "home_title": "مرحباً بك في CyberShield AI",
        "home_desc": "أداة مساعدة لتحليل الرسائل والروابط واكتشاف مؤشرات الاحتيال.",
        "how_title": "كيف يعمل التطبيق؟",
        "how_1": "1. أدخل نص الرسالة أو الرابط المراد فحصه.",
        "how_2": "2. اضغط على زر الفحص لبدء التحليل.",
        "how_3": "3. راجع النتيجة والمؤشرات والتوصيات قبل اتخاذ أي إجراء.",
        "scanner_title": "فحص رسالة أو رابط",
        "input_label": "نص الرسالة أو الرابط",
        "placeholder": "الصق نص الرسالة أو الرابط هنا...",
        "scan": "فحص الآن",
        "clear": "مسح النص",
        "empty": "يرجى إدخال رسالة أو رابط أولاً.",
        "result": "نتيجة التحليل",
        "safe": "مؤشرات الخطورة منخفضة",
        "suspicious": "رسالة أو رابط مشبوه",
        "high": "خطورة مرتفعة",
        "medium": "خطورة متوسطة",
        "low": "خطورة منخفضة",
        "score": "مؤشر الخطورة",
        "recommendation": "التوصية",
        "safe_advice": "لا تظهر مؤشرات واضحة على الاحتيال، لكن تحقق من المصدر قبل فتح الروابط أو مشاركة المعلومات.",
        "suspicious_advice": "تجنب فتح الروابط أو إرسال الأموال أو مشاركة كلمات المرور والبيانات الشخصية حتى يتم التحقق من المصدر.",
        "signals": "المؤشرات المكتشفة",
        "no_signals": "لم تُكتشف مؤشرات واضحة وفق القواعد الحالية.",
        "url_title": "تحليل الروابط",
        "url_found": "الروابط الموجودة في النص",
        "vt_title": "فحص الرابط عبر VirusTotal",
        "vt_not_configured": "لم يتم إعداد مفتاح VirusTotal. يمكن إضافته في Streamlit Secrets باسم VT_API_KEY.",
        "vt_none": "لا يوجد رابط في النص لفحصه.",
        "vt_check": "فحص الرابط عبر VirusTotal",
        "vt_wait": "جارٍ إرسال الرابط للفحص...",
        "vt_error": "تعذر إكمال فحص VirusTotal حالياً. يمكن إعادة المحاولة لاحقاً.",
        "vt_pending": "تم إرسال الرابط للفحص، لكن النتيجة لم تجهز بعد. حاول مرة أخرى بعد قليل.",
        "vt_stats": "نتائج VirusTotal",
        "vt_malicious": "مصادر صنّفت الرابط على أنه ضار",
        "vt_suspicious": "مصادر صنّفت الرابط على أنه مشبوه",
        "vt_harmless": "مصادر صنّفت الرابط على أنه سليم",
        "vt_undetected": "مصادر لم تصدر حكماً",
        "history_title": "سجل عمليات الفحص",
        "no_history": "لا توجد عمليات فحص في هذه الجلسة حتى الآن.",
        "download": "تنزيل السجل بصيغة CSV",
        "clear_history": "مسح السجل",
        "about_title": "حول CyberShield AI",
        "about_text": "مشروع توعوي يستخدم نموذج تعلم آلي وقواعد إرشادية لفحص النصوص والروابط. النتائج تقديرية وليست ضماناً لسلامة الرسائل أو المواقع.",
        "privacy": "تجنب إدخال كلمات المرور أو المعلومات الشخصية الحساسة في أي أداة فحص.",
        "qr_title": "إنشاء QR للرابط",
        "qr_label": "أدخل رابطاً لإنشاء رمز QR",
        "qr_button": "إنشاء رمز QR",
        "qr_empty": "أدخل رابطاً صالحاً أولاً.",
        "disclaimer": "تنبيه: قد يخطئ النموذج في التصنيف. لا تعتمد على النتيجة وحدها لاتخاذ قرارات أمنية.",
        "time": "الوقت",
        "message": "النص المفحوص",
        "classification": "التصنيف",
        "no_url": "لم يتم العثور على رابط في النص.",
    },
    "en": {
        "app_name": "CyberShield AI",
        "tagline": "Smart screening for suspicious messages and links",
        "language": "Interface language",
        "home": "Home",
        "scanner": "Scanner",
        "history": "History",
        "about": "About",
        "home_title": "Welcome to CyberShield AI",
        "home_desc": "An assistant tool for analyzing messages and links for potential scam indicators.",
        "how_title": "How it works",
        "how_1": "1. Enter the message text or link you want to check.",
        "how_2": "2. Select Scan to start the analysis.",
        "how_3": "3. Review the result and indicators before taking action.",
        "scanner_title": "Scan a message or link",
        "input_label": "Message text or link",
        "placeholder": "Paste the message or link here...",
        "scan": "Scan now",
        "clear": "Clear text",
        "empty": "Please enter a message or link first.",
        "result": "Analysis result",
        "safe": "Low risk indicators",
        "suspicious": "Suspicious message or link",
        "high": "High risk",
        "medium": "Medium risk",
        "low": "Low risk",
        "score": "Risk indicator",
        "recommendation": "Recommendation",
        "safe_advice": "No clear scam indicators were detected, but verify the source before opening links or sharing information.",
        "suspicious_advice": "Avoid opening links, sending money, or sharing passwords and personal data until the source is verified.",
        "signals": "Detected indicators",
        "no_signals": "No clear indicators were detected by the current rules.",
        "url_title": "Link analysis",
        "url_found": "Links found in the text",
        "vt_title": "VirusTotal link check",
        "vt_not_configured": "VirusTotal is not configured. Add VT_API_KEY in Streamlit Secrets.",
        "vt_none": "There is no link in the text to check.",
        "vt_check": "Check link with VirusTotal",
        "vt_wait": "Submitting the URL for analysis...",
        "vt_error": "VirusTotal could not complete the check right now. Try again later.",
        "vt_pending": "The URL was submitted, but the result is not ready yet. Try again shortly.",
        "vt_stats": "VirusTotal results",
        "vt_malicious": "Sources marked the URL as malicious",
        "vt_suspicious": "Sources marked the URL as suspicious",
        "vt_harmless": "Sources marked the URL as harmless",
        "vt_undetected": "Sources with no verdict",
        "history_title": "Scan history",
        "no_history": "No scans have been recorded in this session yet.",
        "download": "Download history as CSV",
        "clear_history": "Clear history",
        "about_title": "About CyberShield AI",
        "about_text": "An awareness project using a machine-learning model and heuristic rules to screen text and URLs. Results are estimates, not guarantees of message or website safety.",
        "privacy": "Avoid entering passwords or sensitive personal information into any scanning tool.",
        "qr_title": "Create a QR code for a link",
        "qr_label": "Enter a URL to create a QR code",
        "qr_button": "Create QR code",
        "qr_empty": "Enter a valid URL first.",
        "disclaimer": "Notice: The model can misclassify content. Do not rely on this result alone for security decisions.",
        "time": "Time",
        "message": "Scanned text",
        "classification": "Classification",
        "no_url": "No URL was found in the text.",
    },
}

if "language" not in st.session_state:
    st.session_state.language = "ar"
if "history" not in st.session_state:
    st.session_state.history = []

lang_choice = st.sidebar.selectbox(
    "لغة الواجهة / Interface language",
    options=["العربية", "English"],
    index=0 if st.session_state.language == "ar" else 1,
)
st.session_state.language = "ar" if lang_choice == "العربية" else "en"
T = TEXT[st.session_state.language]

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 1.8rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25);
           background: linear-gradient(120deg, rgba(70,90,180,.12), rgba(20,180,160,.10));}
    .small-note {font-size: .9rem; opacity: .85;}
    div.stButton > button {border-radius: 10px; min-height: 2.7rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='hero'><h1>🛡️ {T['app_name']}</h1><p>{T['tagline']}</p></div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [T["home"], T["scanner"], T["history"], T["about"]],
    label_visibility="collapsed",
)

# -------------------- Lightweight ML model --------------------
safe_samples = [
    "Your appointment is confirmed for tomorrow at 10 AM.",
    "Thank you for attending the meeting. The notes are attached.",
    "Can we move our study session to Friday?",
    "تم تأكيد موعد المحاضرة غداً الساعة العاشرة.",
    "شكراً على حضور الاجتماع، أرفقنا ملخص النقاش.",
    "هل يمكن تأجيل موعد الدراسة إلى يوم الجمعة؟",
    "Your order has shipped. Track it through the official store app.",
    "تم استلام طلبك، ويمكن متابعة الحالة من التطبيق الرسمي.",
]
suspicious_samples = [
    "Congratulations you won a prize click this link now and claim your reward.",
    "Urgent: your bank account will be blocked. Verify your password immediately.",
    "Send your password and verification code to receive compensation.",
    "You have been selected for a grant. Pay a processing fee to release the money.",
    "عاجل تم اختيارك لاستلام تعويض مالي أرسل بياناتك الشخصية ورقم الحساب.",
    "مبروك ربحت جائزة اضغط الرابط الآن لاستلام الهدية.",
    "سيتم إغلاق حسابك البنكي أرسل كلمة المرور ورمز التحقق فوراً.",
    "ادفع رسوم المعالجة لاستلام المنحة المالية.",
]
ml_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
    ("classifier", MultinomialNB()),
])
ml_model.fit(safe_samples + suspicious_samples, [0] * len(safe_samples) + [1] * len(suspicious_samples))

SUSPICIOUS_TERMS = [
    "password", "verify your account", "urgent", "claim your prize", "winner",
    "send money", "processing fee", "bank account", "gift card", "compensation",
    "wire transfer", "click here", "verification code", "crypto", "investment guaranteed",
    "كلمة المرور", "رمز التحقق", "بياناتك الشخصية", "تعويض مالي", "جائزة",
    "ارسل بياناتك", "أرسل بياناتك", "رسوم المعالجة", "حسابك البنكي", "اضغط الرابط",
    "ربحت", "فزت", "عاجل", "تحويل مالي", "منحة مالية", "استلام الأموال",
]
URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+|www\.[^\s<>\]\[\"']+", re.IGNORECASE)

def extract_urls(text):
    return URL_RE.findall(text or "")

def inspect_url(url):
    normalized = url.strip().rstrip(".,;!?)")
    if normalized.lower().startswith("www."):
        normalized = "http://" + normalized
    try:
        from urllib.parse import urlparse
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        reasons = []
        if not parsed.scheme.startswith("http"):
            reasons.append("Non-HTTP URL scheme")
        if "@" in host:
            reasons.append("The URL contains an @ symbol in its host")
        if host and re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host.split(":")[0]):
            reasons.append("The host uses an IP address")
        if host.count("-") >= 3:
            reasons.append("The domain contains many hyphens")
        if len(normalized) > 120:
            reasons.append("The URL is unusually long")
        if any(token in normalized.lower() for token in ["login", "verify", "free-gift", "claim", "wallet", "password"]):
            reasons.append("The URL path contains commonly abused terms")
        if not host or "." not in host:
            reasons.append("The domain format looks unusual")
        return {"url": normalized, "host": host or "Unknown", "reasons": reasons}
    except Exception:
        return {"url": normalized, "host": "Unknown", "reasons": ["Could not parse this URL"]}

def analyze_message(message):
    cleaned = message.strip()
    ml_probability = float(ml_model.predict_proba([cleaned])[0][1])
    lowered = cleaned.lower()
    signals = []
    for term in SUSPICIOUS_TERMS:
        if term.lower() in lowered and term not in signals:
            signals.append(term)
    urls = extract_urls(cleaned)
    url_details = [inspect_url(url) for url in urls]
    for item in url_details:
        signals.extend(item["reasons"])
    score = int(ml_probability * 35)
    score += min(len(signals) * 12, 48)
    if re.search(r"\b(password|pin|otp|verification code|كلمة المرور|رمز التحقق)\b", lowered):
        score += 18
    score = min(score, 100)
    level = "high" if score >= 65 else "medium" if score >= 35 else "low"
    suspicious = score >= 35 or ml_probability >= 0.70
    return {
        "score": score,
        "level": level,
        "suspicious": suspicious,
        "signals": list(dict.fromkeys(signals)),
        "urls": url_details,
        "ml_probability": ml_probability,
    }

def virustotal_check(url):
    api_key = None
    try:
        api_key = st.secrets.get("VT_API_KEY", None)
    except Exception:
        api_key = os.getenv("VT_API_KEY")
    if not api_key:
        return {"status": "not_configured"}
    try:
        headers = {"x-apikey": api_key}
        submit = requests.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data={"url": url},
            timeout=20,
        )
        if submit.status_code not in (200, 201):
            return {"status": "error", "detail": f"HTTP {submit.status_code}"}
        analysis_id = submit.json().get("data", {}).get("id")
        if not analysis_id:
            return {"status": "error"}
        time.sleep(2)
        result = requests.get(
            f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
            headers=headers,
            timeout=20,
        )
        if result.status_code != 200:
            return {"status": "pending"}
        attrs = result.json().get("data", {}).get("attributes", {})
        stats = attrs.get("stats", {})
        if not stats:
            return {"status": "pending"}
        return {
            "status": "ok",
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0),
            "undetected": stats.get("undetected", 0),
        }
    except requests.RequestException:
        return {"status": "error"}
    except Exception:
        return {"status": "error"}

def record_scan(message, result):
    st.session_state.history.insert(0, {
        T["time"]: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        T["message"]: message[:500],
        T["classification"]: T["suspicious"] if result["suspicious"] else T["safe"],
        T["score"]: result["score"],
    })

if page == T["home"]:
    st.subheader(T["home_title"])
    st.write(T["home_desc"])
    st.markdown(f"### {T['how_title']}")
    st.write(T["how_1"])
    st.write(T["how_2"])
    st.write(T["how_3"])
    st.info(T["disclaimer"])
    st.caption(T["privacy"])

elif page == T["scanner"]:
    st.subheader(T["scanner_title"])
    message = st.text_area(T["input_label"], placeholder=T["placeholder"], height=180, key="scan_input")
    col1, col2 = st.columns(2)
    with col1:
        scan_clicked = st.button(T["scan"], type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button(T["clear"], use_container_width=True)
    if clear_clicked:
        st.session_state.scan_input = ""
        st.rerun()

    if scan_clicked:
        if not message.strip():
            st.warning(T["empty"])
        else:
            result = analyze_message(message)
            record_scan(message, result)
            st.markdown("---")
            st.subheader(T["result"])
            if result["level"] == "high":
                st.error(f"🚨 {T['high']}")
            elif result["suspicious"]:
                st.warning(f"⚠️ {T['suspicious']}")
            else:
                st.success(f"✅ {T['safe']}")
            st.metric(T["score"], f"{result['score']} / 100")
            st.progress(result["score"] / 100)
            st.markdown(f"**{T['recommendation']}**")
            st.write(T["suspicious_advice"] if result["suspicious"] else T["safe_advice"])

            with st.expander(T["signals"], expanded=True):
                if result["signals"]:
                    for signal in result["signals"]:
                        st.write(f"• {signal}")
                else:
                    st.write(T["no_signals"])

            st.markdown(f"### 🔗 {T['url_title']}")
            if result["urls"]:
                for item in result["urls"]:
                    st.code(item["url"])
                    st.caption(f"Domain: {item['host']}" if st.session_state.language == "en" else f"النطاق: {item['host']}")
                    if item["reasons"]:
                        for reason in item["reasons"]:
                            st.write(f"• {reason}")
                st.markdown(f"### 🧪 {T['vt_title']}")
                if st.button(T["vt_check"], key="vt_check_button"):
                    target_url = result["urls"][0]["url"]
                    with st.spinner(T["vt_wait"]):
                        vt = virustotal_check(target_url)
                    if vt["status"] == "not_configured":
                        st.info(T["vt_not_configured"])
                    elif vt["status"] == "pending":
                        st.info(T["vt_pending"])
                    elif vt["status"] != "ok":
                        st.warning(T["vt_error"])
                    else:
                        st.write(f"**{T['vt_malicious']}:** {vt['malicious']}")
                        st.write(f"**{T['vt_suspicious']}:** {vt['suspicious']}")
                        st.write(f"**{T['vt_harmless']}:** {vt['harmless']}")
                        st.write(f"**{T['vt_undetected']}:** {vt['undetected']}")
            else:
                st.write(T["no_url"])
            st.info(T["disclaimer"])

    st.markdown("---")
    st.markdown(f"### {T['qr_title']}")
    qr_url = st.text_input(T["qr_label"], key="qr_input")
    if st.button(T["qr_button"]):
        if not re.match(r"^https?://", qr_url.strip(), re.IGNORECASE):
            st.warning(T["qr_empty"])
        else:
            qr = qrcode.make(qr_url.strip())
            buffer = BytesIO()
            qr.save(buffer, format="PNG")
            st.image(buffer.getvalue(), caption="QR Code")
            st.download_button(
                "Download QR" if st.session_state.language == "en" else "تنزيل رمز QR",
                data=buffer.getvalue(),
                file_name="cybershield_qr.png",
                mime="image/png",
            )

elif page == T["history"]:
    st.subheader(T["history_title"])
    if not st.session_state.history:
        st.info(T["no_history"])
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            T["download"],
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="cybershield_history.csv",
            mime="text/csv",
        )
        if st.button(T["clear_history"]):
            st.session_state.history = []
            st.rerun()

elif page == T["about"]:
    st.subheader(T["about_title"])
    st.write(T["about_text"])
    st.info(T["privacy"])
    st.warning(T["disclaimer"])

st.markdown("---")
st.caption("CyberShield AI • Educational prototype")
