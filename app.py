
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
.stButton > button {
    width: 100%;
    border-radius: 14px;
    font-weight: bold;
    min-height: 45px;
}
.main-title {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. TRAINING DATA
#    Labels are generated from actual list lengths.
# =====================================================

english_safe = [
    "Hello, how are you?",
    "Good morning, have a nice day.",
    "I will see you tomorrow.",
    "Thank you for your help.",
    "Can you send me the lecture notes?",
    "I am studying for my exam.",
    "The meeting starts at ten.",
    "I arrived home safely.",
    "Please call me after class.",
    "I will send you the project file.",
    "Are you available this afternoon?",
    "Let's study together tomorrow.",
    "I am going to the library.",
    "The weather is nice today.",
    "I forgot my notebook at home.",
    "Please review my assignment.",
    "I will be late for the meeting.",
    "Happy birthday and best wishes.",
    "Can we talk after work?",
    "I received your email, thank you.",
    "The lecture was very useful.",
    "I am preparing dinner now.",
    "See you next week.",
    "Please confirm the meeting time.",
    "I need help with my homework.",
    "The report is ready for review.",
    "I am traveling to visit my family.",
    "Your appointment is tomorrow at nine.",
    "I will bring the documents to class.",
    "Thank you for explaining the lesson."
]

english_suspicious = [
    "Congratulations, you won one million dollars!",
    "Click here to claim your free cash prize.",
    "Your account will be blocked unless you act now.",
    "Send your password to verify your bank account.",
    "You have been selected for a free luxury vacation.",
    "Pay a small fee to receive your huge inheritance.",
    "Claim your lottery winnings immediately.",
    "Your compensation payment is waiting for you.",
    "Send your full name and home address to receive money.",
    "The United Nations has selected you for a cash reward.",
    "The World Bank will send you a compensation payment.",
    "Final reminder: claim your prize before it expires.",
    "You won a free phone, click this link now.",
    "Your bank account needs urgent verification.",
    "Invest ten dollars and receive five hundred today.",
    "Contact our agent to receive your inheritance.",
    "Your unclaimed funds will be cancelled today.",
    "Provide your date of birth and bank details.",
    "You are guaranteed to earn thousands without working.",
    "Your package is held, pay immediately using this link.",
    "Your tax refund is ready, provide your banking details.",
    "You must respond now or lose your reward.",
    "Congratulations, your free gift card is ready.",
    "Your loan is approved, pay the processing fee first.",
    "We need your personal information to release the funds.",
    "You have won a prize in a lottery you never entered.",
    "Verify your identity immediately to unlock your reward.",
    "Send your card number to receive your payment.",
    "Exclusive offer, earn money instantly with no effort.",
    "Your account has received a suspicious login alert. Click here."
]

arabic_safe = [
    "السلام عليكم، كيف حالك؟",
    "صباح الخير، أتمنى لك يوماً سعيداً.",
    "شكراً جزيلاً على مساعدتك.",
    "سأراك غداً في الجامعة.",
    "هل يمكنك إرسال المحاضرة لي؟",
    "أنا أدرس الآن استعداداً للامتحان.",
    "موعد الاجتماع الساعة العاشرة صباحاً.",
    "وصلت إلى المنزل بسلام.",
    "يرجى الاتصال بي بعد المحاضرة.",
    "سأرسل إليك ملف المشروع اليوم.",
    "هل أنت متاح بعد الظهر؟",
    "دعنا نذاكر معاً غداً.",
    "أنا في المكتبة حالياً.",
    "الطقس جميل اليوم.",
    "نسيت دفتري في المنزل.",
    "من فضلك راجع الواجب.",
    "سأتأخر قليلاً عن الاجتماع.",
    "كل عام وأنت بخير.",
    "هل يمكننا التحدث بعد العمل؟",
    "وصلني بريدك الإلكتروني، شكراً لك.",
    "كانت المحاضرة مفيدة جداً.",
    "أنا أعد العشاء الآن.",
    "أراك الأسبوع القادم.",
    "يرجى تأكيد موعد الاجتماع.",
    "أحتاج إلى مساعدة في الواجب.",
    "التقرير جاهز للمراجعة.",
    "سأسافر لزيارة عائلتي.",
    "موعدك غداً الساعة التاسعة.",
    "سأحضر المستندات إلى الجامعة.",
    "شكراً على شرح الدرس.",
    "هل استلمت الملف الذي أرسلته؟",
    "أتمنى لك التوفيق في الامتحان.",
    "سأعود إلى المنزل بعد انتهاء العمل.",
    "هل يمكنك توضيح هذه النقطة؟",
    "سأرسل التقرير إلى المشرف مساءً."
]

arabic_suspicious = [
    "مبروك لقد ربحت مليون دولار، اضغط لاستلام جائزتك.",
    "تم اختيارك للفوز بجائزة مالية ضخمة.",
    "حسابك البنكي سيتوقف ما لم تؤكد بياناتك فوراً.",
    "أرسل كلمة المرور للتحقق من حسابك.",
    "ادفع رسوماً بسيطة لاستلام ميراثك الكبير.",
    "استلم أرباح اليانصيب الآن قبل انتهاء المهلة.",
    "تعويضك المالي جاهز، تواصل معنا لاستلامه.",
    "أرسل اسمك الكامل وعنوان منزلك لاستلام الأموال.",
    "الأمم المتحدة اختارتك للحصول على مكافأة نقدية.",
    "البنك الدولي سيحول إليك مبلغ تعويض كبير.",
    "تذكير أخير، طالب بجائزتك قبل إلغائها.",
    "ربحت هاتفاً مجانياً، اضغط الرابط الآن.",
    "يلزم التحقق العاجل من حسابك المصرفي.",
    "استثمر مبلغاً صغيراً واربح أضعافه اليوم.",
    "تواصل مع وكيلنا لاستلام أموال الميراث.",
    "سيتم إلغاء أموالك غير المطالب بها اليوم.",
    "أرسل تاريخ ميلادك وبيانات حسابك البنكي.",
    "اربح آلاف الدولارات دون أي عمل.",
    "شحنتك محتجزة، ادفع الرسوم فوراً عبر الرابط.",
    "استردادك الضريبي جاهز، أرسل بياناتك المصرفية.",
    "رد الآن وإلا ستخسر مكافأتك.",
    "بطاقة الهدية المجانية جاهزة للاستلام.",
    "تمت الموافقة على قرضك، ادفع رسوم المعالجة أولاً.",
    "نحتاج معلوماتك الشخصية للإفراج عن الأموال.",
    "ربحت في يانصيب لم تشارك فيه.",
    "أكد هويتك فوراً لفتح المكافأة.",
    "أرسل رقم بطاقتك لاستلام الدفعة المالية.",
    "عرض حصري للربح الفوري دون مجهود.",
    "تم رصد دخول مشبوه لحسابك، اضغط هنا.",
    "أرسل رقم هاتفك وعنوانك للحصول على التعويض.",
    "أموالك جاهزة عبر بطاقة فيزا، تواصل مع المسؤول.",
    "يرجى الرد العاجل لتجنب إلغاء المبلغ المستحق.",
    "مبروك، تم تخصيص صندوق تعويضات بقيمة مليون دولار لك.",
    "أدخل بياناتك الشخصية لاستلام الجائزة قبل انتهاء الصلاحية.",
    "لديك أموال غير مطالب بها، تواصل معنا فوراً."
]


# Generate labels from real list lengths.
texts = (
    english_safe
    + english_suspicious
    + arabic_safe
    + arabic_suspicious
)

labels = (
    ["Safe Message"] * len(english_safe)
    + ["Suspicious Message"] * len(english_suspicious)
    + ["Safe Message"] * len(arabic_safe)
    + ["Suspicious Message"] * len(arabic_suspicious)
)


# =====================================================
# 3. MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def train_model():
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

    model.fit(texts, labels)
    return model


model = train_model()


# =====================================================
# 4. FRAUD INDICATOR ANALYZER
# =====================================================

FRAUD_PATTERNS = [
    (
        "وعد بمبلغ ضخم أو تعويض أو ميراث غير متوقع",
        r"مليون|مليار|مبلغ مالي ضخم|جائزة كبرى|"
        r"ربحت|لقد فزت|فزت بمبلغ|تعويضات|تعويض مالي|"
        r"صندوق التعويضات|الميراث|أموال غير مطالب بها|"
        r"استلام الأموال|مكافأة نقدية|أموالكم|"
        r"million dollars|won money|cash prize|"
        r"compensation fund|inheritance|lottery winnings|"
        r"unclaimed funds|claim your reward",
        2
    ),
    (
        "استخدام أسماء جهات رسمية أو مصرفية لإضفاء المصداقية",
        r"البنك الدولي|الأمم المتحدة|الاتحاد الأوروبي|"
        r"رئيس البنك|الأمين العام|نائبة الأمين العام|"
        r"الرئيس التنفيذي|البنك المركزي|"
        r"world bank|united nations|european union|"
        r"chief executive|secretary general",
        1
    ),
    (
        "طلب بيانات شخصية أو مالية حساسة",
        r"اسمكم الكامل|اسمك الكامل|عنوان منزلكم|عنوان منزلك|"
        r"تاريخ ميلادكم|تاريخ ميلادك|جنسكم|مهنتكم|"
        r"رقم الهاتف المباشر|بريدكم الإلكتروني الشخصي|"
        r"تفاصيل التسليم|بياناتك البنكية|بيانات حسابك|"
        r"كلمة المرور|رقم البطاقة|"
        r"full name|home address|date of birth|"
        r"bank details|personal information|password|"
        r"card number",
        2
    ),
    (
        "ضغط للاستجابة بسرعة أو التهديد بإلغاء الأموال",
        r"عاجل|الرد العاجل|التذكير الأخير|"
        r"قبل انتهاء|إلغاء أموالكم|إلغاء المبلغ|"
        r"أموال غير مطالب بها|فوراً|فورا|دون تأخير|"
        r"خلال اليوم|على وجه السرعة|"
        r"urgent|act now|final reminder|immediately|"
        r"unclaimed funds|before it expires",
        1
    ),
    (
        "توجيه إلى بريد إلكتروني أو رابط للتواصل",
        r"https?://|www\.|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|"
        r"اضغط على الرابط|اضغط هنا|تواصلوا مع|"
        r"يرجى التواصل|أرسل بياناتك|"
        r"click here|contact us|reply immediately",
        1
    ),
    (
        "وعد بأرباح مضمونة أو طلب رسوم مقدمة",
        r"اربح آلاف|اربح المال بسرعة|مضمون|"
        r"دون أي مجهود|بدون مجهود|رسوم المعالجة|"
        r"ادفع رسوماً|ادفع رسوم|ضاعف أموالك|"
        r"guaranteed profit|earn money fast|"
        r"processing fee|no effort|double your money",
        1
    )
]


def analyze_fraud_risk(message):
    score = 0
    reasons = []
    normalized = message.casefold()

    for reason, pattern, weight in FRAUD_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            score += weight
            reasons.append(reason)

    # Multiple independent indicators are required.
    suspicious = score >= 3 and len(reasons) >= 2

    return suspicious, score, reasons


# =====================================================
# 5. SESSION STATE
# =====================================================

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "result" not in st.session_state:
    st.session_state.result = None


def clear_all():
    st.session_state.user_input = ""
    st.session_state.result = None


# =====================================================
# 6. USER INTERFACE
# =====================================================

st.markdown(
    "<div class='main-title'>"
    "<h1>🛡️ نظام فحص وتصنيف الرسائل الذكي</h1>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "### Message Safety Detection System"
)

st.write(
    "يفحص النظام الرسائل العربية والإنجليزية باستخدام "
    "تعلم الآلة وقواعد لاكتشاف مؤشرات الاحتيال."
)

st.info(
    "تنبيه: التصنيف آلي وتقديري. لا تشارك بياناتك الشخصية "
    "ولا تضغط الروابط المشبوهة بناءً على نتيجة هذا الفحص وحدها."
)


# =====================================================
# 7. MESSAGE INPUT
# =====================================================

st.text_area(
    "✍️ أدخل الرسالة التي تريد فحصها:",
    key="user_input",
    height=220,
    placeholder="الصق الرسالة كاملة هنا، حتى لو كانت طويلة..."
)


# =====================================================
# 8. BUTTONS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    check_btn = st.button(
        "🔍 فحص الرسالة",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ مسح الكل",
        on_click=clear_all,
        use_container_width=True
    )


# =====================================================
# 9. CLASSIFICATION
# =====================================================

if check_btn:
    message = st.session_state.user_input.strip()

    if not message:
        st.warning("الرجاء كتابة رسالة أولاً.")
        st.session_state.result = None

    else:
        ml_prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_

        ml_confidence = float(
            probabilities[list(classes).index(ml_prediction)]
        )

        risk_suspicious, risk_score, risk_reasons = (
            analyze_fraud_risk(message)
        )

        final_suspicious = (
            ml_prediction == "Suspicious Message"
            or risk_suspicious
        )

        st.session_state.result = {
            "message": message,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "risk_suspicious": risk_suspicious,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "final_suspicious": final_suspicious
        }


# =====================================================
# 10. DISPLAY RESULTS
# =====================================================

result = st.session_state.result

if result:
    st.markdown("---")
    st.subheader("📊 نتيجة التحليل")

    if result["final_suspicious"]:
        st.error(
            "🚨 تحذير: توجد مؤشرات على أن الرسالة مشبوهة."
        )

        st.markdown("#### لماذا ظهرت هذه النتيجة؟")

        if result["risk_reasons"]:
            for reason in result["risk_reasons"]:
                st.write(f"⚠️ {reason}")
        else:
            st.write(
                "نموذج تعلم الآلة صنف الرسالة على أنها مشبوهة، "
                "لكن القواعد لم تحدد مؤشرات إضافية."
            )

        st.markdown("#### 🛡️ نصائح السلامة")

        st.write("- لا ترسل معلوماتك الشخصية أو البنكية.")
        st.write("- لا تدفع رسوماً لاستلام جائزة غير متوقعة.")
        st.write("- تحقق من الجهة عبر موقعها الرسمي المستقل.")
        st.write("- لا تستخدم بريد التواصل الموجود في الرسالة وحده للتحقق.")

    else:
        st.success(
            "✅ لم يكتشف النظام مؤشرات كافية لتصنيف الرسالة كمشبوهة."
        )

        st.warning(
            "عدم اكتشاف مؤشرات لا يعني أن الرسالة آمنة؛ "
            "قد تحتوي على أساليب احتيال لم يتعلمها النظام."
        )

    st.markdown("#### 🤖 نتيجة نموذج تعلم الآلة")

    st.write(
        "تصنيف النموذج:",
        result["ml_prediction"]
    )

    st.write(
        f"ثقة النموذج في تصنيفه: "
        f"{result['ml_confidence'] * 100:.2f}%"
    )

    st.caption(
        "هذه النسبة ليست احتمالاً موثوقاً بأن الرسالة آمنة أو احتيالية، "
        "خصوصاً لأن بيانات التدريب محدودة."
    )

    st.markdown("#### 🔎 تحليل المؤشرات")

    st.write(
        f"عدد فئات المؤشرات المكتشفة: "
        f"{len(result['risk_reasons'])}"
    )

    st.write(
        f"مجموع نقاط القواعد: {result['risk_score']}"
    )


# =====================================================
# 11. ABOUT THE SYSTEM
# =====================================================

with st.expander("ℹ️ معلومات عن النظام"):
    st.write("اسم النظام: Message Safety Detector")
    st.write("النموذج: Multinomial Naive Bayes")
    st.write("تحويل النص: TF-IDF Character N-Grams")
    st.write("اللغات المدعومة: العربية والإنجليزية")
    st.write(f"عدد أمثلة التدريب: {len(texts)}")
    st.write("التصنيفات: Safe Message / Suspicious Message")
    st.write(
        "يستخدم النظام نموذج تعلم آلة مع قواعد مستقلة "
        "لرصد مؤشرات الاحتيال."
    )
    st.write(
        "البيانات الحالية تجريبية؛ يجب تقييم النظام على "
        "مجموعة رسائل مستقلة قبل الاعتماد عليه."
    )
