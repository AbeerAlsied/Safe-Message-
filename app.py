import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================================================
# 1. إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered"
)


# =========================================================
# 2. قاعدة بيانات التدريب
# =========================================================

texts = [

    # ==================== English Safe Messages ====================

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

    # Additional English safe messages
    *[f"Hope you are doing well today {i}" for i in range(61, 301)],


    # ==================== English Suspicious Messages ====================

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
    "Click here for exclusive adult content",
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

    *[f"Urgent alert click link to claim reward {i}" for i in range(71, 301)],


    # ==================== Arabic Safe Messages ====================

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

    *[f"رسالة محادثة طبيعية آمنة رقم {i}" for i in range(57, 301)],


    # ==================== Arabic Suspicious Messages ====================

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
    "احصل على بطاقة هدية أمازون",
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

    *[f"تنبيه احتيال رسالة مشبوهة رقم {i}" for i in range(67, 301)]
]


# =========================================================
# 3. التصنيفات
# =========================================================

# الرسائل الفعلية الموجودة في كل قسم
english_safe_count = 300
english_suspicious_count = 300
arabic_safe_count = 300

# كل الرسائل المتبقية تعتبر Arabic Suspicious
arabic_suspicious_count = len(texts) - (
    english_safe_count +
    english_suspicious_count +
    arabic_safe_count
)

labels = (
    ["Safe Message"] * english_safe_count +
    ["Suspicious Message"] * english_suspicious_count +
    ["Safe Message"] * arabic_safe_count +
    ["Suspicious Message"] * arabic_suspicious_count
)


# =========================================================
# 4. التحقق من عدد البيانات
# =========================================================

if len(texts) != len(labels):

    st.error(
        f"خطأ في البيانات: عدد الرسائل = {len(texts)} "
        f"بينما عدد التصنيفات = {len(labels)}"
    )

    st.stop()


# =========================================================
# 5. تدريب نموذج الذكاء الاصطناعي
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
# 6. واجهة التطبيق
# =========================================================

st.title("🛡️ نظام فحص وتصنيف الرسائل الذكي")

st.markdown(
    "### Message Safety Detection System (Bilingual)"
)

st.write(
    "هذا النظام يدعم اللغتين العربية والإنجليزية، "
    "ويستخدم تقنيات تعلم الآلة لتحليل الرسائل "
    "وتصنيفها إلى رسائل آمنة أو مشبوهة."
)


# =========================================================
# 7. إدخال الرسالة
# =========================================================

user_message = st.text_area(
    "✍️ أدخل نص الرسالة هنا (عربي أو إنجليزي):",
    height=130,
    placeholder="اكتب الرسالة هنا..."
)


# =========================================================
# 8. الأزرار
# =========================================================

col1, col2 = st.columns(2)

with col1:

    check_btn = st.button(
        "🔍 فحص الرسالة / Check",
        use_container_width=True
    )

with col2:

    clear_btn = st.button(
        "🗑️ مسح / Clear",
        use_container_width=True
    )


# =========================================================
# 9. مسح الرسالة
# =========================================================

if clear_btn:

    st.rerun()


# =========================================================
# 10. فحص الرسالة
# =========================================================

if check_btn:

    if not user_message.strip():

        st.warning(
            "⚠️ الرجاء كتابة رسالة أولاً ليتمكن النظام من فحصها!"
        )

    else:

        # تحويل الرسالة إلى تمثيل رقمي
        msg_vector = vectorizer.transform([user_message])

        # التنبؤ
        prediction = model.predict(msg_vector)[0]

        # حساب الاحتمالية
        probabilities = model.predict_proba(msg_vector)[0]

        max_probability = probabilities.max()

        st.markdown("---")

        st.subheader("📊 نتيجة التحليل / Result:")


        # =================================================
        # رسالة مشبوهة
        # =================================================

        if prediction == "Suspicious Message":

            st.error(
                "🚨 تحذير: الرسالة تبدو مشبوهة "
                "(Suspicious Message)"
            )

            st.markdown(
                """
                **السبب (Reason):**

                تحتوي الرسالة على أنماط أو كلمات
                تشبه الرسائل الاحتيالية أو الإعلانية المزعجة.

                **التوصية (Advice):**

                - لا تضغط على الروابط غير الموثوقة.
                - لا تشارك كلمات المرور أو المعلومات الشخصية.
                - تحقق من مصدر الرسالة قبل اتخاذ أي إجراء.
                """
            )


        # =================================================
        # رسالة آمنة
        # =================================================

        else:

            st.success(
                "✅ الرسالة تبدو آمنة "
                "(Safe Message)"
            )

            st.markdown(
                """
                **السبب (Reason):**

                الرسالة تشبه نمط المحادثات الطبيعية
                الموجودة في بيانات التدريب.

                **التوصية (Advice):**

                يمكنك التعامل معها بشكل طبيعي، مع
                الاستمرار في الحذر من الروابط والطلبات
                غير المتوقعة.
                """
            )


        # =================================================
        # نسبة ثقة النموذج
        # =================================================

        st.info(
            f"🤖 درجة ثقة النموذج التقريبية: "
            f"{max_probability * 100:.2f}%"
        )


# =========================================================
# 11. معلومات عن النظام
# =========================================================

with st.expander("ℹ️ عن النظام"):

    st.write(
        f"""
        **نوع النموذج:** Multinomial Naive Bayes

        **طريقة تحويل النص:** TF-IDF Character N-Grams

        **عدد رسائل التدريب:** {len(texts)}

        **اللغات المدعومة:** العربية والإنجليزية

        **التصنيفات:**
        - Safe Message
        - Suspicious Message
        """
    )
