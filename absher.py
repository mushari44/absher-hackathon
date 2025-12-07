#streamlit run abcher4.py
import streamlit as st
from datetime import datetime
import pandas as pd
import pyttsx3
import requests
import whisper
from streamlit_mic_recorder import mic_recorder
import time

# =====================================================
# مكتبات نموذج NLU المُخصص
# =====================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =====================================================
# إعداد صفحة ستريملت + تنسيقات (الخلفية والأيقونات الرهيبة)
# =====================================================
st.set_page_config(page_title="🤖 أبشر مساعد 🇸🇦", layout="wide")

st.markdown("""
    <style>
    /* ألوان أبشر الرسمية */
    :root {
        --absher-green: #006C3C; /* أخضر غامق (لون الهوية) */
        --absher-light: #eaf4f0; /* خلفية فاتحة صديقة للعين */
        --absher-dark: #004d2a;
        --absher-accent: #c7ddd2; /* لون الحدود الخفيفة */
    }
    h1 {
        color: var(--absher-green);
        border-bottom: 3px solid var(--absher-green);
        padding-bottom: 5px;
        margin-top: 0;
        /* تنسيق الأيقونة المدمجة في العنوان */
        font-size: 2.5rem; 
    }
    .stButton>button {
        background-color: var(--absher-green);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: var(--absher-dark);
    }
    /* تنسيق خاص لكروت النتائج باللون الأخضر */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: var(--absher-light);
        border: 2px solid var(--absher-green);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        font-size: 1.1rem;
    }
    /* استبدال الـ Blue Box (st.info) بالأخضر الغامق */
    .stAlert.info {
        background-color: var(--absher-light); 
        border-left: 6px solid var(--absher-green);
        color: var(--absher-dark); /* لون الخط داخل البوكس */
    }
    /* شريط جانبي بلون أخضر فاتح */
    .css-1d391kg {
        background-color: var(--absher-light) !important;
    }
    /* الـ Alert والأيقونات الأخرى */
    .stAlert {
        border-left: 6px solid var(--absher-dark);
        background-color: var(--absher-light);
    }
    .mic-wrapper {
        padding: 16px;
        border-radius: 20px;
        animation: mic-pulse 1.4s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# 1) ERD: USERS (المواطن، المقيم، الأجنبي)
# =====================================================

USERS = {
    "Mohamed": {  # المواطن (محمد الدوسري)
        "user_type": "المواطن",
        "user_id": "1001",
        "national_id": "1012345678",
        "name": "محمد الدوسري",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        "license_status": "Valid",
        "violations": 0,
    },
    "Ahmed": {  # المقيم (أحمد الرفاعي)
        "user_type": "المقيم",
        "user_id": "1002",
        "national_id": "2098765432",
        "name": "أحمد الرفاعي",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=13)).strftime("%Y-%m-%d"),  # تنتهي خلال 13 يوماً
        "license_status": "Expired Medical",
        "violations": 500,
    },
    "Alex": {  # الأجنبي (Alex Smith)
        "user_type": "الأجنبي (غير عربي)",
        "user_id": "1003",
        "national_id": "3012345678",
        "name": "Alex Smith",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        "license_status": "Valid",
        "violations": 0,
    },
}

SERVICES = {
    "ID_RENEWAL": {"service_id": "2001", "name": "تجديد الهوية/الإقامة", "category": "Identity API"},
    "ID_STATUS": {"service_id": "2002", "name": "الاستعلام عن الصلاحية", "category": "Identity API"},
    "DRIVER_LICENSE_RENEWAL": {"service_id": "3001", "name": "تجديد رخصة القيادة", "category": "Vehicle API"},
    "PASSPORT_RENEWAL": {"service_id": "4001", "name": "تجديد جواز السفر (للمواطن)", "category": "Passport API"},
    "APPOINTMENT_BOOK": {"service_id": "5001", "name": "حجز موعد", "category": "Appointment API"},
}
SERVICE_BY_ID = {s["service_id"]: s for s in SERVICES.values()}

REQUESTS: list[dict] = []


# دوال مساعدة لحالة الجلسة والطلبات
def create_request(user_key: str, service_key: str, status: str = "initiated") -> dict:
    request = {
        "request_id": f"R-{len(REQUESTS) + 1:04d}",
        "user_id": USERS[user_key]["user_id"],
        "service_id": SERVICES[service_key]["service_id"],
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    REQUESTS.append(request)
    return request


if "current_user" not in st.session_state: st.session_state.current_user = "Mohamed"
if "just_reset" not in st.session_state: st.session_state.just_reset = False
if "last_visual" not in st.session_state: st.session_state.last_visual = ""
if "last_tts" not in st.session_state: st.session_state.last_tts = ""

# =====================================================
# 2) بيانات التدريب (تضمين كلمات إنجليزية لدعم غير الناطقين)
# =====================================================

NLU_TRAINING_DATA = {
    'text': [
        "أجدد رخصتي", "رخصتي منتهية", "أبي تجديد قيادة", "جدد لي الرخصة",
        "جوازي خلص", "أحتاج تجديد جواز السفر", "متى أقدر أجدد الجواز؟",
        "احجز لي موعد في الأحوال", "أبغى موعد جوازات", "كيف أضبط موعد؟",
        "كم باقي على صلاحية إقامتي؟", "متى تنتهي هويتي الوطنية؟", "الهوية سارية متى تنتهي؟",
        "أجدد الهوية", "أبي أجدد الإقامة", "الإقامة خلصت", "الهوية خلصت",
        "كم مخالفة علي؟", "وش وضع سيارتي؟",
        "غير المستخدم إلى أحمد الرفاعي", "حولني لمحمد الدوسري", "change user to alex",
        "أهلاً وسهلاً", "مرحبا", "please help me with my iqama", "I need service", "I need help",
    ],
    'intent': [
        "تجديد_رخصة", "تجديد_رخصة", "تجديد_رخصة", "تجديد_رخصة",
        "تجديد_جواز", "تجديد_جواز", "تجديد_جواز",
        "حجز_موعد", "حجز_موعد", "حجز_موعد",
        "استعلام_صلاحية", "استعلام_صلاحية", "استعلام_صلاحية",
        "تجديد_هوية_اقامة", "تجديد_هوية_اقامة", "تجديد_هوية_اقامة", "تجديد_هوية_اقامة",
        "استعلام_مخالفات", "استعلام_مخالفات",
        "تغيير_مستخدم", "تغيير_مستخدم", "تغيير_مستخدم",
        "ترحيب_وافتتاح", "ترحيب_وافتتاح", "طلب_مساعدة_غامض", "طلب_مساعدة_غامض", "طلب_مساعدة_غامض",
    ]
}


# تدريب النموذج وتخزين Whisper
@st.cache_resource
def train_nlu_model():
    df = pd.DataFrame(NLU_TRAINING_DATA)
    nlu_model = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), analyzer='word')),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000))
    ])
    nlu_model.fit(df['text'], df['intent'])
    return nlu_model


if "nlu_model" not in st.session_state:
    with st.spinner("⏳ جارٍ تدريب نموذج NLU المُخصص..."):
        st.session_state.nlu_model = train_nlu_model()
    st.sidebar.success("✅ تم تفعيل نموذج NLU المُخصص.")
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = whisper.load_model("base")


# =====================================================
# 3) TTS و LLM وإعدادات الشريط الجانبي (Sidebar)
# =====================================================

def speak(text: str):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            if "male" in v.name.lower() or "arab" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception:

        pass
# OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
USE_LLM = bool(OPENROUTER_API_KEY)
def call_llm(prompt: str) -> str:
    if not USE_LLM: return ""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": OPENROUTER_MODEL, "temperature": 0.4, "messages": [
            {"role": "system",
             "content": "أنت 'أبشر مساعد'، وكيل رقمي ذكي للخدمات الحكومية السعودية. اكتب ردوداً رسمية، واضحة ومختصرة."},
            {"role": "user", "content": prompt},
        ]}
        resp = requests.post(url, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# إعداد Sidebar
st.sidebar.markdown("### 👤 اختر المستخدم للديمو:")
USER_DISPLAY_MAP = {
    f"محمد الدوسري (المواطن)": "Mohamed",
    f"أحمد الرفاعي (المقيم/حالة حَرِجة)": "Ahmed",
    f"Alex Smith (الأجنبي/غير عربي)": "Alex",
}
selected_display = st.sidebar.radio(
    "",
    list(USER_DISPLAY_MAP.keys()),
    index=list(USER_DISPLAY_MAP.values()).index(st.session_state.current_user),
)
st.session_state.current_user = USER_DISPLAY_MAP[selected_display]

if st.sidebar.button("إعادة التهيئة الآن 🔄"):
    st.session_state.current_user = "Mohamed"
    REQUESTS.clear()
    st.session_state.last_visual = ""
    st.session_state.last_tts = ""
    st.session_state.just_reset = True
    st.rerun()

st.sidebar.markdown("---")
if USE_LLM:
    st.sidebar.success("🔗 LLM مفعّل (توجيهات رسمية).")
else:
    st.sidebar.warning("LLM غير مفعّل (اختياري).")
st.sidebar.markdown("---")


# =====================================================
# 4) منطق فهم الأوامر (Core Business Logic)
# =====================================================

def normalize(text: str) -> str:
    return text.replace("؟", " ").replace("،", " ").strip().lower()


def process_command(text: str):
    raw = text
    text = normalize(text)
    user_key = st.session_state.current_user
    user = USERS[user_key]

    # محاكاة للتأخير
    time.sleep(0.5)

    intent = st.session_state.nlu_model.predict([text])[0]
    st.sidebar.markdown(f"**🔬 النية المحددة (NLU):** `<{intent}>`", unsafe_allow_html=True)

    # 1. تغيير المستخدم
    if intent == "تغيير_مستخدم":
        if "محمد" in text or "دوسري" in text:
            st.session_state.current_user = "Mohamed"
        elif "أحمد" in text or "رفاعي" in text:
            st.session_state.current_user = "Ahmed"
        elif "alex" in text or "smith" in text:
            st.session_state.current_user = "Alex"
        new_user = USERS[st.session_state.current_user]
        visual = f"🔄 تم تغيير المستخدم إلى: **{new_user['name']}** ({new_user['user_type']})"
        tts = f"تم تغيير المستخدم إلى {new_user['name']}."
        return visual, tts

    # 2. تجديد الهوية/الإقامة (صياغة رقم الطلب مُحسَّنة)
    if intent == "تجديد_هوية_اقامة":
        req = create_request(user_key, "ID_RENEWAL", status="submitted")
        doc_type = "الإقامة" if user["user_type"] != "المواطن" else "الهوية الوطنية"
        tts_id = req['request_id'].replace("R-", "")

        visual = (
            f"✅ 🪪 **Identity API – تم تسجيل طلب تجديد {doc_type}**\n\n"
            f"- **رقم الطلب المرجعي:** `{req['request_id']}`\n"
            f"- المستخدم: **{user['name']}**\n"
        )
        extra = call_llm(f"المستخدم {doc_type}، طلب تجديد وثيقته. اكتب توجيهات رسمية عن خطوات المتابعة.")
        if extra: visual += f"\n**🧠 توجيه أبشر مساعد:**\n{extra}"

        tts = f"تم تسجيل طلب التجديد بنجاح برقم {tts_id}."
        return visual, tts

    # 3. الاستعلام عن الصلاحية (منطق التنبيه الحرج)
    if intent == "استعلام_صلاحية":
        create_request(user_key, "ID_STATUS", status="done")
        expiry_date_str = user.get("identity_expiry")
        remaining = (datetime.strptime(expiry_date_str, "%Y-%m-%d").date() - datetime.now().date()).days
        doc_type = "إقامتك" if user["user_type"] != "المواطن" else "هويتك الوطنية"

        if remaining < 30:
            visual = (
                f"🚨 ⏳ **Identity API – تنبيه صلاحية {doc_type} (حرج!)**\n\n"
                f"{doc_type} ستنتهي خلال **{remaining} يوماً** فقط.\n"
                f"💡 **توصية عاجلة:** يمكنك قول: **جدد {doc_type}** لبدء طلب التجديد الآن."
            )
            tts = f"تنبيه! {doc_type} ستنتهي خلال {remaining} يوم فقط. يرجى التجديد."
        else:
            visual = (
                f"✅ **Identity API – {doc_type} سارية**\n"
                f"يتبقى: **{remaining} يوماً**."
            )
            tts = f"{doc_type} سارية ومتبقي {remaining} يوم."
        return visual, tts

    # 4. تجديد رخصة القيادة (منطق الشروط المعقد)
    if intent == "تجديد_رخصة":
        if user["license_status"] == "Expired Medical":
            visual = (f"❌ 🚗 **لا يمكن تجديد الرخصة: فحص طبي منتهي**\n"
                      f"يرجى حجز موعد فحص طبي أولاً (Appointment API).")
            tts = "لا أستطيع تجديد رخصة القيادة، الفحص الطبي منتهي."
            return visual, tts
        if user["violations"] > 0:
            visual = (f"❌ 🚗 **لا يمكن تجديد الرخصة: مخالفات مرورية**\n"
                      f"على المستخدم مخالفات بقيمة {user['violations']} ريال. يجب سدادها.")
            tts = f"يوجد عليك مخالفات مرورية بقيمة {user['violations']} ريال، يجب سدادها."
            return visual, tts

        req = create_request(user_key, "DRIVER_LICENSE_RENEWAL", status="submitted")
        tts_id = req['request_id'].replace("R-", "")
        visual = (f"✅ 🚗 **Vehicle API – تم تسجيل طلب تجديد رخصة القيادة**\n"
                  f"- **رقم الطلب المرجعي:** `{req['request_id']}`")
        tts = f"تم تسجيل طلب التجديد بنجاح برقم {tts_id}."
        return visual, tts

    # 5. تجديد جواز السفر (متاح للمواطن فقط)
    if intent == "تجديد_جواز":
        if user["user_type"] != "المواطن":
            visual = "❌ 🛂 **Passport API – هذه الخدمة مخصصة للمواطنين السعوديين فقط**"
            tts = "خدمة تجديد جواز السفر مخصصة للمواطنين فقط."
            return visual, tts

        req = create_request(user_key, "PASSPORT_RENEWAL", status="submitted")
        tts_id = req['request_id'].replace("R-", "")
        visual = (f"✅ 🛂 **Passport API – تم تسجيل طلب تجديد جواز السفر**\n"
                  f"- **رقم الطلب المرجعي:** `{req['request_id']}`")
        tts = f"تم تسجيل طلب تجديد الجواز بنجاح برقم {tts_id}."
        return visual, tts

    # 6. الترحيب (مهم جداً للبداية)
    if intent == "ترحيب_وافتتاح":
        visual = "👋 أهلاً بك في أبشر مساعد! كيف يمكنني مساعدتك اليوم؟"
        tts = "أهلاً بك في أبشر مساعد! كيف يمكنني مساعدتك اليوم؟"
        return visual, tts

    # 7. أوامر غامضة أو أجنبية (Fallback)
    if intent == "طلب_مساعدة_غامض":
        if user["user_type"] == "الأجنبي (غير عربي)":
            visual = (
                f"💬 **Non-Arabic Support / General Help**\n"
                f"We recognize you need assistance, Alex. For quick service, please say simple commands like:\n"
                f"- 'Renew iqama'\n- 'Check my iqama expiry'\n- 'I need a traffic appointment'"
            )
            tts = "We recognize you need assistance. Please try again with a simple command."
        else:
            visual = f"❔ لم أفهم طلبك (النية: `{intent}`). يرجى استخدام أمر واضح."
            tts = "لم أفهم طلبك."
        return visual, tts

    # 8. افتراضياً (لأي نية لم يتم التعامل معها في الشروط السابقة)
    else:
        visual = f"❔ لم أفهم طلبك (النية: `{intent}`). يرجى استخدام أمر واضح."
        tts = "لم أفهم طلبك."
        return visual, tts


# =====================================================
# 5) واجهة Streamlit الرئيسية
# =====================================================

st.title("🤖 أبشر مساعد 🇸🇦")
current_user = USERS[st.session_state.current_user]
# مربع المعلومات للمستخدم تم تعديله ليصبح أخضر غامق بدلاً من الأزرق
st.info(
    f"👤 المستخدم الحالي: **{current_user['name']}** ({current_user['user_type']}) "
    f"| رقم الهوية/الإقامة: {current_user['national_id']}"
)
st.caption("يعمل المساعد على فهم النوايا (الخدمات) بغض النظر عن لغة المستخدم.")

tab_voice, tab_text = st.tabs(["🎙️ إدخال صوتي", "⌨️ إدخال نصي"])

with tab_voice:
    st.caption(
        "جربي الأوامر: 'جدد رخصتي' (لحالة حَرِجة)، 'كم بقي على إقامتي' (تنتهي في 13 يومًا)، أو 'change user to alex'.")
    st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
    audio = mic_recorder(start_prompt="🎤 اضغطي وابدأي بالتحدث...", stop_prompt="◼️ اضغطي مرة أخرى للإيقاف",
                         just_once=True, format="wav", key="voice_input")
    st.markdown('</div>', unsafe_allow_html=True)

    if audio:
        with st.spinner("⏳ يجري التحليل وفهم النية..."):
            audio_bytes = audio["bytes"]
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            model = st.session_state.whisper_model
            result = model.transcribe(temp_path, language="ar", task="transcribe", fp16=False)
            text = result.get("text", "").strip()

            st.info(f"📥 النص المستخرج: **{text or 'لم يتم التعرف على الكلام'}**")

            if text.strip():
                visual, tts_msg = process_command(text)
                st.session_state.last_visual = visual
                st.session_state.last_tts = tts_msg
                speak(tts_msg)
            else:
                st.warning("لم يتم التعرف على أمر واضح.")

with tab_text:
    text_input = st.text_input("أمرك النصي هنا:", placeholder="مثلاً: جدد رخصة القيادة، أو كم بقي على إقامتي؟",
                               key="manual_input_text")
    if st.button("إرسال الأمر النصي"):
        cleaned = text_input.strip()
        if cleaned:
            visual, tts_msg = process_command(cleaned)
            st.session_state.last_visual = visual
            st.session_state.last_tts = tts_msg
            speak(tts_msg)
        else:
            st.warning("رجاءً اكتبي أمراً واضحاً قبل الإرسال.")

# ---- النتيجة الأخيرة (Decision Output) ----
st.markdown("---")
st.markdown("### 📌 النتيجة الأخيرة (Decision Output)")

if st.session_state.last_visual:
    st.markdown(f"<div class='result-card'>{st.session_state.last_visual}</div>", unsafe_allow_html=True)
else:
    st.info("لم يتم تنفيذ أي أمر بعد. ابدأي بأمر 'مرحبا' أو أمر خدمة.")

# ---- سجل الطلبات ----
st.markdown("---")
st.markdown("### 🗃️ سجل الطلبات المنفَّذة")
if REQUESTS:
    for req in list(REQUESTS)[-3:][::-1]:
        service = SERVICE_BY_ID.get(req["service_id"])
        st.code(f"رقم الطلب المرجعي: {req['request_id']} | الخدمة: {service['name']} | الحالة: {req['status']}")
else:
    st.info("لا توجد طلبات في السجل بعد.")