from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import whisper
import uvicorn
import io
import time

# ============================================
# USERS + SYSTEM STATE
# ============================================

USERS = {
    "Mohamed": {
        "user_type": "المواطن",
        "user_id": "1001",
        "national_id": "1012345678",
        "name": "محمد الدوسري",
        "gender": "male",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        "license_status": "Valid",
        "violations": 0,
    },  
    "Sarah": {
        "user_type": "المواطن",
        "user_id": "1002",
        "national_id": "4012345678",
        "name": "ساره القحطاني",
        "gender": "female",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
        "license_status": "Valid",
        "violations": 0,
    },  
    "Ahmed": {
        "user_type": "المقيم",
        "user_id": "1003",
        "national_id": "2098765432",
        "name": "أحمد الرفاعي",
        "gender": "male",
        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=13)).strftime("%Y-%m-%d"),
        "license_status": "Expired Medical",
        "violations": 500,
    },
    "Alex": {
        "user_type": "الأجنبي (غير عربي)",
        "user_id": "1004",
        "national_id": "3012345678",
        "name": "Alex Smith",
        "gender": "male",

        "identity_expiry": (datetime.now().date() + pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        "license_status": "Valid",
        "violations": 0,
    },
}

SERVICES = {
    "ID_RENEWAL": {"service_id": "2001", "name": "تجديد الهوية/الإقامة"},
    "ID_STATUS": {"service_id": "2002", "name": "الاستعلام عن الصلاحية"},
    "DRIVER_LICENSE_RENEWAL": {"service_id": "3001", "name": "تجديد رخصة القيادة"},
    "PASSPORT_RENEWAL": {"service_id": "4001", "name": "تجديد جواز السفر"},
}

REQUESTS = []

STATE = {
    "current_user_key": "Mohamed",
    "last_visual": "",
    "recent_requests": []
}


app = FastAPI()

# CORS (allow frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




import google.generativeai as genai
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCJFoBe7oMe3apapQSyqwhOO_HSwQ5DJdE")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ Warning: GEMINI_API_KEY not set. Please set it as an environment variable.")

gemini_model = genai.GenerativeModel('gemini-2.5-flash')

whisper_model = whisper.load_model("large-v3")

def generate_action_steps(intent: str, user_text: str) -> str:
    """
    LLM #2: Takes the intent from LLM #1 and generates the ABSHER steps.
    Example:
      intent = "renew_passport"
      return: "1. افتح منصة أبشر ... 2. اختر خدماتي ... إلخ"
    """

    prompt = f"""
أنت مساعد مختص بشرح خطوات تنفيذ خدمات منصة أبشر للمستخدمين.

المهمة:
- لديك نية inferred intent حددها نظام آخر: "{intent}"
- ولديك النص الأصلي للمستخدم: "{user_text}"

أعطِ خطوات واضحة ومختصرة لتنفيذ هذه الخدمة على منصة أبشر.
اكتب فقط الخطوات بدون أي شرح إضافي.

أمثلة النوايا:
- renew_id = خطوات تجديد الهوية/الإقامة
- renew_passport = خطوات تجديد الجواز
- renew_license = خطوات تجديد رخصة القيادة
- appointment = خطوات حجز موعد
- check_expiry = خطوات الاستعلام عن صلاحية الهوية

ابدأ الرد مباشرة بالخطوات.
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"Error in LLM Action Generator: {e}")
        return "تعذر جلب خطوات الخدمة حالياً."

def detect_intent_with_gemini(user_text: str) -> str:
    """
    Use Gemini LLM to intelligently detect the intent of the user's query.
    Returns one of: renew_license, renew_passport, appointment, check_expiry,
                    renew_id, switch_user, greeting, unknown
    """
    prompt = f"""You are an intent classifier for a Saudi Arabian government services system (ABSHER).

Given the user's input text (in Arabic or English), classify it into ONE of these intents:

**Available Intents:**
1. **renew_license** - User wants to renew their driver's license (رخصة القيادة)
2. **renew_passport** - User wants to renew their passport (جواز السفر)
3. **appointment** - User wants to book an appointment (موعد)
4. **check_expiry** - User wants to check when their ID/residence expires (صلاحية الهوية/الإقامة)
5. **renew_id** - User wants to renew their national ID or residence permit (الهوية/الإقامة)
6. **switch_user** - User wants to change the current user/account
7. **greeting** - User is greeting (hello, hi, مرحبا, etc.)
8. **unknown** - None of the above intents match

**User Input:** "{user_text}"

**Instructions:**
- Respond with ONLY the intent name (e.g., "renew_license")
- Do not include any explanation, just the intent name
- Be flexible with Arabic dialects and variations
- Consider context and common phrasings

**Your Response (intent only):**"""

    try:
        response = gemini_model.generate_content(prompt)
        print("Gemini response:", response.text)
        intent = response.text.strip().lower()
        print("Detected intent:", intent)
        # Validate the intent is one of the expected values
        valid_intents = ["renew_license", "renew_passport", "appointment",
                        "check_expiry", "renew_id", "switch_user", "greeting", "unknown"]

        if intent in valid_intents:
            return intent
        else:
            # If Gemini returns something unexpected, try to map it
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    return valid_intent
            return "unknown"
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        # Fallback to basic keyword matching if Gemini fails
        text_lower = user_text.lower()
        if any(word in text_lower for word in ["رخصة", "قيادة", "license", "driving"]):
            return "renew_license"
        elif any(word in text_lower for word in ["جواز", "passport"]):
            return "renew_passport"
        elif any(word in text_lower for word in ["موعد", "appointment", "احجز"]):
            return "appointment"
        elif any(word in text_lower for word in ["صلاحية", "expiry", "تنتهي", "باقي"]):
            return "check_expiry"
        elif any(word in text_lower for word in ["هوية", "إقامة", "identity", "residence"]):
            return "renew_id"
        elif any(word in text_lower for word in ["switch", "change", "غير", "user"]):
            return "switch_user"
        elif any(word in text_lower for word in ["مرحبا", "hello", "hi", "السلام"]):
            return "greeting"
        return "unknown"


def create_request(user_key, service_id, status="submitted"):
    req = {
        "request_id": f"R-{len(REQUESTS)+1:04d}",
        "service_id": service_id,
        "status": status,
    }
    REQUESTS.append(req)
    STATE["recent_requests"] = REQUESTS[-3:]
    return req


def normalize(text):
    return text.lower().replace("؟", "").strip()


def handle_intent(user_key, intent):
    user = USERS[user_key]

    if intent == "switch_user":
        if "ahmed" in user_key.lower():
            STATE["current_user_key"] = "Ahmed"
        elif "alex" in user_key.lower():
            STATE["current_user_key"] = "Alex"
        else:
            STATE["current_user_key"] = "Mohamed"
        return "🔄 تم تغيير المستخدم."

    if intent == "renew_id":
        req = create_request(user_key, "ID_RENEWAL")
        return f"تم تقديم طلب تجديد الهوية. رقم الطلب {req['request_id']}"

    if intent == "check_expiry":
        expiry = user["identity_expiry"]
        return f"تنتهي هويتك بتاريخ {expiry}"

    if intent == "renew_license":
        if user["violations"] > 0:
            return f"لا يمكن التجديد. مخالفاتك: {user['violations']}"
        req = create_request(user_key, "DRIVER_LICENSE_RENEWAL")
        return f"تم تقديم طلب تجديد رخصة القيادة {req['request_id']}"

    if intent == "renew_passport":
        if user["user_type"] != "المواطن":
            return "الخدمة مخصصة للمواطنين فقط."
        req = create_request(user_key, "PASSPORT_RENEWAL")
        return f"تم تقديم طلب تجديد الجواز {req['request_id']}"

    return "لم أفهم أمرك."



@app.get("/api/users")
def get_users():
    return USERS


@app.get("/api/state")
def get_state():
    return STATE


class TextCommand(BaseModel):
    text: str


@app.post("/api/command")
def process_text(cmd: TextCommand):
    text = normalize(cmd.text)

    # 1) LLM #1 — Intent
    intent = detect_intent_with_gemini(text)

    # 2) Execute the system logic
    cur = STATE["current_user_key"]
    visual = handle_intent(cur, intent)

    # 3) LLM #2 — Action Steps
    steps = generate_action_steps(intent, cmd.text)
    print("Generated Steps:", steps)
    STATE["last_visual"] = visual

    return {
        "intent": intent,
        "text": cmd.text,
        "current_user": USERS[cur],
        "visual": visual,
        "action_steps": steps,
        "recent_requests": STATE["recent_requests"]
    }


@app.post("/api/switch-user")
def switch_user(user_key: str = Form(...)):
    STATE["current_user_key"] = user_key
    return {
        "current_user": USERS[user_key]
    }


import subprocess
import tempfile
import uuid
import os
import tempfile
import subprocess

@app.post("/api/voice")
async def process_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Create explicit temp paths
    temp_dir = tempfile.gettempdir()
    webm_path = os.path.join(temp_dir, f"{uuid.uuid4()}.webm")
    wav_path  = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

    # Save webm file
    with open(webm_path, "wb") as f:
        f.write(audio_bytes)

    # Convert via ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", webm_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Whisper STT
    result = whisper_model.transcribe(wav_path, language="ar", fp16=False)
    text = normalize(result["text"])
    # Clean up temp files
    print("text:", text)
    os.remove(webm_path)
    os.remove(wav_path)

    intent = detect_intent_with_gemini(text)
    cur = STATE["current_user_key"]

    visual = handle_intent(cur, intent)
    STATE["last_visual"] = visual

    steps = generate_action_steps(intent, text)

    return {
        "text": text,
        "intent": intent,
        "current_user": USERS[cur],
        "visual": visual,
        "action_steps": steps,
        "recent_requests": STATE["recent_requests"]
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
