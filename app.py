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

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI()

# CORS (allow frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Load Whisper + NLU Model
# ============================================

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Train NLU model
NLU_DATA = {
    "text": [
        "أجدد رخصتي", "رخصتي منتهية", "أبي تجديد قيادة",
        "جوازي خلص", "أحتاج تجديد جواز السفر",
        "احجز موعد", "أبغى موعد جوازات",
        "كم باقي على الإقامة", "متى تنتهي الهوية",
        "أجدد الهوية", "الإقامة خلصت",
        "change user to alex", "غير المستخدم",
        "مرحبا", "hello",
    ],
    "intent": [
        "renew_license", "renew_license", "renew_license",
        "renew_passport", "renew_passport",
        "appointment", "appointment",
        "check_expiry", "check_expiry",
        "renew_id", "renew_id",
        "switch_user", "switch_user",
        "greeting", "greeting",
    ],
}

df = pd.DataFrame(NLU_DATA)

nlu = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
nlu.fit(df["text"], df["intent"])

whisper_model = whisper.load_model("large-v3")


# ============================================
# Helper Functions
# ============================================

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


# ============================================
# API ROUTES
# ============================================

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
    intent = nlu.predict([text])[0]
    cur = STATE["current_user_key"]

    visual = handle_intent(cur, intent)
    STATE["last_visual"] = visual

    return {
        "current_user": USERS[STATE["current_user_key"]],
        "visual": visual,
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

    intent = nlu.predict([text])[0]
    cur = STATE["current_user_key"]

    visual = handle_intent(cur, intent)
    STATE["last_visual"] = visual

    return {
        "text": text,
        "current_user": USERS[cur],
        "visual": visual,
        "recent_requests": STATE["recent_requests"]
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
