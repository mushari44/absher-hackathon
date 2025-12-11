from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import whisper
import uvicorn
import io
import time
import os
import subprocess
import tempfile
import uuid
from dotenv import load_dotenv
load_dotenv()


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is not set!")

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4.1-mini"


import torch
import whisper
import os

# Force GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load Whisper on GPU
whisper_model = whisper.load_model("large-v3", device="cuda")

print("🔥 Whisper is running on:", torch.cuda.get_device_name(0))# whisper_model = whisper.load_model("base")


def detect_intent(user_text: str) -> str:
    prompt = f"""
You are an intent classifier for a Saudi government services assistant (ABSHER).
Classify the following user text into ONE intent:

renew_license
renew_passport
appointment
check_expiry
renew_id
switch_user
greeting
unknown

User text: "{user_text}"
Return ONLY the intent name.
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        intent = response.choices[0].message.content.strip().lower()

        valid = [
            "renew_license", "renew_passport", "appointment",
            "check_expiry", "renew_id", "switch_user",
            "greeting", "unknown"
        ]

        return intent if intent in valid else "unknown"

    except Exception as e:
        print("❌ Intent detection failed:", e)
        return "unknown"


def generate_action_steps(intent: str, user_text: str) -> str:
    prompt = f"""
أنت مساعد يشرح خطوات تنفيذ خدمات منصة أبشر.

النية: {intent}
النص: {user_text}

اكتب خطوات تنفيذ الخدمة فقط، مرقمة، بدون أي كلام إضافي.
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Steps generation failed:", e)
        return "تعذر جلب خطوات الخدمة حالياً."



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
    print(f"➡️ Handling intent '{intent}' for user '{user_key}'")
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
        return f"تنتهي هويتك بتاريخ {user['identity_expiry']}"

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

def text_to_speech(text: str) -> bytes:
    """
    Convert text to speech using OpenAI TTS.
    Returns audio bytes (MP3 format).
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",  # Use tts-1 or tts-1-hd
            voice="alloy",   # voices: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="mp3"
        )
        return response.read()
    except Exception as e:
        print("❌ TTS Error:", e)
        return None

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

    # 1) intent with GPT
    intent = detect_intent(text)

    # 2) execute logic
    cur = STATE["current_user_key"]
    visual = handle_intent(cur, intent)

    # 3) action steps
    steps = generate_action_steps(intent, cmd.text)
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
    return {"current_user": USERS[user_key]}


import base64

@app.post("/api/voice")
async def process_voice(file: UploadFile = File(...)):
    webm_path = None
    wav_path = None

    try:
        audio_bytes = await file.read()

        # Temporary paths
        temp_dir = tempfile.gettempdir()
        webm_path = os.path.join(temp_dir, f"{uuid.uuid4()}.webm")
        wav_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

        # Save uploaded audio
        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to wav (whisper requirement)
        cmd = [
            "ffmpeg", "-y",
            "-i", webm_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if FFmpeg conversion succeeded
        if result.returncode != 0:
            print(f"❌ FFmpeg Error: {result.stderr.decode()}")
            return {"error": "Audio conversion failed"}

        if not os.path.exists(wav_path):
            print("❌ WAV file not created")
            return {"error": "Audio conversion failed"}

        # Whisper STT
        try:
            transcription_result = whisper_model.transcribe(wav_path, language="ar", fp16=False)
            text = normalize(transcription_result["text"])
        except Exception as e:
            print(f"❌ Whisper Error: {e}")
            return {"error": "Speech transcription failed"}

        # Intent → Action
        intent = detect_intent(text)
        cur = STATE["current_user_key"]
        visual = handle_intent(cur, intent)
        steps = generate_action_steps(intent, text)

        # Create the full reply for the user
        final_text = f"{visual}\n\nالخطوات:\n{steps}"

        # Convert text → speech
        audio_output = text_to_speech(final_text)

        if audio_output is None:
            return {"error": "TTS failed"}

        # Encode audio as base64 to include in JSON response
        audio_base64 = base64.b64encode(audio_output).decode('utf-8')

        # Return JSON response with text data AND audio
        STATE["last_visual"] = visual
        return {
            "intent": intent,
            "text": text,  # The transcribed text from user's voice
            "current_user": USERS[cur],
            "visual": visual,
            "action_steps": steps,
            "recent_requests": STATE["recent_requests"],
            "audio": audio_base64,  # Base64 encoded audio for playback
            "audio_format": "mp3"
        }

    except Exception as e:
        print(f"❌ Voice processing error: {e}")
        return {"error": f"Voice processing failed: {str(e)}"}

    finally:
        # Always cleanup temp files
        if webm_path and os.path.exists(webm_path):
            try:
                os.remove(webm_path)
            except Exception as e:
                print(f"⚠️ Failed to remove {webm_path}: {e}")

        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                print(f"⚠️ Failed to remove {wav_path}: {e}")

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
