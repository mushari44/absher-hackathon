

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, field_validator, Field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import whisper
import uvicorn
import os
import subprocess
import tempfile
import uuid
import hashlib
import re
import base64
import time
import logging
from contextlib import asynccontextmanager
import json
from functools import wraps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")

# Separate audit log file
audit_handler = logging.FileHandler("audit.log")
audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

# ============================================
# SECURITY UTILITIES
# ============================================

def sanitize_for_logging(text: str) -> str:
    """Redact sensitive info from logs"""
    # Redact national IDs (10 digits)
    text = re.sub(r'\b\d{10}\b', '[REDACTED_ID]', text)
    # Redact phone numbers
    text = re.sub(r'\b05\d{8}\b', '[REDACTED_PHONE]', text)
    return text

def sanitize_for_llm(text: str) -> str:
    """Prevent prompt injection by sanitizing user input"""
    # Remove control characters
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    # Limit length
    text = text[:500]
    # Escape quotes to prevent prompt breaking
    text = text.replace('"', '\\"').replace("'", "\\'")
    return text

def validate_temp_path(path: str) -> bool:
    """Ensure temp file path only contains safe characters"""
    return bool(re.match(r'^[a-zA-Z0-9_\-/:\\\.]+$', path))

def log_audit_event(user_id: str, action: str, details: dict, ip: str = "unknown"):
    """Log audit events for compliance and security monitoring"""
    audit_logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "action": action,
        "details": details,
        "ip": ip
    }))

# ============================================
# RATE LIMITING (Simple in-memory)
# ============================================

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is within rate limit"""
        now = time.time()

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests outside window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window_seconds
        ]

        # Check if limit exceeded
        if len(self.requests[key]) >= max_requests:
            return False

        # Add current request
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter()

def rate_limit(max_requests: int, window_seconds: int):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"{client_ip}:{request.url.path}"

            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds"
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# ============================================
# DATA MODELS
# ============================================

USERS = {
    "Ahmed": {
        "user_type": "المواطن",
        "user_id": "1001",
        "name": "أحمد",
        "gender": "male",
        "national_id": {
            "status": "expired",
            "expiry_date": "2025-09-10"
        },
        "driver_license": {
            "status": "valid",
            "expiry_date": "2027-04-15"
        },
        "passport": {
            "status": "near_expiry",
            "expiry_date": "2026-02-15"
        },
        "violations": {
            "count": 7,
            "total_amount": 3250,
            "service_block": True
        },
        "wallet": {
            "plates": ["ABC123", "XYZ777", "KSA999"]
        }
    },
    "Sara": {
        "user_type": "المواطن",
        "user_id": "1002",
        "name": "ساره",
        "gender": "female",
        "national_id": {
            "status": "valid",
            "expiry_date": "2030-05-11"
        },
        "driver_license": {
            "status": "near_expiry",
            "expiry_date": "2026-01-14"
        },
        "violations": {
            "count": 1,
            "total_amount": 300,
            "service_block": False
        },
        "wallet": {
            "plates": ["DEF456"]
        }
    },
    "Mohammed": {
        "user_type": "المواطن",
        "user_id": "1003",
        "name": "محمد",
        "gender": "male",
        "national_id": {
            "status": "near_expiry",
            "expiry_date": "2026-01-11",
            "needs_photo_update": True
        },
        "driver_license": {
            "status": "valid",
            "expiry_date": "2028-02-18"
        },
        "violations": {
            "count": 2,
            "total_amount": 250,
            "service_block": False
        },
        "wallet": {
            "plates": ["GHI789", "JKL012"]
        }
    },
    "Alex": {
        "user_type": "المقيم",
        "user_id": "1004",
        "name": "Alex",
        "gender": "male",
        "iqama": {
            "status": "valid",
            "expiry_date": "2026-10-01"
        },
        "driver_license": {
            "status": "near_expiry",
            "expiry_date": "2026-01-10"
        },
        "violations": {
            "count": 0,
            "total_amount": 0
        },
        "wallet": {
            "plates": []
        }
    }
}

SERVICES = {
    "ID_RENEWAL": {"service_id": "2001", "name": "تجديد الهوية/الإقامة"},
    "ID_STATUS": {"service_id": "2002", "name": "الاستعلام عن الصلاحية"},
    "DRIVER_LICENSE_RENEWAL": {"service_id": "3001", "name": "تجديد رخصة القيادة"},
    "PASSPORT_RENEWAL": {"service_id": "4001", "name": "تجديد جواز السفر"},
    "PLATE_TRANSFER": {"service_id": "5001", "name": "نقل ملكية لوحة مركبة"},
}

REQUESTS = []
PLATE_TRANSFER_LOG = []  # Immutable audit log for plate transfers

# Per-user request history
USER_REQUESTS = {
    "Ahmed": [],
    "Sara": [],
    "Mohammed": [],
    "Alex": []
}

STATE = {
    "current_user_key": "Ahmed",
    "last_visual": "",
    "recent_requests": [],
    "conversation_history": [],
    "pending_action": None,
    "pending_intent": None
}

# ============================================
# APP INITIALIZATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks"""
    # Startup
    logger.info("🚀 Starting ABSHER Backend...")

    # Validate critical environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"❌ Missing required environment variables: {missing}")

    # Load Whisper model
    logger.info("📥 Loading Whisper model...")
    global whisper_model, DEVICE, USE_FP16

    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability()
        USE_FP16 = compute_cap[0] >= 7 and compute_cap[0] < 10

        try:
            test_tensor = torch.randn(1, 1, device=DEVICE, dtype=torch.float16)
            del test_tensor
        except Exception:
            USE_FP16 = False
    else:
        USE_FP16 = False

    whisper_model = whisper.load_model("large-v3", device=DEVICE)
    logger.info(f"✅ Whisper loaded on {DEVICE}, FP16={USE_FP16}")

    # Initialize OpenAI client
    global client
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("✅ OpenAI client initialized")

    yield

    # Shutdown
    logger.info("🛑 Shutting down ABSHER Backend...")

app = FastAPI(
    title="ABSHER Digital Services API",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================
# CORS CONFIGURATION
# ============================================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Validate no wildcards in production
if "*" in ALLOWED_ORIGINS and os.getenv("ENVIRONMENT") == "production":
    raise RuntimeError("❌ Wildcard CORS origin not allowed in production!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
    max_age=3600,
)

# ============================================
# OPENAI SETUP
# ============================================

GPT_MODEL = "gpt-4o-mini"

# ============================================
# VALIDATION FUNCTIONS
# ============================================

class TextCommand(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    user_key: Optional[str] = None

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")

        # Remove control characters and null bytes
        v = "".join(ch for ch in v if ch.isprintable() or ch.isspace())
        v = v.replace('\x00', '')

        if len(v) > 1000:
            raise ValueError("Text too long (max 1000 characters)")

        return v.strip()

class SwitchUserRequest(BaseModel):
    user_key: str

    @field_validator('user_key')
    @classmethod
    def validate_user_key(cls, v):
        if v not in USERS:
            raise ValueError(f"Invalid user_key. Must be one of: {list(USERS.keys())}")
        return v

class PaymentRequest(BaseModel):
    user_id: str
    amount: float = Field(..., gt=0, le=100000)
    service: str

class PlateTransferRequest(BaseModel):
    from_user: str
    to_user: str
    plate: str
    price: Optional[float] = Field(default=0, ge=0, le=1000000)

    @field_validator('plate')
    @classmethod
    def validate_plate(cls, v):
        # Saudi plate format: 3-4 letters/numbers
        if not re.match(r'^[A-Z0-9]{3,7}$', v.upper()):
            raise ValueError("Invalid plate format")
        return v.upper()

# ============================================
# HELPER FUNCTIONS
# ============================================

def normalize(text: str) -> str:
    """Normalize text for processing"""
    return text.lower().replace("؟", "").replace("،", "").strip()

def create_request(user_key: str, service_id: str, status: str = "submitted") -> dict:
    """Create a service request"""
    req = {
        "request_id": f"R-{len(REQUESTS)+1:04d}",
        "service_id": service_id,
        "user_key": user_key,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    REQUESTS.append(req)

    # Add to user-specific history
    if user_key in USER_REQUESTS:
        USER_REQUESTS[user_key].append(req)

    # Update recent requests with user's last 3 requests
    STATE["recent_requests"] = USER_REQUESTS.get(user_key, [])[-3:]

    # Log audit event
    log_audit_event(
        user_key,
        f"service_request_created",
        {"request_id": req["request_id"], "service": service_id}
    )

    return req

# ============================================
# INTENT DETECTION
# ============================================

def detect_intent(user_text: str) -> str:
    """Detect user intent using GPT with sanitized input"""
    user_text_safe = sanitize_for_llm(user_text)

    # Quick pattern matching for common intents (faster and more reliable)
    text_lower = user_text.lower().strip()

    # Dashboard patterns
    if any(pattern in text_lower for pattern in [
        "عرض جميع خدماتي", "عرض الخدمات", "وش خدماتي", "ملخص حسابي",
        "عرض الملخص", "وش وضعي", "أبي أشوف كل شي", "show my services",
        "my services", "dashboard", "خدماتي"
    ]):
        return "dashboard"

    # Pay violations patterns
    if any(pattern in text_lower for pattern in [
        "ادفع مخالفاتي", "ابي ادفع مخالفات", "دفع المخالفات", "سداد المخالفات",
        "ابغى ادفع المخالفات", "أبي أدفع مخالفاتي", "pay violations", "pay my violations"
    ]):
        return "pay_violations"

    # Greeting patterns
    if text_lower in ["مرحبا", "مرحباً", "hello", "hi", "السلام عليكم"]:
        return "greeting"

    prompt = f"""
You are an intent classifier for a Saudi government services assistant (ABSHER).
Classify the following user text into ONE intent:

SERVICE INTENTS (specific services - these have PRIORITY over info):
- dashboard: User wants to see ALL their services summary at once (عرض جميع الخدمات، خدماتي، ملخص)
- id_renewal: User wants to renew ID/Iqama (تجديد الهوية/الإقامة)
- id_status: User wants to check ID/Iqama expiry status (الاستعلام عن الصلاحية)
- driver_license_renewal: User wants to renew driver license (تجديد رخصة القيادة)
- passport_renewal: User wants to renew passport (تجديد جواز السفر)
- plate_transfer: User wants to transfer vehicle plate ownership (نقل ملكية لوحة)
- pay_violations: User wants to pay traffic violations directly (دفع المخالفات، سداد المخالفات)

OTHER INTENTS (use these ONLY if no service intent matches):
- info: General questions about HOW services work, requirements, procedures (معلومات عامة عن كيفية العمل)
- fraud_scam: User asking if service requires payment, asking about suspicious requests for money, verifying if something is legitimate (احتيال، طلب أموال)
- switch_user: User wants to change account
- greeting: Simple greetings (hello, hi, مرحبا)
- unknown: Anything else

EXAMPLES:
- "جدد رخصتي" → driver_license_renewal
- "كم باقي على الإقامة؟" → id_status
- "جدد هويتي" → id_renewal
- "أبغى جواز سفر جديد" → passport_renewal
- "أبي أنقل اللوحة للمشتري" → plate_transfer
- "انقل اللوحة من محفظتي لمحفظة أحمد" → plate_transfer
- "محفظتي" → plate_transfer
- "وش اللوحات اللي عندي" → plate_transfer
- "تحقق من اللوحات في محفظتي" → plate_transfer
- "شوف لوحاتي" → plate_transfer
- "show my plates" → plate_transfer
- "ابي ادفع مخالفاتي" → pay_violations
- "دفع المخالفات" → pay_violations
- "سداد المخالفات" → pay_violations
- "وش خدماتي؟" → dashboard
- "عرض جميع خدماتي" → dashboard
- "show my services" → dashboard
- "ملخص حسابي" → dashboard
- "عرض الملخص" → dashboard
- "أبي أشوف كل شي" → dashboard
- "وش وضعي" → dashboard
- "كيف أجدد رخصة القيادة؟" → info
- "هل الخدمة مجانية؟" → info
- "وصلتني رسالة تطلب دفع رسوم، هل هذا صحيح؟" → fraud_scam

User text: "{user_text_safe}"
Return ONLY the intent name (lowercase with underscores).
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20
        )
        intent = response.choices[0].message.content.strip().lower()

        valid = [
            "id_renewal", "id_status", "driver_license_renewal", "passport_renewal",
            "plate_transfer", "info", "fraud_scam", "switch_user", "greeting", "unknown"
        ]

        result = intent if intent in valid else "unknown"
        logger.info(f"Intent detected: {result} for text: {sanitize_for_logging(user_text)}")
        return result

    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        return "unknown"

# ============================================
# TEXT TO SPEECH
# ============================================

def text_to_speech(text: str) -> Optional[bytes]:
    """
    Convert text to speech using OpenAI TTS API with Arabic optimization.
    Returns mp3 bytes or None on failure.

    Improvements for Arabic:
    - Uses 'alloy' voice (better multilingual support than 'onyx')
    - Uses 'tts-1-hd' model for higher quality
    - Preprocesses text for better Arabic pronunciation
    """
    try:
        # Truncate very long text
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters for TTS")

        # Preprocess Arabic text for better TTS
        text = preprocess_arabic_for_tts(text)

        # Try multiple voices in order of Arabic quality preference
        # 'alloy' and 'shimmer' have better multilingual/Arabic support than 'onyx'
        voices = ["alloy", "shimmer", "nova"]

        for voice in voices:
            try:
                response = client.audio.speech.create(
                    model="tts-1-hd",  # High-definition model
                    voice=voice,
                    input=text,
                    response_format="mp3",
                    speed=0.95  # Slightly slower for Arabic clarity
                )
                audio_bytes = response.read()

                if audio_bytes and len(audio_bytes) >= 100:
                    logger.info(f"TTS successful with voice: {voice}")
                    return audio_bytes

            except Exception as voice_error:
                logger.warning(f"TTS failed with voice {voice}: {voice_error}")
                continue

        logger.error("TTS failed with all available voices")
        return None

    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)
        return None


def preprocess_arabic_for_tts(text: str) -> str:
    """
    Preprocess Arabic text for better TTS pronunciation.

    Improvements:
    - Normalize Arabic characters
    - Add spacing around numbers for better pronunciation
    - Replace common abbreviations with full words
    - Clean emoji/special characters that may cause issues
    """
    import re

    # Normalize Arabic characters
    text = text.replace('ي', 'ی')  # Normalize ya
    text = text.replace('ك', 'ک')  # Normalize kaf

    # Add spaces around numbers for better pronunciation
    text = re.sub(r'(\d+)', r' \1 ', text)

    # Replace common abbreviations with full words for better pronunciation
    replacements = {
        'ریال': 'ريال سعودي',
        'km': 'كيلومتر',
        'ID': 'رقم الهوية',
        'R-': 'طلب رقم ',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ============================================
# CONVERSATIONAL AI
# ============================================

def generate_conversational_response(user_text: str, user_key: str) -> str:
    """Generate intelligent conversational response with full context"""
    user = USERS[user_key]

    # Get identity info
    identity = user.get("national_id") or user.get("iqama")
    identity_expiry = identity.get("expiry_date") if identity else "غير متوفر"
    identity_status = identity.get("status") if identity else "غير متوفر"

    # Get license info
    license_info = user.get("driver_license", {})
    license_status = license_info.get("status", "غير متوفر")

    # Get violations info
    violations = user.get("violations", {})
    violations_text = f"{violations.get('count', 0)} مخالفة بقيمة {violations.get('total_amount', 0)} ریال"

    messages = [
        {"role": "system", "content": f"""أنت مساعد ذكي لمنصة أبشر الحكومية السعودية.

معلومات المستخدم الحالي:
- الاسم: {user['name']}
- النوع: {user['user_type']}
- حالة الهوية/الإقامة: {identity_status}
- تاريخ انتهاء الهوية/الإقامة: {identity_expiry}
- حالة رخصة القيادة: {license_status}
- المخالفات المرورية: {violations_text}

مهامك:
1. الرد على الأسئلة بشكل طبيعي ومحادث
2. تذكّر السياق من المحادثة السابقة
3. تقديم معلومات دقيقة عن خدمات أبشر
4. استخدام اللغة العربية الفصحى البسيطة
5. كن مفيداً وودوداً

ملاحظات مهمة:
• جميع الخدمات مجانية أو برسوم رسمية فقط
• لا تطلب أبشر أبداً دفعات عبر رسائل أو مكالمات
• نقل ملكية اللوحات يتم عبر نظام آمن ومُوثق"""}
    ]

    # Add conversation history (last 10 messages)
    for msg in STATE["conversation_history"][-10:]:
        messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": sanitize_for_llm(user_text)})

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        assistant_response = response.choices[0].message.content.strip()

        # Update conversation history
        STATE["conversation_history"].append({"role": "user", "content": user_text})
        STATE["conversation_history"].append({"role": "assistant", "content": assistant_response})

        # Keep only last 20 messages
        if len(STATE["conversation_history"]) > 20:
            STATE["conversation_history"] = STATE["conversation_history"][-20:]

        return assistant_response

    except Exception as e:
        logger.error(f"Conversational response failed: {e}", exc_info=True)
        return "عذراً، حدث خطأ. يمكنك المحاولة مرة أخرى."

# ============================================
# PLATE TRANSFER LOGIC
# ============================================

def validate_plate_transfer(from_user_key: str, to_user_key: str, plate: str) -> dict:
    """Validate plate transfer requirements"""
    errors = []

    # Check users exist
    if from_user_key not in USERS:
        errors.append(f"المستخدم البائع '{from_user_key}' غير موجود")
    if to_user_key not in USERS:
        errors.append(f"المستخدم المشتري '{to_user_key}' غير موجود")

    if errors:
        return {"valid": False, "errors": errors}

    from_user = USERS[from_user_key]
    to_user = USERS[to_user_key]

    # Check plate exists in seller's wallet
    wallet = from_user.get("wallet", {})
    plates = wallet.get("plates", [])

    if plate not in plates:
        errors.append(f"اللوحة {plate} غير موجودة في محفظة البائع")

    # Check seller identity status
    seller_identity = from_user.get("national_id") or from_user.get("iqama")
    if seller_identity and seller_identity.get("status") == "expired":
        errors.append("هوية البائع منتهية، يجب التجديد أولاً")

    # Check buyer identity status
    buyer_identity = to_user.get("national_id") or to_user.get("iqama")
    if buyer_identity and buyer_identity.get("status") == "expired":
        errors.append("هوية المشتري منتهية، يجب التجديد أولاً")

    # Check for violations blocking service
    if from_user.get("violations", {}).get("service_block"):
        errors.append("يوجد إيقاف خدمات على البائع بسبب مخالفات")

    if to_user.get("violations", {}).get("service_block"):
        errors.append("يوجد إيقاف خدمات على المشتري بسبب مخالفات")

    if errors:
        return {"valid": False, "errors": errors}

    return {"valid": True, "errors": []}

def detect_plate_transfer_fraud(from_user_key: str, to_user_key: str, plate: str, price: float) -> dict:
    """Detect suspicious plate transfer patterns"""
    warnings = []
    fraud_score = 0

    # Check 1: Unrealistic price
    if price < 100:
        warnings.append("⚠️ السعر منخفض جداً (أقل من 100 ريال) - قد يكون احتيال")
        fraud_score += 3
    elif price > 500000:
        warnings.append("⚠️ السعر مرتفع جداً (أكثر من 500,000 ريال) - تحقق من المشتري")
        fraud_score += 2

    # Check 2: Recent transfers of same plate
    recent_transfers = [
        t for t in PLATE_TRANSFER_LOG
        if t["plate"] == plate and
        (datetime.now() - datetime.fromisoformat(t["timestamp"])).days < 7
    ]

    if len(recent_transfers) > 0:
        warnings.append(f"⚠️ تم نقل هذه اللوحة {len(recent_transfers)} مرة خلال آخر 7 أيام")
        fraud_score += 2

    # Check 3: Same user as buyer/seller
    if from_user_key == to_user_key:
        warnings.append("❌ لا يمكن نقل اللوحة للمستخدم نفسه")
        fraud_score += 5

    # Check 4: Check transfer history patterns
    user_transfers = [
        t for t in PLATE_TRANSFER_LOG
        if t["from_user"] == from_user_key and
        (datetime.now() - datetime.fromisoformat(t["timestamp"])).days < 30
    ]

    if len(user_transfers) >= 5:
        warnings.append("⚠️ البائع قام بنقل 5 لوحات أو أكثر خلال آخر 30 يوم")
        fraud_score += 2

    # Determine fraud level
    if fraud_score >= 5:
        return {
            "fraud_detected": True,
            "fraud_score": fraud_score,
            "warnings": warnings,
            "recommendation": "BLOCK - احتمال احتيال عالي"
        }
    elif fraud_score >= 3:
        return {
            "fraud_detected": True,
            "fraud_score": fraud_score,
            "warnings": warnings,
            "recommendation": "CONFIRM - يحتاج تأكيد إضافي"
        }
    else:
        return {
            "fraud_detected": False,
            "fraud_score": fraud_score,
            "warnings": warnings if warnings else ["✅ لا توجد مؤشرات احتيال"],
            "recommendation": "ALLOW"
        }

def execute_plate_transfer(from_user_key: str, to_user_key: str, plate: str, price: float) -> dict:
    """Execute plate transfer atomically"""
    try:
        # Create transaction ID
        transaction_id = f"PLT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        # Snapshot before state
        before_state = {
            "seller": {
                "name": USERS[from_user_key]["name"],
                "plates": USERS[from_user_key]["wallet"]["plates"].copy()
            },
            "buyer": {
                "name": USERS[to_user_key]["name"],
                "plates": USERS[to_user_key]["wallet"]["plates"].copy()
            }
        }

        # Execute transfer (atomic operation)
        USERS[from_user_key]["wallet"]["plates"].remove(plate)
        USERS[to_user_key]["wallet"]["plates"].append(plate)

        # Snapshot after state
        after_state = {
            "seller": {
                "name": USERS[from_user_key]["name"],
                "plates": USERS[from_user_key]["wallet"]["plates"].copy()
            },
            "buyer": {
                "name": USERS[to_user_key]["name"],
                "plates": USERS[to_user_key]["wallet"]["plates"].copy()
            }
        }

        # Log to immutable audit trail
        transfer_record = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now().isoformat(),
            "from_user": from_user_key,
            "to_user": to_user_key,
            "plate": plate,
            "price": price,
            "before_state": before_state,
            "after_state": after_state,
            "status": "completed"
        }
        PLATE_TRANSFER_LOG.append(transfer_record)

        # Audit log
        log_audit_event(
            from_user_key,
            "plate_transfer_completed",
            {
                "transaction_id": transaction_id,
                "to_user": to_user_key,
                "plate": plate,
                "price": price
            }
        )

        return {
            "success": True,
            "transaction_id": transaction_id,
            "before_state": before_state,
            "after_state": after_state,
            "message": f"تم نقل اللوحة {plate} بنجاح من {USERS[from_user_key]['name']} إلى {USERS[to_user_key]['name']}"
        }

    except Exception as e:
        logger.error(f"Plate transfer execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"فشل تنفيذ النقل: {str(e)}"
        }

# ============================================
# INTENT HANDLERS
# ============================================

def handle_intent(user_key: str, intent: str, user_text: Optional[str] = None) -> str:
    """Handle user intents"""
    user = USERS[user_key]
    logger.info(f"Handling intent '{intent}' for user '{user_key}'")

    # Switch user
    if intent == "switch_user":
        target_key = None
        text_lower = (user_text or "").lower()

        for key in USERS.keys():
            if key.lower() in text_lower:
                target_key = key
                break

        if target_key and target_key in USERS:
            STATE["current_user_key"] = target_key
            STATE["conversation_history"] = []
            STATE["pending_action"] = None
            STATE["pending_intent"] = None

            log_audit_event(user_key, "user_switched", {"to_user": target_key})

            return f"🔄 تم تغيير المستخدم إلى {USERS[target_key]['name']}."

        return "⚠️ لم أتعرف على المستخدم المطلوب. المستخدمون المتاحون: Ahmed, Sara, Mohammed, Alex."

    # Greeting
    if intent == "greeting":
        return f"مرحباً {user['name']}! 👋 كيف يمكنني مساعدتك اليوم؟"

    # ID/Iqama Status Check
    if intent == "id_status":
        identity = user.get("national_id") or user.get("iqama")
        doc_type = "هويتك" if user["user_type"] == "المواطن" else "إقامتك"

        if not identity:
            return f"⚠️ لا يمكن العثور على معلومات {doc_type}."

        status = identity.get("status")
        expiry_date_str = identity.get("expiry_date")

        if status == "expired":
            return f"⚠️ تنبيه: {doc_type} منتهية!\nتاريخ الانتهاء: {expiry_date_str}\nيرجى المبادرة بالتجديد فوراً."

        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        days_left = (expiry_date - datetime.now()).days

        if days_left < 0:
            return f"⚠️ تنبيه: {doc_type} منتهية منذ {abs(days_left)} يوم!\nيرجى المبادرة بالتجديد فوراً."
        elif status == "near_expiry" or days_left <= 30:
            return f"⚠️ تنبيه: {doc_type} تنتهي خلال {days_left} يوم!\nننصح بالتجديد في أقرب وقت."
        else:
            return f"✅ {doc_type} سارية المفعول.\nتاريخ الانتهاء: {expiry_date_str}\nمتبقي: {days_left} يوم"

    # ID Renewal
    if intent == "id_renewal":
        identity = user.get("national_id") or user.get("iqama")
        doc_type = "الهوية الوطنية" if user["user_type"] == "المواطن" else "الإقامة"

        if not identity:
            return f"⚠️ لا يمكن العثور على معلومات {doc_type}."

        # Check requirements
        missing_requirements = []

        # Check if user has service block
        if user.get("service_block"):
            missing_requirements.append("⚠️ يوجد إيقاف خدمات - يرجى مراجعة الجهة المختصة")

        # For residents, check health insurance
        if user["user_type"] != "المواطن":
            # Assume they need health insurance (in real system, check expiry)
            # missing_requirements.append("⚠️ التأمين الصحي منتهي - يرجى التجديد")
            pass

        # If there are missing requirements, return pending status
        if missing_requirements:
            req = create_request(user_key, "ID_RENEWAL", status="pending_documents")
            missing_text = "\n".join([f"• {req}" for req in missing_requirements])

            return f"""⏳ طلب تجديد {doc_type} بانتظار استكمال المتطلبات

رقم الطلب: {req['request_id']}
📅 تاريخ الطلب: {datetime.now().strftime("%Y-%m-%d %H:%M")}

❌ المتطلبات الناقصة:
{missing_text}

📋 يرجى استكمال المتطلبات لمتابعة الطلب"""

        # All requirements met - create approved request
        req = create_request(user_key, "ID_RENEWAL", status="submitted")

        return f"""✅ طلب تجديد {doc_type} تم بنجاح!

رقم الطلب: {req['request_id']}
📅 تاريخ الطلب: {datetime.now().strftime("%Y-%m-%d %H:%M")}
الحالة: قيد المعالجة ✅

📋 المتطلبات المستوفاة:
• صورة شخصية حديثة ✅
• سداد أي رسوم متأخرة ✅
• التأمين الصحي ساري ✅
• عدم وجود إيقاف خدمات ✅

⏱️ مدة التنفيذ: 1-3 أيام عمل
📍 يمكنك استلام {doc_type} من أقرب مكتب أحوال مدنية"""

    # Driver License Renewal
    if intent == "driver_license_renewal":
        license = user.get("driver_license", {})

        if not license:
            return "⚠️ لا يمكن العثور على معلومات رخصة القيادة."

        # Check violations
        violations = user.get("violations", {})
        violations_count = violations.get("count", 0)
        violations_amount = violations.get("total_amount", 0)

        # Check requirements
        missing_requirements = []

        # Check if there are unpaid violations
        if violations_count > 0:
            missing_requirements.append(f"⚠️ سداد المخالفات المرورية ({violations_count} مخالفة - {violations_amount} ریال)")

        # Check service block
        if user.get("service_block"):
            missing_requirements.append("⚠️ يوجد إيقاف خدمات - يرجى مراجعة الجهة المختصة")

        # If there are missing requirements (violations), check if there's already a pending request
        if missing_requirements:
            # Check if there's already a pending request for this user
            existing_request = None
            if user_key in USER_REQUESTS:
                for req in USER_REQUESTS[user_key]:
                    if req.get("service_id") == "DRIVER_LICENSE_RENEWAL" and req.get("status") == "pending_payment":
                        existing_request = req
                        break

            # Only create new request if none exists
            if not existing_request:
                req = create_request(user_key, "DRIVER_LICENSE_RENEWAL", status="pending_payment")
            else:
                req = existing_request

            missing_text = "\n".join([f"• {req}" for req in missing_requirements])

            return f"""⏳ طلب تجديد رخصة القيادة بانتظار سداد المخالفات

رقم الطلب: {req['request_id']}
📅 تاريخ الطلب: {datetime.now().strftime("%Y-%m-%d %H:%M")}
الحالة: بانتظار السداد 💰

❌ المتطلبات الناقصة:
{missing_text}

💳 [اضغط هنا للدفع|http://localhost:8000/payment.html?request_id={req['request_id']}&amount={violations_amount}&user={user_key}]

📋 يرجى سداد المخالفات أولاً ثم إعادة المحاولة"""

        # All requirements met - check if there's a pending request that was just paid
        existing_request = None
        if user_key in USER_REQUESTS:
            for req in USER_REQUESTS[user_key]:
                if req.get("service_id") == "DRIVER_LICENSE_RENEWAL" and req.get("status") == "submitted":
                    existing_request = req
                    break

        # If there's already a submitted request, return success with that request
        if existing_request:
            req = existing_request
        else:
            # Create new request only if no submitted request exists
            req = create_request(user_key, "DRIVER_LICENSE_RENEWAL", status="submitted")

        return f"""✅ طلب تجديد رخصة القيادة تم بنجاح!

رقم الطلب: {req['request_id']}
📅 تاريخ الطلب: {datetime.now().strftime("%Y-%m-%d %H:%M")}
الحالة: قيد المعالجة ✅

📋 المتطلبات المستوفاة:
• فحص طبي / نظر ✅
• التأمين ساري المفعول ✅
• دفع رسوم التجديد ✅
• لا توجد مخالفات غير مسددة ✅"""

    # Passport Renewal
    if intent == "passport_renewal":
        passport = user.get("passport", {})

        if not passport:
            return "⚠️ لا يمكن العثور على معلومات جواز السفر."

        # Create service request
        create_request(user_key, "PASSPORT_RENEWAL")

        return f"""✅ طلب تجديد جواز السفر تم بنجاح!

📋 المتطلبات:
• صلاحية الجواز (قبل انتهاءه بـ 6 أشهر)
• دفع رسوم التجديد
• عدم وجود بلاغ فقدان
• الإقامة سارية (للمقيمين)

⏱️ مدة التنفيذ: 1-3 أيام عمل
📍 يمكنك استلام الجواز من مكتب الجوازات أو عبر البريد"""

    # Pay Violations - Direct payment for traffic violations
    if intent == "pay_violations":
        violations = user.get("violations", {})
        violations_count = violations.get("count", 0)
        violations_amount = violations.get("total_amount", 0)

        # Check if there are any violations to pay
        if violations_count == 0:
            return "✅ لا توجد مخالفات مرورية مسجلة باسمك حالياً"

        # Create a payment request (no service request needed, just for tracking)
        req_id = f"PAY-{len(REQUESTS)+1:04d}"

        return f"""💰 سداد المخالفات المرورية

📋 التفاصيل:
• عدد المخالفات: {violations_count} مخالفة
• المبلغ الإجمالي: {violations_amount} ریال

💳 [اضغط هنا للدفع|http://localhost:8000/payment.html?request_id={req_id}&amount={violations_amount}&user={user_key}&service=violations]

ℹ️ بعد الدفع، سيتم إزالة جميع المخالفات من سجلك فوراً"""

    # Fraud/Scam Detection
    if intent == "fraud_scam":
        return """🚨 تحذير من الاحتيال:

✅ الحقائق:
• جميع خدمات أبشر الحكومية مجانية أو برسوم رسمية محددة
• لا يتم طلب أي دفعات عبر رسائل نصية أو مكالمات
• الدفع يتم فقط عبر تطبيق أبشر الرسمي أو منصة سداد
• نقل ملكية اللوحات يتم عبر نظام موثق وآمن

❌ احذر من:
• المكالمات أو الرسائل التي تطلب دفع أموال
• طلبات مشاركة بياناتك الشخصية
• روابط مشبوهة تدّعي أنها من أبشر
• عروض بيع لوحات بأسعار مشبوهة

📞 للبلاغ عن الاحتيال:
• اتصل على 1909 (مركز الاتصال الموحد)
• قدم بلاغ عبر تطبيق كلنا أمن"""

    # Dashboard - Show all services
    if intent == "dashboard":
        # Helper function to calculate days until expiry
        def days_until(expiry_str):
            try:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
                return (expiry - datetime.now()).days
            except:
                return None

        # Collect all service information
        identity = user.get("national_id") or user.get("iqama")
        license = user.get("driver_license", {})
        passport = user.get("passport", {})
        violations = user.get("violations", {})
        wallet = user.get("wallet", {})
        doc_type = "الهوية الوطنية" if user["user_type"] == "المواطن" else "الإقامة"

        # Build comprehensive dashboard message
        dashboard_msg = f"📊 **ملخص الخدمات لـ {user['name']}**\n\n"

        # 1. Identity Status
        if identity:
            id_days = days_until(identity.get("expiry_date", ""))
            if id_days is not None:
                if id_days < 0:
                    dashboard_msg += f"🔴 **{doc_type}:** منتهية منذ {abs(id_days)} يوم - جدد فوراً!\n"
                elif id_days <= 30:
                    dashboard_msg += f"🟡 **{doc_type}:** تنتهي خلال {id_days} يوم - جدد قريباً\n"
                else:
                    dashboard_msg += f"✅ **{doc_type}:** سارية ({id_days} يوم متبقي)\n"

        # 2. Driver License
        if license:
            license_days = days_until(license.get("expiry_date", ""))
            if license_days is not None:
                if license_days < 0:
                    dashboard_msg += f"🔴 **رخصة القيادة:** منتهية منذ {abs(license_days)} يوم - جدد فوراً!\n"
                elif license_days <= 30:
                    dashboard_msg += f"🟡 **رخصة القيادة:** تنتهي خلال {license_days} يوم\n"
                else:
                    dashboard_msg += f"✅ **رخصة القيادة:** سارية ({license_days} يوم متبقي)\n"

        # 3. Passport
        if passport:
            passport_days = days_until(passport.get("expiry_date", ""))
            if passport_days is not None:
                if passport_days < 0:
                    dashboard_msg += f"🔴 **جواز السفر:** منتهي منذ {abs(passport_days)} يوم\n"
                elif passport_days <= 180:  # 6 months warning for passport
                    dashboard_msg += f"🟡 **جواز السفر:** تنتهي خلال {passport_days} يوم\n"
                else:
                    dashboard_msg += f"✅ **جواز السفر:** ساري ({passport_days} يوم متبقي)\n"

        # 4. Violations
        if violations:
            count = violations.get("count", 0)
            total = violations.get("total_amount", 0)
            if count > 0:
                dashboard_msg += f"⚠️ **المخالفات المرورية:** {count} مخالفة - المبلغ: {total} ریال\n"
            else:
                dashboard_msg += f"✅ **المخالفات المرورية:** لا توجد مخالفات\n"

        # 5. Vehicle Plates in Wallet
        if wallet:
            plates = wallet.get("plates", [])
            if plates:
                dashboard_msg += f"🚗 **محفظة اللوحات:** {len(plates)} لوحة - {', '.join(plates)}\n"
            else:
                dashboard_msg += f"📭 **محفظة اللوحات:** فارغة\n"

        # 6. Recent Requests
        recent_requests = USER_REQUESTS.get(user_key, [])[-3:]
        if recent_requests:
            dashboard_msg += f"\n📋 **آخر الطلبات ({len(recent_requests)}):**\n"
            for req in recent_requests:
                status_emoji = "✅" if req["status"] == "completed" else "⏳"
                dashboard_msg += f"  {status_emoji} {req['service_id']} - {req['status']}\n"

        dashboard_msg += f"\n💡 **نصيحة:** يمكنك قول \"جدد رخصتي\" أو \"انقل اللوحة ABC123\" لبدء أي خدمة"

        log_audit_event(user_key, "dashboard_viewed", {})
        return dashboard_msg

    # Plate Transfer
    if intent == "plate_transfer":
        wallet = user.get("wallet", {})
        plates = wallet.get("plates", [])

        if not plates:
            return "❌ لا توجد لوحات في محفظتك حالياً."

        # Try to extract transfer details from user text using GPT
        if user_text:
            extraction_prompt = f"""
Extract vehicle plate transfer details from this Arabic text.
Return ONLY valid JSON with these exact fields (use null if not found):
{{
  "plate": "ABC123",
  "to_user": "Sara",
  "price": 2000
}}

Available users: Ahmed, Sara, Mohammed, Alex
User's available plates: {', '.join(plates)}

User text: "{sanitize_for_llm(user_text)}"

Return ONLY the JSON object, nothing else.
"""
            try:
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.1,
                    max_tokens=100
                )

                import json
                extracted = json.loads(response.choices[0].message.content.strip())

                plate = extracted.get("plate")
                to_user = extracted.get("to_user")
                price = extracted.get("price", 0)

                # If we have enough info, show confirmation instead of executing immediately
                if plate and to_user:
                    # Validate
                    validation = validate_plate_transfer(user_key, to_user, plate)
                    if not validation["valid"]:
                        errors_text = "\n".join([f"• {err}" for err in validation["errors"]])
                        return f"❌ لا يمكن إتمام النقل:\n\n{errors_text}"

                    # Check fraud
                    fraud_check = detect_plate_transfer_fraud(user_key, to_user, plate, price or 0)

                    if fraud_check["recommendation"] == "BLOCK":
                        warnings_text = "\n".join(fraud_check["warnings"])
                        return f"🚫 تم إيقاف العملية:\n\n{warnings_text}\n\nيرجى مراجعة أقرب مكتب مرور."

                    # Store pending transfer details in STATE
                    STATE["pending_transfer"] = {
                        "from_user": user_key,
                        "to_user": to_user,
                        "plate": plate,
                        "price": price or 0,
                        "fraud_check": fraud_check
                    }

                    # Get buyer name
                    buyer = USERS.get(to_user, {})
                    buyer_name = buyer.get("name", to_user)
                    seller_name = user.get("name", user_key)

                    warnings_text = "\n".join(fraud_check["warnings"]) if fraud_check["warnings"] else "✅ لا توجد مؤشرات احتيال"

                    # Return confirmation message instead of executing
                    return f"""⏳ تأكيد نقل ملكية اللوحة

📋 تفاصيل العملية المطلوبة:
• البائع: {seller_name}
• المشتري: {buyer_name}
• اللوحة: {plate}
• السعر: {price or 0} ريال

{warnings_text}

⚠️ تنبيه: النقل نهائي ولا يمكن التراجع عنه

[✅ تأكيد النقل|CONFIRM_TRANSFER:{user_key}:{to_user}:{plate}:{price or 0}]
[❌ إلغاء|CANCEL]"""

            except Exception as e:
                logger.warning(f"Failed to extract transfer details: {e}")
                # Fall through to show general info

        # If extraction failed or no text, show general info
        return f"""ℹ️ خدمة نقل ملكية اللوحات:

📋 اللوحات المتوفرة في محفظتك:
{', '.join(plates)}

لنقل لوحة، يرجى تحديد:
• اللوحة المراد نقلها
• اسم المشتري
• السعر (اختياري)

مثال: "انقل اللوحة ABC123 إلى محمد بسعر 5000 ريال"

⚠️ ملاحظات مهمة:
• النقل نهائي ولا يمكن التراجع عنه
• يجب التأكد من هوية المشتري
• العملية موثقة في سجل دائم"""

    # Unknown
    return "عذراً، لم أفهم طلبك. يمكنك تجربة:\n• جدد رخصتي\n• كم باقي على الإقامة؟\n• انقل لوحة\n• هل الخدمة مجانية؟"

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {"status": "ok", "service": "ABSHER Digital Services API", "version": "2.0.0"}

@app.get("/payment.html")
async def get_payment_page():
    """Serve the payment page"""
    payment_file_path = os.path.join(os.path.dirname(__file__), "payment.html")
    if os.path.exists(payment_file_path):
        return FileResponse(payment_file_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Payment page not found")

@app.get("/api/users")
def get_users(request: Request):
    """Get all users (for demo only - in production, use authentication)"""
    log_audit_event("system", "users_list_accessed", {"ip": request.client.host})
    return USERS

@app.get("/api/state")
def get_state():
    """Get current state"""
    return STATE

@app.get("/api/notification/{user_key}")
def get_user_notification(user_key: str):
    """Get personalized notification for a user"""
    if user_key not in USERS:
        raise HTTPException(status_code=404, detail="User not found")

    user = USERS[user_key]
    identity = user.get("national_id") or user.get("iqama")

    if identity:
        identity_status = identity.get("status", "غير متوفر")
        identity_expiry = identity.get("expiry_date", "غير متوفر")
    else:
        identity_status = "غير متوفر"
        identity_expiry = "غير متوفر"

    violations = user.get("violations", {})
    violations_count = violations.get("count", 0)
    violations_amount = violations.get("total_amount", 0)

    notification = f"مرحباً {user['name']}! 👋 "

    if identity_status == "expired":
        notification += f"❌ الهوية/الإقامة منتهية. يرجى التجديد فوراً."
    elif identity_status == "near_expiry":
        notification += f"⚠️ الهوية/الإقامة تقترب من الانتهاء ({identity_expiry})."
    else:
        notification += f"✅ الهوية/الإقامة سارية حتى {identity_expiry}."

    if violations_count > 0:
        notification += f" لديك {violations_count} مخالفة بقيمة {violations_amount} ريال."

    return {
        "user_key": user_key,
        "notification": notification,
        "user": user
    }

@app.post("/api/command")
async def process_text(cmd: TextCommand, request: Request):
    """Process text command"""
    try:
        text = normalize(cmd.text)
        cur = cmd.user_key if cmd.user_key and cmd.user_key in USERS else STATE["current_user_key"]
        STATE["current_user_key"] = cur

        log_audit_event(cur, "text_command", {"text": sanitize_for_logging(cmd.text)}, request.client.host)

        # Detect intent
        intent = detect_intent(text)

        # Generate response
        if intent in ["info", "unknown"]:
            visual = generate_conversational_response(cmd.text, cur)
        else:
            visual = handle_intent(cur, intent, cmd.text)
            cur = STATE["current_user_key"]

        STATE["last_visual"] = visual

        return {
            "intent": intent,
            "text": cmd.text,
            "current_user": USERS[cur],
            "visual": visual,
            "action_steps": "",
            "recent_requests": STATE["recent_requests"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Command processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في معالجة الطلب")

@app.post("/api/switch-user")
def switch_user(request_data: SwitchUserRequest):
    """Switch current user"""
    user_key = request_data.user_key

    STATE["current_user_key"] = user_key
    STATE["conversation_history"] = []
    STATE["pending_action"] = None
    STATE["pending_intent"] = None

    # Load user's recent requests
    STATE["recent_requests"] = USER_REQUESTS.get(user_key, [])[-3:]

    log_audit_event(user_key, "user_switched", {"from": STATE.get("current_user_key")})

    return {"current_user": USERS[user_key]}

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...), request: Request = None):
    """Upload user photo for identity document renewal"""
    try:
        contents = await file.read()

        # Validate file
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="الملف فارغ")

        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"نوع الملف غير مدعوم: {file.content_type}")

        max_size = 5 * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(status_code=400, detail="حجم الصورة كبير جداً (الحد الأقصى 5MB)")

        # Magic byte validation
        if not (contents[:2] == b'\xff\xd8' or contents[:8] == b'\x89PNG\r\n\x1a\n'):
            raise HTTPException(status_code=400, detail="الملف ليس صورة صحيحة")

        user_key = STATE["current_user_key"]

        log_audit_event(
            user_key,
            "photo_uploaded",
            {"filename": file.filename, "size": len(contents)},
            request.client.host if request else "unknown"
        )

        return {
            "success": True,
            "message": "تم رفع الصورة بنجاح! ✅",
            "file_name": file.filename,
            "file_size": len(contents),
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Photo upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء رفع الصورة")

@app.post("/api/transfer-plate")
async def transfer_plate(transfer: PlateTransferRequest, request: Request):
    """Transfer vehicle plate ownership"""
    try:
        # Rate limit: 5 transfers per hour per user
        if not rate_limiter.is_allowed(f"transfer:{transfer.from_user}", 5, 3600):
            raise HTTPException(status_code=429, detail="تم تجاوز الحد الأقصى لعمليات النقل (5 في الساعة)")

        # Validate transfer
        validation = validate_plate_transfer(transfer.from_user, transfer.to_user, transfer.plate)
        if not validation["valid"]:
            return {
                "success": False,
                "errors": validation["errors"]
            }

        # Check for fraud
        fraud_check = detect_plate_transfer_fraud(
            transfer.from_user,
            transfer.to_user,
            transfer.plate,
            transfer.price
        )

        # Block if high fraud score
        if fraud_check["recommendation"] == "BLOCK":
            log_audit_event(
                transfer.from_user,
                "plate_transfer_blocked_fraud",
                {
                    "to_user": transfer.to_user,
                    "plate": transfer.plate,
                    "fraud_score": fraud_check["fraud_score"],
                    "warnings": fraud_check["warnings"]
                },
                request.client.host
            )

            return {
                "success": False,
                "fraud_detected": True,
                "fraud_score": fraud_check["fraud_score"],
                "warnings": fraud_check["warnings"],
                "message": "تم إيقاف العملية بسبب اكتشاف نشاط مشبوه. يرجى مراجعة أقرب مكتب مرور."
            }

        # Require confirmation if suspicious
        if fraud_check["recommendation"] == "CONFIRM":
            return {
                "success": False,
                "requires_confirmation": True,
                "fraud_score": fraud_check["fraud_score"],
                "warnings": fraud_check["warnings"],
                "message": "تم اكتشاف مؤشرات تحتاج تأكيد إضافي. يرجى التأكد من البيانات والمتابعة."
            }

        # Execute transfer
        result = execute_plate_transfer(
            transfer.from_user,
            transfer.to_user,
            transfer.plate,
            transfer.price
        )

        if result["success"]:
            return {
                "success": True,
                "transaction_id": result["transaction_id"],
                "message": result["message"],
                "before_state": result["before_state"],
                "after_state": result["after_state"],
                "warnings": fraud_check["warnings"]
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plate transfer error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في عملية النقل")

@app.post("/api/payment")
async def process_payment(payment: PaymentRequest, request: Request):
    """Process payment for violations or services"""
    try:
        # Rate limit: 10 payments per hour per user
        if not rate_limiter.is_allowed(f"payment:{payment.user_id}", 10, 3600):
            raise HTTPException(status_code=429, detail="تم تجاوز الحد الأقصى لعمليات الدفع (10 في الساعة)")

        # Validate user exists
        if payment.user_id not in USERS:
            raise HTTPException(status_code=404, detail="المستخدم غير موجود")

        user = USERS[payment.user_id]

        # Log payment event
        log_audit_event(
            payment.user_id,
            "payment_completed",
            {
                "amount": payment.amount,
                "service": payment.service
            },
            request.client.host
        )

        # Clear violations if payment is for violations
        if payment.service == "violations" or payment.service == "DRIVER_LICENSE_RENEWAL":
            violations = user.get("violations", {})
            if violations.get("total_amount", 0) > 0:
                user["violations"] = {"count": 0, "total_amount": 0, "details": []}

                # Only update license renewal requests if payment is for DRIVER_LICENSE_RENEWAL
                if payment.service == "DRIVER_LICENSE_RENEWAL":
                    # Update any pending requests to submitted status in USER_REQUESTS
                    if payment.user_id in USER_REQUESTS:
                        for req in USER_REQUESTS[payment.user_id]:
                            if req.get("status") == "pending_payment" and req.get("service_id") == "DRIVER_LICENSE_RENEWAL":
                                req["status"] = "submitted"
                                req["payment_timestamp"] = datetime.now().isoformat()

                    # Also update in global REQUESTS list
                    for req in REQUESTS:
                        if req.get("user_key") == payment.user_id and req.get("status") == "pending_payment" and req.get("service_id") == "DRIVER_LICENSE_RENEWAL":
                            req["status"] = "submitted"
                            req["payment_timestamp"] = datetime.now().isoformat()

                    # Update STATE recent_requests
                    STATE["recent_requests"] = USER_REQUESTS.get(payment.user_id, [])[-3:]

                return {
                    "success": True,
                    "message": f"تم سداد المبلغ {payment.amount} ریال بنجاح! ✅",
                    "receipt_id": f"PAY-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
                    "violations_cleared": True
                }

        # Generic payment success
        return {
            "success": True,
            "message": f"تم سداد المبلغ {payment.amount} ریال بنجاح! ✅",
            "receipt_id": f"PAY-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في عملية الدفع")

@app.post("/api/request/approve/{request_id}")
async def approve_request(request_id: str, request: Request):
    """Approve and complete a service request"""
    try:
        # Find the request
        target_request = None
        user_key = None

        for req in REQUESTS:
            if req.get("request_id") == request_id:
                target_request = req
                user_key = req.get("user_key")
                break

        if not target_request:
            raise HTTPException(status_code=404, detail="الطلب غير موجود")

        if not user_key or user_key not in USERS:
            raise HTTPException(status_code=404, detail="المستخدم غير موجود")

        user = USERS[user_key]
        service_id = target_request.get("service_id")

        # Mark request as completed
        target_request["status"] = "completed"
        target_request["completed_timestamp"] = datetime.now().isoformat()

        # Update the request in USER_REQUESTS as well
        if user_key in USER_REQUESTS:
            for req in USER_REQUESTS[user_key]:
                if req.get("request_id") == request_id:
                    req["status"] = "completed"
                    req["completed_timestamp"] = datetime.now().isoformat()

        # Update STATE recent_requests
        STATE["recent_requests"] = USER_REQUESTS.get(user_key, [])[-3:]

        # Perform service-specific actions
        if service_id == "ID_RENEWAL":
            # Renew the ID/Iqama - extend expiry by 5 years (Gregorian)
            if user["user_type"] == "المواطن" and "national_id" in user:
                current_expiry = datetime.strptime(user["national_id"]["expiry_date"], "%Y-%m-%d")
                new_expiry = current_expiry + timedelta(days=365*5)
                user["national_id"]["expiry_date"] = new_expiry.strftime("%Y-%m-%d")
                user["national_id"]["status"] = "valid"  # Update status to valid
            elif "iqama" in user:
                current_expiry = datetime.strptime(user["iqama"]["expiry_date"], "%Y-%m-%d")
                new_expiry = current_expiry + timedelta(days=365*5)
                user["iqama"]["expiry_date"] = new_expiry.strftime("%Y-%m-%d")
                user["iqama"]["status"] = "valid"  # Update status to valid

        elif service_id == "DRIVER_LICENSE_RENEWAL":
            # Renew driver license - extend expiry by 10 years
            if "driver_license" in user:
                current_expiry = datetime.strptime(user["driver_license"]["expiry_date"], "%Y-%m-%d")
                new_expiry = current_expiry + timedelta(days=365*10)
                user["driver_license"]["expiry_date"] = new_expiry.strftime("%Y-%m-%d")

        elif service_id == "PASSPORT_RENEWAL":
            # Renew passport - extend expiry by 5 years
            if "passport" in user:
                current_expiry = datetime.strptime(user["passport"]["expiry_date"], "%Y-%m-%d")
                new_expiry = current_expiry + timedelta(days=365*5)
                user["passport"]["expiry_date"] = new_expiry.strftime("%Y-%m-%d")

        # Log audit event
        log_audit_event(
            user_key,
            "request_approved",
            {
                "request_id": request_id,
                "service_id": service_id
            },
            request.client.host
        )

        return {
            "success": True,
            "message": f"تم اعتماد الطلب {request_id} بنجاح! ✅",
            "request": target_request,
            "user": user
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request approval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في اعتماد الطلب")

@app.post("/api/plate-transfer/confirm")
async def confirm_plate_transfer(request: Request):
    """Confirm and execute pending plate transfer"""
    try:
        # Get pending transfer from STATE
        pending = STATE.get("pending_transfer")
        if not pending:
            raise HTTPException(status_code=400, detail="لا يوجد طلب نقل معلق")

        from_user = pending["from_user"]
        to_user = pending["to_user"]
        plate = pending["plate"]
        price = pending["price"]
        fraud_check = pending["fraud_check"]

        # Validate users still exist
        if from_user not in USERS or to_user not in USERS:
            raise HTTPException(status_code=404, detail="المستخدم غير موجود")

        # Execute transfer
        result = execute_plate_transfer(from_user, to_user, plate, price)

        if result["success"]:
            # Create request record for seller
            transfer_request = create_request(
                from_user,
                "PLATE_TRANSFER",
                status="completed"
            )

            # Add additional transfer details
            transfer_request.update({
                "transaction_id": result['transaction_id'],
                "plate": plate,
                "from_user": from_user,
                "to_user": to_user,
                "price": price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Update recent requests
            STATE["recent_requests"] = REQUESTS[-3:]

            # Clear pending transfer
            STATE["pending_transfer"] = None

            # Prepare payment link for buyer
            buyer = USERS[to_user]
            payment_link = f"http://localhost:8000/payment.html?request_id={transfer_request['request_id']}&amount={price}&user={to_user}&service=PLATE_TRANSFER"

            warnings_text = "\n".join(fraud_check["warnings"]) if fraud_check["warnings"] else "✅ لا توجد مؤشرات احتيال"

            # Create notification for buyer
            buyer_notification = f"""📩 إشعار: نقل لوحة جديدة

تم نقل اللوحة **{plate}** إليك من {result['before_state']['seller']['name']}

📋 التفاصيل:
• اللوحة: {plate}
• السعر: {price} ريال
• رقم المعاملة: {result['transaction_id']}

💳 يرجى إتمام الدفع لاستكمال النقل:
{payment_link}"""

            return {
                "success": True,
                "message": f"✅ {result['message']}",
                "transaction_id": result['transaction_id'],
                "request_id": transfer_request['request_id'],
                "seller_name": result['before_state']['seller']['name'],
                "buyer_name": result['after_state']['buyer']['name'],
                "plate": plate,
                "price": price,
                "timestamp": transfer_request['timestamp'],
                "warnings": warnings_text,
                "payment_link": payment_link,
                "buyer_notification": buyer_notification,
                "buyer_key": to_user
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "خطأ غير معروف")
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plate transfer confirmation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في تأكيد النقل")

@app.post("/api/voice")
async def process_voice(
    request: Request,
    file: UploadFile = File(...),
    user_key: Optional[str] = Form(None)
):
    """Process voice input with rate limiting"""
    # Rate limit: 10 voice requests per minute
    client_ip = request.client.host
    if not rate_limiter.is_allowed(f"voice:{client_ip}", 10, 60):
        raise HTTPException(status_code=429, detail="تم تجاوز الحد الأقصى للطلبات الصوتية (10 في الدقيقة)")

    webm_path = None
    wav_path = None

    try:
        # Validate content type
        allowed_audio_types = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav", "application/octet-stream", ""}
        base_content_type = (file.content_type or "").split(";")[0].strip()

        if base_content_type not in allowed_audio_types:
            raise HTTPException(status_code=400, detail=f"نوع الملف الصوتي غير مدعوم: {file.content_type}")

        # Read and validate file size
        audio_bytes = await file.read()
        max_size = 10 * 1024 * 1024

        if len(audio_bytes) > max_size:
            raise HTTPException(status_code=400, detail="حجم الملف الصوتي كبير جداً (الحد الأقصى 10MB)")

        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="الملف الصوتي صغير جداً أو فارغ")

        # Create temp files with validated paths
        temp_dir = tempfile.gettempdir()
        webm_path = os.path.join(temp_dir, f"voice_{uuid.uuid4().hex}.webm")
        wav_path = os.path.join(temp_dir, f"voice_{uuid.uuid4().hex}.wav")

        if not validate_temp_path(webm_path) or not validate_temp_path(wav_path):
            raise HTTPException(status_code=500, detail="خطأ في إنشاء الملفات المؤقتة")

        # Save uploaded audio
        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to wav using FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", webm_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )

        if result.returncode != 0:
            stderr_msg = result.stderr.decode('utf-8', errors='ignore')
            logger.error(f"FFmpeg conversion failed: {stderr_msg}")
            raise HTTPException(status_code=500, detail="فشل تحويل الملف الصوتي")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise HTTPException(status_code=500, detail="فشل تحويل الملف الصوتي")

        # Whisper transcription
        try:
            transcription_result = whisper_model.transcribe(
                wav_path,
                language="ar",
                fp16=USE_FP16,
                beam_size=5,
                patience=1.0,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6
            )
            text = normalize(transcription_result["text"])

            logger.info(f"Transcribed: {sanitize_for_logging(text)}")

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="فشل تحويل الصوت إلى نص")

        # Process intent
        cur = user_key if user_key and user_key in USERS else STATE["current_user_key"]
        STATE["current_user_key"] = cur

        log_audit_event(cur, "voice_command", {"text": sanitize_for_logging(text)}, client_ip)

        intent = detect_intent(text)

        if intent in ["info", "unknown"]:
            visual = generate_conversational_response(text, cur)
        else:
            visual = handle_intent(cur, intent, text)
            cur = STATE["current_user_key"]

        # Generate TTS response
        audio_output = text_to_speech(visual)

        if audio_output is None:
            logger.warning("TTS failed, returning text-only response")
            return {
                "intent": intent,
                "text": text,
                "current_user": USERS[cur],
                "visual": visual,
                "action_steps": "",
                "recent_requests": STATE["recent_requests"],
                "error": "تعذر تحويل النص إلى صوت"
            }

        audio_base64 = base64.b64encode(audio_output).decode('utf-8')

        STATE["last_visual"] = visual

        return {
            "intent": intent,
            "text": text,
            "current_user": USERS[cur],
            "visual": visual,
            "action_steps": "",
            "recent_requests": STATE["recent_requests"],
            "audio": audio_base64,
            "audio_format": "mp3"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ في معالجة الصوت")

    finally:
        # Cleanup temp files
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up: {os.path.basename(path)}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "app_secure:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
