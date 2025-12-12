

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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

    prompt = f"""
You are an intent classifier for a Saudi government services assistant (ABSHER).
Classify the following user text into ONE intent:

SERVICE INTENTS (specific services):
- id_renewal: User wants to renew ID/Iqama (تجديد الهوية/الإقامة)
- id_status: User wants to check ID/Iqama expiry status (الاستعلام عن الصلاحية)
- driver_license_renewal: User wants to renew driver license (تجديد رخصة القيادة)
- passport_renewal: User wants to renew passport (تجديد جواز السفر)
- plate_transfer: User wants to transfer vehicle plate ownership (نقل ملكية لوحة)

OTHER INTENTS:
- info: General questions about services, how things work, requirements, procedures (معلومات عامة)
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
    Convert text to speech using OpenAI TTS API.
    Returns mp3 bytes or None on failure.
    """
    try:
        # Truncate very long text
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters for TTS")

        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="onyx",
            input=text,
            response_format="mp3"
        )
        audio_bytes = response.read()

        if not audio_bytes or len(audio_bytes) < 100:
            logger.error("TTS returned empty or invalid audio")
            return None

        return audio_bytes

    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)
        return None

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

                # If we have enough info, execute the transfer
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

                    # Execute transfer
                    result = execute_plate_transfer(user_key, to_user, plate, price or 0)

                    if result["success"]:
                        # Create request record for both users
                        transfer_request = create_request(
                            user_key,
                            "PLATE_TRANSFER",
                            status="completed"
                        )

                        # Add additional transfer details to the request
                        transfer_request.update({
                            "transaction_id": result['transaction_id'],
                            "plate": plate,
                            "from_user": user_key,
                            "to_user": to_user,
                            "price": price,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # Update recent requests
                        STATE["recent_requests"] = REQUESTS[-3:]

                        warnings_text = "\n".join(fraud_check["warnings"])
                        return f"""✅ {result['message']}

📋 تفاصيل العملية:
• رقم المعاملة: {result['transaction_id']}
• رقم الطلب: {transfer_request['request_id']}
• من: {result['before_state']['seller']['name']}
• إلى: {result['after_state']['buyer']['name']}
• اللوحة: {plate}
• السعر: {price} ريال
• التاريخ: {transfer_request['timestamp']}

{warnings_text}

📝 تم تسجيل العملية في السجل الدائم."""
                    else:
                        return f"❌ فشل النقل: {result.get('error', 'خطأ غير معروف')}"

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
