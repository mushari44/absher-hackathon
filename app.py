from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
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
        "vehicle_registration": {
            "status": "expired",
            "expiry_date": "2025-07-20"
        },
        "insurance": {
            "status": "expired",
            "expiry_date": "2025-08-01"
        },
        "periodic_check": {
            "status": "expired",
            "expiry_date": "2025-06-30"
        },
        "medical_check": {
            "required": True
        },
        "violations": {
            "count": 7,
            "total_amount": 3250,
            "service_block": True
        },
        "dependents": [
            {
                "name": "Fahad",
                "age": 15,
                "needs_first_id": True
            },
            {
                "name": "Lama",
                "age": 9,
                "needs_family_card_update": True
            }
        ],
        "domestic_workers": [
            {
                "worker_name": "Mary",
                "iqama_expiry": "2025-12-25",
                "needs_renewal": True
            }
        ],
        "reports": {
            "lost_id_report": "closed"
        },
        "national_address": {
            "updated": False
        },
        "driving_authorization": {
            "status": "expired"
        },
        "weapon_license": {
            "has_license": False
        },
        "hajj_permit": {
            "status": "none"
        },
        "travel_permit": {
            "for_son": True
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
        "passport": {
            "status": "valid",
            "expiry_date": "2031-09-20"
        },
        "vehicle_registration": {
            "status": "valid",
            "expiry_date": "2027-02-01"
        },
        "insurance": {
            "status": "valid",
            "expiry_date": "2026-03-10"
        },
        "periodic_check": {
            "status": "valid",
            "expiry_date": "2026-05-22"
        },
        "medical_check": {
            "required": False
        },
        "violations": {
            "count": 1,
            "total_amount": 300,
            "service_block": False
        },
        "reports": {
            "lost_bank_card": "transferred"
        },
        "national_address": {
            "updated": True
        },
        "driving_authorization": {
            "status": "valid",
            "authorized_for": "her_sister"
        },
        "hajj_permit": {
            "status": "old",
            "year": 2024
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
        "passport": {
            "status": "near_expiry",
            "expiry_date": "2026-04-20"
        },
        "vehicle_registration": {
            "status": "valid",
            "expiry_date": "2027-07-15"
        },
        "insurance": {
            "status": "expired",
            "expiry_date": "2025-09-25"
        },
        "periodic_check": {
            "status": "valid",
            "expiry_date": "2026-06-01"
        },
        "medical_check": {
            "required": True
        },
        "violations": {
            "count": 2,
            "total_amount": 250,
            "service_block": False
        },
        "reports": {
            "accident_hit_and_run": True
        },
        "national_address": {
            "updated": False
        },
        "driving_authorization": {
            "status": "needs_cancel",
            "reason": "vehicle_sold"
        },
        "weapon_license": {
            "status": "expired",
            "expiry_date": "2024-11-30"
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
        "passport": {
            "status": "valid",
            "expiry_date": "2029-03-15"
        },
        "vehicle_registration": {
            "status": "valid",
            "expiry_date": "2027-08-22"
        },
        "insurance": {
            "status": "expired",
            "expiry_date": "2025-10-10"
        },
        "periodic_check": {
            "status": "expired",
            "expiry_date": "2025-09-01"
        },
        "medical_check": {
            "required": False
        },
        "violations": {
            "count": 0,
            "total_amount": 0
        },
        "national_address": {
            "updated": False
        },
        "driving_authorization": {
            "status": "valid",
            "authorized_for": "co-worker"
        },
        "reports": {
            "lost_license": "closed"
        }
    }
}

SERVICES = {
    "ID_RENEWAL": {"service_id": "2001", "name": "تجديد الهوية/الإقامة"},
    "ID_STATUS": {"service_id": "2002", "name": "الاستعلام عن الصلاحية"},
    "DRIVER_LICENSE_RENEWAL": {"service_id": "3001", "name": "تجديد رخصة القيادة"},
    "PASSPORT_RENEWAL": {"service_id": "4001", "name": "تجديد جواز السفر"},
}

REQUESTS = []

STATE = {
    "current_user_key": "Ahmed",
    "last_visual": "",
    "recent_requests": [],
    "conversation_history": [],  # Store conversation context for intelligent responses
    "pending_action": None,  # Track pending actions that need user confirmation
    "pending_intent": None  # Track the intent that's waiting for confirmation
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

SERVICE INTENTS (specific services):
- id_renewal: User wants to renew ID/Iqama (تجديد الهوية/الإقامة)
- id_status: User wants to check ID/Iqama expiry status (الاستعلام عن الصلاحية)
- driver_license_renewal: User wants to renew driver license (تجديد رخصة القيادة)
- passport_renewal: User wants to renew passport (تجديد جواز السفر)

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
- "كيف أجدد رخصة القيادة؟" → info
- "هل الخدمة مجانية؟" → info
- "وصلتني رسالة تطلب دفع رسوم، هل هذا صحيح؟" → fraud_scam
- "شخص اتصل بي وطلب فلوس لتجديد الإقامة" → fraud_scam

User text: "{user_text}"
Return ONLY the intent name (lowercase with underscores).
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        intent = response.choices[0].message.content.strip().lower()

        valid = [
            "id_renewal", "id_status", "driver_license_renewal", "passport_renewal",
            "info", "fraud_scam", "switch_user", "greeting", "unknown"
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


def generate_conversational_response(user_text: str, user_key: str) -> str:
    """
    Generate intelligent conversational response using GPT-4 with full context.
    Uses conversation history for context-aware responses.
    """
    user = USERS[user_key]

    # Build service requirements text
    requirements_text = ""
    for service_intent, requirements in SERVICE_REQUIREMENTS.items():
        service_names = {
            "id_renewal": "تجديد الهوية/الإقامة",
            "driver_license_renewal": "تجديد رخصة القيادة",
            "passport_renewal": "تجديد جواز السفر"
        }
        service_name = service_names.get(service_intent, service_intent)
        requirements_text += f"\n{service_name}:\n"
        for req in requirements:
            requirements_text += f"  • {req}\n"

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

    # Build context from conversation history
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
3. تقديم معلومات دقيقة عن خدمات أبشر ومتطلباتها
4. استخدام اللغة العربية الفصحى البسيطة
5. إذا سُئلت عن خدمة، اشرحها بوضوح مع ذكر متطلباتها
6. إذا سُئلت عن المتطلبات، اذكرها من القائمة أدناه
7. كن مفيداً وودوداً

الخدمات المتاحة ومتطلباتها:
{requirements_text}

ملاحظات مهمة:
• تجديد رخصة القيادة: يتطلب عدم وجود مخالفات مرورية وفحص طبي ساري
• تجديد جواز السفر: متاح للمواطنين السعوديين فقط
• تجديد الهوية/الإقامة: للمقيمين يجب التأكد من التأمين الصحي الساري
• جميع الخدمات مجانية أو برسوم رسمية فقط. لا تطلب أبشر أبداً دفعات عبر رسائل أو مكالمات."""}
    ]

    # Add conversation history
    for msg in STATE["conversation_history"][-10:]:  # Last 10 messages for context
        messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": user_text})

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

        # Keep only last 20 messages (10 exchanges)
        if len(STATE["conversation_history"]) > 20:
            STATE["conversation_history"] = STATE["conversation_history"][-20:]

        return assistant_response

    except Exception as e:
        print(f"❌ Conversational response failed: {e}")
        return "عذراً، حدث خطأ. يمكنك المحاولة مرة أخرى."



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

    # Switch user
    if intent == "switch_user":
        if "ahmed" in user_key.lower():
            STATE["current_user_key"] = "Ahmed"
        elif "sara" in user_key.lower():
            STATE["current_user_key"] = "Sara"
        elif "mohammed" in user_key.lower():
            STATE["current_user_key"] = "Mohammed"
        elif "alex" in user_key.lower():
            STATE["current_user_key"] = "Alex"
        else:
            STATE["current_user_key"] = "Ahmed"  # Default to Ahmed
        return "🔄 تم تغيير المستخدم."

    # Greeting
    if intent == "greeting":
        return f"مرحباً {user['name']}! 👋 كيف يمكنني مساعدتك اليوم؟"

    # ID/Iqama Renewal
    if intent == "id_renewal":
        # Validate requirements first
        validation = validate_service_requirements(user_key, intent)
        if not validation["valid"]:
            return validation["message"]

        # All requirements met, process automatically
        req = create_request(user_key, "ID_RENEWAL")
        return f"✅ تم تجديد {'هويتك' if user['user_type'] == 'المواطن' else 'إقامتك'} بنجاح!\n\nرقم الطلب: {req['request_id']}\nالحالة: قيد المعالجة\n\nسيتم إرسال رسالة نصية عند جاهزية الوثيقة للاستلام."

    # Check ID/Iqama Status
    if intent == "id_status":
        # Get identity document
        identity = user.get("national_id") or user.get("iqama")
        doc_type = "هويتك" if user["user_type"] == "المواطن" else "إقامتك"

        if not identity:
            return f"⚠️ لا يمكن العثور على معلومات {doc_type}."

        status = identity.get("status")
        expiry_date_str = identity.get("expiry_date")

        if status == "expired":
            return f"⚠️ تنبيه: {doc_type} منتهية!\nتاريخ الانتهاء: {expiry_date_str}\nيرجى المبادرة بالتجديد فوراً."

        # Calculate days left
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        days_left = (expiry_date - datetime.now()).days

        if days_left < 0:
            return f"⚠️ تنبيه: {doc_type} منتهية منذ {abs(days_left)} يوم!\nتاريخ الانتهاء: {expiry_date_str}\nيرجى المبادرة بالتجديد فوراً."
        elif status == "near_expiry" or days_left <= 30:
            return f"⚠️ تنبيه: {doc_type} تنتهي خلال {days_left} يوم!\nتاريخ الانتهاء: {expiry_date_str}\nننصح بالتجديد في أقرب وقت."
        else:
            return f"✅ {doc_type} سارية المفعول.\nتاريخ الانتهاء: {expiry_date_str}\nمتبقي: {days_left} يوم"

    # Driver License Renewal
    if intent == "driver_license_renewal":
        # Validate requirements first
        validation = validate_service_requirements(user_key, intent)
        if not validation["valid"]:
            return validation["message"]

        # All requirements met, process automatically
        req = create_request(user_key, "DRIVER_LICENSE_RENEWAL")
        return f"✅ تم تجديد رخصة القيادة بنجاح!\n\nرقم الطلب: {req['request_id']}\nالحالة: قيد المعالجة\nالرسوم: 400 ریال\n\nسيتم إرسال رسالة نصية لك عند جاهزية الرخصة الجديدة."

    # Passport Renewal
    if intent == "passport_renewal":
        # Validate requirements first
        validation = validate_service_requirements(user_key, intent)
        if not validation["valid"]:
            return validation["message"]

        # All requirements met, process automatically
        req = create_request(user_key, "PASSPORT_RENEWAL")
        return f"✅ تم تجديد جواز السفر بنجاح!\n\nرقم الطلب: {req['request_id']}\nالحالة: قيد المعالجة\nالمدة المتوقعة: 3-5 أيام عمل\n\nسيتم إرسال رسالة نصية عند جاهزية الجواز للاستلام."

    # General Information
    if intent == "info":
        return """ℹ️ معلومات عن خدمات أبشر:

📋 الخدمات المتاحة:
• تجديد الهوية/الإقامة
• تجديد رخصة القيادة
• تجديد جواز السفر
• الاستعلام عن صلاحية الوثائق

💰 جميع الخدمات الحكومية عبر أبشر مجانية أو برسوم رسمية فقط.
🔒 لا تشارك بياناتك مع أي شخص.
⚠️ احذر من الرسائل المشبوهة التي تطلب دفع رسوم."""

    # Fraud/Scam Detection
    if intent == "fraud_scam":
        return """🚨 تحذير من الاحتيال:

✅ الحقائق:
• جميع خدمات أبشر الحكومية مجانية أو برسوم رسمية محددة
• لا يتم طلب أي دفعات عبر رسائل نصية أو مكالمات
• الدفع يتم فقط عبر تطبيق أبشر الرسمي أو منصة سداد

❌ احذر من:
• المكالمات أو الرسائل التي تطلب دفع أموال
• طلبات مشاركة بياناتك الشخصية
• روابط مشبوهة تدّعي أنها من أبشر

📞 للبلاغ عن الاحتيال:
• اتصل على 1909 (مركز الاتصال الموحد)
• قدم بلاغ عبر تطبيق كلنا أمن

🔐 بياناتك في أمان مع أبشر الرسمي فقط."""

    # Unknown
    return "عذراً، لم أفهم طلبك. يمكنك تجربة:\n• جدد رخصتي\n• كم باقي على الإقامة؟\n• جدد جوازي\n• هل الخدمة مجانية؟"

def text_to_speech(text: str) -> bytes:
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",     # higher quality
            voice="onyx",         # deep male voice
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


def detect_user_confirmation(user_text: str, context: str) -> bool:
    """
    Use LLM to intelligently detect if user is confirming/accepting help.
    More flexible than keyword matching.
    """
    prompt = f"""
هل المستخدم يوافق/يؤكد/يطلب المساعدة في الرسالة التالية؟

السياق: {context}

رسالة المستخدم: "{user_text}"

أجب بـ "yes" إذا كان المستخدم يوافق أو يطلب المساعدة، أو "no" إذا كان يرفض أو يتحدث عن موضوع آخر.

أمثلة:
- "نعم" → yes
- "اي" → yes
- "ساعدني" → yes
- "طيب" → yes
- "لا شكراً" → no
- "كم باقي على الإقامة؟" → no
- "جدد جوازي" → no (طلب جديد وليس تأكيد)

أجب بكلمة واحدة فقط: yes أو no
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "yes"
    except Exception as e:
        print(f"❌ Confirmation detection failed: {e}")
        return False


def generate_payment_link(user_key: str, amount: float, service_type: str = "violations") -> str:
    """
    Generate a mock payment link for violations or other services.
    In production, this would integrate with Sadad or Absher payment gateway.
    """
    user = USERS[user_key]
    user_id = user.get("user_id", "0000")

    # Mock payment link (in production, this would be real Sadad/Absher payment gateway)
    payment_link = f"https://sadad.gov.sa/payment?service={service_type}&user_id={user_id}&amount={amount}"

    return payment_link


def handle_pending_action_help(user_key: str, pending_intent: str, pending_action: dict) -> str:
    """
    Handle user confirmation to get help with missing requirements.
    Provides guidance on how to complete each missing requirement with actionable links.
    """
    user = USERS[user_key]
    missing_requirements = pending_action.get("missing_requirements", [])
    missing_fields = pending_action.get("missing_fields", [])

    service_names = {
        "id_renewal": "تجديد الهوية/الإقامة",
        "driver_license_renewal": "تجديد رخصة القيادة",
        "passport_renewal": "تجديد جواز السفر"
    }

    service_name = service_names.get(pending_intent, pending_intent)

    response = f"حسناً، دعني أساعدك في إكمال متطلبات {service_name}:\n\n"

    # Provide specific guidance for each missing requirement with actionable links
    for i, req in enumerate(missing_requirements, 1):
        if req == "service_blocking_violations":
            violations = user.get("violations", {})
            amount = violations.get("total_amount", 0)
            count = violations.get("count", 0)

            # Generate payment link
            payment_link = generate_payment_link(user_key, amount, "violations")

            response += f"{i}. 💳 المخالفات المرورية (مطلوب السداد):\n"
            response += f"   • عدد المخالفات: {count} مخالفة\n"
            response += f"   • المبلغ الإجمالي: {amount} ریال\n"
            response += f"   • يجب سداد المخالفات لإتمام التجديد\n\n"
            response += f"   🔗 اضغط هنا للسداد الفوري:\n"
            response += f"   {payment_link}\n\n"
            response += f"   📱 أو يمكنك السداد عبر:\n"
            response += f"     - تطبيق أبشر → المخالفات → سداد\n"
            response += f"     - تطبيق سداد (رقم الفاتورة: {user.get('user_id')})\n"
            response += f"     - أي صراف آلي (اختر: خدمات حكومية)\n\n"

        elif req == "unpaid_violations":
            violations = user.get("violations", {})
            amount = violations.get("total_amount", 0)
            count = violations.get("count", 0)

            # Generate payment link
            payment_link = generate_payment_link(user_key, amount, "violations")

            response += f"{i}. ⚠️ المخالفات المرورية (يُنصح بالسداد):\n"
            response += f"   • عدد المخالفات: {count} مخالفة\n"
            response += f"   • المبلغ الإجمالي: {amount} ریال\n"
            response += f"   • يُنصح بسداد المخالفات قبل التجديد\n\n"
            response += f"   🔗 اضغط هنا للسداد الفوري:\n"
            response += f"   {payment_link}\n\n"
            response += f"   أو يمكنك السداد عبر: أبشر، سداد، أو الصراف الآلي\n\n"

        elif req == "identity_not_near_expiry":
            identity = user.get("national_id") or user.get("iqama")
            if identity:
                expiry_date = identity.get("expiry_date")
                expiry_obj = datetime.strptime(expiry_date, "%Y-%m-%d")
                days_left = (expiry_obj - datetime.now()).days

                response += f"{i}. 📅 موعد التجديد:\n"
                response += f"   • تاريخ انتهاء هويتك/إقامتك: {expiry_date}\n"
                response += f"   • متبقي: {days_left} يوم\n"
                response += f"   • التجديد متاح قبل 60 يوم من الانتهاء\n"
                response += f"   • يمكنك العودة لاحقاً عندما يقترب الموعد\n\n"

        elif req == "photo_update_needed":
            response += f"{i}. 📸 الصورة الشخصية:\n"
            response += f"   • يلزم تحديث الصورة الشخصية\n\n"
            response += f"   📤 يمكنك رفع الصورة مباشرة عبر API:\n"
            response += f"   POST /api/upload-photo\n\n"
            response += f"   ⚠️ متطلبات الصورة:\n"
            response += f"     - صيغة: JPG أو PNG\n"
            response += f"     - الحد الأقصى للحجم: 5 ميجابايت\n"
            response += f"     - صورة حديثة وواضحة مع خلفية بيضاء\n"
            response += f"     - بدون نظارات أو غطاء رأس (إلا للنساء)\n\n"

        elif req == "not_citizen":
            response += f"{i}. 🇸🇦 الجنسية:\n"
            response += f"   • خدمة تجديد جواز السفر متاحة للمواطنين السعوديين فقط\n"
            response += f"   • للمقيمين: يمكن تجديد وثيقة السفر عبر الجوازات\n"
            response += f"   • للاستفسار: اتصل على 920000920\n\n"

    response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    response += "📌 بعد إتمام هذه المتطلبات، يمكنك طلب التجديد مرة أخرى.\n"
    response += "\n💬 هل تحتاج مساعدة إضافية في أي من هذه المتطلبات؟"

    # Clear the pending action after handling
    STATE["pending_action"] = None
    STATE["pending_intent"] = None

    return response


@app.post("/api/command")
def process_text(cmd: TextCommand):
    text = normalize(cmd.text)
    cur = STATE["current_user_key"]

    # 1) Check if user is responding to a pending action
    if STATE["pending_action"]:
        # Use LLM to detect if this is a confirmation
        context = "تم سؤال المستخدم: هل تريد المساعدة في إكمال المتطلبات الناقصة؟"
        is_confirming = detect_user_confirmation(cmd.text, context)

        if is_confirming:
            # User confirmed they want help with missing requirements
            visual = handle_pending_action_help(cur, STATE["pending_intent"], STATE["pending_action"])
            STATE["last_visual"] = visual

            return {
                "intent": "help_with_requirements",
                "text": cmd.text,
                "current_user": USERS[cur],
                "visual": visual,
                "action_steps": "",
                "recent_requests": STATE["recent_requests"]
            }
        else:
            # User didn't confirm, clear pending action and treat as new request
            STATE["pending_action"] = None
            STATE["pending_intent"] = None

    # 2) Detect intent
    intent = detect_intent(text)

    # 3) For info and unknown intents, use conversational AI with context
    if intent in ["info", "unknown"]:
        visual = generate_conversational_response(cmd.text, cur)
    else:
        # 4) Execute specific service logic (chatbot handles it automatically)
        visual = handle_intent(cur, intent)

    STATE["last_visual"] = visual

    return {
        "intent": intent,
        "text": cmd.text,
        "current_user": USERS[cur],
        "visual": visual,
        "action_steps": "",  # No manual steps - chatbot does it automatically
        "recent_requests": STATE["recent_requests"]
    }


class SwitchUserRequest(BaseModel):
    user_key: str

@app.post("/api/switch-user")
def switch_user(request: SwitchUserRequest):
    user_key = request.user_key

    # Validate user exists
    if user_key not in USERS:
        print(f"❌ User '{user_key}' not found in USERS. Available users: {list(USERS.keys())}")
        return {"error": f"User '{user_key}' not found", "current_user": None}

    print(f"✅ Switching to user: {user_key}")
    STATE["current_user_key"] = user_key
    # Clear conversation history and pending actions when switching users
    STATE["conversation_history"] = []
    STATE["pending_action"] = None
    STATE["pending_intent"] = None

    return {"current_user": USERS[user_key]}


def generate_welcome_notification(user_key: str) -> str:
    """
    Generate personalized welcome notification using GPT based on user info.
    Highlights important alerts (expiring documents, violations, etc.)
    """
    # Validate user exists
    if user_key not in USERS:
        return f"مرحباً! المستخدم غير موجود."

    user = USERS[user_key]

    # Get identity info with proper error handling
    identity = user.get("national_id") or user.get("iqama")

    if identity:
        identity_status = identity.get("status", "غير متوفر")
        identity_expiry = identity.get("expiry_date", "غير متوفر")
    else:
        identity_status = "غير متوفر"
        identity_expiry = "غير متوفر"

    # Calculate days until expiry if available
    days_until_expiry = "غير معروف"
    if identity and identity.get("expiry_date"):
        try:
            expiry_date = datetime.strptime(identity["expiry_date"], "%Y-%m-%d")
            today = datetime.now()
            days_until_expiry = (expiry_date - today).days
        except Exception as e:
            print(f"⚠️ Error calculating expiry days for {user_key}: {e}")
            days_until_expiry = "غير معروف"

    # Get license info with default empty dict
    license_info = user.get("driver_license", {})
    license_status = license_info.get("status", "غير متوفر")

    # Get violations info with defaults
    violations = user.get("violations", {})
    violations_count = violations.get("count", 0) if violations else 0
    violations_amount = violations.get("total_amount", 0) if violations else 0
    service_block = violations.get("service_block", False) if violations else False

    prompt = f"""
أنت مساعد ذكي لمنصة أبشر. قم بإنشاء رسالة ترحيب شخصية للمستخدم التالي:

معلومات المستخدم:
- الاسم: {user['name']}
- النوع: {user['user_type']}
- حالة الهوية/الإقامة: {identity_status}
- تاريخ انتهاء الهوية/الإقامة: {identity_expiry} (متبقي {days_until_expiry} يوم)
- حالة رخصة القيادة: {license_status}
- المخالفات المرورية: {violations_count} مخالفة بقيمة {violations_amount} ريال{"- تمنع الخدمات!" if service_block else ""}

تعليمات:
1. ابدأ بترحيب شخصي باسم المستخدم
2. إذا كان هناك مشاكل عاجلة (هوية تنتهي قريباً، مخالفات، رخصة منتهية)، نبّه عليها بوضوح
3. اذكر الإيجابيات إن وجدت (كل شيء صالح، لا مخالفات)
4. كن موجزاً ومباشراً (2-4 جمل فقط)
5. استخدم أيقونات مناسبة (✅ ⚠️ ❌ 📅 🚗)

مثال للتنبيهات:
- إذا كانت الهوية expired: تنبيه عاجل
- إذا كانت الهوية near_expiry: تنبيه بالتجديد قريباً
- إذا كانت المخالفات تمنع الخدمات: تنبيه عاجل بضرورة السداد
- إذا كانت المخالفات موجودة لكن لا تمنع: اذكرها كتذكير

اكتب الرسالة فقط بدون أي شرح إضافي:
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Welcome notification generation failed: {e}")
        return f"مرحباً {user['name']}! 👋"


@app.get("/api/notification/{user_key}")
def get_user_notification(user_key: str):
    """Get personalized notification for a user"""
    if user_key not in USERS:
        return {"error": "User not found"}

    notification = generate_welcome_notification(user_key)
    return {
        "user_key": user_key,
        "notification": notification,
        "user": USERS[user_key]
    }


@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    """
    Upload user photo for identity document renewal.
    In production, this would save to cloud storage and update user record.
    """
    try:
        # Read file
        contents = await file.read()

        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if file.content_type not in allowed_types:
            return {
                "success": False,
                "error": "نوع الملف غير مدعوم. يرجى رفع صورة بصيغة JPG أو PNG فقط."
            }

        # Validate file size (max 5MB)
        max_size = 5 * 1024 * 1024  # 5MB
        if len(contents) > max_size:
            return {
                "success": False,
                "error": "حجم الصورة كبير جداً. الحد الأقصى 5 ميجابايت."
            }

        # In production: Upload to cloud storage (S3, Azure Blob, etc.)
        # For now, we'll just simulate success
        user_key = STATE["current_user_key"]

        # Mock: Update user's photo status
        if user_key in USERS:
            identity = USERS[user_key].get("national_id") or USERS[user_key].get("iqama")
            if identity and identity.get("needs_photo_update"):
                identity["needs_photo_update"] = False
                identity["photo_uploaded"] = True
                identity["photo_upload_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "message": "تم رفع الصورة بنجاح! ✅",
            "file_name": file.filename,
            "file_size": len(contents),
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"❌ Photo upload error: {e}")
        return {
            "success": False,
            "error": f"حدث خطأ أثناء رفع الصورة: {str(e)}"
        }


@app.get("/payment")
async def payment_page(service: str = "violations", user_id: str = "0000", amount: float = 0):
    """
    Mock payment page that simulates Sadad payment gateway.
    When user clicks pay, it processes the payment and redirects back.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>سداد - بوابة الدفع الإلكتروني</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .payment-card {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 500px;
                width: 100%;
                padding: 40px;
                text-align: center;
            }}
            .logo {{
                font-size: 48px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #2d3748;
                font-size: 28px;
                margin-bottom: 10px;
            }}
            .subtitle {{
                color: #718096;
                font-size: 14px;
                margin-bottom: 30px;
            }}
            .payment-details {{
                background: #f7fafc;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 30px;
                text-align: right;
            }}
            .detail-row {{
                display: flex;
                justify-content: space-between;
                padding: 12px 0;
                border-bottom: 1px solid #e2e8f0;
            }}
            .detail-row:last-child {{
                border-bottom: none;
                padding-top: 20px;
                margin-top: 10px;
                border-top: 2px solid #667eea;
            }}
            .detail-label {{
                color: #718096;
                font-size: 14px;
            }}
            .detail-value {{
                color: #2d3748;
                font-weight: 600;
                font-size: 16px;
            }}
            .total {{
                font-size: 24px;
                color: #667eea;
            }}
            .pay-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 16px 48px;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }}
            .pay-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 12px 28px rgba(102, 126, 234, 0.5);
            }}
            .pay-btn:active {{
                transform: translateY(0);
            }}
            .security-note {{
                margin-top: 24px;
                color: #718096;
                font-size: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }}
            .processing {{
                display: none;
                margin-top: 20px;
            }}
            .processing.active {{
                display: block;
            }}
            .spinner {{
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .success {{
                display: none;
            }}
            .success.active {{
                display: block;
            }}
            .success-icon {{
                font-size: 64px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="payment-card">
            <div id="payment-form">
                <div class="logo">💳</div>
                <h1>بوابة سداد للدفع الإلكتروني</h1>
                <p class="subtitle">نظام الدفع الإلكتروني الموحد</p>

                <div class="payment-details">
                    <div class="detail-row">
                        <span class="detail-label">نوع الخدمة:</span>
                        <span class="detail-value">{'مخالفات مرورية' if service == 'violations' else 'خدمة حكومية'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">رقم المستخدم:</span>
                        <span class="detail-value">{user_id}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">رقم الفاتورة:</span>
                        <span class="detail-value">INV-{user_id}-{int(datetime.now().timestamp())}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">المبلغ المطلوب:</span>
                        <span class="detail-value total">{amount:,.2f} ریال</span>
                    </div>
                </div>

                <button class="pay-btn" onclick="processPayment()">
                    ✓ تأكيد الدفع
                </button>

                <div class="security-note">
                    🔒 عملية دفع آمنة ومشفرة
                </div>
            </div>

            <div class="processing" id="processing">
                <div class="spinner"></div>
                <p style="color: #667eea; font-weight: 600;">جارٍ معالجة الدفع...</p>
            </div>

            <div class="success" id="success">
                <div class="success-icon">✅</div>
                <h2 style="color: #48bb78; margin-bottom: 16px;">تم الدفع بنجاح!</h2>
                <p style="color: #718096; margin-bottom: 24px;">
                    تم سداد مبلغ <strong>{amount:,.2f} ریال</strong> بنجاح
                </p>
                <div style="background: #f0fff4; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                    <p style="color: #2f855a; font-size: 14px;">
                        رقم العملية: TXN-{int(datetime.now().timestamp())}
                    </p>
                </div>
                <p style="color: #718096; font-size: 14px;">
                    سيتم تحديث حالتك تلقائياً في نظام أبشر
                </p>
            </div>
        </div>

        <script>
            async function processPayment() {{
                // Hide form, show processing
                document.getElementById('payment-form').style.display = 'none';
                document.getElementById('processing').classList.add('active');

                // Simulate payment processing (2 seconds)
                await new Promise(resolve => setTimeout(resolve, 2000));

                // Call backend to update user status
                try {{
                    const response = await fetch('/api/process-payment', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            user_id: '{user_id}',
                            amount: {amount},
                            service: '{service}'
                        }})
                    }});

                    const data = await response.json();
                    console.log('Payment processed:', data);
                }} catch (error) {{
                    console.error('Payment API error:', error);
                }}

                // Show success
                document.getElementById('processing').classList.remove('active');
                document.getElementById('success').classList.add('active');

                // Redirect back after 3 seconds
                setTimeout(() => {{
                    window.close(); // Try to close the tab
                    // If can't close, show message
                    alert('تم الدفع بنجاح! يمكنك إغلاق هذه النافذة والعودة لأبشر');
                }}, 3000);
            }}
        </script>
    </body>
    </html>
    """

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


class PaymentRequest(BaseModel):
    user_id: str
    amount: float
    service: str


@app.post("/api/process-payment")
async def process_payment(payment: PaymentRequest):
    """
    Process mock payment and update user violations status.
    """
    try:
        # Find user by user_id
        user_key = None
        for key, user in USERS.items():
            if user.get("user_id") == payment.user_id:
                user_key = key
                break

        if not user_key:
            return {"success": False, "error": "User not found"}

        user = USERS[user_key]

        # Clear violations
        if payment.service == "violations":
            violations = user.get("violations", {})
            violations["count"] = 0
            violations["total_amount"] = 0
            violations["service_block"] = False
            violations["last_payment_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            violations["last_payment_amount"] = payment.amount

        return {
            "success": True,
            "message": "تم سداد المخالفات بنجاح",
            "transaction_id": f"TXN-{int(datetime.now().timestamp())}",
            "user_id": payment.user_id,
            "amount": payment.amount
        }

    except Exception as e:
        print(f"❌ Payment processing error: {e}")
        return {"success": False, "error": str(e)}


import base64


# ============================================
# SERVICE REQUIREMENTS VALIDATION
# ============================================

def parse_services_requirements():
    """
    Parse services.txt to extract requirements for each service.
    Returns dict mapping service names to their requirements list.
    """
    services_file = "services.txt"

    if not os.path.exists(services_file):
        print(f"⚠️ Warning: {services_file} not found")
        return {}

    try:
        with open(services_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the structure: each service starts with ١), ٢), etc.
        services_data = {}

        # National ID / Iqama
        services_data["id_renewal"] = [
            "صلاحية الهوية أو الإقامة (هل انتهت أو قربت تنتهي)",
            "صورة شخصية حديثة (للتجديد)",
            "سداد أي رسوم متأخرة",
            "إذا مقيم: التأكد من سريان التأمين الصحي",
            "ما عليه إيقاف خدمات"
        ]

        # Driver License
        services_data["driver_license_renewal"] = [
            "صلاحية الرخصة (منتهية أو قرب الانتهاء)",
            "سداد المخالفات المرورية (إذا موجودة)",
            "فحص طبي / نظر (حسب نوع الرخصة أو العمر)",
            "تأمين المركبة ساري (إذا مرتبطة بالمركبة)"
        ]

        # Passport
        services_data["passport_renewal"] = [
            "صلاحية الجواز (قبل انتهاءه بـ 6 أشهر عادة)",
            "دفع رسوم التجديد",
            "لا يوجد بلاغ فقدان",
            "للمقيمين: الإقامة سارية"
        ]

        return services_data

    except Exception as e:
        print(f"❌ Error parsing services.txt: {e}")
        return {}


# Load service requirements at startup
SERVICE_REQUIREMENTS = parse_services_requirements()


def validate_service_requirements(user_key: str, intent: str) -> dict:
    """
    Validate if user meets all requirements for a service.

    Returns:
        {
            "valid": bool,
            "missing_requirements": list of str (technical list),
            "missing_fields": list of str (fields to ask user for),
            "message": str (conversational Arabic message asking for missing info)
        }
    """
    user = USERS[user_key]
    missing_requirements = []
    missing_fields = []

    # ID/Iqama Renewal
    if intent == "id_renewal":
        # Get identity document (national_id for citizens, iqama for residents)
        identity = user.get("national_id") or user.get("iqama")

        if identity:
            status = identity.get("status")
            expiry_date_str = identity.get("expiry_date")

            # Check if document needs renewal
            if status == "valid":
                # Calculate days until expiry
                expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
                days_left = (expiry_date - datetime.now()).days

                if days_left > 60:
                    missing_requirements.append("identity_not_near_expiry")
                    missing_fields.append(f"الهوية/الإقامة لم تقرب من الانتهاء (متبقي {days_left} يوم)")

            # Check for photo update requirement
            if identity.get("needs_photo_update"):
                missing_requirements.append("photo_update_needed")
                missing_fields.append("صورة شخصية حديثة")

        # Check for violations that block service
        violations = user.get("violations", {})
        if violations.get("service_block"):
            missing_requirements.append("service_blocking_violations")
            missing_fields.append(f"مخالفات مرورية تمنع الخدمة بقيمة {violations.get('total_amount')} ريال")
        elif violations.get("total_amount", 0) > 0:
            missing_requirements.append("unpaid_violations")
            missing_fields.append(f"مخالفات مرورية بقيمة {violations.get('total_amount')} ريال (يُنصح بسدادها)")

    # Driver License Renewal
    elif intent == "driver_license_renewal":
        # Check violations only (removed medical check - not fully automatable)
        violations = user.get("violations", {})
        if violations.get("service_block"):
            missing_requirements.append("service_blocking_violations")
            missing_fields.append(f"مخالفات مرورية تمنع الخدمة بقيمة {violations.get('total_amount')} ريال")
        elif violations.get("total_amount", 0) > 0:
            missing_requirements.append("unpaid_violations")
            missing_fields.append(f"مخالفات مرورية بقيمة {violations.get('total_amount')} ريال")

    # Passport Renewal
    elif intent == "passport_renewal":
        # Check if citizen
        if user["user_type"] != "المواطن":
            missing_requirements.append("not_citizen")
            missing_fields.append("الجنسية السعودية (الخدمة للمواطنين فقط)")

        # Check for violations
        violations = user.get("violations", {})
        if violations.get("total_amount", 0) > 0:
            missing_requirements.append("unpaid_violations")
            missing_fields.append(f"مخالفات مرورية بقيمة {violations.get('total_amount')} ريال (يُنصح بسدادها)")

    # Build conversational response
    if missing_requirements:
        # Create a conversational message asking for missing info
        if len(missing_fields) == 1:
            message = f"لتجديد {'الهوية' if intent == 'id_renewal' else 'رخصة القيادة' if intent == 'driver_license_renewal' else 'جواز السفر'}، يلزم:\n\n• {missing_fields[0]}\n\nهل تريد المساعدة في إكمال هذا المتطلب؟"
        else:
            fields_text = "\n".join([f"• {field}" for field in missing_fields])
            message = f"لتجديد {'الهوية' if intent == 'id_renewal' else 'رخصة القيادة' if intent == 'driver_license_renewal' else 'جواز السفر'}، يلزم:\n\n{fields_text}\n\nهل تريد المساعدة في إكمال هذه المتطلبات؟"

        # Store pending action in STATE for follow-up
        STATE["pending_intent"] = intent
        STATE["pending_action"] = {
            "missing_requirements": missing_requirements,
            "missing_fields": missing_fields
        }

        return {
            "valid": False,
            "missing_requirements": missing_requirements,
            "missing_fields": missing_fields,
            "message": message
        }

    return {
        "valid": True,
        "missing_requirements": [],
        "missing_fields": [],
        "message": "✅ جميع المتطلبات متوفرة. سأقوم بتجهيز الطلب الآن..."
    }


def get_service_requirements_info(intent: str) -> str:
    """
    Get human-readable requirements info for a service.
    Used for answering 'info' questions about service requirements.
    """
    if intent not in SERVICE_REQUIREMENTS:
        return ""

    requirements = SERVICE_REQUIREMENTS[intent]

    service_names = {
        "id_renewal": "تجديد الهوية/الإقامة",
        "driver_license_renewal": "تجديد رخصة القيادة",
        "passport_renewal": "تجديد جواز السفر"
    }

    service_name = service_names.get(intent, intent)

    requirements_text = "\n".join([f"  • {req}" for req in requirements])

    return f"""📋 متطلبات {service_name}:

{requirements_text}

ℹ️ تأكد من توفر جميع المتطلبات قبل تقديم الطلب."""


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
        cur = STATE["current_user_key"]

        # 1) Check if user is responding to a pending action
        if STATE["pending_action"]:
            # Use LLM to detect if this is a confirmation
            context = "تم سؤال المستخدم: هل تريد المساعدة في إكمال المتطلبات الناقصة؟"
            is_confirming = detect_user_confirmation(text, context)

            if is_confirming:
                # User confirmed they want help with missing requirements
                visual = handle_pending_action_help(cur, STATE["pending_intent"], STATE["pending_action"])
                intent = "help_with_requirements"
            else:
                # User didn't confirm, clear pending action and treat as new request
                STATE["pending_action"] = None
                STATE["pending_intent"] = None
                intent = detect_intent(text)

                # For info and unknown intents, use conversational AI with context
                if intent in ["info", "unknown"]:
                    visual = generate_conversational_response(text, cur)
                else:
                    # Execute specific service logic
                    visual = handle_intent(cur, intent)
        else:
            # No pending action, process normally
            intent = detect_intent(text)

            # For info and unknown intents, use conversational AI with context
            if intent in ["info", "unknown"]:
                visual = generate_conversational_response(text, cur)
            else:
                # Execute specific service logic (chatbot handles it automatically)
                visual = handle_intent(cur, intent)

        # Use the visual message directly (no manual steps)
        final_text = visual

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
            "action_steps": "",  # No manual steps - chatbot does it automatically
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
