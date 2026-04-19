from fastapi import FastAPI, Request
from fastapi.responses import Response
from groq import Groq
import json
import re
import datetime
import os
from dotenv import load_dotenv
import gspread
from gspread.exceptions import APIError, SpreadsheetNotFound
from google.oauth2.service_account import Credentials

# Load .env (local only)
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

app = FastAPI()

# Google Sheets scopes
_GS_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)

_groq_client = None
_sheet = None


# -------------------------------
# 🔐 Google Credentials from ENV
# -------------------------------
def get_google_credentials():
    creds_json = os.getenv("GOOGLE_CREDS_JSON")

    if not creds_json:
        raise RuntimeError("GOOGLE_CREDS_JSON is not set")

    try:
        creds_dict = json.loads(creds_json)
    except Exception as e:
        raise RuntimeError(f"Invalid GOOGLE_CREDS_JSON: {e}")

    return Credentials.from_service_account_info(
        creds_dict, scopes=_GS_SCOPES
    )


def get_sheet():
    global _sheet

    if _sheet is None:
        creds = get_google_credentials()
        gs_client = gspread.authorize(creds)

        sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()

        if not sheet_id:
            raise RuntimeError("GOOGLE_SHEET_ID is not set")

        sh = gs_client.open_by_key(sheet_id)
        _sheet = sh.sheet1

        try:
            _sheet.update_acell("A1", "Date")
        except Exception:
            pass

    return _sheet


# -------------------------------
# 🧠 Groq Client
# -------------------------------
def get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = Groq(api_key=key)
    return _groq_client


# -------------------------------
# 🧾 Fallback regex parser
# -------------------------------
def extract_expense_simple(text):
    if not text or not str(text).strip():
        return {"category": "ignore", "amount": 0, "notes": ""}

    raw = str(text).strip()
    low = raw.lower()
    amount = 0.0

    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", low)
    if m:
        amount = float(m.group(1)) * 1000
    else:
        m = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)", low)
        if m:
            amount = float(m.group(1).replace(",", ""))
        else:
            m = re.search(r"\b(\d{4,})\b", low.replace(",", ""))
            if m:
                amount = float(m.group(1))
            else:
                m = re.search(r"\b(\d+(?:\.\d+)?)\b", low)
                if m:
                    amount = float(m.group(1))

    category = "general"
    if any(w in low for w in ("labour", "labor", "mason", "worker")):
        category = "labour"
    elif any(w in low for w in ("cement", "sand", "brick", "steel")):
        category = "materials"
    elif any(w in low for w in ("diesel", "fuel", "truck")):
        category = "transport"

    if amount <= 0:
        return {"category": "ignore", "amount": 0, "notes": raw}

    return {"category": category, "amount": int(round(amount)), "notes": raw}


# -------------------------------
# 🧠 LLM + fallback
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Extract expense from:
{text}

Return:
{{ "category": "", "amount": number, "notes": "" }}

If not expense:
{{ "category": "ignore", "amount": 0, "notes": "" }}
"""
                }
            ],
            temperature=0,
            max_tokens=100
        )

        raw = completion.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception:
        pass

    return extract_expense_simple(text)


# -------------------------------
# 🌐 Webhook endpoints
# -------------------------------
@app.get("/webhook")
async def webhook_probe():
    return Response(
        content="OK — configure this URL in Twilio",
        media_type="text/plain"
    )


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.form()
    message = data.get("Body")
    sender = data.get("From")

    print("📩 Incoming:", message)

    parsed = extract_expense(message)

    if parsed["category"] == "ignore":
        return Response(
            content="<Response><Message>Ignored ❌</Message></Response>",
            media_type="application/xml"
        )

    recorded_at = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    try:
        get_sheet().append_row([
            recorded_at,
            sender,
            parsed["category"],
            parsed["amount"],
            parsed["notes"],
            message
        ])
    except (SpreadsheetNotFound, APIError, PermissionError) as e:
        print("Google Sheets error:", e)
        return Response(
            content="<Response><Message>Sheet access error ❌</Message></Response>",
            media_type="application/xml"
        )

    print("✅ Saved:", parsed)

    return Response(
        content=f"<Response><Message>Saved ✅ ₹{parsed['amount']}</Message></Response>",
        media_type="application/xml"
    )
