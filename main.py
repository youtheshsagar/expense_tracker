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

# Load .env next to this file (works even if uvicorn's cwd is not the project folder)
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

app = FastAPI()

# 📊 Google Sheets / Drive (use current scopes; legacy oauth2client "feeds" scope often causes 403)
_GS_SCOPES = (
  "https://www.googleapis.com/auth/spreadsheets",
  "https://www.googleapis.com/auth/drive",
)

_groq_client = None
_sheet = None


def _service_account_email():
  with open(os.path.join(_ROOT, "creds.json"), encoding="utf-8") as f:
    return json.load(f)["client_email"]


@app.on_event("startup")
def _log_sheets_sharing_hint():
  try:
    email = _service_account_email()
    print(
      f"Google Sheets: share your spreadsheet with this service account (Editor): {email}"
    )
  except OSError:
    pass


def get_groq():
  """Groq API (api.groq.com). OpenRouter is blocked on many JLL networks (Netskope)."""
  global _groq_client
  if _groq_client is None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
      raise RuntimeError("GROQ_API_KEY is not set (check .env)")
    _groq_client = Groq(api_key=key)
  return _groq_client


def get_sheet():
  global _sheet
  if _sheet is None:
    creds_path = os.path.join(_ROOT, "creds.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=_GS_SCOPES)
    gs_client = gspread.authorize(creds)
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if sheet_id:
      sh = gs_client.open_by_key(sheet_id)
    else:
      name = os.getenv("GOOGLE_SHEET_NAME", "Construction Expenses").strip()
      sh = gs_client.open(name)
    _sheet = sh.sheet1
    # Ensure first column header is "Date" (matches row 1 in your sheet)
    try:
      _sheet.update_acell("A1", "Date")
    except Exception:
      pass
  return _sheet


def extract_expense_simple(text):
  """
  Offline parsing when Groq/OpenRouter are blocked (JLL Netskope GenAI policy).
  Handles amounts like 15k, 15000, ₹500, rs 2000.
  """
  if not text or not str(text).strip():
    return {"category": "ignore", "amount": 0, "notes": ""}

  raw = str(text).strip()
  low = raw.lower()

  amount = 0.0
  m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", low)
  if m:
    amount = float(m.group(1)) * 1000
  else:
    m = re.search(r"(?:₹|rs\.?|inr)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", low)
    if m:
      amount = float(m.group(1).replace(",", ""))
    else:
      m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{4,})\b", raw.replace(",", ""))
      if m:
        amount = float(m.group(1).replace(",", ""))
      else:
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", low)
        if m:
          amount = float(m.group(1))

  category = "general"
  if any(w in low for w in ("labour", "labor", "mason", "worker", "wage")):
    category = "labour"
  elif any(w in low for w in ("cement", "sand", "aggregate", "brick", "steel", "material")):
    category = "materials"
  elif any(w in low for w in ("transport", "diesel", "fuel", "petrol", "truck")):
    category = "transport"
  elif any(w in low for w in ("electric", "wiring", "plumb", "pipe")):
    category = "services"

  if amount <= 0:
    return {"category": "ignore", "amount": 0, "notes": raw}

  amt = int(round(amount))
  return {"category": category, "amount": amt, "notes": raw}


# 🧠 LLM extraction (falls back to regex if corporate proxy blocks api.groq.com)
def extract_expense(text):

  if os.getenv("EXTRACTION_MODE", "").strip().lower() in ("simple", "offline", "regex"):
    return extract_expense_simple(text)

  try:
    client = get_groq()

    completion = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[
        {
          "role": "system",
          "content": "You are a strict JSON generator. Return ONLY valid JSON."
        },
        {
          "role": "user",
          "content": f"""
Extract expense details from:

{text}

Return ONLY JSON:
{{
 "category": "",
 "amount": number,
 "notes": ""
}}

If not an expense, return:
{{
 "category": "ignore",
 "amount": 0,
 "notes": ""
}}
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

    return extract_expense_simple(text)

  except Exception as e:
    print("Groq unavailable (blocked or error), using regex fallback:", type(e).__name__)
    return extract_expense_simple(text)


# 📩 Twilio webhook (Twilio uses POST; GET avoids 405 if someone opens the URL in a browser)
@app.get("/webhook")
async def webhook_probe():
  return Response(
    content="OK — configure this URL in Twilio as POST webhook for incoming messages.",
    media_type="text/plain",
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
      content="""
      <Response>
        <Message>Ignored ❌</Message>
      </Response>
      """,
      media_type="application/xml"
    )

  # 📅 When the message was recorded (DD/MM/YYYY HH:MM:SS)
  recorded_at = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

  # 💾 Save to Google Sheets
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
    try:
      sa = _service_account_email()
    except OSError:
      sa = "(see creds.json client_email)"
    print("Google Sheets access failed:", type(e).__name__, getattr(e, "response", e))
    print("Fix: Google Sheets → Share → add", sa, "as Editor (403 means not shared with this account).")
    return Response(
      content=f"""
      <Response>
        <Message>Cannot access Google Sheet (403). Share it with {sa} as Editor, then try again.</Message>
      </Response>
      """,
      media_type="application/xml",
    )

  print("✅ Saved:", parsed)

  # 📤 Reply to WhatsApp
  return Response(
    content=f"""
    <Response>
      <Message>Saved ✅ ₹{parsed["amount"]}</Message>
    </Response>
    """,
    media_type="application/xml"
  )