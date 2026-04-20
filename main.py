from fastapi import FastAPI, Request
from fastapi.responses import Response
from groq import Groq
import json
import re
import datetime
import os
import psycopg2
import requests
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import pytesseract
from requests.auth import HTTPBasicAuth

# -------------------------------
# ENV
# -------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

app = FastAPI()

_groq_client = None


# -------------------------------
# Groq Client
# -------------------------------
def get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing")
        _groq_client = Groq(api_key=key)
    return _groq_client


# -------------------------------
# Download Twilio image (AUTH REQUIRED)
# -------------------------------
def download_image(url):
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")

    r = requests.get(
        url,
        auth=HTTPBasicAuth(sid, token),
        headers={"User-Agent": "Mozilla/5.0"}
    )

    print("IMAGE STATUS:", r.status_code)

    if r.status_code != 200:
        raise Exception(f"Image download failed: {r.status_code}")

    return r.content


# -------------------------------
# OCR extraction
# -------------------------------
def extract_text_ocr(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return text


# -------------------------------
# Base64 converter
# -------------------------------
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


# -------------------------------
# TEXT fallback parser
# -------------------------------
def extract_expense_simple(text):
    if not text:
        return {"category": "ignore", "amount": 0, "notes": ""}

    raw = str(text).strip().lower()
    amount = 0

    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", raw)
    if m:
        amount = float(m.group(1)) * 1000
    else:
        m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*)", raw)
        if m:
            amount = float(m.group(1).replace(",", ""))

    category = "general"
    if "cement" in raw or "sand" in raw:
        category = "materials"
    elif "labour" in raw:
        category = "labour"
    elif "diesel" in raw or "fuel" in raw:
        category = "transport"

    if amount <= 0:
        return {"category": "ignore", "amount": 0, "notes": raw}

    return {
        "category": category,
        "amount": int(amount),
        "notes": raw
    }


# -------------------------------
# HYBRID OCR + VISION ENGINE
# -------------------------------
def hybrid_extract(image_url):
    try:
        client = get_groq()

        # 1. Download image
        img_bytes = download_image(image_url)

        # 2. OCR TEXT
        ocr_text = extract_text_ocr(img_bytes)
        print("OCR TEXT:", ocr_text)

        # 3. Vision AI
        img_b64 = image_to_base64(img_bytes)

        res = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
You are an expense extraction system.

Use BOTH OCR text and image.

OCR TEXT:
{ocr_text}

TASK:
- Fix OCR mistakes
- Extract expense
- Identify category

Return ONLY JSON:
{{
  "category": "materials|labour|transport|general",
  "amount": number,
  "notes": "short summary"
}}

If no expense:
{{"category":"ignore","amount":0,"notes":"none"}}
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0
        )

        raw = res.choices[0].message.content
        print("VISION OUTPUT:", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        print("HYBRID ERROR:", e)

    return {"category": "ignore", "amount": 0, "notes": "failed"}


# -------------------------------
# DB SAVE
# -------------------------------
def save_to_db(row):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO expenses 
        (recorded_at, phone, category, amount, notes, raw_message)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, row)

    conn.commit()
    cur.close()
    conn.close()


# -------------------------------
# WEBHOOK
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.form()

    message = data.get("Body")
    sender = data.get("From")
    num_media = int(data.get("NumMedia", 0))

    print("Incoming:", message, "Media:", num_media)

    # IMAGE FLOW
    if num_media > 0:
        media_url = data.get("MediaUrl0")
        print("Media URL:", media_url)

        parsed = hybrid_extract(media_url)

    # TEXT FLOW
    else:
        parsed = extract_expense_simple(message)

    # IGNORE
    if parsed["category"] == "ignore":
        return Response(
            "<Response><Message>Could not read receipt ❌</Message></Response>",
            media_type="application/xml"
        )

    # SAVE
    try:
        save_to_db([
            datetime.datetime.now(),
            sender,
            parsed["category"],
            parsed["amount"],
            parsed["notes"],
            message if message else "image"
        ])
    except Exception as e:
        print("DB ERROR:", e)
        return Response(
            "<Response><Message>DB error ❌</Message></Response>",
            media_type="application/xml"
        )

    return Response(
        f"<Response><Message>Saved ₹{parsed['amount']} ✅</Message></Response>",
        media_type="application/xml"
    )


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def root():
    return {"status": "running"}