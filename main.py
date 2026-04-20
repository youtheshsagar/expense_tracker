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
load_dotenv()

app = FastAPI()

_groq_client = None


# -------------------------------
# GROQ CLIENT
# -------------------------------
def get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# -------------------------------
# TWILIO IMAGE DOWNLOAD (FIXED)
# -------------------------------
def download_image(url):
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")

    r = requests.get(
        url,
        auth=HTTPBasicAuth(sid, token),
        headers={"User-Agent": "Mozilla/5.0"}
    )

    print("📸 IMAGE STATUS:", r.status_code)

    if r.status_code != 200:
        raise Exception(f"Image download failed: {r.status_code}")

    return r.content


# -------------------------------
# OCR
# -------------------------------
def extract_text_ocr(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        print("🧾 OCR TEXT:\n", text)
        return text.strip()
    except Exception as e:
        print("OCR ERROR:", e)
        return ""


# -------------------------------
# BASE64
# -------------------------------
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode()


# -------------------------------
# TEXT FALLBACK
# -------------------------------
def extract_expense_simple(text):
    if not text:
        return {"category": "ignore", "amount": 0, "notes": ""}

    raw = text.lower()
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
    elif "diesel" in raw:
        category = "transport"

    if amount <= 0:
        return {"category": "ignore", "amount": 0, "notes": raw}

    return {
        "category": category,
        "amount": int(amount),
        "notes": raw
    }


# -------------------------------
# HYBRID OCR + VISION (MAIN ENGINE)
# -------------------------------
def hybrid_extract(image_url):
    try:
        client = get_groq()

        img_bytes = download_image(image_url)

        # OCR
        ocr_text = extract_text_ocr(img_bytes)

        # Vision
        img_b64 = image_to_base64(img_bytes)

        res = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
You are a receipt extractor.

OCR TEXT:
{ocr_text}

RULES:
- Extract expense even if partially visible
- Fix OCR mistakes
- Always return valid JSON

FORMAT:
{{
  "category": "materials|labour|transport|general",
  "amount": number,
  "notes": "short summary"
}}

If nothing found:
{{"category":"ignore","amount":0,"notes":"no data"}}
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
        print("🧠 VISION RAW:\n", raw)

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

    print("📩 Incoming:", message, "Media:", num_media)

    try:
        if num_media > 0:
            media_url = data.get("MediaUrl0")
            print("📸 MEDIA URL:", media_url)
            parsed = hybrid_extract(media_url)
        else:
            parsed = extract_expense_simple(message)

        print("✅ PARSED RESULT:", parsed)

        if parsed["category"] == "ignore":
            return Response(
                "<Response><Message>Could not read receipt ❌</Message></Response>",
                media_type="application/xml"
            )

        save_to_db([
            datetime.datetime.now(),
            sender,
            parsed["category"],
            parsed["amount"],
            parsed["notes"],
            message if message else "image"
        ])

        return Response(
            f"<Response><Message>Saved ₹{parsed['amount']} ✅</Message></Response>",
            media_type="application/xml"
        )

    except Exception as e:
        print("🔥 WEBHOOK ERROR:", e)
        return Response(
            "<Response><Message>Server error ❌</Message></Response>",
            media_type="application/xml"
        )


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def root():
    return {"status": "running"}