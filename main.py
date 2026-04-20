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
# Download image
# -------------------------------
def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    return response.content


# -------------------------------
# Convert image to base64
# -------------------------------
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


# -------------------------------
# Text fallback parser
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
        m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)", raw)
        if m:
            amount = float(m.group(1).replace(",", ""))

    category = "general"
    if any(x in raw for x in ["cement", "sand", "brick", "steel"]):
        category = "materials"
    elif any(x in raw for x in ["labour", "worker", "mason"]):
        category = "labour"
    elif any(x in raw for x in ["diesel", "fuel"]):
        category = "transport"

    if amount <= 0:
        return {"category": "ignore", "amount": 0, "notes": raw}

    return {
        "category": category,
        "amount": int(round(amount)),
        "notes": raw
    }


# -------------------------------
# LLM text extraction
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY JSON"},
                {"role": "user", "content": f"""
Extract expense:

{text}

Return:
{{"category":"", "amount":number, "notes":""}}
"""}
            ],
            temperature=0
        )

        raw = res.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:
        print("TEXT LLM ERROR:", e)

    return extract_expense_simple(text)


# -------------------------------
# IMAGE → EXPENSE (VISION MODEL)
# -------------------------------
def extract_from_image(image_url):
    try:
        client = get_groq()

        img_bytes = download_image(image_url)
        img_b64 = image_to_base64(img_bytes)

        res = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Extract expense from this receipt image.

Return ONLY JSON:
{
  "category": "",
  "amount": number,
  "notes": ""
}

If not an expense:
{
  "category": "ignore",
  "amount": 0,
  "notes": ""
}
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:
        print("IMAGE LLM ERROR:", e)

    return {"category": "ignore", "amount": 0, "notes": "image_failed"}


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
# HEALTH CHECK
# -------------------------------
@app.get("/webhook")
async def webhook_probe():
    return {"status": "ok"}


# -------------------------------
# MAIN WEBHOOK (TEXT + IMAGE)
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.form()

    message = data.get("Body")
    sender = data.get("From")
    num_media = int(data.get("NumMedia", 0))

    print("Incoming:", message, "Media:", num_media)

    # -----------------------
    # IMAGE FLOW
    # -----------------------
    if num_media > 0:
        media_url = data.get("MediaUrl0")
        print("Image URL:", media_url)

        parsed = extract_from_image(media_url)

    # -----------------------
    # TEXT FLOW
    # -----------------------
    else:
        parsed = extract_expense(message)

    # -----------------------
    # Ignore non-expense
    # -----------------------
    if parsed["category"] == "ignore":
        return Response(
            "<Response><Message>Ignored ❌</Message></Response>",
            media_type="application/xml"
        )

    # -----------------------
    # SAVE
    # -----------------------
    try:
        save_to_db([
            datetime.datetime.now(),
            sender,
            parsed["category"],
            parsed["amount"],
            parsed["notes"],
            message if message else "image_receipt"
        ])
    except Exception as e:
        print("DB ERROR:", e)
        return Response(
            "<Response><Message>DB Error ❌</Message></Response>",
            media_type="application/xml"
        )

    return Response(
        f"<Response><Message>Saved ✅ ₹{parsed['amount']}</Message></Response>",
        media_type="application/xml"
    )