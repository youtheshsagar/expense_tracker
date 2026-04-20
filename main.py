from fastapi import FastAPI, Request
from twilio.twiml.messaging_response import MessagingResponse
from supabase import create_client
import psycopg2
import requests
import json
import uuid
import io
import re
import os
from dotenv import load_dotenv

import pytesseract
from PIL import Image
import pdfplumber

from groq import Groq

load_dotenv()

app = FastAPI()

# =========================
# ENV
# =========================
SUPABASE_URL = "https://zasmsfjahuuwafjiajqp.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
BUCKET = "expense-files"

GROQ_KEY = os.getenv("GROQ_API_KEY")
DB_CONN_STRING = os.getenv("DATABASE_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = Groq(api_key=GROQ_KEY)

# =========================
# DB
# =========================
def get_conn():
    return psycopg2.connect(DB_CONN_STRING)

# =========================
# SAFE JSON PARSER
# =========================
def safe_json_parse(text):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# =========================
# ROBUST TWILIO DOWNLOAD
# =========================
def download_media(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*"
    }

    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=20)
        return r.content, r.headers.get("Content-Type", "")
    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        return None, None

# =========================
# FILE VALIDATION
# =========================
def is_valid_file(file_bytes):
    if not file_bytes:
        return False

    # HTML response detection (VERY COMMON Twilio issue)
    if file_bytes[:4] == b"<htm" or b"<html" in file_bytes[:100]:
        return False

    if len(file_bytes) < 200:
        return False

    return True

# =========================
# OCR + PDF FIXED
# =========================
def extract_text(file_bytes, content_type):
    try:
        if not is_valid_file(file_bytes):
            print("INVALID FILE DETECTED")
            return None

        # ======================
        # PDF HANDLING
        # ======================
        if "pdf" in content_type:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                    return text if text.strip() else None
            except Exception as e:
                print("PDF ERROR:", e)
                return None

        # ======================
        # IMAGE HANDLING
        # ======================
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()

            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)

            return text if text.strip() else None

        except Exception as e:
            print("IMAGE OCR ERROR:", e)
            return None

    except Exception as e:
        print("GENERAL OCR ERROR:", e)
        return None

# =========================
# LLM EXTRACTION
# =========================
def extract_expense_with_llm(text):
    prompt = f"""
Extract expense data.

Return ONLY JSON:
{{
  "amount": number,
  "category": string,
  "merchant": string,
  "date": string
}}

TEXT:
{text}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return safe_json_parse(res.choices[0].message.content)

# =========================
# SUPABASE UPLOAD
# =========================
def upload_file(file_bytes, filename):
    name = f"{uuid.uuid4()}_{filename}"

    supabase.storage.from_(BUCKET).upload(
        name,
        file_bytes,
        file_options={"content-type": "application/octet-stream"}
    )

    return supabase.storage.from_(BUCKET).get_public_url(name)

# =========================
# DB INSERT
# =========================
def insert_expense(data, raw_text):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id SERIAL PRIMARY KEY,
        amount FLOAT,
        category TEXT,
        merchant TEXT,
        expense_date TEXT,
        raw_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
        INSERT INTO expenses (amount, category, merchant, expense_date, raw_text)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        data.get("amount"),
        data.get("category"),
        data.get("merchant"),
        data.get("date"),
        raw_text
    ))

    conn.commit()
    cur.close()
    conn.close()

# =========================
# FILE LOG TABLE
# =========================
def insert_file_record(url, data):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses_file_upload (
        id SERIAL PRIMARY KEY,
        url TEXT,
        content JSONB,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
        INSERT INTO expenses_file_upload (url, content)
        VALUES (%s, %s)
    """, (url, json.dumps(data)))

    conn.commit()
    cur.close()
    conn.close()

# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"status": "running"}

# =========================
# TWILIO WEBHOOK (FINAL FIXED)
# =========================
@app.post("/twilio/webhook")
async def webhook(request: Request):
    response = MessagingResponse()

    try:
        form = await request.form()
        msg = form.get("Body", "")
        num_media = int(form.get("NumMedia", 0) or 0)

        # ======================
        # TEXT FLOW
        # ======================
        if num_media == 0:
            structured = extract_expense_with_llm(msg)

            insert_expense(structured, msg)

            response.message(
                f"✅ Saved ₹{structured.get('amount')} ({structured.get('category')})"
            )
            return str(response)

        # ======================
        # FILE FLOW
        # ======================
        media_url = form.get("MediaUrl0")

        file_bytes, content_type = download_media(media_url)

        if not file_bytes:
            response.message("❌ Failed to download file")
            return str(response)

        file_url = upload_file(file_bytes, "expense_file")

        raw_text = extract_text(file_bytes, content_type)

        if not raw_text:
            response.message("⚠️ Could not read image/PDF")
            return str(response)

        structured = extract_expense_with_llm(raw_text)

        insert_file_record(file_url, {
            "raw_text": raw_text,
            "llm": structured,
            "type": content_type
        })

        insert_expense(structured, raw_text)

        response.message("📊 Expense processed successfully")

    except Exception as e:
        print("FULL ERROR:", str(e))
        response.message("❌ Failed to process request")

    return str(response)