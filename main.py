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
# SAFE JSON
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
# CREATE TABLES (FIXED)
# =========================
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # EXPENSES TABLE (FULL RESET SAFE VERSION)
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

    # FILE TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses_file_upload (
            id SERIAL PRIMARY KEY,
            url TEXT,
            content JSONB,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

# run once at startup
init_db()

# =========================
# OCR
# =========================
def extract_text(file_bytes, content_type):
    try:
        if "pdf" in content_type:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join([p.extract_text() or "" for p in pdf.pages])

        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("OCR ERROR:", e)
        return "OCR_FAILED"

# =========================
# LLM
# =========================
def extract_expense_with_llm(text):
    prompt = f"""
Extract expense info.

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
# INSERT HELPERS
# =========================
def insert_file_record(url, data):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO expenses_file_upload (url, content)
        VALUES (%s, %s)
    """, (url, json.dumps(data)))

    conn.commit()
    cur.close()
    conn.close()


def insert_expense(data, raw_text):
    conn = get_conn()
    cur = conn.cursor()

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
# ROOT (FIX FOR 404)
# =========================
@app.get("/")
def home():
    return {"status": "running"}

# =========================
# TWILIO WEBHOOK (FIXED)
# =========================
@app.post("/twilio/webhook")
async def webhook(request: Request):
    response = MessagingResponse()

    try:
        form = await request.form()
        msg = form.get("Body")
        num_media = int(form.get("NumMedia", 0))

        # TEXT
        if num_media == 0:
            insert_expense(
                {"amount": None, "category": None, "merchant": None, "date": None},
                msg
            )
            response.message("✅ Text expense saved")
            return str(response)

        # FILE
        media_url = form.get("MediaUrl0")
        content_type = form.get("MediaContentType0")

        file_bytes = requests.get(media_url).content

        file_url = upload_file(file_bytes, "file")

        raw_text = extract_text(file_bytes, content_type)

        structured = extract_expense_with_llm(raw_text)

        insert_file_record(file_url, {
            "raw_text": raw_text,
            "llm": structured,
            "type": content_type
        })

        insert_expense(structured, raw_text)

        response.message("📊 Expense processed successfully")

    except Exception as e:
        print("FULL ERROR:", e)
        response.message("❌ Failed to process file")

    return str(response)