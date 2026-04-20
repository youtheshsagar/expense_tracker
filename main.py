from fastapi import FastAPI, Request
from twilio.twiml.messaging_response import MessagingResponse
from supabase import create_client
import psycopg2
import requests
import json
import uuid
import io
import re

# OCR
import pytesseract
from PIL import Image
import pdfplumber

# LLM (GROQ)
from groq import Groq

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# =========================
# SUPABASE STORAGE
# =========================
SUPABASE_URL = "https://zasmsfjahuuwafjiajqp.supabase.co"
SUPABASE_KEY = "sb_secret_iIdMxUb7v3TkVsYAeluidw_xswTv5kp"
BUCKET = "expense-files"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# POSTGRES DB
# =========================
DB_CONN_STRING = "postgresql://postgres.lcseodogwkycxnfqxjib:Youthi%40220387@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"

def get_conn():
    return psycopg2.connect(DB_CONN_STRING)

# =========================
# GROQ LLM CLIENT
# =========================
client = Groq(api_key="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")

# =========================
# SAFE JSON PARSER
# =========================
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# =========================
# LLM EXPENSE EXTRACTION
# =========================
def extract_expense_with_llm(text):
    prompt = f"""
You are an expense extraction system.

Extract structured data from the text below.

Return ONLY valid JSON:
{{
  "amount": number,
  "category": string,
  "merchant": string,
  "date": string
}}

TEXT:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return safe_json_parse(response.choices[0].message.content)

# =========================
# OCR / PDF TEXT EXTRACTION
# =========================
def extract_text(file_bytes, content_type):
    if "pdf" in content_type:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])

    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)

# =========================
# SUPABASE FILE UPLOAD
# =========================
def upload_file(file_bytes, filename):
    unique_name = f"{uuid.uuid4()}_{filename}"

    supabase.storage.from_(BUCKET).upload(unique_name, file_bytes)

    url = supabase.storage.from_(BUCKET).get_public_url(unique_name)

    return url

# =========================
# DATABASE FUNCTIONS
# =========================
def insert_text_expense(text):
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
        INSERT INTO expenses (raw_text)
        VALUES (%s)
    """, (text,))

    conn.commit()
    cur.close()
    conn.close()


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


def insert_structured_expense(data, raw_text):
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
# TWILIO WEBHOOK
# =========================
@app.post("/twilio/webhook")
async def twilio_webhook(request: Request):
    form = await request.form()
    response = MessagingResponse()

    try:
        msg = form.get("Body")
        num_media = int(form.get("NumMedia", 0))

        # =========================
        # TEXT MESSAGE
        # =========================
        if num_media == 0:
            insert_text_expense(msg)
            response.message("✅ Expense saved from text")
            return str(response)

        # =========================
        # FILE MESSAGE
        # =========================
        media_url = form.get("MediaUrl0")
        content_type = form.get("MediaContentType0")

        file_bytes = requests.get(media_url).content

        # Upload to Supabase
        file_url = upload_file(file_bytes, "expense_file")

        # Extract text (OCR / PDF)
        raw_text = extract_text(file_bytes, content_type)

        # LLM processing
        structured = extract_expense_with_llm(raw_text)

        # Save file upload record
        insert_file_record(file_url, {
            "raw_text": raw_text,
            "llm_output": structured,
            "media_type": content_type
        })

        # Save structured expense
        insert_structured_expense(structured, raw_text)

        response.message("📊 Expense extracted & saved successfully")

    except Exception as e:
        print("ERROR:", e)
        response.message("❌ Failed to process file")

    return str(response)
