from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response
from groq import Groq
import psycopg2
import os
import re
import json
import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import resend
import requests
from twilio.rest import Client
from supabase import create_client, Client as SupabaseClient

# -------------------------------
# Load ENV
# -------------------------------
load_dotenv()
app = FastAPI()

# -------------------------------
# DB Connection
# -------------------------------
def get_db_connection():
    return psycopg2.connect(os.getenv("DB_SUPABASE_URL"))

# -------------------------------
# Supabase
# -------------------------------
def get_supabase() -> SupabaseClient:
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# -------------------------------
# Twilio
# -------------------------------
def get_twilio():
    return Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

# -------------------------------
# LLM
# -------------------------------
def get_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# OCR via Vision Model
# -------------------------------
def extract_text_from_file(file_url, content_type):
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all readable text from this receipt."},
                    {"type": "image_url", "image_url": {"url": file_url}}
                ]
            }],
            temperature=0
        )

        return res.choices[0].message.content

    except Exception as e:
        print(f"OCR error: {e}")
        return ""

# -------------------------------
# LLM Extraction
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": f"""
Extract expense info from:
{text}

Return:
{{
 "category": "materials|labour|transport|food|general",
 "amount": number,
 "merchant": "name",
 "notes": "desc"
}}

If none:
{{"category":"ignore","amount":0,"merchant":"unknown","notes":"{text}"}}
"""}
            ],
            temperature=0
        )

        content = res.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:
        print("LLM error:", e)

    return {"category": "ignore", "amount": 0, "merchant": "unknown", "notes": text}

# -------------------------------
# Save Expense
# -------------------------------
def save_expense(parsed, raw_text):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO expense (amount, category, merchant, expense_date, raw_text)
            VALUES (%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            parsed["amount"],
            parsed["category"],
            parsed["merchant"],
            datetime.date.today(),
            raw_text
        ))

        eid = cur.fetchone()[0]
        conn.commit()
        print("Saved expense:", eid)
        return eid

    except Exception as e:
        conn.rollback()
        print("DB error:", e)
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Process Media (FULL PIPELINE)
# -------------------------------
def process_media(message_sid, phone, raw_text):
    try:
        twilio = get_twilio()
        supabase = get_supabase()

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")

        media_list = twilio.messages(message_sid).media.list()

        conn = get_db_connection()
        cur = conn.cursor()

        for media in media_list:
            # --- Download media ---
            meta_url = f"https://api.twilio.com{media.uri}.json"
            meta = requests.get(meta_url, auth=(account_sid, auth_token)).json()

            media_url = f"https://api.twilio.com{media.uri}"
            file_data = requests.get(media_url, auth=(account_sid, auth_token)).content

            content_type = meta.get("content_type", "image/jpeg")
            ext = content_type.split("/")[-1]

            filename = f"{phone}_{datetime.datetime.now().timestamp()}.{ext}"
            path = f"expense-media/{filename}"

            # --- Upload to Supabase ---
            supabase.storage.from_("expense-files").upload(
                path,
                file_data,
                {"content-type": content_type}
            )

            file_url = supabase.storage.from_("expense-files").get_public_url(path)

            print("Uploaded:", file_url)

            # --- OCR ---
            extracted_text = extract_text_from_file(file_url, content_type)
            final_text = f"{raw_text}\n{extracted_text}"

            print("OCR TEXT:", extracted_text)

            # --- LLM Parse ---
            parsed = extract_expense(final_text)

            # --- Save file record ---
            cur.execute("""
                INSERT INTO expense_file_upload (url, content, merchant, expense_date, raw_text)
                VALUES (%s,%s,%s,%s,%s)
            """, (
                file_url,
                content_type,
                parsed.get("merchant"),
                datetime.date.today(),
                final_text
            ))

            conn.commit()

            # --- Save expense ---
            if parsed["amount"] > 0:
                save_expense(parsed, final_text)

        cur.close()
        conn.close()

    except Exception as e:
        print("Media error:", e)

# -------------------------------
# Webhook
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.form()

    msg = data.get("Body", "")
    sender = data.get("From", "")
    sid = data.get("MessageSid", "")
    num_media = int(data.get("NumMedia", 0))

    # TEXT
    if num_media == 0 and msg:
        parsed = extract_expense(msg)

        if parsed["amount"] > 0:
            save_expense(parsed, msg)
            return Response(f"<Response><Message>Saved ₹{parsed['amount']}</Message></Response>", media_type="application/xml")

        return Response("<Response><Message>No expense found</Message></Response>", media_type="application/xml")

    # MEDIA
    if num_media > 0:
        background_tasks.add_task(process_media, sid, sender, msg)
        return Response("<Response><Message>Processing receipt...</Message></Response>", media_type="application/xml")

    return Response("<Response><Message>Send text or receipt</Message></Response>", media_type="application/xml")

# -------------------------------
# Health
# -------------------------------
@app.get("/")
def health():
    return {"status": "running"}