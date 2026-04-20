from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from groq import Groq
import psycopg2
import psycopg2.extras
import os
import re
import json
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import resend
import requests
from twilio.rest import Client
from supabase import create_client, Client as SupabaseClient
import base64

load_dotenv()
app = FastAPI()

# -------------------------------
# Connections
# -------------------------------
def get_db_connection():
    return psycopg2.connect(os.getenv("DB_SUPABASE_URL"))

def get_supabase() -> SupabaseClient:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def get_twilio():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    return Client(account_sid, auth_token)

def get_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Fallback Extraction
# -------------------------------
def extract_expense_simple(text):
    raw = text.lower()
    amount = 0

    m = re.search(r"(\d+)\s*k", raw)
    if m:
        amount = int(m.group(1)) * 1000
    else:
        m = re.search(r"\d+", raw)
        if m:
            amount = int(m.group())

    category = "general"
    merchant = "unknown"
   
    if "cement" in raw or "sand" in raw or "material" in raw:
        category = "materials"
    elif "labour" in raw or "worker" in raw:
        category = "labour"
    elif "diesel" in raw or "fuel" in raw:
        category = "transport"
    elif "food" in raw or "lunch" in raw or "dinner" in raw:
        category = "food"

    if amount == 0:
        return {"category": "ignore", "amount": 0, "merchant": merchant, "notes": text}

    return {"category": category, "amount": amount, "merchant": merchant, "notes": text}

# -------------------------------
# Text Expense Extraction
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expense extraction assistant. Return only valid JSON."},
                {"role": "user", "content": f"""
Extract expense information from: "{text}"

Return JSON:
{{
    "category": "materials|labour|transport|food|general",
    "amount": <number>,
    "merchant": "<name or 'unknown'>",
    "notes": "<description>"
}}

Examples:
- "10k labour" → {{"category":"labour", "amount":10000, "merchant":"unknown", "notes":"labour expenses"}}
- "5000 ABC Cement" → {{"category":"materials", "amount":5000, "merchant":"ABC Cement", "notes":"cement"}}

If no expense: {{"category":"ignore", "amount":0, "merchant":"unknown", "notes":"{text}"}}
"""}
            ],
            temperature=0
        )

        content = res.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if all(k in parsed for k in ["category", "amount", "merchant", "notes"]):
                return parsed

    except Exception as e:
        print(f"❌ LLM extraction error: {e}")

    return extract_expense_simple(text)

# -------------------------------
# Image Expense Extraction
# -------------------------------
def extract_expense_from_image(image_url):
    try:
        print(f"🔍 Analyzing image: {image_url}")
       
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
       
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
       
        client = get_groq()
       
        prompt = """Extract expense from this receipt/bill image.

Return ONLY valid JSON:
{
  "category": "materials|labour|transport|food|general",
  "amount": <number>,
  "merchant": "<shop name>",
  "notes": "<brief description>"
}

Find the TOTAL amount (remove ₹, Rs, $, commas).
Choose appropriate category based on items purchased."""

        completion = client.chat.completions.create(
            model="llama-4-scout",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=300
        )
       
        content = completion.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content).strip()
       
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
           
            if all(k in parsed for k in ["category", "amount", "merchant", "notes"]):
                amount = float(parsed["amount"])
               
                if amount <= 0:
                    return {
                        "category": "general",
                        "amount": 100,
                        "merchant": "Review needed",
                        "notes": f"Extracted amount was {amount}"
                    }
               
                parsed["amount"] = amount
               
                valid_cats = ["materials", "labour", "transport", "food", "general"]
                if parsed["category"] not in valid_cats:
                    parsed["category"] = "general"
               
                print(f"✅ Extracted: ₹{amount:,.2f} - {parsed['category']}")
                return parsed

    except Exception as e:
        print(f"❌ Vision error: {e}")

    return {
        "category": "general",
        "amount": 100,
        "merchant": "Manual review needed",
        "notes": "Extraction failed"
    }

# -------------------------------
# Database Operations
# -------------------------------
def save_text_expense(phone, parsed_data, raw_text):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO expense (amount, category, merchant, expense_date, raw_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            parsed_data["amount"],
            parsed_data["category"],
            parsed_data["merchant"],
            datetime.date.today(),
            raw_text
        ))
       
        expense_id = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Saved text expense ID: {expense_id}")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving text expense: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def save_expense_from_file(parsed_data, file_url):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        print(f"💾 Saving: ₹{parsed_data['amount']:,.2f} - {parsed_data['category']}")
       
        cur.execute("""
            INSERT INTO expense (amount, category, merchant, expense_date, raw_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            parsed_data["amount"],
            parsed_data["category"],
            parsed_data["merchant"],
            datetime.date.today(),
            f"Image: {file_url}\n{parsed_data['notes']}"
        ))
       
        expense_id = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Saved ID: {expense_id}")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Media Processing
# -------------------------------
def process_media(message_sid, phone, raw_text):
    try:
        print(f"📱 Processing media - SID: {message_sid}")
       
        twilio_client = get_twilio()
        supabase = get_supabase()
       
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
       
        media_list = twilio_client.messages(message_sid).media.list()
       
        if not media_list:
            print("❌ No media found")
            return
       
        print(f"✅ Found {len(media_list)} file(s)")
       
        conn = get_db_connection()
        cur = conn.cursor()
       
        for idx, media in enumerate(media_list, 1):
            print(f"📄 Processing file {idx}/{len(media_list)}")
           
            uri = media.uri or ""
            json_path = uri if uri.endswith(".json") else f"{uri}.json"
            meta_url = f"https://api.twilio.com{json_path}"
            meta_resp = requests.get(meta_url, auth=(account_sid, auth_token))
            meta_resp.raise_for_status()
            meta = meta_resp.json()
           
            binary_path = json_path.removesuffix(".json")
            media_url = f"https://api.twilio.com{binary_path}"
            response = requests.get(media_url, auth=(account_sid, auth_token))
            response.raise_for_status()
           
            content_type = meta.get("content_type") or media.content_type or "application/octet-stream"
            ext = content_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
           
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{phone.replace('+', '')}_{timestamp}_{media.sid}.{ext}"
           
            storage_path = f"expense-media/{filename}"
            supabase.storage.from_("expense-files").upload(
                storage_path,
                response.content,
                {"content-type": content_type}
            )
           
            file_url = supabase.storage.from_("expense-files").get_public_url(storage_path)
            print(f"✅ Uploaded: {file_url}")
           
            content_metadata = {
                "content_type": content_type,
                "file_size": len(response.content),
                "filename": filename,
                "storage_path": storage_path,
                "twilio_media_sid": media.sid,
                "uploaded_at": datetime.datetime.now().isoformat()
            }
           
            try:
                cur.execute("""
                    INSERT INTO expense_file_upload (url, content, merchant, expense_date, raw_text)
                    VALUES (%s, %s::jsonb, %s, %s, %s)
                    RETURNING id
                """, (
                    file_url,
                    json.dumps(content_metadata),
                    "unknown",
                    datetime.date.today(),
                    raw_text or ""
                ))
               
                file_upload_id = cur.fetchone()[0]
                conn.commit()
                print(f"✅ Saved file upload ID: {file_upload_id}")
               
            except Exception as e:
                conn.rollback()
                print(f"❌ Error saving file upload: {e}")
                continue
           
            if content_type.startswith("image/"):
                print(f"🖼️ Analyzing image with Llama 4 Scout...")
               
                parsed_expense = extract_expense_from_image(file_url)
                expense_id = save_expense_from_file(parsed_expense, file_url)
               
                if expense_id:
                    try:
                        content_metadata["llm_extraction"] = {
                            "extracted_at": datetime.datetime.now().isoformat(),
                            "model": "llama-4-scout",
                            "expense_id": expense_id,
                            "extracted_data": parsed_expense
                        }
                       
                        cur.execute("""
                            UPDATE expense_file_upload
                            SET merchant = %s, content = %s::jsonb
                            WHERE id = %s
                        """, (
                            parsed_expense["merchant"],
                            json.dumps(content_metadata),
                            file_upload_id
                        ))
                        conn.commit()
                       
                    except Exception as e:
                        conn.rollback()
                        print(f"❌ Error updating file_upload: {e}")
       
        cur.close()
        conn.close()
        print(f"✅ Media processing complete")
       
    except Exception as e:
        print(f"❌ Media processing error: {e}")

# -------------------------------
# SIMPLE EMAIL REPORT ENDPOINT
# -------------------------------
@app.get("/send-report")
def send_email_report():
    """
    Simple endpoint to send expense report email.
    Call this URL: http://your-domain.com/send-report
    """
    try:
        print("📧 Generating report...")
       
        # Get today's expenses
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
       
        cur.execute("""
            SELECT
                id, amount, category, merchant, raw_text,
                TO_CHAR(expense_date, 'YYYY-MM-DD HH24:MI') as date
            FROM expense
            WHERE expense_date >= CURRENT_DATE
            ORDER BY expense_date DESC, id DESC
        """)
       
        expenses = cur.fetchall()
       
        if not expenses:
            cur.close()
            conn.close()
            return {
                "status": "no_data",
                "message": "No expenses found today"
            }
       
        # Calculate totals by category
        cur.execute("""
            SELECT
                category,
                COUNT(*) as count,
                SUM(amount) as total
            FROM expense
            WHERE expense_date >= CURRENT_DATE
            GROUP BY category
            ORDER BY total DESC
        """)
       
        summary = cur.fetchall()
        cur.close()
        conn.close()
       
        total_amount = sum(float(row['total']) for row in summary)
       
        print(f"✅ Found {len(expenses)} expenses, Total: ₹{total_amount:,.2f}")
       
        # Build simple HTML email
        category_rows = ""
        for row in summary:
            percentage = (float(row['total']) / total_amount * 100) if total_amount > 0 else 0
            category_rows += f"""
            <tr>
                <td style="padding: 10px;">{row['category'].title()}</td>
                <td style="padding: 10px; text-align: right; font-weight: bold;">₹{float(row['total']):,.2f}</td>
                <td style="padding: 10px; text-align: center;">{int(row['count'])}</td>
                <td style="padding: 10px; text-align: right;">{percentage:.1f}%</td>
            </tr>
            """
       
        expense_rows = ""
        for exp in expenses:
            expense_rows += f"""
            <tr>
                <td style="padding: 8px; font-size: 13px;">{exp['date']}</td>
                <td style="padding: 8px; font-size: 13px;">{exp['category'].title()}</td>
                <td style="padding: 8px; font-size: 13px;">{exp['merchant']}</td>
                <td style="padding: 8px; text-align: right; font-weight: bold;">₹{float(exp['amount']):,.2f}</td>
            </tr>
            """
       
        html = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
            <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
               
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                    <h1 style="margin: 0; color: white; font-size: 28px;">📊 Daily Expense Report</h1>
                    <p style="margin: 10px 0 0; color: rgba(255,255,255,0.9);">{datetime.date.today().strftime('%B %d, %Y')}</p>
                </div>
               
                <!-- Total -->
                <div style="background: #f9fafb; padding: 25px; text-align: center;">
                    <p style="margin: 0; color: #666; font-size: 12px; text-transform: uppercase;">Total Expenses</p>
                    <p style="margin: 10px 0 0; color: #1f2937; font-size: 40px; font-weight: bold;">₹{total_amount:,.2f}</p>
                    <p style="margin: 5px 0 0; color: #666;">{len(expenses)} transactions</p>
                </div>
               
                <!-- Category Summary -->
                <div style="padding: 25px;">
                    <h2 style="margin: 0 0 15px; font-size: 20px;">📊 By Category</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: #f3f4f6;">
                                <th style="padding: 10px; text-align: left;">Category</th>
                                <th style="padding: 10px; text-align: right;">Total</th>
                                <th style="padding: 10px; text-align: center;">Count</th>
                                <th style="padding: 10px; text-align: right;">%</th>
                            </tr>
                        </thead>
                        <tbody>
                            {category_rows}
                        </tbody>
                    </table>
                </div>
               
                <!-- Transactions -->
                <div style="padding: 25px; background: #f9fafb;">
                    <h2 style="margin: 0 0 15px; font-size: 20px;">📝 All Transactions</h2>
                    <table style="width: 100%; border-collapse: collapse; background: white;">
                        <thead>
                            <tr style="background: #667eea; color: white;">
                                <th style="padding: 10px; text-align: left;">Date</th>
                                <th style="padding: 10px; text-align: left;">Category</th>
                                <th style="padding: 10px; text-align: left;">Merchant</th>
                                <th style="padding: 10px; text-align: right;">Amount</th>
                            </tr>
                        </thead>
                        <tbody>
                            {expense_rows}
                        </tbody>
                    </table>
                </div>
               
                <!-- Footer -->
                <div style="padding: 20px; text-align: center; background: #1f2937; color: white;">
                    <p style="margin: 0; font-size: 12px;">Expense Tracker Pro - Powered by Llama AI</p>
                </div>
               
            </div>
        </body>
        </html>
        """
       
        # Send email using Resend
        resend.api_key = os.getenv("RESEND_API_KEY")
       
        email_params = {
            "from": os.getenv("EMAIL_FROM", "Expense Tracker <onboarding@resend.dev>"),
            "to": [os.getenv("EMAIL_TO")],
            "subject": f"📊 Expense Report - {datetime.date.today().strftime('%B %d, %Y')}",
            "html": html
        }
       
        response = resend.Emails.send(email_params)
       
        print(f"✅ Email sent! ID: {response.get('id')}")
       
        return {
            "status": "success",
            "message": "Report sent successfully",
            "email_id": response.get('id'),
            "date": datetime.date.today().isoformat(),
            "total": f"₹{total_amount:,.2f}",
            "transactions": len(expenses),
            "breakdown": {row['category']: float(row['total']) for row in summary}
        }
       
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
       
        return {
            "status": "error",
            "message": str(e)
        }

# -------------------------------
# WhatsApp Webhook
# -------------------------------
@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        form_data = await request.form()
        data = dict(form_data)
       
        from_number = data.get("From", "")
        body = data.get("Body", "").strip()
        message_sid = data.get("MessageSid", "")
        num_media = int(data.get("NumMedia", 0))
       
        print(f"\n📩 WhatsApp from {from_number}: {body}")
       
        if num_media > 0:
            print(f"📎 {num_media} file(s) attached")
            background_tasks.add_task(process_media, message_sid, from_number, body)
            return Response(content="", media_type="text/plain")
       
        if not body or len(body) < 2:
            return Response(content="", media_type="text/plain")
       
        parsed = extract_expense(body)
       
        if parsed["category"] != "ignore" and parsed["amount"] > 0:
            expense_id = save_text_expense(from_number, parsed, body)
           
            if expense_id:
                print(f"✅ Saved: ₹{parsed['amount']} ({parsed['category']})")
       
        return Response(content="", media_type="text/plain")
       
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content="", media_type="text/plain", status_code=500)

# -------------------------------
# Basic Endpoints
# -------------------------------
@app.get("/")
def root():
    return {"status": "running", "message": "Expense Tracker API"}

@app.get("/expenses/today")
def get_today_expenses():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
   
    cur.execute("""
        SELECT id, amount, category, merchant,
               TO_CHAR(expense_date, 'YYYY-MM-DD HH24:MI') as date
        FROM expense
        WHERE expense_date >= CURRENT_DATE
        ORDER BY expense_date DESC
    """)
   
    expenses = cur.fetchall()
    total = sum(float(e['amount']) for e in expenses)
   
    cur.close()
    conn.close()
   
    return {
        "date": datetime.date.today().isoformat(),
        "total": total,
        "count": len(expenses),
        "expenses": expenses
    }