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
# Database Connection
# -------------------------------
def get_db_connection():
    """Get PostgreSQL connection to Supabase."""
    return psycopg2.connect(os.getenv("DB_SUPABASE_URL"))

# -------------------------------
# Supabase Client (for Storage)
# -------------------------------
def get_supabase() -> SupabaseClient:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

# -------------------------------
# Twilio Client
# -------------------------------
def get_twilio():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    return Client(account_sid, auth_token)

# -------------------------------
# LLM Client
# -------------------------------
def get_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Expense Extraction (fallback)
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
# LLM Extraction
# -------------------------------
def extract_expense(text):
    """Extract expense information using LLM."""
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expense extraction assistant. Return only valid JSON."},
                {"role": "user", "content": f"""
Extract expense information from this unstructured message:
"{text}"

Return JSON in this exact format:
{{
    "category": "materials|labour|transport|food|general",
    "amount": <number>,
    "merchant": "<merchant name or 'unknown'>",
    "notes": "<brief description>"
}}

Examples:
- "10k spend on labour today" → {{"category":"labour", "amount":10000, "merchant":"unknown", "notes":"labour expenses"}}
- "Paid 5000 to ABC Cement" → {{"category":"materials", "amount":5000, "merchant":"ABC Cement", "notes":"cement purchase"}}

If no expense info found, return: {{"category":"ignore", "amount":0, "merchant":"unknown", "notes":"{text}"}}
"""}
            ],
            temperature=0
        )

        content = res.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            # Validate required fields
            if all(k in parsed for k in ["category", "amount", "merchant", "notes"]):
                return parsed

    except Exception as e:
        print(f"LLM extraction error: {e}")

    return extract_expense_simple(text)

# -------------------------------
# Save Text Expense to DB
# -------------------------------
def save_text_expense(phone, parsed_data, raw_text):
    """Save text-based expense to expense table."""
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
       
        print(f"Saved text expense ID: {expense_id}")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"Error saving text expense: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Download and Upload Media
# -------------------------------
def process_media(message_sid, phone, raw_text):
    """Download media from Twilio, upload to Supabase, and save to expense_file_upload table."""
    try:
        twilio_client = get_twilio()
        supabase = get_supabase()
       
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
       
        # Get media list
        media_list = twilio_client.messages(message_sid).media.list()
       
        if not media_list:
            print("No media found")
            return
       
        conn = get_db_connection()
        cur = conn.cursor()
       
        for media in media_list:
            # Get media metadata
            uri = media.uri or ""
            json_path = uri if uri.endswith(".json") else f"{uri}.json"
            meta_url = f"https://api.twilio.com{json_path}"
            meta_resp = requests.get(meta_url, auth=(account_sid, auth_token))
            meta_resp.raise_for_status()
            meta = meta_resp.json()
           
            # Download binary content
            binary_path = json_path.removesuffix(".json")
            media_url = f"https://api.twilio.com{binary_path}"
            response = requests.get(media_url, auth=(account_sid, auth_token))
            response.raise_for_status()
           
            # Determine file extension
            content_type = meta.get("content_type") or media.content_type or "application/octet-stream"
            ext = content_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
           
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{phone.replace('+', '')}_{timestamp}_{media.sid}.{ext}"
           
            # Upload to Supabase Storage (bucket: expense-files)
            storage_path = f"expense-media/{filename}"
            supabase.storage.from_("expense-files").upload(
                storage_path,
                response.content,
                {"content-type": content_type}
            )
           
            # Get public URL
            file_url = supabase.storage.from_("expense-files").get_public_url(storage_path)
           
            # Extract expense info from raw_text using LLM
            parsed = extract_expense(raw_text) if raw_text else {
                "category": "general",
                "merchant": "unknown"
            }
           
            # Save to expense_file_upload table
            try:
                cur.execute("""
                    INSERT INTO expense_file_upload (url, content, merchant, expense_date, raw_text)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    file_url,
                    content_type,
                    parsed.get("merchant", "unknown"),
                    datetime.date.today(),
                    raw_text or ""
                ))
               
                file_id = cur.fetchone()[0]
                conn.commit()
               
                print(f"Uploaded media ID: {file_id}, URL: {file_url}")
               
            except Exception as e:
                conn.rollback()
                print(f"Error saving file upload record: {e}")
       
        cur.close()
        conn.close()
       
    except Exception as e:
        print(f"Media processing error: {e}")

# -------------------------------
# Query today's data
# -------------------------------
def get_today_summary():
    """Get today's expense summary by category."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT category, SUM(amount)
        FROM expense
        WHERE expense_date = CURRENT_DATE
        GROUP BY category
    """)

    data = cur.fetchall()

    cur.close()
    conn.close()
    return data

# -------------------------------
# Chart generation
# -------------------------------
def generate_chart(data):
    """Generate bar chart of expenses by category."""
    if not data:
        return None
   
    categories = [x[0] for x in data]
    amounts = [float(x[1]) for x in data]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, amounts, color='steelblue')
    plt.title("Daily Expenses by Category")
    plt.xlabel("Category")
    plt.ylabel("Amount (₹)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    path = "/tmp/chart.png"
    plt.savefig(path)
    plt.close()

    return path

# -------------------------------
# LLM Analysis
# -------------------------------
def analyze_with_llm(data):
    """Generate expense analysis using LLM."""
    client = get_groq()

    data_dict = {cat: float(amt) for cat, amt in data}
    total = sum(data_dict.values())

    prompt = f"""
You are a financial analyst for construction expenses.

Today's expense data by category:
{json.dumps(data_dict, indent=2)}

Total: ₹{total:,.2f}

Provide a brief analysis covering:
1. Total spend for the day
2. Highest spending category and percentage
3. Any anomalies or unusual patterns
4. One actionable suggestion for cost optimization

Keep the response concise and professional (max 150 words).
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return res.choices[0].message.content

# -------------------------------
# Email via Resend
# -------------------------------
def send_email(report, chart_path):
    """Send daily report email with chart attachment."""
    resend.api_key = os.getenv("RESEND_API_KEY")

    attachments = []
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            chart = f.read()
        attachments.append({
            "filename": "chart.png",
            "content": chart
        })

    resend.Emails.send({
        "from": "Expense Tracker <onboarding@resend.dev>",
        "to": [os.getenv("EMAIL_TO")],
        "subject": "Daily Expense Report",
        "html": f"<pre>{report}</pre>",
        "attachments": attachments
    })

# -------------------------------
# Webhook
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming Twilio messages."""
    data = await request.form()

    msg = data.get("Body", "").strip()
    sender = data.get("From", "")
    message_sid = data.get("MessageSid", "")
    num_media = int(data.get("NumMedia", 0))

    print(f"Received message from {sender}: {msg}, Media: {num_media}")

    # Case 1: Only text message (no media)
    if num_media == 0 and msg:
        parsed = extract_expense(msg)

        if parsed["category"] == "ignore":
            return Response(
                "<Response><Message>No expense information found in your message.</Message></Response>",
                media_type="application/xml"
            )

        # Save to expense table
        expense_id = save_text_expense(sender, parsed, msg)
       
        if expense_id:
            response_msg = f"✓ Saved ₹{parsed['amount']:,} to {parsed['category']}."
        else:
            response_msg = "Error saving expense. Please try again."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Case 2: Media present (with or without text)
    elif num_media > 0:
        # Process media in background
        background_tasks.add_task(process_media, message_sid, sender, msg)
       
        response_msg = f"✓ Received {num_media} file(s). Processing..."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Case 3: Empty message
    else:
        return Response(
            "<Response><Message>Please send expense details or attach a receipt.</Message></Response>",
            media_type="application/xml"
        )

# -------------------------------
# Daily Report Endpoint
# -------------------------------
@app.get("/send-daily-report")
def send_daily():
    """Generate and send daily expense report."""
    data = get_today_summary()

    if not data:
        return {"status": "no data for today"}

    chart = generate_chart(data)
    report = analyze_with_llm(data)

    send_email(report, chart)

    return {
        "status": "sent",
        "categories": len(data),
        "total_expenses": sum(float(x[1]) for x in data)
    }

# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def health_check():
    """API health check endpoint."""
    return {
        "status": "active",
        "service": "expense-tracker",
        "version": "2.0"
    }

# -------------------------------
# Test Expense Extraction
# -------------------------------
@app.post("/test-extract")
async def test_extract(request: Request):
    """Test endpoint for expense extraction."""
    data = await request.json()
    text = data.get("text", "")
   
    if not text:
        return {"error": "No text provided"}
   
    result = extract_expense(text)
    return result