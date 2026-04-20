from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response
from groq import Groq
import psycopg2
import psycopg2.extras
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
    """Extract expense from text using Groq."""
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
# Image Expense Extraction (GROQ VISION)
# -------------------------------
def extract_expense_from_image(image_url):
    """Extract expense from receipt using Groq Llama 3.2 Vision."""
    try:
        print(f"\n{'='*60}")
        print(f"🔍 GROQ VISION ANALYSIS")
        print(f"{'='*60}")
        print(f"Image URL: {image_url}")
       
        # Download and encode image
        print(f"⬇️  Downloading image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
       
        # Convert to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
       
        print(f"✅ Image downloaded and encoded ({len(response.content)} bytes)")
       
        # Get Groq client
        client = get_groq()
       
        # Prompt
        prompt = """You are an expert at reading receipts, bills, and invoices. Analyze this image carefully.

**TASK:** Extract expense information from this receipt/bill.

Look for:
1. **Total Amount** - Find the FINAL TOTAL (look for: "Total", "Grand Total", "Amount", "Net Amount", "Balance Due")
2. **Merchant Name** - Shop/company name (usually at the top)
3. **Category** - Based on items purchased
4. **Items** - Brief description of what was bought

**IMPORTANT RULES:**
- Extract the FINAL TOTAL amount (not subtotal or individual item prices)
- Remove currency symbols (₹, Rs, $) and commas - return just the number
- If multiple totals exist, choose the largest one near "Total"
- Make your best estimate even if image is slightly unclear

**RESPONSE FORMAT** (MUST be valid JSON only, no other text):
{
  "category": "materials",
  "amount": 15000,
  "merchant": "ABC Hardware",
  "notes": "cement bags, sand"
}

**Categories** (choose most appropriate):
- materials: cement, sand, bricks, steel, paint, hardware, building supplies
- labour: wages, worker payments, contractor fees
- transport: diesel, fuel, vehicle, delivery charges
- food: meals, groceries, restaurant
- general: anything else

If image is completely unreadable:
{
  "category": "general",
  "amount": 0,
  "merchant": "unreadable",
  "notes": "Image quality too poor"
}"""

        # Create vision request
        print(f"🤖 Sending to Groq Llama Vision...")
       
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",  # Groq's vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=500
        )
       
        content = completion.choices[0].message.content.strip()
       
        print(f"\n📄 GROQ RESPONSE:")
        print(f"{'-'*60}")
        print(content)
        print(f"{'-'*60}\n")
       
        # Clean response
        content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = content.strip()
       
        # Parse JSON
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if match:
            json_str = match.group()
            print(f"📋 Extracted JSON: {json_str}\n")
           
            parsed = json.loads(json_str)
           
            # Validate
            if all(k in parsed for k in ["category", "amount", "merchant", "notes"]):
                amount = float(parsed["amount"])
               
                if amount <= 0:
                    print(f"⚠️  Zero/negative amount: {amount}")
                    return {
                        "category": "general",
                        "amount": 100,
                        "merchant": "Review needed",
                        "notes": f"Extracted amount was {amount}. URL: {image_url}"
                    }
               
                parsed["amount"] = amount
               
                valid_cats = ["materials", "labour", "transport", "food", "general"]
                if parsed["category"] not in valid_cats:
                    parsed["category"] = "general"
               
                print(f"✅ SUCCESS - Extracted:")
                print(f"   Amount: ₹{amount:,.2f}")
                print(f"   Category: {parsed['category']}")
                print(f"   Merchant: {parsed['merchant']}")
                print(f"{'='*60}\n")
               
                return parsed
            else:
                print(f"⚠️  Missing required fields")
        else:
            print(f"⚠️  No JSON found in response")

    except requests.RequestException as e:
        print(f"❌ Download error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
    except Exception as e:
        print(f"❌ Groq vision error: {e}")
        import traceback
        traceback.print_exc()

    # Fallback
    print(f"⚠️  Using fallback values")
    return {
        "category": "general",
        "amount": 100,
        "merchant": "Manual review needed",
        "notes": f"Extraction failed. URL: {image_url}"
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
        print(f"\n💾 SAVING TO DATABASE")
        print(f"   Amount: ₹{parsed_data['amount']:,.2f}")
        print(f"   Category: {parsed_data['category']}")
        print(f"   Merchant: {parsed_data['merchant']}")
       
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
       
        print(f"✅ Expense saved! ID: {expense_id}\n")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Media Processing
# -------------------------------
def process_media(message_sid, phone, raw_text):
    try:
        print(f"\n{'#'*60}")
        print(f"📱 PROCESSING MEDIA - SID: {message_sid}")
        print(f"{'#'*60}\n")
       
        twilio_client = get_twilio()
        supabase = get_supabase()
       
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
       
        media_list = twilio_client.messages(message_sid).media.list()
       
        if not media_list:
            print("❌ No media found")
            return
       
        print(f"✅ Found {len(media_list)} file(s)\n")
       
        conn = get_db_connection()
        cur = conn.cursor()
       
        for idx, media in enumerate(media_list, 1):
            print(f"{'-'*60}")
            print(f"📄 FILE {idx}/{len(media_list)}")
            print(f"{'-'*60}")
           
            # Get metadata
            uri = media.uri or ""
            json_path = uri if uri.endswith(".json") else f"{uri}.json"
            meta_url = f"https://api.twilio.com{json_path}"
            meta_resp = requests.get(meta_url, auth=(account_sid, auth_token))
            meta_resp.raise_for_status()
            meta = meta_resp.json()
           
            # Download
            binary_path = json_path.removesuffix(".json")
            media_url = f"https://api.twilio.com{binary_path}"
            response = requests.get(media_url, auth=(account_sid, auth_token))
            response.raise_for_status()
           
            print(f"✅ Downloaded ({len(response.content)} bytes)")
           
            # File info
            content_type = meta.get("content_type") or media.content_type or "application/octet-stream"
            ext = content_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
           
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{phone.replace('+', '')}_{timestamp}_{media.sid}.{ext}"
           
            # Upload to Supabase
            storage_path = f"expense-media/{filename}"
            supabase.storage.from_("expense-files").upload(
                storage_path,
                response.content,
                {"content-type": content_type}
            )
           
            file_url = supabase.storage.from_("expense-files").get_public_url(storage_path)
            print(f"✅ Uploaded: {storage_path}")
            print(f"✅ URL: {file_url}")
           
            # Metadata
            content_metadata = {
                "content_type": content_type,
                "file_size": len(response.content),
                "filename": filename,
                "storage_path": storage_path,
                "twilio_media_sid": media.sid,
                "uploaded_at": datetime.datetime.now().isoformat()
            }
           
            # Save to expense_file_upload
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
                print(f"✅ Saved to expense_file_upload, ID: {file_upload_id}")
               
            except Exception as e:
                conn.rollback()
                print(f"❌ Error saving file upload: {e}")
                continue
           
            # Process images with Groq Vision
            if content_type.startswith("image/"):
                print(f"\n🖼️  IMAGE DETECTED - Starting Groq Vision analysis...\n")
               
                parsed_expense = extract_expense_from_image(file_url)
               
                # Save to expense table
                expense_id = save_expense_from_file(parsed_expense, file_url)
               
                if expense_id:
                    # Update file_upload with extraction data
                    try:
                        content_metadata["llm_extraction"] = {
                            "extracted_at": datetime.datetime.now().isoformat(),
                            "model": "llama-3.2-90b-vision-preview",
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
                        print(f"✅ Updated file_upload with extraction data")
                       
                    except Exception as e:
                        conn.rollback()
                        print(f"❌ Error updating file_upload: {e}")
            else:
                print(f"⚠️  Non-image file ({content_type})")
       
        cur.close()
        conn.close()
       
        print(f"\n{'#'*60}")
        print(f"✅ MEDIA PROCESSING COMPLETE")
        print(f"{'#'*60}\n")
       
    except Exception as e:
        print(f"\n❌ MEDIA PROCESSING ERROR: {e}")
        import traceback
        traceback.print_exc()

# -------------------------------
# Analytics Functions
# -------------------------------
def get_today_summary():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT category, SUM(amount)
        FROM expense
        WHERE expense_date = CURRENT_DATE
        GROUP BY category
        ORDER BY SUM(amount) DESC
    """)

    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def generate_chart(data):
    if not data:
        return None
   
    categories = [x[0] for x in data]
    amounts = [float(x[1]) for x in data]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, amounts, color='steelblue')
    plt.title("Daily Expenses by Category", fontsize=14, fontweight='bold')
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Amount (₹)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
   
    for i, v in enumerate(amounts):
        plt.text(i, v, f'₹{v:,.0f}', ha='center', va='bottom')
   
    plt.tight_layout()

    path = "/tmp/chart.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path

def analyze_with_llm(data):
    client = get_groq()

    data_dict = {cat: float(amt) for cat, amt in data}
    total = sum(data_dict.values())

    prompt = f"""Financial analyst for construction expenses.

Today's data:
{json.dumps(data_dict, indent=2)}

Total: ₹{total:,.2f}

Provide brief analysis (max 150 words):
1. Total spend
2. Highest category & %
3. Anomalies
4. Cost optimization tip
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return res.choices[0].message.content

def send_email(report, chart_path):
    resend.api_key = os.getenv("RESEND_API_KEY")

    attachments = []
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            chart = f.read()
        attachments.append({
            "filename": "expense_chart.png",
            "content": chart
        })

    resend.Emails.send({
        "from": "Expense Tracker <onboarding@resend.dev>",
        "to": [os.getenv("EMAIL_TO")],
        "subject": f"Daily Expense Report - {datetime.date.today().strftime('%d %b %Y')}",
        "html": f"<h2>Daily Expense Analysis</h2><pre>{report}</pre>",
        "attachments": attachments
    })

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.form()

    msg = data.get("Body", "").strip()
    sender = data.get("From", "")
    message_sid = data.get("MessageSid", "")
    num_media = int(data.get("NumMedia", 0))

    print(f"\n📨 From {sender}: {msg} ({num_media} files)")

    # Text only
    if num_media == 0 and msg:
        parsed = extract_expense(msg)

        if parsed["category"] == "ignore":
            return Response(
                "<Response><Message>❌ No expense info. Please specify amount and category.</Message></Response>",
                media_type="application/xml"
            )

        expense_id = save_text_expense(sender, parsed, msg)
       
        if expense_id:
            response_msg = f"✅ Saved ₹{parsed['amount']:,} ({parsed['category']})"
        else:
            response_msg = "❌ Error saving. Try again."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Media
    elif num_media > 0:
        background_tasks.add_task(process_media, message_sid, sender, msg)
       
        response_msg = f"✅ Received {num_media} file(s). Analyzing with Groq Vision AI..."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Empty
    else:
        return Response(
            "<Response><Message>Send expense details or attach receipt image.</Message></Response>",
            media_type="application/xml"
        )

@app.get("/send-daily-report")
def send_daily():
    data = get_today_summary()

    if not data:
        return {"status": "no data"}

    chart = generate_chart(data)
    report = analyze_with_llm(data)

    send_email(report, chart)

    total = sum(float(x[1]) for x in data)

    return {
        "status": "sent",
        "date": datetime.date.today().isoformat(),
        "total": f"₹{total:,.2f}",
        "breakdown": {cat: float(amt) for cat, amt in data}
    }

@app.get("/")
def health():
    return {
        "status": "active",
        "service": "expense-tracker",
        "version": "4.0-groq-vision",
        "models": {
            "text": "llama-3.1-8b-instant",
            "vision": "llama-3.2-90b-vision-preview",
            "analysis": "llama-3.3-70b-versatile"
        },
        "provider": "Groq (100% free)"
    }

@app.get("/expenses/today")
def get_today():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, amount, category, merchant, raw_text,
               TO_CHAR(expense_date, 'YYYY-MM-DD') as date
        FROM expense
        WHERE expense_date = CURRENT_DATE
        ORDER BY id DESC
    """)

    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.close()
    conn.close()

    total = sum(float(r['amount']) for r in results)

    return {
        "date": datetime.date.today().isoformat(),
        "count": len(results),
        "total": f"₹{total:,.2f}",
        "expenses": results
    }

@app.get("/file-uploads/today")
def get_uploads():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT id, url, content, merchant,
               TO_CHAR(expense_date, 'YYYY-MM-DD') as date, raw_text
        FROM expense_file_upload
        WHERE expense_date = CURRENT_DATE
        ORDER BY id DESC
    """)

    results = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "date": datetime.date.today().isoformat(),
        "count": len(results),
        "uploads": results
    }

@app.post("/test-extract-image")
async def test_image(request: Request):
    data = await request.json()
    image_url = data.get("image_url", "")
   
    if not image_url:
        return {"error": "No image_url provided"}
   
    result = extract_expense_from_image(image_url)
    return {
        "status": "success",
        "extracted": result,
        "needs_review": result.get("merchant") == "Manual review needed"
    }