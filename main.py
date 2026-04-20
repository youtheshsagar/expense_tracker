
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
# LLM Extraction from Text
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
# LLM Vision - Extract from Image (IMPROVED)
# -------------------------------
def extract_expense_from_image(image_url):
    """Extract expense information from receipt/bill image using LLM vision."""
    try:
        print(f"🔍 Starting LLM vision analysis for: {image_url}")
        client = get_groq()

        # Using llama-3.3-70b-versatile with vision capabilities
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert at reading receipts, bills, and invoices. Analyze this image carefully.

TASK: Extract the following information:
1. Total amount (look for words like: Total, Grand Total, Amount, Net Amount, Balance)
2. Merchant/Shop name (usually at the top of the receipt)
3. Category of expense based on items purchased
4. Brief description of items

IMPORTANT RULES:
- Look for the FINAL TOTAL amount (not subtotals or item prices)
- If you see multiple numbers, the largest one near "Total" is usually correct
- Common amount formats: 1,234.56 or 1234.56 or Rs.1234 or ₹1234
- If the image is unclear, make your best estimate
- Even if text is partially visible, try to extract what you can see

RESPONSE FORMAT (must be valid JSON only, no other text):
{
  "category": "materials",
  "amount": 15000,
  "merchant": "ABC Hardware Store",
  "notes": "10 bags cement, 1 ton sand"
}

Categories (choose one):
- materials: cement, sand, bricks, steel, paint, hardware, building supplies
- labour: wages, worker payments, contractor fees
- transport: diesel, fuel, vehicle expenses, delivery charges
- food: restaurant, meals, groceries
- general: anything else

If you cannot read the image clearly, still try to provide reasonable estimates based on visible numbers and text."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,  # Lower temperature for more consistent extraction
            max_tokens=1000
        )

        content = res.choices[0].message.content.strip()
        print(f"📄 LLM Vision Raw Response:\n{content}\n")
       
        # Remove markdown code blocks if present
        content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = content.strip()
       
        # Try to extract JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if match:
            json_str = match.group()
            print(f"📋 Extracted JSON string:\n{json_str}\n")
           
            parsed = json.loads(json_str)
            print(f"✅ Parsed JSON successfully:\n{json.dumps(parsed, indent=2)}\n")
           
            # Validate required fields
            if all(k in parsed for k in ["category", "amount", "merchant", "notes"]):
                # Ensure amount is a number and positive
                try:
                    amount = float(parsed["amount"])
                    if amount <= 0:
                        print(f"⚠️  Amount is zero or negative: {amount}")
                        return create_fallback_expense(image_url, "Zero amount extracted")
                   
                    parsed["amount"] = amount
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Invalid amount format: {parsed['amount']}, error: {e}")
                    return create_fallback_expense(image_url, f"Invalid amount: {parsed['amount']}")
               
                # Validate category
                valid_categories = ["materials", "labour", "transport", "food", "general"]
                if parsed["category"] not in valid_categories:
                    print(f"⚠️  Invalid category: {parsed['category']}, defaulting to 'general'")
                    parsed["category"] = "general"
               
                print(f"✅ Validation passed - returning parsed data")
                return parsed
            else:
                missing = [k for k in ["category", "amount", "merchant", "notes"] if k not in parsed]
                print(f"⚠️  Missing required fields: {missing}")
        else:
            print(f"⚠️  No JSON object found in response")

    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"   Content was: {content}")
    except Exception as e:
        print(f"❌ LLM vision extraction error: {e}")
        import traceback
        traceback.print_exc()

    # Fallback with warning
    return create_fallback_expense(image_url, "Unable to extract")

def create_fallback_expense(image_url, reason):
    """Create a fallback expense entry when extraction fails."""
    print(f"⚠️  Creating fallback expense entry. Reason: {reason}")
    return {
        "category": "general",
        "amount": 100,  # Default small amount instead of 0
        "merchant": "Manual review needed",
        "notes": f"IMAGE NEEDS REVIEW: {reason}. URL: {image_url}"
    }

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
       
        print(f"✓ Saved text expense ID: {expense_id}")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"✗ Error saving text expense: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Save Expense from File
# -------------------------------
def save_expense_from_file(parsed_data, file_url):
    """Save expense extracted from file to expense table."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        print(f"💾 Attempting to save expense to database:")
        print(f"   Amount: ₹{parsed_data['amount']}")
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
            f"Extracted from image: {file_url}\nNotes: {parsed_data['notes']}"
        ))
       
        expense_id = cur.fetchone()[0]
        conn.commit()
       
        print(f"✅ Successfully saved expense to database, ID: {expense_id}")
        return expense_id
       
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving expense from file: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Download and Upload Media
# -------------------------------
def process_media(message_sid, phone, raw_text):
    """
    Complete media processing workflow:
    1. Download media from Twilio
    2. Upload to Supabase Storage (expense-files bucket)
    3. Save to expense_file_upload table with URL and content metadata (JSONB)
    4. Use LLM vision to extract expense data from images
    5. Save extracted expense data to expense table
    6. Update expense_file_upload with merchant info
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 Starting media processing for message: {message_sid}")
        print(f"{'='*60}")
       
        twilio_client = get_twilio()
        supabase = get_supabase()
       
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
       
        # Get media list
        media_list = twilio_client.messages(message_sid).media.list()
       
        if not media_list:
            print("❌ No media found")
            return
       
        print(f"✅ Found {len(media_list)} media file(s)")
       
        conn = get_db_connection()
        cur = conn.cursor()
       
        for idx, media in enumerate(media_list, 1):
            print(f"\n{'─'*60}")
            print(f"📄 Processing file {idx}/{len(media_list)}")
            print(f"{'─'*60}")
           
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
           
            print(f"✅ Downloaded from Twilio ({len(response.content)} bytes)")
           
            # Determine file extension and content type
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
           
            print(f"✅ Uploaded to storage: {storage_path}")
            print(f"✅ Public URL: {file_url}")
           
            # Prepare JSONB content metadata
            content_metadata = {
                "content_type": content_type,
                "file_size": len(response.content),
                "filename": filename,
                "storage_path": storage_path,
                "twilio_media_sid": media.sid,
                "uploaded_at": datetime.datetime.now().isoformat()
            }
           
            # STEP 1: Save to expense_file_upload table with JSONB content
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
               
                print(f"✅ Saved to expense_file_upload table, ID: {file_upload_id}")
               
            except Exception as e:
                conn.rollback()
                print(f"❌ Error saving to expense_file_upload: {e}")
                import traceback
                traceback.print_exc()
                continue
           
            # STEP 2: Process image files with LLM Vision
            if content_type.startswith("image/"):
                print(f"\n🔍 Image detected - starting LLM vision analysis...")
               
                parsed_expense = extract_expense_from_image(file_url)
               
                print(f"\n📊 Extracted expense data:")
                print(f"   Category: {parsed_expense['category']}")
                print(f"   Amount: ₹{parsed_expense['amount']:,.2f}")
                print(f"   Merchant: {parsed_expense['merchant']}")
                print(f"   Notes: {parsed_expense['notes']}")
               
                # STEP 3: Save to expense table (always save, even with fallback values)
                print(f"\n💾 Saving to expense table...")
                expense_id = save_expense_from_file(parsed_expense, file_url)
               
                if expense_id:
                    print(f"✅ Successfully created expense record ID: {expense_id}")
                   
                    # STEP 4: Update expense_file_upload with extracted merchant and LLM data
                    try:
                        content_metadata["llm_extraction"] = {
                            "extracted_at": datetime.datetime.now().isoformat(),
                            "model": "llama-3.3-70b-versatile",
                            "expense_id": expense_id,
                            "extracted_data": parsed_expense,
                            "extraction_success": parsed_expense["amount"] > 100  # Mark if likely successful
                        }
                       
                        cur.execute("""
                            UPDATE expense_file_upload
                            SET merchant = %s,
                                content = %s::jsonb
                            WHERE id = %s
                        """, (
                            parsed_expense["merchant"],
                            json.dumps(content_metadata),
                            file_upload_id
                        ))
                        conn.commit()
                        print(f"✅ Updated expense_file_upload with LLM extraction data")
                       
                    except Exception as e:
                        conn.rollback()
                        print(f"❌ Error updating expense_file_upload: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ Failed to save expense to database")
                   
            else:
                print(f"⚠️  Non-image file ({content_type}), skipping LLM vision processing")
       
        cur.close()
        conn.close()
       
        print(f"\n{'='*60}")
        print(f"✅ Media processing completed successfully")
        print(f"{'='*60}\n")
       
    except Exception as e:
        print(f"\n❌ Media processing error: {e}")
        import traceback
        traceback.print_exc()

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
        ORDER BY SUM(amount) DESC
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
    plt.title("Daily Expenses by Category", fontsize=14, fontweight='bold')
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Amount (₹)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
   
    # Add value labels on bars
    for i, v in enumerate(amounts):
        plt.text(i, v, f'₹{v:,.0f}', ha='center', va='bottom')
   
    plt.tight_layout()

    path = "/tmp/chart.png"
    plt.savefig(path, dpi=300)
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
2. Highest spending category and its percentage of total
3. Any anomalies or unusual patterns
4. One actionable suggestion for cost optimization

Keep the response concise and professional (max 150 words).
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    print(f"\n📨 Received from {sender}")
    print(f"   Message: {msg}")
    print(f"   Media files: {num_media}")

    # Case 1: Only text message (no media)
    if num_media == 0 and msg:
        parsed = extract_expense(msg)

        if parsed["category"] == "ignore":
            return Response(
                "<Response><Message>❌ No expense information found. Please specify amount and category.</Message></Response>",
                media_type="application/xml"
            )

        # Save to expense table
        expense_id = save_text_expense(sender, parsed, msg)
       
        if expense_id:
            response_msg = f"✅ Saved ₹{parsed['amount']:,} ({parsed['category']})"
        else:
            response_msg = "❌ Error saving expense. Please try again."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Case 2: Media present (with or without text)
    elif num_media > 0:
        # Process media in background
        background_tasks.add_task(process_media, message_sid, sender, msg)
       
        response_msg = f"✅ Received {num_media} file(s). Analyzing receipt... Check database for results."

        return Response(
            f"<Response><Message>{response_msg}</Message></Response>",
            media_type="application/xml"
        )

    # Case 3: Empty message
    else:
        return Response(
            "<Response><Message>Please send expense details (e.g., '5000 for cement') or attach a receipt image.</Message></Response>",
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

    total = sum(float(x[1]) for x in data)

    return {
        "status": "sent",
        "date": datetime.date.today().isoformat(),
        "categories": len(data),
        "total_expenses": f"₹{total:,.2f}",
        "breakdown": {cat: float(amt) for cat, amt in data}
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
        "version": "3.1",
        "models": {
            "text": "llama-3.1-8b-instant",
            "vision": "llama-3.3-70b-versatile",
            "analysis": "llama-3.3-70b-versatile"
        }
    }

# -------------------------------
# Get Today's Expenses
# -------------------------------
@app.get("/expenses/today")
def get_today_expenses():
    """Get all expenses for today."""
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

# -------------------------------
# Get File Uploads
# -------------------------------
@app.get("/file-uploads/today")
def get_today_file_uploads():
    """Get all file uploads for today with content metadata."""
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

# -------------------------------
# Test Expense Extraction
# -------------------------------
@app.post("/test-extract")
async def test_extract(request: Request):
    """Test endpoint for text expense extraction."""
    data = await request.json()
    text = data.get("text", "")
   
    if not text:
        return {"error": "No text provided"}
   
    result = extract_expense(text)
    return result

# -------------------------------
# Test Image Extraction
# -------------------------------
@app.post("/test-extract-image")
async def test_extract_image(request: Request):
    """Test endpoint for image expense extraction."""
    data = await request.json()
    image_url = data.get("image_url", "")
   
    if not image_url:
        return {"error": "No image_url provided"}
   
    result = extract_expense_from_image(image_url)
    return {
        "status": "success",
        "extracted_data": result,
        "needs_review": result.get("merchant") == "Manual review needed"
    }