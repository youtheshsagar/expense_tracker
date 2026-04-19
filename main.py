from fastapi import FastAPI, Request
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

# -------------------------------
# Load ENV
# -------------------------------
load_dotenv()

app = FastAPI()

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
    if "cement" in raw or "sand" in raw:
        category = "materials"
    elif "labour" in raw:
        category = "labour"
    elif "diesel" in raw:
        category = "transport"

    if amount == 0:
        return {"category": "ignore", "amount": 0, "notes": text}

    return {"category": category, "amount": amount, "notes": text}

# -------------------------------
# LLM Extraction
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return only JSON"},
                {"role": "user", "content": f"""
Extract expense:
{text}

Return:
{{"category":"", "amount":number, "notes":""}}
"""}
            ],
            temperature=0
        )

        match = re.search(r"\{.*\}", res.choices[0].message.content)
        if match:
            return json.loads(match.group())

    except:
        pass

    return extract_expense_simple(text)

# -------------------------------
# Save to DB
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
# Query today's data
# -------------------------------
def get_today_summary():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    cur.execute("""
        SELECT category, SUM(amount)
        FROM expenses
        WHERE DATE(recorded_at) = CURRENT_DATE
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
    categories = [x[0] for x in data]
    amounts = [float(x[1]) for x in data]

    plt.figure()
    plt.bar(categories, amounts)
    plt.title("Daily Expenses")

    path = "/tmp/chart.png"
    plt.savefig(path)
    plt.close()

    return path

# -------------------------------
# LLM Analysis
# -------------------------------
def analyze_with_llm(data):
    client = get_groq()

    prompt = f"""
You are a financial analyst.

Data:
{data}

Tasks:
1. Total spend
2. Highest category
3. Detect anomaly
4. Give 1 suggestion

Keep it short.
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
    resend.api_key = os.getenv("RESEND_API_KEY")

    with open(chart_path, "rb") as f:
        chart = f.read()

    resend.Emails.send({
        "from": "Expense Tracker <onboarding@resend.dev>",
        "to": [os.getenv("EMAIL_TO")],
        "subject": "Daily Expense Report",
        "html": f"<pre>{report}</pre>",
        "attachments": [{
            "filename": "chart.png",
            "content": chart
        }]
    })

# -------------------------------
# Webhook
# -------------------------------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.form()

    msg = data.get("Body")
    sender = data.get("From")

    parsed = extract_expense(msg)

    if parsed["category"] == "ignore":
        return Response("<Response><Message>Ignored</Message></Response>", media_type="application/xml")

    save_to_db([
        datetime.datetime.now(),
        sender,
        parsed["category"],
        parsed["amount"],
        parsed["notes"],
        msg
    ])

    return Response(
        f"<Response><Message>Saved ₹{parsed['amount']}</Message></Response>",
        media_type="application/xml"
    )

# -------------------------------
# Daily Report Endpoint
# -------------------------------
@app.get("/send-daily-report")
def send_daily():
    data = get_today_summary()

    if not data:
        return {"status": "no data"}

    chart = generate_chart(data)
    report = analyze_with_llm(data)

    send_email(report, chart)

    return {"status": "sent"}
