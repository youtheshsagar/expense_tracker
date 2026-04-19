from fastapi import FastAPI, Request
from fastapi.responses import Response
from groq import Groq
import json
import re
import datetime
import os
from dotenv import load_dotenv
import psycopg2

# Load .env (for local)
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

app = FastAPI()

_groq_client = None


# -------------------------------
# 🧠 Groq Client
# -------------------------------
def get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = Groq(api_key=key)
    return _groq_client


# -------------------------------
# 🧾 Fallback parser
# -------------------------------
def extract_expense_simple(text):
    if not text or not str(text).strip():
        return {"category": "ignore", "amount": 0, "notes": ""}

    raw = str(text).strip()
    low = raw.lower()
    amount = 0.0

    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", low)
    if m:
        amount = float(m.group(1)) * 1000
    else:
        m = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)", low)
        if m:
            amount = float(m.group(1).replace(",", ""))
        else:
            m = re.search(r"\b(\d{4,})\b", low.replace(",", ""))
            if m:
                amount = float(m.group(1))
            else:
                m = re.search(r"\b(\d+(?:\.\d+)?)\b", low)
                if m:
                    amount = float(m.group(1))

    category = "general"
    if any(w in low for w in ("labour", "labor", "mason", "worker")):
        category = "labour"
    elif any(w in low for w in ("cement", "sand", "brick", "steel")):
        category = "materials"
    elif any(w in low for w in ("diesel", "fuel", "truck")):
        category = "transport"

    if amount <= 0:
        return {"category": "ignore", "amount": 0, "notes": raw}

    return {
        "category": category,
        "amount": int(round(amount)),
        "notes": raw
    }


# -------------------------------
# 🧠 LLM extraction
# -------------------------------
def extract_expense(text):
    try:
        client = get_groq()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {
                    "role": "user",
                    "content": f"""
Extract expense from:
{text}

Return:
{{ "category": "", "amount": number, "notes": "" }}

If not expense:
{{ "category": "ignore", "amount": 0, "notes": "" }}
"""
                }
            ],
            temperature=0,
            max_tokens=100
        )

        raw = completion.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception:
        pass

    return extract_expense_simple(text)


# -------------------------------
# 🗄 PostgreSQL Save
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
# 🌐 Webhook endpoints
# -------------------------------
@app.get("/webhook")
async def webhook_probe():
    return Response(
        content="OK — configure this URL in Twilio",
        media_type="text/plain"
    )


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.form()
    message = data.get("Body")
    sender = data.get("From")

    print("📩 Incoming:", message)

    parsed = extract_expense(message)

    if parsed["category"] == "ignore":
        return Response(
            content="<Response><Message>Ignored ❌</Message></Response>",
            media_type="application/xml"
        )

    recorded_at = datetime.datetime.now()

    try:
        save_to_db([
            recorded_at,
            sender,
            parsed["category"],
            parsed["amount"],
            parsed["notes"],
            message
        ])
    except Exception as e:
        print("DB Error:", e)
        return Response(
            content="<Response><Message>DB error ❌</Message></Response>",
            media_type="application/xml"
        )

    print("✅ Saved:", parsed)

    return Response(
        content=f"<Response><Message>Saved ✅ ₹{parsed['amount']}</Message></Response>",
        media_type="application/xml"
    )
