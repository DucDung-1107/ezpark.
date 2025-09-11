import hmac, hashlib, os, csv, io
from fastapi import HTTPException
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
PAYMENT_WEBHOOK_SECRET = os.environ.get("PAYMENT_WEBHOOK_SECRET", "replace_me")
VAT_RATE = float(os.environ.get("VAT_RATE", "0.10"))
CURRENCY = os.environ.get("CURRENCY", "VND")

def verify_hmac(body: bytes, signature_hex: str):
    computed = hmac.new(PAYMENT_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed, signature_hex):
        raise HTTPException(status_code=400, detail="Invalid signature")
    return True

def compute_invoice_for_session(session, rate_per_30min=5000):
    from math import ceil
    if not session.out_at or not session.in_at:
        return 0,0,0
    duration_minutes = (session.out_at - session.in_at).total_seconds() / 60.0
    slots = ceil(duration_minutes / 30.0)
    net = slots * rate_per_30min
    vat = int(round(net * VAT_RATE))
    gross = net + vat
    return net, vat, gross

def generate_invoice_csv_rows(rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["invoice_id","invoice_date","site_id","session_id","spot_code","plate","in_at","out_at","net_amount","vat_rate","vat_amount","total_amount", "currency"])
    for r in rows:
        writer.writerow([
            r.get("invoice_id"),
            r.get("invoice_date"),
            r.get("site_id"),
            r.get("session_id"),
            r.get("spot_code"),
            r.get("plate"),
            r.get("in_at"),
            r.get("out_at"),
            r.get("net_amount"),
            r.get("vat_rate"),
            r.get("vat_amount"),
            r.get("total_amount"),
            CURRENCY
        ])
    return buf.getvalue()