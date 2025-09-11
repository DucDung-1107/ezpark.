from flask import Flask, request, render_template_string, redirect
import os, hmac, hashlib, requests
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
BACKEND = os.environ.get("BACKEND_URL", "http://backend:8000")
PAYMENT_WEBHOOK_SECRET = os.environ.get("PAYMENT_WEBHOOK_SECRET", "replace_me")

HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Payment Sandbox</title></head>
<body>
  <h1>Payment Sandbox</h1>
  <p>Payment_id: {{payment_id}}, session: {{session_id}}, amount: {{amount}}</p>
  <button onclick="pay('succeeded')">Pay (Success)</button>
  <button onclick="pay('failed')">Simulate Fail</button>
  <script>
    function pay(status){
      fetch('/do_pay', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({payment_id:'{{payment_id}}', session_id:{{session_id}}, status:status, amount:{{amount}}})})
      .then(()=>{ alert('Webhook sent: '+status); window.location.href='/';});
    }
  </script>
</body></html>
"""

@app.route('/pay')
def pay():
    payment_id = request.args.get('payment_id')
    session_id = int(request.args.get('session_id', '0'))
    amount = int(request.args.get('amount', '0'))
    return render_template_string(HTML, payment_id=payment_id, session_id=session_id, amount=amount)

@app.route('/do_pay', methods=['POST'])
def do_pay():
    data = request.json
    payload = {
        "payment_id": data.get("payment_id"),
        "session_id": data.get("session_id"),
        "status": data.get("status"),
        "amount": data.get("amount"),
        "provider_payload": {"demo": True}
    }
    body = json_bytes = bytes(str(payload), 'utf-8')
    signature = hmac.new(PAYMENT_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    headers = {"X-Signature": signature, "Content-Type": "application/json"}
    # send webhook to backend (use internal service name)
    try:
        r = requests.post(f"{BACKEND}/api/payments/webhook", json=payload, headers=headers, timeout=5)
        return ("ok", 200) if r.status_code==200 else ("backend error", 500)
    except Exception as e:
        return (f"error: {e}", 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)