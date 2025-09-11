from fastapi import FastAPI, Request, Header, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import os, json, uuid
from .database import engine, get_session
from .models import *
from .schemas import SlotUpdatePayload, PaymentWebhookPayload
from .utils import verify_hmac, compute_invoice_for_session, generate_invoice_csv_rows
from sqlmodel import Session, select, SQLModel
from sqlalchemy.exc import OperationalError
from datetime import datetime
from dotenv import load_dotenv
import time
load_dotenv()

app = FastAPI(title="ParkWave Demo Backend")

def wait_for_db(timeout: int = 60, interval: float = 1.0):
    """
    Chờ Postgres sẵn sàng trước khi gọi create_all.
    timeout: giây tối đa chờ
    """
    start = time.time()
    while True:
        try:
            # thử kết nối nhanh
            with engine.connect():
                print("Database is ready.")
                return
        except Exception as e:
            elapsed = time.time() - start
            if elapsed >= timeout:
                raise RuntimeError(f"Database not ready after {timeout}s") from e
            print(f"Database not ready yet ({int(elapsed)}s elapsed). Retrying in {interval}s...")
            time.sleep(interval)

# trong event startup, dùng wait_for_db trước khi create_all / seed
@app.on_event("startup")
def on_startup():
    # chờ DB sẵn sàng (tối đa 60s)
    wait_for_db(timeout=60)
    # tạo schema + seed (nếu cần)
    SQLModel.metadata.create_all(engine)
    from .seed import seed
    seed()

# serve admin page
@app.get("/", response_class=HTMLResponse)
def admin_page():
    p = os.path.join(os.path.dirname(__file__), "static", "admin.html")
    return open(p, "r", encoding="utf-8").read()

# kiosk page
@app.get("/site/{site_id}", response_class=HTMLResponse)
def kiosk_page(site_id: str):
    p = os.path.join(os.path.dirname(__file__), "static", "kiosk.html")
    return open(p, "r", encoding="utf-8").read().replace("{{SITE_ID}}", site_id)

# get spots
@app.get("/api/spots")
def get_spots(site_id: str = None):
    with Session(engine) as s:
        q = select(Spot)
        if site_id:
            q = q.where(Spot.site_id == site_id)
        spots = s.exec(q).all()
        return {"spots":[sp.dict() for sp in spots]}

# events endpoint from cv-worker
@app.post("/api/events/slot_update")
def slot_update(payload: SlotUpdatePayload):
    with Session(engine) as s:
        # record event
        ev = Event(event_type="slot_update", site_id=payload.site_id, payload=payload.dict())
        s.add(ev)
        # update spot status
        spot = s.get(Spot, payload.spot_id)
        if not spot:
            # create spot if not exist
            spot = Spot(id=payload.spot_id, site_id=payload.site_id, code=payload.spot_id)
            s.add(spot)
            s.commit()
            s.refresh(spot)
        spot.status = "occupied" if payload.status == "occupied" else "free"
        spot.last_seen_at = datetime.utcnow()
        s.add(spot)
        # if occupied -> create parking_session if none active
        if payload.status == "occupied":
            # optionally create vehicle
            vehicle_id = None
            if payload.plate:
                v = s.exec(select(Vehicle).where(Vehicle.plate == payload.plate)).first()
                if not v:
                    v = Vehicle(plate=payload.plate)
                    s.add(v)
                    s.commit()
                    s.refresh(v)
                vehicle_id = v.id
            # check if active session exists for spot
            active = s.exec(select(ParkingSession).where(ParkingSession.spot_id == spot.id, ParkingSession.status == "active")).first()
            if not active:
                ps = ParkingSession(site_id=payload.site_id, spot_id=spot.id, vehicle_id=vehicle_id)
                s.add(ps)
        else:
            # free -> close active session for this spot
            active = s.exec(select(ParkingSession).where(ParkingSession.spot_id == spot.id, ParkingSession.status == "active")).first()
            if active:
                active.out_at = datetime.utcnow()
                active.status = "completed"
                # compute billed amount
                net, vat, gross = compute_invoice_for_session(active)
                active.billed_amount = gross
        s.commit()
    return {"status":"ok"}

# list sessions
@app.get("/api/sessions")
def list_sessions():
    with Session(engine) as s:
        rows = s.exec(select(ParkingSession)).all()
        return {"sessions":[r.dict() for r in rows]}

# create payment
@app.post("/api/payments/create")
def create_payment(session_id: int, amount: int):
    payment_id = "pay_" + uuid.uuid4().hex[:8]
    payment_url = f"http://localhost:9000/pay?payment_id={payment_id}&session_id={session_id}&amount={amount}"
    with Session(engine) as s:
        p = Payment(payment_id=payment_id, session_id=session_id, provider="sandbox", amount=amount, status="created")
        s.add(p)
        s.commit()
    return {"payment_id":payment_id, "payment_url":payment_url}

# webhook for payment
@app.post("/api/payments/webhook")
async def payment_webhook(request: Request, x_signature: str = Header(None)):
    body = await request.body()
    verify_hmac(body, x_signature)
    data = await request.json()
    payment_id = data.get("payment_id")
    status = data.get("status")
    session_id = data.get("session_id")
    amount = data.get("amount")
    with Session(engine) as s:
        p = s.exec(select(Payment).where(Payment.payment_id == payment_id)).first()
        if not p:
            # create record
            p = Payment(payment_id=payment_id, session_id=session_id, provider="sandbox", amount=amount, status=status, provider_payload=data)
            s.add(p)
        else:
            p.status = status
            p.provider_payload = data
        # if succeeded -> mark session billed and create invoice entry
        if status == "succeeded" and session_id:
            ps = s.get(ParkingSession, session_id)
            if ps:
                # mark billed amount if not already
                if not ps.billed_amount:
                    from .utils import compute_invoice_for_session
                    # make sure out_at is set
                    if not ps.out_at:
                        ps.out_at = datetime.utcnow()
                    net, vat, gross = compute_invoice_for_session(ps)
                    ps.billed_amount = gross
                # create invoice record (simple)
                inv = Invoice(site_id=ps.site_id, period_from=ps.in_at, period_to=ps.out_at, net_total=ps.billed_amount, vat=0, gross_total=ps.billed_amount)
                s.add(inv)
        s.commit()
    return {"status":"ok"}

# export invoices CSV (simple)
@app.get("/admin/invoices")
def export_invoices(site_id: str = None, from_date: str = None, to_date: str = None):
    # build rows from sessions that have billed_amount
    with Session(engine) as s:
        q = select(ParkingSession).where(ParkingSession.billed_amount != None)
        if site_id:
            q = q.where(ParkingSession.site_id == site_id)
        sessions = s.exec(q).all()
        rows = []
        for ps in sessions:
            v = s.get(Vehicle, ps.vehicle_id) if ps.vehicle_id else None
            rows.append({
                "invoice_id": f"inv_{ps.id}",
                "invoice_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "site_id": ps.site_id,
                "session_id": ps.id,
                "spot_code": ps.spot_id,
                "plate": v.plate if v else "",
                "in_at": ps.in_at.isoformat(),
                "out_at": ps.out_at.isoformat() if ps.out_at else "",
                "net_amount": int(ps.billed_amount or 0),
                "vat_rate": float(os.environ.get("VAT_RATE", 0.1)),
                "vat_amount": 0,
                "total_amount": int(ps.billed_amount or 0)
            })
        csv_text = generate_invoice_csv_rows(rows)
        return Response(content=csv_text, media_type="text/csv", headers={"Content-Disposition":"attachment; filename=invoices.csv"})