from pydantic import BaseModel
from typing import Optional

class SlotUpdatePayload(BaseModel):
    event_id: str
    site_id: str
    spot_id: str
    status: str  # "occupied" or "free"
    image_url: Optional[str] = None
    plate: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None

class PaymentWebhookPayload(BaseModel):
    payment_id: str
    session_id: int
    status: str
    amount: int
    provider_payload: Optional[dict] = None