from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional
from datetime import datetime

class Site(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    config: Optional[dict] = Field(sa_column=Column(JSON), default={})
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Spot(SQLModel, table=True):
    id: str = Field(primary_key=True)
    site_id: str = Field(index=True)
    code: str
    label: Optional[str] = None
    status: str = "free"  # free, occupied, reserved, maintenance
    last_seen_at: Optional[datetime] = None

class Vehicle(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    plate: str
    source: str = "anpr"

class ParkingSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    site_id: str
    spot_id: str
    vehicle_id: Optional[int] = None
    in_at: datetime = Field(default_factory=datetime.utcnow)
    out_at: Optional[datetime] = None
    status: str = "active"  # active / completed
    billed_amount: Optional[int] = None

class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    site_id: str
    payload: dict = Field(sa_column=Column(JSON))
    event_time: datetime = Field(default_factory=datetime.utcnow)

class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    payment_id: str
    session_id: Optional[int]
    provider: str
    amount: int
    status: str
    provider_payload: Optional[dict] = Field(sa_column=Column(JSON))

class Invoice(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    site_id: str
    period_from: Optional[datetime] = None
    period_to: Optional[datetime] = None
    net_total: Optional[int] = 0
    vat: Optional[int] = 0
    gross_total: Optional[int] = 0  