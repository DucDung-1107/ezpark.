import json, os
from sqlmodel import SQLModel
from .database import engine
from .models import Site, Spot
from datetime import datetime

def seed():
    SQLModel.metadata.create_all(engine)
    # load spots.json
    p = os.path.join(os.path.dirname(__file__), "..", "spots.json")
    if not os.path.exists(p):
        print("spots.json not found")
        return
    doc = json.load(open(p, "r", encoding="utf-8"))
    site_id = doc.get("site_id", "site_demo_1")
    # insert site and spots if not exist
    from sqlmodel import Session, select
    with Session(engine) as s:
        existing = s.exec(select(Site).where(Site.id == site_id)).first()
        if not existing:
            site = Site(id=site_id, name=doc.get("site_name","Demo Site"), config=doc)
            s.add(site)
        for sp in doc.get("spots", []):
            existing_sp = s.exec(select(Spot).where(Spot.id == sp["id"])).first()
            if not existing_sp:
                spot = Spot(id=sp["id"], site_id=site_id, code=sp["code"], label=sp.get("label"))
                s.add(spot)
        s.commit()
    print("Seed completed")

if __name__ == "__main__":
    seed()