import json, os
from sqlmodel import SQLModel
from .database import engine
from .models import Site, Spot
from datetime import datetime

def load_spots_config():
    # Try multiple locations (mounted by docker-compose to /spots.json)
    candidates = [
        "/spots.json",
        os.path.join(os.path.dirname(__file__), "..", "spots.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "spots.json"),
        "./spots.json"
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                print("Loaded spots.json from:", p)
                return doc
        except Exception as e:
            print("Error loading spots.json from", p, ":", e)
    print("No spots.json found in candidates:", candidates)
    return None

def seed():
    SQLModel.metadata.create_all(engine)
    doc = load_spots_config()
    if not doc:
        print("No spots config to seed.")
        return
    site_id = doc.get("site_id", "site_demo_1")
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