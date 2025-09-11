# cv-worker/worker.py (robust version)
import os, time, json, random, requests, sys
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.environ.get("BACKEND_URL", "http://backend:8000")
SITE_ID = os.environ.get("SITE_ID", "site_demo_1")

# Candidate paths to find spots.json robustly
candidate_paths = [
    "/spots.json",
    "/app/../spots.json",
    "/app/spots.json",
    "/usr/src/app/spots.json",
    "./spots.json"
]

cfg = None
for p in candidate_paths:
    try:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print("Loaded spots.json from:", p)
            break
    except Exception as e:
        print("Error reading", p, ":", e)

if cfg is None:
    print("ERROR: spots.json not found. Checked paths:", candidate_paths)
    sys.exit(1)

spots = [s["id"] for s in cfg.get("spots", [])]
if not spots:
    print("No spots found in spots.json - nothing to simulate")
    sys.exit(1)

print("cv-worker running in SIM mode. Spots:", spots)
state = {s: "free" for s in spots}

def send_event(spot_id, status, plate=None):
    payload = {
        "event_id": f"evt_{spot_id}_{int(time.time())}",
        "site_id": SITE_ID,
        "spot_id": spot_id,
        "status": status,
        "plate": plate
    }
    try:
        r = requests.post(f"{BACKEND}/api/events/slot_update", json=payload, timeout=6)
        print("sent event", payload, "->", r.status_code)
    except Exception as e:
        print("error sending event:", e)

# simulate: randomly toggle a spot every 4-8 seconds
plates = ['30A-12345','29B-54321','59C-11111','43A-22222']
while True:
    time.sleep(random.uniform(4,8))
    spot = random.choice(spots)
    # flip state
    new = "occupied" if state[spot] == "free" else "free"
    state[spot] = new
    plate = random.choice(plates) if new == "occupied" and random.random() < 0.7 else None
    send_event(spot, new, plate)