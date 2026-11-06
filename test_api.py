"""Test the FastAPI app for real: start uvicorn, call /health and
POST /predict with three payloads, save the request/response pairs."""
import json
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
OUT = "results/api_test.json"

cases = {
    "likely_to_book": {
        "trip_type": "flight",
        "price_pkr": 28000,
        "days_advance": 45,
        "party_size": 2,
        "trip_duration_days": 4,
        "device": "desktop",
        "channel": "direct",
        "pages_viewed": 18,
        "time_on_site_min": 22.5,
        "returning_visitor": 1,
        "weekend_departure": 0,
    },
    "unlikely_to_book": {
        "trip_type": "umrah",
        "price_pkr": 420000,
        "days_advance": 3,
        "party_size": 5,
        "trip_duration_days": 14,
        "device": "mobile",
        "channel": "social",
        "pages_viewed": 2,
        "time_on_site_min": 0.8,
        "returning_visitor": 0,
        "weekend_departure": 1,
    },
    "edge_case_minimal_engagement": {
        "trip_type": "hotel",
        "price_pkr": 12000,
        "days_advance": 90,
        "party_size": 1,
        "trip_duration_days": 1,
        "device": "mobile",
        "channel": "organic",
        "pages_viewed": 1,
        "time_on_site_min": 0.2,
        "returning_visitor": 0,
        "weekend_departure": 0,
    },
}


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


server = subprocess.Popen(
    [".venv/bin/uvicorn", "api:app", "--port", "8000"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
log = []
try:
    for _ in range(30):
        try:
            status, body = call("GET", "/health")
            break
        except Exception:
            time.sleep(1)
    log.append({"request": "GET /health", "status": status, "response": body})
    print("GET /health:", status, body)
    for name, payload in cases.items():
        status, body = call("POST", "/predict", payload)
        log.append({"request": "POST /predict " + name, "status": status,
                    "payload": payload, "response": body})
        print("POST /predict", name, "->", status, body)
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()

with open(OUT, "w") as fh:
    json.dump(log, fh, indent=2)
print("Saved:", OUT)
