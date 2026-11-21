"""Demonstrate drift monitoring.  Needs the API running (and API_KEY if you set one).

    python scripts/simulate_drift.py                  # normal traffic -> /drift says "ok"
    python scripts/simulate_drift.py --shift          # drifted traffic -> PSI alarm
"""
import argparse
import json
import random
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--shift", action="store_true", help="send drifted traffic (older, longer hours, new job)")
ap.add_argument("--api-key", default="")
a = ap.parse_args()
random.seed(1)


def call(path, body=None):
    headers = {"Content-Type": "application/json"}
    if a.api_key:
        headers["X-API-Key"] = a.api_key
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode() if body else None, headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=15))  # noqa: S310 (user-supplied local URL)


info = call("/model-info")
cats = info["categories"]


def person():
    p = {
        "age": min(90, max(17, int(random.gauss(39, 13)))), "education_num": min(16, max(1, int(random.gauss(10, 2.5)))),
        "capital_gain": 0 if random.random() > 0.08 else random.randint(500, 20000), "capital_loss": 0,
        "hours_per_week": min(99, max(1, int(random.gauss(40, 11)))),
        **{c: random.choice(v) for c, v in cats.items()},
    }
    if a.shift:
        p["age"] = min(100, p["age"] + 25)
        p["hours_per_week"] = min(99, p["hours_per_week"] + 15)
    return p


sent = 0
while sent < a.n:
    batch = [person() for _ in range(25)]
    call("/predict", {"instances": batch})
    sent += len(batch)
print(f"sent {sent} {'DRIFTED' if a.shift else 'normal'} requests")
rep = call("/drift")
print(f"status={rep['status']} max_psi={rep['max_psi']} window={rep['window_size']}")
for f, v in sorted(rep["features"].items(), key=lambda kv: -kv[1]["psi"])[:5]:
    print(f"  {f:<16} PSI={v['psi']:<8} {v['status']}")
