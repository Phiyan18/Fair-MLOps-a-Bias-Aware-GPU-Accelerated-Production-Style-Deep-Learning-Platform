"""Tiny dependency-free load generator (use it to watch the Kubernetes autoscaler react).

    python scripts/load_test.py --url http://localhost:8000 --concurrency 32 --seconds 60
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--seconds", type=int, default=30)
ap.add_argument("--batch", type=int, default=50, help="rows per /predict request")
ap.add_argument("--api-key", default="", help="value for the X-API-Key header (if the server requires it)")
args = ap.parse_args()

base = json.loads((Path(__file__).parent.parent / "examples" / "predict_request.json").read_text())["instances"][0]
payload = json.dumps({"instances": [base] * args.batch}).encode()
latencies, errors = [], 0
lock = threading.Lock()
stop = time.time() + args.seconds


def worker():
    global errors
    while time.time() < stop:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(f"{args.url}/predict", data=payload,
                                         headers={"Content-Type": "application/json",
                                                  **({"X-API-Key": args.api_key} if args.api_key else {})})
            urllib.request.urlopen(req, timeout=10).read()
            with lock:
                latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:  # noqa: BLE001
            with lock:
                errors += 1


threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
[t.start() for t in threads]
[t.join() for t in threads]

if latencies:
    latencies.sort()
    n = len(latencies)
    print(f"requests: {n}  errors: {errors}  rps: {n / args.seconds:.1f}  rows/s: {n * args.batch / args.seconds:.0f}")
    print(f"latency ms  p50={statistics.median(latencies):.1f}  p95={latencies[int(n * .95) - 1]:.1f}  "
          f"p99={latencies[int(n * .99) - 1]:.1f}")
else:
    print("no successful requests - is the server running?")
