#!/usr/bin/env python3
"""Latency benchmark for the deployed endpoint: p50/p90/p95/p99 of round trip, queue and execution.

    python scripts/bench.py -n 200                     # sequential, one request in flight
    python scripts/bench.py -n 200 -c 4                # four concurrent clients
    python scripts/bench.py -n 100 --interval 1.5       # a request every ~1.5 s after the last
    python scripts/bench.py -n 100 --input-file job.json

Warm-up requests (default 3) run first and are excluded, so a cold start doesn't pollute the
numbers; pass --warmup 0 to include it. Round trip is measured on this machine; `delayTime`
(queue + dispatch) and `executionTime` are RunPod's own figures for each job.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".runpod-deploy.json")
DEFAULT_JOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_input.json")


def pct(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=100, help="measured requests")
    p.add_argument("-c", type=int, default=1, help="concurrent clients")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--interval", type=float, default=0.0,
                   help="seconds each client waits between requests (models sparse traffic)")
    p.add_argument("--input-file", default=DEFAULT_JOB)
    p.add_argument("--endpoint")
    p.add_argument("--key-file")
    p.add_argument("--json-out", help="write every sample here")
    args = p.parse_args()

    if args.key_file:
        with open(os.path.expanduser(args.key_file), encoding="utf-8") as f:
            key = f.read().strip()
    else:
        key = os.environ.get("RUNPOD_API_KEY", "").strip() or sys.exit("set RUNPOD_API_KEY or pass --key-file")
    endpoint = args.endpoint or os.environ.get("RUNPOD_ENDPOINT_ID")
    if not endpoint and os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            endpoint = json.load(f)["endpoint_id"]
    if not endpoint:
        sys.exit("no endpoint: pass --endpoint, set RUNPOD_ENDPOINT_ID, or run scripts/deploy.py first")
    with open(args.input_file, encoding="utf-8") as f:
        body = f.read().encode()
    url = "https://api.runpod.ai/v2/%s/runsync" % endpoint
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
               "User-Agent": "laya-runpod/1.0"}

    def one():
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                res = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            res = {"status": "HTTP %d" % e.code}
        except Exception as e:  # noqa: BLE001 -- a benchmark counts failures, it doesn't stop on them
            res = {"status": type(e).__name__}
        return {"rtt_ms": (time.perf_counter() - t0) * 1000, "status": res.get("status"),
                "delay_ms": res.get("delayTime"), "exec_ms": res.get("executionTime"),
                "worker": res.get("workerId")}

    for i in range(args.warmup):
        s = one()
        print("warmup %d: %s %.0f ms" % (i + 1, s["status"], s["rtt_ms"]), file=sys.stderr)

    samples, lock, remaining = [], threading.Lock(), [args.n]

    def client():
        while True:
            with lock:
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
            s = one()
            if args.interval:
                time.sleep(args.interval)
            with lock:
                samples.append(s)
                if len(samples) % 25 == 0:
                    print("  %d/%d" % (len(samples), args.n), file=sys.stderr)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(args.c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [s for s in samples if s["status"] == "COMPLETED"]
    print("\n%d requests, %d concurrent, %d ok, %.1f req/s, workers used: %d"
          % (len(samples), args.c, len(ok), len(samples) / wall, len({s["worker"] for s in ok})))
    print("%-22s %8s %8s %8s %8s %8s %8s" % ("ms", "min", "p50", "p90", "p95", "p99", "max"))
    for label, field in (("round trip (client)", "rtt_ms"), ("queue+dispatch (RP)", "delay_ms"),
                         ("execution (RP)", "exec_ms")):
        v = [s[field] for s in ok if s[field] is not None]
        print("%-22s %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f"
              % (label, min(v), pct(v, 50), pct(v, 90), pct(v, 95), pct(v, 99), max(v)))
    failed = [s["status"] for s in samples if s["status"] != "COMPLETED"]
    if failed:
        print("failures:", {f: failed.count(f) for f in set(failed)})
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(samples, f)


if __name__ == "__main__":
    main()
