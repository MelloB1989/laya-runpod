#!/usr/bin/env python3
"""Send one job to the deployed Laya endpoint and print the result.

    python scripts/invoke.py "I was charged twice, refund me" --preset triage
    python scripts/invoke.py --input-file test_input.json
    python scripts/invoke.py "Mein Konto wurde zweimal belastet" --route
    python scripts/invoke.py --health

The endpoint id comes from --endpoint, RUNPOD_ENDPOINT_ID, or .runpod-deploy.json (written by
scripts/deploy.py); the key from RUNPOD_API_KEY or --key-file. Standard library only.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".runpod-deploy.json")
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def call(key, method, url, body=None, timeout=120):
    req = urllib.request.Request(url, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit("%s %s -> HTTP %d: %s" % (method, url, e.code, e.read().decode(errors="replace")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("state", nargs="?", help="text to decide on")
    p.add_argument("--preset", default="triage", help="question preset (triage, email, guard, moderation, router)")
    p.add_argument("--questions", help="JSON file holding a questions object; replaces --preset")
    p.add_argument("--model", help="pin a checkpoint: english, multilingual, typed-decisions")
    p.add_argument("--route", action="store_true", help="routing decision only")
    p.add_argument("--health", action="store_true", help="worker health (loaded checkpoints, GPU)")
    p.add_argument("--endpoint-health", action="store_true", help="RunPod queue/worker counts, no job")
    p.add_argument("--input-file", help="JSON file with a full job ({\"input\": {...}})")
    p.add_argument("--endpoint", help="endpoint id")
    p.add_argument("--key-file", help="read the RunPod API key from this file")
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
    base = "https://api.runpod.ai/v2/%s" % endpoint

    if args.endpoint_health:
        print(json.dumps(call(key, "GET", base + "/health"), indent=2))
        return
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            job = json.load(f)
    elif args.health:
        job = {"input": {"action": "health"}}
    else:
        if not args.state:
            p.error("give a state, --input-file, --health or --endpoint-health")
        inp = {"state": args.state}
        if args.route:
            inp["action"] = "route"
        elif args.questions:
            with open(args.questions, encoding="utf-8") as f:
                inp["questions"] = json.load(f)
        else:
            inp["preset"] = args.preset
        if args.model:
            inp["model"] = args.model
        job = {"input": inp}

    t0 = time.time()
    res = call(key, "POST", base + "/runsync", job)
    # runsync hands back a job id instead of the output when the job outlives its wait window
    # (cold start included); poll until it settles.
    while res.get("status") not in DONE:
        time.sleep(2)
        res = call(key, "GET", "%s/status/%s" % (base, res["id"]))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("# %s in %.1fs (queue %s ms, execution %s ms)"
          % (res["status"], time.time() - t0, res.get("delayTime"), res.get("executionTime")), file=sys.stderr)
    sys.exit(0 if res["status"] == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
