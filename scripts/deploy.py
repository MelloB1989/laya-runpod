#!/usr/bin/env python3
"""Create or update the Laya serverless template + endpoint on RunPod (idempotent, by name).

    export RUNPOD_API_KEY=...            # or --key-file path/to/key
    python scripts/deploy.py --image ghcr.io/<owner>/laya-runpod:latest

Re-running with a new --image rolls the endpoint's workers onto it. Uses only the standard
library and RunPod's REST API (https://rest.runpod.io/v1). The endpoint id is written to
.runpod-deploy.json for scripts/invoke.py.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".runpod-deploy.json")

# 16-24 GB cards: every Laya checkpoint together needs ~3 GB of VRAM, so the cheapest pools win.
DEFAULT_GPUS = [
    "NVIDIA RTX A4000",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4500",
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
]
# The image ships torch cu128 wheels, which need a host driver that reports CUDA >= 12.8.
DEFAULT_CUDA = ["12.8", "12.9", "13.0"]


def api_key(args) -> str:
    if args.key_file:
        with open(os.path.expanduser(args.key_file), encoding="utf-8") as f:
            return f.read().strip()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("set RUNPOD_API_KEY or pass --key-file")
    return key


def call(key: str, method: str, path: str, body=None):
    req = urllib.request.Request(
        API + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit("%s %s -> HTTP %d: %s" % (method, path, e.code, e.read().decode(errors="replace")))
    return json.loads(raw) if raw.strip() else None


def parse_env(pairs):
    env = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            sys.exit("--env expects KEY=VALUE, got %r" % pair)
        env[name] = value
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="container image, e.g. ghcr.io/<owner>/laya-runpod:latest")
    p.add_argument("--name", default="laya", help="template/endpoint name (default: laya)")
    p.add_argument("--key-file", help="read the RunPod API key from this file instead of RUNPOD_API_KEY")
    p.add_argument("--gpu", action="append", help="GPU type id (repeatable); default: 16-24 GB cards")
    p.add_argument("--cuda", action="append", help="allowed host CUDA version (repeatable); default 12.8+")
    p.add_argument("--workers-min", type=int, default=0, help="always-on workers (billed while idle)")
    p.add_argument("--workers-max", type=int, default=2)
    p.add_argument("--idle-timeout", type=int, default=30, help="seconds a worker stays up after its last job")
    p.add_argument("--execution-timeout", type=int, default=300, help="per-job limit in seconds")
    p.add_argument("--scaler-value", type=int, default=4, help="queue delay (s) before scaling up")
    p.add_argument("--container-disk", type=int, default=20, help="GB")
    p.add_argument("--no-flashboot", action="store_true")
    p.add_argument("--registry-auth-id", help="RunPod container registry auth id, for a private image")
    p.add_argument("--env", action="append", metavar="KEY=VALUE", help="extra worker env (repeatable)")
    args = p.parse_args()
    key = api_key(args)

    env = {"LAYA_DEVICE": "cuda"}
    env.update(parse_env(args.env))
    template_body = {
        "name": args.name,
        "imageName": args.image,
        "isServerless": True,
        "category": "NVIDIA",
        "containerDiskInGb": args.container_disk,
        "env": env,
        "ports": [],
        "readme": "Laya decision engine (https://github.com/NandhaKishorM/laya) as a RunPod queue worker.",
    }
    if args.registry_auth_id:
        template_body["containerRegistryAuthId"] = args.registry_auth_id

    template = next((t for t in call(key, "GET", "/templates") or [] if t.get("name") == args.name), None)
    if template:
        # isServerless and category are fixed at creation; the update route refuses them.
        update = {k: v for k, v in template_body.items() if k not in ("isServerless", "category")}
        template = call(key, "POST", "/templates/%s/update" % template["id"], update)
        print("updated template %s (%s)" % (template["id"], args.image))
    else:
        template = call(key, "POST", "/templates", template_body)
        print("created template %s (%s)" % (template["id"], args.image))

    endpoint_body = {
        "name": args.name,
        "templateId": template["id"],
        "computeType": "GPU",
        "gpuTypeIds": args.gpu or DEFAULT_GPUS,
        "gpuCount": 1,
        "allowedCudaVersions": args.cuda or DEFAULT_CUDA,
        "workersMin": args.workers_min,
        "workersMax": args.workers_max,
        "idleTimeout": args.idle_timeout,
        "executionTimeoutMs": args.execution_timeout * 1000,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": args.scaler_value,
        "flashboot": not args.no_flashboot,
    }
    endpoint = next((e for e in call(key, "GET", "/endpoints") or [] if e.get("name") == args.name), None)
    if endpoint:
        update = {k: v for k, v in endpoint_body.items() if k != "computeType"}
        endpoint = call(key, "POST", "/endpoints/%s/update" % endpoint["id"], update)
        print("updated endpoint %s" % endpoint["id"])
    else:
        endpoint = call(key, "POST", "/endpoints", endpoint_body)
        print("created endpoint %s" % endpoint["id"])

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"endpoint_id": endpoint["id"], "template_id": template["id"], "image": args.image}, f, indent=2)
    base = "https://api.runpod.ai/v2/%s" % endpoint["id"]
    print("\n  sync:   POST %s/runsync\n  async:  POST %s/run  ->  GET %s/status/<job id>\n  health: GET  %s/health"
          % (base, base, base, base))


if __name__ == "__main__":
    main()
