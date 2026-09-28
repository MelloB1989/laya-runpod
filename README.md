# laya-runpod

Self-hosted [Laya](https://github.com/NandhaKishorM/laya) on RunPod serverless. Laya is a
non-autoregressive decision model: it answers typed questions (`choice`, `score`, `noul`) about a
piece of text in one forward pass. This repo packages it as a RunPod queue worker. Workers scale to
zero when idle and pull an image that already contains all three checkpoints (`english`,
`multilingual`, `typed-decisions`), so a cold start never downloads weights.

```
src/handler.py          RunPod handler: predict / batch / route / health
Dockerfile              python:3.11-slim + torch 2.11 cu128 + laya 0.3.21 + baked checkpoints
scripts/bake_models.py  image-build step that downloads and test-loads the checkpoints
scripts/deploy.py       create/update the RunPod template + endpoint over the REST API
scripts/invoke.py       send a job to the endpoint from the terminal
scripts/bench.py        latency percentiles (round trip, queue, execution) against the endpoint
tests/                  handler contract tests (fake router, no weights needed)
.github/workflows/      test, then build and push ghcr.io/<owner>/<repo>
```

## API

Each job's `input` uses the same shape as the body of `laya-serve`'s `POST /v1/systemone`, and a
predict job's output is exactly what `Router.predict` returns (the Jev-compatible `answers` /
`usage` / `routing` object).

```jsonc
// predict: your own questions
{"input": {"state": "I was charged twice, refund me",
           "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                   "criteria": {"billing": "refunds", "tech": "bugs", "other": "else"}}}}}

// predict: a built-in preset (triage, email, guard, moderation, router). A plain-string state is
// placed under the field the preset reads (message, body, prompt, post, request).
{"input": {"state": "My payment failed twice", "preset": "triage"}}

// batch: requests share forward passes. Output: {"results": [...]}, in input order.
{"input": {"requests": [{"state": "...", "preset": "guard"}, {"state": "...", "questions": {...}}],
           "batch_size": 32, "sort_by_length": true}}

// route only: which checkpoint would answer, and why. No forward pass.
{"input": {"action": "route", "state": "Mein Konto wurde zweimal belastet"}}

// worker health: loaded checkpoints, revisions, device, GPU name, versions
{"input": {"action": "health"}}
```

Optional fields on a predict request, or on each item in a batch: `model` (`english`,
`multilingual`, `typed-decisions`, an alias, or a HF id; unknown values such as a Jev model id mean
"let the router choose"), `max_len`, `head_max_len` (e.g. `max_len: 8192` for long documents on
`multilingual`), `lang`, and `min_confidence`. `min_confidence` can also be set once for a whole
batch.

The limits are laya-serve's own: 64 questions, a 50,000-character state, 100 options per choice, 32
score levels, and 512 options across all questions. A batch accepts up to 256 requests
(`LAYA_MAX_BATCH`). An invalid job returns `{"error": "<status>: <detail>"}`, using laya-serve's
status codes (400 malformed, 413 over a limit, 422 rejected by Laya, 500 inference failure with the
traceback in the worker log). RunPod reports these jobs as `FAILED`.

## Build and publish the image

RunPod pulls the image from a registry. Pick one of these routes:

**GitHub Actions → GHCR (no local GPU or registry login needed).** Push this repo to GitHub. The
`image` workflow runs the tests, builds the CUDA image, and pushes `ghcr.io/<owner>/<repo>:latest`
plus a `sha-<short>` tag. If the repo is public, the package is public too and RunPod can pull it
anonymously. If the repo is private, create a RunPod registry auth with a PAT that has
`read:packages` and pass `--registry-auth-id` to `deploy.py`.

**Local build and push:**

```bash
docker build -t <registry>/laya-runpod:0.3.21 .
docker push <registry>/laya-runpod:0.3.21
```

Build arguments: `TORCH_INDEX` (`cu128`, or `cpu` for a local smoke test), `TORCH_VERSION`,
`LAYA_VERSION`, and `LAYA_BAKE_MODELS` (a comma list; leave it empty to download checkpoints on
cold start instead, and then also set `HF_HUB_OFFLINE=0` in the endpoint env).

### Local smoke test (no GPU)

```bash
docker build --build-arg TORCH_INDEX=cpu -t laya-runpod:cpu .
docker run --rm -e LAYA_DEVICE=cpu laya-runpod:cpu \
  python -u handler.py --test_input "$(cat test_input.json)"
# or serve RunPod's local API on :8000 and POST /runsync:
docker run --rm -p 8000:8000 -e LAYA_DEVICE=cpu laya-runpod:cpu \
  python -u handler.py --rp_serve_api --rp_api_host 0.0.0.0
```

## Deploy

```bash
export RUNPOD_API_KEY=...        # or pass --key-file to either script
python scripts/deploy.py --image ghcr.io/<owner>/laya-runpod:latest
python scripts/invoke.py --health
python scripts/invoke.py "Hi, we were billed twice for March, refund it or we cancel" --preset triage
python scripts/invoke.py --input-file test_input.json
```

`deploy.py` matches the template and endpoint by `--name` (default `laya`), so re-running it with a
new `--image` updates them in place. Defaults: 0 min / 2 max workers, a 30 s idle timeout,
FlashBoot on, 16-24 GB GPUs (all three checkpoints need about 3 GB of VRAM), and host CUDA ≥ 12.8
to match the cu128 wheels. Worker env overrides go through `--env KEY=VALUE`, e.g.
`--env LAYA_MODELS=english,multilingual` to load only two checkpoints, or
`--env LAYA_CUDA_AMP=fp16`.

Latency measured on the first deployment (RTX 4090 worker, all three checkpoints resident):

| | end to end |
|---|---|
| warm worker | ~1 s round trip, 75-400 ms execution |
| cold start, image already on the host (FlashBoot) | ~18 s |
| first job on a host that has never pulled the image | ~8 min (a one-time ~7 GB pull) |

Set `--workers-min 1` if the 18 s cold start matters. That worker is billed while idle.

Calling the endpoint directly:

```bash
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' \
  -d @test_input.json
```

## Worker environment

| env | default | meaning |
|---|---|---|
| `LAYA_DEVICE` | `cuda` | torch device; laya falls back to CPU if the GPU is unusable (`health` shows the real device) |
| `LAYA_MODELS` | all | checkpoints to preload at boot |
| `LAYA_PRELOAD` | `1` | load checkpoints at worker start rather than on first use |
| `LAYA_WARMUP` | `1` | one tiny prediction per checkpoint at boot, so the first job doesn't pay CUDA init |
| `LAYA_AUTO_TASK` | `0` | auto-route typed-decisions workflows to that checkpoint |
| `LAYA_MAX_TOKEN_BUDGET` | `8192` | cap on per-request `max_len` / `head_max_len` |
| `LAYA_MAX_BATCH` | `256` | cap on `requests` per batch job |
| `LAYA_CONCURRENCY` | `4` | jobs a worker holds at once; they still run one at a time on the GPU, but the worker fetches the next job while the current one computes |
| `LAYA_CUDA_AMP` | checkpoint's own | `fp16` or `bf16` autocast |
| `HF_HUB_OFFLINE` | `1` | the baked cache is complete; set to `0` if you bake nothing |

## Tests

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python "laya[serve]==0.3.21" -r requirements.txt pytest
.venv/bin/python -m pytest -q tests
```

Laya is Apache-2.0, © Convai Innovations. The model checkpoints come from
[huggingface.co/convaiinnovations](https://huggingface.co/convaiinnovations).
