"""RunPod serverless worker for Laya (https://github.com/NandhaKishorM/laya).

One job = one `input` object. The shapes mirror `laya-serve`'s `/v1/systemone` body, so a
request that works against a self-hosted laya-serve works here unchanged:

    {"state": ..., "questions": {...}}                   -> predict (Jev-shaped answers)
    {"state": "...", "preset": "triage"}                 -> predict with a built-in question set
    {"requests": [{...}, {...}], "batch_size": 32}       -> predict_batch, {"results": [...]}
    {"action": "route", "state": ..., "questions": ...}  -> routing decision only, no forward pass
    {"action": "health"}                                 -> loaded checkpoints, device, versions

Validation and model-name resolution are laya-serve's own helpers, so the limits (64 questions,
50k-char state, 100 options per choice, ...) and error texts are identical. Their `HTTPException`s
become `{"error": "<status>: <detail>"}`, which RunPod reports as a FAILED job.

Configuration is laya-serve's environment (LAYA_DEVICE, LAYA_MODELS, LAYA_PRELOAD, LAYA_THREADS,
LAYA_AUTO_TASK, LAYA_MAX_LOADED, LAYA_MAX_TOKEN_BUDGET) plus:

    LAYA_WARMUP       run one tiny prediction per loaded checkpoint at boot (default 1)
    LAYA_MAX_BATCH    cap on `requests` per batch job (default 256)
    LAYA_CONCURRENCY  jobs a worker holds at once (default 4); they still run one at a time
"""
import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from laya import serve as laya_serve
from laya.cli import PRESETS, PRESET_STATE_KEYS

logging.basicConfig(level=os.environ.get("LAYA_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("laya.runpod")

DEFAULT_MAX_BATCH = 256
DEFAULT_CONCURRENCY = 4

# Built once per worker by `init()`, before the first job; tests assign a fake directly.
_router = None


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def init(router: Optional[Any] = None):
    """Build (or inject) the Router and warm it up. Runs at worker start, not per job."""
    global _router
    t0 = time.perf_counter()
    _router = router if router is not None else laya_serve.build_router()
    _log.info("router ready in %.1fs, loaded=%s", time.perf_counter() - t0, _router.loaded)
    if laya_serve._env_bool("LAYA_WARMUP", True):
        _warmup()
    return _router


def _warmup():
    # First CUDA forward pass pays context setup and kernel selection; do it here so the first
    # real job doesn't, and so a broken image fails at boot instead of on every request.
    probe = {"probe": {"type": "noul", "instructions": "Is this a warmup request?"}}
    for name in list(_router.loaded or []):
        t0 = time.perf_counter()
        _router.predict("warmup request", probe, model=name)
        _log.info("warmup %s: %.0f ms", name, (time.perf_counter() - t0) * 1000)


def _state_and_questions(inp: Dict[str, Any]):
    """Resolve `preset` into questions, placing a plain-text state under the field it reads."""
    state = inp.get("state")
    preset = inp.get("preset")
    if preset is None:
        if "questions" not in inp:
            raise HTTPException(status_code=400, detail="input needs a 'questions' object or a 'preset'")
        return state, inp["questions"]
    if "questions" in inp:
        raise HTTPException(status_code=400, detail="pass either 'questions' or 'preset', not both")
    if preset not in PRESETS:
        raise HTTPException(status_code=400,
                            detail="unknown preset %r; choose one of %s" % (preset, sorted(PRESETS)))
    if isinstance(state, str):
        state = {PRESET_STATE_KEYS[preset]: state}
    return state, PRESETS[preset]()


def _min_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=422, detail="min_confidence must be a number between 0 and 1")
    return float(value)


def _prepare(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one request and turn it into Router keyword arguments."""
    if not isinstance(inp, dict):
        raise HTTPException(status_code=400, detail="each request must be an object")
    state, questions = _state_and_questions(inp)
    laya_serve._check_request_limits(state, questions)
    cap = laya_serve._resolve_max_token_budget()
    request = {"state": state, "questions": questions,
               "model": laya_serve._resolve_model(inp.get("model"))}
    for key in ("max_len", "head_max_len"):
        value = laya_serve._validate_budget_param(inp, key, cap)
        if value is not None:
            request[key] = value
    lang = inp.get("lang")
    if lang is not None:
        if not isinstance(lang, str) or not lang.strip():
            raise HTTPException(status_code=422, detail="lang must be a non-empty language code")
        request["lang"] = lang.strip()
    return request


def _predict(inp: Dict[str, Any]) -> Dict[str, Any]:
    request = _prepare(inp)
    return _router.predict(**request, min_confidence=_min_confidence(inp.get("min_confidence")))


def _predict_batch(inp: Dict[str, Any]) -> Dict[str, Any]:
    requests = inp["requests"]
    if not isinstance(requests, list) or not requests:
        raise HTTPException(status_code=400, detail="'requests' must be a non-empty list")
    max_batch = _env_int("LAYA_MAX_BATCH", DEFAULT_MAX_BATCH)
    if len(requests) > max_batch:
        raise HTTPException(status_code=413,
                            detail="too many requests in batch (%d > %d)" % (len(requests), max_batch))
    prepared: List[Dict[str, Any]] = []
    for i, item in enumerate(requests):
        try:
            prepared.append(_prepare(item))
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code, detail="requests[%d]: %s" % (i, e.detail))
    batch_size = inp.get("batch_size")
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                                   or batch_size <= 0):
        raise HTTPException(status_code=422, detail="batch_size must be a positive integer")
    results = _router.predict_batch(
        prepared,
        batch_size=batch_size,
        min_confidence=_min_confidence(inp.get("min_confidence")),
        sort_by_length=bool(inp.get("sort_by_length", False)),
    )
    return {"results": results}


def _route(inp: Dict[str, Any]) -> Dict[str, Any]:
    state = inp.get("state")
    if state is None:
        raise HTTPException(status_code=400, detail="'state' is required")
    questions = inp.get("questions")
    if questions is not None and not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    decision = _router.route(state, questions, model=laya_serve._resolve_model(inp.get("model")),
                             lang=inp.get("lang"))
    return dict(decision)


def _health() -> Dict[str, Any]:
    import laya
    import torch
    from laya.mcp.device import agent_device, resolve_device, router_agent

    devices = {}
    for name in _router.loaded or []:
        device = agent_device(router_agent(_router, name))
        if device:
            devices[name] = device
    return {
        "status": "ok",
        "revision": os.environ.get("LAYA_RUNPOD_REVISION"),
        "concurrency": concurrency(1),
        "laya_version": laya.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded": _router.loaded,
        "revisions": getattr(_router, "loaded_revisions", {}),
        "device": next(iter(devices.values()), None) or resolve_device(),
        "checkpoint_devices": devices,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input")
    try:
        if not isinstance(inp, dict):
            raise HTTPException(status_code=400, detail="job input must be an object")
        action = inp.get("action") or ("batch" if "requests" in inp else "predict")
        if action == "predict":
            return _predict(inp)
        if action == "batch":
            if "requests" not in inp:
                raise HTTPException(status_code=400, detail="batch input needs a 'requests' list")
            return _predict_batch(inp)
        if action == "route":
            return _route(inp)
        if action == "health":
            return _health()
        raise HTTPException(status_code=400, detail="unknown action %r; use predict, batch, route or health"
                            % (action,))
    except HTTPException as e:
        return {"error": "%d: %s" % (e.status_code, e.detail)}
    except ValueError as e:
        # Laya's question validation errors name the question and what to fix: safe to return.
        return {"error": "422: %s" % e}
    except Exception:  # noqa: BLE001 -- same policy as laya-serve: the traceback goes to the log only
        _log.exception("inference failed (job %s)", job.get("id"))
        return {"error": "500: inference failed"}


# One forward pass at a time, off the event loop. A sync handler would run on the SDK's loop, and
# at concurrency 1 the SDK's job fetcher sleeps a fixed second while a job is in flight
# (runpod/serverless/modules/rp_scale.py), so a job arriving just after the last one finished
# waited out the rest of that second: ~0.9 s of queue delay back to back, ~1 job/s per worker
# under load. Holding several job slots keeps a job-take poll open while the GPU computes.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")


async def async_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    return await asyncio.get_running_loop().run_in_executor(_pool, handler, job)


def concurrency(_current: int) -> int:
    return _env_int("LAYA_CONCURRENCY", DEFAULT_CONCURRENCY)


def take_one_job_per_request() -> bool:
    """Make the SDK fetch jobs one at a time even though the worker holds several slots.

    With free slots > 1 the SDK asks the batch job-take API for that many jobs, and RunPod
    starts the execution clock before that call returns: +~80 ms per job on the same GPU and
    datacenter (A4500, EU-RO-1: 235 -> 315 ms p50). The single-job API has no such wait, and the
    fetch loop re-polls straight away while slots remain, so the prefetch is kept. JobScaler
    reads `rp_scale.get_job` when it is constructed; runpod is pinned in requirements.txt.
    """
    from runpod.serverless.modules import rp_job, rp_scale

    if getattr(rp_scale, "get_job", None) is not rp_job.get_job:
        _log.warning("runpod SDK layout changed; leaving its batch job-take in place")
        return False

    async def get_one_job(session, num_jobs: int = 1):
        return await rp_job.get_job(session, 1)

    rp_scale.get_job = get_one_job
    return True


if __name__ == "__main__":
    import runpod

    init()
    take_one_job_per_request()
    runpod.serverless.start({"handler": async_handler, "concurrency_modifier": concurrency})
