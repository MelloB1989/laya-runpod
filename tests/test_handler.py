"""Handler contract tests against a fake Router, so they run without weights or a GPU."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import handler  # noqa: E402

QUESTIONS = {"dept": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "money", "tech": "bugs"}}}


class FakeRouter:
    loaded = ["english"]
    loaded_revisions = {"english": "abc123"}

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append(("predict", state, questions, kwargs))
        if questions.get("bad", {}).get("type") == "nope":
            raise ValueError("question 'bad' has unknown type 'nope'")
        if state == "explode":
            raise RuntimeError("CUDA out of memory at /secret/path")
        return {"model": "laya-rl-agent", "answers": {q: {"type": s["type"]} for q, s in questions.items()},
                "usage": {"input_tokens": 1, "output_tokens": 0}, "routing": {"model": "english"}}

    def predict_batch(self, requests, **kwargs):
        self.calls.append(("predict_batch", requests, kwargs))
        return [{"answers": {}, "state": r["state"]} for r in requests]

    def route(self, state, questions=None, **kwargs):
        self.calls.append(("route", state, questions, kwargs))
        return {"model": "multilingual", "reason": "non-Latin script"}


@pytest.fixture
def router(monkeypatch):
    monkeypatch.setenv("LAYA_WARMUP", "0")
    fake = FakeRouter()
    handler.init(fake)
    return fake


def run(inp):
    return handler.handler({"id": "job-1", "input": inp})


def test_predict_passes_state_questions_and_overrides(router):
    out = run({"state": "refund me", "questions": QUESTIONS, "model": "ml", "max_len": 2048,
               "min_confidence": 0.6, "lang": "de"})
    assert out["answers"] == {"dept": {"type": "choice"}}
    _, state, questions, kwargs = router.calls[-1]
    assert (state, questions) == ("refund me", QUESTIONS)
    assert kwargs == {"model": "multilingual", "max_len": 2048, "min_confidence": 0.6, "lang": "de"}


def test_jev_model_id_falls_back_to_auto_routing(router):
    run({"state": "x", "questions": QUESTIONS, "model": "jev-1"})
    assert router.calls[-1][3]["model"] is None


def test_preset_fills_questions_and_state_field(router):
    run({"state": "My payment failed twice", "preset": "triage"})
    _, state, questions, _ = router.calls[-1]
    assert state == {"message": "My payment failed twice"}
    assert questions == handler.PRESETS["triage"]()


@pytest.mark.parametrize("inp, expected", [
    ({"questions": QUESTIONS}, "400: 'state' is required"),
    ({"state": "x"}, "400: input needs a 'questions' object or a 'preset'"),
    ({"state": "x", "preset": "nope"}, "400: unknown preset 'nope'"),
    ({"state": "x", "preset": "triage", "questions": QUESTIONS}, "400: pass either"),
    ({"state": "x" * 50001, "questions": QUESTIONS}, "413: state too large"),
    ({"state": "x", "questions": {str(i): QUESTIONS["dept"] for i in range(65)}}, "413: too many questions"),
    ({"state": "x", "questions": QUESTIONS, "max_len": 99999}, "422: max_len exceeds server limit"),
    ({"state": "x", "questions": QUESTIONS, "min_confidence": 2}, "422: min_confidence"),
    ({"state": "x", "questions": QUESTIONS, "action": "nope"}, "400: unknown action"),
])
def test_invalid_input_is_a_failed_job(router, inp, expected):
    out = run(inp)
    assert out["error"].startswith(expected), out
    assert not any(c[0] == "predict" for c in router.calls[1:])


def test_non_object_input(router):
    assert run(["x"])["error"] == "400: job input must be an object"


def test_laya_value_error_is_422(router):
    out = run({"state": "x", "questions": {"bad": {"type": "nope", "instructions": "?"}}})
    assert out == {"error": "422: question 'bad' has unknown type 'nope'"}


def test_internal_error_does_not_leak(router):
    out = run({"state": "explode", "questions": QUESTIONS})
    assert out == {"error": "500: inference failed"}


def test_batch(router):
    out = run({"requests": [{"state": "a", "questions": QUESTIONS},
                            {"state": "b", "preset": "guard", "model": "english"}],
               "batch_size": 8, "sort_by_length": True})
    assert [r["state"] for r in out["results"]] == ["a", {"prompt": "b"}]
    _, requests, kwargs = router.calls[-1]
    assert requests[1]["model"] == "english"
    assert kwargs == {"batch_size": 8, "min_confidence": None, "sort_by_length": True}


def test_batch_error_names_the_request(router):
    out = run({"requests": [{"state": "a", "questions": QUESTIONS}, {"questions": QUESTIONS}]})
    assert out["error"] == "400: requests[1]: 'state' is required"


def test_batch_cap(router, monkeypatch):
    monkeypatch.setenv("LAYA_MAX_BATCH", "2")
    out = run({"requests": [{"state": "a", "questions": QUESTIONS}] * 3})
    assert out["error"] == "413: too many requests in batch (3 > 2)"


def test_route(router):
    out = run({"action": "route", "state": "नमस्ते"})
    assert out == {"model": "multilingual", "reason": "non-Latin script"}


def test_warmup_runs_each_loaded_checkpoint(monkeypatch):
    monkeypatch.setenv("LAYA_WARMUP", "1")
    fake = FakeRouter()
    handler.init(fake)
    assert [c[3]["model"] for c in fake.calls] == ["english"]


def test_async_handler_runs_the_same_handler(router):
    import asyncio
    out = asyncio.run(handler.async_handler({"id": "job-2", "input": {"state": "x", "questions": QUESTIONS}}))
    assert out["answers"] == {"dept": {"type": "choice"}}


def test_concurrency_from_env(monkeypatch):
    assert handler.concurrency(1) == handler.DEFAULT_CONCURRENCY
    monkeypatch.setenv("LAYA_CONCURRENCY", "2")
    assert handler.concurrency(1) == 2
    monkeypatch.setenv("LAYA_CONCURRENCY", "0")
    assert handler.concurrency(1) == handler.DEFAULT_CONCURRENCY


def test_job_fetcher_takes_one_job_per_request(monkeypatch):
    import asyncio
    from runpod.serverless.modules import rp_job, rp_scale

    asked = []

    async def fake_get_job(session, num_jobs=1):
        asked.append(num_jobs)
        return [{"id": "j", "input": {}}]

    monkeypatch.setattr(rp_job, "get_job", fake_get_job)
    monkeypatch.setattr(rp_scale, "get_job", fake_get_job)
    assert handler.take_one_job_per_request()
    assert asyncio.run(rp_scale.get_job(None, 4)) == [{"id": "j", "input": {}}]
    assert asked == [1]


def test_job_fetcher_patch_skips_an_unknown_sdk_layout(monkeypatch):
    from runpod.serverless.modules import rp_scale

    async def other(session, num_jobs=1):
        return None

    monkeypatch.setattr(rp_scale, "get_job", other)
    assert handler.take_one_job_per_request() is False
    assert rp_scale.get_job is other
