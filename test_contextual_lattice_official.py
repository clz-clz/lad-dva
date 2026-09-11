from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import multi_agent_v2
from live_backbone import LiveBackboneResult


REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
SERVED_MODEL = f"Qwen/Qwen3-32B-AWQ@{REVISION}"
MODEL_HASH = "b" * 64


def _settings():
    return {
        "provider": "vllm", "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": SERVED_MODEL, "revision": REVISION,
        "structured_api": "chat-completions-json-schema",
    }


def _record(stage, status="live"):
    return {
        "stage": stage, "status": status, **_settings(),
        "response_model": SERVED_MODEL if status == "live" else None,
        "system_fingerprint": "fp" if status == "live" else None,
        "usage": {"prompt_tokens": 2, "completion_tokens": 1} if status == "live" else {},
        "response_status": "completed" if status == "live" else None,
        "finish_reason": "stop" if status == "live" else None,
        "incomplete_reason": None,
    }


def test_official_contextual_verifier_uses_structured_requester_and_records_live_evidence(monkeypatch):
    calls = []

    def requester(stage, payload):
        calls.append((stage, payload))
        return LiveBackboneResult({"tags": ["B-ORG", "O"]}, _record(stage))

    class ForbiddenLLM:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("official Verifier must not use global ChatOpenAI")

    monkeypatch.setattr(multi_agent_v2, "llm", ForbiddenLLM())
    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", lambda *_args, **_kwargs: [])
    result = asyncio.run(multi_agent_v2.verifier_node({
        "official": True, "tokens": ["Acme", "arrived"],
        "dirty_tags": ["B-ORG", "O"], "candidate_paths": [["B-ORG", "O"], ["B-PER", "O"]],
        "rag_weights": [0.5, 0.5], "dataset_name": "conll2003", "noise_type": "ATF",
        "use_verifier": True, "verify_all": False, "verifier_topk": 4,
        "structured_requester": requester, "provider_settings": _settings(),
        "provider_metadata": {"coder": [_record("coder")], "reviewer": [_record("reviewer")], "verifier": []},
    }))

    assert [stage for stage, _payload in calls] == ["verifier"]
    assert calls[0][1]["name"] == "selectdenoise_verifier"
    assert calls[0][1]["schema"]["properties"]["tags"]["minItems"] == 2
    assert result["current_tags"] == ["B-ORG", "O"]
    assert result["provider_metadata"]["verifier"] == [_record("verifier")]


def test_official_contextual_verifier_records_an_explicit_skip():
    result = asyncio.run(multi_agent_v2.verifier_node({
        "official": True, "tokens": ["Acme"], "dirty_tags": ["B-ORG"],
        "candidate_paths": [["B-ORG"]], "rag_weights": [1.0],
        "dataset_name": "conll2003", "noise_type": "ATF", "use_verifier": True,
        "verify_all": False, "provider_settings": _settings(),
        "provider_metadata": {"coder": [_record("coder")], "reviewer": [_record("reviewer")], "verifier": []},
    }))

    assert result["provider_metadata"]["verifier"][0]["status"] == "skipped_uncontested"


def test_official_contextual_evidence_rejects_live_records_without_usage():
    evidence = {stage: [_record(stage)] for stage in ("coder", "reviewer", "verifier")}
    evidence["reviewer"][0]["usage"] = {}

    with pytest.raises(RuntimeError, match="live evidence"):
        multi_agent_v2._validate_contextual_official_evidence(evidence)


@pytest.mark.parametrize(
    "stage, status",
    [("coder", "skipped"), ("reviewer", "failed"), ("verifier", "local")],
)
def test_official_contextual_evidence_rejects_undocumented_nonlive_status(stage, status):
    evidence = {stage_name: [_record(stage_name)] for stage_name in ("coder", "reviewer", "verifier")}
    record = _record(stage, status)
    record.update({"response_model": None, "response_status": None, "finish_reason": None, "usage": {}})
    evidence[stage] = [record]

    with pytest.raises(RuntimeError, match="status"):
        multi_agent_v2._validate_contextual_official_evidence(evidence)


def test_contextual_terminal_persists_anchor_and_rejects_invalid_terminal_result():
    terminal = SimpleNamespace(
        model_hash=MODEL_HASH,
        fallback_count=0,
        sentence_deer_stats=lambda _tokens: {},
        decode=lambda **_kwargs: SimpleNamespace(
            tags=["B-ORG", "O"], model_hash=MODEL_HASH, used_anchor=True,
            predicted_gain=0.25,
        ),
    )
    result = multi_agent_v2._apply_contextual_lattice_terminal({
        "tokens": ["Acme", "arrived"], "dirty_tags": ["B-ORG", "O"],
        "current_tags": ["B-ORG", "O"], "candidate_paths": [["B-ORG", "O"]],
        "rag_weights": [1.0], "dataset_name": "conll2003", "official": True,
    }, terminal)
    assert result["terminal_anchor_tags"] == ["B-ORG", "O"]

    terminal.decode = lambda **_kwargs: SimpleNamespace(
        tags=["I-ORG", "O"], model_hash=MODEL_HASH, used_anchor=False,
        predicted_gain=0.0,
    )
    with pytest.raises(multi_agent_v2.ContextualLatticeError, match="terminal result"):
        multi_agent_v2._apply_contextual_lattice_terminal({
            "tokens": ["Acme", "arrived"], "dirty_tags": ["B-ORG", "O"],
            "current_tags": ["B-ORG", "O"], "candidate_paths": [["B-ORG", "O"]],
            "rag_weights": [1.0], "dataset_name": "conll2003", "official": True,
        }, terminal)


def test_contextual_replay_passes_only_the_locked_terminal_allowlist():
    captured = {}

    class Terminal:
        model_hash = MODEL_HASH
        fallback_count = 0

        def sentence_deer_stats(self, tokens):
            return {"entity": 0.2, "type": 0.3, "semantic": 0.25, "oov": 0.1}

        def decode(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                tags=["B-ORG", "O"], model_hash=MODEL_HASH,
                used_anchor=True, predicted_gain=0.1,
            )

    result = multi_agent_v2.run_contextual_lattice_replay(
        tokens=["Acme", "arrived"],
        dirty_tags=["B-ORG", "O"],
        provider_record={
            "anchor_tags": ["B-ORG", "O"],
            "candidate_paths": [["B-ORG", "O"]],
            "rag_weights": [1.0],
        },
        dataset_name="conll2003",
        terminal=Terminal(),
    )

    assert set(captured) == {
        "tokens", "dirty_tags", "anchor_tags", "candidate_paths",
        "reviewer_weights", "valid_types", "deer_stats",
    }
    assert "gold_tags" not in captured
    assert result["terminal_model_hash"] == MODEL_HASH
    assert result["terminal_fallback_count"] == 0
