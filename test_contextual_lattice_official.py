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
        "enable_thinking": True, "thinking_mode": "thinking",
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


def test_official_contextual_coder_preserves_illegal_candidate_for_lattice_filtering(monkeypatch):
    def requester(stage, _payload):
        return LiveBackboneResult({"tags": ["O", "I-PER"]}, _record(stage))

    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", lambda *_args, **_kwargs: [])
    result = asyncio.run(multi_agent_v2.coder_node({
        "official": True,
        "tokens": ["Stefano", "Bordon"],
        "dirty_tags": ["B-PER", "O"],
        "dataset_name": "conll2003",
        "noise_type": "BT",
        "structured_requester": requester,
        "provider_settings": _settings(),
        "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "verifier")},
    }))

    assert result["candidate_paths"] == [["B-PER", "I-PER"]] * 3
    assert result["terminal_candidate_paths"] == [["O", "I-PER"]] * 3
    assert result["fallback_used"] is False


def test_diagnostic_coder_path_filter_calls_only_the_requested_path(monkeypatch):
    calls = []

    def requester(stage, payload):
        calls.append((stage, payload["name"]))
        return LiveBackboneResult({"tags": ["B-PER", "O"]}, _record(stage))

    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", lambda *_args, **_kwargs: [])
    result = asyncio.run(multi_agent_v2.coder_node({
        "official": True,
        "tokens": ["Stefano", "Bordon"],
        "dirty_tags": ["B-PER", "O"],
        "dataset_name": "conll2003",
        "noise_type": "BT",
        "structured_requester": requester,
        "provider_settings": _settings(),
        "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "verifier")},
        "__diagnostic_coder_path__": 5,
    }))

    assert calls == [("coder", "selectdenoise_coder_path_5")]
    assert len(result["terminal_candidate_paths"]) == 1


def test_official_contextual_preterminal_returns_terminal_candidate_view(monkeypatch):
    class Graph:
        async def ainvoke(self, _state):
            return {
                "current_tags": ["B-PER", "O"],
                "candidate_paths": [["B-PER", "O"]],
                "terminal_candidate_paths": [["O", "I-PER"]],
                "rag_weights": [1.0],
                "provider_metadata": {
                    stage: [_record(stage)] for stage in ("coder", "reviewer", "verifier")
                },
                "fallback_used": False,
            }

    monkeypatch.setattr(multi_agent_v2, "_select_pipeline_graph", lambda _config: Graph())
    result = asyncio.run(multi_agent_v2.run_agent_pipeline(
        ["Stefano", "Bordon"],
        ["B-PER", "O"],
        {
            "official": True,
            "terminal_decoder": "contextual-lattice-v1",
            "preterminal_only": True,
            "__return_candidates__": True,
        },
        dataset_name="conll2003",
    ))

    assert result["candidate_paths"] == [["O", "I-PER"]]


def test_contextual_terminal_receives_terminal_candidate_view():
    captured = {}
    terminal = SimpleNamespace(
        model_hash=MODEL_HASH,
        fallback_count=0,
        sentence_deer_stats=lambda _tokens: {},
        decode=lambda **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(
                tags=("B-PER", "O"), model_hash=MODEL_HASH,
                used_anchor=True, predicted_gain=0.0,
            )
        ),
    )

    multi_agent_v2._apply_contextual_lattice_terminal({
        "tokens": ["Stefano", "Bordon"],
        "dirty_tags": ["B-PER", "O"],
        "current_tags": ["B-PER", "O"],
        "candidate_paths": [["B-PER", "O"]],
        "terminal_candidate_paths": [["O", "I-PER"]],
        "rag_weights": [1.0],
        "dataset_name": "conll2003",
    }, terminal)

    assert captured["candidate_paths"] == [["O", "I-PER"]]


def test_contextual_checkpoint_path_resolves_workspace_relative_manifest(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    bundle = workspace / "unified_experiment_20260816" / "locked" / "bundle"
    checkpoint = workspace / "unified_experiment_20260816" / "hf_home" / "snapshot"
    bundle.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.chdir(worktree)

    resolved = multi_agent_v2._resolve_contextual_checkpoint_path(
        "unified_experiment_20260816/hf_home/snapshot", bundle
    )

    assert resolved == checkpoint.resolve()


def test_contextual_terminal_falls_back_from_empty_terminal_candidate_view():
    captured = {}
    terminal = SimpleNamespace(
        model_hash=MODEL_HASH,
        fallback_count=0,
        sentence_deer_stats=lambda _tokens: {},
        decode=lambda **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(
                tags=("B-PER", "O"), model_hash=MODEL_HASH,
                used_anchor=True, predicted_gain=0.0,
            )
        ),
    )

    multi_agent_v2._apply_contextual_lattice_terminal({
        "tokens": ["Stefano", "Bordon"],
        "dirty_tags": ["B-PER", "O"],
        "current_tags": ["B-PER", "O"],
        "candidate_paths": [["B-PER", "O"]],
        "terminal_candidate_paths": [],
        "rag_weights": [1.0],
        "dataset_name": "conll2003",
    }, terminal)

    assert captured["candidate_paths"] == [["B-PER", "O"]]


def test_official_contextual_verifier_uses_structured_requester_and_records_live_evidence(monkeypatch):
    calls = []
    deer_cache_modes = []

    def requester(stage, payload):
        calls.append((stage, payload))
        return LiveBackboneResult({"tags": ["B-ORG", "O"]}, _record(stage))

    class ForbiddenLLM:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("official Verifier must not use global ChatOpenAI")

    monkeypatch.setattr(multi_agent_v2, "llm", ForbiddenLLM())
    def deer_examples(*_args, **kwargs):
        deer_cache_modes.append(kwargs.get("cache_only"))
        return []

    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", deer_examples)
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
    assert deer_cache_modes == [True]
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


def test_official_contextual_coder_reviewer_verifier_use_configured_nothink():
    calls = []
    settings = {
        **_settings(), "enable_thinking": False, "thinking_mode": "nothink",
    }

    def record(stage):
        value = _record(stage)
        value.update({"enable_thinking": False, "thinking_mode": "nothink"})
        return value

    def requester(stage, payload):
        calls.append((stage, payload))
        if stage == "reviewer":
            return LiveBackboneResult({"weights": [0.8, 0.2]}, record(stage))
        return LiveBackboneResult({"tags": ["B-ORG", "O"]}, record(stage))

    # The test must exercise the graph nodes themselves, not only the adapter.
    import multi_agent_v2 as module
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "_get_deer_examples", lambda *_args, **_kwargs: [])
    try:
        asyncio.run(module.coder_node({
            "official": True, "tokens": ["Acme", "arrived"],
            "dirty_tags": ["B-ORG", "O"], "dataset_name": "conll2003",
            "noise_type": "BT", "structured_requester": requester,
            "provider_settings": settings,
            "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "verifier")},
        }))
        asyncio.run(module.reviewer_node({
            "official": True, "tokens": ["Acme", "arrived"],
            "dirty_tags": ["B-ORG", "O"],
            "candidate_paths": [["B-ORG", "O"], ["B-PER", "O"]],
            "dataset_name": "conll2003", "use_lads": True,
            "structured_requester": requester, "provider_settings": settings,
            "provider_metadata": {"coder": [record("coder")], "reviewer": [], "verifier": []},
        }))
        asyncio.run(module.verifier_node({
            "official": True, "tokens": ["Acme", "arrived"],
            "dirty_tags": ["B-ORG", "O"],
            "candidate_paths": [["B-ORG", "O"], ["B-PER", "O"]],
            "rag_weights": [0.5, 0.5], "dataset_name": "conll2003", "noise_type": "ATF",
            "use_verifier": True, "verify_all": False, "verifier_topk": 4,
            "structured_requester": requester, "provider_settings": settings,
            "provider_metadata": {"coder": [record("coder")], "reviewer": [record("reviewer")], "verifier": []},
        }))
    finally:
        monkeypatch.undo()

    assert calls and all(payload["enable_thinking"] is False for _, payload in calls)
    assert {stage for stage, _ in calls} == {"coder", "reviewer", "verifier"}


def test_contextual_replay_preserves_illegal_candidate_for_lattice_filtering():
    captured = {}

    class Terminal:
        model_hash = MODEL_HASH
        fallback_count = 0

        def sentence_deer_stats(self, _tokens):
            return {}

        def decode(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                tags=("B-PER", "O"), model_hash=MODEL_HASH,
                used_anchor=True, predicted_gain=0.0,
            )

    multi_agent_v2.run_contextual_lattice_replay(
        tokens=["Stefano", "Bordon"],
        dirty_tags=["O", "I-PER"],
        provider_record={
            "anchor_tags": ["B-PER", "O"],
            "candidate_paths": [["O", "I-PER"]],
            "rag_weights": [1.0],
        },
        dataset_name="conll2003",
        terminal=Terminal(),
    )

    assert captured["candidate_paths"] == [["O", "I-PER"]]


@pytest.mark.parametrize("candidate", [["O"], ["O", "B-DATE"]])
def test_contextual_replay_still_rejects_candidate_shape_and_ontology(candidate):
    with pytest.raises(multi_agent_v2.ContextualLatticeError, match="candidate_paths"):
        multi_agent_v2.run_contextual_lattice_replay(
            tokens=["Stefano", "Bordon"],
            dirty_tags=["O", "I-PER"],
            provider_record={
                "anchor_tags": ["B-PER", "O"],
                "candidate_paths": [candidate],
                "rag_weights": [1.0],
            },
            dataset_name="conll2003",
            terminal=object(),
        )
