from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path

import pytest

import multi_agent_v2
import run_multiseed


def test_backbone_env_configures_both_clients():
    import langchain_openai

    real_chat_openai = langchain_openai.ChatOpenAI
    saved_env = {
        key: os.environ.get(key)
        for key in ("BACKBONE_MODEL", "BACKBONE_BASE_URL", "BACKBONE_API_KEY")
    }
    init_calls: list[dict] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(dict(kwargs))

    try:
        os.environ["BACKBONE_MODEL"] = "meta-llama/test"
        os.environ["BACKBONE_BASE_URL"] = "http://localhost:8000/v1"
        os.environ["BACKBONE_API_KEY"] = "offline-key"
        langchain_openai.ChatOpenAI = FakeChatOpenAI
        importlib.reload(multi_agent_v2)
    finally:
        langchain_openai.ChatOpenAI = real_chat_openai
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(multi_agent_v2)

    assert init_calls == [
        {
            "model": "meta-llama/test",
            "temperature": 0.7,
            "base_url": "http://localhost:8000/v1",
            "api_key": "offline-key",
        },
        {
            "model": "meta-llama/test",
            "temperature": 1.0,
            "base_url": "http://localhost:8000/v1",
            "api_key": "offline-key",
        },
    ]


def test_run_agent_pipeline_keeps_default_return_type(monkeypatch: pytest.MonkeyPatch):
    async def fake_invoke(state):
        return {
            "current_tags": ["B-PER", "I-PER", "O"],
            "candidate_paths": [["B-PER"], ["B-PER", "I-PER", "O", "B-LOC"]],
            "rag_weights": [0.25],
        }

    monkeypatch.setattr(multi_agent_v2.multi_agent_graph, "ainvoke", fake_invoke)

    result = asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice", "Smith", "arrived"],
            ["O", "O", "O"],
            {"use_verifier": False},
            dataset_name="conll2003",
        )
    )

    assert result == ["B-PER", "I-PER", "O"]


def test_run_agent_pipeline_returns_length_safe_candidate_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_invoke(state):
        return {
            "current_tags": ["B-PER", "I-PER", "O"],
            "candidate_paths": [["B-PER"], ["B-PER", "I-PER", "O", "B-LOC"]],
            "rag_weights": [0.25],
        }

    monkeypatch.setattr(multi_agent_v2.multi_agent_graph, "ainvoke", fake_invoke)

    result = asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice", "Smith", "arrived"],
            ["O", "O", "O"],
            {"__return_candidates__": True, "use_verifier": False},
            dataset_name="conll2003",
        )
    )

    assert result["pred_tags"] == ["B-PER", "I-PER", "O"]
    assert result["candidate_paths"] == [
        ["B-PER", "O", "O"],
        ["B-PER", "I-PER", "O"],
    ]
    assert result["rag_weights"] == [0.25, 1.0]
    assert result["confidence"] == pytest.approx([1.0, 0.8, 1.0])


def test_run_multiseed_persists_candidate_evidence_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    noisy_path = run_multiseed._noisy_path("conll2003", "BT", 13, 1)
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_path.write_text(
        json.dumps(
            {
                "tokens": ["Alice", "Smith", "arrived"],
                "ner_tags": ["B-PER", "I-PER", "O"],
                "dirty_tags": ["O", "O", "O"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured_cfg = {}

    async def fake_pipeline(tokens, dirty_tags, config=None, dataset_name=None):
        captured_cfg.update(config or {})
        return {
            "pred_tags": ["B-PER", "I-PER", "O"],
            "candidate_paths": [["B-PER", "O", "O"], ["B-PER", "I-PER", "O"]],
            "rag_weights": [0.25, 1.0],
            "confidence": [1.0, 0.8, 1.0],
        }

    asyncio.run(
        run_multiseed._run_one_cell(
            "selectdenoise_full",
            run_multiseed.CONFIGURATIONS["selectdenoise_full"],
            "conll2003",
            "BT",
            13,
            1,
            (fake_pipeline, {}),
            max_concurrency=1,
            dummy=False,
        )
    )

    pred_path = run_multiseed._pred_path("selectdenoise_full", "conll2003", "BT", 13)
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    assert captured_cfg["__return_candidates__"] is True
    assert rows == [
        {
            "tokens": ["Alice", "Smith", "arrived"],
            "gold_tags": ["B-PER", "I-PER", "O"],
            "pred_tags": ["B-PER", "I-PER", "O"],
            "candidate_paths": [["B-PER", "O", "O"], ["B-PER", "I-PER", "O"]],
            "rag_weights": [0.25, 1.0],
            "confidence": [1.0, 0.8, 1.0],
        }
    ]
