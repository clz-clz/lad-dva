import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import multi_agent_v2
import run_multiseed
from live_backbone import LiveBackboneResult, LiveBackboneSettings


CONTEXTUAL_REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"


def _env(provider="vllm"):
    values = {
        "BACKBONE_PROVIDER": provider,
        "BACKBONE_MODEL": "Qwen/Qwen3-32B-AWQ",
        "BACKBONE_BASE_URL": "http://127.0.0.1:8000/v1?secret=no",
        "BACKBONE_API_KEY": "test-key",
        "BACKBONE_TAG": "qwen32b-r1",
        "BACKBONE_REVISION": "a" * 40,
    }
    if provider == "deepseek":
        values["BACKBONE_MODEL"] = "deepseek-v4-flash"
        values["BACKBONE_BASE_URL"] = "https://api.deepseek.com/v1"
        values.pop("BACKBONE_REVISION")
    return values


def _identity():
    revision = "a" * 40
    return {
        "provider": "vllm",
        "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": f"Qwen/Qwen3-32B-AWQ@{revision}",
        "revision": revision,
        "structured_api": "chat-completions-json-schema",
    }


def _live_record(stage):
    identity = _identity()
    return {
        "stage": stage,
        "status": "live",
        **identity,
        "response_model": identity["served_model"],
        "structured_api": identity["structured_api"],
        "response_status": "completed",
        "finish_reason": "stop",
        "incomplete_reason": None,
        "system_fingerprint": "fp-test",
        "usage": {},
    }


def _status_record(stage, status):
    identity = _identity()
    return {
        "stage": stage,
        "status": status,
        **_identity(),
        "response_model": None,
        "structured_api": identity["structured_api"],
        "response_status": None,
        "finish_reason": None,
        "incomplete_reason": None,
        "system_fingerprint": None,
        "usage": {},
    }


def _full_evidence(*, ror_status="not_triggered", gasd_status="live",
                   coder_stages=(1, 2, 5)):
    return {
        "coder": [_live_record(f"coder_path_{index}") for index in coder_stages],
        "reviewer": [_live_record("reviewer")],
        "ror": [_status_record("ror", ror_status)],
        "gasd": ([_live_record("gasd_r")] if gasd_status == "live"
                 else [_status_record("gasd", gasd_status)]),
    }


def _contextual_identity():
    return {
        "provider": "vllm",
        "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": f"Qwen/Qwen3-32B-AWQ@{CONTEXTUAL_REVISION}",
        "revision": CONTEXTUAL_REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": True,
        "thinking_mode": "thinking",
    }


def _contextual_live_record(stage):
    identity = _contextual_identity()
    return {
        "stage": stage, "status": "live", **identity,
        "response_model": identity["served_model"],
        "response_status": "completed", "finish_reason": "stop",
        "incomplete_reason": None, "system_fingerprint": "fp-contextual",
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


def _contextual_skipped_record(stage, status):
    return {
        "stage": stage, "status": status, **_contextual_identity(),
        "response_model": None, "response_status": None, "finish_reason": None,
        "incomplete_reason": None, "system_fingerprint": None, "usage": {},
    }


def test_official_contextual_profile_accepts_three_stage_prediction_and_anchor(tmp_path):
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
        revision=CONTEXTUAL_REVISION,
    )
    run_multiseed._validate_official_settings_for_configs(
        settings, ["selectdenoise_contextual_lattice"]
    )
    evidence = {
        "coder": [_contextual_live_record("coder") for _ in range(3)],
        "reviewer": [_contextual_skipped_record("reviewer", "skipped_identical")],
        "verifier": [_contextual_skipped_record("verifier", "skipped_uncontested")],
    }
    row = {
        "tokens": ["Alice", "works"], "gold_tags": ["B-PER", "O"],
        "pred_tags": ["B-PER", "O"], "terminal_anchor_tags": ["B-PER", "O"],
        "terminal_model_hash": "a" * 64, "provider_metadata": evidence,
        "fallback_used": False,
    }
    path = tmp_path / "contextual.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(200)), encoding="utf-8")

    run_multiseed._validate_official_prediction(
        path, "conll2003", "selectdenoise_contextual_lattice", "BT",
        provider_identity=_contextual_identity(),
    )


@pytest.mark.parametrize(
    "stage, status",
    [("coder", "skipped"), ("reviewer", "failed"), ("verifier", "local")],
)
def test_official_contextual_evidence_rejects_undocumented_nonlive_status(stage, status):
    evidence = {
        "coder": [_contextual_live_record("coder") for _ in range(3)],
        "reviewer": [_contextual_skipped_record("reviewer", "skipped_identical")],
        "verifier": [_contextual_skipped_record("verifier", "skipped_uncontested")],
    }
    evidence[stage] = [_contextual_skipped_record(stage, status)]

    with pytest.raises(RuntimeError, match="[Cc]oder|[Rr]eviewer|[Vv]erifier"):
        run_multiseed._validate_official_provider_evidence(
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            evidence, _contextual_identity(), None, noise_type="BT",
        )


@pytest.mark.parametrize('smoke_indices', [None, [2, 17]])
def test_official_contextual_runner_persists_terminal_anchor_evidence(tmp_path, monkeypatch, smoke_indices):
    noisy_dir, pred_dir = tmp_path / "noisy", tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "contextual-offline")
    _write_noisy(noisy_dir / "noisy_seed13__BT__conll2003__N200.jsonl", count=200)
    evidence = {
        "coder": [_contextual_live_record("coder") for _ in range(3)],
        "reviewer": [_contextual_skipped_record("reviewer", "skipped_identical")],
        "verifier": [_contextual_skipped_record("verifier", "skipped_uncontested")],
    }

    class Adapter:
        provider_metadata = staticmethod(_contextual_identity)
        structured_requester = staticmethod(lambda *_args: (_ for _ in ()).throw(
            AssertionError("fake pipeline must not invoke a provider")
        ))

    observed_contexts = []
    async def pipeline(tokens, dirty, _config, **_kwargs):
        from request_diagnostics import _sentence
        observed_contexts.append(await asyncio.to_thread(_sentence.get))
        return {
            "pred_tags": ["B-PER", "O"], "candidate_paths": [["B-PER", "O"]],
            "rag_weights": [1.0], "confidence": [1.0, 1.0],
            "terminal_anchor_tags": ["B-PER", "O"], "terminal_model_hash": "a" * 64,
            "provider_metadata": evidence, "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "conll2003", "BT", 13, 200, (pipeline, {}), max_concurrency=8,
        dummy=False, official=smoke_indices is None, failure_policy="abort", request_timeout=10.0,
        adapter_factory=Adapter,
        paid_smoke_size=len(smoke_indices) if smoke_indices else None,
        row_indices=smoke_indices,
        prediction_path=pred_dir / 'smoke.jsonl' if smoke_indices else None,
    ))

    output = next(pred_dir.glob("*.jsonl"))
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["terminal_anchor_tags"] == ["B-PER", "O"]
    assert {c['row_index'] for c in observed_contexts} == set(smoke_indices or range(200))
    assert all(c['dataset'] == 'conll2003' and c['noise'] == 'BT' and c['seed'] == 13
               for c in observed_contexts)


def test_provider_cache_cell_runs_preterminal_graph_once_and_is_exactly_resumable(tmp_path, monkeypatch):
    """Changing a source digest or manifest identity must prevent cache reuse."""
    noisy_dir, cache_root = tmp_path / "noisy", tmp_path / "cache"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    calls = []

    class Adapter:
        def __init__(self):
            self.closed = False

        provider_metadata = staticmethod(_contextual_identity)
        structured_requester = staticmethod(lambda *_args: None)

        def close(self):
            self.closed = True

    adapter = Adapter()

    async def pipeline(tokens, dirty, config, **_kwargs):
        calls.append(dict(config))
        return {
            "pred_tags": ["B-PER", "O"],
            "candidate_paths": [["B-PER", "O"]],
            "rag_weights": [1.0], "confidence": [1.0, 1.0],
            "provider_metadata": {
                "coder": [_contextual_live_record("coder") for _ in range(3)],
                "reviewer": [_contextual_skipped_record("reviewer", "skipped_identical")],
                "verifier": [_contextual_skipped_record("verifier", "skipped_uncontested")],
            },
            "fallback_used": False,
        }

    first = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice", run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 200, (pipeline, {}), cache_root=cache_root,
        cache_tag="qwen-test", max_concurrency=20, request_timeout=10.0,
        adapter_factory=lambda: adapter, git_sha="a" * 40, bundle_hash="b" * 64,
    ))

    assert len(calls) == 200
    assert all(config["terminal_decoder"] == "contextual-lattice-v1" for config in calls)
    assert all(config["preterminal_only"] is True for config in calls)
    assert first["row_count"] == 200
    assert first["sha256"] == hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()
    assert adapter.closed is False
    calls.clear()

    second = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice", run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 200, (pipeline, {}), cache_root=cache_root,
        cache_tag="qwen-test", max_concurrency=20, request_timeout=10.0,
        adapter_factory=lambda: adapter, git_sha="a" * 40, bundle_hash="b" * 64,
    ))

    assert second == first
    assert calls == []


def test_contextual_provider_cache_uses_real_pipeline_preterminal_mode(monkeypatch):
    class Graph:
        async def ainvoke(self, state):
            assert state["official"] is True
            assert state["provider_metadata"]
            return {
                **state,
                "current_tags": ["B-PER", "O"],
                "candidate_paths": [["B-PER", "O"]],
                "rag_weights": [1.0],
                "provider_metadata": {
                    "coder": [_contextual_live_record("coder") for _ in range(3)],
                    "reviewer": [_contextual_skipped_record("reviewer", "skipped_identical")],
                    "verifier": [_contextual_skipped_record("verifier", "skipped_uncontested")],
                },
            }

    monkeypatch.setattr(multi_agent_v2, "_select_pipeline_graph", lambda config: Graph())
    monkeypatch.setattr(
        multi_agent_v2, "_load_contextual_lattice_terminal",
        lambda: (_ for _ in ()).throw(AssertionError("offline terminal must be skipped")),
    )
    config = {
        **run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "official": True,
        "preterminal_only": True,
        "provider_metadata": _contextual_identity(),
        "__return_candidates__": True,
    }

    result = asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice", "works"], ["B-PER", "O"], config, dataset_name="msra"
        )
    )

    assert result["pred_tags"] == ["B-PER", "O"]
    assert result["fallback_used"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("datasets", ["msra"]), ("noise_types", ["BT"]), ("seeds", [13])],
)
def test_provider_cache_phase_rejects_partial_canonical_matrix(field, value):
    kwargs = {
        "datasets": run_multiseed.DATASETS,
        "noise_types": run_multiseed.NOISE_TYPES,
        "seeds": run_multiseed.SEEDS,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        run_multiseed._validate_provider_cache_launch(
            ["selectdenoise_contextual_lattice"], size=200, ratios=[0.15],
            max_concurrency=20, failure_policy="abort", dummy=False, **kwargs,
        )


def test_provider_cache_does_not_create_historical_prediction_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKBONE_PROVIDER", "vllm")
    monkeypatch.setenv("BACKBONE_MODEL", "Qwen/Qwen3-32B-AWQ")
    monkeypatch.setenv("BACKBONE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("BACKBONE_API_KEY", "offline-key")
    monkeypatch.setenv("BACKBONE_TAG", "qwen-cache-test")
    monkeypatch.setenv("BACKBONE_REVISION", CONTEXTUAL_REVISION)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(run_multiseed, "_import_pipeline", lambda _dummy: (None, {}))
    class Factory:
        def close(self):
            pass

    monkeypatch.setattr(run_multiseed, "_CachedAdapterFactory", lambda _factory: Factory())
    monkeypatch.setattr(
        run_multiseed, "_ensure_official_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("historical manifest used")),
    )
    calls = []

    async def fake_cell(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(run_multiseed, "_run_provider_cache_cell", fake_cell)

    run_multiseed.main([
        "--phase", "provider-cache", "--official", "--size", "200",
        "--max-concurrency", "20", "--bundle-hash", "b" * 64,
        "--provider-cache-root", str(tmp_path / "cache"),
    ])

    assert len(calls) == 45
    assert not list((tmp_path / "predictions").glob("run_manifest__*.json"))


def test_provider_cache_phase_rejects_non_contextual_or_noncanonical_launches():
    args = run_multiseed._parse_args(["--phase", "provider-cache"])
    assert args.phase == "provider-cache"
    with pytest.raises(ValueError, match="contextual-lattice"):
        run_multiseed._validate_provider_cache_launch(
            ["lad_rg_full"], size=200, ratios=[0.15], max_concurrency=20,
            failure_policy="abort", dummy=False,
        )
    with pytest.raises(ValueError, match="max-concurrency 20"):
        run_multiseed._validate_provider_cache_launch(
            ["selectdenoise_contextual_lattice"], size=200, ratios=[0.15],
            max_concurrency=19, failure_policy="abort", dummy=False,
        )


def _write_noisy(path: Path, count=1):
    row = {
        "tokens": ["Alice", "works"],
        "ner_tags": ["B-PER", "O"],
        "dirty_tags": ["B-PER", "O"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8")


def _write_official_prediction(path: Path, *, evidence=None, count=200):
    row = {
        "tokens": ["Alice", "works"],
        "gold_tags": ["B-PER", "O"],
        "pred_tags": ["B-PER", "O"],
        "ror_reasoning_source": "not_triggered",
        "gasd_variant_requested": "r",
        "gasd_variant_used": "r",
        "provider_metadata": evidence or _full_evidence(),
        "fallback_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("noise_type", "expected_stages"),
    [
        ("BT", ["coder_path_1", "coder_path_2", "coder_path_5"]),
        ("IF", ["coder_path_1", "coder_path_2", "coder_path_5"]),
        ("ATF", [f"coder_path_{index}" for index in range(1, 6)]),
    ],
)
def test_real_coder_evidence_validates_exact_noise_stage_set(
    monkeypatch, noise_type, expected_stages
):
    identity = _identity()

    def structured_requester(stage, payload):
        return LiveBackboneResult(
            {"tags": ["B-PER", "O"]},
            {
                **identity,
                "stage": stage,
                "status": "live",
                "response_model": identity["served_model"],
                "response_status": "completed",
                "finish_reason": "stop",
                "incomplete_reason": None,
                "system_fingerprint": "fp-test",
                "usage": {},
            },
        )

    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", lambda *args, **kwargs: [])
    result = asyncio.run(multi_agent_v2.coder_node({
        "tokens": ["Alice", "works"],
        "dirty_tags": ["B-PER", "O"],
        "dataset_name": "msra",
        "noise_type": noise_type,
        "official": True,
        "provider_settings": identity,
        "provider_metadata": {},
        "structured_requester": structured_requester,
    }))
    evidence = _full_evidence()
    evidence["coder"] = result["provider_metadata"]["coder"]

    run_multiseed._validate_official_provider_evidence(
        run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
        evidence,
        identity,
        "not_triggered",
        noise_type=noise_type,
    )

    assert [record["stage"] for record in evidence["coder"]] == expected_stages


@pytest.mark.parametrize(
    "coder_records",
    [
        [_live_record("coder_path_1"), _live_record("coder_path_2")],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _live_record("coder_path_5"), _live_record("coder_path_5"),
        ],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _live_record("coder_path_3"), _live_record("coder_path_5"),
        ],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _status_record("coder_path_5", "skipped"),
        ],
    ],
    ids=("missing", "duplicate", "extra", "non-live"),
)
def test_official_coder_evidence_rejects_invalid_bt_stage_sets(coder_records):
    evidence = _full_evidence()
    evidence["coder"] = coder_records

    with pytest.raises(RuntimeError, match="[Cc]oder"):
        run_multiseed._validate_official_provider_evidence(
            run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            evidence,
            _identity(),
            "not_triggered",
            noise_type="BT",
        )


def test_official_coder_evidence_rejects_unsupported_noise_type():
    with pytest.raises(RuntimeError, match="unsupported official noise type"):
        run_multiseed._validate_official_provider_evidence(
            run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            _full_evidence(),
            _identity(),
            "not_triggered",
            noise_type="OTHER",
        )


@pytest.mark.parametrize("record_count", [199, 201])
def test_official_cell_rejects_non_200_input_before_adapter_or_calls(
    tmp_path, monkeypatch, record_count
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-count")
    _write_noisy(
        noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=record_count
    )
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-count.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")

    def adapter_factory():
        raise AssertionError("adapter must not be constructed for a non-200 cell")

    async def pipeline(*args, **kwargs):
        raise AssertionError("pipeline must not run for a non-200 cell")

    with pytest.raises(RuntimeError, match="exactly 200"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort",
            request_timeout=10.0, adapter_factory=adapter_factory,
        ))

    assert not target_tmp.exists()


def test_official_adapter_construction_failure_cleans_exact_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-adapter-fail")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-adapter-fail.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="adapter failed"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(RuntimeError("adapter failed")),
        ))

    assert not target_tmp.exists()


def test_official_noisy_load_failure_cleans_only_exact_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-load-fail")
    noisy_path = noisy_dir / "noisy_seed13__BT__msra__N200.jsonl"
    noisy_path.parent.mkdir(parents=True)
    noisy_path.write_text("not-json\n", encoding="utf-8")
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-load-fail.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_full__msra__BT__qwen-load-fail.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("adapter must not be constructed after a noisy-load failure")
            ),
        ))

    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_official_resume_rejects_non_200_input_before_adapter(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-resume-count")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=199)
    _write_official_prediction(
        pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen-resume-count.jsonl"
    )

    with pytest.raises(RuntimeError, match="exactly 200"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("adapter must not be constructed for a non-200 resume")
            ),
        ))


def test_official_resume_rejects_mislabelled_stage_evidence_and_cleans_exact_temp(
    tmp_path, monkeypatch
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-resume-evidence")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    evidence = _full_evidence()
    evidence["coder"][0]["stage"] = "gasd_r"
    target = pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen-resume-evidence.jsonl"
    _write_official_prediction(target, evidence=evidence)
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_gasd_r__msra__BT__qwen-resume-evidence.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    class Adapter:
        provider_metadata = staticmethod(_identity)

    with pytest.raises(RuntimeError, match="stage"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=Adapter,
        ))

    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_cli_defaults_abort_and_require_explicit_legacy_dirty():
    official = run_multiseed._parse_args(["--official", "--size", "200"])
    legacy = run_multiseed._parse_args([])
    explicit_dirty = run_multiseed._parse_args(["--failure-policy", "dirty"])

    assert (official.failure_policy, official.request_timeout) == ("abort", 3600.0)
    assert (legacy.failure_policy, legacy.request_timeout) == ("abort", 180.0)
    assert explicit_dirty.failure_policy == "dirty"


def test_official_request_model_configuration_uses_immutable_served_name(monkeypatch):
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision="a" * 40,
    )
    monkeypatch.setenv("BACKBONE_SERVED_MODEL", "")
    monkeypatch.setenv("LAD_RG_OFFICIAL_REQUESTS", "0")

    assert run_multiseed._configure_official_request_model(settings) == settings.served_model
    assert run_multiseed.os.environ["BACKBONE_SERVED_MODEL"] == settings.served_model
    assert run_multiseed.os.environ["LAD_RG_OFFICIAL_REQUESTS"] == "1"


def test_official_coder_and_reviewer_clients_use_manifest_sdk_controls():
    environment = dict(os.environ)
    environment.update({
        "LAD_RG_OFFICIAL_REQUESTS": "1",
        "BACKBONE_MODEL": "deepseek-v4-flash",
        "BACKBONE_SERVED_MODEL": "deepseek-v4-flash",
        "BACKBONE_BASE_URL": "http://127.0.0.1:9/v1",
        "BACKBONE_API_KEY": "offline-key",
    })
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, multi_agent_v2 as m; "
                "print(json.dumps({"
                "'reviewer_timeout': m.llm.request_timeout, "
                "'reviewer_retries': m.llm.max_retries, "
                "'reviewer_trust_env': getattr(m.llm.http_client, '_trust_env', None), "
                "'reviewer_max_connections': m.llm.http_client._transport._pool._max_connections, "
                "'reviewer_max_keepalive': m.llm.http_client._transport._pool._max_keepalive_connections, "
                "'coder_timeout': m.coder_llm.request_timeout, "
                "'coder_retries': m.coder_llm.max_retries, "
                "'coder_trust_env': getattr(m.coder_llm.http_client, '_trust_env', None), "
                "'coder_max_connections': m.coder_llm.http_client._transport._pool._max_connections, "
                "'coder_max_keepalive': m.coder_llm.http_client._transport._pool._max_keepalive_connections}))"
            ),
        ],
        cwd=Path(__file__).parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
        check=True,
    )
    controls = json.loads(completed.stdout.strip())

    assert controls == {
        "reviewer_timeout": 120.0,
        "reviewer_retries": 2,
        "reviewer_trust_env": False,
        "reviewer_max_connections": 1000,
        "reviewer_max_keepalive": 100,
        "coder_timeout": 120.0,
        "coder_retries": 2,
        "coder_trust_env": False,
        "coder_max_connections": 1000,
        "coder_max_keepalive": 100,
    }
    assert multi_agent_v2._provider_client_options({}) == {}


def test_cached_adapter_factory_reuses_one_adapter_and_closes_it_once():
    factory_type = getattr(run_multiseed, "_CachedAdapterFactory", None)
    assert factory_type is not None
    created = []

    class Adapter:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    def create():
        adapter = Adapter()
        created.append(adapter)
        return adapter

    factory = factory_type(create)
    first = factory()
    assert factory() is first
    assert created == [first]
    factory.close()
    factory.close()
    assert first.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        factory()


def test_official_environment_is_explicit_and_rejects_deepseek_gasd_r():
    with pytest.raises(ValueError, match="BACKBONE_TAG"):
        run_multiseed._official_settings_from_env(["lad_rg_full"], {})
    with pytest.raises(ValueError, match="GASD-R/Both"):
        run_multiseed._official_settings_from_env(["lad_rg_gasd_r"], _env("deepseek"))


def test_qwen_nothink_environment_is_bound_to_manifest_and_requires_fresh_tag():
    environment = _env()
    environment.update({
        "QWEN_ENABLE_THINKING": "false",
        "BACKBONE_TAG": "qwen32b-contextual-nothink",
        "BACKBONE_REVISION": CONTEXTUAL_REVISION,
    })
    settings, tag = run_multiseed._official_settings_from_env(
        ["selectdenoise_contextual_lattice"], environment,
    )
    assert settings.enable_thinking is False
    assert settings.thinking_mode == "nothink"
    manifest = run_multiseed._build_official_manifest(settings, tag, "1" * 40, 3600.0)
    assert manifest["backbone"]["enable_thinking"] is False
    assert manifest["backbone"]["thinking_mode"] == "nothink"
    assert "nothink" in manifest["backbone"]["tag"]

    environment["BACKBONE_TAG"] = "qwen32b-contextual-thinking"
    with pytest.raises(ValueError, match="nothink"):
        run_multiseed._official_settings_from_env(
            ["selectdenoise_contextual_lattice"], environment,
        )


def test_manifest_is_canonical_sanitized_and_required_for_resume(tmp_path):
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://user:pass@127.0.0.1:8000/v1?token=x",
        api_key="secret", revision="a" * 40,
    )
    manifest = run_multiseed._build_official_manifest(
        settings, "qwen32b-r1", "1" * 40, 600.0
    )
    assert manifest["backbone"]["endpoint_origin"] == "http://127.0.0.1:8000"
    assert manifest["backbone"]["served_model"] == settings.served_model
    assert "secret" not in run_multiseed._canonical_json(manifest)
    assert manifest["decoder_constants"] == {
        "contract_version": "lad-rg-official-decoder-v2",
        "structured_api": "chat-completions-json-schema",
        "hard_constraint": "strict-iob2-v1",
        "provider_timeout_seconds": 120.0,
        "sdk_max_retries": 2,
        "runner_request_timeout_seconds": 600.0,
        "gasd_beta_omega": 2.0,
        "gasd_gamma_proposal": 1.0,
        "gasd_candidate_scale_g": 1.0,
        "gasd_candidate_scale_r": 0.0,
        "gasd_reason_bonus": 2.0,
        "ror_omega_quantile": 0.6,
        "ror_confidence_threshold": 0.6,
        "voting_entity_boost": 1.0,
        "voting_consensus_ratio": 0.6,
    }

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    stale = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen32b-r1.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing manifest"):
        run_multiseed._ensure_official_manifest(manifest, pred_dir, "qwen32b-r1")


def test_official_cell_reuses_one_adapter_and_persists_launch_evidence(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen32b-r1")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    adapters = []

    class Adapter:
        def __init__(self):
            adapters.append(self)

        def ror_reasoner(self, stage, payload):
            raise AssertionError("not called by fake pipeline")

        def gasd_reason_decoder(self, payload):
            raise AssertionError("not called by fake pipeline")

        def provider_metadata(self):
            return _identity()

    async def pipeline(tokens, dirty, config, dataset_name=None):
        assert config["official"] is True
        assert config["ror_reasoner"].__self__ is adapters[0]
        assert config["gasd_reason_decoder"].__self__ is adapters[0]
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "r",
            "gasd_variant_used": "r",
            "provider_metadata": _full_evidence(),
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=2,
        dummy=False, ratio=0.15, official=True, failure_policy="abort",
        request_timeout=10.0, adapter_factory=Adapter,
    ))

    assert len(adapters) == 1
    rows = run_multiseed._load_noisy(
        pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen32b-r1.jsonl"
    )
    assert len(rows) == 200
    assert rows[0]["gasd_variant_used"] == "r"
    assert rows[0]["fallback_used"] is False


def test_paid_smoke_uses_first_20_canonical_rows_with_strict_evidence(
    tmp_path, monkeypatch
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    smoke_output = pred_dir / "smoke" / "deepseek.jsonl"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "deepseek-smoke")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    identity = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "served_model": "deepseek-v4-flash",
        "revision": None,
        "structured_api": "responses-json-schema",
    }
    calls = []

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(lambda: dict(identity))

    def record(stage, status, response_model=None):
        return {
            "stage": stage,
            "status": status,
            **identity,
            "response_model": response_model,
            "response_status": "completed" if response_model else None,
            "finish_reason": None,
            "incomplete_reason": None,
            "system_fingerprint": "fp-smoke" if response_model else None,
            "usage": {},
        }

    async def pipeline(tokens, dirty, config, dataset_name=None):
        calls.append((list(tokens), dataset_name))
        assert config["official"] is True
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "g",
            "provider_metadata": {
                "coder": [
                    record("coder_path_1", "live", identity["served_model"]),
                    record("coder_path_2", "live", identity["served_model"]),
                    record("coder_path_5", "live", identity["served_model"]),
                ],
                "reviewer": [record("reviewer", "live", identity["served_model"])],
                "ror": [record("ror", "not_triggered")],
                "gasd": [record("gasd", "local")],
            },
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=2,
        dummy=False, ratio=0.15, official=False, paid_smoke_size=20,
        prediction_path=smoke_output, failure_policy="abort",
        request_timeout=10.0, adapter_factory=Adapter,
    ))

    rows = run_multiseed._load_noisy(smoke_output)
    assert len(calls) == len(rows) == 20
    assert all(row["fallback_used"] is False for row in rows)
    assert not (pred_dir / "run_manifest__deepseek-smoke.json").exists()


def test_paid_smoke_rejects_noncanonical_source_size_before_adapter(
    tmp_path, monkeypatch
):
    noisy_dir = tmp_path / "noisy"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N20.jsonl", count=20)

    with pytest.raises(ValueError, match="canonical size 200"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 20, (lambda *args: None, {}),
            max_concurrency=1, dummy=False, official=False, paid_smoke_size=20,
            prediction_path=tmp_path / "smoke.jsonl", failure_policy="abort",
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("adapter must not be constructed")
            ),
        ))


def test_official_failure_removes_only_exact_cell_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "tag")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__tag.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("partial", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_full__msra__BT__tag.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(lambda: {})

    async def broken(*args, **kwargs):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (broken, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort",
            request_timeout=10.0, adapter_factory=Adapter,
        ))

    assert not target.exists()
    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_official_failure_cancels_and_drains_pending_sentences(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "fail-fast")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(_identity)

    calls = 0
    cancelled = 0

    async def pipeline(tokens, dirty, config, dataset_name=None):
        nonlocal calls, cancelled
        calls += 1
        if calls == 1:
            await asyncio.sleep(0)
            raise RuntimeError("provider failed")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "g",
            "provider_metadata": _full_evidence(gasd_status="local"),
            "fallback_used": False,
        }

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=5,
            dummy=False, official=True, failure_policy="abort",
            request_timeout=10.0, adapter_factory=Adapter,
        ))

    assert 1 <= calls < 200
    assert cancelled >= 1
    assert not list(pred_dir.glob("*fail-fast*"))


def test_official_cell_rejects_empty_stage_evidence(tmp_path, monkeypatch):
    from tqdm.asyncio import tqdm as async_tqdm

    async def forbidden_progress_gather(*args, **kwargs):
        raise AssertionError("official failures must not use orphan-prone tqdm gather")

    monkeypatch.setattr(async_tqdm, "gather", forbidden_progress_gather)
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-empty-evidence")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(_identity)

    async def pipeline(tokens, dirty, config, dataset_name=None):
        return {
            "pred_tags": list(dirty), "candidate_paths": [list(dirty)],
            "rag_weights": [1.0], "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "r", "gasd_variant_used": "r",
            "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "ror", "gasd")},
            "fallback_used": False,
        }

    with pytest.raises(RuntimeError, match="provider evidence"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=5,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=Adapter,
        ))


def test_official_gasd_disabled_cell_accepts_disabled_variant_evidence(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen32b-g-off")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(_identity)

    async def pipeline(tokens, dirty, config, dataset_name=None):
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "disabled",
            "provider_metadata": _full_evidence(gasd_status="disabled"),
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_no_gasd", run_multiseed.CONFIGURATIONS["lad_rg_no_gasd"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=1,
        dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
        adapter_factory=Adapter,
    ))

    output = pred_dir / "pred_seed13__lad_rg_no_gasd__msra__BT__qwen32b-g-off.jsonl"
    assert run_multiseed._load_noisy(output)[0]["gasd_variant_used"] == "disabled"


def test_official_runner_injects_one_adapter_structured_requester_into_pipeline(
    tmp_path, monkeypatch
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-structured-v2")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    adapters = []

    class Adapter:
        def __init__(self):
            adapters.append(self)

        def structured_requester(self, stage, payload):
            raise AssertionError("fake pipeline should receive, not call, requester")

        def ror_reasoner(self, stage, payload):
            raise AssertionError("fake pipeline should not call RoR")

        def gasd_reason_decoder(self, payload):
            raise AssertionError("fake pipeline should not call GASD")

        def provider_metadata(self):
            return _identity()

    async def pipeline(tokens, dirty, config, dataset_name=None):
        assert config["structured_requester"].__self__ is adapters[0]
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "g",
            "provider_metadata": _full_evidence(gasd_status="local"),
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=1,
        dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
        adapter_factory=Adapter,
    ))

    assert len(adapters) == 1
