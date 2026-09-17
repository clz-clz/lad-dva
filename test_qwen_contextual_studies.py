import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import asyncio

import pytest

import contextual_studies as study
import run_qwen_contextual_studies as runner


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "DATASETS", ("msra",))
    monkeypatch.setattr(study, "NOISE_TYPES", ("BT",))
    monkeypatch.setattr(study, "SEEDS", (13,))
    monkeypatch.setattr(study, "GRADIENT_RATIOS", (0.15,))
    monkeypatch.setattr(study, "TEST_GROUPS_PER_DATASET", 1)
    monkeypatch.setattr(study, "SAMPLE_SIZE", 2)
    source = tmp_path / "deepseek-study"
    relative = Path("inputs/r15/noisy_seed13__BT__msra__N2.jsonl")
    rows = [
        {"tokens": ["Alice"], "ner_tags": ["B-PER"], "dirty_tags": ["O"]},
        {"tokens": ["Paris"], "ner_tags": ["B-LOC"], "dirty_tags": ["B-LOC"]},
    ]
    _write_jsonl(source / relative, rows)
    selection = [{
        "dataset": "msra", "family": "BT", "seed": 13, "row_index": 0,
        "tokens_sha256": "a" * 64, "group_digest": "b" * 64,
    }]
    _write_json(source / "selection.json", selection)
    _write_json(source / "input_manifest.json", {"fixture": True})
    _write_json(source / "manifest.json", {
        "schema_version": study.STUDY_MANIFEST_SCHEMA,
        "selection_sha256": _sha(source / "selection.json"),
        "input_manifest_sha256": _sha(source / "input_manifest.json"),
        "input_files_sha256": {relative.as_posix(): _sha(source / relative)},
    })
    monkeypatch.setattr(study, "FROZEN_SOURCE_MANIFEST_SHA256", _sha(source / "manifest.json"))
    monkeypatch.setattr(study, "FROZEN_INPUT_MANIFEST_SHA256", _sha(source / "input_manifest.json"))
    monkeypatch.setattr(study, "FROZEN_SELECTION_SHA256", _sha(source / "selection.json"))
    return source, relative, rows


def test_prepare_inputs_copies_only_sha_verified_frozen_bank(tmp_path, monkeypatch):
    source, relative, rows = _frozen_fixture(tmp_path, monkeypatch)
    target = tmp_path / "qwen-study"

    manifest = study.prepare_inputs(target, frozen_source_root=source)

    assert study.load_jsonl(target / relative) == rows
    assert manifest["selection_rows"] == 1
    assert manifest["input_files_sha256"] == {relative.as_posix(): _sha(source / relative)}
    assert json.loads((target / "selection.json").read_text(encoding="utf-8"))[0]["seed"] == 13


def test_prepare_inputs_rejects_tampered_frozen_file(tmp_path, monkeypatch):
    source, relative, _rows = _frozen_fixture(tmp_path, monkeypatch)
    with (source / relative).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="SHA mismatch"):
        study.prepare_inputs(tmp_path / "qwen-study", frozen_source_root=source)


def test_every_phase_rehashes_prepared_frozen_inputs(tmp_path, monkeypatch):
    source, relative, _rows = _frozen_fixture(tmp_path, monkeypatch)
    target = tmp_path / "qwen-study"
    study.prepare_inputs(target, frozen_source_root=source)

    assert runner.verify_frozen_inputs(target) == {
        relative.as_posix(): _sha(source / relative),
    }
    with (target / relative).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.verify_frozen_inputs(target)


def test_prepared_manifest_cannot_redeclare_a_tampered_input(tmp_path, monkeypatch):
    source, relative, _rows = _frozen_fixture(tmp_path, monkeypatch)
    target = tmp_path / "qwen-study"
    study.prepare_inputs(target, frozen_source_root=source)

    with (target / relative).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    local_manifest_path = target / "input_manifest.json"
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    local_manifest["input_files_sha256"][relative.as_posix()] = _sha(target / relative)
    _write_json(local_manifest_path, local_manifest)

    with pytest.raises(ValueError, match="provenance is invalid"):
        runner.verify_frozen_inputs(target)


def test_qwen_manifest_is_no_thinking_and_secret_free():
    manifest = study.build_study_manifest(
        study_tag="qwen32b-contextual-nothink-studies-v1",
        source_tag=study.SOURCE_TAG,
        git_sha="a" * 40,
        source_files={"run": "b" * 64},
        provider={"provider": "vllm", "api_key": "must-not-leak"},
        protocol={"failure_policy": "abort"},
    )

    assert manifest["backbone"]["model"] == "Qwen/Qwen3-32B-AWQ"
    assert manifest["backbone"]["enable_thinking"] is False
    assert "must-not-leak" not in json.dumps(manifest)


def test_qwen_runner_protocol_matches_deepseek_study_matrix():
    protocol = runner._protocol()

    assert protocol["ablation_variants"] == [
        "full", "minus_contextual", "minus_gate",
        "minus_reviewer_weighting", "minus_verifier",
        "minus_atf_deanchor",
    ]
    assert protocol["gradient_ratios"] == [0.05, 0.15, 0.25, 0.35, 0.45]
    assert protocol["sample_size"] == 200
    assert protocol["primary_ablation_rows"] == 450
    assert protocol["gradient_rows"] == 2250
    assert protocol["thinking"] is False
    assert protocol["no_dfa"] is True
    assert protocol["no_exception_fallback"] is True


def test_decorate_preserves_coordinates_and_locked_selection():
    row = {
        "tokens": ["Alice"], "gold_tags": ["B-PER"], "pred_tags": ["B-PER"],
        "terminal_anchor_tags": ["B-PER"], "terminal_model_hash": "a" * 64,
        "terminal_used_anchor": True, "terminal_predicted_gain": 0.0,
        "terminal_fallback_count": 0, "fallback_used": False,
    }
    selection = [{
        "dataset": "msra", "family": "BT", "seed": 42, "row_index": 7,
        "group_digest": "b" * 64,
    }]

    decorated = runner._decorate(
        [row], kind="noise-gradient", variant="full", dataset="msra",
        noise="BT", ratio=0.25, coordinates=[{"seed": 42, "row_index": 7}],
        origin="provider_cache_replay", selection=selection,
    )

    assert decorated[0]["study_seed"] == 42
    assert decorated[0]["study_row_index"] == 7
    assert decorated[0]["study_group_digest"] == "b" * 64
    assert decorated[0]["study_ratio"] == 0.25


def _auditable_row():
    from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

    def evidence(stage, status):
        return {
            "stage": stage, "status": status, "provider": "vllm",
            "model": "Qwen/Qwen3-32B-AWQ",
            "served_model": (
                "Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8"
            ),
            "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
            "structured_api": "chat-completions-json-schema",
            "enable_thinking": False, "thinking_mode": "nothink",
        }

    return {
        "tokens": ["Alice"], "gold_tags": ["B-PER"], "pred_tags": ["B-PER"],
        "terminal_anchor_tags": ["B-PER"],
        "terminal_model_hash": LOCKED_DECODER_MODEL_HASH,
        "terminal_used_anchor": True, "terminal_predicted_gain": 0.0,
        "terminal_fallback_count": 0, "fallback_used": False,
        "provider_cache_sha256": "c" * 64,
        "candidate_paths": [["B-PER"], ["O"]],
        "rag_weights": [0.8, 0.2], "confidence": [0.8],
        "provider_metadata": {
            "coder": [evidence("coder", "live")],
            "reviewer": [evidence("reviewer", "live")],
            "verifier": [evidence("verifier", "live")],
        },
    }


def _audit_binding(row, noisy, *, tag=None):
    row["input_digest"] = runner.run_multiseed._provider_input_digest(noisy)
    record = {
        "input_digest": row["input_digest"],
        "anchor_tags": row["terminal_anchor_tags"],
        "candidate_paths": row["candidate_paths"],
        "rag_weights": row["rag_weights"],
        "confidence": row["confidence"],
        "provider_metadata": row["provider_metadata"],
    }
    return {
        row["provider_cache_sha256"]: {
            "tag": tag or runner.SOURCE_CACHE_TAG,
            "records": [record],
        },
    }


def _audit_terminal():
    from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

    return SimpleNamespace(
        model_hash=LOCKED_DECODER_MODEL_HASH,
        fallback_count=0,
        sentence_deer_stats=lambda _tokens: {},
        decode=lambda **kwargs: SimpleNamespace(
            tags=tuple(kwargs["anchor_tags"]),
            model_hash=LOCKED_DECODER_MODEL_HASH,
            used_anchor=True,
            predicted_gain=0.0,
        ),
    )


def test_audit_row_requires_pinned_nothinking_evidence_and_cache_sha():
    row = _auditable_row()
    noisy = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    registry = _audit_binding(row, noisy)
    kwargs = dict(
        kind="ablation", variant="full", noise="BT", ratio=0.15,
        terminal=_audit_terminal(),
    )
    runner._audit_prediction_row(row, "msra", noisy, registry, **kwargs)

    row["provider_metadata"]["coder"][0]["enable_thinking"] = True
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._audit_prediction_row(row, "msra", noisy, registry, **kwargs)


def test_audit_row_rejects_source_only_fields_and_unknown_cache():
    row = _auditable_row()
    noisy = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    registry = _audit_binding(row, noisy)
    kwargs = dict(
        kind="ablation", variant="full", noise="BT", ratio=0.15,
        terminal=_audit_terminal(),
    )
    row["dirty_tags"] = ["O"]
    with pytest.raises(ValueError, match="cache SHA"):
        runner._audit_prediction_row(row, "msra", noisy, {}, **kwargs)
    with pytest.raises(ValueError, match="source-only"):
        runner._audit_prediction_row(row, "msra", noisy, registry, **kwargs)


def test_audit_row_rejects_wrong_cache_tag_and_modified_candidate_evidence():
    row = _auditable_row()
    noisy = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    registry = _audit_binding(row, noisy, tag=runner.REVIEWER_CACHE_TAG)
    kwargs = dict(
        kind="ablation", variant="full", noise="BT", ratio=0.15,
        terminal=_audit_terminal(),
    )
    with pytest.raises(ValueError, match="wrong provider-cache tag"):
        runner._audit_prediction_row(row, "msra", noisy, registry, **kwargs)

    registry = _audit_binding(row, noisy)
    row["candidate_paths"] = [["O"], ["B-PER"]]
    with pytest.raises(ValueError, match="exactly present"):
        runner._audit_prediction_row(row, "msra", noisy, registry, **kwargs)


def test_audit_recomputes_minus_contextual_semantics():
    row = _auditable_row()
    noisy = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    row.update({
        "pred_tags": ["O"], "terminal_used_anchor": False,
        "terminal_predicted_gain": 1.0,
    })
    registry = _audit_binding(row, noisy)

    with pytest.raises(ValueError, match="minus_contextual terminal semantics"):
        runner._audit_prediction_row(
            row, "msra", noisy, registry, kind="ablation",
            variant="minus_contextual", noise="BT", ratio=0.15,
            terminal=_audit_terminal(),
        )


def test_source_cache_routes_frozen_rows_through_explicit_study_contract(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(study, "DATASETS", ("msra",))
    monkeypatch.setattr(study, "NOISE_TYPES", ("BT",))
    monkeypatch.setattr(study, "SEEDS", (13,))
    monkeypatch.setattr(study, "SAMPLE_SIZE", 1)
    root = tmp_path / "study"
    input_path = study.study_input_path(root, "msra", "BT", 13, 0.15)
    _write_jsonl(input_path, [{
        "tokens": ["Alice"], "ner_tags": ["B-PER"], "dirty_tags": ["O"],
    }])
    monkeypatch.setattr(runner, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "ensure_manifest", lambda *_a, **_k: {})
    monkeypatch.setattr(runner.run_multiseed, "_import_pipeline", lambda _dummy: (object(), {}))

    class Factory:
        closed = False

        def close(self):
            self.closed = True

    factory = Factory()
    monkeypatch.setattr(runner, "_live_factory", lambda _tag: factory)
    observed = []

    async def fake_cache(*args, **kwargs):
        observed.append((args, kwargs))
        return {"row_count": 1}

    monkeypatch.setattr(runner.run_multiseed, "_run_provider_cache_cell", fake_cache)

    runner.run_source_cache(root, tmp_path / "cache", max_concurrency=2)

    assert factory.closed is True
    assert len(observed) == 1
    args, kwargs = observed[0]
    assert args[2:6] == ("msra", "BT", 13, 1)
    assert kwargs["explicit_source_rows"][0]["tokens"] == ["Alice"]
    assert kwargs["study_identity"]["variant"] == "full"
    assert kwargs["study_identity"]["coordinates_sha256"] == runner._sha256_json([
        {"seed": 13, "row_index": 0},
    ])


def test_reviewer_weighting_row_cache_resumes_without_repeating_paid_rows(
    tmp_path, monkeypatch,
):
    import multi_agent_v2
    from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

    monkeypatch.setattr(study, "SAMPLE_SIZE", 2)
    root, cache_root = tmp_path / "study", tmp_path / "cache"

    def live(stage):
        return {
            "stage": stage, "status": "live", "provider": "vllm",
            "model": "Qwen/Qwen3-32B-AWQ",
            "served_model": (
                "Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8"
            ),
            "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
            "structured_api": "chat-completions-json-schema",
            "enable_thinking": False, "thinking_mode": "nothink",
            "response_model": (
                "Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8"
            ),
            "response_status": "completed", "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    source_rows = []
    noisy_rows = []
    for index in range(2):
        source_rows.append({
            "tokens": [f"name-{index}"], "gold_tags": ["B-PER"],
            "pred_tags": ["B-PER"], "candidate_paths": [["B-PER"], ["B-ORG"]],
            "rag_weights": [0.8, 0.2], "confidence": [0.8],
            "terminal_anchor_tags": ["B-PER"],
            "terminal_model_hash": LOCKED_DECODER_MODEL_HASH,
            "terminal_used_anchor": True, "terminal_predicted_gain": 0.0,
            "terminal_fallback_count": 0, "provider_cache_sha256": "d" * 64,
            "provider_metadata": {
                "coder": [live("coder")], "reviewer": [live("reviewer")],
                "verifier": [live("verifier")],
            },
            "fallback_used": False,
        })
        noisy_rows.append({
            "tokens": [f"name-{index}"], "ner_tags": ["B-PER"],
            "dirty_tags": ["B-ORG"],
        })
    study._write_jsonl_atomic(
        study.study_prediction_path(root, "ablation", "full", "msra", "ATF", 0.15, 13),
        source_rows,
    )
    study._write_jsonl_atomic(
        study.study_input_path(root, "msra", "ATF", 13, 0.15), noisy_rows,
    )

    provider_identity = {
        "provider": "vllm", "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": (
            "Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8"
        ),
        "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": False, "thinking_mode": "nothink",
        "timeout_seconds": 600.0,
    }

    class Adapter:
        def provider_metadata(self):
            return dict(provider_identity)

        def structured_requester(self, *_args, **_kwargs):
            raise AssertionError("fake verifier should intercept the provider call")

    class Factory:
        def __call__(self):
            return Adapter()

    terminal = SimpleNamespace(
        model_hash=LOCKED_DECODER_MODEL_HASH,
        fallback_count=0,
        sentence_deer_stats=lambda _tokens: {},
        decode=lambda **kwargs: SimpleNamespace(
            tags=tuple(kwargs["anchor_tags"]),
            model_hash=LOCKED_DECODER_MODEL_HASH,
            used_anchor=True, predicted_gain=0.0,
        ),
    )
    monkeypatch.setattr(multi_agent_v2, "_init_deer", lambda *_a, **_k: None)
    monkeypatch.setattr(multi_agent_v2, "_base_decode", lambda *_a, **_k: ["B-PER"])
    monkeypatch.setattr(multi_agent_v2, "legalize_noise_aware", lambda tags, *_a: tags)
    monkeypatch.setattr(multi_agent_v2, "_is_type_contested", lambda _paths: True)
    monkeypatch.setattr(
        multi_agent_v2, "_per_position_confidence", lambda *_a, **_k: [1.0],
    )
    calls = []

    async def flaky_verifier(state):
        calls.append(state["tokens"][0])
        if len(calls) == 2:
            raise RuntimeError("second row failed")
        return {
            "current_tags": ["B-PER"],
            "provider_metadata": {
                "coder": state["provider_metadata"]["coder"],
                "reviewer": state["provider_metadata"]["reviewer"],
                "verifier": [live("verifier")],
            },
        }

    monkeypatch.setattr(multi_agent_v2, "verifier_node", flaky_verifier)
    common = dict(
        root=root, cache_root=cache_root, factory=Factory(), terminal=terminal,
        dataset="msra", noise="ATF", seed=13, selection=[],
        max_concurrency=1, git_sha="a" * 40,
    )
    with pytest.raises(RuntimeError, match="second row failed"):
        asyncio.run(runner._reviewer_weighting_cell(**common))
    fragment = (
        cache_root / runner.REVIEWER_CACHE_TAG / "rows" / "msra" / "ATF"
        / "seed13" / "row000.jsonl"
    )
    assert fragment.is_file()
    progress = json.loads((fragment.parent / "progress.json").read_text(encoding="utf-8"))
    assert progress["rows"]["0"] == runner._sha256_file(fragment)
    assert not study.study_prediction_path(
        root, "ablation", "minus_reviewer_weighting", "msra", "ATF", 0.15, 13,
    ).exists()

    original_fragment = fragment.read_bytes()
    with fragment.open("ab") as handle:
        handle.write(b" \n")
    with pytest.raises(ValueError, match="progress index"):
        asyncio.run(runner._reviewer_weighting_cell(**common))
    fragment.write_bytes(original_fragment)

    result = asyncio.run(runner._reviewer_weighting_cell(**common))

    assert calls == ["name-0", "name-1", "name-1"]
    assert result.is_file()
    assert runner._sha256_file(
        runner.run_multiseed._provider_cache_cell_path(
            cache_root, runner.REVIEWER_CACHE_TAG, runner.CONFIG_NAME,
            "msra", "ATF", 13,
        )
    ) in {
        row["provider_cache_sha256"] for row in study.load_jsonl(result)
    }


def test_gradient_selection_rejects_duplicate_or_substituted_coordinates():
    tokens = ["Alice"]
    token_sha = hashlib.sha256(
        study._canonical_tokens(tokens).encode("utf-8")
    ).hexdigest()
    expected = [
        {"seed": 13, "row_index": 1, "group_digest": "a" * 64,
         "tokens_sha256": token_sha},
        {"seed": 42, "row_index": 2, "group_digest": "b" * 64,
         "tokens_sha256": token_sha},
    ]
    rows = [
        {"tokens": tokens, "study_seed": 13, "study_row_index": 1,
         "study_group_digest": "a" * 64},
        {"tokens": tokens, "study_seed": 13, "study_row_index": 1,
         "study_group_digest": "a" * 64},
    ]

    with pytest.raises(ValueError, match="coordinates"):
        runner._validate_gradient_selection(rows, expected)


def test_cache_registry_rehashes_indexed_cells_before_audit(tmp_path, monkeypatch):
    from contextual_lattice_runtime import LOCKED_BUNDLE_MANIFEST_HASH

    cache_root = tmp_path / "cache"
    root = tmp_path / "study"
    tag = runner.SOURCE_CACHE_TAG
    monkeypatch.setattr(runner, "_expected_cache_counts", lambda _root: {tag: 1})
    monkeypatch.setattr(study, "SAMPLE_SIZE", 1)
    row = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    _write_json(root / "selection.json", [])
    input_path = study.study_input_path(root, "msra", "BT", 13, 0.15)
    _write_jsonl(input_path, [{**row, "ner_tags": ["B-PER"]}])
    identity = runner._study_identity(
        root, kind="ablation", variant="full", dataset="msra", noise="BT",
        seed=13, ratio=0.15, coordinates=[{"seed": 13, "row_index": 0}],
    )
    manifest = runner.run_multiseed._provider_cache_manifest(
        [row], runner.run_multiseed.CONFIGURATIONS[runner.CONFIG_NAME],
        dataset="msra", noise="BT", seed=13, git_sha="a" * 40,
        bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH, enable_thinking=False,
        study_identity=identity,
    )
    prediction = _auditable_row()
    record = {
        "row_index": 0,
        "input_digest": runner.run_multiseed._provider_input_digest(row),
        "anchor_tags": ["B-PER"],
        "candidate_paths": prediction["candidate_paths"],
        "rag_weights": prediction["rag_weights"],
        "confidence": prediction["confidence"],
        "provider_metadata": prediction["provider_metadata"],
        "fallback_used": False,
    }
    cell_path = runner.run_multiseed._provider_cache_cell_path(
        cache_root, tag, runner.CONFIG_NAME, "msra", "BT", 13,
    )
    sha = runner.run_multiseed.write_provider_cell(cell_path, [record], manifest)
    runner.run_multiseed._update_provider_cache_index(
        cache_root, tag,
        key=runner.run_multiseed._provider_cache_cell_key(
            runner.CONFIG_NAME, "msra", "BT", 13,
        ),
        cell={
            "path": str(cell_path), "sha256": sha, "row_count": 1,
            "git_sha": "a" * 40, "model_revision": runner.QWEN_REVISION,
            "provider_fingerprint": "e" * 64,
        },
    )

    assert runner._validated_cache_registry(root, cache_root)[sha]["records"] == [record]

    wrong_manifest = json.loads(json.dumps(manifest))
    wrong_manifest["configuration"]["study_pipeline_config"]["deanchor_atf"] = False
    wrong_sha = runner.run_multiseed.write_provider_cell(
        cell_path, [record], wrong_manifest,
    )
    runner.run_multiseed._update_provider_cache_index(
        cache_root, tag,
        key=runner.run_multiseed._provider_cache_cell_key(
            runner.CONFIG_NAME, "msra", "BT", 13,
        ),
        cell={
            "path": str(cell_path), "sha256": wrong_sha, "row_count": 1,
            "git_sha": "a" * 40, "model_revision": runner.QWEN_REVISION,
            "provider_fingerprint": "e" * 64,
        },
    )
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        runner._validated_cache_registry(root, cache_root)

    sha = runner.run_multiseed.write_provider_cell(cell_path, [record], manifest)
    runner.run_multiseed._update_provider_cache_index(
        cache_root, tag,
        key=runner.run_multiseed._provider_cache_cell_key(
            runner.CONFIG_NAME, "msra", "BT", 13,
        ),
        cell={
            "path": str(cell_path), "sha256": sha, "row_count": 1,
            "git_sha": "a" * 40, "model_revision": runner.QWEN_REVISION,
            "provider_fingerprint": "e" * 64,
        },
    )
    with cell_path.open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner._validated_cache_registry(root, cache_root)


def test_reviewer_cache_must_reuse_exact_source_coder_evidence(tmp_path, monkeypatch):
    from contextual_lattice_runtime import LOCKED_BUNDLE_MANIFEST_HASH

    root, cache_root = tmp_path / "study", tmp_path / "cache"
    monkeypatch.setattr(study, "SAMPLE_SIZE", 1)
    monkeypatch.setattr(runner, "_expected_cache_counts", lambda _root: {
        runner.SOURCE_CACHE_TAG: 1, runner.REVIEWER_CACHE_TAG: 1,
    })
    _write_json(root / "selection.json", [])
    noisy = {"tokens": ["Alice"], "dirty_tags": ["O"]}
    _write_jsonl(
        study.study_input_path(root, "msra", "BT", 13, 0.15),
        [{**noisy, "ner_tags": ["B-PER"]}],
    )
    prediction = _auditable_row()
    source_record = {
        "row_index": 0,
        "input_digest": runner.run_multiseed._provider_input_digest(noisy),
        "anchor_tags": ["B-PER"],
        "candidate_paths": prediction["candidate_paths"],
        "rag_weights": prediction["rag_weights"],
        "confidence": prediction["confidence"],
        "provider_metadata": prediction["provider_metadata"],
        "fallback_used": False,
    }

    def publish(tag, identity, record):
        manifest = runner.run_multiseed._provider_cache_manifest(
            [noisy], runner.run_multiseed.CONFIGURATIONS[runner.CONFIG_NAME],
            dataset="msra", noise="BT", seed=13, git_sha="a" * 40,
            bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH, enable_thinking=False,
            study_identity=identity,
        )
        path = runner.run_multiseed._provider_cache_cell_path(
            cache_root, tag, runner.CONFIG_NAME, "msra", "BT", 13,
        )
        sha = runner.run_multiseed.write_provider_cell(path, [record], manifest)
        runner.run_multiseed._update_provider_cache_index(
            cache_root, tag,
            key=runner.run_multiseed._provider_cache_cell_key(
                runner.CONFIG_NAME, "msra", "BT", 13,
            ),
            cell={
                "path": str(path), "sha256": sha, "row_count": 1,
                "git_sha": "a" * 40, "model_revision": runner.QWEN_REVISION,
                "provider_fingerprint": "e" * 64,
            },
        )
        return sha

    coordinates = [{"seed": 13, "row_index": 0}]
    source_identity = runner._study_identity(
        root, kind="ablation", variant="full", dataset="msra", noise="BT",
        seed=13, ratio=0.15, coordinates=coordinates,
    )
    source_sha = publish(runner.SOURCE_CACHE_TAG, source_identity, source_record)
    reviewer_identity = runner._study_identity(
        root, kind="ablation", variant="minus_reviewer_weighting",
        dataset="msra", noise="BT", seed=13, ratio=0.15,
        coordinates=coordinates,
    )
    reviewer_identity.update({
        "source_provider_cache_sha256": source_sha,
        "reviewer_weighting": "uniform", "reviewer_stage": "disabled",
        "verifier_policy": "type_contested_only",
    })
    reviewer_record = json.loads(json.dumps(source_record))
    reviewer_record["rag_weights"] = [0.5, 0.5]
    reviewer_record["provider_metadata"]["reviewer"][0]["status"] = "disabled"
    publish(runner.REVIEWER_CACHE_TAG, reviewer_identity, reviewer_record)

    assert len(runner._validated_cache_registry(root, cache_root)) == 2

    reviewer_record["provider_metadata"]["coder"][0]["request_id"] = "new-call"
    publish(runner.REVIEWER_CACHE_TAG, reviewer_identity, reviewer_record)
    with pytest.raises(ValueError, match="preserve source coder evidence"):
        runner._validated_cache_registry(root, cache_root)
