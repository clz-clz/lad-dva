import hashlib
import json
from pathlib import Path
import asyncio
import concurrent.futures
import threading
import time

import pytest

import official_provider_cache
import run_multiseed
from official_contract import (
    VerifierSemanticRetryExhausted,
    VerifierSemanticRetryInterrupted,
    validate_verifier_semantic_retry_evidence,
)


_MODEL = "Qwen/Qwen3-32B-AWQ"
_REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
_SERVED_MODEL = f"{_MODEL}@{_REVISION}"


def _evidence(stage, status):
    evidence = {
        "stage": stage,
        "status": status,
        "provider": "vllm",
        "model": _MODEL,
        "served_model": _SERVED_MODEL,
        "revision": _REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": False,
        "thinking_mode": "nothink",
    }
    if status == "live":
        evidence.update({
            "response_model": _SERVED_MODEL,
            "response_status": "completed",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
    return evidence


def _record(index=0, digest=None):
    return {
        "row_index": index,
        "input_digest": digest or hashlib.sha256(f"row-{index}".encode()).hexdigest(),
        "anchor_tags": ["B-PER", "O"],
        "candidate_paths": [["B-PER", "O"]],
        "rag_weights": [1.0],
        "confidence": [1.0, 1.0],
        "provider_metadata": {
            "coder": [_evidence("coder", "live")],
            "reviewer": [_evidence("reviewer", "skipped_identical")],
            "verifier": [_evidence("verifier", "skipped_uncontested")],
        },
        "fallback_used": False,
    }


def _manifest(*records):
    return {
        "schema": "selectdenoise-contextual-provider-cache-v1",
        "source_digests": {str(row["row_index"]): row["input_digest"] for row in records},
        "git_sha": "a" * 40,
        "model_revision": _REVISION,
        "bundle_hash": "b" * 64,
        "configuration": {
            "terminal_decoder": "contextual-lattice-v1",
            "enable_thinking": False,
            "thinking_mode": "nothink",
        },
    }


def test_write_then_read_returns_exact_rows_and_cell_digest(tmp_path):
    records = [_record(0), _record(1)]
    cell = tmp_path / "provider.jsonl"

    digest = official_provider_cache.write_provider_cell(cell, records, _manifest(*records))

    assert digest == hashlib.sha256(cell.read_bytes()).hexdigest()
    assert official_provider_cache.read_provider_cell(cell, digest, 2) == records


def test_atomic_write_removes_temporary_cell_when_publish_fails(tmp_path, monkeypatch):
    record = _record()
    cell = tmp_path / "provider.jsonl"
    temporary = cell.with_suffix(".jsonl.tmp")
    original_replace = Path.replace

    def fail_publish(self, target):
        if self == temporary:
            raise OSError("publish failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        official_provider_cache.write_provider_cell(cell, [record], _manifest(record))

    assert not temporary.exists()
    assert not cell.exists()


@pytest.mark.parametrize("expected_rows", [0, 2])
def test_read_rejects_an_incorrect_exact_row_count(tmp_path, expected_rows):
    record = _record()
    cell = tmp_path / "provider.jsonl"
    digest = official_provider_cache.write_provider_cell(cell, [record], _manifest(record))

    with pytest.raises(ValueError, match="row count"):
        official_provider_cache.read_provider_cell(cell, digest, expected_rows)


def test_read_rejects_a_sha_mismatch_before_returning_rows(tmp_path):
    record = _record()
    cell = tmp_path / "provider.jsonl"
    official_provider_cache.write_provider_cell(cell, [record], _manifest(record))

    with pytest.raises(ValueError, match="SHA-256"):
        official_provider_cache.read_provider_cell(cell, "0" * 64, 1)


def test_write_rejects_duplicate_row_indices(tmp_path):
    records = [_record(0), _record(0)]

    with pytest.raises(ValueError, match="duplicate row_index"):
        official_provider_cache.write_provider_cell(
            tmp_path / "provider.jsonl", records, _manifest(*records)
        )


def test_write_rejects_source_digest_mismatch(tmp_path):
    record = _record()
    manifest = _manifest(record)
    manifest["source_digests"] = {"0": "f" * 64}

    with pytest.raises(ValueError, match="source digest"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], manifest)


def test_write_rejects_gold_labels_nested_in_candidate_paths(tmp_path):
    record = _record()
    record["candidate_paths"] = [{"gold_tags": ["B-PER", "O"]}]

    with pytest.raises(ValueError, match="gold labels"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


def test_write_rejects_gold_labels_nested_in_manifest(tmp_path):
    record = _record()
    manifest = _manifest(record)
    manifest["launch"] = {"provider_evidence": [{"ner_tags": ["B-PER", "O"]}]}

    with pytest.raises(ValueError, match="gold labels"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], manifest)


def test_write_rejects_gold_labels_nested_in_provider_evidence(tmp_path):
    record = _record()
    record["provider_metadata"]["coder"][0]["details"] = {
        "response": {"gold_tags": ["B-PER", "O"]}
    }

    with pytest.raises(ValueError, match="gold labels"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


def test_write_rejects_gold_labels_key_nested_in_candidate_paths(tmp_path):
    record = _record()
    record["candidate_paths"] = [{"details": {"gold_labels": ["B-PER", "O"]}}]

    with pytest.raises(ValueError, match="gold labels"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


@pytest.mark.parametrize("identity_field", ["git_sha", "model_revision", "bundle_hash", "configuration"])
def test_write_requires_replay_identity_fields(tmp_path, identity_field):
    record = _record()
    manifest = _manifest(record)
    del manifest[identity_field]

    with pytest.raises(ValueError, match=identity_field):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], manifest)


def test_write_rejects_noncanonical_replay_identity_values(tmp_path):
    record = _record()
    manifest = _manifest(record)
    manifest["git_sha"] = "not-a-sha"

    with pytest.raises(ValueError, match="git_sha"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], manifest)


@pytest.mark.parametrize(
    ("stage", "field", "value", "match"),
    [
        ("coder", "stage", "reviewer", "stage"),
        ("reviewer", "served_model", "other@revision", "served identity"),
        ("verifier", "revision", "a" * 40, "served identity"),
    ],
)
def test_write_rejects_unbound_or_unpinned_provider_evidence(tmp_path, stage, field, value, match):
    record = _record()
    record["provider_metadata"][stage][0][field] = value

    with pytest.raises(ValueError, match=match):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


def test_write_rejects_malformed_provider_metadata(tmp_path):
    record = _record()
    record["provider_metadata"] = {"coder": []}

    with pytest.raises(ValueError, match="provider_metadata"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


def test_write_rejects_incomplete_stage_evidence(tmp_path):
    record = _record()
    del record["provider_metadata"]["verifier"]

    with pytest.raises(ValueError, match="provider_metadata"):
        official_provider_cache.write_provider_cell(tmp_path / "provider.jsonl", [record], _manifest(record))


def test_failed_cell_is_not_resumable_as_a_completed_cell(tmp_path):
    record = _record()
    cell = tmp_path / "provider.jsonl"
    cell.with_suffix(".jsonl.tmp").write_text("partial\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        official_provider_cache.read_provider_cell(cell, "0" * 64, 1)


@pytest.mark.parametrize("identity_field", ["git_sha", "model_revision", "bundle_hash", "configuration"])
def test_contextual_replay_rejects_provider_cache_identity_mismatches(identity_field):
    """The future replay phase must bind every cache to its launch identity."""
    record = _record()
    expected = _manifest(record)
    cached = dict(expected)
    cached[identity_field] = (
        {"terminal_decoder": "other"}
        if identity_field == "configuration"
        else "c" * len(str(expected[identity_field]))
    )

    with pytest.raises(ValueError, match=identity_field):
        run_multiseed._validate_replay_cache_identity(cached, expected)


def test_contextual_cache_thinking_modes_are_not_interchangeable():
    record = _record()
    expected = _manifest(record)
    actual = json.loads(json.dumps(expected))
    actual["configuration"]["enable_thinking"] = True
    actual["configuration"]["thinking_mode"] = "thinking"

    with pytest.raises(ValueError, match="configuration"):
        run_multiseed._validate_replay_cache_identity(actual, expected)


def test_provider_cache_manifest_records_bounded_verifier_semantic_retry_policy():
    row = {"tokens": ["Acme"], "dirty_tags": ["B-ORG"]}
    manifest = run_multiseed._provider_cache_manifest(
        [row], run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        dataset="conll2003", noise="ATF", seed=42, git_sha="a" * 40,
        bundle_hash="b" * 64, enable_thinking=False,
    )

    assert manifest["verifier_semantic_retry"] == {
        "max_retries": 2,
        "max_attempts": 3,
        "retryable_error": "illegal-iob2-transition-v1",
        "feedback": "first-invalid-transition-v1",
        "no_dfa": True,
        "no_fallback": True,
    }


def test_replay_requires_retry_policy_for_current_producer_but_accepts_legacy_compatible():
    row = {"tokens": ["Acme"], "dirty_tags": ["B-ORG"]}
    expected = run_multiseed._provider_cache_manifest(
        [row], run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        dataset="conll2003", noise="ATF", seed=42, git_sha="a" * 40,
        bundle_hash="b" * 64, enable_thinking=False,
    )
    missing_policy = json.loads(json.dumps(expected))
    missing_policy.pop("verifier_semantic_retry")

    with pytest.raises(ValueError, match="semantic retry policy"):
        run_multiseed._validate_replay_cache_identity(missing_policy, expected)

    missing_policy["git_sha"] = "c" * 40
    run_multiseed._validate_replay_cache_identity(
        missing_policy, expected, compatible_git_shas=("c" * 40,),
    )

    same_legacy_producer = json.loads(json.dumps(expected))
    same_legacy_producer.pop("verifier_semantic_retry")
    run_multiseed._validate_replay_cache_identity(
        same_legacy_producer, expected, compatible_git_shas=("a" * 40,),
    )


def test_contextual_provider_evidence_accepts_auditable_verifier_semantic_retry():
    rejected = _evidence("verifier", "live")
    rejected.update({
        "semantic_attempt": 1,
        "semantic_outcome": "rejected_illegal_iob2",
        "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
        "rejected_tags": ["O", "I-ORG"],
        "response_sha256": "d" * 64,
    })
    accepted = _evidence("verifier", "live")
    accepted.update({"semantic_attempt": 2, "semantic_outcome": "accepted"})
    evidence = {
        "coder": [_evidence("coder", "live") for _ in range(5)],
        "reviewer": [_evidence("reviewer", "live")],
        "verifier": [rejected, accepted],
    }
    identity = {
        "provider": "vllm", "model": _MODEL, "served_model": _SERVED_MODEL,
        "revision": _REVISION, "structured_api": "chat-completions-json-schema",
        "enable_thinking": False, "thinking_mode": "nothink",
    }

    rejected["response_sha256"] = run_multiseed._sha256_json({
        "tags": rejected["rejected_tags"],
    })
    run_multiseed._validate_contextual_provider_evidence(evidence, identity, "ATF")


@pytest.mark.parametrize(
    "mutation",
    [
        {"illegal_transition": {"index": 1, "previous": "O", "current": "O"}},
        {"illegal_transition": {"index": 99, "previous": "O", "current": "I-ORG"}},
        {"response_sha256": "0" * 64},
    ],
)
def test_verifier_retry_provenance_rejects_false_or_unbound_illegal_edges(mutation):
    rejected = _evidence("verifier", "live")
    rejected.update({
        "semantic_attempt": 1,
        "semantic_outcome": "rejected_illegal_iob2",
        "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
        "rejected_tags": ["O", "I-ORG"],
        "response_sha256": run_multiseed._sha256_json({"tags": ["O", "I-ORG"]}),
    })
    rejected.update(mutation)
    accepted = _evidence("verifier", "live")
    accepted.update({"semantic_attempt": 2, "semantic_outcome": "accepted"})

    with pytest.raises(ValueError):
        validate_verifier_semantic_retry_evidence([rejected, accepted])


def test_semantic_retry_failure_provenance_is_persisted_without_target_labels(tmp_path):
    attempts = []
    for number in range(1, 4):
        record = _evidence("verifier", "live")
        record.update({
            "semantic_attempt": number,
            "semantic_outcome": "rejected_illegal_iob2",
            "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
            "rejected_tags": ["O", "I-ORG"],
            "response_sha256": run_multiseed._sha256_json({"tags": ["O", "I-ORG"]}),
        })
        attempts.append(record)
    error = VerifierSemanticRetryExhausted(attempts)

    path = run_multiseed._write_verifier_semantic_failure_provenance(
        cache_root=tmp_path,
        cache_tag="qwen32b-contextual-nothink-v1",
        config_name="selectdenoise_contextual_lattice",
        dataset="fewnerd",
        noise="ATF",
        seed=42,
        row_index=137,
        input_digest="e" * 64,
        git_sha="a" * 40,
        attempts=error.records,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "selectdenoise-verifier-semantic-failure-v1"
    assert payload["row_index"] == 137
    assert len(payload["attempts"]) == 3
    assert "gold_tags" not in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp"))


def test_semantic_failure_provenance_whitelists_attempt_metadata(tmp_path):
    attempts = []
    for number in range(1, 4):
        record = _evidence("verifier", "live")
        record.update({
            "semantic_attempt": number,
            "semantic_outcome": "rejected_illegal_iob2",
            "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
            "rejected_tags": ["O", "I-ORG"],
            "response_sha256": run_multiseed._sha256_json({"tags": ["O", "I-ORG"]}),
            "api_key": "must-not-be-written",
            "gold_tags": ["B-ORG", "I-ORG"],
        })
        record["usage"] = {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "secret": "must-not-be-written",
        }
        attempts.append(record)

    path = run_multiseed._write_verifier_semantic_failure_provenance(
        cache_root=tmp_path,
        cache_tag="qwen32b-contextual-nothink-v1",
        config_name="selectdenoise_contextual_lattice",
        dataset="fewnerd",
        noise="ATF",
        seed=42,
        row_index=137,
        input_digest="e" * 64,
        git_sha="a" * 40,
        attempts=attempts,
    )

    text = path.read_text(encoding="utf-8")
    assert "must-not-be-written" not in text
    assert "api_key" not in text
    assert "gold_tags" not in text
    assert "secret" not in text
    payload = json.loads(text)
    assert payload["attempts"][0]["usage"] == {
        "prompt_tokens": 7, "completion_tokens": 2,
    }


def test_contextual_cache_allows_explicitly_compatible_producer_with_new_request_limits():
    record = _record()
    expected = _manifest(record)
    expected["request_limits"] = {
        "provider_timeout_seconds": 600.0,
        "runner_timeout_seconds": 7200.0,
        "max_concurrency": 32,
    }
    cached = json.loads(json.dumps(expected))
    cached["git_sha"] = "b" * 40
    cached["request_limits"] = {
        "provider_timeout_seconds": 300.0,
        "runner_timeout_seconds": 3600.0,
        "max_concurrency": 20,
    }

    run_multiseed._validate_replay_cache_identity(
        cached, expected, compatible_git_shas=["b" * 40]
    )


def test_reduced_profile_reuses_only_verified_prefix_of_complete_source_cell(tmp_path):
    config = run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"]
    source_rows = [
        {"tokens": [f"token-{index}"], "dirty_tags": ["O"], "ner_tags": ["O"]}
        for index in range(200)
    ]
    source_records = []
    for index, row in enumerate(source_rows):
        record = _record(index, run_multiseed._provider_input_digest(row))
        record["provider_metadata"]["coder"] = [
            _evidence("coder", "live") for _ in (1, 2, 5)
        ]
        source_records.append(record)
    source_tag = "qwen32b-contextual-nothink-v1"
    target_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    source_cell = run_multiseed._provider_cache_cell_path(
        tmp_path, source_tag, "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    source_manifest = run_multiseed._provider_cache_manifest(
        source_rows, config, dataset="msra", noise="BT", seed=13,
        git_sha="d" * 40, bundle_hash="b" * 64, enable_thinking=False,
    )
    source_sha = official_provider_cache.write_provider_cell(
        source_cell, source_records, source_manifest,
    )
    run_multiseed._update_provider_cache_index(
        tmp_path, source_tag,
        key=run_multiseed._provider_cache_cell_key(
            "selectdenoise_contextual_lattice", "msra", "BT", 13,
        ),
        cell={"path": str(source_cell), "sha256": source_sha, "row_count": 200,
              "git_sha": "d" * 40, "model_revision": _REVISION,
              "provider_fingerprint": "e" * 64},
    )
    target_rows = source_rows[:100]
    target_manifest = run_multiseed._provider_cache_manifest(
        target_rows, config, dataset="msra", noise="BT", seed=13,
        git_sha="f" * 40, bundle_hash="b" * 64, enable_thinking=False,
    )
    identity = {
        "timeout_seconds": 600.0, "provider": "vllm", "model": _MODEL,
        "served_model": _SERVED_MODEL, "revision": _REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": False, "thinking_mode": "nothink",
    }

    cell = run_multiseed._reuse_provider_cache_prefix(
        cache_root=tmp_path, source_tag=source_tag, target_tag=target_tag,
        config_name="selectdenoise_contextual_lattice", dataset="msra", noise="BT",
        seed=13, rows=target_rows, target_manifest=target_manifest,
        provider_identity=identity,
    )

    assert cell["row_count"] == 100
    target_path = Path(cell["path"])
    assert official_provider_cache.read_provider_cell(target_path, cell["sha256"], 100)
    derived_manifest = run_multiseed._read_provider_cache_manifest(target_path)
    assert derived_manifest["derived_from"] == {
        "cache_tag": source_tag,
        "sha256": source_sha,
        "source_row_count": 200,
        "selected_rows": [0, 100],
    }
    assert official_provider_cache.read_provider_cell(source_cell, source_sha, 200)


def _write_reduced_source_for_full_continuation(
    cache_root, rows, *, source_tag="qwen32b-contextual-nothink-s13-n100-v1",
    bundle_hash="b" * 64, enable_thinking=False,
):
    config = run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"]
    source_rows = rows[:100]
    records = []
    for index, row in enumerate(source_rows):
        record = _record(index, run_multiseed._provider_input_digest(row))
        record["provider_metadata"]["coder"] = [
            _evidence("coder", "live") for _ in range(3)
        ]
        if enable_thinking:
            for stage_records in record["provider_metadata"].values():
                for evidence in stage_records:
                    evidence["enable_thinking"] = True
                    evidence["thinking_mode"] = "thinking"
        records.append(record)
    source_manifest = run_multiseed._provider_cache_manifest(
        source_rows, config, dataset="msra", noise="BT", seed=13,
        git_sha="d" * 40, bundle_hash=bundle_hash,
        enable_thinking=enable_thinking,
    )
    source_manifest["request_limits"] = {
        "provider_timeout_seconds": 300.0,
        "runner_timeout_seconds": 3600.0,
        "max_concurrency": 20,
    }
    source_cell = run_multiseed._provider_cache_cell_path(
        cache_root, source_tag, "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    source_sha = official_provider_cache.write_provider_cell(
        source_cell, records, source_manifest,
    )
    run_multiseed._update_provider_cache_index(
        cache_root, source_tag,
        key=run_multiseed._provider_cache_cell_key(
            "selectdenoise_contextual_lattice", "msra", "BT", 13,
        ),
        cell={"path": str(source_cell), "sha256": source_sha, "row_count": 100,
              "git_sha": "d" * 40, "model_revision": _REVISION,
              "provider_fingerprint": "e" * 64},
    )
    return source_cell, source_sha, records


def _continuation_adapter():
    class Adapter:
        def provider_metadata(self):
            return {
                "timeout_seconds": 600.0, "provider": "vllm", "model": _MODEL,
                "served_model": _SERVED_MODEL, "revision": _REVISION,
                "structured_api": "chat-completions-json-schema",
                "enable_thinking": False, "thinking_mode": "nothink",
            }

        def structured_requester(self, stage, payload):
            return {}

    return Adapter()


def test_study_provider_cache_accepts_explicit_frozen_rows_and_binds_identity(
    tmp_path, monkeypatch,
):
    rows = [
        {"tokens": [f"study-{index}"], "dirty_tags": ["O"], "ner_tags": ["O"]}
        for index in range(3)
    ]
    requested = []

    async def pipeline(tokens, dirty_tags, config, dataset_name=None):
        requested.append(tokens[0])
        return {
            "pred_tags": list(dirty_tags),
            "candidate_paths": [list(dirty_tags)] * 3,
            "rag_weights": [1.0, 1.0, 1.0],
            "confidence": [1.0] * len(tokens),
            "provider_metadata": {
                "coder": [_evidence("coder", "live") for _ in range(3)],
                "reviewer": [_evidence("reviewer", "skipped_identical")],
                "verifier": [_evidence("verifier", "skipped_uncontested")],
            },
            "fallback_used": False,
        }

    def canonical_access_is_forbidden(*_args, **_kwargs):
        raise AssertionError("study cache must not read the canonical noisy namespace")

    monkeypatch.setattr(run_multiseed, "_noisy_path", canonical_access_is_forbidden)
    monkeypatch.setattr(run_multiseed, "_load_official_source_rows", canonical_access_is_forbidden)
    study_identity = {
        "schema": "qwen-contextual-studies-v1",
        "kind": "ablation",
        "variant": "full",
        "ratio": 0.15,
        "source_file_sha256": "c" * 64,
        "coordinates_sha256": "d" * 64,
    }
    cache_root = tmp_path / "cache"
    tag = "qwen32b-contextual-studies-nothink-source-r15-v1"
    cell = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, len(rows), (pipeline, {}),
        cache_root=cache_root, cache_tag=tag, max_concurrency=2,
        request_timeout=7200, adapter_factory=_continuation_adapter,
        git_sha="a" * 40, bundle_hash="b" * 64,
        explicit_source_rows=rows, study_identity=study_identity,
    ))

    assert requested == ["study-0", "study-1", "study-2"]
    manifest = run_multiseed._read_provider_cache_manifest(Path(cell["path"]))
    assert manifest["configuration"]["study"] == study_identity
    assert manifest["configuration"]["study_pipeline_config"] == {
        "terminal_decoder": "contextual-lattice-v1",
        "deanchor_atf": True,
        "use_verifier": True,
        "verifier_semantic_max_retries": 2,
    }
    assert cell["row_count"] == 3

    requested.clear()
    resumed = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, len(rows), (pipeline, {}),
        cache_root=cache_root, cache_tag=tag, max_concurrency=32,
        request_timeout=7200, adapter_factory=_continuation_adapter,
        git_sha="a" * 40, bundle_hash="b" * 64,
        explicit_source_rows=rows, study_identity=study_identity,
    ))
    assert resumed["sha256"] == cell["sha256"]
    assert requested == []

    changed_identity = {**study_identity, "variant": "minus_atf_deanchor"}
    with pytest.raises(ValueError, match="configuration identity mismatch"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, len(rows), (pipeline, {}),
            cache_root=cache_root, cache_tag=tag, max_concurrency=2,
            request_timeout=7200, adapter_factory=_continuation_adapter,
            git_sha="a" * 40, bundle_hash="b" * 64,
            explicit_source_rows=rows, study_identity=changed_identity,
        ))


def test_study_provider_cache_rejects_reuse_and_mismatched_size(tmp_path):
    rows = [{"tokens": ["x"], "dirty_tags": ["O"], "ner_tags": ["O"]}]
    common = dict(
        config_name="selectdenoise_contextual_lattice",
        config=run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        dataset="msra", noise="BT", seed=13, pipelines=(None, {}),
        cache_root=tmp_path / "cache", cache_tag="study-nothink-v1",
        max_concurrency=2, request_timeout=7200,
        adapter_factory=_continuation_adapter, git_sha="a" * 40,
        bundle_hash="b" * 64, explicit_source_rows=rows,
        study_identity={"schema": "qwen-contextual-studies-v1"},
    )

    with pytest.raises(ValueError, match="size must equal"):
        asyncio.run(run_multiseed._run_provider_cache_cell(size=2, **common))
    with pytest.raises(ValueError, match="cannot reuse"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            size=1, reuse_cache_tag="old-nothink", **common,
        ))


def test_study_contextual_replay_uses_same_explicit_identity(tmp_path, monkeypatch):
    rows = [
        {"tokens": ["Alice"], "dirty_tags": ["O"], "ner_tags": ["B-PER"]},
    ]
    cache_root = tmp_path / "cache"
    tag = "qwen32b-contextual-studies-nothink-source-r15-v1"
    identity = {
        "schema": "qwen-contextual-studies-v1", "kind": "ablation",
        "variant": "full", "ratio": 0.15,
    }

    async def pipeline(tokens, dirty_tags, config, dataset_name=None):
        return {
            "pred_tags": ["B-PER"],
            "candidate_paths": [["B-PER"], ["O"]],
            "rag_weights": [0.8, 0.2], "confidence": [0.8],
            "provider_metadata": {
                "coder": [_evidence("coder", "live") for _ in range(3)],
                "reviewer": [_evidence("reviewer", "live")],
                "verifier": [_evidence("verifier", "live")],
            },
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 1, (pipeline, {}), cache_root=cache_root,
        cache_tag=tag, max_concurrency=1, request_timeout=7200,
        adapter_factory=_continuation_adapter, git_sha="a" * 40,
        bundle_hash="b" * 64, explicit_source_rows=rows,
        study_identity=identity,
    ))
    monkeypatch.setattr(run_multiseed, "_noisy_path", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("study replay must not read canonical input")
    ))

    def replay(**kwargs):
        from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

        assert kwargs["provider_record"]["anchor_tags"] == ["B-PER"]
        return {
            "pred_tags": ["B-PER"], "terminal_anchor_tags": ["B-PER"],
            "terminal_model_hash": LOCKED_DECODER_MODEL_HASH,
            "terminal_used_anchor": True,
            "terminal_predicted_gain": 0.0, "terminal_fallback_count": 0,
        }

    output = asyncio.run(run_multiseed._run_contextual_replay_cell(
        "msra", "BT", 13, 1, cache_root=cache_root, cache_tag=tag,
        bundle_hash="b" * 64, git_sha="a" * 40, replay_fn=replay,
        terminal=object(), max_concurrency=1,
        prediction_path=tmp_path / "prediction.jsonl", enable_thinking=False,
        explicit_source_rows=rows, study_identity=identity,
    ))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["candidate_paths"] == [["B-PER"], ["O"]]
    assert result["input_digest"] == run_multiseed._provider_input_digest(rows[0])


def _full_continuation_rows(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    rows = [
        {"tokens": [f"token-{index}", "x"], "dirty_tags": ["O", "O"],
         "ner_tags": ["O", "O"]}
        for index in range(200)
    ]
    run_multiseed._noisy_path("msra", "BT", 13, 200, 0.15).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    return rows


def test_full_profile_continues_verified_n100_prefix_and_requests_only_suffix(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    target_tag = "qwen32b-contextual-nothink-v1"
    _source_cell, source_sha, source_records = _write_reduced_source_for_full_continuation(
        cache_root, rows, source_tag=source_tag,
    )
    source_index_path = run_multiseed._provider_cache_index_path(cache_root, source_tag)
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_key = run_multiseed._provider_cache_cell_key(
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    source_index["cells"][source_key]["path"] = str(
        Path("C:/previous-worktree/provider_cache") / source_tag
        / Path(_source_cell).name
    )
    source_index_path.write_text(json.dumps(source_index), encoding="utf-8")
    requested = []

    async def suffix_pipeline(tokens, dirty_tags, config, dataset_name=None):
        requested.append(int(tokens[0].split("-")[1]))
        return {
            "pred_tags": list(dirty_tags),
            "candidate_paths": [list(dirty_tags)] * 3,
            "rag_weights": [1.0, 1.0, 1.0],
            "confidence": [1.0] * len(tokens),
            "provider_metadata": {
                "coder": [_evidence("coder", "live") for _ in range(3)],
                "reviewer": [_evidence("reviewer", "skipped_identical")],
                "verifier": [_evidence("verifier", "skipped_uncontested")],
            },
            "fallback_used": False,
        }

    cell = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 200, (suffix_pipeline, {}),
        cache_root=cache_root, cache_tag=target_tag, max_concurrency=32,
        request_timeout=7200, adapter_factory=_continuation_adapter,
        git_sha="a" * 40, bundle_hash="b" * 64,
        reuse_cache_tag=source_tag,
        compatible_git_shas=("d" * 40,),
    ))

    assert sorted(requested) == list(range(100, 200))
    assert cell["row_count"] == 200
    target = Path(cell["path"])
    combined = official_provider_cache.read_provider_cell(target, cell["sha256"], 200)
    assert combined[:100] == source_records
    manifest = run_multiseed._read_provider_cache_manifest(target)
    assert manifest["git_sha"] == "a" * 40
    assert manifest["request_limits"] == {
        "provider_timeout_seconds": 600.0,
        "runner_timeout_seconds": 7200,
        "max_concurrency": 32,
    }
    assert manifest["derived_from"] == {
        "cache_tag": source_tag,
        "sha256": source_sha,
        "source_row_count": 100,
        "selected_rows": [0, 100],
        "producer_git_sha": "d" * 40,
    }

    requested.clear()
    target_index_path = run_multiseed._provider_cache_index_path(cache_root, target_tag)
    target_index = json.loads(target_index_path.read_text(encoding="utf-8"))
    target_key = run_multiseed._provider_cache_cell_key(
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    target_index["cells"][target_key]["path"] = str(
        Path("C:/previous-worktree/provider_cache") / target_tag / target.name
    )
    target_index_path.write_text(json.dumps(target_index), encoding="utf-8")

    async def must_not_run(*_args, **_kwargs):
        pytest.fail("a verified existing N200 target cell must make zero requests")

    resumed = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 200, (must_not_run, {}),
        cache_root=cache_root, cache_tag=target_tag, max_concurrency=32,
        request_timeout=7200, adapter_factory=_continuation_adapter,
        git_sha="a" * 40, bundle_hash="b" * 64,
        reuse_cache_tag=source_tag,
        compatible_git_shas=("d" * 40,),
    ))
    assert resumed["path"] == str(target)
    assert resumed["sha256"] == cell["sha256"]
    assert requested == []


@pytest.mark.parametrize("mismatch", ["thinking", "bundle", "input", "sha"])
def test_full_profile_reuse_rejects_mismatch_or_tamper_without_publishing(
    tmp_path, monkeypatch, mismatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    source_rows = list(rows)
    if mismatch == "input":
        source_rows[0] = {
            "tokens": ["different", "x"], "dirty_tags": ["O", "O"],
            "ner_tags": ["O", "O"],
        }
    source_cell, _sha, _records = _write_reduced_source_for_full_continuation(
        cache_root, source_rows, source_tag=source_tag,
        bundle_hash=("c" * 64 if mismatch == "bundle" else "b" * 64),
        enable_thinking=mismatch == "thinking",
    )
    if mismatch == "sha":
        source_cell.write_bytes(source_cell.read_bytes() + b"\n")

    async def must_not_run(*_args, **_kwargs):
        pytest.fail("invalid reuse must fail before provider requests")

    with pytest.raises(ValueError):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (must_not_run, {}),
            cache_root=cache_root, cache_tag="qwen32b-contextual-nothink-v1",
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64, reuse_cache_tag=source_tag,
            compatible_git_shas=("d" * 40,),
        ))

    target = run_multiseed._provider_cache_cell_path(
        cache_root, "qwen32b-contextual-nothink-v1",
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_full_profile_suffix_failure_leaves_no_target_or_temporary_cache(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    _write_reduced_source_for_full_continuation(cache_root, rows, source_tag=source_tag)

    async def fail_suffix(*_args, **_kwargs):
        raise RuntimeError("suffix failed")

    with pytest.raises(RuntimeError, match="suffix failed"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (fail_suffix, {}),
            cache_root=cache_root, cache_tag="qwen32b-contextual-nothink-v1",
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64, reuse_cache_tag=source_tag,
            compatible_git_shas=("d" * 40,),
        ))

    target = run_multiseed._provider_cache_cell_path(
        cache_root, "qwen32b-contextual-nothink-v1",
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    assert not run_multiseed._provider_cache_index_path(
        cache_root, "qwen32b-contextual-nothink-v1",
    ).exists()


def test_exhausted_semantic_retries_persist_failure_provenance_without_a_cell(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    _write_reduced_source_for_full_continuation(cache_root, rows, source_tag=source_tag)
    attempts = []
    for number in range(1, 4):
        record = _evidence("verifier", "live")
        record.update({
            "semantic_attempt": number,
            "semantic_outcome": "rejected_illegal_iob2",
            "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
            "rejected_tags": ["O", "I-ORG"],
            "response_sha256": run_multiseed._sha256_json({"tags": ["O", "I-ORG"]}),
        })
        attempts.append(record)

    async def exhaust_retries(*_args, **_kwargs):
        raise VerifierSemanticRetryExhausted(attempts)

    with pytest.raises(VerifierSemanticRetryExhausted, match="after 3 attempts"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (exhaust_retries, {}),
            cache_root=cache_root, cache_tag="qwen32b-contextual-nothink-v1",
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64, reuse_cache_tag=source_tag,
            compatible_git_shas=("d" * 40,),
        ))

    target = run_multiseed._provider_cache_cell_path(
        cache_root, "qwen32b-contextual-nothink-v1",
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    failures = list((target.parent / "failures").glob("*.json"))
    assert failures
    assert not target.exists()
    assert not list(target.parent.rglob("*.tmp"))


def test_exhausted_semantic_retries_can_publish_explicit_invalid_rows(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    attempts = []
    for number in range(1, 4):
        record = _evidence("verifier", "live")
        record.update({
            "semantic_attempt": number,
            "semantic_outcome": "rejected_illegal_iob2",
            "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
            "rejected_tags": ["O", "I-ORG"],
            "response_sha256": run_multiseed._sha256_json(
                {"tags": ["O", "I-ORG"]}
            ),
        })
        attempts.append(record)
    context = {
        "candidate_paths": [["O", "I-ORG"], ["O", "O"]],
        "rag_weights": [0.5, 0.5],
        "provider_metadata": {
            "coder": [_evidence("coder", "live") for _ in range(3)],
            "reviewer": [_evidence("reviewer", "skipped_identical")],
            "verifier": attempts,
        },
    }

    async def exhaust_retries(*_args, **_kwargs):
        raise VerifierSemanticRetryExhausted(attempts, context=context)

    cell = asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "BT", 13, 200, (exhaust_retries, {}),
        cache_root=cache_root,
        cache_tag="qwen32b-contextual-nothink-invalid-v1",
        max_concurrency=32, request_timeout=7200,
        adapter_factory=_continuation_adapter, git_sha="a" * 40,
        bundle_hash="b" * 64, allow_verifier_exhausted=True,
    ))

    records = official_provider_cache.read_provider_cell(
        Path(cell["path"]), cell["sha256"], 200,
    )
    assert len(records) == 200
    assert all(record["provider_outcome_status"] == "invalid_verifier_iob2"
               for record in records)
    assert all(record["anchor_tags"] == [] and record["confidence"] == []
               for record in records)
    assert all(record["provider_metadata"]["verifier"] == attempts
               for record in records)


def test_contextual_replay_preserves_published_invalid_verifier_row(tmp_path):
    rows = [{"tokens": ["Alice", "x"], "dirty_tags": ["O", "O"],
             "ner_tags": ["B-PER", "O"]}]
    cache_root = tmp_path / "cache"
    tag = "qwen32b-contextual-nothink-invalid-v1"
    attempts = []
    for number in range(1, 4):
        record = _evidence("verifier", "live")
        record.update({
            "semantic_attempt": number,
            "semantic_outcome": "rejected_illegal_iob2",
            "illegal_transition": {"index": 1, "previous": "O", "current": "I-PER"},
            "rejected_tags": ["O", "I-PER"],
            "response_sha256": run_multiseed._sha256_json(
                {"tags": ["O", "I-PER"]}
            ),
        })
        attempts.append(record)
    context = {
        "candidate_paths": [["O", "I-PER"], ["O", "O"]],
        "rag_weights": [0.5, 0.5],
        "provider_metadata": {
            "coder": [_evidence("coder", "live") for _ in range(5)],
            "reviewer": [_evidence("reviewer", "skipped_identical")],
            "verifier": attempts,
        },
    }

    async def exhaust(*_args, **_kwargs):
        raise VerifierSemanticRetryExhausted(attempts, context=context)

    identity = {
        "schema": "qwen-contextual-studies-v1", "kind": "ablation",
        "variant": "minus_atf_deanchor", "ratio": 0.15,
    }
    asyncio.run(run_multiseed._run_provider_cache_cell(
        "selectdenoise_contextual_lattice",
        run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        "msra", "ATF", 13, 1, (exhaust, {}), cache_root=cache_root,
        cache_tag=tag, max_concurrency=1, request_timeout=7200,
        adapter_factory=_continuation_adapter, git_sha="a" * 40,
        bundle_hash="b" * 64, explicit_source_rows=rows,
        study_identity=identity, allow_verifier_exhausted=True,
    ))

    def must_not_replay(**_kwargs):
        raise AssertionError("invalid Verifier rows must bypass terminal replay")

    output = asyncio.run(run_multiseed._run_contextual_replay_cell(
        "msra", "ATF", 13, 1, cache_root=cache_root, cache_tag=tag,
        bundle_hash="b" * 64, git_sha="a" * 40, replay_fn=must_not_replay,
        terminal=object(), max_concurrency=1,
        prediction_path=tmp_path / "prediction.jsonl", enable_thinking=False,
        explicit_source_rows=rows, study_identity=identity,
        allow_invalid_verifier=True,
    ))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["prediction_status"] == "invalid_verifier_iob2"
    assert result["pred_tags"] == ["O", "I-PER"]
    assert result["terminal_model_hash"] is None
    assert result["confidence"] == []


def test_interrupted_semantic_retries_persist_rejected_prefix_without_a_cell(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    _write_reduced_source_for_full_continuation(cache_root, rows, source_tag=source_tag)
    rejected = _evidence("verifier", "live")
    rejected.update({
        "semantic_attempt": 1,
        "semantic_outcome": "rejected_illegal_iob2",
        "illegal_transition": {"index": 1, "previous": "O", "current": "I-ORG"},
        "rejected_tags": ["O", "I-ORG"],
        "response_sha256": run_multiseed._sha256_json({"tags": ["O", "I-ORG"]}),
    })

    async def interrupt_retries(*_args, **_kwargs):
        raise VerifierSemanticRetryInterrupted([rejected], ValueError("bad ontology"))

    with pytest.raises(VerifierSemanticRetryInterrupted):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (interrupt_retries, {}),
            cache_root=cache_root, cache_tag="qwen32b-contextual-nothink-v1",
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64, reuse_cache_tag=source_tag,
            compatible_git_shas=("d" * 40,),
        ))

    target = run_multiseed._provider_cache_cell_path(
        cache_root, "qwen32b-contextual-nothink-v1",
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    failures = list((target.parent / "failures").glob("*.json"))
    assert failures
    payload = json.loads(failures[0].read_text(encoding="utf-8"))
    assert payload["terminal_outcome"] == "nonsemantic_interruption"
    assert payload["terminal_error_type"] == "ValueError"
    assert len(payload["attempts"]) == 1
    assert not target.exists()


def test_full_profile_missing_source_cell_runs_normally_even_when_source_index_exists(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    source_tag = "qwen32b-contextual-nothink-s13-n100-v1"
    _write_reduced_source_for_full_continuation(cache_root, rows, source_tag=source_tag)
    target_manifest = run_multiseed._provider_cache_manifest(
        rows, run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        dataset="msra", noise="BT", seed=42, git_sha="a" * 40,
        bundle_hash="b" * 64, enable_thinking=False,
    )

    result = run_multiseed._load_provider_cache_continuation_prefix(
        cache_root=cache_root, source_tag=source_tag,
        target_tag="qwen32b-contextual-nothink-v1",
        config_name="selectdenoise_contextual_lattice", dataset="msra",
        noise="BT", seed=42, rows=rows, target_manifest=target_manifest,
        provider_identity=_continuation_adapter().provider_metadata(),
    )

    assert result is None


def test_formal_full_profile_rejects_missing_seed13_prefix_before_requests(
    tmp_path, monkeypatch,
):
    _full_continuation_rows(tmp_path, monkeypatch)

    async def must_not_run(*_args, **_kwargs):
        pytest.fail("missing approved seed13 prefix must fail before paid requests")

    with pytest.raises(ValueError, match="seed13.*prefix"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (must_not_run, {}),
            cache_root=tmp_path / "cache",
            cache_tag=run_multiseed.QWEN_FORMAL_CACHE_TAG,
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64,
            reuse_cache_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
            compatible_git_shas=("d" * 40,),
        ))


def test_formal_full_profile_rejects_unapproved_prefix_producer(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    _write_reduced_source_for_full_continuation(
        cache_root, rows, source_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
    )

    async def must_not_run(*_args, **_kwargs):
        pytest.fail("unapproved source producer must fail before paid requests")

    with pytest.raises(ValueError, match="producer Git SHA"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (must_not_run, {}),
            cache_root=cache_root,
            cache_tag=run_multiseed.QWEN_FORMAL_CACHE_TAG,
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64,
            reuse_cache_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
            compatible_git_shas=("e" * 40,),
        ))


def test_formal_full_profile_cleans_published_cell_when_index_update_fails(
    tmp_path, monkeypatch,
):
    rows = _full_continuation_rows(tmp_path, monkeypatch)
    cache_root = tmp_path / "cache"
    _write_reduced_source_for_full_continuation(
        cache_root, rows, source_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
    )

    async def suffix_pipeline(tokens, dirty_tags, config, dataset_name=None):
        return {
            "pred_tags": list(dirty_tags),
            "candidate_paths": [list(dirty_tags)] * 3,
            "rag_weights": [1.0, 1.0, 1.0],
            "confidence": [1.0] * len(tokens),
            "provider_metadata": {
                "coder": [_evidence("coder", "live") for _ in range(3)],
                "reviewer": [_evidence("reviewer", "skipped_identical")],
                "verifier": [_evidence("verifier", "skipped_uncontested")],
            },
            "fallback_used": False,
        }

    original_update = run_multiseed._update_provider_cache_index

    def fail_target_index(root, tag, **kwargs):
        if tag == run_multiseed.QWEN_FORMAL_CACHE_TAG:
            raise OSError("index publish failed")
        return original_update(root, tag, **kwargs)

    monkeypatch.setattr(run_multiseed, "_update_provider_cache_index", fail_target_index)
    with pytest.raises(OSError, match="index publish failed"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (suffix_pipeline, {}),
            cache_root=cache_root,
            cache_tag=run_multiseed.QWEN_FORMAL_CACHE_TAG,
            max_concurrency=32, request_timeout=7200,
            adapter_factory=_continuation_adapter, git_sha="a" * 40,
            bundle_hash="b" * 64,
            reuse_cache_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
            compatible_git_shas=("d" * 40,),
        ))

    target = run_multiseed._provider_cache_cell_path(
        cache_root, run_multiseed.QWEN_FORMAL_CACHE_TAG,
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    assert not run_multiseed._provider_cache_index_path(
        cache_root, run_multiseed.QWEN_FORMAL_CACHE_TAG,
    ).exists()


def test_formal_full_profile_rejects_timeout_drift_before_requests(
    tmp_path, monkeypatch,
):
    _full_continuation_rows(tmp_path, monkeypatch)
    adapter = _continuation_adapter()
    adapter.provider_metadata = lambda: {
        **_continuation_adapter().provider_metadata(), "timeout_seconds": 300.0,
    }

    async def must_not_run(*_args, **_kwargs):
        pytest.fail("timeout drift must fail before provider requests")

    with pytest.raises(ValueError, match="600"):
        asyncio.run(run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (must_not_run, {}),
            cache_root=tmp_path / "cache",
            cache_tag=run_multiseed.QWEN_FORMAL_CACHE_TAG,
            max_concurrency=32, request_timeout=7200,
            adapter_factory=lambda: adapter, git_sha="a" * 40,
            bundle_hash="b" * 64,
            reuse_cache_tag=run_multiseed.QWEN_REDUCED_CACHE_TAG,
            compatible_git_shas=("d" * 40,),
        ))


def test_contextual_replay_phase_parses_an_explicit_cache_tag_and_root(tmp_path):
    args = run_multiseed._parse_args([
        "--phase", "contextual-replay",
        "--provider-cache-root", str(tmp_path),
        "--provider-cache-tag", "qwen-replay-v1",
    ])

    assert args.phase == "contextual-replay"
    assert args.provider_cache_root == tmp_path
    assert args.provider_cache_tag == "qwen-replay-v1"


def test_contextual_replay_bundle_accepts_snapshot_directory_checkpoint(
    tmp_path, monkeypatch
):
    import contextual_lattice_runtime as runtime

    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    model_file = checkpoint / "model.safetensors"
    model_file.write_bytes(b"locked checkpoint")
    checkpoint_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    decoder = bundle / "decoder.pt"
    gate = bundle / "gate.joblib"
    decoder.write_bytes(b"decoder")
    gate.write_bytes(b"gate")
    decoder_hash = hashlib.sha256(decoder.read_bytes()).hexdigest()
    gate_hash = hashlib.sha256(gate.read_bytes()).hexdigest()
    split_hash = "1" * 64
    decoder_model_hash = "2" * 64
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_hash": checkpoint_hash,
                "split_hash": split_hash,
                "decoder_model_hash": decoder_model_hash,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    embedding_cache = tmp_path / "embedding_cache"
    embedding_cache.mkdir()

    monkeypatch.setattr(runtime, "LOCKED_CHECKPOINT_HASH", checkpoint_hash)
    monkeypatch.setattr(runtime, "LOCKED_SPLIT_HASH", split_hash)
    monkeypatch.setattr(runtime, "LOCKED_BUNDLE_MANIFEST_HASH", manifest_hash)
    monkeypatch.setattr(runtime, "LOCKED_DECODER_FILE_HASH", decoder_hash)
    monkeypatch.setattr(runtime, "LOCKED_GATE_FILE_HASH", gate_hash)
    monkeypatch.setattr(runtime, "LOCKED_DECODER_MODEL_HASH", decoder_model_hash)
    monkeypatch.setenv("CONTEXTUAL_LATTICE_BUNDLE", str(bundle))
    monkeypatch.setenv("CONTEXTUAL_LATTICE_ENCODER_CACHE", str(embedding_cache))

    assert run_multiseed._validate_contextual_replay_bundle(manifest_hash) == (
        bundle.resolve(),
        embedding_cache.resolve(),
    )


def test_contextual_replay_cell_rechecks_source_rows_and_persists_cache_provenance(tmp_path, monkeypatch):
    import asyncio

    noisy_dir = tmp_path / "noisy"
    prediction_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    source_rows = [
        {"tokens": ["Acme", "arrived"], "ner_tags": ["B-ORG", "O"],
         "dirty_tags": ["B-ORG", "O"]}
        for _ in range(200)
    ]
    noisy_dir.mkdir(parents=True)
    noisy_path = run_multiseed._noisy_path("conll2003", "BT", 13, 200, 0.15)
    noisy_path.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    cache_root = tmp_path / "cache"
    tag = "qwen-replay-nothink-v1"
    config = run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"]
    manifest = run_multiseed._provider_cache_manifest(
        source_rows, config, dataset="conll2003", noise="BT", seed=13,
        git_sha="a" * 40, bundle_hash="b" * 64, enable_thinking=False,
    )
    records = []
    for index, row in enumerate(source_rows):
        record = _record(index, run_multiseed._provider_input_digest(row))
        record["provider_metadata"]["coder"] = [
            _evidence("coder", "live") for _ in range(3)
        ]
        records.append(record)
    cell = run_multiseed._provider_cache_cell_path(
        cache_root, tag, "selectdenoise_contextual_lattice", "conll2003", "BT", 13
    )
    sha = official_provider_cache.write_provider_cell(cell, records, manifest)
    run_multiseed._update_provider_cache_index(
        cache_root, tag,
        key=run_multiseed._provider_cache_cell_key(
            "selectdenoise_contextual_lattice", "conll2003", "BT", 13
        ),
        cell={"path": str(cell), "sha256": sha, "row_count": 200,
              "git_sha": "a" * 40, "model_revision": _REVISION,
              "provider_fingerprint": "d" * 64},
    )

    def replay_fn(**kwargs):
        assert set(kwargs["provider_record"]) == {
            "row_index", "input_digest", "anchor_tags", "candidate_paths",
            "rag_weights", "confidence", "provider_metadata", "fallback_used",
        }
        return {
            "pred_tags": ["B-ORG", "O"],
            "terminal_anchor_tags": ["B-ORG", "O"],
            "terminal_model_hash": "2b45770a128cc5a716d3c7dbdc75ab059afc1a1b9ed471ea5cf1bd4f2a150e4e",
            "terminal_used_anchor": True,
            "terminal_predicted_gain": 0.0,
            "terminal_fallback_count": 0,
        }

    output = asyncio.run(run_multiseed._run_contextual_replay_cell(
        "conll2003", "BT", 13, 200, cache_root=cache_root, cache_tag=tag,
        bundle_hash="b" * 64, git_sha="a" * 40, replay_fn=replay_fn,
        terminal=object(), max_concurrency=4, prediction_path=prediction_path,
        enable_thinking=False,
    ))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 200
    assert all(row["provider_cache_sha256"] == sha for row in rows)
    assert all(row["terminal_fallback_count"] == 0 for row in rows)


def test_contextual_replay_environment_clears_credentials_and_blocks_sockets(monkeypatch):
    import socket

    monkeypatch.setenv("BACKBONE_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with run_multiseed._offline_contextual_replay_environment():
        assert run_multiseed.os.environ["BACKBONE_API_KEY"] == ""
        assert run_multiseed.os.environ["OPENAI_API_KEY"] == ""
        with pytest.raises(RuntimeError, match="network access is disabled"):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    assert run_multiseed.os.environ["BACKBONE_API_KEY"] == "secret"


def test_provider_cache_phase_is_a_distinct_runner_phase():
    args = run_multiseed._parse_args(["--phase", "provider-cache"])

    assert args.phase == "provider-cache"


def test_provider_cache_caps_live_requests_not_only_sentences(tmp_path, monkeypatch):
    """Three Coder paths per sentence must still honor the 32-request cap."""
    from multi_agent_v2 import _invoke_structured_requester

    noisy_dir = tmp_path / "noisy"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    noisy_dir.mkdir()
    row = {
        "tokens": ["Alice"],
        "ner_tags": ["B-PER"],
        "dirty_tags": ["B-PER"],
    }
    run_multiseed._noisy_path("msra", "BT", 13, 200, 0.15).write_text(
        "".join(json.dumps(row) + "\n" for _ in range(200)), encoding="utf-8"
    )

    class PeakTrackingAdapter:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0
            self.stages = []
            self.timeout_seconds = 120

        def provider_metadata(self):
            return {
                "timeout_seconds": self.timeout_seconds,
                "provider": "vllm",
                "model": _MODEL,
                "served_model": _SERVED_MODEL,
                "revision": _REVISION,
                "structured_api": "chat-completions-json-schema",
                "enable_thinking": False,
                "thinking_mode": "nothink",
            }

        def structured_requester(self, stage, payload):
            with self.lock:
                self.stages.append(stage)
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.05)
                return {}
            finally:
                with self.lock:
                    self.active -= 1

    adapter = PeakTrackingAdapter()

    async def three_path_pipeline(tokens, dirty_tags, config, dataset_name=None):
        requester = config["structured_requester"]
        await asyncio.gather(*[
            _invoke_structured_requester(requester, "coder", {}) for _ in range(3)
        ])
        await _invoke_structured_requester(requester, "reviewer", {})
        await _invoke_structured_requester(requester, "verifier", {})
        return {
            "pred_tags": list(dirty_tags),
            "candidate_paths": [list(dirty_tags)] * 3,
            "rag_weights": [1.0, 1.0, 1.0],
            "confidence": [1.0],
            "provider_metadata": {
                "coder": [_evidence("coder", "live") for _ in range(3)],
                "reviewer": [_evidence("reviewer", "live")],
                "verifier": [_evidence("verifier", "live")],
            },
            "fallback_used": False,
        }

    async def run_cell(runner_timeout=10, git_sha="a" * 40, compatible_git_shas=()):
        asyncio.get_running_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=64)
        )
        await run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (three_path_pipeline, {}),
                cache_root=tmp_path / "cache", cache_tag="request-cap-nothink-test",
            max_concurrency=32, request_timeout=runner_timeout,
            adapter_factory=lambda: adapter, git_sha=git_sha,
            bundle_hash="b" * 64,
            compatible_git_shas=compatible_git_shas,
        )

    asyncio.run(run_cell())

    assert adapter.peak <= 32
    assert set(adapter.stages) == {"coder", "reviewer", "verifier"}
    stage_count = len(adapter.stages)
    adapter.timeout_seconds = 300
    asyncio.run(run_cell(
        runner_timeout=11,
        git_sha="c" * 40,
        compatible_git_shas=["a" * 40],
    ))
    assert len(adapter.stages) == stage_count


@pytest.mark.filterwarnings(
    "ignore:The loop argument is deprecated.*:DeprecationWarning"
)
def test_provider_cache_abort_does_not_start_waiting_provider_requests(
    tmp_path, monkeypatch
):
    """A failed live call must cancel queued calls before another one starts."""
    from multi_agent_v2 import _invoke_structured_requester

    noisy_dir = tmp_path / "noisy"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    noisy_dir.mkdir()
    row = {
        "tokens": ["Alice"],
        "ner_tags": ["B-PER"],
        "dirty_tags": ["B-PER"],
    }
    run_multiseed._noisy_path("msra", "BT", 13, 200, 0.15).write_text(
        "".join(json.dumps(row) + "\n" for _ in range(200)), encoding="utf-8"
    )

    class FailingAdapter:
        def __init__(self):
            self.lock = threading.Lock()
            self.started = 0
            self.first_wave = threading.Barrier(32)
            self.failure_raised = threading.Event()
            self.release = threading.Event()

        def provider_metadata(self):
            return {
                "provider": "vllm",
                "model": _MODEL,
                "served_model": _SERVED_MODEL,
                "revision": _REVISION,
                "structured_api": "chat-completions-json-schema",
            }

        def structured_requester(self, stage, payload):
            with self.lock:
                self.started += 1
                call_number = self.started
            if call_number <= 32:
                self.first_wave.wait(timeout=5)
            if call_number == 32:
                self.failure_raised.set()
                raise RuntimeError("provider failed")
            self.release.wait(timeout=5)
            return {}

    adapter = FailingAdapter()

    async def failing_pipeline(tokens, dirty_tags, config, dataset_name=None):
        requester = config["structured_requester"]
        await asyncio.gather(*[
            _invoke_structured_requester(requester, "coder", {})
            for _ in range(3)
        ])

    def release_started_calls():
        adapter.failure_raised.wait(timeout=5)
        time.sleep(0.1)
        adapter.release.set()

    controller = threading.Thread(target=release_started_calls, daemon=True)
    controller.start()

    async def run_cell():
        asyncio.get_running_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=64)
        )
        await run_multiseed._run_provider_cache_cell(
            "selectdenoise_contextual_lattice",
            run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            "msra", "BT", 13, 200, (failing_pipeline, {}),
            cache_root=tmp_path / "cache", cache_tag="abort-request-cap-test",
            max_concurrency=32, request_timeout=10,
            adapter_factory=lambda: adapter, git_sha="a" * 40,
            bundle_hash="b" * 64,
        )

    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            asyncio.run(run_cell())
    finally:
        adapter.release.set()
        controller.join(timeout=5)

    assert adapter.started == 32
    assert not list((tmp_path / "cache").rglob("*.jsonl"))


def test_offline_replay_precreates_windows_event_loop_before_blocking_network():
    with run_multiseed._offline_contextual_replay_loop() as loop:
        assert loop.run_until_complete(asyncio.sleep(0, result="ok")) == "ok"
        with pytest.raises(RuntimeError, match="network access is disabled"):
            run_multiseed.socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_reduced_replay_validates_new_prediction_with_requested_size(monkeypatch, tmp_path):
    source = [{"tokens": ["x"], "dirty_tags": ["O"], "ner_tags": ["O"]}
              for _ in range(100)]
    cached = [{"input_digest": run_multiseed._provider_input_digest(row),
               "provider_metadata": {}} for row in source]
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("present", encoding="utf-8")
    cell_path = tmp_path / "cell.jsonl"
    cell_path.write_text("present", encoding="utf-8")
    expected_manifest = {"git_sha": "a" * 40, "model_revision": _REVISION}
    monkeypatch.setattr(run_multiseed, "_validate_contextual_replay_launch", lambda *a, **k: None)
    monkeypatch.setattr(run_multiseed, "_noisy_path", lambda *a, **k: source_path)
    monkeypatch.setattr(run_multiseed, "_load_official_source_rows", lambda *a, **k: source)
    monkeypatch.setattr(run_multiseed, "_provider_cache_manifest", lambda *a, **k: expected_manifest)
    monkeypatch.setattr(run_multiseed, "_provider_cache_cell_path", lambda *a, **k: cell_path)
    monkeypatch.setattr(run_multiseed, "_read_provider_cache_index_cell", lambda *a, **k: {
        "path": str(cell_path), "row_count": 100, "sha256": "b" * 64,
        "provider_fingerprint": "c" * 64, "git_sha": "a" * 40,
        "model_revision": _REVISION,
    })
    monkeypatch.setattr(run_multiseed, "read_provider_cell", lambda *a, **k: cached)
    monkeypatch.setattr(run_multiseed, "_read_provider_cache_manifest", lambda *a, **k: expected_manifest)
    monkeypatch.setattr(run_multiseed, "_validate_replay_cache_identity", lambda *a, **k: None)
    monkeypatch.setattr(run_multiseed, "_validate_replay_source_rows", lambda *a, **k: None)
    monkeypatch.setattr(run_multiseed, "_validate_contextual_provider_evidence", lambda *a, **k: None)
    observed = []
    monkeypatch.setattr(
        run_multiseed, "_validate_contextual_replay_prediction",
        lambda *a, **k: observed.append(k.get("expected_count")),
    )

    def replay(**_kwargs):
        return {
            "pred_tags": ["O"], "terminal_anchor_tags": ["O"],
            "terminal_model_hash": "d" * 64, "terminal_used_anchor": False,
            "terminal_predicted_gain": 0.0, "terminal_fallback_count": 0,
        }

    asyncio.run(run_multiseed._run_contextual_replay_cell(
        "msra", "BT", 13, 100, cache_root=tmp_path, cache_tag="nothink-test",
        bundle_hash="e" * 64, git_sha="a" * 40, replay_fn=replay,
        terminal=object(), max_concurrency=32, prediction_path=tmp_path / "pred.jsonl",
        enable_thinking=False,
    ))
    assert observed == [100]
