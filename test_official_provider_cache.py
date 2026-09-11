import hashlib
import json
from pathlib import Path

import pytest

import official_provider_cache
import run_multiseed


_MODEL = "Qwen/Qwen3-32B-AWQ"
_REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
_SERVED_MODEL = f"{_MODEL}@{_REVISION}"


def _evidence(stage, status):
    return {
        "stage": stage,
        "status": status,
        "model": _MODEL,
        "served_model": _SERVED_MODEL,
        "revision": _REVISION,
    }


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
            "reviewer": [_evidence("reviewer", "skipped")],
            "verifier": [_evidence("verifier", "skipped")],
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
        "configuration": {"terminal_decoder": "contextual-lattice-v1"},
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
@pytest.mark.skip(reason="Task 4 owns offline contextual replay identity validation.")
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


def test_provider_cache_phase_is_a_distinct_runner_phase():
    args = run_multiseed._parse_args(["--phase", "provider-cache"])

    assert args.phase == "provider-cache"
