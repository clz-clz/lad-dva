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
    evidence = {
        "stage": stage,
        "status": status,
        "provider": "vllm",
        "model": _MODEL,
        "served_model": _SERVED_MODEL,
        "revision": _REVISION,
        "structured_api": "chat-completions-json-schema",
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
    tag = "qwen-replay-v1"
    config = run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"]
    manifest = run_multiseed._provider_cache_manifest(
        source_rows, config, dataset="conll2003", noise="BT", seed=13,
        git_sha="a" * 40, bundle_hash="b" * 64,
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
