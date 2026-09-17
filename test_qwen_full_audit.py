import json
import os
from pathlib import Path

import official_provider_cache
import qwen_full_audit
import run_multiseed
from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH
from metrics import compute_prf1, compute_ser


TAG = "qwen32b-contextual-nothink-v1"
REVISION = official_provider_cache.QWEN_REVISION
SERVED = official_provider_cache.QWEN_SERVED_MODEL


def _evidence(stage, status):
    value = {
        "stage": stage, "status": status, "provider": "vllm",
        "model": official_provider_cache.QWEN_MODEL, "served_model": SERVED,
        "revision": REVISION, "structured_api": "chat-completions-json-schema",
        "enable_thinking": False, "thinking_mode": "nothink",
    }
    if status == "live":
        value.update({
            "response_model": SERVED, "response_status": "completed",
            "finish_reason": "stop", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
    return value


def _write_minimal_matrix(
    tmp_path, *, credential=False, row_count=2, prefix_source=False,
    manifest_git_sha="a" * 40,
):
    noisy_root = tmp_path / "noisy"
    cache_root = tmp_path / "cache"
    predictions_root = tmp_path / "predictions"
    noisy_root.mkdir()
    predictions_root.mkdir()
    rows = [
        {"tokens": [f"t{index}"], "dirty_tags": ["O"], "ner_tags": ["O"]}
        for index in range(row_count)
    ]
    noisy = noisy_root / "noisy_seed13__BT__msra__N200.jsonl"
    noisy.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    records = []
    for index, row in enumerate(rows):
        evidence = {
            "coder": [_evidence("coder", "live") for _ in range(3)],
            "reviewer": [_evidence("reviewer", "skipped_identical")],
            "verifier": [_evidence("verifier", "skipped_uncontested")],
        }
        records.append({
            "row_index": index,
            "input_digest": run_multiseed._provider_input_digest(row),
            "anchor_tags": ["O"], "candidate_paths": [["O"]] * 3,
            "rag_weights": [1.0, 1.0, 1.0], "confidence": [1.0],
            "provider_metadata": evidence, "fallback_used": False,
        })
    manifest = run_multiseed._provider_cache_manifest(
        rows, run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
        dataset="msra", noise="BT", seed=13, git_sha=manifest_git_sha,
        bundle_hash="b" * 64, enable_thinking=False,
    )
    manifest["request_limits"] = {
        "provider_timeout_seconds": 600.0,
        "runner_timeout_seconds": 7200.0,
        "max_concurrency": 32,
    }
    if credential:
        manifest["credential"] = "must-not-be-present"
    if prefix_source:
        source_manifest = run_multiseed._provider_cache_manifest(
            rows[:100], run_multiseed.CONFIGURATIONS["selectdenoise_contextual_lattice"],
            dataset="msra", noise="BT", seed=13, git_sha="d" * 40,
            bundle_hash="b" * 64, enable_thinking=False,
        )
        source_cell = run_multiseed._provider_cache_cell_path(
            cache_root, run_multiseed.QWEN_REDUCED_CACHE_TAG,
            "selectdenoise_contextual_lattice", "msra", "BT", 13,
        )
        source_digest = official_provider_cache.write_provider_cell(
            source_cell, records[:100], source_manifest,
        )
        run_multiseed._update_provider_cache_index(
            cache_root, run_multiseed.QWEN_REDUCED_CACHE_TAG,
            key=run_multiseed._provider_cache_cell_key(
                "selectdenoise_contextual_lattice", "msra", "BT", 13,
            ),
            cell={"path": str(source_cell), "sha256": source_digest,
                  "row_count": 100, "git_sha": "d" * 40,
                  "model_revision": REVISION, "provider_fingerprint": "e" * 64},
        )
        manifest["derived_from"] = {
            "cache_tag": run_multiseed.QWEN_REDUCED_CACHE_TAG,
            "sha256": source_digest, "source_row_count": 100,
            "selected_rows": [0, 100], "producer_git_sha": "d" * 40,
        }
    cell = run_multiseed._provider_cache_cell_path(
        cache_root, TAG, "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    digest = official_provider_cache.write_provider_cell(cell, records, manifest)
    run_multiseed._update_provider_cache_index(
        cache_root, TAG,
        key=run_multiseed._provider_cache_cell_key(
            "selectdenoise_contextual_lattice", "msra", "BT", 13,
        ),
        cell={"path": str(cell), "sha256": digest, "row_count": row_count,
              "git_sha": manifest_git_sha, "model_revision": REVISION,
              "provider_fingerprint": "c" * 64},
    )

    prediction = predictions_root / (
        f"pred_seed13__selectdenoise_contextual_lattice__msra__BT__{TAG}.jsonl"
    )
    prediction_rows = []
    for row, record in zip(rows, records):
        prediction_rows.append({
            "tokens": row["tokens"], "gold_tags": row["ner_tags"],
            "pred_tags": ["O"], "terminal_anchor_tags": ["O"],
            "terminal_model_hash": LOCKED_DECODER_MODEL_HASH,
            "terminal_used_anchor": False, "terminal_predicted_gain": 0.0,
            "terminal_fallback_count": 0, "provider_cache_sha256": digest,
            "provider_metadata": record["provider_metadata"], "fallback_used": False,
        })
    prediction.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8",
    )
    aggregate = predictions_root / f"aggregated__{TAG}.json"
    gold = [row["gold_tags"] for row in prediction_rows]
    pred = [row["pred_tags"] for row in prediction_rows]
    metrics = compute_prf1(gold, pred)
    ser = compute_ser(pred)
    aggregate.write_text(json.dumps({
        "selectdenoise_contextual_lattice|msra|BT": {
            "n_seeds": 1,
            "f1_mean": metrics["f1"], "f1_std": 0.0,
            "p_mean": metrics["precision"], "p_std": 0.0,
            "r_mean": metrics["recall"], "r_std": 0.0,
            "ser_mean": ser, "ser_std": 0.0,
            "per_seed": [{"seed": 13, **metrics, "ser": ser}],
        }
    }), encoding="utf-8")
    return noisy_root, cache_root, predictions_root, aggregate, prediction


def _audit(tmp_path, *, credential=False):
    noisy, cache, predictions, aggregate, prediction = _write_minimal_matrix(
        tmp_path, credential=credential,
    )
    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )
    return report, aggregate, prediction


def test_exact_qwen_audit_accepts_cache_prediction_iob2_and_fresh_aggregate(tmp_path):
    report, _aggregate, _prediction = _audit(tmp_path)

    assert report["ok"] is True
    assert report["checks"]["matrix"] == {
        "expected_cells": 1, "cache_cells": 1, "prediction_cells": 1,
        "rows_per_cell": 2, "total_rows": 2,
    }
    assert report["checks"]["reuse"] == {
        "existing_full_cells": 0, "prefix_cells": 0,
        "continuation_full_cells": 0, "current_full_cells": 1,
    }


def test_exact_qwen_audit_distinguishes_original_reuse_from_continuation_cells(tmp_path):
    noisy, cache, predictions, _aggregate, _prediction = _write_minimal_matrix(
        tmp_path, manifest_git_sha="c" * 40,
    )

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=("c" * 40,),
        existing_full_producer_git_shas=(),
        datasets=("msra",), noises=("BT",), seeds=(13,), expected_rows=2,
        expected_existing_full_cells=0, expected_prefix_cells=0,
    )

    assert report["ok"] is True
    assert report["checks"]["reuse"] == {
        "existing_full_cells": 0, "prefix_cells": 0,
        "continuation_full_cells": 1, "current_full_cells": 0,
    }

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=("c" * 40,),
        existing_full_producer_git_shas=("c" * 40,),
        datasets=("msra",), noises=("BT",), seeds=(13,), expected_rows=2,
        expected_existing_full_cells=1, expected_prefix_cells=0,
    )

    assert report["ok"] is True
    assert report["checks"]["reuse"] == {
        "existing_full_cells": 1, "prefix_cells": 0,
        "continuation_full_cells": 0, "current_full_cells": 0,
    }


def test_exact_qwen_audit_rejects_stale_aggregate(tmp_path):
    report, aggregate, prediction = _audit(tmp_path)
    assert report["ok"] is True
    older = prediction.stat().st_mtime - 10
    os.utime(aggregate, (older, older))

    noisy = tmp_path / "noisy"
    cache = tmp_path / "cache"
    predictions = tmp_path / "predictions"
    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )

    assert report["ok"] is False
    assert "stale_aggregate" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_rejects_credential_material_in_cache(tmp_path):
    report, _aggregate, _prediction = _audit(tmp_path, credential=True)

    assert report["ok"] is False
    assert "credential_leak" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_recursively_rejects_failure_tmp_files(tmp_path):
    noisy, cache, predictions, _aggregate, _prediction = _write_minimal_matrix(tmp_path)
    failures = cache / TAG / "failures"
    failures.mkdir()
    (failures / "interrupted.json.tmp").write_text("partial", encoding="utf-8")

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )

    assert "temporary_cache" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_rejects_sensitive_failure_artifact(tmp_path):
    noisy, cache, predictions, _aggregate, _prediction = _write_minimal_matrix(tmp_path)
    failures = cache / TAG / "failures"
    failures.mkdir()
    (failures / "verifier_semantic_failure__bad.json").write_text(json.dumps({
        "schema": "selectdenoise-verifier-semantic-failure-v1",
        "credential": "must-not-be-present",
        "gold_tags": ["O"],
    }), encoding="utf-8")

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )

    assert "failure_provenance" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_rejects_any_terminal_fallback(tmp_path):
    report, _aggregate, prediction = _audit(tmp_path)
    assert report["ok"] is True
    rows = [json.loads(line) for line in prediction.read_text(encoding="utf-8").splitlines()]
    rows[0]["terminal_fallback_count"] = 1
    prediction.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=tmp_path / "noisy", cache_root=tmp_path / "cache",
        predictions_root=tmp_path / "predictions", cache_tag=TAG,
        bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )

    assert report["ok"] is False
    assert "prediction_validation" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_rejects_missing_or_incorrect_aggregate_metric(tmp_path):
    report, aggregate, _prediction = _audit(tmp_path)
    assert report["ok"] is True
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    cell = payload["selectdenoise_contextual_lattice|msra|BT"]
    del cell["p_mean"]
    cell["f1_mean"] = 0.25
    aggregate.write_text(json.dumps(payload), encoding="utf-8")

    report = qwen_full_audit.audit_qwen_full(
        noisy_root=tmp_path / "noisy", cache_root=tmp_path / "cache",
        predictions_root=tmp_path / "predictions", cache_tag=TAG,
        bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=(), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=2, expected_existing_full_cells=0,
        expected_prefix_cells=0,
    )

    assert report["ok"] is False
    assert "aggregate_validation" in {item["code"] for item in report["blockers"]}


def test_exact_qwen_audit_rereads_and_hash_verifies_prefix_source(tmp_path):
    noisy, cache, predictions, _aggregate, _prediction = _write_minimal_matrix(
        tmp_path, row_count=100, prefix_source=True,
    )
    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=("d" * 40,), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=100, expected_existing_full_cells=0,
        expected_prefix_cells=1,
    )
    assert report["ok"] is True

    source = run_multiseed._provider_cache_cell_path(
        cache, run_multiseed.QWEN_REDUCED_CACHE_TAG,
        "selectdenoise_contextual_lattice", "msra", "BT", 13,
    )
    source.write_bytes(source.read_bytes() + b"\n")
    report = qwen_full_audit.audit_qwen_full(
        noisy_root=noisy, cache_root=cache, predictions_root=predictions,
        cache_tag=TAG, bundle_hash="b" * 64, producer_git_sha="a" * 40,
        compatible_git_shas=("d" * 40,), datasets=("msra",), noises=("BT",),
        seeds=(13,), expected_rows=100, expected_existing_full_cells=0,
        expected_prefix_cells=1,
    )
    assert report["ok"] is False
    assert "cache_validation" in {item["code"] for item in report["blockers"]}
