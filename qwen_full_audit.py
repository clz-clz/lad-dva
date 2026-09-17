"""Exact read-only audit for the formal Qwen no-thinking full experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from official_provider_cache import QWEN_REVISION, read_provider_cell
from official_contract import validate_verifier_semantic_retry_evidence
from metrics import compute_prf1, compute_ser
from run_multiseed import (
    CONFIGURATIONS,
    DATASETS,
    NOISE_TYPES,
    OFFICIAL_SAMPLE_SIZE,
    QWEN_REDUCED_CACHE_TAG,
    SEEDS,
    _is_legal_iob2,
    _provider_cache_cell_key,
    _provider_cache_cell_path,
    _provider_cache_index_path,
    _provider_cache_index_path_matches,
    _provider_cache_manifest,
    _read_provider_cache_index_cell,
    _read_provider_cache_manifest,
    _validate_contextual_provider_evidence,
    _validate_contextual_replay_prediction,
    _validate_replay_cache_identity,
    _validate_replay_source_rows,
    _valid_tags,
)


FORMAL_TAG = "qwen32b-contextual-nothink-v1"
CONFIG_NAME = "selectdenoise_contextual_lattice"
_FORBIDDEN_CREDENTIAL_FIELDS = frozenset({
    "api_key", "authorization", "password", "secret", "access_token",
    "credential", "credentials",
})
_FORBIDDEN_GOLD_FIELDS = frozenset({"gold_tags", "ner_tags", "gold_labels"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def _contains_credential_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_CREDENTIAL_FIELDS:
                return True
            if _contains_credential_field(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_credential_field(item) for item in value)
    return False


def _contains_gold_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_GOLD_FIELDS:
                return True
            if _contains_gold_field(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_gold_field(item) for item in value)
    return False


def _block(blockers: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    blockers.append({"code": code, "message": message, **details})


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _require_close(actual: Any, expected: float, field: str) -> None:
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError(f"aggregate {field} does not match verified predictions")


def _noisy_path(root: Path, dataset: str, noise: str, seed: int) -> Path:
    return Path(root) / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"


def _prediction_path(root: Path, tag: str, dataset: str, noise: str, seed: int) -> Path:
    return Path(root) / (
        f"pred_seed{seed}__{CONFIG_NAME}__{dataset}__{noise}__{tag}.jsonl"
    )


def audit_qwen_full(
    *,
    noisy_root: Path,
    cache_root: Path,
    predictions_root: Path,
    cache_tag: str,
    bundle_hash: str,
    producer_git_sha: str,
    compatible_git_shas: Sequence[str],
    existing_full_producer_git_shas: Sequence[str] = (),
    datasets: Sequence[str] = DATASETS,
    noises: Sequence[str] = NOISE_TYPES,
    seeds: Sequence[int] = SEEDS,
    expected_rows: int = OFFICIAL_SAMPLE_SIZE,
    expected_existing_full_cells: int = 8,
    expected_prefix_cells: int = 12,
) -> dict[str, Any]:
    """Validate the exact cache/prediction/aggregate matrix without mutation."""
    noisy_root, cache_root, predictions_root = map(
        Path, (noisy_root, cache_root, predictions_root)
    )
    blockers: list[dict[str, Any]] = []
    expected_cells = len(datasets) * len(noises) * len(seeds)
    expected_keys = {
        _provider_cache_cell_key(CONFIG_NAME, dataset, noise, seed)
        for dataset in datasets for noise in noises for seed in seeds
    }
    if cache_tag != FORMAL_TAG:
        _block(blockers, "invalid_tag", f"audit tag must be exactly {FORMAL_TAG}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", bundle_hash):
        _block(blockers, "invalid_bundle_hash", "bundle hash must be 64 hexadecimal characters")
    allowed_git_shas = {producer_git_sha.lower(), *(
        str(value).lower() for value in compatible_git_shas
    )}
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in allowed_git_shas):
        _block(blockers, "invalid_git_sha", "producer Git SHAs must be 40 hexadecimal characters")
    existing_full_producers = {
        str(value).lower() for value in existing_full_producer_git_shas
    }
    if (any(not re.fullmatch(r"[0-9a-f]{40}", value)
            for value in existing_full_producers)
            or not existing_full_producers <= allowed_git_shas
            or producer_git_sha.lower() in existing_full_producers):
        _block(
            blockers, "invalid_existing_full_producer_git_sha",
            "original full-cache producer Git SHAs must be compatible prior producers",
        )

    index_path = _provider_cache_index_path(cache_root, cache_tag)
    index_cells: Mapping[str, Any] = {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidate = index.get("cells") if isinstance(index, Mapping) else None
        if not isinstance(candidate, Mapping):
            raise ValueError("index has no cells mapping")
        index_cells = candidate
        actual_keys = set(index_cells)
        if actual_keys != expected_keys:
            _block(
                blockers, "cache_matrix", "provider-cache index is not the exact matrix",
                missing=sorted(expected_keys - actual_keys),
                extra=sorted(actual_keys - expected_keys),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _block(blockers, "cache_index", f"provider-cache index is invalid: {exc}")

    expected_cache_paths = {
        _provider_cache_cell_path(
            cache_root, cache_tag, CONFIG_NAME, dataset, noise, seed,
        ).resolve()
        for dataset in datasets for noise in noises for seed in seeds
    }
    target_dir = cache_root / cache_tag
    actual_cache_paths = {
        path.resolve() for path in target_dir.glob("provider_*.jsonl")
    } if target_dir.is_dir() else set()
    if actual_cache_paths != expected_cache_paths:
        _block(
            blockers, "cache_files", "provider-cache files are not the exact matrix",
            missing=sorted(str(path) for path in expected_cache_paths - actual_cache_paths),
            extra=sorted(str(path) for path in actual_cache_paths - expected_cache_paths),
        )
    temporary_cache = sorted(str(path) for path in target_dir.rglob("*.tmp")) \
        if target_dir.is_dir() else []
    if temporary_cache:
        _block(blockers, "temporary_cache", "temporary provider-cache files remain",
               files=temporary_cache)

    failure_paths = sorted(target_dir.glob("failures/*.json")) \
        if target_dir.is_dir() else []
    for failure_path in failure_paths:
        try:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            if not isinstance(failure, Mapping):
                raise ValueError("failure provenance is not an object")
            if (_contains_credential_field(failure) or _contains_gold_field(failure)):
                raise ValueError("failure provenance contains credential or gold fields")
            outcome = failure.get("terminal_outcome")
            interrupted = outcome == "nonsemantic_interruption"
            required = {
                "schema", "cache_tag", "config", "dataset", "noise", "seed",
                "row_index", "input_digest", "git_sha", "terminal_outcome",
                "attempts",
            } | ({"terminal_error_type"} if interrupted else set())
            if set(failure) != required:
                raise ValueError("failure provenance fields are malformed")
            if (failure.get("schema") != "selectdenoise-verifier-semantic-failure-v1"
                    or failure.get("cache_tag") != cache_tag
                    or failure.get("config") != CONFIG_NAME
                    or failure.get("dataset") not in datasets
                    or failure.get("noise") not in noises
                    or failure.get("seed") not in seeds
                    or type(failure.get("row_index")) is not int
                    or not 0 <= failure["row_index"] < expected_rows
                    or re.fullmatch(r"[0-9a-f]{64}", str(failure.get("input_digest", ""))) is None
                    or str(failure.get("git_sha", "")).lower() not in allowed_git_shas
                    or outcome not in {
                        "semantic_retries_exhausted", "nonsemantic_interruption",
                    }):
                raise ValueError("failure provenance identity is malformed")
            if interrupted and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,127}",
                str(failure.get("terminal_error_type", "")),
            ) is None:
                raise ValueError("failure provenance interruption type is malformed")
            attempts = failure.get("attempts")
            if not isinstance(attempts, list):
                raise ValueError("failure provenance attempts are malformed")
            validate_verifier_semantic_retry_evidence(
                attempts, exhausted=not interrupted, interrupted=interrupted,
            )
        except Exception as exc:  # noqa: BLE001 - audit every failure artifact
            _block(
                blockers, "failure_provenance", str(exc),
                file=str(failure_path),
            )

    expected_noisy_paths = {
        _noisy_path(noisy_root, dataset, noise, seed).resolve()
        for dataset in datasets for noise in noises for seed in seeds
    }
    actual_noisy_paths = {
        path.resolve() for path in noisy_root.glob("noisy_seed*__N200.jsonl")
    } if noisy_root.is_dir() else set()
    if actual_noisy_paths != expected_noisy_paths:
        _block(
            blockers, "noisy_matrix", "canonical noisy files are not the exact matrix",
            missing=sorted(str(path) for path in expected_noisy_paths - actual_noisy_paths),
            extra=sorted(str(path) for path in actual_noisy_paths - expected_noisy_paths),
        )

    provider_identity = {
        "provider": "vllm", "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": f"Qwen/Qwen3-32B-AWQ@{QWEN_REVISION}",
        "revision": QWEN_REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": False, "thinking_mode": "nothink",
    }
    cache_cells = prediction_cells = total_rows = 0
    prefix_cells = existing_full_cells = 0
    continuation_full_cells = current_full_cells = 0
    source_hashes: dict[str, str] = {}
    prediction_paths: list[Path] = []
    verified_metrics: dict[tuple[str, str, int], dict[str, float]] = {}
    aggregate_keys = {
        f"{CONFIG_NAME}|{dataset}|{noise}"
        for dataset in datasets for noise in noises
    }

    for dataset in datasets:
        valid_tags = _valid_tags(dataset)
        for noise in noises:
            for seed in seeds:
                key = _provider_cache_cell_key(CONFIG_NAME, dataset, noise, seed)
                noisy_path = _noisy_path(noisy_root, dataset, noise, seed)
                try:
                    source_rows = _read_jsonl(noisy_path)
                    source_hashes[noisy_path.name] = _sha256_file(noisy_path)
                    if len(source_rows) != expected_rows:
                        raise ValueError(
                            f"expected {expected_rows} rows, found {len(source_rows)}"
                        )
                    for index, row in enumerate(source_rows):
                        tokens, dirty, gold = (
                            row.get("tokens"), row.get("dirty_tags"), row.get("ner_tags")
                        )
                        if (not all(isinstance(value, list) for value in (tokens, dirty, gold))
                                or not len(tokens) == len(dirty) == len(gold)
                                or not all(isinstance(item, str)
                                           for value in (tokens, dirty, gold) for item in value)
                                or any(tag not in valid_tags for tag in dirty + gold)
                                or not _is_legal_iob2(gold)):
                            raise ValueError(f"row {index} violates the noisy-data contract")
                except Exception as exc:  # noqa: BLE001 - audit every cell
                    _block(blockers, "noisy_validation", str(exc), cell=key)
                    continue

                expected_manifest = _provider_cache_manifest(
                    source_rows, CONFIGURATIONS[CONFIG_NAME], dataset=dataset,
                    noise=noise, seed=seed, git_sha=producer_git_sha.lower(),
                    bundle_hash=bundle_hash.lower(), enable_thinking=False,
                )
                cell_path = _provider_cache_cell_path(
                    cache_root, cache_tag, CONFIG_NAME, dataset, noise, seed,
                )
                try:
                    indexed = _read_provider_cache_index_cell(cache_root, cache_tag, key)
                    if (not _provider_cache_index_path_matches(indexed["path"], cell_path)
                            or indexed["row_count"] != expected_rows):
                        raise ValueError("cache index path or row count mismatch")
                    cached = read_provider_cell(
                        cell_path, str(indexed["sha256"]), expected_rows,
                    )
                    manifest = _read_provider_cache_manifest(cell_path)
                    _validate_replay_cache_identity(
                        manifest, expected_manifest,
                        compatible_git_shas=compatible_git_shas,
                    )
                    if (indexed["git_sha"] != manifest.get("git_sha")
                            or indexed["model_revision"] != manifest.get("model_revision")):
                        raise ValueError("cache index provenance mismatch")
                    _validate_replay_source_rows(source_rows, cached)
                    for record in cached:
                        _validate_contextual_provider_evidence(
                            record["provider_metadata"], provider_identity, noise,
                        )
                    if _contains_credential_field({"manifest": manifest, "records": cached}):
                        _block(blockers, "credential_leak", "credential field in provider cache",
                               cell=key)
                    derived = manifest.get("derived_from")
                    if derived is not None:
                        required = {
                            "cache_tag", "sha256", "source_row_count",
                            "selected_rows", "producer_git_sha",
                        }
                        if (not isinstance(derived, Mapping) or set(derived) != required
                                or derived.get("source_row_count") != 100
                                or derived.get("selected_rows") != [0, 100]
                                or not re.fullmatch(
                                    r"[0-9a-f]{64}", str(derived.get("sha256", ""))
                                )
                                or not re.fullmatch(
                                    r"[0-9a-f]{40}",
                                    str(derived.get("producer_git_sha", "")),
                                )):
                            raise ValueError("cache prefix provenance is malformed")
                        if derived["cache_tag"] != QWEN_REDUCED_CACHE_TAG:
                            raise ValueError("cache prefix source tag is not the approved reduced tag")
                        source_producer = str(derived["producer_git_sha"]).lower()
                        if source_producer not in allowed_git_shas:
                            raise ValueError("cache prefix producer Git SHA is not approved")
                        source_path = _provider_cache_cell_path(
                            cache_root, QWEN_REDUCED_CACHE_TAG, CONFIG_NAME,
                            dataset, noise, seed,
                        )
                        source_index = _read_provider_cache_index_cell(
                            cache_root, QWEN_REDUCED_CACHE_TAG, key,
                        )
                        if (not _provider_cache_index_path_matches(
                                source_index["path"], source_path)
                                or source_index["row_count"] != 100
                                or str(source_index["sha256"]).lower()
                                != str(derived["sha256"]).lower()
                                or str(source_index["git_sha"]).lower() != source_producer
                                or source_index["model_revision"] != QWEN_REVISION):
                            raise ValueError("cache prefix source index provenance mismatch")
                        source_cached = read_provider_cell(
                            source_path, str(source_index["sha256"]), 100,
                        )
                        source_manifest = _read_provider_cache_manifest(source_path)
                        source_expected = _provider_cache_manifest(
                            source_rows[:100], CONFIGURATIONS[CONFIG_NAME],
                            dataset=dataset, noise=noise, seed=seed,
                            git_sha=source_producer, bundle_hash=bundle_hash.lower(),
                            enable_thinking=False,
                        )
                        _validate_replay_cache_identity(
                            source_manifest, source_expected,
                            compatible_git_shas=[source_producer],
                        )
                        if (str(source_manifest.get("git_sha", "")).lower()
                                != source_producer
                                or source_index["git_sha"] != source_manifest.get("git_sha")):
                            raise ValueError("cache prefix source manifest provenance mismatch")
                        _validate_replay_source_rows(source_rows[:100], source_cached)
                        for record in source_cached:
                            _validate_contextual_provider_evidence(
                                record["provider_metadata"], provider_identity, noise,
                            )
                        if cached[:100] != source_cached:
                            raise ValueError("cache prefix rows differ from the approved source")
                        if _contains_credential_field(
                                {"manifest": source_manifest, "records": source_cached}):
                            raise ValueError("credential field in cache prefix source")
                        prefix_cells += 1
                    else:
                        manifest_git_sha = str(manifest.get("git_sha", "")).lower()
                        if manifest_git_sha == producer_git_sha.lower():
                            current_full_cells += 1
                        elif manifest_git_sha in existing_full_producers:
                            existing_full_cells += 1
                        else:
                            continuation_full_cells += 1
                    cache_cells += 1
                except Exception as exc:  # noqa: BLE001 - audit every cell
                    code = "gold_leak" if "gold labels" in str(exc).lower() else "cache_validation"
                    _block(blockers, code, str(exc), cell=key)
                    continue

                prediction_path = _prediction_path(
                    predictions_root, cache_tag, dataset, noise, seed,
                )
                try:
                    _validate_contextual_replay_prediction(
                        prediction_path, dataset, noise, str(indexed["sha256"]),
                        expected_count=expected_rows, enable_thinking=False,
                    )
                    prediction_rows = _read_jsonl(prediction_path)
                    for index, (source, prediction) in enumerate(
                            zip(source_rows, prediction_rows)):
                        if (prediction.get("tokens") != source.get("tokens")
                                or prediction.get("gold_tags") != source.get("ner_tags")):
                            raise ValueError(f"prediction source mismatch at row {index}")
                    if _contains_credential_field(prediction_rows):
                        _block(blockers, "credential_leak", "credential field in predictions",
                               cell=key)
                    prediction_cells += 1
                    total_rows += len(prediction_rows)
                    prediction_paths.append(prediction_path)
                    gold = [row["gold_tags"] for row in prediction_rows]
                    pred = [row["pred_tags"] for row in prediction_rows]
                    verified_metrics[(dataset, noise, seed)] = {
                        **compute_prf1(gold, pred), "ser": compute_ser(pred),
                    }
                except Exception as exc:  # noqa: BLE001 - audit every cell
                    _block(blockers, "prediction_validation", str(exc), cell=key)

    expected_prediction_paths = {
        _prediction_path(predictions_root, cache_tag, dataset, noise, seed).resolve()
        for dataset in datasets for noise in noises for seed in seeds
    }
    actual_prediction_paths = {
        path.resolve() for path in predictions_root.glob(f"pred_*__{cache_tag}.jsonl")
    } if predictions_root.is_dir() else set()
    if actual_prediction_paths != expected_prediction_paths:
        _block(
            blockers, "prediction_matrix", "prediction files are not the exact matrix",
            missing=sorted(str(path) for path in expected_prediction_paths - actual_prediction_paths),
            extra=sorted(str(path) for path in actual_prediction_paths - expected_prediction_paths),
        )
    temporary_predictions = sorted(
        str(path) for path in predictions_root.glob(f"pred_*__{cache_tag}.jsonl.tmp")
    ) if predictions_root.is_dir() else []
    if temporary_predictions:
        _block(blockers, "temporary_prediction", "temporary prediction files remain",
               files=temporary_predictions)

    aggregate_path = predictions_root / f"aggregated__{cache_tag}.json"
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not isinstance(aggregate, Mapping) or set(aggregate) != aggregate_keys:
            raise ValueError("aggregate keys do not match the exact dataset/noise matrix")
        for key, cell in aggregate.items():
            per_seed = cell.get("per_seed") if isinstance(cell, Mapping) else None
            if (cell.get("n_seeds") != len(seeds)
                    or not isinstance(per_seed, list)
                    or {row.get("seed") for row in per_seed if isinstance(row, Mapping)}
                    != set(seeds)):
                raise ValueError(f"aggregate seed coverage is invalid for {key}")
            _config, dataset, noise = key.split("|", 2)
            try:
                expected_per_seed = {
                    seed: verified_metrics[(dataset, noise, seed)] for seed in seeds
                }
            except KeyError as exc:
                raise ValueError(
                    f"aggregate cannot be verified because prediction metrics are missing for {key}"
                ) from exc
            actual_per_seed = {row["seed"]: row for row in per_seed}
            for seed, expected in expected_per_seed.items():
                for field in ("precision", "recall", "f1", "ser"):
                    _require_close(
                        actual_per_seed[seed].get(field), expected[field],
                        f"{key}.per_seed[{seed}].{field}",
                    )
            aggregate_fields = {
                "p": "precision", "r": "recall", "f1": "f1", "ser": "ser",
            }
            for prefix, metric_name in aggregate_fields.items():
                values = [expected_per_seed[seed][metric_name] for seed in seeds]
                mean, std = _mean_std(values)
                _require_close(cell.get(f"{prefix}_mean"), mean, f"{key}.{prefix}_mean")
                _require_close(cell.get(f"{prefix}_std"), std, f"{key}.{prefix}_std")
        if prediction_paths and aggregate_path.stat().st_mtime < max(
                path.stat().st_mtime for path in prediction_paths):
            _block(blockers, "stale_aggregate", "aggregate predates a prediction file")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _block(blockers, "aggregate_validation", str(exc))

    if existing_full_cells != expected_existing_full_cells:
        _block(
            blockers, "existing_full_reuse_count",
            f"expected {expected_existing_full_cells} existing full cells, found {existing_full_cells}",
        )
    if prefix_cells != expected_prefix_cells:
        _block(
            blockers, "prefix_reuse_count",
            f"expected {expected_prefix_cells} prefix cells, found {prefix_cells}",
        )

    return {
        "schema": "qwen-contextual-nothink-full-audit-v1",
        "ok": not blockers,
        "tag": cache_tag,
        "checks": {
            "matrix": {
                "expected_cells": expected_cells,
                "cache_cells": cache_cells,
                "prediction_cells": prediction_cells,
                "rows_per_cell": expected_rows,
                "total_rows": total_rows,
            },
            "reuse": {
                "existing_full_cells": existing_full_cells,
                "prefix_cells": prefix_cells,
                "continuation_full_cells": continuation_full_cells,
                "current_full_cells": current_full_cells,
            },
            "source_sha256": source_hashes,
            "aggregate": str(aggregate_path),
        },
        "blockers": blockers,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-root", type=Path, default=Path("results_multiseed"))
    parser.add_argument("--provider-cache-root", type=Path, default=Path("provider_cache"))
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions_multiseed"))
    parser.add_argument("--provider-cache-tag", choices=(FORMAL_TAG,), default=FORMAL_TAG)
    parser.add_argument("--bundle-hash", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--compatible-provider-cache-git-sha", action="append", default=[])
    parser.add_argument("--existing-full-producer-git-sha", action="append", default=[])
    parser.add_argument("--expected-existing-full-cells", type=int, default=8)
    parser.add_argument("--expected-prefix-cells", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_qwen_full(
        noisy_root=args.noisy_root, cache_root=args.provider_cache_root,
        predictions_root=args.predictions_root, cache_tag=args.provider_cache_tag,
        bundle_hash=args.bundle_hash, producer_git_sha=args.producer_git_sha,
        compatible_git_shas=args.compatible_provider_cache_git_sha,
        existing_full_producer_git_shas=args.existing_full_producer_git_sha,
        expected_existing_full_cells=args.expected_existing_full_cells,
        expected_prefix_cells=args.expected_prefix_cells,
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
