"""Staged Qwen no-thinking runner for the locked contextual studies.

Provider work is written only to immutable, gold-free provider-cache cells.
Predictions are produced later from those cells in a credential-free replay.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import contextual_studies as study
import run_multiseed
from contextual_lattice_runtime import LOCKED_BUNDLE_MANIFEST_HASH
from official_provider_cache import QWEN_MODEL, QWEN_REVISION, QWEN_SERVED_MODEL


STUDY_TAG = "qwen32b-contextual-nothink-studies-v1"
SOURCE_CACHE_TAG = f"{STUDY_TAG}-source-r15"
REVIEWER_CACHE_TAG = f"{STUDY_TAG}-minus-reviewer-weighting-r15"
DEFAULT_ROOT = Path("contextual_studies") / STUDY_TAG
CONFIG_NAME = "selectdenoise_contextual_lattice"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_sha() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip().lower()
    if not GIT_SHA_RE.fullmatch(value):
        raise RuntimeError("git rev-parse did not return a full Git SHA")
    return value


def _require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if status:
        raise RuntimeError("paid study stages require a clean Git worktree")


def _source_files() -> dict[str, str]:
    names = (
        "contextual_studies.py", "run_qwen_contextual_studies.py",
        "analyze_contextual_studies.py", "run_multiseed.py",
        "multi_agent_v2.py", "live_backbone.py", "official_provider_cache.py",
        "official_contract.py", "contextual_lattice_runtime.py",
        "selectdenoise_contextual_lattice.py",
    )
    return {name: _sha256_file(Path(name)) for name in names}


def verify_frozen_inputs(root: Path) -> dict[str, str]:
    """Re-hash the exact 225-file bank before every study phase."""
    input_manifest_path = root / "input_manifest.json"
    frozen_source_manifest_path = root / "frozen_source_manifest.json"
    frozen_input_manifest_path = root / "frozen_input_manifest.json"
    selection_path = root / "selection.json"
    if (not input_manifest_path.is_file()
            or not frozen_source_manifest_path.is_file()
            or not frozen_input_manifest_path.is_file()
            or not selection_path.is_file()):
        raise FileNotFoundError(
            "prepared input manifests and selection are required"
        )
    if (_sha256_file(frozen_source_manifest_path)
            != study.FROZEN_SOURCE_MANIFEST_SHA256
            or _sha256_file(frozen_input_manifest_path)
            != study.FROZEN_INPUT_MANIFEST_SHA256):
        raise ValueError("prepared frozen manifest trust anchor is invalid")
    frozen_source_manifest = json.loads(
        frozen_source_manifest_path.read_text(encoding="utf-8")
    )
    authoritative = frozen_source_manifest.get("input_files_sha256")
    if (frozen_source_manifest.get("schema_version")
            != study.STUDY_MANIFEST_SCHEMA
            or frozen_source_manifest.get("input_manifest_sha256")
            != study.FROZEN_INPUT_MANIFEST_SHA256
            or frozen_source_manifest.get("selection_sha256")
            != study.FROZEN_SELECTION_SHA256
            or not isinstance(authoritative, Mapping)):
        raise ValueError("prepared frozen source manifest identity is invalid")
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("input_files_sha256")
    if (manifest.get("frozen_source_manifest_sha256")
            != study.FROZEN_SOURCE_MANIFEST_SHA256
            or manifest.get("frozen_selection_sha256")
            != study.FROZEN_SELECTION_SHA256
            or not isinstance(expected, Mapping)
            or dict(expected) != dict(authoritative)):
        raise ValueError("prepared frozen-input provenance is invalid")
    expected_keys = set(expected)
    actual_paths = sorted((root / "inputs").rglob("*.jsonl"))
    actual_keys = {path.relative_to(root).as_posix() for path in actual_paths}
    if actual_keys != expected_keys:
        raise ValueError("prepared frozen-input namespace is incomplete or contains extras")
    for relative, expected_sha in expected.items():
        path = root / relative
        if (not isinstance(expected_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
                or _sha256_file(path) != expected_sha):
            raise ValueError(f"prepared frozen-input SHA mismatch: {relative}")
    if _sha256_file(selection_path) != study.FROZEN_SELECTION_SHA256:
        raise ValueError("prepared frozen selection SHA mismatch")
    return dict(expected)


def _protocol() -> dict[str, Any]:
    return {
        "ablation_ratio": study.OFFICIAL_RATIO,
        "ablation_variants": list(study.ABLATION_VARIANTS),
        "datasets": list(study.DATASETS),
        "noise_types": list(study.NOISE_TYPES),
        "seeds": list(study.SEEDS),
        "sample_size": study.SAMPLE_SIZE,
        "gradient_ratios": list(study.GRADIENT_RATIOS),
        "test_groups_per_dataset": study.TEST_GROUPS_PER_DATASET,
        "primary_ablation_rows": 450,
        "gradient_rows": 2250,
        "failure_policy": "abort",
        "provider_timeout_seconds": 600.0,
        "outer_timeout_seconds": 7200.0,
        "sentence_concurrency": 32,
        "provider_request_concurrency": 32,
        "sdk_max_retries": 2,
        "thinking": False,
        "no_dfa": True,
        "no_exception_fallback": True,
        "noise_strength_not_exposed_to_prompt": True,
        "frozen_input_source": study.SOURCE_TAG,
    }


def ensure_manifest(root: Path, git_sha: str) -> dict[str, Any]:
    verify_frozen_inputs(root)
    input_manifest = root / "input_manifest.json"
    selection = root / "selection.json"
    if not input_manifest.is_file() or not selection.is_file():
        raise FileNotFoundError("prepare the frozen study inputs before creating the manifest")
    manifest = study.build_study_manifest(
        study_tag=STUDY_TAG,
        source_tag=study.SOURCE_TAG,
        git_sha=git_sha,
        source_files=_source_files(),
        provider={
            "provider": "vllm", "model": QWEN_MODEL,
            "served_model": QWEN_SERVED_MODEL, "revision": QWEN_REVISION,
            "structured_api": "chat-completions-json-schema",
            "schema_transport": "openai-chat-json-schema",
        },
        protocol=_protocol(),
    )
    manifest.update({
        "input_manifest_sha256": _sha256_file(input_manifest),
        "selection_sha256": _sha256_file(selection),
        "coverage": {
            "ablation_cells": 270,
            "ablation_rows_full_namespace": 54000,
            "ablation_rows_primary_test": 2700,
            "gradient_cells": 75,
            "gradient_rows": 2250,
        },
    })
    path = root / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("study manifest collision with different content")
    else:
        study._write_json_atomic(path, manifest)
    return manifest


def _input_rows(root: Path, dataset: str, noise: str, seed: int, ratio: float) -> list[dict[str, Any]]:
    rows = study._input_cell_rows(root, dataset, noise, seed, ratio)
    valid = study._valid_tags(dataset)
    for index, row in enumerate(rows):
        tokens, gold, dirty = row.get("tokens"), row.get("ner_tags"), row.get("dirty_tags")
        if (not isinstance(tokens, list) or not isinstance(gold, list)
                or not isinstance(dirty, list)
                or not len(tokens) == len(gold) == len(dirty)
                or any(tag not in valid for tag in gold + dirty)):
            raise ValueError(f"malformed study input row {dataset}/{noise}/{seed}:{index}")
    return rows


def _study_identity(
    root: Path, *, kind: str, variant: str, dataset: str, noise: str,
    seed: int, ratio: float, coordinates: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    input_path = study.study_input_path(root, dataset, noise, seed, ratio)
    return {
        "schema": "qwen-contextual-studies-v1",
        "study_tag": STUDY_TAG,
        "kind": kind,
        "variant": variant,
        "dataset": dataset,
        "noise": noise,
        "seed": seed,
        "ratio": float(ratio),
        "source_file_sha256": _sha256_file(input_path),
        "coordinates_sha256": _sha256_json(list(coordinates)),
    }


def _live_factory(expected_tag: str):
    environment = dict(os.environ)
    environment["BACKBONE_TAG"] = expected_tag
    settings, resolved_tag = run_multiseed._official_settings_from_env(
        [CONFIG_NAME], environment,
    )
    if resolved_tag != expected_tag:
        raise RuntimeError("provider tag normalization changed the study tag")
    run_multiseed._configure_official_request_model(settings)
    if (settings.provider != "vllm" or settings.model != QWEN_MODEL
            or settings.served_model != QWEN_SERVED_MODEL
            or settings.revision != QWEN_REVISION
            or settings.enable_thinking is not False
            or settings.timeout_seconds != 600.0
            or settings.max_retries != 2):
        raise RuntimeError("study provider settings are not the pinned Qwen no-thinking identity")
    from live_backbone import OpenAICompatibleLADRGAdapter

    return run_multiseed._CachedAdapterFactory(
        lambda: OpenAICompatibleLADRGAdapter(settings)
    )


def run_source_cache(root: Path, cache_root: Path, *, max_concurrency: int = 32) -> None:
    """Build the 45-cell r15 provider source used by every ablation."""
    _require_clean_worktree()
    git_sha = _git_sha()
    ensure_manifest(root, git_sha)
    factory = _live_factory(STUDY_TAG)
    pipelines = run_multiseed._import_pipeline(False)
    try:
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                for seed in study.SEEDS:
                    rows = _input_rows(root, dataset, noise, seed, study.OFFICIAL_RATIO)
                    coords = [{"seed": seed, "row_index": index} for index in range(len(rows))]
                    identity = _study_identity(
                        root, kind="ablation", variant="full", dataset=dataset,
                        noise=noise, seed=seed, ratio=study.OFFICIAL_RATIO,
                        coordinates=coords,
                    )
                    asyncio.run(run_multiseed._run_provider_cache_cell(
                        CONFIG_NAME, run_multiseed.CONFIGURATIONS[CONFIG_NAME],
                        dataset, noise, seed, len(rows), pipelines,
                        cache_root=cache_root, cache_tag=SOURCE_CACHE_TAG,
                        max_concurrency=max_concurrency, request_timeout=7200.0,
                        adapter_factory=factory, git_sha=git_sha,
                        bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH,
                        explicit_source_rows=rows, study_identity=identity,
                    ))
    finally:
        factory.close()


def _decorate(
    rows: Sequence[Mapping[str, Any]], *, kind: str, variant: str,
    dataset: str, noise: str, ratio: float,
    coordinates: Sequence[Mapping[str, int]], origin: str,
    selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != len(coordinates):
        raise ValueError("study prediction and coordinate counts differ")
    selected = {
        (int(item["seed"]), int(item["row_index"])): item
        for item in selection
        if item.get("dataset") == dataset and item.get("family") == noise
    }
    decorated = []
    for source, coordinate in zip(rows, coordinates):
        record = dict(source)
        seed = int(coordinate["seed"])
        row_index = int(coordinate["row_index"])
        selected_row = selected.get((seed, row_index), {})
        record.update({
            "study_kind": kind, "study_variant": variant,
            "study_dataset": dataset, "study_noise": noise,
            "study_ratio": float(ratio), "study_seed": seed,
            "study_row_index": row_index, "study_origin": origin,
            "study_source_tag": study.SOURCE_TAG,
            "study_group_digest": selected_row.get(
                "group_digest",
                study.group_digest(
                    dataset, record["tokens"],
                    salt="selectdenoise-contextual-lattice-v1-20260820",
                ),
            ),
        })
        study.validate_study_prediction_row(record, dataset)
        decorated.append(record)
    return decorated


def run_source_replay(root: Path, cache_root: Path, *, max_concurrency: int = 32) -> None:
    """Replay the source cache locally into the full ablation namespace."""
    git_sha = _git_sha()
    ensure_manifest(root, git_sha)
    selection = study.load_selection(root)
    from multi_agent_v2 import _load_contextual_lattice_terminal, run_contextual_lattice_replay

    terminal = _load_contextual_lattice_terminal()
    for dataset in study.DATASETS:
        for noise in study.NOISE_TYPES:
            for seed in study.SEEDS:
                output = study.study_prediction_path(
                    root, "ablation", "full", dataset, noise,
                    study.OFFICIAL_RATIO, seed,
                )
                if output.exists():
                    study.validate_study_prediction_file(
                        output, dataset, expected_count=study.SAMPLE_SIZE,
                    )
                    continue
                rows = _input_rows(root, dataset, noise, seed, study.OFFICIAL_RATIO)
                coords = [{"seed": seed, "row_index": index} for index in range(len(rows))]
                identity = _study_identity(
                    root, kind="ablation", variant="full", dataset=dataset,
                    noise=noise, seed=seed, ratio=study.OFFICIAL_RATIO,
                    coordinates=coords,
                )
                staging = output.with_suffix(".replay.jsonl")
                asyncio.run(run_multiseed._run_contextual_replay_cell(
                    dataset, noise, seed, len(rows), cache_root=cache_root,
                    cache_tag=SOURCE_CACHE_TAG,
                    bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH, git_sha=git_sha,
                    replay_fn=run_contextual_lattice_replay, terminal=terminal,
                    max_concurrency=max_concurrency, prediction_path=staging,
                    enable_thinking=False, explicit_source_rows=rows,
                    study_identity=identity,
                ))
                replayed = study.load_jsonl(staging)
                decorated = _decorate(
                    replayed, kind="ablation", variant="full", dataset=dataset,
                    noise=noise, ratio=study.OFFICIAL_RATIO,
                    coordinates=coords, origin="provider_cache_replay",
                    selection=selection,
                )
                study._write_variant_cell(
                    root, "full", dataset, noise, study.OFFICIAL_RATIO,
                    seed, decorated,
                )
                staging.unlink(missing_ok=True)


def _terminal_replay(
    source: Mapping[str, Any], noisy: Mapping[str, Any], dataset: str,
    terminal: Any, *, anchor: Sequence[str] | None = None,
    weights: Sequence[float] | None = None, margin: float | None = None,
) -> dict[str, Any]:
    from multi_agent_v2 import run_contextual_lattice_replay

    result = run_contextual_lattice_replay(
        tokens=list(noisy["tokens"]), dirty_tags=list(noisy["dirty_tags"]),
        provider_record={
            "anchor_tags": list(anchor if anchor is not None else source["terminal_anchor_tags"]),
            "candidate_paths": [list(path) for path in source["candidate_paths"]],
            "rag_weights": list(weights if weights is not None else source["rag_weights"]),
            "confidence": list(source["confidence"]),
        },
        dataset_name=dataset, terminal=terminal, terminal_margin=margin,
    )
    record = dict(source)
    record.update({
        "pred_tags": list(result["pred_tags"]),
        "terminal_anchor_tags": list(result["terminal_anchor_tags"]),
        "terminal_model_hash": result["terminal_model_hash"],
        "terminal_used_anchor": result["terminal_used_anchor"],
        "terminal_predicted_gain": result["terminal_predicted_gain"],
        "terminal_fallback_count": result["terminal_fallback_count"],
        "fallback_used": False,
    })
    return record


def run_offline_ablation(root: Path) -> None:
    """Derive every provider-free ablation from the validated Qwen source."""
    import multi_agent_v2

    selection = study.load_selection(root)
    terminal = multi_agent_v2._load_contextual_lattice_terminal()
    for dataset in study.DATASETS:
        multi_agent_v2._init_deer(dataset, cache_only=True)
        valid_types = set(multi_agent_v2.DATASET_ENTITY_TYPES[dataset])
        for noise in study.NOISE_TYPES:
            dangling_policy = "demote" if noise == "IF" else "promote"
            for seed in study.SEEDS:
                source_path = study.study_prediction_path(
                    root, "ablation", "full", dataset, noise,
                    study.OFFICIAL_RATIO, seed,
                )
                source = study.validate_study_prediction_file(
                    source_path, dataset, expected_count=study.SAMPLE_SIZE,
                )
                noisy = _input_rows(root, dataset, noise, seed, study.OFFICIAL_RATIO)
                coords = [{"seed": seed, "row_index": index} for index in range(len(source))]
                for variant in (
                    "minus_contextual", "minus_gate", "minus_verifier",
                    "minus_atf_deanchor",
                ):
                    if variant == "minus_atf_deanchor" and noise == "ATF":
                        continue
                    output = study.study_prediction_path(
                        root, "ablation", variant, dataset, noise,
                        study.OFFICIAL_RATIO, seed,
                    )
                    if output.exists():
                        study.validate_study_prediction_file(
                            output, dataset, expected_count=study.SAMPLE_SIZE,
                        )
                        continue
                    if variant == "minus_contextual":
                        derived = []
                        for row in source:
                            item = dict(row)
                            item.update({
                                "pred_tags": list(row["terminal_anchor_tags"]),
                                "terminal_used_anchor": True,
                                "terminal_predicted_gain": 0.0,
                                "terminal_fallback_count": 0,
                                "fallback_used": False,
                            })
                            derived.append(item)
                    elif variant == "minus_gate":
                        derived = [
                            _terminal_replay(
                                row, noisy[index], dataset, terminal,
                                margin=-1.0e9,
                            )
                            for index, row in enumerate(source)
                        ]
                    elif variant == "minus_verifier":
                        derived = []
                        for index, row in enumerate(source):
                            base = multi_agent_v2._base_decode(
                                [list(path) for path in row["candidate_paths"]],
                                [float(value) for value in row["rag_weights"]],
                                list(row["tokens"]), list(noisy[index]["dirty_tags"]),
                                dataset, noise,
                            )
                            base = multi_agent_v2.legalize_noise_aware(
                                base, valid_types, dangling_policy,
                            )
                            derived.append(_terminal_replay(
                                row, noisy[index], dataset, terminal, anchor=base,
                            ))
                    else:
                        derived = [dict(row) for row in source]
                    decorated = _decorate(
                        derived, kind="ablation", variant=variant,
                        dataset=dataset, noise=noise,
                        ratio=study.OFFICIAL_RATIO, coordinates=coords,
                        origin="offline_replay", selection=selection,
                    )
                    study._write_variant_cell(
                        root, variant, dataset, noise, study.OFFICIAL_RATIO,
                        seed, decorated,
                    )


def _run_live_variant_cell(
    root: Path, cache_root: Path, factory: Any, pipelines: Any,
    *, dataset: str, noise: str, seed: int, ratio: float,
    variant: str, cache_tag: str, config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], coordinates: Sequence[Mapping[str, int]],
    kind: str, max_concurrency: int,
) -> Path:
    git_sha = _git_sha()
    identity = _study_identity(
        root, kind=kind, variant=variant, dataset=dataset, noise=noise,
        seed=seed, ratio=ratio, coordinates=coordinates,
    )
    asyncio.run(run_multiseed._run_provider_cache_cell(
        CONFIG_NAME, config, dataset, noise, seed, len(rows), pipelines,
        cache_root=cache_root, cache_tag=cache_tag,
        max_concurrency=max_concurrency, request_timeout=7200.0,
        adapter_factory=factory, git_sha=git_sha,
        bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH,
        explicit_source_rows=rows, study_identity=identity,
    ))
    from multi_agent_v2 import _load_contextual_lattice_terminal, run_contextual_lattice_replay

    terminal = _load_contextual_lattice_terminal()
    staging = root / "predictions" / "staging" / kind / cache_tag / (
        f"pred_seed{seed}__{dataset}__{noise}.jsonl"
    )
    return asyncio.run(run_multiseed._run_contextual_replay_cell(
        dataset, noise, seed, len(rows), cache_root=cache_root,
        cache_tag=cache_tag, bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH,
        git_sha=git_sha, replay_fn=run_contextual_lattice_replay,
        terminal=terminal, max_concurrency=max_concurrency,
        prediction_path=staging, enable_thinking=False,
        explicit_source_rows=rows, study_identity=identity,
        config_override=config,
    ))


def run_atf_deanchor(root: Path, cache_root: Path, *, max_concurrency: int = 32) -> None:
    """Run only the 15 ATF cells whose provider prompts change."""
    _require_clean_worktree()
    ensure_manifest(root, _git_sha())
    selection = study.load_selection(root)
    tag = f"{STUDY_TAG}-minus-atf-deanchor-r15"
    factory = _live_factory(STUDY_TAG)
    pipelines = run_multiseed._import_pipeline(False)
    config = dict(run_multiseed.CONFIGURATIONS[CONFIG_NAME])
    config["deanchor_atf"] = False
    try:
        for dataset in study.DATASETS:
            for seed in study.SEEDS:
                output = study.study_prediction_path(
                    root, "ablation", "minus_atf_deanchor", dataset,
                    "ATF", study.OFFICIAL_RATIO, seed,
                )
                if output.exists():
                    study.validate_study_prediction_file(
                        output, dataset, expected_count=study.SAMPLE_SIZE,
                    )
                    continue
                rows = _input_rows(root, dataset, "ATF", seed, study.OFFICIAL_RATIO)
                coords = [{"seed": seed, "row_index": index} for index in range(len(rows))]
                staging = _run_live_variant_cell(
                    root, cache_root, factory, pipelines, dataset=dataset,
                    noise="ATF", seed=seed, ratio=study.OFFICIAL_RATIO,
                    variant="minus_atf_deanchor", cache_tag=tag,
                    config=config, rows=rows, coordinates=coords,
                    kind="ablation", max_concurrency=max_concurrency,
                )
                decorated = _decorate(
                    study.load_jsonl(staging), kind="ablation",
                    variant="minus_atf_deanchor", dataset=dataset,
                    noise="ATF", ratio=study.OFFICIAL_RATIO,
                    coordinates=coords, origin="provider_cache_replay",
                    selection=selection,
                )
                study._write_variant_cell(
                    root, "minus_atf_deanchor", dataset, "ATF",
                    study.OFFICIAL_RATIO, seed, decorated,
                )
                staging.unlink(missing_ok=True)
    finally:
        factory.close()


async def _reviewer_weighting_cell(
    root: Path, cache_root: Path, factory: Any, terminal: Any, *, dataset: str,
    noise: str, seed: int, selection: Sequence[Mapping[str, Any]],
    max_concurrency: int, git_sha: str,
) -> Path:
    import multi_agent_v2

    output = study.study_prediction_path(
        root, "ablation", "minus_reviewer_weighting", dataset,
        noise, study.OFFICIAL_RATIO, seed,
    )
    if output.exists():
        study.validate_study_prediction_file(
            output, dataset, expected_count=study.SAMPLE_SIZE,
        )
        return output
    source = study.validate_study_prediction_file(
        study.study_prediction_path(
            root, "ablation", "full", dataset, noise,
            study.OFFICIAL_RATIO, seed,
        ),
        dataset, expected_count=study.SAMPLE_SIZE,
    )
    noisy = _input_rows(root, dataset, noise, seed, study.OFFICIAL_RATIO)
    adapter = factory()
    provider_identity = adapter.provider_metadata()
    multi_agent_v2._init_deer(dataset, cache_only=True)
    sem = asyncio.Semaphore(max_concurrency)
    provider_records: list[dict[str, Any] | None] = [None] * len(source)
    coords = [{"seed": seed, "row_index": index} for index in range(len(source))]
    config = dict(run_multiseed.CONFIGURATIONS[CONFIG_NAME])
    cell_identity = _study_identity(
        root, kind="ablation", variant="minus_reviewer_weighting",
        dataset=dataset, noise=noise, seed=seed, ratio=study.OFFICIAL_RATIO,
        coordinates=coords,
    )
    source_cache_shas = sorted({str(row["provider_cache_sha256"]) for row in source})
    if len(source_cache_shas) != 1:
        raise ValueError("reviewer-weighting source rows do not share one provider-cache SHA")
    cell_identity.update({
        "source_provider_cache_sha256": source_cache_shas[0],
        "reviewer_weighting": "uniform",
        "reviewer_stage": "disabled",
        "verifier_policy": "type_contested_only",
    })
    full_manifest = run_multiseed._provider_cache_manifest(
        noisy, config, dataset=dataset, noise=noise, seed=seed,
        git_sha=git_sha, bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH,
        enable_thinking=False, study_identity=cell_identity,
    )
    full_manifest["request_limits"] = {
        "provider_timeout_seconds": 600.0,
        "runner_timeout_seconds": 7200.0,
        "max_concurrency": max_concurrency,
    }
    fragment_root = (
        cache_root / REVIEWER_CACHE_TAG / "rows" / dataset / noise
        / f"seed{seed}"
    )
    fragment_index_path = fragment_root / "progress.json"
    fragment_index: dict[str, Any] = {
        "schema": "qwen-reviewer-row-progress-v1",
        "cell_identity_sha256": _sha256_json(cell_identity),
        "rows": {},
    }
    if fragment_index_path.exists():
        loaded_index = json.loads(fragment_index_path.read_text(encoding="utf-8"))
        if (not isinstance(loaded_index, Mapping)
                or loaded_index.get("schema") != fragment_index["schema"]
                or loaded_index.get("cell_identity_sha256")
                != fragment_index["cell_identity_sha256"]
                or not isinstance(loaded_index.get("rows"), Mapping)):
            raise ValueError("reviewer-weighting row progress index is malformed")
        fragment_index = {
            **fragment_index,
            "rows": dict(loaded_index["rows"]),
        }
    fragment_index_lock = asyncio.Lock()
    cell_path = run_multiseed._provider_cache_cell_path(
        cache_root, REVIEWER_CACHE_TAG, CONFIG_NAME, dataset, noise, seed,
    )
    cell_key = run_multiseed._provider_cache_cell_key(
        CONFIG_NAME, dataset, noise, seed,
    )

    if cell_path.exists():
        indexed = run_multiseed._read_provider_cache_index_cell(
            cache_root, REVIEWER_CACHE_TAG, cell_key,
        )
        cached_manifest = run_multiseed._read_provider_cache_manifest(cell_path)
        run_multiseed._validate_replay_cache_identity(cached_manifest, full_manifest)
        if (not run_multiseed._provider_cache_index_path_matches(
                indexed["path"], cell_path)
                or indexed["row_count"] != len(noisy)):
            raise ValueError("reviewer-weighting cache index identity mismatch")
        loaded = run_multiseed.read_provider_cell(
            cell_path, str(indexed["sha256"]), len(noisy),
        )
        provider_records[:] = loaded

    async def process(index: int) -> None:
        async with sem:
            row = source[index]
            candidates = [list(path) for path in row["candidate_paths"]]
            if not candidates:
                raise RuntimeError(f"source row {index} has no candidates")
            weights = [1.0 / len(candidates)] * len(candidates)
            row_identity = {
                **cell_identity,
                "row_index": index,
                "uniform_weights_sha256": _sha256_json(weights),
            }
            row_manifest = run_multiseed._provider_cache_manifest(
                [noisy[index]], config, dataset=dataset, noise=noise, seed=seed,
                git_sha=git_sha, bundle_hash=LOCKED_BUNDLE_MANIFEST_HASH,
                enable_thinking=False, study_identity=row_identity,
            )
            fragment = fragment_root / f"row{index:03d}.jsonl"
            if fragment.exists():
                fragment_sha = fragment_index["rows"].get(str(index))
                if (not isinstance(fragment_sha, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", fragment_sha)
                        or _sha256_file(fragment) != fragment_sha):
                    raise ValueError(
                        "reviewer-weighting row fragment is not bound to its progress index"
                    )
                cached_manifest = run_multiseed._read_provider_cache_manifest(fragment)
                run_multiseed._validate_replay_cache_identity(
                    cached_manifest, row_manifest,
                )
                cached = run_multiseed.read_provider_cell(fragment, fragment_sha, 1)[0]
                cached["row_index"] = index
                provider_records[index] = cached
                return
            valid_types = set(multi_agent_v2.DATASET_ENTITY_TYPES[dataset])
            dangling_policy = "demote" if noise == "IF" else "promote"
            base = multi_agent_v2._base_decode(
                candidates, weights, list(row["tokens"]),
                list(noisy[index]["dirty_tags"]), dataset, noise,
            )
            base = multi_agent_v2.legalize_noise_aware(
                base, valid_types, dangling_policy,
            )
            reviewer_record = multi_agent_v2._stage_status_record(
                "reviewer", "disabled", provider_identity,
            )
            coder_evidence = [dict(item) for item in row["provider_metadata"]["coder"]]
            state = {
                "tokens": list(row["tokens"]),
                "dirty_tags": list(noisy[index]["dirty_tags"]),
                "candidate_paths": candidates, "rag_weights": weights,
                "dataset_name": dataset, "noise_type": noise,
                "official": True, "contextual_study": True,
                "use_verifier": True, "verify_all": False,
                "verifier_topk": 4,
                "verifier_semantic_max_retries": 2,
                "provider_settings": dict(provider_identity),
                "provider_metadata": {
                    "coder": coder_evidence, "reviewer": [reviewer_record],
                    "verifier": [],
                },
                "structured_requester": adapter.structured_requester,
            }
            if multi_agent_v2._is_type_contested(candidates):
                verifier_result = await asyncio.wait_for(
                    multi_agent_v2.verifier_node(state), timeout=7200.0,
                )
                anchor = list(verifier_result["current_tags"])
                provider_metadata = verifier_result["provider_metadata"]
            else:
                anchor = base
                provider_metadata = {
                    "coder": coder_evidence, "reviewer": [reviewer_record],
                    "verifier": [multi_agent_v2._stage_status_record(
                        "verifier", "skipped_uncontested", provider_identity,
                    )],
                }
            multi_agent_v2._validate_contextual_official_evidence(
                provider_metadata, allow_reviewer_disabled=True,
            )
            record = {
                "row_index": 0,
                "input_digest": run_multiseed._provider_input_digest(noisy[index]),
                "anchor_tags": anchor, "candidate_paths": candidates,
                "rag_weights": weights,
                "confidence": multi_agent_v2._per_position_confidence(
                    candidates, weights, anchor,
                ),
                "provider_metadata": provider_metadata,
                "fallback_used": False,
            }
            run_multiseed.write_provider_cell(fragment, [record], row_manifest)
            fragment_sha = _sha256_file(fragment)
            async with fragment_index_lock:
                fragment_index["rows"][str(index)] = fragment_sha
                study._write_json_atomic(fragment_index_path, fragment_index)
            record["row_index"] = index
            provider_records[index] = record

    tasks = [
        asyncio.create_task(process(index))
        for index in range(len(source))
        if provider_records[index] is None
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if any(row is None for row in provider_records):
        raise RuntimeError("reviewer-weighting provider cache is incomplete")
    if not cell_path.exists():
        records = [row for row in provider_records if row is not None]
        cache_sha = run_multiseed.write_provider_cell(
            cell_path, records, full_manifest,
        )
        cell = {
            "path": str(cell_path), "sha256": cache_sha,
            "row_count": len(records), "git_sha": git_sha,
            "model_revision": QWEN_REVISION,
            "provider_fingerprint": _sha256_json(provider_identity),
        }
        try:
            run_multiseed._update_provider_cache_index(
                cache_root, REVIEWER_CACHE_TAG, key=cell_key, cell=cell,
            )
        except BaseException:
            cell_path.unlink(missing_ok=True)
            raise
    else:
        cache_sha = str(run_multiseed._read_provider_cache_index_cell(
            cache_root, REVIEWER_CACHE_TAG, cell_key,
        )["sha256"])

    predictions = []
    for index, cached in enumerate(provider_records):
        assert cached is not None
        replay = await asyncio.to_thread(
            multi_agent_v2.run_contextual_lattice_replay,
            tokens=list(source[index]["tokens"]),
            dirty_tags=list(noisy[index]["dirty_tags"]),
            provider_record=cached, dataset_name=dataset, terminal=terminal,
        )
        predictions.append({
            "tokens": list(source[index]["tokens"]),
            "gold_tags": list(source[index]["gold_tags"]),
            "pred_tags": list(replay["pred_tags"]),
            "input_digest": cached["input_digest"],
            "candidate_paths": cached["candidate_paths"],
            "rag_weights": cached["rag_weights"],
            "confidence": cached["confidence"],
            "terminal_anchor_tags": list(replay["terminal_anchor_tags"]),
            "terminal_model_hash": replay["terminal_model_hash"],
            "terminal_used_anchor": replay["terminal_used_anchor"],
            "terminal_predicted_gain": replay["terminal_predicted_gain"],
            "terminal_fallback_count": replay["terminal_fallback_count"],
            "provider_cache_sha256": cache_sha,
            "provider_metadata": cached["provider_metadata"],
            "fallback_used": False,
        })
    decorated = _decorate(
        predictions, kind="ablation",
        variant="minus_reviewer_weighting", dataset=dataset, noise=noise,
        ratio=study.OFFICIAL_RATIO, coordinates=coords,
        origin="source_coder_verifier_live", selection=selection,
    )
    return study._write_variant_cell(
        root, "minus_reviewer_weighting", dataset, noise,
        study.OFFICIAL_RATIO, seed, decorated,
    )


def run_reviewer_weighting(
    root: Path, cache_root: Path, *, max_concurrency: int = 32,
) -> None:
    _require_clean_worktree()
    git_sha = _git_sha()
    ensure_manifest(root, git_sha)
    selection = study.load_selection(root)
    factory = _live_factory(STUDY_TAG)
    from multi_agent_v2 import _load_contextual_lattice_terminal

    terminal = _load_contextual_lattice_terminal()
    try:
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                for seed in study.SEEDS:
                    asyncio.run(_reviewer_weighting_cell(
                        root, cache_root, factory, terminal, dataset=dataset,
                        noise=noise, seed=seed, selection=selection,
                        max_concurrency=max_concurrency, git_sha=git_sha,
                    ))
    finally:
        factory.close()


def _write_official_gradient(root: Path, selection: Sequence[Mapping[str, Any]]) -> None:
    for dataset in study.DATASETS:
        for noise in study.NOISE_TYPES:
            selected = study.selected_indices(selection, dataset, noise)
            rows = []
            coords = []
            for item in selected:
                seed, row_index = int(item["seed"]), int(item["row_index"])
                source = study.validate_study_prediction_file(
                    study.study_prediction_path(
                        root, "ablation", "full", dataset, noise,
                        study.OFFICIAL_RATIO, seed,
                    ),
                    dataset, expected_count=study.SAMPLE_SIZE,
                )
                rows.append(dict(source[row_index]))
                coords.append({"seed": seed, "row_index": row_index})
            decorated = _decorate(
                rows, kind="noise-gradient", variant="full",
                dataset=dataset, noise=noise, ratio=study.OFFICIAL_RATIO,
                coordinates=coords, origin="source_reuse", selection=selection,
            )
            study._write_gradient_cell(
                root, "full", dataset, noise, study.OFFICIAL_RATIO, decorated,
            )


def run_noise_gradient(root: Path, cache_root: Path, *, max_concurrency: int = 32) -> None:
    _require_clean_worktree()
    ensure_manifest(root, _git_sha())
    selection = study.load_selection(root)
    _write_official_gradient(root, selection)
    factory = _live_factory(STUDY_TAG)
    pipelines = run_multiseed._import_pipeline(False)
    config = dict(run_multiseed.CONFIGURATIONS[CONFIG_NAME])
    try:
        for ratio in study.GRADIENT_RATIOS:
            if abs(ratio - study.OFFICIAL_RATIO) < 1e-12:
                continue
            tag = f"{STUDY_TAG}-gradient-{study.rate_token(ratio)}"
            for dataset in study.DATASETS:
                for noise in study.NOISE_TYPES:
                    final_path = study.study_prediction_path(
                        root, "noise-gradient", "full", dataset, noise, ratio,
                    )
                    if final_path.exists():
                        study.validate_study_prediction_file(
                            final_path, dataset,
                            expected_count=study.TEST_GROUPS_PER_DATASET,
                        )
                        continue
                    selected = study.selected_indices(selection, dataset, noise)
                    by_seed: dict[int, list[dict[str, Any]]] = {}
                    for item in selected:
                        by_seed.setdefault(int(item["seed"]), []).append(item)
                    staged: dict[tuple[int, int], dict[str, Any]] = {}
                    for seed, items in sorted(by_seed.items()):
                        full_rows = _input_rows(root, dataset, noise, seed, ratio)
                        rows = [full_rows[int(item["row_index"])] for item in items]
                        coords = [
                            {"seed": seed, "row_index": int(item["row_index"])}
                            for item in items
                        ]
                        staging = _run_live_variant_cell(
                            root, cache_root, factory, pipelines,
                            dataset=dataset, noise=noise, seed=seed,
                            ratio=ratio, variant="full", cache_tag=tag,
                            config=config, rows=rows, coordinates=coords,
                            kind="noise-gradient", max_concurrency=max_concurrency,
                        )
                        decorated = _decorate(
                            study.load_jsonl(staging), kind="noise-gradient",
                            variant="full", dataset=dataset, noise=noise,
                            ratio=ratio, coordinates=coords,
                            origin="provider_cache_replay", selection=selection,
                        )
                        staged.update({
                            (int(row["study_seed"]), int(row["study_row_index"])): row
                            for row in decorated
                        })
                        staging.unlink(missing_ok=True)
                    combined = [
                        staged[(int(item["seed"]), int(item["row_index"]))]
                        for item in selected
                    ]
                    study._write_gradient_cell(
                        root, "full", dataset, noise, ratio, combined,
                    )
    finally:
        factory.close()


def _expected_cache_counts(root: Path) -> dict[str, int]:
    selection = study.load_selection(root)
    counts = {
        SOURCE_CACHE_TAG: len(study.DATASETS) * len(study.NOISE_TYPES) * len(study.SEEDS),
        REVIEWER_CACHE_TAG: len(study.DATASETS) * len(study.NOISE_TYPES) * len(study.SEEDS),
        f"{STUDY_TAG}-minus-atf-deanchor-r15": len(study.DATASETS) * len(study.SEEDS),
    }
    for ratio in study.GRADIENT_RATIOS:
        if abs(ratio - study.OFFICIAL_RATIO) < 1e-12:
            continue
        groups = 0
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                selected = study.selected_indices(selection, dataset, noise)
                groups += len({int(item["seed"]) for item in selected})
        counts[f"{STUDY_TAG}-gradient-{study.rate_token(ratio)}"] = groups
    return counts


def _validated_cache_registry(
    root: Path, cache_root: Path,
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    selection = study.load_selection(root)
    for tag, expected_count in _expected_cache_counts(root).items():
        index_path = run_multiseed._provider_cache_index_path(cache_root, tag)
        if not index_path.is_file():
            raise FileNotFoundError(f"missing provider-cache index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        cells = index.get("cells") if isinstance(index, Mapping) else None
        if (index.get("schema") != run_multiseed.PROVIDER_CACHE_INDEX_SCHEMA
                or not isinstance(cells, Mapping)):
            raise ValueError(f"provider-cache index is malformed: {index_path}")
        if len(cells) != expected_count:
            raise ValueError(
                f"provider-cache tag {tag} has {len(cells)} cells, expected {expected_count}"
            )
        for key, cell in cells.items():
            if not isinstance(cell, Mapping):
                raise ValueError(f"provider-cache cell index is malformed: {tag}/{key}")
            indexed_path = Path(str(cell.get("path", "")))
            actual_path = cache_root / tag / indexed_path.name
            row_count = cell.get("row_count")
            sha = str(cell.get("sha256", ""))
            if (not isinstance(row_count, int) or row_count <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or not actual_path.is_file()
                    or not run_multiseed._provider_cache_index_path_matches(
                        str(cell.get("path", "")), actual_path
                    )):
                raise ValueError(f"provider-cache index identity mismatch: {tag}/{key}")
            records = run_multiseed.read_provider_cell(
                actual_path, sha, row_count,
            )
            manifest = run_multiseed._read_provider_cache_manifest(actual_path)
            configuration = manifest.get("configuration", {})
            identity = configuration.get("study") if isinstance(configuration, Mapping) else None
            if tag == SOURCE_CACHE_TAG:
                expected_kind, expected_variant = "ablation", "full"
                expected_ratio = study.OFFICIAL_RATIO
            elif tag == REVIEWER_CACHE_TAG:
                expected_kind = "ablation"
                expected_variant = "minus_reviewer_weighting"
                expected_ratio = study.OFFICIAL_RATIO
            elif tag == f"{STUDY_TAG}-minus-atf-deanchor-r15":
                expected_kind = "ablation"
                expected_variant = "minus_atf_deanchor"
                expected_ratio = study.OFFICIAL_RATIO
            else:
                expected_kind, expected_variant = "noise-gradient", "full"
                ratio_by_tag = {
                    f"{STUDY_TAG}-gradient-{study.rate_token(ratio)}": ratio
                    for ratio in study.GRADIENT_RATIOS
                    if abs(ratio - study.OFFICIAL_RATIO) >= 1e-12
                }
                if tag not in ratio_by_tag:
                    raise ValueError(f"unknown study cache tag: {tag}")
                expected_ratio = ratio_by_tag[tag]
            if not isinstance(identity, Mapping):
                raise ValueError(f"provider-cache study identity is missing: {tag}/{key}")
            dataset = identity.get("dataset")
            noise = identity.get("noise")
            seed = identity.get("seed")
            if (dataset not in study.DATASETS or noise not in study.NOISE_TYPES
                    or seed not in study.SEEDS):
                raise ValueError(f"provider-cache study coordinates are invalid: {tag}/{key}")
            if tag == f"{STUDY_TAG}-minus-atf-deanchor-r15" and noise != "ATF":
                raise ValueError(f"deanchor cache contains a non-ATF cell: {tag}/{key}")
            if expected_kind == "noise-gradient":
                expected_coordinates = [
                    {"seed": int(item["seed"]), "row_index": int(item["row_index"])}
                    for item in study.selected_indices(selection, dataset, noise)
                    if int(item["seed"]) == seed
                ]
            else:
                expected_coordinates = [
                    {"seed": seed, "row_index": row_index}
                    for row_index in range(study.SAMPLE_SIZE)
                ]
            input_path = study.study_input_path(
                root, dataset, noise, seed, expected_ratio,
            )
            required_identity = {
                "schema": "qwen-contextual-studies-v1",
                "study_tag": STUDY_TAG,
                "kind": expected_kind,
                "variant": expected_variant,
                "dataset": dataset,
                "noise": noise,
                "seed": seed,
                "ratio": float(expected_ratio),
                "source_file_sha256": _sha256_file(input_path),
                "coordinates_sha256": _sha256_json(expected_coordinates),
            }
            expected_pipeline_config = {
                "terminal_decoder": "contextual-lattice-v1",
                "deanchor_atf": tag
                != f"{STUDY_TAG}-minus-atf-deanchor-r15",
                "use_verifier": True,
                "verifier_semantic_max_retries": 2,
            }
            if tag == REVIEWER_CACHE_TAG:
                required_identity.update({
                    "reviewer_weighting": "uniform",
                    "reviewer_stage": "disabled",
                    "verifier_policy": "type_contested_only",
                })
                if not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(identity.get("source_provider_cache_sha256", "")),
                ):
                    raise ValueError(
                        f"reviewer cache source binding is invalid: {tag}/{key}"
                    )
            if (cell.get("git_sha") != manifest.get("git_sha")
                    or cell.get("model_revision") != QWEN_REVISION
                    or manifest.get("model_revision") != QWEN_REVISION
                    or manifest.get("bundle_hash") != LOCKED_BUNDLE_MANIFEST_HASH
                    or configuration.get("config") != CONFIG_NAME
                    or configuration.get("dataset") != dataset
                    or configuration.get("noise") != noise
                    or configuration.get("seed") != seed
                    or configuration.get("study_pipeline_config")
                    != expected_pipeline_config
                    or any(identity.get(field) != value
                           for field, value in required_identity.items())
                    or key != run_multiseed._provider_cache_cell_key(
                        CONFIG_NAME, dataset, noise, seed,
                    )
                    or row_count != len(expected_coordinates)):
                raise ValueError(f"provider-cache manifest identity mismatch: {tag}/{key}")
            if sha in registry:
                raise ValueError("duplicate provider-cache SHA across study cells")
            registry[sha] = {
                "tag": tag,
                "key": key,
                "manifest": manifest,
                "identity": dict(identity),
                "records": records,
            }
    source_cells = {
        (
            binding["identity"]["dataset"],
            binding["identity"]["noise"],
            binding["identity"]["seed"],
        ): (sha, binding)
        for sha, binding in registry.items()
        if binding["tag"] == SOURCE_CACHE_TAG
    }
    for reviewer_sha, reviewer in registry.items():
        if reviewer["tag"] != REVIEWER_CACHE_TAG:
            continue
        identity = reviewer["identity"]
        coordinate = (
            identity["dataset"], identity["noise"], identity["seed"],
        )
        source_entry = source_cells.get(coordinate)
        if source_entry is None:
            raise ValueError("reviewer cache has no matching source cache cell")
        source_sha, source = source_entry
        if identity.get("source_provider_cache_sha256") != source_sha:
            raise ValueError("reviewer cache is bound to the wrong source cache cell")
        source_by_index = {
            record.get("row_index"): record
            for record in source["records"]
        }
        for record in reviewer["records"]:
            source_record = source_by_index.get(record.get("row_index"))
            candidates = record.get("candidate_paths")
            expected_weights = (
                [1.0 / len(candidates)] * len(candidates)
                if isinstance(candidates, list) and candidates else None
            )
            source_metadata = (
                source_record.get("provider_metadata")
                if isinstance(source_record, Mapping) else None
            )
            reviewer_metadata = record.get("provider_metadata")
            if (not isinstance(source_record, Mapping)
                    or record.get("input_digest")
                    != source_record.get("input_digest")
                    or candidates != source_record.get("candidate_paths")
                    or record.get("rag_weights") != expected_weights
                    or not isinstance(source_metadata, Mapping)
                    or not isinstance(reviewer_metadata, Mapping)
                    or reviewer_metadata.get("coder")
                    != source_metadata.get("coder")
                    or not all(
                        isinstance(item, Mapping)
                        and item.get("stage") == "reviewer"
                        and item.get("status") == "disabled"
                        for item in reviewer_metadata.get("reviewer", [])
                    )):
                raise ValueError(
                    "reviewer cache does not preserve source coder evidence exactly"
                )
    return registry


def _expected_prediction_cache_tag(
    *, kind: str, variant: str, noise: str, ratio: float,
) -> str:
    if kind == "noise-gradient":
        if abs(float(ratio) - study.OFFICIAL_RATIO) < 1e-12:
            return SOURCE_CACHE_TAG
        return f"{STUDY_TAG}-gradient-{study.rate_token(ratio)}"
    if variant == "minus_reviewer_weighting":
        return REVIEWER_CACHE_TAG
    if variant == "minus_atf_deanchor" and noise == "ATF":
        return f"{STUDY_TAG}-minus-atf-deanchor-r15"
    return SOURCE_CACHE_TAG


def _audit_prediction_row(
    row: Mapping[str, Any], dataset: str, noisy_row: Mapping[str, Any],
    cache_registry: Mapping[str, Mapping[str, Any]], *,
    kind: str, variant: str, noise: str, ratio: float, terminal: Any,
) -> None:
    from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

    if row.get("terminal_model_hash") != LOCKED_DECODER_MODEL_HASH:
        raise ValueError("terminal model hash mismatch")
    cache_sha = row.get("provider_cache_sha256")
    if cache_sha not in cache_registry:
        raise ValueError("prediction is not bound to a verified provider-cache SHA")
    binding = cache_registry[cache_sha]
    expected_tag = _expected_prediction_cache_tag(
        kind=kind, variant=variant, noise=noise, ratio=ratio,
    )
    if binding.get("tag") != expected_tag:
        raise ValueError("prediction is bound to the wrong provider-cache tag")
    expected_digest = run_multiseed._provider_input_digest(noisy_row)
    if row.get("input_digest") != expected_digest:
        raise ValueError("prediction input digest does not match its current noisy row")
    candidates, weights, confidence = (
        row.get("candidate_paths"), row.get("rag_weights"), row.get("confidence")
    )
    if (not isinstance(candidates, list) or not candidates
            or not isinstance(weights, list) or len(weights) != len(candidates)
            or not isinstance(confidence, list)
            or len(confidence) != len(row["tokens"])):
        raise ValueError("candidate evidence is incomplete")
    valid = study._valid_tags(dataset)
    if any(
        not isinstance(path, list) or len(path) != len(row["tokens"])
        or any(tag not in valid for tag in path)
        for path in candidates
    ):
        raise ValueError("candidate evidence violates the ontology")
    metadata = row.get("provider_metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {"coder", "reviewer", "verifier"}:
        raise ValueError("provider evidence stages are incomplete")
    for stage, records in metadata.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"provider evidence for {stage} is empty")
        for record in records:
            if (not isinstance(record, Mapping) or record.get("stage") != stage
                    or record.get("model") != QWEN_MODEL
                    or record.get("served_model") != QWEN_SERVED_MODEL
                    or record.get("revision") != QWEN_REVISION
                    or record.get("enable_thinking") is not False
                    or record.get("thinking_mode") != "nothink"):
                raise ValueError(f"provider evidence identity mismatch for {stage}")
    serialized = json.dumps(row, ensure_ascii=False).lower()
    if any(marker in serialized for marker in (
        "api_key", "authorization", "bearer ", "backbone_api_key",
    )):
        raise ValueError("credential-bearing field detected")
    if "ner_tags" in row or "dirty_tags" in row:
        raise ValueError("prediction leaks source-only label fields")
    comparable_fields = (
        "input_digest", "candidate_paths", "rag_weights", "confidence",
        "provider_metadata",
    )
    matching_records = [
        record for record in binding["records"]
        if all(record.get(field) == row.get(field) for field in comparable_fields)
    ]
    if not matching_records:
        raise ValueError("prediction evidence is not exactly present in its provider-cache cell")
    if variant != "minus_verifier" and not any(
        record.get("anchor_tags") == row.get("terminal_anchor_tags")
        for record in matching_records
    ):
        raise ValueError("prediction anchor is not present in its provider-cache cell")

    cached = dict(matching_records[0])
    if kind == "ablation" and variant == "minus_contextual":
        expected_terminal = {
            "pred_tags": list(cached["anchor_tags"]),
            "terminal_anchor_tags": list(cached["anchor_tags"]),
            "terminal_model_hash": LOCKED_DECODER_MODEL_HASH,
            "terminal_used_anchor": True,
            "terminal_predicted_gain": 0.0,
            "terminal_fallback_count": 0,
        }
    else:
        import multi_agent_v2

        terminal_margin = None
        if kind == "ablation" and variant == "minus_gate":
            terminal_margin = -1.0e9
        if kind == "ablation" and variant == "minus_verifier":
            valid_types = set(multi_agent_v2.DATASET_ENTITY_TYPES[dataset])
            dangling_policy = "demote" if noise == "IF" else "promote"
            anchor = multi_agent_v2._base_decode(
                [list(path) for path in row["candidate_paths"]],
                [float(value) for value in row["rag_weights"]],
                list(row["tokens"]), list(noisy_row["dirty_tags"]),
                dataset, noise,
            )
            cached["anchor_tags"] = multi_agent_v2.legalize_noise_aware(
                anchor, valid_types, dangling_policy,
            )
        expected_terminal = multi_agent_v2.run_contextual_lattice_replay(
            tokens=list(noisy_row["tokens"]),
            dirty_tags=list(noisy_row["dirty_tags"]),
            provider_record=cached, dataset_name=dataset, terminal=terminal,
            terminal_margin=terminal_margin,
        )
    terminal_fields = (
        "pred_tags", "terminal_anchor_tags", "terminal_model_hash",
        "terminal_used_anchor", "terminal_predicted_gain",
        "terminal_fallback_count",
    )
    if any(row.get(field) != expected_terminal.get(field) for field in terminal_fields):
        raise ValueError(f"prediction does not implement {variant} terminal semantics")


def _validate_gradient_selection(
    rows: Sequence[Mapping[str, Any]],
    expected_selection: Sequence[Mapping[str, Any]],
) -> None:
    expected_coordinates = [
        (int(item["seed"]), int(item["row_index"]))
        for item in expected_selection
    ]
    actual_coordinates = [
        (row.get("study_seed"), row.get("study_row_index"))
        for row in rows
    ]
    if actual_coordinates != expected_coordinates:
        raise ValueError("gradient coordinates do not match frozen selection")
    for row, selected_item in zip(rows, expected_selection):
        tokens_sha = hashlib.sha256(
            study._canonical_tokens(row["tokens"]).encode("utf-8")
        ).hexdigest()
        if (row.get("study_group_digest") != selected_item.get("group_digest")
                or tokens_sha != selected_item.get("tokens_sha256")):
            raise ValueError("gradient frozen-selection digest mismatch")


def audit_outputs(root: Path, cache_root: Path) -> dict[str, Any]:
    """Fail closed on exact matrix, metadata, ontology, and provenance."""
    errors: list[str] = []
    cache_registry = _validated_cache_registry(root, cache_root)
    import multi_agent_v2
    from multi_agent_v2 import _load_contextual_lattice_terminal

    terminal = _load_contextual_lattice_terminal()
    ablation_cells = ablation_rows = gradient_cells = gradient_rows = 0
    for variant in study.ABLATION_VARIANTS:
        for dataset in study.DATASETS:
            if variant == "minus_verifier":
                multi_agent_v2._init_deer(dataset, cache_only=True)
            for noise in study.NOISE_TYPES:
                for seed in study.SEEDS:
                    path = study.study_prediction_path(
                        root, "ablation", variant, dataset, noise,
                        study.OFFICIAL_RATIO, seed,
                    )
                    try:
                        rows = study.validate_study_prediction_file(
                            path, dataset, expected_count=study.SAMPLE_SIZE,
                        )
                        noisy_rows = _input_rows(
                            root, dataset, noise, seed, study.OFFICIAL_RATIO,
                        )
                        for row_index, row in enumerate(rows):
                            expected_origin = "offline_replay"
                            if variant == "full":
                                expected_origin = "provider_cache_replay"
                            elif variant == "minus_reviewer_weighting":
                                expected_origin = "source_coder_verifier_live"
                            elif variant == "minus_atf_deanchor" and noise == "ATF":
                                expected_origin = "provider_cache_replay"
                            if (row.get("study_kind") != "ablation"
                                    or row.get("study_variant") != variant
                                    or row.get("study_dataset") != dataset
                                    or row.get("study_noise") != noise
                                    or row.get("study_seed") != seed
                                    or row.get("study_ratio") != study.OFFICIAL_RATIO
                                    or row.get("study_origin") != expected_origin):
                                raise ValueError("ablation metadata mismatch")
                            if row.get("fallback_used") is not False:
                                raise ValueError("fallback detected")
                            if (row.get("tokens") != noisy_rows[row_index].get("tokens")
                                    or row.get("gold_tags") != noisy_rows[row_index].get("ner_tags")):
                                raise ValueError("ablation input alignment mismatch")
                            _audit_prediction_row(
                                row, dataset, noisy_rows[row_index], cache_registry,
                                kind="ablation", variant=variant, noise=noise,
                                ratio=study.OFFICIAL_RATIO, terminal=terminal,
                            )
                        ablation_cells += 1
                        ablation_rows += len(rows)
                    except Exception as exc:
                        errors.append(f"{path}: {exc}")
    for ratio in study.GRADIENT_RATIOS:
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                path = study.study_prediction_path(
                    root, "noise-gradient", "full", dataset, noise, ratio,
                )
                try:
                    rows = study.validate_study_prediction_file(
                        path, dataset, expected_count=study.TEST_GROUPS_PER_DATASET,
                    )
                    expected_selection = study.selected_indices(
                        study.load_selection(root), dataset, noise,
                    )
                    _validate_gradient_selection(rows, expected_selection)
                    for row in rows:
                        if (row.get("study_kind") != "noise-gradient"
                                or row.get("study_variant") != "full"
                                or row.get("study_dataset") != dataset
                                or row.get("study_noise") != noise
                                or row.get("study_ratio") != ratio
                                or row.get("study_origin") != (
                                    "source_reuse" if abs(
                                        ratio - study.OFFICIAL_RATIO
                                    ) < 1e-12 else "provider_cache_replay"
                                )
                                or row.get("fallback_used") is not False):
                            raise ValueError("gradient metadata mismatch")
                        seed = row.get("study_seed")
                        row_index = row.get("study_row_index")
                        if seed not in study.SEEDS or not isinstance(row_index, int):
                            raise ValueError("gradient coordinate is malformed")
                        noisy_rows = _input_rows(root, dataset, noise, seed, ratio)
                        if (not 0 <= row_index < len(noisy_rows)
                                or row.get("tokens") != noisy_rows[row_index].get("tokens")
                                or row.get("gold_tags") != noisy_rows[row_index].get("ner_tags")):
                            raise ValueError("gradient input alignment mismatch")
                        _audit_prediction_row(
                            row, dataset, noisy_rows[row_index], cache_registry,
                            kind="noise-gradient", variant="full", noise=noise,
                            ratio=ratio, terminal=terminal,
                        )
                    gradient_cells += 1
                    gradient_rows += len(rows)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
    tmp_files = [str(path) for path in root.rglob("*.tmp")]
    tmp_files.extend(str(path) for path in cache_root.rglob("*.tmp"))
    if tmp_files:
        errors.append(f"temporary files remain: {tmp_files[:5]}")
    analysis_paths = [
        root / "analysis" / "ablation_results.json",
        root / "analysis" / "noise_gradient_results.json",
    ]
    prediction_paths = [
        path for path in (root / "predictions").rglob("*.jsonl")
        if "staging" not in path.parts
    ]
    if not all(path.is_file() for path in analysis_paths):
        errors.append("analysis outputs are incomplete")
    elif not errors:
        try:
            previous_ablation = json.loads(analysis_paths[0].read_text(encoding="utf-8"))
            previous_gradient = json.loads(analysis_paths[1].read_text(encoding="utf-8"))
            import analyze_contextual_studies

            expected_ablation = analyze_contextual_studies.analyze_ablation(
                root, study.SOURCE_TAG,
            )
            expected_gradient = analyze_contextual_studies.analyze_gradient(
                root, study.SOURCE_TAG,
            )
            if (previous_ablation != expected_ablation
                    or previous_gradient != expected_gradient):
                errors.append("analysis outputs do not match a fresh recomputation")
        except Exception as exc:
            errors.append(f"analysis output validation failed: {exc}")
    if (not errors and prediction_paths
            and min(path.stat().st_mtime_ns for path in analysis_paths) < max(
                path.stat().st_mtime_ns for path in prediction_paths
            )):
        errors.append("analysis outputs are stale")
    report = {
        "schema": "qwen-contextual-studies-audit-v1",
        "study_tag": STUDY_TAG,
        "ablation_cells": ablation_cells,
        "ablation_rows": ablation_rows,
        "gradient_cells": gradient_cells,
        "gradient_rows": gradient_rows,
        "errors": errors,
        "passed": not errors and ablation_cells == 270
        and ablation_rows == 54000 and gradient_cells == 75
        and gradient_rows == 2250,
    }
    study._write_json_atomic(root / "audit.json", report)
    if not report["passed"]:
        raise RuntimeError(f"study audit failed with {len(errors)} errors")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=(
            "prepare", "source-cache", "source-replay", "offline-ablation",
            "atf-deanchor", "reviewer-weighting", "noise-gradient",
            "analyze", "audit",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("provider_cache"))
    parser.add_argument("--frozen-source-root", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = args.root.resolve()
    if args.phase == "prepare":
        if args.frozen_source_root is None:
            raise ValueError("prepare requires --frozen-source-root")
        study.prepare_inputs(
            root, frozen_source_root=args.frozen_source_root.resolve(),
        )
        return
    verify_frozen_inputs(root)
    if args.max_concurrency <= 0 or args.max_concurrency > 32:
        raise ValueError("study max-concurrency must be in [1, 32]")
    loop = asyncio.new_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64))
    asyncio.set_event_loop(loop)
    if args.phase == "source-cache":
        run_source_cache(root, args.cache_root.resolve(), max_concurrency=args.max_concurrency)
    elif args.phase == "source-replay":
        run_source_replay(root, args.cache_root.resolve(), max_concurrency=args.max_concurrency)
    elif args.phase == "offline-ablation":
        run_offline_ablation(root)
    elif args.phase == "atf-deanchor":
        run_atf_deanchor(root, args.cache_root.resolve(), max_concurrency=args.max_concurrency)
    elif args.phase == "reviewer-weighting":
        run_reviewer_weighting(
            root, args.cache_root.resolve(), max_concurrency=args.max_concurrency,
        )
    elif args.phase == "noise-gradient":
        run_noise_gradient(root, args.cache_root.resolve(), max_concurrency=args.max_concurrency)
    elif args.phase == "analyze":
        import analyze_contextual_studies

        analyze_contextual_studies.analyze_ablation(root, study.SOURCE_TAG)
        analyze_contextual_studies.analyze_gradient(root, study.SOURCE_TAG)
    else:
        audit_outputs(root, args.cache_root.resolve())


if __name__ == "__main__":
    main()
