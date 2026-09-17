"""Reproducible Qwen no-thinking Contextual-Lattice studies.

This module owns the study namespace around the already completed official
45-cell run.  It deliberately keeps the official runner/configuration intact:
study inputs and predictions live below ``contextual_studies/<tag>/`` and all
provider-backed work uses the pinned Qwen/vLLM adapter.

The two supported studies are:

* six one-factor Contextual-Lattice ablations at the official 15% protocol;
* a five-rate BT/IF/ATF gradient on the locked 150-group held-out split.

The module is intentionally useful in offline mode as well.  ``prepare`` and
the derived ablations never need an API key; live stages are explicit CLI
subcommands and fail closed on malformed rows.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from contextual_lattice_runtime import (
    LOCKED_BUNDLE_MANIFEST_HASH,
    LOCKED_CHECKPOINT_HASH,
    LOCKED_DECODER_FILE_HASH,
    LOCKED_DECODER_MODEL_HASH,
    LOCKED_GATE_FILE_HASH,
    LOCKED_SPLIT_HASH,
)
from official_provider_cache import QWEN_MODEL, QWEN_REVISION, QWEN_SERVED_MODEL


STUDY_MANIFEST_SCHEMA = "selectdenoise-contextual-studies-v1"
SOURCE_TAG = "deepseek-flash-contextual-studies-v1-20260914-frozen-inputs"
FROZEN_SOURCE_MANIFEST_SHA256 = "a6a06552472d9c1243376dd3f71ef76d95d424db620667ffcfa81e2469694bbf"
FROZEN_INPUT_MANIFEST_SHA256 = "a7f69c202fbe1abb68263689d92270c05d32a85ef98840200240bfe06a131cda"
FROZEN_SELECTION_SHA256 = "f4ed672e68d90ccbfad9652133813f6ea598eba06c6436f2517afb7bc502f9fe"
DATASETS = ("msra", "conll2003", "wnut17", "fewnerd", "ontonotes5")
NOISE_TYPES = ("BT", "IF", "ATF")
SEEDS = (13, 42, 2024)
GRADIENT_RATIOS = (0.05, 0.15, 0.25, 0.35, 0.45)
OFFICIAL_RATIO = 0.15
SAMPLE_SIZE = 200
TEST_GROUPS_PER_DATASET = 30
ABLATION_VARIANTS = (
    "full",
    "minus_contextual",
    "minus_gate",
    "minus_reviewer_weighting",
    "minus_verifier",
    "minus_atf_deanchor",
)

_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

DATASET_ENTITY_TYPES = {
    "msra": ("PER", "LOC", "ORG"),
    "conll2003": ("PER", "LOC", "ORG", "MISC"),
    "wnut17": ("PER", "LOC", "ORG", "MISC"),
    "fewnerd": ("PER", "LOC", "ORG", "MISC"),
    "ontonotes5": ("PER", "LOC", "ORG", "MISC"),
}

CONTEXTUAL_LATTICE_PROVENANCE = {
    "bundle_manifest_hash": LOCKED_BUNDLE_MANIFEST_HASH,
    "checkpoint_hash": LOCKED_CHECKPOINT_HASH,
    "decoder_file_hash": LOCKED_DECODER_FILE_HASH,
    "decoder_model_hash": LOCKED_DECODER_MODEL_HASH,
    "gate_file_hash": LOCKED_GATE_FILE_HASH,
    "margin": -0.1,
    "split_hash": LOCKED_SPLIT_HASH,
}


def rate_token(ratio: float) -> str:
    """Return the canonical filesystem token for an entity corruption rate."""
    value = float(ratio)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid noise ratio: {ratio!r}")
    percentage = int(round(value * 100))
    if abs(value * 100 - percentage) > 1e-9:
        raise ValueError("study ratios must be whole percentage points")
    return f"r{percentage}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _valid_tags(dataset: str) -> set[str]:
    if dataset not in DATASET_ENTITY_TYPES:
        raise ValueError(f"unknown dataset ontology: {dataset!r}")
    return {"O"} | {
        f"{prefix}-{entity_type}"
        for entity_type in DATASET_ENTITY_TYPES[dataset]
        for prefix in ("B", "I")
    }


def _is_legal_iob2(tags: Sequence[str]) -> bool:
    previous = "O"
    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if previous not in {f"B-{entity_type}", f"I-{entity_type}"}:
                return False
        previous = tag
    return True


def validate_study_prediction_row(
    row: Mapping[str, Any],
    dataset: str,
    *,
    require_terminal: bool = True,
) -> None:
    """Fail closed on one study prediction row.

    This validator intentionally does not pad, truncate, reorder, or replace
    tags.  It is used for both offline-derived rows and live study output.
    """
    tokens = row.get("tokens")
    gold = row.get("gold_tags")
    pred = row.get("pred_tags")
    if not all(isinstance(value, list) for value in (tokens, gold, pred)):
        raise ValueError("study row sequences are not aligned")
    if not all(
        isinstance(item, str)
        for values in (tokens, gold, pred)
        for item in values
    ):
        raise ValueError("study row sequences are not aligned")
    if not len(tokens) == len(gold) == len(pred):
        raise ValueError("study row sequences are not aligned")
    valid = _valid_tags(dataset)
    if any(tag not in valid for values in (gold, pred) for tag in values):
        raise ValueError("study row contains an unknown ontology tag")
    if not _is_legal_iob2(pred):
        raise ValueError("study row prediction is not legal IOB2")
    if not require_terminal:
        return
    anchor = row.get("terminal_anchor_tags")
    if (not isinstance(anchor, list)
            or len(anchor) != len(tokens)
            or any(not isinstance(tag, str) or tag not in valid for tag in anchor)
            or not _is_legal_iob2(anchor)):
        raise ValueError("study terminal anchor is invalid")
    model_hash = row.get("terminal_model_hash")
    if not isinstance(model_hash, str) or not _SHA256_RE.fullmatch(model_hash):
        raise ValueError("study terminal model hash is invalid")
    if not isinstance(row.get("terminal_used_anchor"), bool):
        raise ValueError("study terminal anchor flag is invalid")
    gain = row.get("terminal_predicted_gain")
    if (isinstance(gain, bool) or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))):
        raise ValueError("study terminal predicted gain is invalid")
    fallback_count = row.get("terminal_fallback_count")
    if (isinstance(fallback_count, bool) or not isinstance(fallback_count, int)
            or fallback_count != 0):
        raise ValueError("study terminal fallback count must be zero")
    if row.get("fallback_used") is not False:
        raise ValueError("study pipeline fallback must be false")


def validate_study_prediction_file(
    path: Path,
    dataset: str,
    *,
    expected_count: int | None = None,
    require_terminal: bool = True,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"study prediction has {len(rows)} rows, expected {expected_count}: {path}"
        )
    for row in rows:
        validate_study_prediction_row(row, dataset, require_terminal=require_terminal)
    return rows


def build_study_manifest(
    *,
    study_tag: str,
    source_tag: str,
    git_sha: str,
    source_files: Mapping[str, str],
    provider: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a secret-free, closed-world study manifest."""
    if not _TAG_RE.fullmatch(study_tag):
        raise ValueError("study tag contains invalid characters")
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise ValueError("study manifest requires a 40-hex Git SHA")
    if source_tag != SOURCE_TAG:
        raise ValueError("study source must be the locked DeepSeek frozen input bank")
    safe_provider = {
        key: value for key, value in dict(provider).items()
        if key.lower() not in {"api_key", "apikey", "authorization", "token", "secret"}
    }
    manifest = {
        "schema_version": STUDY_MANIFEST_SCHEMA,
        "study_tag": study_tag,
        "source": {
            "tag": source_tag,
            "git_sha": git_sha,
            "files_sha256": dict(sorted(source_files.items())),
        },
        "backbone": {
            "provider": safe_provider.get("provider", "vllm"),
            "model": safe_provider.get("model", QWEN_MODEL),
            "served_model": safe_provider.get("served_model", QWEN_SERVED_MODEL),
            "revision": safe_provider.get("revision", QWEN_REVISION),
            "structured_api": safe_provider.get(
                "structured_api", "chat-completions-json-schema"
            ),
            "schema_transport": safe_provider.get(
                "schema_transport", "openai-chat-json-schema"
            ),
            "enable_thinking": False,
            "thinking_mode": "nothink",
        },
        "terminal_provenance": dict(CONTEXTUAL_LATTICE_PROVENANCE),
        "protocol": dict(protocol),
        "created_by": "contextual_studies.py",
    }
    for key in (
        "base_url", "endpoint_origin", "endpoint_path", "thinking_requested",
        "rate_limit_rpm", "rate_limit_tpm", "response_identity_max_retries",
        "provider_order", "provider_only", "allow_fallbacks",
        "require_parameters", "quantizations", "max_price",
        "inference_provider", "allowed_inference_providers", "routing_policy",
        "provider_selection",
    ):
        if key in safe_provider:
            manifest["backbone"][key] = safe_provider[key]
    return manifest


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite numbers in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if not _GIT_SHA_RE.fullmatch(value):
        raise RuntimeError("git rev-parse did not return a 40-hex SHA")
    return value


def _study_source_files() -> dict[str, str]:
    paths = (
        "live_backbone.py",
        "multi_agent_v2.py",
        "run_multiseed.py",
        "contextual_lattice_runtime.py",
        "selectdenoise_contextual_lattice.py",
        "noise_injector.py",
        "contextual_studies.py",
        "run_contextual_studies.py",
        "analyze_contextual_studies.py",
        "official_contract.py",
    )
    return {
        path: sha256_file(Path(path))
        for path in paths
        if Path(path).is_file()
    }


def _assert_no_api_secret_in_manifest(manifest: Mapping[str, Any]) -> None:
    serialized = _canonical_json(manifest).lower()
    for marker in ("api_key", "apikey", "authorization", "bearer ", "secret"):
        if marker in serialized:
            raise ValueError("study manifest contains a credential-bearing field")


def source_prediction_path(source_tag: str, dataset: str, noise: str, seed: int) -> Path:
    return Path("predictions_multiseed") / (
        f"pred_seed{seed}__selectdenoise_contextual_lattice__{dataset}__{noise}__{source_tag}.jsonl"
    )


def source_noisy_path(dataset: str, noise: str, seed: int, ratio: float = OFFICIAL_RATIO) -> Path:
    if abs(float(ratio) - OFFICIAL_RATIO) < 1e-12:
        return Path("results_multiseed") / (
            f"noisy_seed{seed}__{noise}__{dataset}__N{SAMPLE_SIZE}.jsonl"
        )
    return Path("contextual_studies") / "__canonical_inputs__" / (
        f"noisy_seed{seed}__{noise}__{dataset}__N{SAMPLE_SIZE}__{rate_token(ratio)}.jsonl"
    )


def study_input_path(root: Path, dataset: str, noise: str, seed: int, ratio: float) -> Path:
    return Path(root) / "inputs" / rate_token(ratio) / (
        f"noisy_seed{seed}__{noise}__{dataset}__N{SAMPLE_SIZE}.jsonl"
    )


def study_prediction_path(
    root: Path,
    study_kind: str,
    variant: str,
    dataset: str,
    noise: str,
    ratio: float,
    seed: int | None = None,
) -> Path:
    if study_kind == "ablation":
        if seed is None:
            raise ValueError("ablation prediction paths require a seed")
        return Path(root) / "predictions" / "ablation" / variant / (
            f"pred_seed{seed}__{dataset}__{noise}.jsonl"
        )
    if study_kind == "noise-gradient":
        if seed is not None:
            raise ValueError("gradient prediction paths are group-selected, not seed cells")
        return Path(root) / "predictions" / "noise_gradient" / rate_token(ratio) / (
            f"pred__{variant}__{dataset}__{noise}.jsonl"
        )
    raise ValueError(f"unknown study kind: {study_kind!r}")


def _copy_if_absent(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f"study input collision with different content: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_tokens(tokens: Sequence[str]) -> str:
    return json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))


def group_digest(dataset: str, tokens: Sequence[str], *, salt: str) -> str:
    return _sha256_bytes(
        f"{salt}|{dataset}|{_canonical_tokens(tokens)}".encode("utf-8")
    )


def _source_rows(source_tag: str = SOURCE_TAG) -> list[dict[str, Any]]:
    """Load all 45 official cells with audit-only coordinates attached."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for noise in NOISE_TYPES:
            for seed in SEEDS:
                path = source_prediction_path(source_tag, dataset, noise, seed)
                if not path.exists():
                    raise FileNotFoundError(f"missing locked source prediction: {path}")
                cell = load_jsonl(path)
                if len(cell) != SAMPLE_SIZE:
                    raise ValueError(f"locked source cell has {len(cell)} rows: {path}")
                for row_index, row in enumerate(cell):
                    validate_study_prediction_row(row, dataset)
                    record = dict(row)
                    record.update({
                        "dataset": dataset,
                        "family": noise,
                        "seed": seed,
                        "row_index": row_index,
                    })
                    rows.append(record)
    if len(rows) != 45 * SAMPLE_SIZE:
        raise ValueError(f"locked source matrix has {len(rows)} rows, expected 9000")
    return rows


def build_test_selection(source_tag: str = SOURCE_TAG) -> list[dict[str, Any]]:
    """Return the 450 rows in the frozen, group-disjoint test split."""
    from selectdenoise_contextual_lattice import build_historical_split

    rows = _source_rows(source_tag)
    split = build_historical_split(rows, groups_per_dataset=TEST_GROUPS_PER_DATASET)
    selected: list[dict[str, Any]] = []
    for record in split.test:
        item = {
            "dataset": record["dataset"],
            "family": record["family"],
            "seed": int(record["seed"]),
            "row_index": int(record["row_index"]),
            "tokens_sha256": _sha256_bytes(
                _canonical_tokens(record["tokens"]).encode("utf-8")
            ),
            "group_digest": group_digest(
                record["dataset"], record["tokens"],
                salt="selectdenoise-contextual-lattice-v1-20260820",
            ),
        }
        selected.append(item)
    selected.sort(key=lambda item: (
        item["dataset"], item["group_digest"], item["family"], item["seed"], item["row_index"]
    ))
    if len(selected) != len(DATASETS) * TEST_GROUPS_PER_DATASET * len(NOISE_TYPES):
        raise ValueError(f"frozen test selection has {len(selected)} rows, expected 450")
    if len({(item["dataset"], item["group_digest"]) for item in selected}) != 150:
        raise ValueError("frozen test selection does not contain 30 groups per dataset")
    return selected


def _materialize_noise_file(
    dataset: str,
    noise: str,
    seed: int,
    ratio: float,
    target: Path,
) -> None:
    """Generate one current-protocol noisy file without touching old files."""
    if target.exists():
        return
    from noise_injector import inject_noise_corpus

    # Reuse only the canonical 15% corpus's clean token/gold view.  This keeps
    # every gradient rate on exactly the same locked 200-sentence slice while
    # still re-running the current corpus-level injector for the new rate.
    canonical = source_noisy_path(dataset, noise, seed, OFFICIAL_RATIO)
    if not canonical.exists():
        raise FileNotFoundError(f"missing canonical input for gradient regeneration: {canonical}")
    canonical_rows = load_jsonl(canonical)
    if len(canonical_rows) != SAMPLE_SIZE:
        raise ValueError(f"canonical input has {len(canonical_rows)} rows: {canonical}")
    tokens = [list(row["tokens"]) for row in canonical_rows]
    gold = [list(row["ner_tags"]) for row in canonical_rows]
    noisy, _metadata = inject_noise_corpus(
        list(zip(tokens, gold)), noise, ratio, dataset, seed
    )
    rows = [
        {"dataset": dataset, "tokens": list(tokens[index]),
         "ner_tags": list(gold[index]), "dirty_tags": list(noisy[index][1])}
        for index in range(len(tokens))
    ]
    _write_jsonl_atomic(target, rows)


def prepare_inputs(
    root: Path,
    source_tag: str = SOURCE_TAG,
    *,
    frozen_source_root: Path,
) -> dict[str, Any]:
    """Copy and verify the authoritative DeepSeek study input bank.

    The source predictions are intentionally not imported: only the immutable
    225 noisy-input files and paired 450-row selection define cross-backbone
    experimental parity.
    """
    root = Path(root)
    if source_tag != SOURCE_TAG:
        raise ValueError("study source tag does not identify the frozen input bank")
    frozen_source_root = Path(frozen_source_root)
    source_manifest_path = frozen_source_root / "manifest.json"
    source_input_manifest_path = frozen_source_root / "input_manifest.json"
    source_selection_path = frozen_source_root / "selection.json"
    if (not source_manifest_path.is_file() or not source_input_manifest_path.is_file()
            or not source_selection_path.is_file()):
        raise FileNotFoundError(
            "frozen DeepSeek study manifest, input manifest, and selection are required"
        )
    if sha256_file(source_manifest_path) != FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError("frozen DeepSeek study manifest SHA mismatch")
    if sha256_file(source_input_manifest_path) != FROZEN_INPUT_MANIFEST_SHA256:
        raise ValueError("frozen DeepSeek input manifest SHA mismatch")
    if sha256_file(source_selection_path) != FROZEN_SELECTION_SHA256:
        raise ValueError("frozen DeepSeek selection SHA mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_hashes = source_manifest.get("input_files_sha256")
    if (source_manifest.get("schema_version") != STUDY_MANIFEST_SCHEMA
            or not isinstance(expected_hashes, Mapping)
            or source_manifest.get("selection_sha256") != FROZEN_SELECTION_SHA256
            or source_manifest.get("input_manifest_sha256")
            != FROZEN_INPUT_MANIFEST_SHA256):
        raise ValueError("frozen DeepSeek study manifest identity mismatch")
    selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
    expected_selection_rows = len(DATASETS) * TEST_GROUPS_PER_DATASET * len(NOISE_TYPES)
    if not isinstance(selection, list) or len(selection) != expected_selection_rows:
        raise ValueError(
            f"frozen study selection must contain exactly {expected_selection_rows} rows"
        )
    selection_path = root / "selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise ValueError("study selection collision: existing selection differs")
    else:
        _write_json_atomic(selection_path, selection)

    # Keep byte-identical copies of both pinned manifests beside the prepared
    # bank.  Every later phase can therefore derive the authoritative input
    # hash map from a file whose SHA is pinned in source code, instead of
    # trusting the mutable, locally generated input_manifest.json.
    _copy_if_absent(
        source_manifest_path, root / "frozen_source_manifest.json",
    )
    _copy_if_absent(
        source_input_manifest_path, root / "frozen_input_manifest.json",
    )

    copied_hashes: dict[str, str] = {}
    for ratio in GRADIENT_RATIOS:
        for dataset in DATASETS:
            for noise in NOISE_TYPES:
                for seed in SEEDS:
                    relative = Path("inputs") / rate_token(ratio) / (
                        f"noisy_seed{seed}__{noise}__{dataset}__N{SAMPLE_SIZE}.jsonl"
                    )
                    source = frozen_source_root / relative
                    expected = expected_hashes.get(relative.as_posix())
                    if not source.is_file() or expected != sha256_file(source):
                        raise ValueError(f"frozen input SHA mismatch: {relative.as_posix()}")
                    rows = load_jsonl(source)
                    if len(rows) != SAMPLE_SIZE:
                        raise ValueError(
                            f"frozen input must contain exactly {SAMPLE_SIZE} rows: "
                            f"{relative.as_posix()}"
                        )
                    valid = _valid_tags(dataset)
                    for row_index, row in enumerate(rows):
                        tokens = row.get("tokens")
                        gold = row.get("ner_tags")
                        dirty = row.get("dirty_tags")
                        if (not isinstance(tokens, list) or not isinstance(gold, list)
                                or not isinstance(dirty, list)
                                or not (len(tokens) == len(gold) == len(dirty))
                                or any(tag not in valid for tag in gold + dirty)):
                            raise ValueError(
                                f"malformed frozen input row {row_index}: "
                                f"{relative.as_posix()}"
                            )
                    target = root / relative
                    _copy_if_absent(source, target)
                    copied_hashes[relative.as_posix()] = expected
    manifest = {
        "schema_version": STUDY_MANIFEST_SCHEMA,
        "source_tag": source_tag,
        "sample_size": SAMPLE_SIZE,
        "ratios": list(GRADIENT_RATIOS),
        "datasets": list(DATASETS),
        "noise_types": list(NOISE_TYPES),
        "seeds": list(SEEDS),
        "selection_rows": len(selection),
        "frozen_source_manifest_sha256": sha256_file(source_manifest_path),
        "frozen_selection_sha256": sha256_file(source_selection_path),
        "input_files_sha256": dict(sorted(copied_hashes.items())),
        "input_files": sorted(
            str(path.relative_to(root))
            for path in (root / "inputs").rglob("*.jsonl")
        ),
    }
    _assert_no_api_secret_in_manifest(manifest)
    manifest_path = root / "input_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("study input manifest collision: existing manifest differs")
    else:
        _write_json_atomic(manifest_path, manifest)
    return manifest


def load_selection(root: Path) -> list[dict[str, Any]]:
    path = Path(root) / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"run prepare before using the study selection: {path}")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selection, list):
        raise ValueError("study selection must be a JSON array")
    return [dict(item) for item in selection]


def selected_indices(
    selection: Sequence[Mapping[str, Any]],
    dataset: str,
    noise: str,
) -> list[dict[str, Any]]:
    selected = [
        dict(item) for item in selection
        if item.get("dataset") == dataset and item.get("family") == noise
    ]
    if len(selected) != TEST_GROUPS_PER_DATASET:
        raise ValueError(f"selection has {len(selected)} rows for {dataset}/{noise}")
    return sorted(selected, key=lambda item: (int(item["seed"]), int(item["row_index"])))


def _source_cell_rows(source_tag: str, dataset: str, noise: str, seed: int) -> list[dict[str, Any]]:
    path = source_prediction_path(source_tag, dataset, noise, seed)
    rows = load_jsonl(path)
    if len(rows) != SAMPLE_SIZE:
        raise ValueError(f"source cell has {len(rows)} rows: {path}")
    for row in rows:
        validate_study_prediction_row(row, dataset)
    return rows


def _input_cell_rows(root: Path, dataset: str, noise: str, seed: int, ratio: float) -> list[dict[str, Any]]:
    path = study_input_path(root, dataset, noise, seed, ratio)
    rows = load_jsonl(path)
    if len(rows) != SAMPLE_SIZE:
        raise ValueError(f"study input has {len(rows)} rows, expected 200: {path}")
    return rows


def _attach_study_metadata(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    dataset: str,
    noise: str,
    ratio: float,
    source_indices: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        record = dict(source)
        record.update({
            "study_variant": variant,
            "study_dataset": dataset,
            "study_noise": noise,
            "study_ratio": float(ratio),
            "study_row_index": (
                int(source_indices[index]["row_index"])
                if source_indices is not None else index
            ),
            "study_seed": (
                int(source_indices[index]["seed"])
                if source_indices is not None else None
            ),
        })
        out.append(record)
    return out


def _validate_source_alignment(
    pred: Mapping[str, Any], noisy: Mapping[str, Any], dataset: str,
) -> None:
    if pred.get("tokens") != noisy.get("tokens"):
        raise ValueError("source prediction/input token mismatch")
    if pred.get("gold_tags") != noisy.get("ner_tags"):
        raise ValueError("source prediction/input gold mismatch")
    if len(noisy.get("dirty_tags", [])) != len(noisy.get("tokens", [])):
        raise ValueError("source input dirty tags are not aligned")
    if any(tag not in _valid_tags(dataset) for tag in noisy["dirty_tags"]):
        raise ValueError("source input contains an unknown tag")


def _write_variant_cell(
    root: Path,
    variant: str,
    dataset: str,
    noise: str,
    ratio: float,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = study_prediction_path(root, "ablation", variant, dataset, noise, ratio, seed)
    if path.exists():
        existing = validate_study_prediction_file(path, dataset, expected_count=len(rows))
        if existing != [dict(row) for row in rows]:
            raise ValueError(f"ablation output collision with different rows: {path}")
        return path
    _write_jsonl_atomic(path, rows)
    validate_study_prediction_file(path, dataset, expected_count=len(rows))
    return path


def _write_gradient_cell(
    root: Path,
    variant: str,
    dataset: str,
    noise: str,
    ratio: float,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = study_prediction_path(root, "noise-gradient", variant, dataset, noise, ratio)
    if path.exists():
        existing = validate_study_prediction_file(path, dataset, expected_count=len(rows))
        if existing != [dict(row) for row in rows]:
            raise ValueError(f"gradient output collision with different rows: {path}")
        return path
    _write_jsonl_atomic(path, rows)
    validate_study_prediction_file(path, dataset, expected_count=len(rows))
    return path


__all__ = [
    "ABLATION_VARIANTS",
    "DATASETS",
    "GRADIENT_RATIOS",
    "NOISE_TYPES",
    "OFFICIAL_RATIO",
    "SEEDS",
    "SOURCE_TAG",
    "STUDY_MANIFEST_SCHEMA",
    "build_study_manifest",
    "build_test_selection",
    "group_digest",
    "holm_adjust",
    "load_selection",
    "prepare_inputs",
    "rate_token",
    "selected_indices",
    "sha256_file",
    "study_input_path",
    "study_prediction_path",
    "validate_study_prediction_file",
    "validate_study_prediction_row",
]


if __name__ == "__main__":
    print("This module is imported by the study runner; use the project CLI wrapper.")
