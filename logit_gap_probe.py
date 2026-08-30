"""Reproducible raw next-token logit probe for pinned Qwen3-32B-AWQ.

This module is deliberately safe to import in offline test processes: PyTorch,
Transformers, Hugging Face access, and model construction are all deferred to
explicit runtime functions.  The command is intended to run on the rented GPU
only after its vLLM server has been stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-32B-AWQ"
DATASETS = ("msra", "conll2003", "wnut17", "fewnerd", "ontonotes5")
NOISE_TYPES = ("BT", "IF", "ATF")
SEEDS = (13, 42, 2024)
OFFICIAL_ROWS_PER_FILE = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20250317
DEFAULT_CONTROL_SEED = "lad-rg-logit-gap-controls-v1"
TRANSACTION_PROTOCOL = "aggregate-installed-last"
TRANSACTION_VERSION = 1

# Audited copy of multi_agent_v2.DATASET_ENTITY_TYPES.  Importing that module
# would construct OpenAI-compatible clients, which violates this probe's safe
# import contract.
DATASET_ENTITY_TYPES = {
    "msra": ("PER", "LOC", "ORG"),
    "conll2003": ("PER", "LOC", "ORG", "MISC"),
    "wnut17": ("PER", "LOC", "ORG", "MISC"),
    "fewnerd": ("PER", "LOC", "ORG", "MISC"),
    "ontonotes5": ("PER", "LOC", "ORG", "MISC"),
}

_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_SAFE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class OutputTransactionError(RuntimeError):
    """A commit failed and at least one rollback/cleanup action also failed."""

    def __init__(
        self,
        primary_error: BaseException,
        rollback_errors: Sequence[BaseException],
        recovery_paths: Sequence[Path],
    ) -> None:
        self.primary_error = primary_error
        self.rollback_errors = tuple(rollback_errors)
        self.recovery_paths = tuple(recovery_paths)
        failures = "; ".join(
            f"{type(error).__name__}: {error}" for error in self.rollback_errors
        )
        recoverable = ", ".join(str(path) for path in self.recovery_paths) or "none"
        super().__init__(
            f"output transaction failed with {type(primary_error).__name__}: "
            f"{primary_error}; rollback failed: {failures}; "
            f"recoverable files preserved: {recoverable}"
        )


@dataclass(frozen=True)
class ProbeItem:
    dataset: str
    noise: str
    seed: int
    sentence_index: int
    token_index: int
    group: str
    gold_tag: str
    dirty_tag: str
    tokens: tuple[str, ...]
    dirty_tags: tuple[str, ...]


def validate_runtime_identity(revision: str | None, backbone_tag: str | None) -> tuple[str, str]:
    """Validate the immutable model identity and artifact-safe output tag."""
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("BACKBONE_REVISION must be exactly 40 hexadecimal characters")
    if not isinstance(backbone_tag, str) or not backbone_tag:
        raise ValueError("BACKBONE_TAG must be nonempty")
    if not _SAFE_TAG_RE.fullmatch(backbone_tag):
        raise ValueError("BACKBONE_TAG must be a safe filename component")
    return revision.lower(), backbone_tag


def build_codebook(dataset: str) -> dict[str, str]:
    """Return code -> IOB2 tag in the project's fixed ontology order."""
    try:
        entity_types = DATASET_ENTITY_TYPES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset ontology: {dataset!r}") from exc
    tags = ["O"]
    for entity_type in entity_types:
        tags.extend((f"B-{entity_type}", f"I-{entity_type}"))
    return {str(index): tag for index, tag in enumerate(tags)}


def _valid_tags(dataset: str) -> frozenset[str]:
    return frozenset(build_codebook(dataset).values())


def render_target_prompt(
    tokenizer: Any,
    *,
    tokens: Sequence[str],
    dirty_tags: Sequence[str],
    target_index: int,
    dataset: str,
    codebook: Mapping[str, str] | None = None,
) -> str:
    """Render one deterministic Qwen chat context with thinking disabled."""
    if not tokens or len(tokens) != len(dirty_tags):
        raise ValueError("prompt requires equal nonempty tokens and dirty tags")
    if not 0 <= target_index < len(tokens):
        raise ValueError(f"target index {target_index} is out of bounds")
    expected_codebook = build_codebook(dataset)
    selected_codebook = dict(codebook or expected_codebook)
    if selected_codebook != expected_codebook:
        raise ValueError(f"codebook does not match the {dataset} ontology order")

    user_content = "\n".join(
        (
            "Classify the target token with one IOB2 answer code.",
            f"Dataset: {dataset}",
            f"Tokens: {json.dumps(list(tokens), ensure_ascii=False)}",
            f"Dirty tags: {json.dumps(list(dirty_tags), ensure_ascii=False)}",
            f"Target token index: {target_index}",
            f"Target token: {json.dumps(tokens[target_index], ensure_ascii=False)}",
            "Codebook (answer code -> IOB2 tag): "
            + json.dumps(selected_codebook, ensure_ascii=False),
            "Answer with exactly one code from the codebook and no other text.",
            "Answer code:",
        )
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a raw next-token NER classifier. Do not explain, reason aloud, "
                "or emit punctuation."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "the pinned Qwen tokenizer must support enable_thinking=False in its chat template"
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Qwen chat template returned an empty or non-text prompt")
    return rendered


def contextual_code_token_ids(
    tokenizer: Any, prompt: str, codebook: Mapping[str, str]
) -> dict[str, int]:
    """Validate each bare answer code as exactly one token after ``prompt``."""
    context_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not context_ids:
        raise ValueError("rendered prompt tokenized to an empty context")
    token_ids: dict[str, int] = {}
    for code in codebook:
        combined_ids = list(tokenizer.encode(prompt + code, add_special_tokens=False))
        if combined_ids[: len(context_ids)] != context_ids:
            raise ValueError(
                f"answer code {code!r} changes the rendered prompt tokenization"
            )
        suffix = combined_ids[len(context_ids) :]
        if len(suffix) != 1:
            raise ValueError(
                f"answer code {code!r} must be exactly one next-generation token "
                "in the rendered prompt context"
            )
        token_id = suffix[0]
        if not isinstance(token_id, int):
            raise ValueError(f"answer code {code!r} produced a non-integer token ID")
        token_ids[code] = token_id
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("answer codes must map to distinct contextual token IDs")
    return token_ids


def canonical_noisy_paths(
    input_root: Path,
    *,
    datasets: Sequence[str] = DATASETS,
    noise_types: Sequence[str] = NOISE_TYPES,
    seeds: Sequence[int] = SEEDS,
) -> set[Path]:
    root = Path(input_root)
    return {
        root / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
        for dataset in datasets
        for noise in noise_types
        for seed in seeds
    }


def _is_legal_gold_iob2(tags: Sequence[str]) -> bool:
    previous_prefix = "O"
    previous_type: str | None = None
    for tag in tags:
        if tag == "O":
            previous_prefix, previous_type = "O", None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_type == entity_type
        ):
            return False
        previous_prefix, previous_type = prefix, entity_type
    return True


def _row_lists(row: Any, path: Path, sentence_index: int) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(row, dict):
        raise ValueError(f"{path.name} row {sentence_index} must be a JSON object")
    tokens = row.get("tokens")
    gold = row.get("ner_tags")
    dirty = row.get("dirty_tags")
    values = (tokens, gold, dirty)
    if not (
        all(isinstance(value, list) for value in values)
        and len(tokens) > 0
        and len(tokens) == len(gold) == len(dirty)
        and all(isinstance(item, str) for value in values for item in value)
    ):
        raise ValueError(
            f"{path.name} row {sentence_index} requires equal nonempty token/gold/dirty lengths "
            "containing only strings"
        )
    return tokens, gold, dirty


def _stable_control_sample(
    candidates: Sequence[ProbeItem], count: int, *, control_seed: str, filename: str
) -> list[ProbeItem]:
    if len(candidates) < count:
        raise ValueError(
            f"{filename} needs {count} unchanged gold-O controls, found {len(candidates)}"
        )

    def rank(item: ProbeItem) -> tuple[bytes, int, int]:
        material = (
            f"{control_seed}\0{filename}\0{item.sentence_index}\0{item.token_index}"
        ).encode("utf-8")
        return hashlib.sha256(material).digest(), item.sentence_index, item.token_index

    selected = sorted(candidates, key=rank)[:count]
    return sorted(selected, key=lambda item: (item.sentence_index, item.token_index))


def _summarize_filenames(names: Sequence[str]) -> str:
    preview = ", ".join(names[:3])
    remaining = len(names) - 3
    return preview if remaining <= 0 else f"{preview}, ... (+{remaining} more)"


def load_probe_items(
    input_root: Path,
    *,
    datasets: Sequence[str] = DATASETS,
    noise_types: Sequence[str] = NOISE_TYPES,
    seeds: Sequence[int] = SEEDS,
    expected_rows: int = OFFICIAL_ROWS_PER_FILE,
    control_seed: str = DEFAULT_CONTROL_SEED,
) -> list[ProbeItem]:
    """Validate the complete source matrix and select corrupt/control positions."""
    root = Path(input_root)
    if not isinstance(control_seed, str) or not control_seed:
        raise ValueError("control_seed must be nonempty")
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    unknown = [dataset for dataset in datasets if dataset not in DATASET_ENTITY_TYPES]
    if unknown:
        raise ValueError(f"unknown dataset ontologies: {unknown}")
    unknown_noise = [noise for noise in noise_types if noise not in NOISE_TYPES]
    if unknown_noise:
        raise ValueError(f"unknown noise types: {unknown_noise}")

    expected = canonical_noisy_paths(
        root, datasets=datasets, noise_types=noise_types, seeds=seeds
    )
    actual = set(root.glob("noisy_seed*.jsonl")) if root.is_dir() else set()
    missing = sorted(path.name for path in expected - actual)
    extra = sorted(path.name for path in actual - expected)
    if missing:
        raise ValueError(
            f"missing canonical noisy-data files ({len(missing)}): "
            f"{_summarize_filenames(missing)}"
        )
    if extra:
        raise ValueError(
            f"unexpected noisy-data files ({len(extra)}): "
            f"{_summarize_filenames(extra)}"
        )

    items: list[ProbeItem] = []
    ordered_paths = [
        root / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
        for dataset in datasets
        for noise in noise_types
        for seed in seeds
    ]
    for path in ordered_paths:
        parts = path.stem.split("__")
        seed = int(parts[0].removeprefix("noisy_seed"))
        noise = parts[1]
        dataset = parts[2]
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != expected_rows:
            raise ValueError(f"{path.name} has {len(lines)} rows; expected {expected_rows}")

        valid_tags = _valid_tags(dataset)
        corrupted: list[ProbeItem] = []
        controls: list[ProbeItem] = []
        for sentence_index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} row {sentence_index} is not valid JSON"
                ) from exc
            tokens, gold, dirty = _row_lists(row, path, sentence_index)
            invalid = sorted(set(gold + dirty) - valid_tags)
            if invalid:
                raise ValueError(
                    f"{path.name} row {sentence_index} has out-of-ontology tags: {invalid}"
                )
            if not _is_legal_gold_iob2(gold):
                raise ValueError(
                    f"{path.name} row {sentence_index} has illegal gold IOB2 tags"
                )
            token_tuple = tuple(tokens)
            dirty_tuple = tuple(dirty)
            for token_index, (gold_tag, dirty_tag) in enumerate(zip(gold, dirty)):
                if gold_tag != dirty_tag:
                    corrupted.append(
                        ProbeItem(
                            dataset, noise, seed, sentence_index, token_index,
                            "corrupted", gold_tag, dirty_tag, token_tuple, dirty_tuple,
                        )
                    )
                elif gold_tag == "O":
                    controls.append(
                        ProbeItem(
                            dataset, noise, seed, sentence_index, token_index,
                            "control", gold_tag, dirty_tag, token_tuple, dirty_tuple,
                        )
                    )
        items.extend(corrupted)
        items.extend(
            _stable_control_sample(
                controls,
                len(corrupted),
                control_seed=control_seed,
                filename=path.name,
            )
        )
    return items


def gather_code_logits(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    contextual_token_ids: Sequence[Mapping[str, int]],
) -> list[list[float]]:
    """Run one padded batch and gather code logits at each true final token."""
    if not prompts or len(prompts) != len(contextual_token_ids):
        raise ValueError("prompts and contextual token IDs must have equal nonzero length")
    import torch

    encoded = tokenizer(
        list(prompts), add_special_tokens=False, padding=True, return_tensors="pt"
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise ValueError("tokenizer batch must include input_ids and attention_mask")
    device = getattr(model, "device", None)
    model_inputs = {
        key: value.to(device) if device is not None and hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    attention_mask = model_inputs["attention_mask"]
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    sequence_positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).expand_as(attention_mask)
    final_positions = torch.where(
        attention_mask.bool(), sequence_positions, torch.full_like(sequence_positions, -1)
    ).max(dim=1).values
    if bool((final_positions < 0).any().item()):
        raise ValueError("every prompt must contain at least one non-padding token")

    with torch.inference_mode():
        outputs = model(**model_inputs)
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != attention_mask.shape:
            raise ValueError("model logits must have shape [batch, sequence, vocabulary]")
        batch_positions = torch.arange(logits.shape[0], device=logits.device)
        next_logits = logits[batch_positions, final_positions.to(logits.device), :]
        code_id_rows = [list(token_ids.values()) for token_ids in contextual_token_ids]
        width = len(code_id_rows[0])
        if width < 2 or any(len(row) != width for row in code_id_rows):
            raise ValueError("each prompt must provide the same O-plus-entity code IDs")
        code_ids = torch.tensor(code_id_rows, dtype=torch.long, device=logits.device)
        if bool((code_ids < 0).any().item()) or bool((code_ids >= logits.shape[-1]).any().item()):
            raise ValueError("a contextual answer-code token ID is outside the model vocabulary")
        selected = next_logits.gather(1, code_ids)
    return [[float(value) for value in row] for row in selected.detach().cpu().tolist()]


def score_code_logits(codebook: Mapping[str, str], logits: Sequence[float]) -> dict[str, Any]:
    if list(codebook.items())[:1] != [("0", "O")]:
        raise ValueError("codebook must begin with 0=O")
    if len(logits) != len(codebook) or len(logits) < 2:
        raise ValueError("logit count must match a nontrivial codebook")
    numeric = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("all gathered logits must be finite")
    entity_index = max(range(1, len(numeric)), key=numeric.__getitem__)
    entity_code = list(codebook)[entity_index]
    o_logit = numeric[0]
    entity_logit = numeric[entity_index]
    return {
        "o_logit": o_logit,
        "strongest_entity_logit": entity_logit,
        "strongest_entity_label": codebook[entity_code],
        "delta_z": entity_logit - o_logit,
    }


def _derived_bootstrap_seed(seed: int, key: tuple[str, str, str]) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _bootstrap_mean_cis(
    metric_values: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    import numpy as np

    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in metric_values.items()
    }
    sizes = {array.size for array in arrays.values()}
    if not arrays or sizes == {0}:
        raise ValueError("cannot bootstrap an empty group")
    if len(sizes) != 1:
        raise ValueError("bootstrap metrics must have equal lengths")
    sample_size = next(iter(sizes))
    if sample_size == 1:
        return {
            name: {"low": float(array[0]), "high": float(array[0])}
            for name, array in arrays.items()
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    means = {
        name: np.empty(iterations, dtype=np.float64) for name in arrays
    }
    # Cap the temporary index matrix at roughly one million int64 entries.
    chunk_size = min(512, iterations, max(1, 1_000_000 // sample_size))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(
            0, sample_size, size=(stop - start, sample_size)
        )
        for name, array in arrays.items():
            means[name][start:stop] = array[indices].mean(axis=1)
    return {
        name: {
            "low": float(np.percentile(metric_means, 2.5)),
            "high": float(np.percentile(metric_means, 97.5)),
        }
        for name, metric_means in means.items()
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["dataset"]), str(record["noise"]), str(record["group"]))
        grouped[key].append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        delta_values = [float(row["delta_z"]) for row in rows]
        o_values = [float(row["o_logit"]) for row in rows]
        entity_values = [float(row["strongest_entity_logit"]) for row in rows]
        if not all(
            math.isfinite(value)
            for values in (delta_values, o_values, entity_values)
            for value in values
        ):
            raise ValueError(f"non-finite logit in aggregate group {key}")
        count = len(rows)
        group = {
            "dataset": key[0],
            "noise": key[1],
            "group": key[2],
            "count": count,
            "mean_delta_z": math.fsum(delta_values) / count,
            "mean_o_logit": math.fsum(o_values) / count,
            "mean_strongest_entity_logit": math.fsum(entity_values) / count,
        }
        group.update(
            _bootstrap_mean_cis(
                {
                    "mean_delta_z_ci95": delta_values,
                    "mean_o_logit_ci95": o_values,
                    "mean_strongest_entity_logit_ci95": entity_values,
                },
                iterations=bootstrap_iterations,
                seed=_derived_bootstrap_seed(bootstrap_seed, key),
            )
        )
        groups.append(group)
    return {
        "bootstrap": {
            "method": "fixed-seed percentile bootstrap of the mean",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "shared_resample_indices": True,
        },
        "groups": groups,
    }


def output_paths(output_root: Path, backbone_tag: str) -> tuple[Path, Path]:
    validate_runtime_identity("0" * 40, backbone_tag)
    root = Path(output_root)
    base = root / f"logit_gap__{backbone_tag}.jsonl"
    aggregate = root / f"logit_gap__{backbone_tag}__aggregate.json"
    return base, aggregate


def _write_staged(path: Path, content: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    staged = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as primary_error:
        try:
            staged.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            raise OutputTransactionError(
                primary_error, [cleanup_error], [staged] if staged.exists() else []
            ) from primary_error
        raise
    return staged


def _transaction_marker(
    jsonl_path: Path, jsonl_bytes: bytes, record_count: int, transaction_id: str
) -> dict[str, Any]:
    return {
        "protocol": TRANSACTION_PROTOCOL,
        "version": TRANSACTION_VERSION,
        "transaction_id": transaction_id,
        "jsonl_name": jsonl_path.name,
        "jsonl_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "jsonl_bytes": len(jsonl_bytes),
        "jsonl_line_count": jsonl_bytes.count(b"\n"),
        "jsonl_record_count": record_count,
    }


def validate_output_pair(jsonl_path: Path, aggregate_path: Path) -> dict[str, Any]:
    """Validate that the aggregate marker commits the exact canonical JSONL."""
    jsonl_path = Path(jsonl_path)
    aggregate_path = Path(aggregate_path)
    try:
        jsonl_bytes = jsonl_path.read_bytes()
        aggregate_value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("output pair is missing or unreadable") from exc
    if not isinstance(aggregate_value, dict):
        raise ValueError("aggregate output must be a JSON object")
    marker = aggregate_value.get("artifact_transaction")
    if not isinstance(marker, dict):
        raise ValueError("aggregate output has no artifact transaction marker")

    lines = jsonl_bytes.splitlines()
    try:
        for line in lines:
            if not line:
                raise ValueError("blank JSONL record")
            json.loads(line)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("canonical JSONL contains an invalid record") from exc
    expected = {
        "protocol": TRANSACTION_PROTOCOL,
        "version": TRANSACTION_VERSION,
        "transaction_id": marker.get("transaction_id"),
        "jsonl_name": jsonl_path.name,
        "jsonl_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "jsonl_bytes": len(jsonl_bytes),
        "jsonl_line_count": jsonl_bytes.count(b"\n"),
        "jsonl_record_count": len(lines),
    }
    transaction_id = marker.get("transaction_id")
    if not isinstance(transaction_id, str) or not re.fullmatch(r"[0-9a-f]{32}", transaction_id):
        raise ValueError("aggregate transaction id is invalid")
    if marker != expected:
        raise ValueError("canonical JSONL does not match its aggregate commit marker")
    return dict(marker)


def _path_matches(path: Path, expected: bytes | None) -> bool:
    try:
        if expected is None:
            return not path.exists()
        return path.is_file() and path.read_bytes() == expected
    except OSError:
        return False


def _cleanup_paths(paths: Sequence[Path]) -> list[BaseException]:
    errors: list[BaseException] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except BaseException as exc:
            errors.append(exc)
    return errors


def _restore_old_pair(
    *,
    jsonl_path: Path,
    aggregate_path: Path,
    old_jsonl: bytes | None,
    old_aggregate: bytes | None,
    jsonl_backup: Path,
    aggregate_backup: Path,
) -> list[BaseException]:
    """Restore the pre-transaction bytes, restoring the aggregate marker last."""
    errors: list[BaseException] = []
    if _path_matches(jsonl_path, old_jsonl) and _path_matches(
        aggregate_path, old_aggregate
    ):
        return errors

    # No aggregate marker may remain canonical while the JSONL is being restored.
    if aggregate_path.exists() and not _path_matches(aggregate_path, old_aggregate):
        try:
            aggregate_path.unlink()
        except BaseException as exc:
            errors.append(exc)

    if old_jsonl is None:
        if jsonl_path.exists():
            try:
                jsonl_path.unlink()
            except BaseException as exc:
                errors.append(exc)
    elif not _path_matches(jsonl_path, old_jsonl):
        if _path_matches(jsonl_backup, old_jsonl):
            try:
                os.replace(jsonl_backup, jsonl_path)
            except BaseException as exc:
                errors.append(exc)
        else:
            errors.append(RuntimeError("old JSONL backup is unavailable"))

    jsonl_restored = _path_matches(jsonl_path, old_jsonl)
    if not jsonl_restored:
        errors.append(RuntimeError("canonical JSONL was not restored exactly"))
        if aggregate_path.exists():
            try:
                if old_aggregate is not None and _path_matches(
                    aggregate_path, old_aggregate
                ) and not aggregate_backup.exists():
                    os.replace(aggregate_path, aggregate_backup)
                else:
                    aggregate_path.unlink()
            except BaseException as exc:
                errors.append(exc)
        return errors

    if old_aggregate is None:
        if aggregate_path.exists():
            try:
                aggregate_path.unlink()
            except BaseException as exc:
                errors.append(exc)
    elif not _path_matches(aggregate_path, old_aggregate):
        if _path_matches(aggregate_backup, old_aggregate):
            try:
                os.replace(aggregate_backup, aggregate_path)
            except BaseException as exc:
                errors.append(exc)
        else:
            errors.append(RuntimeError("old aggregate backup is unavailable"))

    if not _path_matches(aggregate_path, old_aggregate):
        errors.append(RuntimeError("canonical aggregate was not restored exactly"))
    return errors


def write_outputs_atomic(
    output_root: Path,
    backbone_tag: str,
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Commit JSONL plus aggregate, using the aggregate as the last-installed marker."""
    jsonl_path, aggregate_path = output_paths(output_root, backbone_tag)
    jsonl_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    jsonl_bytes = jsonl_text.encode("utf-8")
    transaction_id = uuid.uuid4().hex
    aggregate_value = dict(aggregate)
    aggregate_value["artifact_transaction"] = _transaction_marker(
        jsonl_path, jsonl_bytes, len(records), transaction_id
    )
    aggregate_text = (
        json.dumps(aggregate_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    final_paths = (jsonl_path, aggregate_path)
    staged_paths: list[Path] = []
    try:
        staged_paths.append(_write_staged(jsonl_path, jsonl_text))
        staged_paths.append(_write_staged(aggregate_path, aggregate_text))
    except BaseException as primary_error:
        cleanup_errors = _cleanup_paths(staged_paths)
        if cleanup_errors:
            recovery_paths = [path for path in staged_paths if path.exists()]
            raise OutputTransactionError(
                primary_error, cleanup_errors, recovery_paths
            ) from primary_error
        raise

    jsonl_staged, aggregate_staged = staged_paths
    jsonl_backup = jsonl_path.with_name(f".{jsonl_path.name}.{transaction_id}.bak")
    aggregate_backup = aggregate_path.with_name(
        f".{aggregate_path.name}.{transaction_id}.bak"
    )
    try:
        old_jsonl = jsonl_path.read_bytes() if jsonl_path.exists() else None
        old_aggregate = aggregate_path.read_bytes() if aggregate_path.exists() else None
    except BaseException as primary_error:
        cleanup_errors = _cleanup_paths(staged_paths)
        if cleanup_errors:
            recovery_paths = [path for path in staged_paths if path.exists()]
            raise OutputTransactionError(
                primary_error, cleanup_errors, recovery_paths
            ) from primary_error
        raise
    try:
        # Remove the old commit marker before any canonical JSONL transition.
        if old_aggregate is not None:
            os.replace(aggregate_path, aggregate_backup)
        if old_jsonl is not None:
            os.replace(jsonl_path, jsonl_backup)
        os.replace(jsonl_staged, jsonl_path)
        # Installing the aggregate is the atomic commit point and must be last.
        os.replace(aggregate_staged, aggregate_path)
        validate_output_pair(jsonl_path, aggregate_path)
    except BaseException as primary_error:
        rollback_errors = _restore_old_pair(
            jsonl_path=jsonl_path,
            aggregate_path=aggregate_path,
            old_jsonl=old_jsonl,
            old_aggregate=old_aggregate,
            jsonl_backup=jsonl_backup,
            aggregate_backup=aggregate_backup,
        )
        recovery_candidates = [
            jsonl_staged,
            aggregate_staged,
            jsonl_backup,
            aggregate_backup,
        ]
        if rollback_errors:
            recovery_paths = [path for path in recovery_candidates if path.exists()]
            raise OutputTransactionError(
                primary_error, rollback_errors, recovery_paths
            ) from primary_error
        _cleanup_paths(recovery_candidates)
        raise

    # Once the marker validates, cleanup is best effort and cannot revoke the commit.
    _cleanup_paths([jsonl_staged, aggregate_staged, jsonl_backup, aggregate_backup])
    return final_paths


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def model_fingerprint(model: Any, revision: str) -> str:
    config = getattr(model, "config", None)
    config_value = config.to_dict() if hasattr(config, "to_dict") else str(config)
    payload = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model": MODEL_ID,
        "revision": revision,
        "config": config_value,
    }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def tokenizer_fingerprint(tokenizer: Any, revision: str) -> str:
    payload = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "model": MODEL_ID,
        "revision": revision,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        for token, token_id in sorted(vocab.items(), key=lambda pair: (pair[1], pair[0])):
            digest.update(_canonical_json_bytes([token_id, token]))
    return "sha256:" + digest.hexdigest()


def load_model_and_tokenizer(model_id: str, revision: str) -> tuple[Any, Any]:
    """Lazily load the exact pinned model for direct local GPU inference."""
    if model_id != MODEL_ID:
        raise ValueError(f"this probe only supports {MODEL_ID}")
    validate_runtime_identity(revision, "runtime-validation")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the logit-gap probe requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("the pinned tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    return tokenizer, model


def _record_from_item(
    item: ProbeItem,
    *,
    codebook: Mapping[str, str],
    code_token_ids: Mapping[str, int],
    logits: Sequence[float],
    revision: str,
    backbone_tag: str,
    model_hash: str,
    tokenizer_hash: str,
) -> dict[str, Any]:
    record = {
        "dataset": item.dataset,
        "noise": item.noise,
        "seed": item.seed,
        "sentence_index": item.sentence_index,
        "token_index": item.token_index,
        "group": item.group,
        "gold_tag": item.gold_tag,
        "dirty_tag": item.dirty_tag,
        "model": MODEL_ID,
        "revision": revision,
        "backbone_tag": backbone_tag,
        "model_fingerprint": model_hash,
        "tokenizer_fingerprint": tokenizer_hash,
        "codebook": dict(codebook),
        "code_token_ids": dict(code_token_ids),
    }
    record.update(score_code_logits(codebook, logits))
    return record


def _ontology_batches(
    items: Sequence[ProbeItem], batch_size: int
) -> list[list[tuple[int, ProbeItem]]]:
    """Group deterministic item positions by dataset and exact codebook shape."""
    buckets: dict[tuple[str, tuple[tuple[str, str], ...]], list[tuple[int, ProbeItem]]] = {}
    for index, item in enumerate(items):
        key = (item.dataset, tuple(build_codebook(item.dataset).items()))
        buckets.setdefault(key, []).append((index, item))
    return [
        bucket[start : start + batch_size]
        for bucket in buckets.values()
        for start in range(0, len(bucket), batch_size)
    ]


def run_probe(
    *,
    revision: str,
    backbone_tag: str,
    input_root: Path = Path("results_multiseed"),
    output_root: Path = Path("predictions_multiseed"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    control_seed: str = DEFAULT_CONTROL_SEED,
    model_loader: Callable[[str, str], tuple[Any, Any]] = load_model_and_tokenizer,
) -> dict[str, Any]:
    revision, backbone_tag = validate_runtime_identity(revision, backbone_tag)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    # Fail before allocating GPU memory if any official source artifact is bad.
    items = load_probe_items(Path(input_root), control_seed=control_seed)
    tokenizer, model = model_loader(MODEL_ID, revision)
    model_hash = model_fingerprint(model, revision)
    tokenizer_hash = tokenizer_fingerprint(tokenizer, revision)

    ordered_records: list[dict[str, Any] | None] = [None] * len(items)
    for indexed_batch in _ontology_batches(items, batch_size):
        batch = [item for _, item in indexed_batch]
        prompts: list[str] = []
        codebooks: list[dict[str, str]] = []
        token_id_maps: list[dict[str, int]] = []
        for item in batch:
            codebook = build_codebook(item.dataset)
            prompt = render_target_prompt(
                tokenizer,
                tokens=item.tokens,
                dirty_tags=item.dirty_tags,
                target_index=item.token_index,
                dataset=item.dataset,
                codebook=codebook,
            )
            token_ids = contextual_code_token_ids(tokenizer, prompt, codebook)
            prompts.append(prompt)
            codebooks.append(codebook)
            token_id_maps.append(token_ids)
        batch_logits = gather_code_logits(model, tokenizer, prompts, token_id_maps)
        for (item_index, item), codebook, token_ids, logits in zip(
            indexed_batch, codebooks, token_id_maps, batch_logits
        ):
            ordered_records[item_index] = _record_from_item(
                item,
                codebook=codebook,
                code_token_ids=token_ids,
                logits=logits,
                revision=revision,
                backbone_tag=backbone_tag,
                model_hash=model_hash,
                tokenizer_hash=tokenizer_hash,
            )

    if any(record is None for record in ordered_records):
        raise RuntimeError("internal error: a probe item was not scored")
    records = [record for record in ordered_records if record is not None]

    aggregate = aggregate_records(
        records,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    aggregate.update(
        {
            "schema_version": "lad-rg-logit-gap-v1",
            "record_count": len(records),
            "model": MODEL_ID,
            "revision": revision,
            "backbone_tag": backbone_tag,
            "model_fingerprint": model_hash,
            "tokenizer_fingerprint": tokenizer_hash,
            "control_sampling": {
                "method": "SHA-256 rank without replacement within each source file",
                "seed": control_seed,
            },
        }
    )
    jsonl_path, aggregate_path = write_outputs_atomic(
        Path(output_root), backbone_tag, records, aggregate
    )
    return {
        "records": len(records),
        "jsonl": str(jsonl_path),
        "aggregate": str(aggregate_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe pinned Qwen/Qwen3-32B-AWQ raw next-token IOB2 logits. "
            "Stop vLLM before running this GPU command."
        )
    )
    parser.add_argument("--revision", default=os.environ.get("BACKBONE_REVISION"))
    parser.add_argument("--backbone-tag", default=os.environ.get("BACKBONE_TAG"))
    parser.add_argument("--input-root", type=Path, default=Path("results_multiseed"))
    parser.add_argument("--output-root", type=Path, default=Path("predictions_multiseed"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--control-seed", default=DEFAULT_CONTROL_SEED)
    parser.add_argument(
        "--confirm-vllm-stopped",
        action="store_true",
        help="Required acknowledgement that vLLM no longer holds the GPU model memory.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    model_loader: Callable[[str, str], tuple[Any, Any]] = load_model_and_tokenizer,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        revision, backbone_tag = validate_runtime_identity(
            args.revision, args.backbone_tag
        )
        if not args.confirm_vllm_stopped:
            raise ValueError(
                "--confirm-vllm-stopped is required before GPU model loading"
            )
        result = run_probe(
            revision=revision,
            backbone_tag=backbone_tag,
            input_root=args.input_root,
            output_root=args.output_root,
            batch_size=args.batch_size,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            control_seed=args.control_seed,
            model_loader=model_loader,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
