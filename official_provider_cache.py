"""Immutable, provider-stage cache cells for Contextual Lattice runs.

This module deliberately owns only the on-disk contract.  The runner stages
that produce and replay these cells are introduced separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROVIDER_CACHE_SCHEMA = "selectdenoise-contextual-provider-cache-v1"
_RECORD_FIELDS = frozenset({
    "row_index", "input_digest", "anchor_tags", "candidate_paths",
    "rag_weights", "confidence", "provider_metadata", "fallback_used",
})
_STAGES = frozenset({"coder", "reviewer", "verifier"})
_FORBIDDEN_GOLD_FIELDS = frozenset({"gold_tags", "ner_tags"})


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{description} must be a SHA-256 hex digest") from exc
    return value.lower()


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("provider-cache manifest must be a mapping")
    normalized = dict(manifest)
    if normalized.get("schema") != PROVIDER_CACHE_SCHEMA:
        raise ValueError(f"provider-cache manifest schema must be {PROVIDER_CACHE_SCHEMA!r}")
    source_digests = normalized.get("source_digests")
    if not isinstance(source_digests, Mapping):
        raise ValueError("provider-cache manifest source_digests must be a mapping")
    normalized["source_digests"] = {
        str(index): _require_sha256(digest, f"source digest for row {index}")
        for index, digest in source_digests.items()
    }
    return normalized


def _validate_provider_metadata(metadata: Any) -> None:
    if not isinstance(metadata, Mapping) or set(metadata) != _STAGES:
        raise ValueError("provider_metadata must contain exactly coder, reviewer, and verifier")
    for stage in _STAGES:
        evidence = metadata[stage]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"provider_metadata.{stage} must contain explicit stage evidence")
        for record in evidence:
            if not isinstance(record, Mapping) or record.get("stage") is None or record.get("status") is None:
                raise ValueError(f"provider_metadata.{stage} contains malformed stage evidence")


def _validate_records(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("provider-cache records must be a sequence")
    source_digests = manifest["source_digests"]
    normalized: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("provider-cache record must be a mapping")
        if _FORBIDDEN_GOLD_FIELDS.intersection(record):
            raise ValueError("provider-cache records must not contain gold labels")
        if set(record) != _RECORD_FIELDS:
            raise ValueError("provider-cache record has an invalid field set")
        index = record["row_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("row_index must be a non-negative integer")
        if index in seen_indices:
            raise ValueError(f"duplicate row_index {index}")
        seen_indices.add(index)
        digest = _require_sha256(record["input_digest"], f"input digest for row {index}")
        if source_digests.get(str(index)) != digest:
            raise ValueError(f"source digest mismatch for row {index}")
        if not isinstance(record["anchor_tags"], list):
            raise ValueError("anchor_tags must be a list")
        if not isinstance(record["candidate_paths"], list):
            raise ValueError("candidate_paths must be a list")
        if not isinstance(record["rag_weights"], list):
            raise ValueError("rag_weights must be a list")
        if not isinstance(record["confidence"], list):
            raise ValueError("confidence must be a list")
        if record["fallback_used"] is not False:
            raise ValueError("provider-cache fallback_used must be false")
        _validate_provider_metadata(record["provider_metadata"])
        normalized.append(dict(record))
    return normalized


def _cell_bytes(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> bytes:
    header = {"schema": PROVIDER_CACHE_SCHEMA, "manifest": manifest}
    lines = [_canonical_json(header)]
    lines.extend(_canonical_json(record) for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_provider_cell(path: Path, records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    """Atomically write a validated cache cell and return its full-file SHA-256."""
    cell_path = Path(path)
    normalized_manifest = _validate_manifest(manifest)
    normalized_records = _validate_records(records, normalized_manifest)
    content = _cell_bytes(normalized_records, normalized_manifest)
    temporary = cell_path.with_suffix(cell_path.suffix + ".tmp")
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_bytes(content)
        temporary.replace(cell_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_bytes(content)


def read_provider_cell(path: Path, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    """Verify a complete cache cell before returning its provider-stage rows."""
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows < 0:
        raise ValueError("expected row count must be a non-negative integer")
    expected = _require_sha256(expected_sha256, "expected SHA-256")
    cell_path = Path(path)
    content = cell_path.read_bytes()
    if _sha256_bytes(content) != expected:
        raise ValueError("provider-cache SHA-256 mismatch")
    try:
        lines = content.decode("utf-8").splitlines()
        header = json.loads(lines[0])
        raw_records = [json.loads(line) for line in lines[1:]]
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider-cache cell is malformed") from exc
    if not isinstance(header, Mapping) or set(header) != {"schema", "manifest"}:
        raise ValueError("provider-cache header is malformed")
    if header["schema"] != PROVIDER_CACHE_SCHEMA:
        raise ValueError("provider-cache schema mismatch")
    manifest = _validate_manifest(header["manifest"])
    records = _validate_records(raw_records, manifest)
    if len(records) != expected_rows:
        raise ValueError(f"provider-cache row count mismatch: expected {expected_rows}, found {len(records)}")
    return records
