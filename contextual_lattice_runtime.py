"""Persistence and thread-safe inference for the locked lattice decoder.

The training/evaluation harness remains in ``run_contextual_lattice_gate``.
This module contains only the gold-free runtime bundle contract used by the
host SelectDenoise pipeline.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import joblib
except ImportError as exc:  # pragma: no cover - dependency is pinned for runtime use
    raise ImportError("The contextual lattice runtime requires joblib") from exc

from selectdenoise_contextual_lattice import (
    ContextualDecodeResult,
    ContextualLatticeDecoder,
    ContextualLatticeInput,
    ResidualGate,
    _SpanScorer,
    iob2_to_spans,
    refuse_overwrite,
)


SCHEMA_VERSION = "contextual-lattice-v1"
# Trusted v1 provenance.  These values are code-level pins in addition to the
# human-readable frozen manifest; a supplied bundle cannot redefine its own
# expected checkpoint or file hashes.
LOCKED_CHECKPOINT_HASH = "2100142f31627531497850659dcb3821c99d5e71c08a8e01a98e4b11ef32a199"
LOCKED_SPLIT_HASH = "fe4b00d32e655a334e8451e57a49f8d505047b313308bb5d60df0084865d421a"
LOCKED_BUNDLE_MANIFEST_HASH = "b24d4d5485e92e5513db246c48ed8d5258bd70b69c31889ae7d393186bda7825"
LOCKED_DECODER_FILE_HASH = "cc12773a2ee56db0c79452fff10b9f1cd026219c0aada21805e728ab02f717ca"
LOCKED_GATE_FILE_HASH = "2cd1febab79bdb1293852e1c607cd3496e3133329d073d67cd9028d3c874097c"
LOCKED_DECODER_MODEL_HASH = "2b45770a128cc5a716d3c7dbdc75ab059afc1a1b9ed471ea5cf1bd4f2a150e4e"
_DECODER_FILE = "decoder.pt"
_GATE_FILE = "gate.joblib"
_MANIFEST_FILE = "manifest.json"
_HASHES_FILE = "files.sha256.json"

# The bundle is deliberately self-describing but closed-world.  A new field
# requires a schema/version change rather than silently becoming model input.
_BUNDLE_METADATA_FIELDS = frozenset(
    {
        "bundle_schema",
        "schema_version",
        "checkpoint",
        "checkpoint_hash",
        "split_hash",
        "seed",
        "margin",
        "decoder_model_hash",
        "type_vocabulary",
        "hidden_size",
        "scalar_size",
        "decoder_file",
        "gate_file",
        "gate_model",
        "rows",
        "train_rows",
        "calibration_rows",
        "test_rows",
        "salt",
        "device",
        "epochs",
        "deer_profile",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_bundle(
    decoder: ContextualLatticeDecoder,
    gate: ResidualGate,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write a non-overwriting, self-describing frozen decoder bundle."""

    if decoder._model is None:  # noqa: SLF001 - persistence is an internal contract
        raise ValueError("Cannot persist an unfitted contextual lattice decoder")
    if gate.model is None:
        raise ValueError("Cannot persist an unfitted residual gate")
    model = decoder._model  # noqa: SLF001
    record = dict(manifest)
    record.setdefault("schema_version", SCHEMA_VERSION)
    record.setdefault("checkpoint_hash", str(decoder.encoder.checkpoint_hash))
    record.setdefault("split_hash", "unspecified")
    record.setdefault("seed", int(decoder.seed))
    record.setdefault("margin", float(gate.margin))
    record["decoder_model_hash"] = decoder.model_hash
    record["type_vocabulary"] = list(decoder.type_vocabulary)
    record["hidden_size"] = int(decoder.encoder.hidden_size)
    record["scalar_size"] = int(model.scalar_projection.in_features)
    record["decoder_file"] = _DECODER_FILE
    record["gate_file"] = _GATE_FILE
    record["gate_model"] = type(gate.model).__name__
    record["bundle_schema"] = SCHEMA_VERSION
    unexpected = set(record) - _BUNDLE_METADATA_FIELDS
    if unexpected:
        raise ValueError(f"Unexpected contextual lattice bundle metadata: {sorted(unexpected)}")

    refuse_overwrite(output_dir)
    output_dir.mkdir(parents=True)
    decoder_payload = {
        "state_dict": _cpu_state_dict(model),
        "hidden_size": int(decoder.encoder.hidden_size),
        "scalar_size": int(model.scalar_projection.in_features),
        "type_vocabulary": list(decoder.type_vocabulary),
        "seed": int(decoder.seed),
        "epochs": int(decoder.epochs),
        "batch_size": int(decoder.batch_size),
        "learning_rate": float(decoder.learning_rate),
        "weight_decay": float(decoder.weight_decay),
    }
    torch.save(decoder_payload, output_dir / _DECODER_FILE)
    joblib.dump(gate.model, output_dir / _GATE_FILE, compress=3)
    _write_json(output_dir / _MANIFEST_FILE, record)

    hashes = {
        _DECODER_FILE: _sha256(output_dir / _DECODER_FILE),
        _GATE_FILE: _sha256(output_dir / _GATE_FILE),
        _MANIFEST_FILE: _sha256(output_dir / _MANIFEST_FILE),
    }
    _write_json(output_dir / _HASHES_FILE, hashes)
    return output_dir


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location=device)


def _validate_hashes(bundle_dir: Path, expected: Mapping[str, str] | None = None) -> None:
    hash_path = bundle_dir / _HASHES_FILE
    if not hash_path.exists():
        raise ValueError(f"Missing bundle hash file: {hash_path}")
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    if not isinstance(hashes, dict):
        raise ValueError("Bundle hashes must be a JSON object")
    expected_names = {_DECODER_FILE, _GATE_FILE, _MANIFEST_FILE}
    if set(hashes) != expected_names:
        raise ValueError("Bundle hash inventory mismatch")
    for name, expected_digest in hashes.items():
        path = bundle_dir / name
        if not path.exists() or _sha256(path) != expected_digest:
            raise ValueError(f"Bundle hash mismatch for {name}")
    if expected:
        for name, digest in expected.items():
            path = bundle_dir / name
            if not path.exists() or _sha256(path) != digest:
                raise ValueError(f"Trusted bundle hash mismatch for {name}")


def load_bundle(
    bundle_dir: Path,
    encoder: Any,
    *,
    expected_checkpoint_hash: str,
    expected_schema_version: str = SCHEMA_VERSION,
    expected_split_hash: str | None = None,
    expected_manifest_hash: str | None = None,
    expected_decoder_file_hash: str | None = None,
    expected_gate_file_hash: str | None = None,
    expected_decoder_model_hash: str | None = None,
) -> "ContextualLatticeTerminal":
    """Load and validate a frozen bundle without fitting any parameters."""

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / _MANIFEST_FILE
    decoder_path = bundle_dir / _DECODER_FILE
    gate_path = bundle_dir / _GATE_FILE
    for path in (manifest_path, decoder_path, gate_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing contextual lattice bundle file: {path}")
    if expected_manifest_hash is not None and _sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Trusted contextual lattice manifest hash mismatch")
    expected_files = {
        name: digest
        for name, digest in (
            (_DECODER_FILE, expected_decoder_file_hash),
            (_GATE_FILE, expected_gate_file_hash),
        )
        if digest is not None
    }
    _validate_hashes(bundle_dir, expected_files)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest must be a JSON object")
    unexpected = set(manifest) - _BUNDLE_METADATA_FIELDS
    if unexpected:
        raise ValueError(f"Unexpected contextual lattice bundle metadata: {sorted(unexpected)}")
    if manifest.get("schema_version") != expected_schema_version or manifest.get("bundle_schema") != expected_schema_version:
        raise ValueError("Contextual lattice bundle schema mismatch")
    if manifest.get("checkpoint_hash") != expected_checkpoint_hash:
        raise ValueError("Contextual lattice bundle checkpoint hash mismatch")
    if expected_split_hash is not None and manifest.get("split_hash") != expected_split_hash:
        raise ValueError("Contextual lattice bundle split hash mismatch")
    if getattr(encoder, "checkpoint_hash", None) != expected_checkpoint_hash:
        raise ValueError("Loaded encoder checkpoint hash mismatch")
    margin = float(manifest.get("margin", float("nan")))
    if not np.isfinite(margin) or margin != -0.1:
        raise ValueError("Locked contextual lattice margin must be finite and equal to -0.1")

    device = torch.device(getattr(encoder, "device", "cpu"))
    payload = _torch_load(decoder_path, device)
    type_vocabulary = tuple(str(x) for x in payload["type_vocabulary"])
    hidden_size = int(payload["hidden_size"])
    scalar_size = int(payload["scalar_size"])
    if hidden_size != int(encoder.hidden_size) or manifest.get("hidden_size") != hidden_size:
        raise ValueError("Contextual lattice encoder dimension mismatch")
    if tuple(manifest.get("type_vocabulary", ())) != type_vocabulary:
        raise ValueError("Contextual lattice type vocabulary mismatch")

    decoder = ContextualLatticeDecoder(
        type_vocabulary,
        encoder=encoder,
        epochs=int(payload.get("epochs", 20)),
        batch_size=int(payload.get("batch_size", 16)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-4)),
        device=device,
        seed=int(payload.get("seed", 731)),
    )
    decoder._model = _SpanScorer(hidden_size, scalar_size, len(type_vocabulary), seed=decoder.seed).to(device)  # noqa: SLF001
    decoder._model.load_state_dict(payload["state_dict"])  # noqa: SLF001
    decoder._model.eval()  # noqa: SLF001
    decoder._model_hash = decoder._hash_model()  # noqa: SLF001
    if decoder.model_hash != manifest.get("decoder_model_hash"):
        raise ValueError("Contextual lattice decoder model hash mismatch")
    if expected_decoder_model_hash is not None and decoder.model_hash != expected_decoder_model_hash:
        raise ValueError("Trusted contextual lattice decoder model hash mismatch")

    gate = ResidualGate(seed=int(payload.get("seed", 731)))
    gate.model = joblib.load(gate_path)
    gate.margin = margin
    decoder.gate = gate
    decoder.margin = margin
    return ContextualLatticeTerminal(decoder, deer_profile=manifest.get("deer_profile", {}))


class ContextualLatticeTerminal:
    """Thread-safe, gold-free terminal decoder with anchor fallback."""

    def __init__(self, decoder: Any, *, deer_profile: Mapping[str, Any] | None = None) -> None:
        self.decoder = decoder
        self.deer_profile = dict(deer_profile or {})
        self._lock = threading.RLock()
        self._fallback_count = 0

    @property
    def fallback_count(self) -> int:
        with self._lock:
            return self._fallback_count

    @property
    def model_hash(self) -> str:
        return str(getattr(self.decoder, "model_hash", "unknown"))

    def sentence_deer_stats(self, tokens: Sequence[str]) -> dict[str, float]:
        token_entity = self.deer_profile.get("token_entity", {})
        token_type = self.deer_profile.get("token_type", {})
        vocabulary = set(self.deer_profile.get("vocabulary", ()))
        if not tokens:
            return {"entity": 0.0, "type": 0.0, "semantic": 0.0, "oov": 0.0}
        entity = float(np.mean([token_entity.get(token, 0.0) for token in tokens]))
        typ = float(np.mean([token_type.get(token, 0.0) for token in tokens]))
        return {
            "entity": entity,
            "type": typ,
            "semantic": 0.5 * (entity + typ),
            "oov": float(np.mean([token not in vocabulary for token in tokens])),
        }

    def decode(
        self,
        *,
        tokens: Sequence[str],
        dirty_tags: Sequence[str],
        anchor_tags: Sequence[str],
        candidate_paths: Sequence[Sequence[str]],
        reviewer_weights: Sequence[float],
        valid_types: frozenset[str],
        deer_stats: Mapping[str, float],
    ) -> ContextualDecodeResult:
        raw_paths = [tuple(path) for path in candidate_paths]
        raw_weights = tuple(float(weight) for weight in reviewer_weights)
        if raw_weights and len(raw_weights) != len(raw_paths):
            raise ValueError("reviewer_weights must match candidate_paths or be empty")
        legal_paths: list[tuple[str, ...]] = []
        legal_weights: list[float] = []
        for index, path in enumerate(raw_paths):
            if len(path) != len(tokens):
                continue
            legal_paths.append(path)
            if raw_weights:
                legal_weights.append(raw_weights[index])
        value = ContextualLatticeInput(
            tokens=tuple(tokens),
            dirty_tags=tuple(dirty_tags),
            anchor_tags=tuple(anchor_tags),
            candidate_paths=tuple(legal_paths),
            reviewer_weights=tuple(legal_weights) if raw_weights else (),
            valid_types=frozenset(valid_types),
            deer_stats=dict(deer_stats),
        )
        with self._lock:
            try:
                return self.decoder.decode(value)
            except Exception:  # noqa: BLE001 - terminal must fail closed to anchor
                self._fallback_count += 1
                return ContextualDecodeResult(
                    tags=value.anchor_tags,
                    raw_lattice_tags=value.anchor_tags,
                    used_anchor=True,
                    predicted_gain=0.0,
                    selected_spans=iob2_to_spans(value.anchor_tags),
                    model_hash=self.model_hash,
                )


__all__ = [
    "ContextualLatticeTerminal",
    "LOCKED_BUNDLE_MANIFEST_HASH",
    "LOCKED_CHECKPOINT_HASH",
    "LOCKED_DECODER_FILE_HASH",
    "LOCKED_DECODER_MODEL_HASH",
    "LOCKED_GATE_FILE_HASH",
    "LOCKED_SPLIT_HASH",
    "SCHEMA_VERSION",
    "load_bundle",
    "save_bundle",
]
