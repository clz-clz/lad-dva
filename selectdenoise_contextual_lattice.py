"""Anchor-aware contextual lattice replacement for SelectDenoise.

This module intentionally keeps the terminal decoder independent of the host
experiment's dataset/family metadata.  Historical/evaluator metadata is
handled by the companion cached-gate harness, never by these model-facing
objects or features.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import string
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

try:  # Torch is present in the mandated environment; keep import errors clear.
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only on an unsupported host
    raise ImportError("The contextual lattice decoder requires torch") from exc

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("The contextual lattice gate requires scikit-learn") from exc


SEED = 731
WINDOW_SIZE = 320
WINDOW_STRIDE = 256
CONTEXTUAL_DIM = 256
SCALAR_DIM = 64
TYPE_DIM = 16
MODEL_INPUT_FIELDS = frozenset(
    {
        "tokens",
        "dirty_tags",
        "anchor_tags",
        "candidate_paths",
        "reviewer_weights",
        "valid_types",
        "deer_stats",
    }
)
FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "dataset",
        "dataset_name",
        "family",
        "noise_type",
        "cell",
        "seed",
        "example_id",
        "sentence_group",
        "__noise_type__",
    }
)


@dataclass(frozen=True, order=True)
class Span:
    """An inclusive/exclusive token interval with an ontology type."""

    start: int
    end: int
    type: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid span interval ({self.start}, {self.end})")
        if not self.type or self.type.startswith(("B-", "I-")):
            raise ValueError(f"Invalid span type {self.type!r}")


def _tag_type(tag: str) -> str | None:
    if tag == "O":
        return None
    if not isinstance(tag, str) or len(tag) < 3 or tag[1] != "-" or tag[0] not in "BI":
        raise ValueError(f"Invalid IOB2 tag {tag!r}")
    return tag[2:]


def validate_iob2(tags: Sequence[str], valid_types: frozenset[str] | None = None) -> None:
    previous: str = "O"
    for tag in tags:
        typ = _tag_type(tag)
        if typ is not None and valid_types is not None and typ not in valid_types:
            raise ValueError(f"Unknown ontology type {typ!r}")
        if tag.startswith("I-") and (previous == "O" or previous[2:] != typ):
            raise ValueError(f"Illegal IOB2 transition {previous!r} -> {tag!r}")
        previous = tag


def _validate_tag_alphabet(tags: Sequence[str], valid_types: frozenset[str]) -> None:
    for tag in tags:
        typ = _tag_type(tag)
        if typ is not None and typ not in valid_types:
            raise ValueError(f"Unknown ontology type {typ!r}")


def iob2_to_spans(tags: Sequence[str], *, strict: bool = True) -> tuple[Span, ...]:
    """Extract exact spans.  Strict mode rejects malformed IOB2 transitions."""

    if strict:
        validate_iob2(tags)
    spans: list[Span] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            typ = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append(Span(i, j, typ))
            i = j
        elif tag.startswith("I-") and not strict:
            # Invalid source paths are filtered before this function in the
            # production path.  Permissive extraction is useful for diagnostics.
            typ = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append(Span(i, j, typ))
            i = j
        else:
            i += 1
    return tuple(spans)


def spans_to_iob2(spans: Sequence[Span], length: int, valid_types: frozenset[str]) -> tuple[str, ...]:
    if length < 0:
        raise ValueError("length must be non-negative")
    out = ["O"] * length
    ordered = sorted(spans, key=lambda s: (s.start, s.end, s.type))
    previous_end = 0
    for span in ordered:
        if span.type not in valid_types:
            raise ValueError(f"Unknown ontology type {span.type!r}")
        if span.end > length or span.start < previous_end:
            raise ValueError("Overlapping/out-of-range spans cannot form IOB2 tags")
        out[span.start] = f"B-{span.type}"
        for i in range(span.start + 1, span.end):
            out[i] = f"I-{span.type}"
        previous_end = span.end
    result = tuple(out)
    validate_iob2(result, valid_types)
    return result


@dataclass(frozen=True)
class ContextualLatticeInput:
    """Allow-listed, noise-blind model input."""

    tokens: tuple[str, ...]
    dirty_tags: tuple[str, ...]
    anchor_tags: tuple[str, ...]
    candidate_paths: tuple[tuple[str, ...], ...]
    reviewer_weights: tuple[float, ...]
    valid_types: frozenset[str]
    deer_stats: Any

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.dirty_tags) or len(self.tokens) != len(self.anchor_tags):
            raise ValueError("tokens, dirty_tags, and anchor_tags must have equal length")
        if not self.valid_types:
            raise ValueError("valid_types cannot be empty")
        if not isinstance(self.tokens, tuple) or not all(isinstance(t, str) for t in self.tokens):
            raise TypeError("tokens must be a tuple of strings")
        try:
            validate_iob2(self.anchor_tags, self.valid_types)
        except ValueError as exc:
            raise ValueError(f"Invalid anchor tags: {exc}") from exc
        # Dirty tags are an observed noisy source and may contain an intentional
        # malformed IOB transition (for example O->I after internal
        # fragmentation).  Their alphabet and length are checked here; strict
        # legality is enforced only for the anchor and model outputs.
        _validate_tag_alphabet(self.dirty_tags, self.valid_types)
        if self.reviewer_weights and len(self.reviewer_weights) != len(self.candidate_paths):
            # Missing weights are intentionally represented by an empty tuple;
            # an explicitly mismatched list is a data-quality error.
            raise ValueError("reviewer_weights must match candidate_paths or be empty")
        for path in self.candidate_paths:
            if len(path) != len(self.tokens):
                raise ValueError("candidate path length differs from tokens")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ContextualLatticeInput":
        unknown = set(payload) - MODEL_INPUT_FIELDS
        forbidden = unknown & FORBIDDEN_METADATA_FIELDS
        if unknown:
            label = "forbidden" if forbidden else "unknown"
            raise ValueError(f"{label} model-facing fields: {sorted(unknown)}")
        missing = MODEL_INPUT_FIELDS - set(payload)
        if missing:
            raise ValueError(f"Missing model-facing fields: {sorted(missing)}")
        return cls(
            tokens=tuple(payload["tokens"]),
            dirty_tags=tuple(payload["dirty_tags"]),
            anchor_tags=tuple(payload["anchor_tags"]),
            candidate_paths=tuple(tuple(path) for path in payload["candidate_paths"]),
            reviewer_weights=tuple(float(x) for x in payload["reviewer_weights"]),
            valid_types=frozenset(payload["valid_types"]),
            deer_stats=payload["deer_stats"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "dirty_tags": list(self.dirty_tags),
            "anchor_tags": list(self.anchor_tags),
            "candidate_paths": [list(path) for path in self.candidate_paths],
            "reviewer_weights": list(self.reviewer_weights),
            "valid_types": sorted(self.valid_types),
            "deer_stats": self.deer_stats,
        }


@dataclass(frozen=True)
class CandidateSpan:
    span: Span
    score: float = 0.0
    exact_support: int = 0
    overlap_support: int = 0
    path_support: int = 0
    weight_mass: float = 0.0
    dirty_presence: bool = False
    anchor_presence: bool = False
    missing_weight: bool = False
    scalar_features: tuple[float, ...] = field(default_factory=tuple, compare=False)
    contextual_features: Any = field(default=None, compare=False)


def _overlap(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def build_lattice(value: ContextualLatticeInput) -> tuple[CandidateSpan, ...]:
    """Build and deduplicate the typed-span lattice from observable sources."""

    sources: list[tuple[str, tuple[str, ...], float]] = [("dirty", value.dirty_tags, 1.0)]
    sources.append(("anchor", value.anchor_tags, 1.0))
    weights_missing = not value.reviewer_weights and bool(value.candidate_paths)
    if weights_missing:
        weights = [1.0] * len(value.candidate_paths)
    else:
        weights = list(value.reviewer_weights)
    for i, path in enumerate(value.candidate_paths):
        try:
            validate_iob2(path, value.valid_types)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Excluding invalid candidate path {i}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        weight = float(weights[i]) if i < len(weights) else 1.0
        if not math.isfinite(weight) or weight < 0:
            warnings.warn(f"Excluding candidate path {i} with invalid reviewer weight", RuntimeWarning, stacklevel=2)
            continue
        sources.append((f"path:{i}", tuple(path), weight))

    by_span: dict[Span, dict[str, Any]] = {}
    for name, tags, weight in sources:
        try:
            source_spans = iob2_to_spans(tags)
        except ValueError as exc:
            if name == "dirty":
                warnings.warn(f"Dirty source is malformed; extracting permissive spans: {exc}", RuntimeWarning, stacklevel=2)
                source_spans = iob2_to_spans(tags, strict=False)
            else:
                # The anchor was validated at input construction.  This branch
                # is defensive for future source additions.
                warnings.warn(f"Excluding invalid source {name}: {exc}", RuntimeWarning, stacklevel=2)
                continue
        for span in source_spans:
            if span.type not in value.valid_types:
                continue
            entry = by_span.setdefault(
                span,
                {
                    "exact_support": 0,
                    "overlap_support": 0,
                    "path_support": 0,
                    "weight_mass": 0.0,
                    "dirty_presence": False,
                    "anchor_presence": False,
                    "missing_weight": weights_missing,
                },
            )
            entry["exact_support"] += 1
            if name == "dirty":
                entry["dirty_presence"] = True
            elif name == "anchor":
                entry["anchor_presence"] = True
            else:
                entry["path_support"] += 1
                entry["weight_mass"] += weight

    spans = sorted(by_span)
    # Exact support is source presence; overlap support captures complementary
    # boundary/merge evidence without introducing unsupported heuristic spans.
    for span in spans:
        entry = by_span[span]
        entry["overlap_support"] = sum(
            1 for other in spans if other != span and _overlap(span, other)
        )
    return tuple(CandidateSpan(span=span, **by_span[span]) for span in spans)


def _predecessor_indices(spans: Sequence[Span]) -> list[int]:
    ordered = sorted(spans, key=lambda s: (s.end, s.start, s.type))
    return [sum(1 for other in ordered[:i] if other.end <= span.start) for i, span in enumerate(ordered)]


@dataclass(frozen=True)
class LatticeDecode:
    spans: tuple[Span, ...]
    score: float


def decode_exact_lattice(candidates: Sequence[CandidateSpan]) -> LatticeDecode:
    ordered = sorted(candidates, key=lambda c: (c.span.end, c.span.start, c.span.type))
    if not ordered:
        return LatticeDecode((), 0.0)
    prev = _predecessor_indices([c.span for c in ordered])
    dp = [0.0] * (len(ordered) + 1)
    choices: list[tuple[Span, ...]] = [()] * (len(ordered) + 1)
    for i, (candidate, p) in enumerate(zip(ordered, prev), start=1):
        skip_score, skip_spans = dp[i - 1], choices[i - 1]
        take_score = float(candidate.score) + dp[p]
        take_spans = choices[p] + (candidate.span,)
        if take_score > skip_score + 1e-12:
            dp[i], choices[i] = take_score, take_spans
        else:
            dp[i], choices[i] = skip_score, skip_spans
    return LatticeDecode(tuple(sorted(choices[-1], key=lambda s: (s.start, s.end, s.type))), float(dp[-1]))


def lattice_log_partition(candidates: Sequence[CandidateSpan]) -> float:
    ordered = sorted(candidates, key=lambda c: (c.span.end, c.span.start, c.span.type))
    if not ordered:
        return 0.0
    prev = _predecessor_indices([c.span for c in ordered])
    logz = [0.0] * (len(ordered) + 1)
    for i, (candidate, p) in enumerate(zip(ordered, prev), start=1):
        logz[i] = float(np.logaddexp(logz[i - 1], float(candidate.score) + logz[p]))
    return logz[-1]


def _torch_log_partition(scores: torch.Tensor, spans: Sequence[Span]) -> torch.Tensor:
    ordered_idx = sorted(range(len(spans)), key=lambda i: (spans[i].end, spans[i].start, spans[i].type))
    ordered_spans = [spans[i] for i in ordered_idx]
    ordered_scores = scores[torch.as_tensor(ordered_idx, device=scores.device)]
    prev = _predecessor_indices(ordered_spans)
    logz: list[torch.Tensor] = [scores.new_zeros(())]
    for i, p in enumerate(prev, start=1):
        logz.append(torch.logaddexp(logz[i - 1], ordered_scores[i - 1] + logz[p]))
    return logz[-1]


def _stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_prediction_hash(predictions: Sequence[Sequence[str]]) -> str:
    payload = json.dumps([list(row) for row in predictions], ensure_ascii=False, separators=(",", ":"))
    return _stable_hash_bytes(payload.encode("utf-8"))


def refuse_overwrite(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {target}")


def _unicode_shape(tokens: Sequence[str]) -> tuple[float, ...]:
    chars = "".join(tokens)
    if not chars:
        return (0.0,) * 6
    n = len(chars)
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in chars) / n
    latin = sum(c.isascii() and c.isalpha() for c in chars) / n
    digit = sum(c.isdigit() for c in chars) / n
    punct = sum(c in string.punctuation for c in chars) / n
    upper = sum(c.isupper() for c in chars) / n
    alpha_num = sum(c.isalnum() for c in chars) / n
    return (cjk, latin, digit, punct, upper, alpha_num)


def _deer_value(stats: Any, *names: str) -> float:
    if stats is None:
        return 0.0
    for name in names:
        if isinstance(stats, Mapping) and name in stats:
            value = stats[name]
        elif hasattr(stats, name):
            value = getattr(stats, name)
        else:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _candidate_scalars(candidate: CandidateSpan, value: ContextualLatticeInput) -> tuple[float, ...]:
    n = max(len(value.tokens), 1)
    span = candidate.span
    span_tokens = value.tokens[span.start : span.end]
    shape = _unicode_shape(span_tokens)
    sent_shape = _unicode_shape(value.tokens)
    source_count = max(len(value.candidate_paths) + 2, 1)
    density = sum(1 for tag in value.dirty_tags if tag != "O") / n
    oov = _deer_value(value.deer_stats, "oov", "oov_fraction", "oov_rate")
    return (
        candidate.exact_support / source_count,
        candidate.overlap_support / source_count,
        candidate.path_support / max(len(value.candidate_paths), 1),
        candidate.weight_mass / max(sum(value.reviewer_weights) if value.reviewer_weights else 1.0, 1e-6),
        float(candidate.dirty_presence),
        float(candidate.anchor_presence),
        float(candidate.missing_weight),
        span.start / n,
        span.end / n,
        (span.end - span.start) / n,
        float(span.start == 0),
        float(span.end == n),
        density,
        _deer_value(value.deer_stats, "entity", "entity_posterior", "entity_mean"),
        _deer_value(value.deer_stats, "type", "type_posterior", "type_mean"),
        _deer_value(value.deer_stats, "semantic", "semantic_posterior", "semantic_mean"),
        oov,
        sum(shape) / len(shape),
        *shape,
        *sent_shape,
        len(span_tokens) / n,
        float(any(tok in string.punctuation for tok in span_tokens)),
        float(any(any(c.isdigit() for c in tok) for tok in span_tokens)),
    )


class ContextEncoder(Protocol):
    hidden_size: int
    checkpoint_hash: str

    def encode(self, tokens: Sequence[str]) -> np.ndarray: ...


class GLiNERContextEncoder:
    """Offline adapter exposing frozen first-subtoken mDeBERTa embeddings."""

    def __init__(
        self,
        checkpoint: str | Path,
        valid_types: Iterable[str],
        *,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # The bundled GLiNER snapshot intentionally omits the base tokenizer;
        # its sibling ``hf_home`` contains the offline mDeBERTa SentencePiece
        # cache.  Discover it without allowing any network fallback.
        for parent in self._checkpoint_parents(checkpoint):
            if parent.name == "hf_home":
                os.environ.setdefault("HF_HOME", str(parent))
                os.environ.setdefault("TRANSFORMERS_CACHE", str(parent / "hub"))
                break
        try:
            from gliner import GLiNER
            import gliner.model as gliner_model
            from tokenizers import Tokenizer
            from transformers import PreTrainedTokenizerFast
            from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Set PYTHONPATH to the cached GLiNER dependency before loading the encoder") from exc
        self.checkpoint = Path(checkpoint)
        self.valid_types = tuple(sorted(valid_types))
        self.device = torch.device(device)
        model_file = self.checkpoint / "model.safetensors"
        self.checkpoint_hash = _file_hash(model_file) if model_file.exists() else _stable_hash_bytes(str(self.checkpoint).encode())
        tokenizer = self._load_fast_tokenizer(Tokenizer, PreTrainedTokenizerFast)
        # The bundled runtime ships a Python-3.9 SentencePiece extension while
        # the mandated interpreter is Python 3.11; importing that extension
        # causes a native access violation.  The cached fast tokenizer is
        # equivalent for the whitespace-tokenized input and avoids that DLL.
        original_loader = gliner_model.BaseGLiNER.__dict__["_load_tokenizer"]
        original_config_loader = gliner_model.BaseGLiNER.__dict__["_load_config"]

        @classmethod
        def _offline_loader(cls, config, model_dir, cache_dir=None, local_files_only=False):
            return tokenizer

        @classmethod
        def _offline_config_loader(cls, config_file, **overrides):
            config = original_config_loader.__func__(cls, config_file, **overrides)
            # The GLiNER checkpoint contains the complete DeBERTa weights, but
            # its compact config omits ``encoder_config``.  Supplying the
            # cached backbone config prevents Transformer from attempting a
            # network AutoConfig lookup while keeping all weights frozen.
            config.encoder_config = DebertaV2Config.from_pretrained(
                str(self._backbone_source), local_files_only=True
            )
            return config

        gliner_model.BaseGLiNER._load_tokenizer = _offline_loader
        gliner_model.BaseGLiNER._load_config = _offline_config_loader
        try:
            self.model = GLiNER.from_pretrained(
                str(self.checkpoint),
                local_files_only=True,
                map_location=str(self.device),
                low_cpu_mem_usage=True,
                load_tokenizer=True,
            )
        finally:
            gliner_model.BaseGLiNER._load_tokenizer = original_loader
            gliner_model.BaseGLiNER._load_config = original_config_loader
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 512))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, np.ndarray] = {}

    @staticmethod
    def _checkpoint_parents(checkpoint: str | Path) -> Iterable[Path]:
        path = Path(checkpoint).resolve()
        yield from path.parents

    def _load_fast_tokenizer(self, tokenizer_cls: Any, fast_tokenizer_cls: Any) -> Any:
        hf_home = os.environ.get("HF_HOME")
        candidates: list[Path] = []
        if hf_home:
            candidates.extend(
                Path(hf_home).glob("hub/models--microsoft--mdeberta-v3-base/snapshots/*")
            )
        if not candidates:
            raise FileNotFoundError("Offline mDeBERTa tokenizer cache was not found")
        source = sorted(candidates)[-1]
        self._backbone_source = source
        tokenizer_file = source / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Cached fast tokenizer is missing: {tokenizer_file}")
        tokenizer = tokenizer_cls.from_file(str(tokenizer_file))
        return fast_tokenizer_cls(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )

    def _key(self, tokens: Sequence[str]) -> str:
        canonical = json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))
        return _stable_hash_bytes((self.checkpoint_hash + "\0" + canonical).encode("utf-8"))

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        key = self._key(tokens)
        if key in self._memory:
            return self._memory[key].copy()
        disk_path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if disk_path and disk_path.exists():
            array = np.load(disk_path, allow_pickle=False).astype(np.float32, copy=False)
            self._memory[key] = array
            return array.copy()

        n = len(tokens)
        if n == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        output = np.zeros((n, self.hidden_size), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        starts = list(range(0, n, WINDOW_STRIDE))
        if starts and starts[-1] + WINDOW_SIZE < n:
            starts.append(n - WINDOW_SIZE)
        for start in starts:
            end = min(start + WINDOW_SIZE, n)
            # Word-piece expansion can exceed the backbone's 384 subtoken
            # window.  Shrink only this window until every original word is
            # represented; the global 320/256 policy remains fixed.
            while True:
                window = tuple(tokens[start:end])
                local, covered = self._encode_word_window(window)
                if covered == len(window):
                    break
                if covered <= 0:
                    raise ValueError("Frozen encoder produced no aligned word subtokens")
                end = start + covered
            local = local[: end - start]
            output[start : start + len(window)] += local
            counts[start : start + len(window)] += 1.0
        array = (output / np.maximum(counts[:, None], 1.0)).astype(np.float32)
        self._memory[key] = array
        if disk_path:
            np.save(disk_path, array, allow_pickle=False)
        return array.copy()

    def _encode_word_window(self, tokens: Sequence[str]) -> tuple[np.ndarray, int]:
        tokenizer = self.model.data_processor.transformer_tokenizer
        encoded = tokenizer(
            list(tokens),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            add_special_tokens=True,
        )
        word_ids = encoded.word_ids(batch_index=0)
        covered = 0
        for word_id in word_ids:
            if word_id is not None:
                covered = max(covered, int(word_id) + 1)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            token_embeddings = self.model.model.token_rep_layer(input_ids, attention_mask)
        if isinstance(token_embeddings, (tuple, list)):
            token_embeddings = token_embeddings[0]
        token_embeddings = token_embeddings[0].detach().float().cpu().numpy()
        pooled = np.zeros((covered, self.hidden_size), dtype=np.float32)
        seen = np.zeros(covered, dtype=bool)
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id >= covered or seen[word_id]:
                continue
            pooled[word_id] = token_embeddings[token_index]
            seen[word_id] = True
        return pooled, covered


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SpanScorer(nn.Module):
    def __init__(self, hidden_size: int, scalar_size: int, type_count: int, seed: int = SEED) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.context_projection = nn.Linear(hidden_size * 6, CONTEXTUAL_DIM)
        self.scalar_projection = nn.Linear(scalar_size, SCALAR_DIM)
        self.type_embedding = nn.Embedding(type_count, TYPE_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(CONTEXTUAL_DIM + SCALAR_DIM + TYPE_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, contextual: torch.Tensor, scalars: torch.Tensor, type_indices: torch.Tensor) -> torch.Tensor:
        contextual = self.context_projection(contextual)
        scalars = self.scalar_projection(scalars)
        types = self.type_embedding(type_indices)
        return self.mlp(torch.cat((contextual, scalars, types), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class ContextualDecodeResult:
    tags: tuple[str, ...]
    raw_lattice_tags: tuple[str, ...]
    used_anchor: bool
    predicted_gain: float
    selected_spans: tuple[Span, ...]
    model_hash: str


@dataclass
class _FeatureBundle:
    value: ContextualLatticeInput
    candidates: tuple[CandidateSpan, ...]
    contextual: np.ndarray
    scalars: np.ndarray
    type_indices: np.ndarray


def _span_context(embeddings: np.ndarray, span: Span) -> np.ndarray:
    n, hidden = embeddings.shape
    if n == 0:
        return np.zeros(hidden * 6, dtype=np.float32)
    start = embeddings[min(span.start, n - 1)]
    end = embeddings[min(span.end - 1, n - 1)]
    middle = embeddings[span.start : min(span.end, n)]
    mean = middle.mean(axis=0) if len(middle) else start
    maximum = middle.max(axis=0) if len(middle) else start
    left = embeddings[max(0, span.start - 2) : span.start].mean(axis=0) if span.start else start
    right = embeddings[span.end : min(n, span.end + 2)].mean(axis=0) if span.end < n else end
    return np.concatenate((start, end, mean, maximum, left, right)).astype(np.float32)


def _feature_bundle(
    value: ContextualLatticeInput,
    encoder: ContextEncoder,
    type_vocabulary: Sequence[str] | None = None,
) -> _FeatureBundle:
    candidates = build_lattice(value)
    embeddings = np.asarray(encoder.encode(value.tokens), dtype=np.float32)
    if embeddings.shape != (len(value.tokens), int(encoder.hidden_size)):
        raise ValueError(f"Encoder returned {embeddings.shape}, expected {(len(value.tokens), int(encoder.hidden_size))}")
    vocabulary = tuple(type_vocabulary or sorted(value.valid_types))
    types = {typ: i for i, typ in enumerate(vocabulary)}
    contextual = np.asarray([_span_context(embeddings, c.span) for c in candidates], dtype=np.float32)
    scalars = np.asarray([_candidate_scalars(c, value) for c in candidates], dtype=np.float32)
    type_indices = np.asarray([types[c.span.type] for c in candidates], dtype=np.int64)
    if not candidates:
        scalar_size = len(_candidate_scalars(CandidateSpan(Span(0, 1, next(iter(value.valid_types)))), value)) if value.tokens else 32
        contextual = np.zeros((0, int(encoder.hidden_size) * 6), dtype=np.float32)
        scalars = np.zeros((0, scalar_size), dtype=np.float32)
        type_indices = np.zeros((0,), dtype=np.int64)
    return _FeatureBundle(value, candidates, contextual, scalars, type_indices)


class ContextualLatticeDecoder:
    """Train and decode the complete candidate interval lattice."""

    def __init__(
        self,
        valid_types: Iterable[str],
        *,
        encoder: ContextEncoder,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
        seed: int = SEED,
    ) -> None:
        self.valid_types = frozenset(valid_types)
        self.encoder = encoder
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device)
        self.seed = int(seed)
        self._model: _SpanScorer | None = None
        self.type_vocabulary = tuple(sorted(self.valid_types))
        self._feature_cache: dict[str, _FeatureBundle] = {}
        self._model_hash = "unfitted"
        self.unreachable_gold: list[tuple[Span, ...]] = []
        self.gate: ResidualGate | None = None
        self.margin: float = 0.0

    def _bundle(self, value: ContextualLatticeInput) -> _FeatureBundle:
        key = _stable_hash_bytes(
            json.dumps(value.to_payload(), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        )
        if key not in self._feature_cache:
            self._feature_cache[key] = _feature_bundle(value, self.encoder, self.type_vocabulary)
        return self._feature_cache[key]

    def _ensure_model(self, bundle: _FeatureBundle) -> _SpanScorer:
        if self._model is None:
            scalar_size = bundle.scalars.shape[1] if bundle.scalars.ndim == 2 else 0
            self._model = _SpanScorer(
                int(self.encoder.hidden_size), scalar_size, len(self.valid_types), seed=self.seed
            ).to(self.device)
        return self._model

    def fit(self, train: Sequence[tuple[Any, ...]]) -> None:
        if not train:
            raise ValueError("Cannot fit contextual lattice decoder on an empty sequence")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        bundles: list[tuple[_FeatureBundle, tuple[Span, ...], float]] = []
        self.unreachable_gold = []
        for record in train:
            if len(record) == 2:
                value, gold_tags = record
                weight = 1.0
            elif len(record) == 3:
                value, gold_tags, weight = record
            else:
                raise ValueError("Training records must be (value, gold_tags[, weight])")
            if not isinstance(value, ContextualLatticeInput):
                raise TypeError("Training record has a non-model input")
            gold = iob2_to_spans(tuple(gold_tags), strict=True)
            bundle = self._bundle(value)
            available = {candidate.span for candidate in bundle.candidates}
            missing = tuple(span for span in gold if span not in available)
            self.unreachable_gold.append(missing)
            bundles.append((bundle, tuple(span for span in gold if span in available), float(weight)))
        model = self._ensure_model(bundles[0][0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        model.train()
        for _epoch in range(self.epochs):
            order = np.arange(len(bundles))
            rng = np.random.default_rng(self.seed + _epoch)
            rng.shuffle(order)
            for begin in range(0, len(order), max(self.batch_size, 1)):
                optimizer.zero_grad(set_to_none=True)
                batch_items: list[tuple[_FeatureBundle, tuple[Span, ...], float]] = []
                for index in order[begin : begin + max(self.batch_size, 1)]:
                    bundle, gold, weight = bundles[int(index)]
                    if not bundle.candidates or weight <= 0:
                        continue
                    batch_items.append((bundle, gold, weight))
                if not batch_items:
                    continue
                # One scorer call per sentence batch; the DP partition remains
                # sentence-local because candidate intervals differ by row.
                contextual = torch.as_tensor(
                    np.concatenate([item[0].contextual for item in batch_items], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                scalars = torch.as_tensor(
                    np.concatenate([item[0].scalars for item in batch_items], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                types = torch.as_tensor(
                    np.concatenate([item[0].type_indices for item in batch_items], axis=0),
                    dtype=torch.long,
                    device=self.device,
                )
                packed_scores = model(contextual, scalars, types)
                batch_count = len(batch_items)
                max_candidates = max(len(item[0].candidates) for item in batch_items)
                padded_scores = packed_scores.new_zeros((batch_count, max_candidates))
                valid_mask = torch.zeros((batch_count, max_candidates), dtype=torch.bool, device=self.device)
                predecessor = torch.zeros((batch_count, max_candidates), dtype=torch.long, device=self.device)
                target_mask = torch.zeros((batch_count, max_candidates), dtype=torch.float32, device=self.device)
                offset = 0
                for batch_index, (bundle, gold, _weight) in enumerate(batch_items):
                    count = len(bundle.candidates)
                    scores = packed_scores[offset : offset + count]
                    offset += count
                    order_by_end = sorted(
                        range(count),
                        key=lambda i: (bundle.candidates[i].span.end, bundle.candidates[i].span.start, bundle.candidates[i].span.type),
                    )
                    ordered_spans = [bundle.candidates[i].span for i in order_by_end]
                    reordered = scores[torch.as_tensor(order_by_end, dtype=torch.long, device=self.device)]
                    padded_scores[batch_index, :count] = reordered
                    valid_mask[batch_index, :count] = True
                    predecessor[batch_index, :count] = torch.as_tensor(
                        _predecessor_indices(ordered_spans), dtype=torch.long, device=self.device
                    )
                    target_positions = {span: position for position, span in enumerate(ordered_spans)}
                    for span in gold:
                        if span in target_positions:
                            target_mask[batch_index, target_positions[span]] = 1.0
                logz_columns: list[torch.Tensor] = [padded_scores.new_zeros((batch_count,))]
                for position in range(max_candidates):
                    previous = predecessor[:, position]
                    history = torch.stack(logz_columns, dim=1)
                    take = padded_scores[:, position] + history.gather(1, previous[:, None]).squeeze(1)
                    update = torch.logaddexp(logz_columns[-1], take)
                    logz_columns.append(torch.where(valid_mask[:, position], update, logz_columns[-1]))
                logz = torch.stack(logz_columns, dim=1)
                gold_score = (padded_scores * target_mask).sum(dim=1)
                weights = torch.as_tensor([item[2] for item in batch_items], dtype=torch.float32, device=self.device)
                loss = ((logz[:, -1] - gold_score) * weights).sum() / max(sum(x[2] for x in bundles), 1e-6)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        self._model_hash = self._hash_model()

    def _hash_model(self) -> str:
        if self._model is None:
            return "unfitted"
        buffer = io.BytesIO()
        torch.save(self._model.state_dict(), buffer)
        return _stable_hash_bytes(buffer.getvalue())

    @property
    def model_hash(self) -> str:
        return self._model_hash

    def score_candidates(self, value: ContextualLatticeInput) -> tuple[_FeatureBundle, np.ndarray]:
        if self._model is None:
            raise RuntimeError("fit must be called before scoring")
        bundle = self._bundle(value)
        if not bundle.candidates:
            return bundle, np.zeros((0,), dtype=np.float32)
        with torch.inference_mode():
            scores = self._model(
                torch.as_tensor(bundle.contextual, dtype=torch.float32, device=self.device),
                torch.as_tensor(bundle.scalars, dtype=torch.float32, device=self.device),
                torch.as_tensor(bundle.type_indices, dtype=torch.long, device=self.device),
            )
        return bundle, scores.detach().float().cpu().numpy()

    def decode(self, value: ContextualLatticeInput, *, margin: float | None = None) -> ContextualDecodeResult:
        bundle, scores = self.score_candidates(value)
        scored = tuple(
            CandidateSpan(
                span=candidate.span,
                score=float(score),
                exact_support=candidate.exact_support,
                overlap_support=candidate.overlap_support,
                path_support=candidate.path_support,
                weight_mass=candidate.weight_mass,
                dirty_presence=candidate.dirty_presence,
                anchor_presence=candidate.anchor_presence,
                missing_weight=candidate.missing_weight,
            )
            for candidate, score in zip(bundle.candidates, scores)
        )
        lattice = decode_exact_lattice(scored)
        raw_tags = spans_to_iob2(lattice.spans, len(value.tokens), value.valid_types)
        anchor_spans = iob2_to_spans(value.anchor_tags)
        anchor_by_span = {candidate.span: candidate for candidate in scored}
        anchor_score = sum(anchor_by_span[span].score for span in anchor_spans if span in anchor_by_span)
        score_gap = float(lattice.score - anchor_score)
        gate_features = self.gate_features(bundle, scores, lattice.spans, anchor_spans)
        gate_prediction = float(self.gate.predict(gate_features[None, :])[0]) if self.gate else score_gap
        threshold = self.margin if margin is None else float(margin)
        use_anchor = gate_prediction <= threshold
        tags = value.anchor_tags if use_anchor else raw_tags
        return ContextualDecodeResult(
            tags=tuple(tags),
            raw_lattice_tags=tuple(raw_tags),
            used_anchor=use_anchor,
            predicted_gain=gate_prediction,
            selected_spans=lattice.spans,
            model_hash=self.model_hash,
        )

    @staticmethod
    def gate_features(
        bundle: _FeatureBundle,
        scores: np.ndarray,
        lattice_spans: Sequence[Span],
        anchor_spans: Sequence[Span],
    ) -> np.ndarray:
        selected = [c for c in bundle.candidates if c.span in set(lattice_spans)]
        selected_scores = np.asarray([scores[i] for i, c in enumerate(bundle.candidates) if c.span in set(lattice_spans)])
        supports = np.asarray([c.exact_support for c in selected], dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-selected_scores)) if len(selected_scores) else np.zeros(1)
        anchor_set = set(anchor_spans)
        changed = sum(c.span not in anchor_set for c in selected) + sum(span not in {c.span for c in selected} for span in anchor_spans)
        score_gap = float(selected_scores.sum() - sum(scores[i] for i, c in enumerate(bundle.candidates) if c.span in anchor_set))
        n = max(len(bundle.value.tokens), 1)
        return np.asarray(
            [
                score_gap,
                float(len(selected)),
                float(changed),
                float(supports.max() if len(supports) else 0.0),
                float(supports.mean() if len(supports) else 0.0),
                float(probabilities.max()),
                float(probabilities.mean()),
                float(len(bundle.value.tokens)),
                sum(tag != "O" for tag in bundle.value.dirty_tags) / n,
                len(lattice_spans) / n,
                len(anchor_spans) / n,
                float(sum(c.anchor_presence for c in selected)),
                float(sum(c.dirty_presence for c in selected)),
            ],
            dtype=np.float32,
        )


class ResidualGate:
    """Cross-fitted residual predictor and one global anchor margin."""

    def __init__(self, *, seed: int = SEED) -> None:
        self.seed = seed
        self.model: HistGradientBoostingRegressor | None = None
        self.margin: float = 0.0

    def fit(self, features: np.ndarray, gains: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        gains = np.asarray(gains, dtype=np.float32)
        if features.ndim != 2 or len(features) != len(gains):
            raise ValueError("Residual features and gains have incompatible shapes")
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.seed,
        )
        self.model.fit(features, gains)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Residual gate is not fitted")
        return np.asarray(self.model.predict(np.asarray(features, dtype=np.float32)), dtype=np.float32)

    def choose_margin(
        self,
        features: np.ndarray,
        anchor_tags: Sequence[Sequence[str]],
        lattice_tags: Sequence[Sequence[str]],
        gold_tags: Sequence[Sequence[str]],
        *,
        retention_null: Sequence[bool] | None = None,
        margins: Sequence[float] = (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75, 1.00),
    ) -> float:
        predictions = self.predict(features)
        if retention_null is None:
            retention_null = [tuple(anchor) == tuple(gold) for anchor, gold in zip(anchor_tags, gold_tags)]
        best: tuple[float, float] | None = None
        for margin in sorted(float(x) for x in margins):
            selected = predictions > margin
            if selected.mean() < 0.05:
                continue
            combined = [lattice_tags[i] if selected[i] else anchor_tags[i] for i in range(len(predictions))]
            unsafe = False
            for i, (anchor, candidate, gold) in enumerate(zip(anchor_tags, combined, gold_tags)):
                if retention_null[i] and _sentence_f1(candidate, gold) < _sentence_f1(anchor, gold) - 1e-12:
                    unsafe = True
                    break
            if unsafe:
                continue
            score = _micro_f1(gold_tags, combined)
            if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and margin > best[1]):
                best = (score, margin)
        if best is None:
            raise RuntimeError("No residual margin satisfies alternative-coverage and retention constraints")
        self.margin = best[1]
        return self.margin


def _span_counts(gold: Sequence[str], pred: Sequence[str]) -> tuple[int, int, int]:
    gs, ps = set(iob2_to_spans(gold)), set(iob2_to_spans(pred))
    tp = len(gs & ps)
    return tp, len(ps) - tp, len(gs) - tp


def _sentence_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    tp, fp, fn = _span_counts(gold, pred)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _micro_f1(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> float:
    counts = np.asarray([_span_counts(g, p) for g, p in zip(gold, pred)], dtype=np.int64).sum(axis=0)
    tp, fp, fn = counts
    return float(2 * tp / max(2 * tp + fp + fn, 1))


@dataclass(frozen=True)
class HistoricalSplit:
    train: tuple[Mapping[str, Any], ...]
    calibration: tuple[Mapping[str, Any], ...]
    test: tuple[Mapping[str, Any], ...]
    test_groups: frozenset[tuple[str, str]]
    calibration_groups: frozenset[tuple[str, str]]


def _canonical_tokens(tokens: Sequence[str]) -> str:
    return json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))


def _group_digest(salt: str, dataset: str, canonical: str) -> str:
    return hashlib.sha256(f"{salt}|{dataset}|{canonical}".encode("utf-8")).hexdigest()


def build_historical_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups_per_dataset: int = 30,
    salt: str = "selectdenoise-contextual-lattice-v1-20260820",
    seeds: Sequence[int] = (13, 42, 2024),
) -> HistoricalSplit:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if "dataset" not in row or "tokens" not in row:
            raise ValueError("Historical rows require audit-only dataset and tokens fields")
        key = (str(row["dataset"]), _canonical_tokens(row["tokens"]))
        grouped[key].append(row)
    # A few historical datasets repeat an identical token sequence at multiple
    # row indices.  The preregistered canonical-token grouping treats these as
    # one sentence group; retain one deterministic seed/family representative
    # and fail if the supposedly identical records disagree on gold tags.
    for key, records in list(grouped.items()):
        representatives: dict[tuple[int, str], Mapping[str, Any]] = {}
        for row in sorted(records, key=lambda item: int(item.get("row_index", 0))):
            sf = (int(row.get("seed", 0)), str(row.get("family", "")))
            previous = representatives.get(sf)
            if previous is not None and tuple(previous.get("gold_tags", ())) != tuple(row.get("gold_tags", ())):
                raise ValueError(f"Conflicting gold tags inside canonical group {key} seed/family {sf}")
            representatives.setdefault(sf, row)
        grouped[key] = list(representatives.values())
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in grouped:
        by_dataset[key[0]].append(key)
    test_groups: set[tuple[str, str]] = set()
    calibration_groups: set[tuple[str, str]] = set()
    for dataset in sorted(by_dataset):
        ordered = sorted(by_dataset[dataset], key=lambda key: (_group_digest(salt, *key), key[1]))
        if len(ordered) < 2 * groups_per_dataset:
            raise ValueError(f"Dataset {dataset} has only {len(ordered)} groups; need {2 * groups_per_dataset}")
        test_groups.update(ordered[:groups_per_dataset])
        calibration_groups.update(ordered[groups_per_dataset : 2 * groups_per_dataset])

    def choose_seed(key: tuple[str, str]) -> int:
        value = int(_group_digest(salt + "-replicate", *key)[:16], 16)
        return int(seeds[value % len(seeds)])

    train: list[Mapping[str, Any]] = []
    calibration: list[Mapping[str, Any]] = []
    test: list[Mapping[str, Any]] = []
    for key, group_rows in grouped.items():
        if key in test_groups:
            selected_seed = choose_seed(key)
            test.extend(row for row in group_rows if int(row.get("seed", selected_seed)) == selected_seed)
        elif key in calibration_groups:
            selected_seed = choose_seed(key)
            calibration.extend(row for row in group_rows if int(row.get("seed", selected_seed)) == selected_seed)
        else:
            train.extend(group_rows)
    expected_cells = len(by_dataset) * groups_per_dataset * 3
    if len(test) != expected_cells or len(calibration) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} rows in each held-out split (three family views per group), "
            f"got test={len(test)}, calibration={len(calibration)}"
        )
    if test_groups & calibration_groups:
        raise AssertionError("Group split overlap")
    return HistoricalSplit(tuple(train), tuple(calibration), tuple(test), frozenset(test_groups), frozenset(calibration_groups))


__all__ = [
    "CandidateSpan",
    "ContextualDecodeResult",
    "ContextualLatticeDecoder",
    "ContextualLatticeInput",
    "GLiNERContextEncoder",
    "HistoricalSplit",
    "ResidualGate",
    "Span",
    "build_historical_split",
    "build_lattice",
    "decode_exact_lattice",
    "iob2_to_spans",
    "lattice_log_partition",
    "make_prediction_hash",
    "refuse_overwrite",
    "spans_to_iob2",
    "validate_iob2",
]
