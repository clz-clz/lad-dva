from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from selectdenoise_contextual_lattice import (
    ContextualLatticeDecoder,
    ContextualLatticeInput,
    CandidateSpan,
    ResidualGate,
    Span,
    build_historical_split,
    build_lattice,
    decode_exact_lattice,
    iob2_to_spans,
    lattice_log_partition,
    make_prediction_hash,
    refuse_overwrite,
    spans_to_iob2,
)
from run_contextual_lattice_gate import evaluate


def _value(**kwargs):
    base = dict(
        tokens=("Alice", "visited", "Paris", "."),
        dirty_tags=("B-PER", "O", "B-LOC", "O"),
        anchor_tags=("B-PER", "O", "B-LOC", "O"),
        candidate_paths=(
            ("B-PER", "O", "B-LOC", "O"),
            ("B-PER", "O", "B-ORG", "O"),
        ),
        reviewer_weights=(0.8, 0.2),
        valid_types=frozenset({"PER", "LOC", "ORG"}),
        deer_stats={"entity": 0.5, "type": 0.4, "semantic": 0.6, "oov": 0.1},
    )
    base.update(kwargs)
    return ContextualLatticeInput(**base)


def test_model_schema_rejects_evaluation_metadata():
    payload = {
        "tokens": ["A"],
        "dirty_tags": ["O"],
        "anchor_tags": ["O"],
        "candidate_paths": [["O"]],
        "reviewer_weights": [1.0],
        "valid_types": ["PER"],
        "deer_stats": {},
        "__noise_type__": "BT",
    }
    with pytest.raises(ValueError, match="unknown|forbidden"):
        ContextualLatticeInput.from_payload(payload)


def test_evaluator_normalizes_jsonable_tag_lists_for_seqeval():
    rows = [
        {
            "dataset": "toy",
            "family": "BT",
            "tokens": ("Alice",),
            "gold_tags": ("B-PER",),
            "anchor_tags": ("O",),
            "dirty_tags": ("O",),
        }
    ]
    report = evaluate(rows, [{"tags": ["B-PER"], "used_anchor": False}])
    assert report["cells"]["toy__BT"]["delta_f1"] > 0


def test_lattice_is_permutation_invariant_and_deduplicates():
    value = _value()
    a = build_lattice(value)
    permuted = _value(
        candidate_paths=(value.candidate_paths[1], value.candidate_paths[0]),
        reviewer_weights=(value.reviewer_weights[1], value.reviewer_weights[0]),
    )
    b = build_lattice(permuted)
    assert [(x.span, x.exact_support, x.weight_mass) for x in a] == [
        (x.span, x.exact_support, x.weight_mass) for x in b
    ]
    assert len(a) == len({x.span for x in a})


def test_invalid_anchor_fails_but_invalid_candidate_path_is_excluded():
    with pytest.raises(ValueError, match="anchor"):
        build_lattice(_value(anchor_tags=("I-PER", "O", "B-LOC", "O")))
    lattice = build_lattice(
        _value(candidate_paths=(("I-PER", "O", "B-LOC", "O"), ("B-PER", "O", "B-ORG", "O")))
    )
    per = next(span for span in lattice if span.span.type == "PER" and span.span.start == 0)
    assert per.path_support == 1  # only the valid second path is counted
    assert any(span.span.type == "ORG" for span in lattice)


def test_malformed_dirty_source_is_observed_but_anchor_remains_strict():
    value = _value(dirty_tags=("O", "I-PER", "O", "O"))
    lattice = build_lattice(value)
    assert any(candidate.span.type == "PER" for candidate in lattice)


def test_exact_interval_dp_matches_bruteforce_and_partition():
    spans = [
        CandidateSpan(Span(0, 1, "PER"), score=1.1),
        CandidateSpan(Span(1, 2, "LOC"), score=0.7),
        CandidateSpan(Span(0, 2, "ORG"), score=1.4),
    ]
    result = decode_exact_lattice(spans)
    legal = []
    for mask in range(1 << len(spans)):
        chosen = [spans[i] for i in range(len(spans)) if mask & (1 << i)]
        if all(a.span.end <= b.span.start or b.span.end <= a.span.start for a, b in itertools.combinations(chosen, 2)):
            legal.append(chosen)
    best = max(sum(x.score for x in choice) for choice in legal)
    assert result.score == pytest.approx(best)
    assert result.spans == (Span(0, 1, "PER"), Span(1, 2, "LOC"))
    expected_z = sum(np.exp(sum(x.score for x in choice)) for choice in legal)
    assert lattice_log_partition(spans) == pytest.approx(np.log(expected_z), rel=1e-6)


def test_iob2_roundtrip_handles_adjacent_same_type_and_length():
    tags = ("B-PER", "I-PER", "B-PER", "O", "B-LOC")
    spans = iob2_to_spans(tags)
    assert spans == (Span(0, 2, "PER"), Span(2, 3, "PER"), Span(4, 5, "LOC"))
    assert spans_to_iob2(spans, len(tags), frozenset({"PER", "LOC"})) == tags


def test_grouped_split_has_exact_cell_counts_and_no_cross_seed_leakage():
    rows = []
    for dataset in ("a", "b"):
        for group in range(65):
            tokens = (dataset, str(group))
            for seed in (13, 42, 2024):
                for family in ("BT", "IF", "ATF"):
                    rows.append({"dataset": dataset, "seed": seed, "family": family, "tokens": tokens})
    split = build_historical_split(rows, groups_per_dataset=30)
    assert len(split.test) == 2 * 30 * 3
    assert len(split.calibration) == 2 * 30 * 3
    assert all(len({r["seed"] for r in split.test if r["tokens"] == tokens}) == 1 for tokens in {r["tokens"] for r in split.test})
    test_groups = {(r["dataset"], r["tokens"]) for r in split.test}
    calib_groups = {(r["dataset"], r["tokens"]) for r in split.calibration}
    assert not test_groups & calib_groups


def test_residual_gate_selects_margin_without_retention_regression():
    gate = ResidualGate()
    features = np.asarray([[0.9, 2.0], [0.1, 1.0], [0.8, 3.0], [0.0, 0.0]])
    gains = np.asarray([1.0, -0.1, 0.8, 0.0])
    anchor = [("B-PER", "O"), ("B-PER", "O"), ("B-PER", "O"), ("O", "O")]
    lattice = [("B-PER", "O"), ("B-PER", "O"), ("O", "O"), ("O", "O")]
    gold = [("B-PER", "O"), ("B-PER", "O"), ("O", "O"), ("O", "O")]
    gate.fit(features, gains)
    margin = gate.choose_margin(features, anchor, lattice, gold, margins=(-0.5, 0.0, 0.5))
    assert margin in (-0.5, 0.0, 0.5)
    assert gate.predict(features).shape == (4,)


class _FakeEncoder:
    checkpoint_hash = "fake-encoder-v1"
    hidden_size = 8

    def encode(self, tokens):
        return np.asarray([[float(i + 1)] * self.hidden_size for i in range(len(tokens))], dtype=np.float32)


def test_decoder_is_deterministic_and_falls_back_to_anchor():
    value = _value()
    decoder = ContextualLatticeDecoder(value.valid_types, encoder=_FakeEncoder(), epochs=1, device="cpu")
    decoder.fit([(value, value.anchor_tags)])
    first = decoder.decode(value, margin=10.0)
    second = decoder.decode(value, margin=10.0)
    assert first.tags == second.tags == value.anchor_tags
    assert len(first.tags) == len(value.tokens)
    assert first.model_hash == second.model_hash
    assert first.used_anchor is True


def test_prediction_hash_and_output_directory_are_safe(tmp_path: Path):
    predictions = [("O", "B-PER"), ("O", "O")]
    assert make_prediction_hash(predictions) == make_prediction_hash(predictions)
    target = tmp_path / "result"
    refuse_overwrite(target)
    target.mkdir()
    with pytest.raises(FileExistsError):
        refuse_overwrite(target)
