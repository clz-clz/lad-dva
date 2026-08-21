from __future__ import annotations

import json
import hashlib
import inspect
import os
import asyncio
from pathlib import Path

import numpy as np
import pytest

from selectdenoise_contextual_lattice import (
    ContextualLatticeDecoder,
    ContextualLatticeInput,
    ResidualGate,
    Span,
    build_lattice,
)
from contextual_lattice_runtime import ContextualLatticeTerminal, load_bundle, save_bundle
from run_contextual_lattice_gate import TrainingDeer
from run_multiseed import CONFIGURATIONS


class _FakeEncoder:
    checkpoint_hash = "fake-checkpoint-v1"
    hidden_size = 8

    def encode(self, tokens):
        return np.asarray([[float(i + 1)] * self.hidden_size for i in range(len(tokens))], dtype=np.float32)


def _value() -> ContextualLatticeInput:
    return ContextualLatticeInput(
        tokens=("Alice", "visited", "Paris"),
        dirty_tags=("B-PER", "O", "B-LOC"),
        anchor_tags=("B-PER", "O", "B-LOC"),
        candidate_paths=(("B-PER", "O", "B-ORG"), ("B-PER", "O", "B-LOC")),
        reviewer_weights=(0.7, 0.3),
        valid_types=frozenset({"PER", "LOC", "ORG", "MISC"}),
        deer_stats={"entity": 0.2, "type": 0.3, "semantic": 0.25, "oov": 0.0},
    )


def _fitted_components():
    value = _value()
    decoder = ContextualLatticeDecoder(value.valid_types, encoder=_FakeEncoder(), epochs=1, device="cpu")
    decoder.fit([(value, value.anchor_tags)])
    gate = ResidualGate()
    features = np.tile(np.arange(13, dtype=np.float32), (25, 1))
    gains = np.linspace(-0.2, 1.0, 25, dtype=np.float32)
    gate.fit(features, gains)
    gate.margin = -0.1
    return decoder, gate


def test_bundle_round_trip_preserves_decoder_and_prediction_hash(tmp_path: Path):
    decoder, gate = _fitted_components()
    bundle = save_bundle(
        decoder,
        gate,
        tmp_path / "bundle",
        {
            "schema_version": "contextual-lattice-v1",
            "checkpoint_hash": "fake-checkpoint-v1",
            "split_hash": "split-v1",
            "deer_profile": {"token_entity": {}, "token_type": {}, "vocabulary": []},
        },
    )
    loaded = load_bundle(bundle, _FakeEncoder(), expected_checkpoint_hash="fake-checkpoint-v1")
    first = loaded.decode(
        tokens=_value().tokens,
        dirty_tags=_value().dirty_tags,
        anchor_tags=_value().anchor_tags,
        candidate_paths=_value().candidate_paths,
        reviewer_weights=_value().reviewer_weights,
        valid_types=_value().valid_types,
        deer_stats=_value().deer_stats,
    )
    second = loaded.decode(
        tokens=_value().tokens,
        dirty_tags=_value().dirty_tags,
        anchor_tags=_value().anchor_tags,
        candidate_paths=_value().candidate_paths,
        reviewer_weights=_value().reviewer_weights,
        valid_types=_value().valid_types,
        deer_stats=_value().deer_stats,
    )
    assert first.tags == second.tags
    assert first.model_hash == second.model_hash == decoder.model_hash
    assert (bundle / "manifest.json").exists()
    assert (bundle / "decoder.pt").exists()
    assert (bundle / "gate.joblib").exists()


def test_bundle_rejects_checkpoint_mismatch(tmp_path: Path):
    decoder, gate = _fitted_components()
    bundle = save_bundle(
        decoder,
        gate,
        tmp_path / "bundle",
        {"schema_version": "contextual-lattice-v1", "checkpoint_hash": "expected"},
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_bundle(bundle, _FakeEncoder(), expected_checkpoint_hash="different")


def test_bundle_rejects_split_schema_and_margin_mismatches(tmp_path: Path):
    decoder, gate = _fitted_components()
    bundle = save_bundle(
        decoder,
        gate,
        tmp_path / "bundle",
        {
            "schema_version": "contextual-lattice-v1",
            "checkpoint_hash": "fake-checkpoint-v1",
            "split_hash": "split-v1",
        },
    )
    with pytest.raises(ValueError, match="schema"):
        load_bundle(
            bundle,
            _FakeEncoder(),
            expected_checkpoint_hash="fake-checkpoint-v1",
            expected_schema_version="contextual-lattice-v0",
        )
    with pytest.raises(ValueError, match="split"):
        load_bundle(
            bundle,
            _FakeEncoder(),
            expected_checkpoint_hash="fake-checkpoint-v1",
            expected_split_hash="other-split",
        )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["margin"] = 0.0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    hashes_path = bundle / "files.sha256.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="margin"):
        load_bundle(bundle, _FakeEncoder(), expected_checkpoint_hash="fake-checkpoint-v1")


def test_bundle_rejects_unexpected_metadata_even_when_hashes_are_updated(tmp_path: Path):
    decoder, gate = _fitted_components()
    bundle = save_bundle(
        decoder,
        gate,
        tmp_path / "bundle",
        {"schema_version": "contextual-lattice-v1", "checkpoint_hash": "fake-checkpoint-v1"},
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["__noise_type__"] = "BT"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    hashes_path = bundle / "files.sha256.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        load_bundle(bundle, _FakeEncoder(), expected_checkpoint_hash="fake-checkpoint-v1")


def test_trusted_bundle_hashes_are_checked_before_gate_deserialization(tmp_path: Path):
    decoder, gate = _fitted_components()
    bundle = save_bundle(
        decoder,
        gate,
        tmp_path / "bundle",
        {"schema_version": "contextual-lattice-v1", "checkpoint_hash": "fake-checkpoint-v1"},
    )
    manifest_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Trusted contextual lattice manifest"):
        load_bundle(
            bundle,
            _FakeEncoder(),
            expected_checkpoint_hash="fake-checkpoint-v1",
            expected_manifest_hash="not-the-manifest",
        )
    decoder_hash = hashlib.sha256((bundle / "decoder.pt").read_bytes()).hexdigest()
    gate_hash = hashlib.sha256((bundle / "gate.joblib").read_bytes()).hexdigest()
    model_hash = decoder.model_hash
    loaded = load_bundle(
        bundle,
        _FakeEncoder(),
        expected_checkpoint_hash="fake-checkpoint-v1",
        expected_manifest_hash=manifest_hash,
        expected_decoder_file_hash=decoder_hash,
        expected_gate_file_hash=gate_hash,
        expected_decoder_model_hash=model_hash,
    )
    assert loaded.model_hash == model_hash
    decoder_path = bundle / "decoder.pt"
    decoder_path.write_bytes(decoder_path.read_bytes() + b"tampered")
    hashes_path = bundle / "files.sha256.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["decoder.pt"] = hashlib.sha256(decoder_path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Trusted bundle hash mismatch for decoder.pt"):
        load_bundle(
            bundle,
            _FakeEncoder(),
            expected_checkpoint_hash="fake-checkpoint-v1",
            expected_manifest_hash=manifest_hash,
            expected_decoder_file_hash=decoder_hash,
            expected_gate_file_hash=gate_hash,
            expected_decoder_model_hash=model_hash,
        )


def test_terminal_fallback_is_legal_and_counts_failures():
    class _FailingDecoder:
        model_hash = "failing-model"

        def decode(self, value):
            raise RuntimeError("synthetic decoder failure")

    terminal = ContextualLatticeTerminal(_FailingDecoder(), deer_profile={})
    result = terminal.decode(
        tokens=("Alice",),
        dirty_tags=("O",),
        anchor_tags=("B-PER",),
        candidate_paths=(("B-PER",),),
        reviewer_weights=(1.0,),
        valid_types=frozenset({"PER"}),
        deer_stats={},
    )
    assert result.tags == ("B-PER",)
    assert result.used_anchor is True
    assert terminal.fallback_count == 1


def test_runtime_excludes_invalid_candidate_paths_but_hard_fails_invalid_anchor():
    class _CaptureDecoder:
        model_hash = "capture"

        def decode(self, value):
            assert tuple(candidate.span for candidate in build_lattice(value)) == (Span(0, 1, "PER"),)
            return type("Result", (), {
                "tags": value.anchor_tags,
                "raw_lattice_tags": value.anchor_tags,
                "used_anchor": True,
                "predicted_gain": 0.0,
                "selected_spans": (),
                "model_hash": self.model_hash,
            })()

    terminal = ContextualLatticeTerminal(_CaptureDecoder())
    result = terminal.decode(
        tokens=("Alice",),
        dirty_tags=("O",),
        anchor_tags=("B-PER",),
        candidate_paths=(("B-PER",), ("I-PER",), ("B-ORG", "O")),
        reviewer_weights=(0.7, 0.2, 0.1),
        valid_types=frozenset({"PER"}),
        deer_stats={},
    )
    assert result.used_anchor is True
    with pytest.raises(ValueError, match="Invalid anchor"):
        terminal.decode(
            tokens=("Alice",),
            dirty_tags=("O",),
            anchor_tags=("I-PER",),
            candidate_paths=(),
            reviewer_weights=(),
            valid_types=frozenset({"PER"}),
            deer_stats={},
        )


def test_bundle_directory_is_non_overwriting(tmp_path: Path):
    decoder, gate = _fitted_components()
    target = tmp_path / "bundle"
    save_bundle(decoder, gate, target, {"schema_version": "contextual-lattice-v1", "checkpoint_hash": "fake"})
    with pytest.raises(FileExistsError):
        save_bundle(decoder, gate, target, {"schema_version": "contextual-lattice-v1", "checkpoint_hash": "fake"})


def test_training_deer_profile_reproduces_runtime_sentence_statistics():
    deer = TrainingDeer(
        token_entity={"Alice": 0.8},
        token_type={"Alice": 0.5},
        vocabulary=frozenset({"Alice", "visited"}),
    )
    terminal = ContextualLatticeTerminal(_FailingDecoderForStats(), deer_profile=deer.to_profile())
    stats = terminal.sentence_deer_stats(("Alice", "unknown"))
    assert stats["entity"] == pytest.approx(0.4)
    assert stats["type"] == pytest.approx(0.25)
    assert stats["semantic"] == pytest.approx(0.325)
    assert stats["oov"] == pytest.approx(0.5)


class _FailingDecoderForStats:
    model_hash = "stats-only"


def test_versioned_configuration_exposes_contextual_default_and_legacy_rollback():
    assert CONFIGURATIONS["selectdenoise_contextual_lattice"]["terminal_decoder"] == "contextual-lattice-v1"
    assert CONFIGURATIONS["selectdenoise_full"] == {"deanchor_atf": True, "use_verifier": True}
    assert CONFIGURATIONS["selectdenoise_full_legacy"]["use_verifier"] is True


def test_host_terminal_adapter_does_not_forward_noise_metadata():
    from multi_agent_v2 import _apply_contextual_lattice_terminal

    class _FakeTerminal:
        model_hash = "fake-terminal"
        fallback_count = 0

        def sentence_deer_stats(self, tokens):
            return {"entity": 0.0, "type": 0.0, "semantic": 0.0, "oov": 0.0}

        def decode(self, **kwargs):
            assert "noise_type" not in kwargs
            assert "dataset_name" not in kwargs
            return type("Result", (), {
                "tags": tuple(kwargs["anchor_tags"]),
                "used_anchor": True,
                "predicted_gain": 0.0,
                "model_hash": self.model_hash,
            })()

    state = {
        "tokens": ["Alice"],
        "dirty_tags": ["O"],
        "current_tags": ["B-PER"],
        "candidate_paths": [["B-PER"]],
        "rag_weights": [1.0],
        "dataset_name": "conll2003",
        "noise_type": "ATF",
    }
    output = _apply_contextual_lattice_terminal(state, _FakeTerminal())
    assert output["current_tags"] == ["B-PER"]
    assert output["terminal_model_hash"] == "fake-terminal"


def test_mocked_host_pipeline_invokes_terminal_without_provider_call(monkeypatch):
    import multi_agent_v2

    class _FakeTerminal:
        model_hash = "mock-model"
        fallback_count = 0

        def sentence_deer_stats(self, tokens):
            return {"entity": 0.0, "type": 0.0, "semantic": 0.0, "oov": 0.0}

        def decode(self, **kwargs):
            assert set(kwargs) == {
                "tokens", "dirty_tags", "anchor_tags", "candidate_paths",
                "reviewer_weights", "valid_types", "deer_stats",
            }
            return type("Result", (), {
                "tags": tuple(kwargs["anchor_tags"]),
                "used_anchor": True,
                "predicted_gain": 0.0,
                "model_hash": self.model_hash,
            })()

    async def fake_invoke(state):
        return {
            "current_tags": ["B-PER"],
            "candidate_paths": [["B-PER"]],
            "rag_weights": [1.0],
        }

    monkeypatch.setattr(multi_agent_v2.multi_agent_graph, "ainvoke", fake_invoke)
    monkeypatch.setattr(multi_agent_v2, "_load_contextual_lattice_terminal", lambda: _FakeTerminal())
    result = asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice"],
            ["B-PER"],
            {"terminal_decoder": "contextual-lattice-v1", "__return_candidates__": True, "__noise_type__": "BT"},
            dataset_name="conll2003",
        )
    )
    assert result["pred_tags"] == ["B-PER"]
    assert result["terminal_model_hash"] == "mock-model"


def test_contextual_cell_failure_does_not_publish_dirty_fallback(tmp_path: Path, monkeypatch):
    import run_multiseed

    monkeypatch.setattr(run_multiseed, "NOISY_DIR", tmp_path / "noisy")
    monkeypatch.setattr(run_multiseed, "PRED_DIR", tmp_path / "pred")
    noisy_path = run_multiseed._noisy_path("conll2003", "BT", 13, 1)
    noisy_path.parent.mkdir(parents=True)
    noisy_path.write_text(
        json.dumps({"tokens": ["Alice"], "ner_tags": ["B-PER"], "dirty_tags": ["O"]}) + "\n",
        encoding="utf-8",
    )

    async def failing_pipeline(*args, **kwargs):
        raise RuntimeError("bundle unavailable")

    with pytest.raises(RuntimeError, match="bundle unavailable"):
        asyncio.run(
            run_multiseed._run_one_cell(
                "selectdenoise_contextual_lattice",
                CONFIGURATIONS["selectdenoise_contextual_lattice"],
                "conll2003",
                "BT",
                13,
                1,
                (failing_pipeline, {}),
                max_concurrency=1,
                dummy=False,
            )
        )
    assert not run_multiseed._pred_path("selectdenoise_contextual_lattice", "conll2003", "BT", 13).exists()


def test_runtime_decode_signature_is_allow_listed_and_rejects_forbidden_metadata():
    parameters = inspect.signature(ContextualLatticeTerminal.decode).parameters
    assert "noise_type" not in parameters
    assert "dataset_name" not in parameters
    with pytest.raises(TypeError):
        ContextualLatticeTerminal(_FailingDecoderForStats()).decode(
            tokens=("Alice",),
            dirty_tags=("O",),
            anchor_tags=("B-PER",),
            candidate_paths=(),
            reviewer_weights=(),
            valid_types=frozenset({"PER"}),
            deer_stats={},
            __noise_type__="BT",
        )


def test_locked_report_manifest_hash_consistency():
    root = Path(
        os.environ.get(
            "CONTEXTUAL_LATTICE_LOCK_DIR",
            r"C:\Users\LinzeChen\AI_Workspace\unified_experiment_20260816\contextual_lattice_run_lock_bundle_20260821",
        )
    )
    if not root.exists():
        pytest.skip("locked external replay artifact is not mounted")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    prediction_hash = (root / "predictions.sha256").read_text(encoding="utf-8").strip()
    assert prediction_hash == report["prediction_hash"] == report["repeat_prediction_hash"]
    bundle_manifest = root / "bundle" / "manifest.json"
    digest = hashlib.sha256(bundle_manifest.read_bytes()).hexdigest()
    assert digest == report["manifest"]["bundle_manifest_hash"]
    assert report["manifest"]["margin"] == -0.1
    assert report["ser"] == 0.0
    assert report["invalid_lengths"] == 0


def test_documented_metrics_and_manifest_have_locked_inventory():
    docs = Path(__file__).parent / "docs" / "contextual_lattice"
    metrics = json.loads((docs / "metrics_450.json").read_text(encoding="utf-8"))
    manifest = json.loads((docs / "frozen_manifest.json").read_text(encoding="utf-8"))
    from contextual_lattice_runtime import (
        LOCKED_BUNDLE_MANIFEST_HASH,
        LOCKED_CHECKPOINT_HASH,
        LOCKED_DECODER_FILE_HASH,
        LOCKED_DECODER_MODEL_HASH,
        LOCKED_GATE_FILE_HASH,
        LOCKED_SPLIT_HASH,
    )
    assert set(metrics["cells"]) == {
        f"{dataset}__{family}"
        for dataset in ("conll2003", "fewnerd", "msra", "ontonotes5", "wnut17")
        for family in ("BT", "IF", "ATF")
    }
    assert metrics["prediction_hash"] == manifest["predictions"]["sha256"]
    assert metrics["repeat_prediction_hash"] == manifest["predictions"]["repeat_sha256"]
    assert manifest["formal_gate"]["observed_positive_cells"] == 12
    assert manifest["trusted_runtime_hash_pins"] == {
        "checkpoint_sha256": LOCKED_CHECKPOINT_HASH,
        "split_sha256": LOCKED_SPLIT_HASH,
        "manifest_sha256": LOCKED_BUNDLE_MANIFEST_HASH,
        "decoder_file_sha256": LOCKED_DECODER_FILE_HASH,
        "gate_file_sha256": LOCKED_GATE_FILE_HASH,
        "decoder_model_sha256": LOCKED_DECODER_MODEL_HASH,
    }
    for name, expected in manifest["source_sha256"].items():
        digest = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        assert digest == expected
