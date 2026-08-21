"""Run the pre-registered cached 450-example contextual-lattice gate.

The script is deliberately separate from ``multi_agent_v2.py``.  It consumes
immutable SelectDenoise cache rows, writes a new gold-free prediction artifact,
and only then joins audit metadata for reporting.  No provider/API is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from metrics import compute_prf1, compute_ser
from contextual_lattice_runtime import (
    LOCKED_BUNDLE_MANIFEST_HASH,
    LOCKED_CHECKPOINT_HASH,
    LOCKED_DECODER_FILE_HASH,
    LOCKED_DECODER_MODEL_HASH,
    LOCKED_GATE_FILE_HASH,
    LOCKED_SPLIT_HASH,
    load_bundle,
    save_bundle,
)
from selectdenoise_contextual_lattice import (
    ContextualLatticeDecoder,
    ContextualLatticeInput,
    GLiNERContextEncoder,
    HistoricalSplit,
    ResidualGate,
    _micro_f1,
    _sentence_f1,
    _span_counts,
    build_historical_split,
    iob2_to_spans,
    make_prediction_hash,
    refuse_overwrite,
)


DATASET_TYPES = {
    "msra": frozenset({"PER", "LOC", "ORG"}),
    "conll2003": frozenset({"PER", "LOC", "ORG", "MISC"}),
    "wnut17": frozenset({"PER", "LOC", "ORG", "MISC"}),
    "fewnerd": frozenset({"PER", "LOC", "ORG", "MISC"}),
    "ontonotes5": frozenset({"PER", "LOC", "ORG", "MISC"}),
}
FAMILIES = ("BT", "IF", "ATF")
SEEDS = (13, 42, 2024)
FILE_RE = re.compile(r"pred_seed(?P<seed>\d+)__selectdenoise_full__(?P<dataset>[^_]+)__(?P<family>BT|IF|ATF)\.jsonl$")
DEFAULT_WORKSPACE = Path(r"C:\Users\LinzeChen\AI_Workspace")
DEFAULT_EXPERIMENT_ROOT = DEFAULT_WORKSPACE / "unified_experiment_20260816"
MODEL_SALT = "selectdenoise-contextual-lattice-v1-20260820"


@dataclass(frozen=True)
class TrainingDeer:
    token_entity: Mapping[str, float]
    token_type: Mapping[str, float]
    vocabulary: frozenset[str]

    def to_profile(self) -> dict[str, Any]:
        """Return the JSON-safe, fold-local profile used by runtime features."""
        return {
            "token_entity": {str(key): float(value) for key, value in self.token_entity.items()},
            "token_type": {str(key): float(value) for key, value in self.token_type.items()},
            "vocabulary": sorted(str(token) for token in self.vocabulary),
        }

    def sentence_stats(self, tokens: Sequence[str]) -> dict[str, float]:
        if not tokens:
            return {"entity": 0.0, "type": 0.0, "semantic": 0.0, "oov": 0.0}
        entity = np.mean([self.token_entity.get(token, 0.0) for token in tokens])
        typ = np.mean([self.token_type.get(token, 0.0) for token in tokens])
        oov = np.mean([token not in self.vocabulary for token in tokens])
        # The semantic statistic is intentionally derived from the same
        # training-only token evidence; it is not a dataset/family feature.
        semantic = 0.5 * (entity + typ)
        return {"entity": float(entity), "type": float(typ), "semantic": float(semantic), "oov": float(oov)}


def load_historical_rows(workspace: Path = DEFAULT_WORKSPACE) -> list[dict[str, Any]]:
    prediction_root = workspace / "predictions_multiseed"
    noisy_root = workspace / "results_multiseed"
    rows: list[dict[str, Any]] = []
    files = sorted(prediction_root.glob("pred_seed*__selectdenoise_full__*__*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No selectdenoise_full cache rows under {prediction_root}")
    for prediction_file in files:
        match = FILE_RE.match(prediction_file.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        dataset = match.group("dataset")
        family = match.group("family")
        noisy_file = noisy_root / f"noisy_seed{seed}__{family}__{dataset}__N200.jsonl"
        if not noisy_file.exists():
            raise FileNotFoundError(f"Missing dirty cache for {prediction_file.name}: {noisy_file}")
        prediction_rows = [_read_json_line(line) for line in prediction_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        noisy_rows = [_read_json_line(line) for line in noisy_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prediction_rows) != len(noisy_rows):
            raise ValueError(f"Row count mismatch for {prediction_file.name}")
        valid_types = DATASET_TYPES.get(dataset)
        if valid_types is None:
            raise ValueError(f"Unknown dataset ontology {dataset}")
        for index, (prediction, noisy) in enumerate(zip(prediction_rows, noisy_rows)):
            if tuple(prediction["tokens"]) != tuple(noisy["tokens"]):
                raise ValueError(f"Token mismatch at {prediction_file.name}:{index}")
            paths = tuple(tuple(path) for path in prediction.get("candidate_paths", ()))
            weights = tuple(float(x) for x in prediction.get("rag_weights", ()))
            # The row metadata remains in this outer record only.  It is stripped
            # before construction of ContextualLatticeInput.
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "seed": seed,
                    "row_index": index,
                    "tokens": tuple(prediction["tokens"]),
                    "dirty_tags": tuple(noisy["dirty_tags"]),
                    "anchor_tags": tuple(prediction["pred_tags"]),
                    "gold_tags": tuple(prediction["gold_tags"]),
                    "candidate_paths": paths,
                    "reviewer_weights": weights,
                    "valid_types": valid_types,
                    "deer_stats": {},
                }
            )
    if len(rows) != 5 * 3 * 3 * 200:
        raise ValueError(f"Expected 9000 historical rows, found {len(rows)}")
    return rows


def _read_json_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("JSONL row must be an object")
    return value


def fit_training_deer(rows: Sequence[Mapping[str, Any]]) -> TrainingDeer:
    entity_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vocabulary: set[str] = set()
    for row in rows:
        tokens = tuple(row["tokens"])
        gold = tuple(row["gold_tags"])
        vocabulary.update(tokens)
        for token, tag in zip(tokens, gold):
            counts = entity_counts[token]
            counts[0] += 1.0
            if tag != "O":
                counts[1] += 1.0
                type_counts[token][tag[2:]] += 1
    token_entity = {token: values[1] / max(values[0], 1.0) for token, values in entity_counts.items()}
    token_type = {}
    for token, counts in type_counts.items():
        total = sum(counts.values())
        token_type[token] = max(counts.values()) / max(total, 1)
    return TrainingDeer(token_entity, token_type, frozenset(vocabulary))


def model_input(row: Mapping[str, Any], deer: TrainingDeer) -> ContextualLatticeInput:
    return ContextualLatticeInput(
        tokens=tuple(row["tokens"]),
        dirty_tags=tuple(row["dirty_tags"]),
        anchor_tags=tuple(row["anchor_tags"]),
        candidate_paths=tuple(tuple(path) for path in row["candidate_paths"]),
        reviewer_weights=tuple(float(x) for x in row["reviewer_weights"]),
        valid_types=frozenset(row["valid_types"]),
        deer_stats=deer.sentence_stats(tuple(row["tokens"])),
    )


def grouped_folds(rows: Sequence[Mapping[str, Any]], folds: int = 5) -> list[tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), json.dumps(list(row["tokens"]), ensure_ascii=False, separators=(",", ":")))
        groups[key].append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{MODEL_SALT}|fold|{key}".encode()).hexdigest())
    buckets = [set() for _ in range(folds)]
    for index, key in enumerate(ordered):
        buckets[index % folds].add(key)
    result = []
    for fold in range(folds):
        valid_keys = buckets[fold]
        train = [row for key, records in groups.items() if key not in valid_keys for row in records]
        valid = [row for key, records in groups.items() if key in valid_keys for row in records]
        result.append((train, valid))
    return result


def _weighted_training_records(rows: Sequence[Mapping[str, Any]], deer: TrainingDeer) -> list[tuple[ContextualLatticeInput, tuple[str, ...], float]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row["dataset"]), json.dumps(list(row["tokens"]), ensure_ascii=False, separators=(",", ":")))
        counts[key] += 1
    records = []
    for row in rows:
        key = (str(row["dataset"]), json.dumps(list(row["tokens"]), ensure_ascii=False, separators=(",", ":")))
        records.append((model_input(row, deer), tuple(row["gold_tags"]), 1.0 / counts[key]))
    return records


def _strict_error(anchor: Sequence[str], lattice: Sequence[str], gold: Sequence[str]) -> float:
    a = _span_counts(gold, anchor)
    b = _span_counts(gold, lattice)
    return float((a[1] + a[2]) - (b[1] + b[2]))


def fit_oof_residual_gate(train_rows: Sequence[Mapping[str, Any]], encoder: Any, *, epochs: int, device: str) -> ResidualGate:
    feature_rows: list[np.ndarray] = []
    gain_rows: list[float] = []
    for fold_train, fold_valid in grouped_folds(train_rows):
        deer = fit_training_deer(fold_train)
        decoder = ContextualLatticeDecoder(
            {"PER", "LOC", "ORG", "MISC"}, encoder=encoder, epochs=epochs, device=device, seed=731
        )
        decoder.fit(_weighted_training_records(fold_train, deer))
        for row in fold_valid:
            value = model_input(row, deer)
            bundle, scores = decoder.score_candidates(value)
            from selectdenoise_contextual_lattice import decode_exact_lattice, CandidateSpan

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
            anchor_spans = iob2_to_spans(value.anchor_tags)
            feature_rows.append(decoder.gate_features(bundle, scores, lattice.spans, anchor_spans))
            lattice_tags = decoder.decode(value, margin=-float("inf")).raw_lattice_tags
            gain_rows.append(_strict_error(value.anchor_tags, lattice_tags, tuple(row["gold_tags"])))
    gate = ResidualGate(seed=731)
    gate.fit(np.asarray(feature_rows, dtype=np.float32), np.asarray(gain_rows, dtype=np.float32))
    return gate


def _prediction_payload(result: Any) -> dict[str, Any]:
    return {
        "tags": list(result.tags),
        "raw_lattice_tags": list(result.raw_lattice_tags),
        "used_anchor": bool(result.used_anchor),
        "predicted_gain": float(result.predicted_gain),
        "selected_spans": [[span.start, span.end, span.type] for span in result.selected_spans],
        "model_hash": result.model_hash,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    iterations: int = 10_000,
) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = f"{row['dataset']}|{json.dumps(list(row['tokens']), ensure_ascii=False, separators=(',', ':'))}"
        groups[key].append(index)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for key in groups:
        by_dataset[key.split("|", 1)[0]].append(key)
    rng = np.random.default_rng(731)
    deltas = np.empty(iterations, dtype=np.float64)
    gold_all = [tuple(row["gold_tags"]) for row in rows]
    anchor_all = [tuple(row["anchor_tags"]) for row in rows]
    pred_all = [tuple(prediction["tags"]) for prediction in predictions]
    for iteration in range(iterations):
        indices: list[int] = []
        for dataset in sorted(by_dataset):
            chosen = rng.choice(by_dataset[dataset], size=len(by_dataset[dataset]), replace=True)
            for key in chosen:
                indices.extend(groups[key])
        deltas[iteration] = _micro_f1([gold_all[i] for i in indices], [pred_all[i] for i in indices]) - _micro_f1(
            [gold_all[i] for i in indices], [anchor_all[i] for i in indices]
        )
    return {
        "delta": float(np.mean(deltas)),
        "lower_95": float(np.quantile(deltas, 0.05)),
        "upper_95": float(np.quantile(deltas, 0.95)),
        "p_nonpositive": float(np.mean(deltas <= 0.0)),
    }


def evaluate(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row["dataset"]), str(row["family"]))].append(index)
    cells = {}
    wins = ties = losses = 0
    deltas: list[float] = []
    anchor_all, pred_all, gold_all = [], [], []
    for (dataset, family), indices in sorted(grouped.items()):
        # ``metrics.compute_prf1`` delegates to seqeval, which requires a
        # list-of-lists rather than a list of tuples.  Keep the serialized
        # representation immutable, but normalize at this evaluator boundary.
        gold = [list(rows[i]["gold_tags"]) for i in indices]
        anchor = [list(rows[i]["anchor_tags"]) for i in indices]
        pred = [list(predictions[i]["tags"]) for i in indices]
        p = compute_prf1(gold, pred)
        a = compute_prf1(gold, anchor)
        delta = p["f1"] - a["f1"]
        deltas.append(delta)
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
        cells[f"{dataset}__{family}"] = {"method": p, "anchor": a, "delta_f1": delta, "n": len(indices)}
        gold_all.extend(gold)
        anchor_all.extend(anchor)
        pred_all.extend(pred)
    clean_indices = [i for i, row in enumerate(rows) if tuple(row["dirty_tags"]) == tuple(row["gold_tags"])]
    clean_changes = [i for i in clean_indices if tuple(predictions[i]["tags"]) != tuple(rows[i]["anchor_tags"])]
    clean_regressions = [
        i
        for i in clean_indices
        if _sentence_f1(tuple(predictions[i]["tags"]), tuple(rows[i]["gold_tags"]))
        < _sentence_f1(tuple(rows[i]["anchor_tags"]), tuple(rows[i]["gold_tags"])) - 1e-12
    ]
    pooled_method = compute_prf1(gold_all, pred_all)
    pooled_anchor = compute_prf1(gold_all, anchor_all)
    oracle = []
    for gold, anchor, pred in zip(gold_all, anchor_all, pred_all):
        oracle.append(pred if _sentence_f1(pred, gold) >= _sentence_f1(anchor, gold) else anchor)
    oracle_gap = _micro_f1(gold_all, oracle) - pooled_anchor["f1"]
    recovered = (pooled_method["f1"] - pooled_anchor["f1"]) / oracle_gap if oracle_gap > 0 else 0.0
    return {
        "cells": cells,
        "pooled": {"method": pooled_method, "anchor": pooled_anchor, "delta_f1": pooled_method["f1"] - pooled_anchor["f1"]},
        "wins_ties_losses": [wins, ties, losses],
        "worst_cell_delta": float(min(deltas) if deltas else 0.0),
        "bootstrap": _stratified_bootstrap(rows, predictions),
        "oracle_gap": float(oracle_gap),
        "oracle_gap_recovery": float(recovered),
        "accepted_alternatives": int(sum(not prediction["used_anchor"] for prediction in predictions)),
        "clean_rows": len(clean_indices),
        "clean_changes": len(clean_changes),
        "clean_change_rate": len(clean_changes) / max(len(clean_indices), 1),
        "clean_regressions": len(clean_regressions),
        "ser": compute_ser([list(prediction["tags"]) for prediction in predictions]),
        "invalid_lengths": sum(len(prediction["tags"]) != len(row["tokens"]) for row, prediction in zip(rows, predictions)),
    }


def run_gate(
    *,
    workspace: Path = DEFAULT_WORKSPACE,
    output: Path | None = None,
    checkpoint: Path | None = None,
    encoder_cache: Path | None = None,
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    epochs: int = 20,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    rows = load_historical_rows(workspace)
    split = build_historical_split(rows, groups_per_dataset=30, salt=MODEL_SALT, seeds=SEEDS)
    if output is None:
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        output = workspace / "unified_experiment_20260816" / f"contextual_lattice_run_{code_hash}_450"
    refuse_overwrite(output)
    output.mkdir(parents=True)

    if checkpoint is None:
        checkpoint = workspace / "unified_experiment_20260816" / "hf_home" / "hub" / "models--urchade--gliner_multi-v2.1" / "snapshots" / "443d26d654e0324125a96bebd8e796c14ff2efe6"
    cache = encoder_cache if encoder_cache is not None else output / "embedding_cache"
    encoder = GLiNERContextEncoder(checkpoint, {"PER", "LOC", "ORG", "MISC"}, device=device, cache_dir=cache)

    # Five-fold grouped OOF gate fitting is completed before the all-training
    # decoder is allowed to inspect calibration/test records.
    oof_gate = fit_oof_residual_gate(split.train, encoder, epochs=epochs, device=device)
    train_deer = fit_training_deer(split.train)
    decoder = ContextualLatticeDecoder({"PER", "LOC", "ORG", "MISC"}, encoder=encoder, epochs=epochs, device=device)
    decoder.fit(_weighted_training_records(split.train, train_deer))
    decoder.gate = oof_gate

    calibration_predictions = []
    calibration_features = []
    calibration_anchor = []
    calibration_lattice = []
    calibration_gold = []
    for row in split.calibration:
        value = model_input(row, train_deer)
        bundle, scores = decoder.score_candidates(value)
        raw = decoder.decode(value, margin=-float("inf"))
        anchor_spans = iob2_to_spans(value.anchor_tags)
        calibration_features.append(decoder.gate_features(bundle, scores, raw.selected_spans, anchor_spans))
        calibration_anchor.append(value.anchor_tags)
        calibration_lattice.append(raw.raw_lattice_tags)
        calibration_gold.append(tuple(row["gold_tags"]))
        calibration_predictions.append(raw)
    margin = oof_gate.choose_margin(
        np.asarray(calibration_features),
        calibration_anchor,
        calibration_lattice,
        calibration_gold,
        retention_null=[tuple(row["dirty_tags"]) == tuple(row["gold_tags"]) for row in split.calibration],
    )
    decoder.margin = margin
    calibration_final = [decoder.decode(model_input(row, train_deer)) for row in split.calibration]
    calibration_report = evaluate(split.calibration, [_prediction_payload(result) for result in calibration_final])
    positive_calibration_cells = sum(cell["delta_f1"] > 0 for cell in calibration_report["cells"].values())
    if (
        calibration_report["pooled"]["delta_f1"] < 0.005
        or positive_calibration_cells < 11
        or calibration_report["accepted_alternatives"] == 0
        or calibration_report["ser"] != 0.0
        or calibration_report["invalid_lengths"] != 0
        or calibration_report["clean_regressions"] != 0
    ):
        raise RuntimeError("Calibration pre-gate failed; test gate was not opened")

    split_hash = hashlib.sha256(
        json.dumps(
            {
                "test": sorted([list(x) for x in split.test_groups]),
                "calibration": sorted([list(x) for x in split.calibration_groups]),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    # Freeze the protocol/model manifest before test inference.
    manifest = {
        "schema_version": "contextual-lattice-v1",
        "salt": MODEL_SALT,
        "rows": len(rows),
        "train_rows": len(split.train),
        "calibration_rows": len(split.calibration),
        "test_rows": len(split.test),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": encoder.checkpoint_hash,
        "decoder_model_hash": decoder.model_hash,
        "gate_model": str(type(oof_gate.model).__name__),
        "margin": margin,
        "seed": 731,
        "epochs": epochs,
        "device": device,
        "split_hash": split_hash,
    }
    bundle_dir = save_bundle(
        decoder,
        oof_gate,
        output / "bundle",
        {
            **manifest,
            "deer_profile": train_deer.to_profile(),
        },
    )
    manifest["bundle_dir"] = str(bundle_dir)
    manifest["bundle_manifest_hash"] = hashlib.sha256(
        (bundle_dir / "manifest.json").read_bytes()
    ).hexdigest()
    _write_json(output / "frozen_manifest.json", manifest)

    # Write only model-facing, gold-free predictions before the evaluator joins
    # dataset/family/gold metadata.  Reload the just-written frozen bundle so
    # the replay validates serialization, rather than only the in-memory model.
    terminal = load_bundle(
        bundle_dir,
        encoder,
        expected_checkpoint_hash=LOCKED_CHECKPOINT_HASH,
        expected_split_hash=LOCKED_SPLIT_HASH,
        expected_manifest_hash=LOCKED_BUNDLE_MANIFEST_HASH,
        expected_decoder_file_hash=LOCKED_DECODER_FILE_HASH,
        expected_gate_file_hash=LOCKED_GATE_FILE_HASH,
        expected_decoder_model_hash=LOCKED_DECODER_MODEL_HASH,
    )

    def runtime_decode(row: Mapping[str, Any]) -> Any:
        value = model_input(row, train_deer)
        return terminal.decode(
            tokens=value.tokens,
            dirty_tags=value.dirty_tags,
            anchor_tags=value.anchor_tags,
            candidate_paths=value.candidate_paths,
            reviewer_weights=value.reviewer_weights,
            valid_types=value.valid_types,
            deer_stats=terminal.sentence_deer_stats(value.tokens),
        )

    test_results = [runtime_decode(row) for row in split.test]
    gold_free = [_prediction_payload(result) for result in test_results]
    repeat_results = [runtime_decode(row) for row in split.test]
    repeat_hash = make_prediction_hash([result.tags for result in repeat_results])
    _write_jsonl(output / "predictions_gold_free.jsonl", gold_free)
    prediction_hash = make_prediction_hash([result["tags"] for result in gold_free])
    if repeat_hash != prediction_hash:
        raise RuntimeError("Two complete inference passes produced different prediction hashes")
    (output / "predictions.sha256").write_text(prediction_hash + "\n", encoding="utf-8")
    report = evaluate(split.test, gold_free)
    report.update(
        {
            "prediction_hash": prediction_hash,
            "repeat_prediction_hash": repeat_hash,
            "model_hash": terminal.model_hash,
            "terminal_fallback_count": terminal.fallback_count,
            "bundle_dir": str(bundle_dir),
            "runtime_seconds": time.perf_counter() - start_time,
            "api_calls": 0,
            "external_cost_usd": 0.0,
            "calibration": calibration_report,
            "manifest": manifest,
        }
    )
    _write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--encoder-cache", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    report = run_gate(
        workspace=args.workspace,
        output=args.output,
        checkpoint=args.checkpoint,
        encoder_cache=args.encoder_cache,
        device=args.device,
        epochs=args.epochs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
