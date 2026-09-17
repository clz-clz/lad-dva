"""Analyze completed Contextual-Lattice study artifacts.

All metrics are computed from the study namespace after strict row validation.
The primary ablation comparisons use the locked 450-row test selection; the
full 9,000-row matrix is reported separately as descriptive deployment scope.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import contextual_studies as study
from metrics import (
    _per_sentence_span_counts,
    compute_prf1,
    compute_ser,
    token_class_rates,
)


LOGGER = logging.getLogger("contextual-study-analysis")
BOOTSTRAP_ITERATIONS = 10_000


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _span_set(tags: Sequence[str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    index = 0
    while index < len(tags):
        tag = tags[index]
        if isinstance(tag, str) and tag.startswith("B-"):
            entity_type = tag[2:]
            end = index + 1
            while end < len(tags) and tags[end] == f"I-{entity_type}":
                end += 1
            spans.add((entity_type, index, end))
            index = end
        else:
            index += 1
    return spans


def _metric_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold = [list(row["gold_tags"]) for row in rows]
    pred = [list(row["pred_tags"]) for row in rows]
    metrics = compute_prf1(gold, pred)
    rates = token_class_rates(pred)
    acceptance = float(np.mean([
        not bool(row.get("terminal_used_anchor")) for row in rows
    ])) if rows else 0.0
    fallback = sum(int(row.get("terminal_fallback_count", 0)) for row in rows)
    return {
        "rows": len(rows),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "ser": compute_ser(pred),
        "o_rate": rates["o_rate"],
        "entity_rate": rates["entity_rate"],
        "terminal_acceptance_rate": acceptance,
        "terminal_anchor_rate": 1.0 - acceptance,
        "terminal_fallback_count": fallback,
    }


def _selection_map(selection: Sequence[Mapping[str, Any]], dataset: str, noise: str) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(item["seed"]), int(item["row_index"])): dict(item)
        for item in selection
        if item.get("dataset") == dataset and item.get("family") == noise
    }


def _load_ablation_rows(root: Path, variant: str) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    loaded: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for dataset in study.DATASETS:
        for noise in study.NOISE_TYPES:
            for seed in study.SEEDS:
                path = study.study_prediction_path(
                    root, "ablation", variant, dataset, noise,
                    study.OFFICIAL_RATIO, seed,
                )
                rows = study.validate_study_prediction_file(
                    path, dataset, expected_count=study.SAMPLE_SIZE,
                )
                for index, row in enumerate(rows):
                    key = (dataset, noise, seed, index)
                    if key in loaded:
                        raise ValueError(f"duplicate ablation coordinate: {key}")
                    loaded[key] = row
    return loaded


def _primary_rows(
    loaded: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selection:
        key = (item["dataset"], item["family"], int(item["seed"]), int(item["row_index"]))
        if key not in loaded:
            raise ValueError(f"ablation output misses primary coordinate: {key}")
        row = dict(loaded[key])
        row["_group"] = item["group_digest"]
        rows.append(row)
    return rows


def _group_bootstrap(
    gold: Sequence[Sequence[str]], pred_a: Sequence[Sequence[str]],
    pred_b: Sequence[Sequence[str]], groups: Sequence[str],
    strata: Sequence[str], *, iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 0,
) -> dict[str, float]:
    """Paired strict-span bootstrap clustered by group within each stratum."""
    if not (len(gold) == len(pred_a) == len(pred_b) == len(groups) == len(strata)):
        raise ValueError("cluster bootstrap inputs are not aligned")
    if not gold:
        return {"f1_A": 0.0, "f1_B": 0.0, "delta": 0.0,
                "p_value": 1.0, "ci_95_low": 0.0, "ci_95_high": 0.0}
    counts_a = _per_sentence_span_counts([list(x) for x in gold], [list(x) for x in pred_a])
    counts_b = _per_sentence_span_counts([list(x) for x in gold], [list(x) for x in pred_b])
    strata_groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (group, strata_key) in enumerate(zip(groups, strata)):
        strata_groups[strata_key][group].append(index)
    # Each locked selection cell contains one row per group.  Refuse accidental
    # duplicate coordinates rather than silently overweighting a sentence.
    if any(len(indices) != 1 for cell in strata_groups.values() for indices in cell.values()):
        raise ValueError("cluster bootstrap requires one row per group within each stratum")
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    sampled_cells = [
        [indices[0] for indices in groups_by_key.values()]
        for groups_by_key in strata_groups.values()
    ]
    for iteration in range(iterations):
        indices: list[int] = []
        for cell_indices in sampled_cells:
            draw = rng.integers(0, len(cell_indices), size=len(cell_indices))
            indices.extend(cell_indices[int(index)] for index in draw)
        a = counts_a[indices].sum(axis=0)
        b = counts_b[indices].sum(axis=0)
        denom_a = 2 * a[0] + a[1] + a[2]
        denom_b = 2 * b[0] + b[1] + b[2]
        f1_a = float(2 * a[0] / denom_a) if denom_a else 0.0
        f1_b = float(2 * b[0] / denom_b) if denom_b else 0.0
        deltas[iteration] = f1_a - f1_b
    a_total = counts_a.sum(axis=0)
    b_total = counts_b.sum(axis=0)
    denom_a = 2 * a_total[0] + a_total[1] + a_total[2]
    denom_b = 2 * b_total[0] + b_total[1] + b_total[2]
    obs_a = float(2 * a_total[0] / denom_a) if denom_a else 0.0
    obs_b = float(2 * b_total[0] / denom_b) if denom_b else 0.0
    return {
        "f1_A": obs_a,
        "f1_B": obs_b,
        "delta": obs_a - obs_b,
        "p_value": float(np.mean(deltas <= 0.0)),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "group_within_dataset_noise_stratum",
    }


def analyze_ablation(root: Path, source_tag: str = study.SOURCE_TAG) -> dict[str, Any]:
    selection = study.load_selection(root)
    primary_by_variant: dict[str, list[dict[str, Any]]] = {}
    full_by_variant: dict[str, dict[tuple[str, str, int, int], dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for variant in study.ABLATION_VARIANTS:
        loaded = _load_ablation_rows(root, variant)
        full_by_variant[variant] = loaded
        primary = _primary_rows(loaded, selection)
        primary_by_variant[variant] = primary
        pooled = _metric_record(primary)
        metric_rows.append({
            "scope": "primary_test_pooled", "variant": variant,
            **pooled,
        })
        metric_rows.append({
            "scope": "full_namespace_pooled", "variant": variant,
            **_metric_record(list(loaded.values())),
        })
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                condition = [
                    row for row in primary
                    if row["study_dataset"] == dataset and row["study_noise"] == noise
                ]
                metric_rows.append({
                    "scope": "primary_test_condition", "variant": variant,
                    "dataset": dataset, "noise": noise,
                    **_metric_record(condition),
                })

    comparisons: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for index, variant in enumerate(study.ABLATION_VARIANTS):
        if variant == "full":
            continue
        full = primary_by_variant["full"]
        ablated = primary_by_variant[variant]
        if len(full) != len(ablated):
            raise ValueError("paired ablation selections have different lengths")
        groups = [f"{row['_group']}" for row in full]
        strata = [f"{row['study_dataset']}|{row['study_noise']}" for row in full]
        result = _group_bootstrap(
            [row["gold_tags"] for row in full],
            [row["pred_tags"] for row in full],
            [row["pred_tags"] for row in ablated],
            groups, strata, seed=20260914 + index,
        )
        comparison = {
            "full_vs": variant,
            "comparison": "Full - ablation",
            **result,
        }
        comparisons.append(comparison)
        raw_p.append(result["p_value"])
    adjusted = study.holm_adjust(raw_p)
    for comparison, adjusted_p in zip(comparisons, adjusted):
        comparison["p_value_holm"] = adjusted_p

    payload = {
        "schema_version": study.STUDY_MANIFEST_SCHEMA,
        "study_tag": root.name,
        "source_tag": source_tag,
        "primary_scope": {
            "rows": len(selection),
            "groups": 150,
            "selection": "locked historical test split",
            "bootstrap": "10000 clustered resamples, dataset/noise stratified",
        },
        "metrics": metric_rows,
        "comparisons": comparisons,
        "terminal_acceptance_definition": "fraction of rows where terminal_used_anchor is false",
    }
    analysis_dir = Path(root) / "analysis"
    study._write_json_atomic(analysis_dir / "ablation_results.json", payload)
    _write_csv(analysis_dir / "ablation_results.csv", metric_rows + comparisons)
    return payload


def _load_gradient_cell(root: Path, dataset: str, noise: str, ratio: float) -> list[dict[str, Any]]:
    path = study.study_prediction_path(root, "noise-gradient", "full", dataset, noise, ratio)
    return study.validate_study_prediction_file(
        path, dataset, expected_count=study.TEST_GROUPS_PER_DATASET,
    )


def _realized_corruption_counts(
    root: Path, dataset: str, noise: str, ratio: float,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    changed = 0
    total = 0
    cache: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["study_seed"])
        if seed not in cache:
            cache[seed] = study._input_cell_rows(root, dataset, noise, seed, ratio)
        noisy = cache[seed][int(row["study_row_index"])]
        gold_spans = _span_set(noisy["ner_tags"])
        total += len(gold_spans)
        for entity_type, start, end in gold_spans:
            if list(noisy["dirty_tags"])[start:end] != list(noisy["ner_tags"])[start:end]:
                changed += 1
    return changed, total


def _realized_corruption(
    root: Path, dataset: str, noise: str, ratio: float,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    changed, total = _realized_corruption_counts(root, dataset, noise, ratio, rows)
    return float(changed / total) if total else 0.0


def _plot_gradient_curves(root: Path, records: Sequence[Mapping[str, Any]]) -> list[str]:
    plot_paths: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib unavailable; curve data remains available in CSV/JSON")
        return plot_paths
    plot_dir = Path(root) / "analysis"
    rates = list(study.GRADIENT_RATIOS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, noise in zip(axes, study.NOISE_TYPES):
        values = [
            next(record["f1"] for record in records
                 if record["scope"] == "pooled_by_noise_rate"
                 and record["noise"] == noise and abs(record["ratio"] - ratio) < 1e-12)
            for ratio in rates
        ]
        axis.plot([rate * 100 for rate in rates], values, marker="o", label="F1")
        axis.set_title(noise)
        axis.set_xlabel("Noise rate (%)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Strict IOB2 F1")
    fig.tight_layout()
    path = plot_dir / "noise_gradient_f1_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path.relative_to(root).as_posix()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, noise in zip(axes, study.NOISE_TYPES):
        values = [
            next(record["terminal_acceptance_rate"] for record in records
                 if record["scope"] == "pooled_by_noise_rate"
                 and record["noise"] == noise and abs(record["ratio"] - ratio) < 1e-12)
            for ratio in rates
        ]
        axis.plot([rate * 100 for rate in rates], values, marker="o", color="tab:orange")
        axis.set_title(noise)
        axis.set_xlabel("Noise rate (%)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Terminal lattice acceptance")
    fig.tight_layout()
    path = plot_dir / "noise_gradient_terminal_acceptance_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path.relative_to(root).as_posix()))
    return plot_paths


def analyze_gradient(root: Path, source_tag: str = study.SOURCE_TAG) -> dict[str, Any]:
    selection = study.load_selection(root)
    condition_records: list[dict[str, Any]] = []
    cell_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for ratio in study.GRADIENT_RATIOS:
        for dataset in study.DATASETS:
            for noise in study.NOISE_TYPES:
                rows = _load_gradient_cell(root, dataset, noise, ratio)
                cell_rows[(dataset, noise, ratio)] = rows
                condition_records.append({
                    "scope": "dataset_noise_rate", "dataset": dataset,
                    "noise": noise, "ratio": ratio,
                    "realized_entity_corruption_rate": _realized_corruption(
                        root, dataset, noise, ratio, rows,
                    ),
                    **_metric_record(rows),
                })
    for ratio in study.GRADIENT_RATIOS:
        for noise in study.NOISE_TYPES:
            rows = [row for dataset in study.DATASETS
                    for row in cell_rows[(dataset, noise, ratio)]]
            changed = total = 0
            for dataset in study.DATASETS:
                current_changed, current_total = _realized_corruption_counts(
                    root, dataset, noise, ratio,
                    cell_rows[(dataset, noise, ratio)],
                )
                changed += current_changed
                total += current_total
            condition_records.append({
                "scope": "pooled_by_noise_rate", "noise": noise,
                "ratio": ratio,
                "realized_entity_corruption_rate": float(changed / total) if total else 0.0,
                **_metric_record(rows),
            })

    deltas: list[dict[str, Any]] = []
    for noise in study.NOISE_TYPES:
        for ratio in study.GRADIENT_RATIOS:
            if abs(ratio - study.OFFICIAL_RATIO) < 1e-12:
                continue
            base_rows = [row for dataset in study.DATASETS
                         for row in cell_rows[(dataset, noise, study.OFFICIAL_RATIO)]]
            target_rows = [row for dataset in study.DATASETS
                           for row in cell_rows[(dataset, noise, ratio)]]
            lookup_base = {(row["study_dataset"], row["study_seed"], row["study_row_index"]): row
                           for row in base_rows}
            paired_base: list[dict[str, Any]] = []
            paired_target: list[dict[str, Any]] = []
            for row in target_rows:
                key = (row["study_dataset"], row["study_seed"], row["study_row_index"])
                paired_base.append(lookup_base[key])
                paired_target.append(row)
            groups = [f"{row['study_dataset']}|{row['study_group_digest']}"
                      for row in paired_target]
            strata = [f"{row['study_dataset']}|{noise}" for row in paired_target]
            result = _group_bootstrap(
                [row["gold_tags"] for row in paired_base],
                [row["pred_tags"] for row in paired_target],
                [row["pred_tags"] for row in paired_base],
                groups, strata,
                seed=20261001 + int(round(ratio * 100)) + study.NOISE_TYPES.index(noise),
            )
            deltas.append({
                "scope": "delta_vs_15_pooled", "noise": noise, "ratio": ratio,
                "reference_ratio": study.OFFICIAL_RATIO,
                "delta_f1": result["delta"],
                "p_value": result["p_value"],
                "ci_95_low": result["ci_95_low"],
                "ci_95_high": result["ci_95_high"],
                "bootstrap_iterations": result["bootstrap_iterations"],
            })

    auc_records: list[dict[str, Any]] = []
    for noise in study.NOISE_TYPES:
        y = [next(record["f1"] for record in condition_records
                  if record["scope"] == "pooled_by_noise_rate"
                  and record["noise"] == noise and abs(record["ratio"] - ratio) < 1e-12)
             for ratio in study.GRADIENT_RATIOS]
        auc_records.append({
            "scope": "pooled_gradient_auc", "noise": noise,
            "auc_f1_vs_rate": float(np.trapezoid(y, list(study.GRADIENT_RATIOS)))
            if hasattr(np, "trapezoid") else float(np.trapz(y, list(study.GRADIENT_RATIOS))),
            "rate_min": min(study.GRADIENT_RATIOS),
            "rate_max": max(study.GRADIENT_RATIOS),
        })

    all_records = condition_records + deltas + auc_records
    plot_paths = _plot_gradient_curves(root, condition_records)
    payload = {
        "schema_version": study.STUDY_MANIFEST_SCHEMA,
        "study_tag": root.name,
        "source_tag": source_tag,
        "selection": {
            "rows_per_condition": study.TEST_GROUPS_PER_DATASET,
            "groups_per_dataset": study.TEST_GROUPS_PER_DATASET,
            "rates": list(study.GRADIENT_RATIOS),
            "bootstrap": "10000 clustered paired resamples, dataset/noise stratified",
        },
        "records": all_records,
        "plots": plot_paths,
        "terminal_acceptance_definition": "fraction of rows where terminal_used_anchor is false",
    }
    analysis_dir = Path(root) / "analysis"
    study._write_json_atomic(analysis_dir / "noise_gradient_results.json", payload)
    _write_csv(analysis_dir / "noise_gradient_results.csv", all_records)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-tag", default=study.SOURCE_TAG)
    parser.add_argument("--only", choices=("ablation", "noise-gradient", "all"), default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = args.root if args.root.is_absolute() else Path(__file__).resolve().parent / args.root
    result: dict[str, Any] = {}
    if args.only in {"ablation", "all"}:
        result["ablation"] = analyze_ablation(root, args.source_tag)
    if args.only in {"noise-gradient", "all"}:
        result["noise_gradient"] = analyze_gradient(root, args.source_tag)
    study._write_json_atomic(root / "analysis" / "summary.json", {
        "schema_version": study.STUDY_MANIFEST_SCHEMA,
        "study_tag": root.name,
        "completed": list(result),
    })
    LOGGER.info("analysis complete: %s", ", ".join(result))


if __name__ == "__main__":
    main()
