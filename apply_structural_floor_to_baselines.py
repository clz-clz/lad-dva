from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from utils import enforce_iob2_syntax


BASELINES = (
    "baseline_zero_shot",
    "baseline_cot_reasoning",
    "baseline_self_refine",
    "baseline_standard_prompting",
)

DATASET_ENTITY_TYPES = {
    "msra": {"PER", "LOC", "ORG"},
    "conll2003": {"PER", "LOC", "ORG", "MISC"},
    "wnut17": {"PER", "LOC", "ORG", "MISC"},
    "fewnerd": {"PER", "LOC", "ORG", "MISC"},
    "ontonotes5": {"PER", "LOC", "ORG", "MISC"},
}

PRED_DIR = Path("predictions_multiseed")


def _source_files(pred_dir: Path, baselines: Iterable[str]) -> Iterator[Path]:
    for baseline in baselines:
        yield from sorted(pred_dir.glob(f"pred_seed*__{baseline}__*__*.jsonl"))


def _parse_pred_name(path: Path) -> tuple[str, str, str, str]:
    parts = path.stem.split("__")
    if len(parts) < 4 or not parts[0].startswith("pred_seed"):
        raise ValueError(f"Unrecognized prediction filename: {path.name}")
    seed = parts[0][len("pred_seed"):]
    baseline = parts[1]
    dataset = parts[2]
    noise = "__".join(parts[3:])
    return seed, baseline, dataset, noise


def _output_path(source: Path) -> Path:
    seed, baseline, dataset, noise = _parse_pred_name(source)
    return source.with_name(
        f"pred_seed{seed}__{baseline}_sfloor__{dataset}__{noise}.jsonl"
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _apply_structural_floor(rows: list[dict], dataset_name: str) -> list[dict]:
    valid = DATASET_ENTITY_TYPES.get(dataset_name, {"PER", "LOC", "ORG", "MISC"})
    repaired = []
    for row in rows:
        updated = dict(row)
        updated["pred_tags"] = enforce_iob2_syntax(list(row["pred_tags"]), valid)
        repaired.append(updated)
    return repaired


def process_file(source: Path, *, force: bool = False) -> tuple[str, str]:
    output = _output_path(source)
    if output.exists() and output.stat().st_size > 0 and not force:
        return ("skipped", output.name)

    _, _, dataset, _ = _parse_pred_name(source)
    rows = _load_jsonl(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(output, _apply_structural_floor(rows, dataset))
    return ("written", output.name)


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(
        description="Apply a structural IOB2 floor to baseline prediction files."
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing structural-floor outputs.")
    args = parser.parse_args(argv)

    summary = {
        "scanned": 0,
        "written": 0,
        "skipped": 0,
        "missing": 0,
    }

    for source in _source_files(PRED_DIR, BASELINES):
        summary["scanned"] += 1
        try:
            status, _ = process_file(source, force=args.force)
        except FileNotFoundError:
            summary["missing"] += 1
            continue
        summary[status] += 1

    print(
        "structural-floor summary: "
        f"scanned={summary['scanned']} "
        f"written={summary['written']} "
        f"skipped={summary['skipped']} "
        f"missing={summary['missing']}"
    )
    return summary


if __name__ == "__main__":
    main()
