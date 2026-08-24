from __future__ import annotations

import json
from pathlib import Path

import pytest

import apply_structural_floor_to_baselines as sfloor


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_illegal_transitions_are_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_dir = tmp_path / "predictions_multiseed"
    source = pred_dir / "pred_seed13__baseline_zero_shot__conll2003__BT.jsonl"
    _write_jsonl(
        source,
        [
            {
                "tokens": ["Alice", "met", "Bob"],
                "gold_tags": ["B-PER", "O", "B-PER"],
                "pred_tags": ["I-PER", "I-LOC", "O"],
            }
        ],
    )
    monkeypatch.setattr(sfloor, "PRED_DIR", pred_dir)

    summary = sfloor.main([])

    out = pred_dir / "pred_seed13__baseline_zero_shot_sfloor__conll2003__BT.jsonl"
    assert _read_jsonl(out) == [
        {
            "tokens": ["Alice", "met", "Bob"],
            "gold_tags": ["B-PER", "O", "B-PER"],
            "pred_tags": ["B-PER", "B-LOC", "O"],
        }
    ]
    assert summary["written"] == 1
    assert summary["skipped"] == 0


def test_legal_rows_and_metadata_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_dir = tmp_path / "predictions_multiseed"
    source = pred_dir / "pred_seed42__baseline_cot_reasoning__ontonotes5__IF.jsonl"
    row = {
        "tokens": ["New", "York"],
        "gold_tags": ["B-LOC", "I-LOC"],
        "pred_tags": ["B-LOC", "I-LOC"],
        "sentence_id": 7,
        "notes": {"source": "manual"},
    }
    _write_jsonl(source, [row])
    monkeypatch.setattr(sfloor, "PRED_DIR", pred_dir)

    sfloor.main([])

    out = pred_dir / "pred_seed42__baseline_cot_reasoning_sfloor__ontonotes5__IF.jsonl"
    assert _read_jsonl(out) == [row]


def test_existing_outputs_are_skipped_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_dir = tmp_path / "predictions_multiseed"
    source = pred_dir / "pred_seed2024__baseline_self_refine__msra__ATF.jsonl"
    output = pred_dir / "pred_seed2024__baseline_self_refine_sfloor__msra__ATF.jsonl"
    _write_jsonl(
        source,
        [
            {
                "tokens": ["Alice"],
                "gold_tags": ["B-PER"],
                "pred_tags": ["I-PER"],
            }
        ],
    )
    _write_jsonl(
        output,
        [
            {
                "tokens": ["keep"],
                "gold_tags": ["O"],
                "pred_tags": ["O"],
                "sentinel": True,
            }
        ],
    )
    monkeypatch.setattr(sfloor, "PRED_DIR", pred_dir)

    summary = sfloor.main([])

    assert _read_jsonl(output) == [
        {
            "tokens": ["keep"],
            "gold_tags": ["O"],
            "pred_tags": ["O"],
            "sentinel": True,
        }
    ]
    assert summary["skipped"] == 1
    assert summary["written"] == 0
