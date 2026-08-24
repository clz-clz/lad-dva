from __future__ import annotations

import json
from pathlib import Path

import pytest

import aggregate_seeds
import run_multiseed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ([], {"o_rate": 0.0, "entity_rate": 0.0}),
        ([["O", "O"]], {"o_rate": 1.0, "entity_rate": 0.0}),
        ([["B-PER", "I-PER"]], {"o_rate": 0.0, "entity_rate": 1.0}),
        ([["O", "B-PER"], ["I-PER", "O"]], {"o_rate": 0.5, "entity_rate": 0.5}),
    ],
)
def test_token_class_rates_cover_empty_and_extremes(tags, expected):
    assert aggregate_seeds.token_class_rates(tags) == expected


def test_aggregate_cell_tracks_entity_rate_and_gold_o_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_dir = tmp_path / "predictions_multiseed"
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "")
    monkeypatch.setattr(aggregate_seeds, "SEEDS", [13])

    pred_path = run_multiseed._pred_path("baseline_zero_shot", "msra", "BT", 13)
    _write_jsonl(
        pred_path,
        [
            {
                "tokens": ["Alice", "met", "Bob", "today"],
                "gold_tags": ["B-PER", "O", "B-PER", "O"],
                "pred_tags": ["B-PER", "O", "O", "O"],
            }
        ],
    )

    cell = aggregate_seeds._aggregate_cell("baseline_zero_shot", "msra", "BT")

    assert cell is not None
    assert cell["o_rate_mean"] == pytest.approx(0.75)
    assert cell["entity_rate_mean"] == pytest.approx(0.25)
    assert cell["gold_o_rate_mean"] == pytest.approx(0.5)
    assert cell["per_seed"] == [
        {
            "seed": 13,
            "precision": pytest.approx(1.0),
            "recall": pytest.approx(0.5),
            "f1": pytest.approx(2 / 3),
            "ser": pytest.approx(0.0),
            "o_rate": pytest.approx(0.75),
            "entity_rate": pytest.approx(0.25),
            "gold_o_rate": pytest.approx(0.5),
        }
    ]


def test_prf_orate_latex_contains_precision_recall_and_gold_reference():
    cells = {
        ("baseline_zero_shot", "msra", "BT"): {
            "p_mean": 0.8,
            "r_mean": 0.6,
            "f1_mean": 0.6857,
            "ser_mean": 0.0,
            "o_rate_mean": 0.7,
            "entity_rate_mean": 0.3,
            "gold_o_rate_mean": 0.55,
        }
    }

    latex = aggregate_seeds.emit_prf_orate_latex(cells, ["baseline_zero_shot"])

    assert r"\textbf{Precision}" in latex
    assert r"\textbf{Recall}" in latex
    assert r"\textbf{F1}" in latex
    assert r"\textbf{SER}" in latex
    assert r"\textbf{O-rate}" in latex
    assert r"\textit{Gold reference}" in latex
    assert "0.550" in latex
    assert "0.700" in latex
