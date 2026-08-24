from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import oracle_gap


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyze_cell_clips_partial_candidates_and_keeps_oracle_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pred_dir = tmp_path / "predictions_multiseed"
    noisy_dir = tmp_path / "results_multiseed"
    monkeypatch.setattr(oracle_gap, "PRED", pred_dir)
    monkeypatch.setattr(oracle_gap, "NOISY", noisy_dir)

    _write_jsonl(
        pred_dir / "pred_seed13__lad_rg_full__conll2003__BT__llama8b.jsonl",
        [
            {
                "tokens": ["Alice", "arrived"],
                "gold_tags": ["B-PER", "O"],
                "pred_tags": ["O", "O"],
                "candidate_paths": [["B-PER"], ["B-PER", "O", "B-LOC"], []],
            },
            {
                "tokens": ["Berlin"],
                "gold_tags": ["B-LOC"],
                "pred_tags": ["B-LOC"],
            },
        ],
    )
    _write_jsonl(
        noisy_dir / "noisy_seed13__BT__conll2003__N200.jsonl",
        [
            {
                "tokens": ["Alice", "arrived"],
                "ner_tags": ["B-PER", "O"],
                "dirty_tags": ["O", "O"],
            },
            {
                "tokens": ["Berlin"],
                "ner_tags": ["B-LOC"],
                "dirty_tags": ["B-LOC"],
            },
        ],
    )

    summary = oracle_gap.analyze_cell(
        dataset="conll2003",
        noise="BT",
        seed=13,
        method="lad_rg_full",
        tag="llama8b",
    )

    assert summary["status"] == "partial_candidate_evidence"
    assert summary["n_rows"] == 2
    assert summary["candidate_rows"] == 1
    assert summary["candidate_coverage"] == pytest.approx(0.5)
    assert summary["candidate_evidence"] == "partial"
    assert summary["oracle_sent"] >= summary["method"]
    assert summary["best_path"] >= summary["method"]
    assert summary["oracle_tok"] >= summary["oracle_sent"]
    assert summary["oracle_tok_legal"] == summary["oracle_tok"]


def test_oracle_token_ceiling_is_length_safe_and_hard_iob2_legal():
    result = oracle_gap._oracle_token(
        ["B-PER", "I-PER", "O"],
        [["O", "I-PER"], ["O", "I-PER", "O", "B-LOC"]],
    )

    assert result == ["O", "O", "O"]
    assert len(result) == 3


def test_analyze_cell_reports_missing_and_candidate_free_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pred_dir = tmp_path / "predictions_multiseed"
    noisy_dir = tmp_path / "results_multiseed"
    monkeypatch.setattr(oracle_gap, "PRED", pred_dir)
    monkeypatch.setattr(oracle_gap, "NOISY", noisy_dir)

    missing = oracle_gap.analyze_cell(
        dataset="conll2003",
        noise="IF",
        seed=13,
        method="lad_rg_full",
    )
    assert missing["status"] == "missing"

    _write_jsonl(
        pred_dir / "pred_seed13__lad_rg_no_ror__conll2003__IF.jsonl",
        [
            {
                "tokens": ["Alice"],
                "gold_tags": ["B-PER"],
                "pred_tags": ["O"],
            }
        ],
    )
    _write_jsonl(
        noisy_dir / "noisy_seed13__IF__conll2003__N200.jsonl",
        [
            {
                "tokens": ["Alice"],
                "ner_tags": ["B-PER"],
                "dirty_tags": ["O"],
            }
        ],
    )

    no_candidates = oracle_gap.analyze_cell(
        dataset="conll2003",
        noise="IF",
        seed=13,
        method="lad_rg_no_ror",
    )
    assert no_candidates["status"] == "no_candidate_evidence"
    assert no_candidates["candidate_coverage"] == 0.0
    assert no_candidates["candidate_evidence"] == "none"
    assert "candidate" in no_candidates["message"].lower()


def test_main_prints_clear_offline_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    pred_dir = tmp_path / "predictions_multiseed"
    noisy_dir = tmp_path / "results_multiseed"
    monkeypatch.setattr(oracle_gap, "PRED", pred_dir)
    monkeypatch.setattr(oracle_gap, "NOISY", noisy_dir)

    _write_jsonl(
        pred_dir / "pred_seed13__lad_rg_full__conll2003__ATF.jsonl",
        [
            {
                "tokens": ["Alice"],
                "gold_tags": ["B-PER"],
                "pred_tags": ["B-PER"],
                "candidate_paths": [["B-PER"]],
            }
        ],
    )
    _write_jsonl(
        noisy_dir / "noisy_seed13__ATF__conll2003__N200.jsonl",
        [
            {
                "tokens": ["Alice"],
                "ner_tags": ["B-PER"],
                "dirty_tags": ["O"],
            }
        ],
    )

    oracle_gap.main(["--dataset", "conll2003", "--noise", "ATF", "--seed", "13", "--method", "lad_rg_full"])
    out = capsys.readouterr().out

    assert "conll2003/ATF" in out
    assert "selGap" in out


def test_main_labels_partial_candidate_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    pred_dir = tmp_path / "predictions_multiseed"
    noisy_dir = tmp_path / "results_multiseed"
    monkeypatch.setattr(oracle_gap, "PRED", pred_dir)
    monkeypatch.setattr(oracle_gap, "NOISY", noisy_dir)
    rows = [
        {"tokens": ["Alice"], "gold_tags": ["B-PER"], "pred_tags": ["B-PER"],
         "candidate_paths": [["B-PER"]]},
        {"tokens": ["left"], "gold_tags": ["O"], "pred_tags": ["O"]},
    ]
    _write_jsonl(pred_dir / "pred_seed13__lad_rg_full__conll2003__BT.jsonl", rows)
    _write_jsonl(
        noisy_dir / "noisy_seed13__BT__conll2003__N200.jsonl",
        [{"tokens": row["tokens"], "ner_tags": row["gold_tags"], "dirty_tags": row["gold_tags"]}
         for row in rows],
    )

    oracle_gap.main(["--dataset", "conll2003", "--noise", "BT", "--seed", "13",
                     "--method", "lad_rg_full"])

    assert "[partial 1/2]" in capsys.readouterr().out
