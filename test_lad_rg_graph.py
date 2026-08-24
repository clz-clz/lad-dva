from __future__ import annotations

import asyncio

import pytest

import multi_agent_v2
from metrics import compute_ser
from run_multiseed import CONFIGURATIONS


def _state(**overrides):
    state = {
        "tokens": ["Alice", "Berlin"],
        "dirty_tags": ["O", "O"],
        "candidate_paths": [["O", "O"], ["B-PER", "B-LOC"]],
        "rag_weights": [0.5, 0.5],
        "current_tags": [],
        "dataset_name": "conll2003",
        "noise_type": "BT",
        "use_ror": True,
        "ror_ungated": False,
        "gasd_potentials": True,
        "ror_proposals": {},
    }
    state.update(overrides)
    return state


def test_lad_rg_configurations_select_lad_rg_graph_and_preserve_selectdenoise_configs():
    assert CONFIGURATIONS["lad_rg_full"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_no_ror"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_no_potentials"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_ror_ungated"]["terminal_graph"] == "lad-rg"

    assert "terminal_graph" not in CONFIGURATIONS["selectdenoise_full"]
    assert CONFIGURATIONS["selectdenoise_contextual_lattice"]["terminal_decoder"] == "contextual-lattice-v1"


def test_run_agent_pipeline_uses_lad_rg_graph_only_for_lad_rg_config(monkeypatch):
    calls: list[str] = []

    class _FakeGraph:
        def __init__(self, name: str):
            self.name = name

        async def ainvoke(self, state):
            calls.append(self.name)
            return {
                "current_tags": ["B-PER"],
                "candidate_paths": [["B-PER"]],
                "rag_weights": [1.0],
            }

    monkeypatch.setattr(multi_agent_v2, "multi_agent_graph", _FakeGraph("selectdenoise"))
    monkeypatch.setattr(multi_agent_v2, "lad_rg_graph", _FakeGraph("lad-rg"))

    assert asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice"],
            ["O"],
            {"terminal_graph": "lad-rg"},
            dataset_name="conll2003",
        )
    ) == ["B-PER"]
    assert asyncio.run(
        multi_agent_v2.run_agent_pipeline(
            ["Alice"],
            ["O"],
            {"use_ror": True, "gasd_potentials": True},
            dataset_name="conll2003",
        )
    ) == ["B-PER"]
    assert calls == ["lad-rg", "selectdenoise"]


def test_ror_gate_uses_omega_and_ungated_diagnostic(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0, 0.1])

    gated = multi_agent_v2.ror_node(_state())
    assert gated["current_tags"] == ["O", "O"]
    assert gated["ror_proposals"] == {0: "PER"}

    ungated = multi_agent_v2.ror_node(_state(ror_ungated=True))
    assert ungated["current_tags"] == ["O", "O"]
    assert ungated["ror_proposals"] == {0: "PER", 1: "LOC"}


def test_ror_output_is_proposals_not_raw_final_tags(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0])
    result = multi_agent_v2.ror_node(
        _state(
            tokens=["Alice"],
            dirty_tags=["O"],
            candidate_paths=[["O"], ["B-PER"]],
            rag_weights=[0.5, 0.5],
        )
    )
    assert result["ror_proposals"] == {0: "PER"}
    assert result["current_tags"] == ["O"]


def test_gasd_decodes_legal_iob2_from_candidates_potentials_and_proposals(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [0.0, 1.0])
    state = _state(
        tokens=["New", "York"],
        dirty_tags=["O", "O"],
        current_tags=["O", "O"],
        candidate_paths=[["O", "I-LOC"], ["O", "I-LOC"], ["O", "O"]],
        rag_weights=[1.0, 1.0, 1.0],
        ror_proposals={1: "LOC"},
        gasd_potentials=True,
    )

    decoded = multi_agent_v2.gasd_node(state)["current_tags"]
    assert compute_ser([decoded]) == 0.0
    assert decoded == ["O", "B-LOC"]


def test_gasd_potentials_false_removes_only_omega_not_proposal_bonus(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [10.0])
    state = _state(
        tokens=["Alice"],
        dirty_tags=["O"],
        current_tags=["O"],
        candidate_paths=[["O"], ["B-PER"]],
        rag_weights=[0.75, 0.25],
        ror_proposals={0: "PER"},
        gasd_potentials=False,
    )

    assert multi_agent_v2.gasd_node(state)["current_tags"] == ["B-PER"]


def test_gasd_falls_back_to_legal_base_without_candidates_or_on_decode_error(monkeypatch):
    no_candidates = multi_agent_v2.gasd_node(
        _state(
            tokens=["Alice", "arrived"],
            current_tags=["I-PER", "O"],
            candidate_paths=[],
            rag_weights=[],
        )
    )["current_tags"]
    assert no_candidates == ["B-PER", "O"]
    assert compute_ser([no_candidates]) == 0.0

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic decode failure")

    monkeypatch.setattr(multi_agent_v2, "_gasd_viterbi_decode", _boom)
    failed_decode = multi_agent_v2.gasd_node(
        _state(
            tokens=["Alice", "arrived"],
            current_tags=["I-PER", "O"],
            candidate_paths=[["O", "O"]],
            rag_weights=[1.0],
        )
    )["current_tags"]
    assert failed_decode == ["B-PER", "O"]
    assert compute_ser([failed_decode]) == 0.0
