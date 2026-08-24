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
    assert CONFIGURATIONS["lad_rg_no_lads"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_no_ror"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_no_gasd"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_lads_ror"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_lads_gasd"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_ror_gasd"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_coder_only"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_no_potentials"]["terminal_graph"] == "lad-rg"
    assert CONFIGURATIONS["lad_rg_ror_ungated"]["terminal_graph"] == "lad-rg"

    assert "terminal_graph" not in CONFIGURATIONS["selectdenoise_full"]
    assert CONFIGURATIONS["selectdenoise_contextual_lattice"]["terminal_decoder"] == "contextual-lattice-v1"


def test_oracle_pool_configs_are_registered_as_offline_only():
    assert CONFIGURATIONS["oracle_pool_sent"]["offline_only"] == "oracle_gap.py"
    assert CONFIGURATIONS["oracle_pool_sent"]["oracle_variant"] == "sentence_selection"
    assert CONFIGURATIONS["oracle_pool_tok"]["offline_only"] == "oracle_gap.py"
    assert CONFIGURATIONS["oracle_pool_tok"]["oracle_variant"] == "token_ceiling"


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


def test_run_agent_pipeline_threads_leave_one_out_and_pairwise_flags_into_lad_rg_state(
    monkeypatch,
):
    seen_states = []

    class _FakeGraph:
        async def ainvoke(self, state):
            seen_states.append(dict(state))
            return {
                "current_tags": ["B-PER"],
                "candidate_paths": [["B-PER"]],
                "rag_weights": [1.0],
            }

    monkeypatch.setattr(multi_agent_v2, "lad_rg_graph", _FakeGraph())

    for config_name in ("lad_rg_no_lads", "lad_rg_no_gasd", "lad_rg_ror_gasd"):
        asyncio.run(
            multi_agent_v2.run_agent_pipeline(
                ["Alice"],
                ["O"],
                CONFIGURATIONS[config_name],
                dataset_name="conll2003",
            )
        )

    assert [state["use_lads"] for state in seen_states] == [False, True, False]
    assert [state["use_gasd"] for state in seen_states] == [True, False, True]
    assert [state["use_ror"] for state in seen_states] == [True, True, True]


def test_reviewer_disables_lads_weighting_without_calling_llm(monkeypatch):
    called = {"n": 0}

    def _boom(_prompt):
        called["n"] += 1
        raise AssertionError("reviewer LLM should be bypassed when use_lads is false")

    class _FakeLLM:
        def invoke(self, prompt):
            return _boom(prompt)

    monkeypatch.setattr(multi_agent_v2, "llm", _FakeLLM())

    result = asyncio.run(
        multi_agent_v2.reviewer_node(
            {
                "tokens": ["Alice", "arrived"],
                "dirty_tags": ["O", "O"],
                "candidate_paths": [["O", "O"], ["B-PER", "O"], ["O", "O"]],
                "dataset_name": "conll2003",
                "use_lads": False,
            }
        )
    )

    assert called["n"] == 0
    assert result["rag_weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_gasd_can_be_disabled_without_running_decoder(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("GASD decoder should be bypassed when use_gasd is false")

    monkeypatch.setattr(multi_agent_v2, "_gasd_viterbi_decode", _boom)
    result = multi_agent_v2.gasd_node(
        _state(
            tokens=["Alice", "arrived"],
            current_tags=["I-PER", "O"],
            use_gasd=False,
        )
    )

    assert result["current_tags"] == ["B-PER", "O"]


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


def test_ror_two_stage_reasoner_detects_span_then_assigns_type_without_applying_raw_tags(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0, 1.0])
    calls = []

    def fake_reasoner(stage, payload):
        calls.append((stage, payload))
        if stage == "span_detection":
            return {"spans": [{"start": 0, "end": 2}]}
        return {"types": [{"start": 0, "end": 2, "type": "ORG"}]}

    result = multi_agent_v2.ror_node(
        _state(
            tokens=["Acme", "Labs"],
            dirty_tags=["O", "O"],
            candidate_paths=[["O", "O"], ["B-PER", "I-PER"]],
            rag_weights=[0.5, 0.5],
            ror_reasoner=fake_reasoner,
        )
    )

    assert [stage for stage, _ in calls] == ["span_detection", "type_assignment"]
    assert result["ror_proposals"] == {0: "ORG", 1: "ORG"}
    assert result["ror_reasoning"]["source"] == "callback"
    assert result["current_tags"] == ["O", "O"]


def test_ror_reasoner_malformed_response_falls_back_to_gated_candidate_vote(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0])

    result = multi_agent_v2.ror_node(
        _state(
            tokens=["Alice"],
            dirty_tags=["O"],
            candidate_paths=[["O"], ["B-PER"]],
            rag_weights=[0.5, 0.5],
            ror_reasoner=lambda stage, payload: {"unexpected": True},
        )
    )

    assert result["ror_proposals"] == {0: "PER"}
    assert result["ror_reasoning"]["source"] == "deterministic_fallback"


def test_ror_returns_legal_length_safe_fallback_when_candidate_pool_is_empty():
    result = multi_agent_v2.ror_node(
        _state(
            tokens=["Alice", "arrived"],
            dirty_tags=["I-PER"],
            candidate_paths=[],
            rag_weights=[],
            current_tags=[],
        )
    )

    assert result["current_tags"] == ["B-PER", "O"]
    assert result["ror_proposals"] == {}
    assert compute_ser([result["current_tags"]]) == 0.0


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


def test_gasd_r_reason_then_constrain_uses_callback_evidence_but_blocks_illegal_iob2():
    seen = []

    def fake_decoder(payload):
        seen.append(payload)
        return {
            "reason": "both tokens form a location",
            "tag_scores": [
                {"I-LOC": 20.0, "B-LOC": 10.0},
                {"I-LOC": 20.0},
            ],
        }

    result = multi_agent_v2.gasd_node(
        _state(
            tokens=["New", "York"],
            dirty_tags=["O", "O"],
            current_tags=["O", "O"],
            candidate_paths=[["O", "O"]],
            rag_weights=[1.0],
            ror_proposals={0: "LOC", 1: "LOC"},
            ror_reasoning={"source": "callback", "spans": [{"start": 0, "end": 2}]},
            gasd_variant="r",
            gasd_reason_decoder=fake_decoder,
        )
    )

    assert seen[0]["ror_reasoning"]["source"] == "callback"
    assert result["current_tags"] == ["B-LOC", "I-LOC"]
    assert compute_ser([result["current_tags"]]) == 0.0
    assert result["gasd_variant_used"] == "r"


def test_gasd_r_without_decoder_falls_back_safely_to_gasd_g(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0])
    result = multi_agent_v2.gasd_node(
        _state(
            tokens=["Alice"],
            dirty_tags=["O"],
            current_tags=["O"],
            candidate_paths=[["O"], ["B-PER"]],
            rag_weights=[0.5, 0.5],
            ror_proposals={0: "PER"},
            gasd_variant="r",
        )
    )

    assert result["current_tags"] == ["B-PER"]
    assert result["gasd_variant_used"] == "g_fallback"


def test_gasd_r_decoder_error_falls_back_to_gasd_g(monkeypatch):
    monkeypatch.setattr(multi_agent_v2, "_omega_weights", lambda tokens, dataset_name: [1.0])

    def failed_decoder(payload):
        raise RuntimeError("offline decoder unavailable")

    result = multi_agent_v2.gasd_node(
        _state(
            tokens=["Alice"],
            current_tags=["O"],
            candidate_paths=[["O"], ["B-PER"]],
            rag_weights=[0.5, 0.5],
            ror_proposals={0: "PER"},
            gasd_variant="r",
            gasd_reason_decoder=failed_decoder,
        )
    )

    assert result["current_tags"] == ["B-PER"]
    assert result["gasd_variant_used"] == "g_fallback"


def test_lad_rg_registers_gasd_g_r_and_both_variants():
    assert CONFIGURATIONS["lad_rg_full"]["gasd_variant"] == "g"
    assert CONFIGURATIONS["lad_rg_gasd_r"]["gasd_variant"] == "r"
    assert CONFIGURATIONS["lad_rg_gasd_both"]["gasd_variant"] == "both"


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
