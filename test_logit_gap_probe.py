import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import logit_gap_probe as probe


class ChatTemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "\n".join(message["content"] for message in messages) + "\n<assistant>\n"


class ContextTokenizer:
    def __init__(self, prompt, *, invalid_code=None, retokenize=False):
        self.prompt = prompt
        self.invalid_code = invalid_code
        self.retokenize = retokenize

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.prompt:
            return [10, 11]
        assert text.startswith(self.prompt)
        code = text[len(self.prompt):]
        if code == self.invalid_code:
            return [10, 11, 70, 71]
        prefix = [10, 99] if self.retokenize and code == "0" else [10, 11]
        return prefix + [100 + int(code)]


class PaddingTokenizer:
    def __call__(self, prompts, **kwargs):
        assert prompts == ["short", "long"]
        assert kwargs == {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": torch.tensor([[5, 6, 0], [7, 8, 9]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.grad_enabled = None

    def __call__(self, **encoded):
        self.grad_enabled = torch.is_grad_enabled()
        logits = torch.zeros((2, 3, 120), dtype=torch.float32)
        logits[0, 1, 100:102] = torch.tensor([1.0, 5.0])
        logits[0, 2, 100:102] = torch.tensor([99.0, 98.0])
        logits[1, 2, 100:102] = torch.tensor([4.0, 2.0])
        return SimpleNamespace(logits=logits)


def _write_matrix_file(root, row, *, dataset="msra", noise="BT", seed=13, count=1):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8")
    return path


def _small_matrix_kwargs():
    return {
        "datasets": ("msra",),
        "noise_types": ("BT",),
        "seeds": (13,),
        "expected_rows": 1,
    }


def test_codebook_and_prompt_are_deterministic_and_disable_qwen_thinking():
    codebook = probe.build_codebook("msra")
    assert list(codebook.items()) == [
        ("0", "O"),
        ("1", "B-PER"),
        ("2", "I-PER"),
        ("3", "B-LOC"),
        ("4", "I-LOC"),
        ("5", "B-ORG"),
        ("6", "I-ORG"),
    ]

    tokenizer = ChatTemplateTokenizer()
    prompt = probe.render_target_prompt(
        tokenizer,
        tokens=["Li", "Ming", "left"],
        dirty_tags=["B-PER", "O", "O"],
        target_index=1,
        dataset="msra",
        codebook=codebook,
    )

    messages, kwargs = tokenizer.calls[0]
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert "Li" in prompt and "B-PER" in prompt
    assert "Target token index: 1" in prompt
    assert '"0": "O"' in prompt and '"6": "I-ORG"' in prompt
    assert "Answer with exactly one code" in messages[-1]["content"]


def test_contextual_code_validation_accepts_exactly_one_suffix_token():
    codebook = {"0": "O", "1": "B-PER"}
    prompt = "rendered prompt\n<assistant>\n"
    tokenizer = ContextTokenizer(prompt)

    assert probe.contextual_code_token_ids(tokenizer, prompt, codebook) == {
        "0": 100,
        "1": 101,
    }


@pytest.mark.parametrize(
    "tokenizer, message",
    [
        (ContextTokenizer("p", invalid_code="1"), "exactly one next-generation token"),
        (ContextTokenizer("p", retokenize=True), "changes the rendered prompt tokenization"),
    ],
)
def test_contextual_code_validation_fails_closed(tokenizer, message):
    with pytest.raises(ValueError, match=message):
        probe.contextual_code_token_ids(tokenizer, "p", {"0": "O", "1": "B-PER"})


def test_deterministic_equal_controls_are_without_replacement(tmp_path):
    row = {
        "tokens": ["Li", "Ming", "met", "a", "friend"],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "O"],
        "dirty_tags": ["O", "O", "O", "O", "O"],
    }
    _write_matrix_file(tmp_path, row)

    first = probe.load_probe_items(tmp_path, control_seed="fixed-control-v1", **_small_matrix_kwargs())
    second = probe.load_probe_items(tmp_path, control_seed="fixed-control-v1", **_small_matrix_kwargs())

    assert first == second
    corrupted = [item for item in first if item.group == "corrupted"]
    controls = [item for item in first if item.group == "control"]
    assert len(corrupted) == len(controls) == 2
    assert len({(item.sentence_index, item.token_index) for item in controls}) == 2
    assert all(item.gold_tag == item.dirty_tag == "O" for item in controls)


def test_control_sampling_fails_when_equal_controls_cannot_be_drawn(tmp_path):
    row = {
        "tokens": ["Li", "Ming", "left"],
        "ner_tags": ["B-PER", "I-PER", "O"],
        "dirty_tags": ["O", "O", "O"],
    }
    _write_matrix_file(tmp_path, row)

    with pytest.raises(ValueError, match="needs 2 unchanged gold-O controls, found 1"):
        probe.load_probe_items(tmp_path, **_small_matrix_kwargs())


def test_canonical_matrix_paths_and_row_contract_fail_closed(tmp_path):
    paths = probe.canonical_noisy_paths(tmp_path)
    assert len(paths) == 45
    assert tmp_path / "noisy_seed2024__ATF__ontonotes5__N200.jsonl" in paths

    bad_row = {
        "tokens": ["Li", "Ming"],
        "ner_tags": ["B-UNKNOWN", "I-PER"],
        "dirty_tags": ["O"],
    }
    _write_matrix_file(tmp_path, bad_row)
    with pytest.raises(ValueError, match="equal nonempty token/gold/dirty lengths"):
        probe.load_probe_items(tmp_path, **_small_matrix_kwargs())


def test_matrix_validation_rejects_missing_extra_and_noncanonical_counts(tmp_path):
    row = {"tokens": ["x"], "ner_tags": ["O"], "dirty_tags": ["O"]}
    expected = _write_matrix_file(tmp_path, row)
    extra = tmp_path / "noisy_seed99__BT__msra__N200.jsonl"
    extra.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected noisy-data files"):
        probe.load_probe_items(tmp_path, **_small_matrix_kwargs())

    extra.unlink()
    expected.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 rows; expected 1"):
        probe.load_probe_items(tmp_path, **_small_matrix_kwargs())


def test_padding_aware_next_token_logit_gather_uses_inference_mode():
    model = FakeModel()
    values = probe.gather_code_logits(
        model,
        PaddingTokenizer(),
        ["short", "long"],
        [{"0": 100, "1": 101}, {"0": 100, "1": 101}],
    )

    assert values == [[1.0, 5.0], [4.0, 2.0]]
    assert model.grad_enabled is False


def test_delta_z_uses_strongest_entity_and_reports_its_label():
    score = probe.score_code_logits(
        {"0": "O", "1": "B-PER", "2": "I-PER"},
        [2.5, 1.0, 4.75],
    )
    assert score == {
        "o_logit": 2.5,
        "strongest_entity_logit": 4.75,
        "strongest_entity_label": "I-PER",
        "delta_z": 2.25,
    }


def test_aggregation_and_bootstrap_ci_are_deterministic():
    records = [
        {"dataset": "msra", "noise": "BT", "group": "corrupted",
         "delta_z": -1.0, "o_logit": 3.0, "strongest_entity_logit": 2.0},
        {"dataset": "msra", "noise": "BT", "group": "corrupted",
         "delta_z": 3.0, "o_logit": 1.0, "strongest_entity_logit": 4.0},
        {"dataset": "msra", "noise": "BT", "group": "control",
         "delta_z": -2.0, "o_logit": 5.0, "strongest_entity_logit": 3.0},
    ]

    first = probe.aggregate_records(records, bootstrap_iterations=1000, bootstrap_seed=77)
    second = probe.aggregate_records(records, bootstrap_iterations=1000, bootstrap_seed=77)
    assert first == second
    groups = {(g["dataset"], g["noise"], g["group"]): g for g in first["groups"]}
    corrupted = groups[("msra", "BT", "corrupted")]
    assert corrupted["count"] == 2
    assert corrupted["mean_delta_z"] == 1.0
    assert corrupted["mean_o_logit"] == 2.0
    assert corrupted["mean_strongest_entity_logit"] == 3.0
    assert corrupted["mean_delta_z_ci95"] == {"low": -1.0, "high": 3.0}
    assert groups[("msra", "BT", "control")]["mean_delta_z_ci95"] == {
        "low": -2.0,
        "high": -2.0,
    }


def test_output_names_and_atomic_write_leave_complete_json_only(tmp_path):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen32b_awq-r1")
    assert jsonl_path.name == "logit_gap__qwen32b_awq-r1.jsonl"
    assert aggregate_path.name == "logit_gap__qwen32b_awq-r1__aggregate.json"

    records = [{"dataset": "msra", "delta_z": 1.0}]
    aggregate = {"groups": []}
    probe.write_outputs_atomic(tmp_path, "qwen32b_awq-r1", records, aggregate)
    assert [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()] == records
    assert json.loads(aggregate_path.read_text(encoding="utf-8")) == aggregate
    assert not list(tmp_path.glob("*.tmp"))

    old_jsonl = jsonl_path.read_text(encoding="utf-8")
    old_aggregate = aggregate_path.read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        probe.write_outputs_atomic(tmp_path, "qwen32b_awq-r1", [{"bad": object()}], aggregate)
    assert jsonl_path.read_text(encoding="utf-8") == old_jsonl
    assert aggregate_path.read_text(encoding="utf-8") == old_aggregate


@pytest.mark.parametrize(
    "revision, tag, message",
    [
        ("abc", "qwen", "40 hexadecimal"),
        ("a" * 40, "", "BACKBONE_TAG"),
        ("a" * 40, "../escape", "safe filename component"),
    ],
)
def test_runtime_identity_validation_fails_before_loading(revision, tag, message):
    with pytest.raises(ValueError, match=message):
        probe.validate_runtime_identity(revision, tag)


def test_normal_import_is_lazy_and_does_not_import_torch_or_transformers():
    code = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "import logit_gap_probe; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    completed = subprocess.run(
        ["D:/py/Anaconda3/python.exe", "-c", code],
        cwd=Path(__file__).parent,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_validation_does_not_call_model_loader(tmp_path):
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model loader must not be called")

    with pytest.raises(ValueError, match="40 hexadecimal"):
        probe.main(
            [
                "--revision", "mutable-main",
                "--backbone-tag", "qwen",
                "--confirm-vllm-stopped",
                "--input-root", str(tmp_path / "inputs"),
                "--output-root", str(tmp_path / "outputs"),
            ],
            model_loader=forbidden_loader,
        )
    assert called is False
