import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    def __init__(self, padding_side):
        self.padding_side = padding_side

    def __call__(self, prompts, **kwargs):
        assert prompts == ["short", "long"]
        assert kwargs == {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.padding_side == "right":
            input_ids = [[5, 6, 0], [7, 8, 9]]
            attention_mask = [[1, 1, 0], [1, 1, 1]]
        else:
            input_ids = [[0, 5, 6], [7, 8, 9]]
            attention_mask = [[0, 1, 1], [1, 1, 1]]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side
        self.grad_enabled = None

    def __call__(self, **encoded):
        self.grad_enabled = torch.is_grad_enabled()
        logits = torch.zeros((2, 3, 120), dtype=torch.float32)
        first_final = 1 if self.padding_side == "right" else 2
        first_padding = 2 if self.padding_side == "right" else 0
        logits[0, first_final, 100:102] = torch.tensor([1.0, 5.0])
        logits[0, first_padding, 100:102] = torch.tensor([99.0, 98.0])
        logits[1, 2, 102:104] = torch.tensor([4.0, 2.0])
        return SimpleNamespace(logits=logits)


class EndToEndTokenizer:
    name_or_path = probe.MODEL_ID
    vocab_size = 32
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<eos>"}
    init_kwargs = {"revision": "fake"}
    padding_side = "right"

    def __init__(self):
        self.prompt_datasets = {}
        self.batch_datasets = []
        self.batch_sizes = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        content = messages[-1]["content"]
        dataset = next(
            line.split(": ", 1)[1]
            for line in content.splitlines()
            if line.startswith("Dataset: ")
        )
        rendered = "\n".join(message["content"] for message in messages) + "\n<assistant>\n"
        self.prompt_datasets[rendered] = dataset
        return rendered

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in self.prompt_datasets:
            return [1, 2]
        for prompt in self.prompt_datasets:
            if text.startswith(prompt):
                suffix = text[len(prompt):]
                return [1, 2, 10 + int(suffix)]
        raise AssertionError("unknown fake prompt")

    def __call__(self, prompts, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
        }
        datasets = {self.prompt_datasets[prompt] for prompt in prompts}
        self.batch_datasets.append(datasets)
        self.batch_sizes.append(len(prompts))
        lengths = [2 + (index % 2) for index in range(len(prompts))]
        width = max(lengths)
        input_ids = [([1] * length) + ([0] * (width - length)) for length in lengths]
        masks = [([1] * length) + ([0] * (width - length)) for length in lengths]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
        }

    def get_vocab(self):
        return {str(code): 10 + code for code in range(9)}


class EndToEndModel:
    device = torch.device("cpu")

    class Config:
        def to_dict(self):
            return {"model_type": "fake-qwen", "vocab_size": 32}

    config = Config()

    def __call__(self, **encoded):
        batch, width = encoded["input_ids"].shape
        logits = torch.zeros((batch, width, 32), dtype=torch.float32)
        for token_id in range(10, 19):
            logits[:, :, token_id] = float(token_id)
        return SimpleNamespace(logits=logits)


class InjectedTransactionFailure(BaseException):
    pass


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


def _write_canonical_matrix(root):
    root.mkdir(parents=True, exist_ok=True)
    corrupted = {
        "tokens": ["Li", "works", "today"],
        "ner_tags": ["B-PER", "O", "O"],
        "dirty_tags": ["O", "O", "O"],
    }
    clean = {
        "tokens": ["A", "plain", "row"],
        "ner_tags": ["O", "O", "O"],
        "dirty_tags": ["O", "O", "O"],
    }
    text = json.dumps(corrupted) + "\n" + "".join(
        json.dumps(clean) + "\n" for _ in range(199)
    )
    for dataset in probe.DATASETS:
        for noise in probe.NOISE_TYPES:
            for seed in probe.SEEDS:
                path = root / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
                path.write_text(text, encoding="utf-8")
    return root


def _forbidden_loader(calls):
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("model loader must not be called")

    return load


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


@pytest.mark.parametrize("failure", ["malformed_json", "illegal_gold_iob2"])
def test_canonical_input_validation_fails_before_model_loading(tmp_path, failure):
    input_root = _write_canonical_matrix(tmp_path / "inputs")
    path = input_root / "noisy_seed13__BT__msra__N200.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    if failure == "malformed_json":
        lines[0] = "{not-json"
        message = "is not valid JSON"
    else:
        row = json.loads(lines[0])
        row["ner_tags"] = ["I-PER", "O", "O"]
        lines[0] = json.dumps(row)
        message = "illegal gold IOB2"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loader_calls = []

    with pytest.raises(ValueError, match=message):
        probe.run_probe(
            revision="a" * 40,
            backbone_tag="qwen",
            input_root=input_root,
            output_root=tmp_path / "outputs",
            model_loader=_forbidden_loader(loader_calls),
        )

    assert loader_calls == []


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_aware_next_token_logit_gather_uses_per_row_code_ids(padding_side):
    model = FakeModel(padding_side)
    values = probe.gather_code_logits(
        model,
        PaddingTokenizer(padding_side),
        ["short", "long"],
        [{"0": 100, "1": 101}, {"0": 102, "1": 103}],
    )

    assert values == [[1.0, 5.0], [4.0, 2.0]]
    assert model.grad_enabled is False


def test_gather_code_logits_rejects_mixed_codebook_widths():
    with pytest.raises(ValueError, match="same O-plus-entity code IDs"):
        probe.gather_code_logits(
            FakeModel(),
            PaddingTokenizer("right"),
            ["short", "long"],
            [{"0": 100, "1": 101}, {"0": 102, "1": 103, "2": 104}],
        )


def test_full_canonical_probe_batches_within_each_dataset_and_preserves_order(tmp_path):
    input_root = _write_canonical_matrix(tmp_path / "inputs")
    output_root = tmp_path / "outputs"
    tokenizer = EndToEndTokenizer()
    model = EndToEndModel()

    result = probe.run_probe(
        revision="a" * 40,
        backbone_tag="fake-qwen",
        input_root=input_root,
        output_root=output_root,
        batch_size=8,
        bootstrap_iterations=32,
        model_loader=lambda model_id, revision: (tokenizer, model),
    )

    assert result["records"] == 90
    assert tokenizer.batch_sizes == [8, 8, 2] * 5
    assert tokenizer.batch_datasets == [{dataset} for dataset in probe.DATASETS for _ in range(3)]
    records = [
        json.loads(line)
        for line in Path(result["jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["dataset"] for record in records] == [
        dataset for dataset in probe.DATASETS for _ in range(18)
    ]
    assert [len(record["codebook"]) for record in records] == [7] * 18 + [9] * 72
    assert {record["strongest_entity_label"] for record in records[:18]} == {"I-ORG"}
    assert {record["strongest_entity_label"] for record in records[18:]} == {"I-MISC"}
    assert {record["delta_z"] for record in records[:18]} == {6.0}
    assert {record["delta_z"] for record in records[18:]} == {8.0}


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


def test_all_bootstrap_cis_match_shared_index_direct_numpy_reference():
    values = [
        (-4.0, 8.0, 4.0),
        (-1.0, 3.0, 2.0),
        (0.5, -2.0, -1.5),
        (2.0, 1.0, 3.0),
        (7.5, -4.0, 3.5),
        (11.0, 6.0, 17.0),
    ]
    records = [
        {
            "dataset": "msra",
            "noise": "IF",
            "group": "corrupted",
            "delta_z": delta,
            "o_logit": o_logit,
            "strongest_entity_logit": entity_logit,
        }
        for delta, o_logit, entity_logit in values
    ]
    seed = 913

    aggregate = probe.aggregate_records(
        records, bootstrap_iterations=10_000, bootstrap_seed=seed
    )
    group = aggregate["groups"][0]

    digest = hashlib.sha256(f"{seed}\0msra\0IF\0corrupted".encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    columns = {
        "mean_delta_z_ci95": np.asarray([row[0] for row in values]),
        "mean_o_logit_ci95": np.asarray([row[1] for row in values]),
        "mean_strongest_entity_logit_ci95": np.asarray([row[2] for row in values]),
    }
    expected = {
        name: dict(zip(("low", "high"), map(float, np.percentile(column[indices].mean(axis=1), [2.5, 97.5]))))
        for name, column in columns.items()
    }

    assert {name: group[name] for name in expected} == expected
    assert aggregate["bootstrap"] == {
        "method": "fixed-seed percentile bootstrap of the mean",
        "iterations": 10_000,
        "seed": seed,
        "confidence": 0.95,
        "shared_resample_indices": True,
    }


def test_output_names_and_aggregate_commit_marker_validate_exact_jsonl(tmp_path):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen32b_awq-r1")
    assert jsonl_path.name == "logit_gap__qwen32b_awq-r1.jsonl"
    assert aggregate_path.name == "logit_gap__qwen32b_awq-r1__aggregate.json"

    records = [
        {"dataset": "msra", "delta_z": 1.0},
        {"dataset": "msra", "delta_z": -2.0},
    ]
    aggregate = {"groups": []}
    probe.write_outputs_atomic(tmp_path, "qwen32b_awq-r1", records, aggregate)
    assert [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()] == records
    installed_aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    marker = installed_aggregate["artifact_transaction"]
    jsonl_bytes = jsonl_path.read_bytes()
    assert installed_aggregate["groups"] == []
    assert marker == {
        "protocol": "aggregate-installed-last",
        "version": 1,
        "transaction_id": marker["transaction_id"],
        "jsonl_name": jsonl_path.name,
        "jsonl_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "jsonl_bytes": len(jsonl_bytes),
        "jsonl_line_count": 2,
        "jsonl_record_count": 2,
    }
    assert len(marker["transaction_id"]) == 32
    assert probe.validate_output_pair(jsonl_path, aggregate_path) == marker
    assert not _transaction_debris(tmp_path)

    jsonl_path.write_bytes(jsonl_bytes + b'{"tampered":true}\n')
    with pytest.raises(ValueError, match="does not match its aggregate commit marker"):
        probe.validate_output_pair(jsonl_path, aggregate_path)
    jsonl_path.write_bytes(jsonl_bytes)

    old_jsonl = jsonl_path.read_text(encoding="utf-8")
    old_aggregate = aggregate_path.read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        probe.write_outputs_atomic(tmp_path, "qwen32b_awq-r1", [{"bad": object()}], aggregate)
    assert jsonl_path.read_text(encoding="utf-8") == old_jsonl
    assert aggregate_path.read_text(encoding="utf-8") == old_aggregate


def _transaction_debris(root):
    return [path for path in root.iterdir() if path.suffix in {".tmp", ".bak"}]


def test_second_stage_failure_removes_first_stage_and_preserves_unrelated_file(tmp_path, monkeypatch):
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("untouched", encoding="utf-8")
    real_write_staged = probe._write_staged
    calls = 0

    def fail_second(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InjectedTransactionFailure("second stage failed")
        return real_write_staged(path, content)

    monkeypatch.setattr(probe, "_write_staged", fail_second)
    with pytest.raises(InjectedTransactionFailure, match="second stage failed"):
        probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    assert unrelated.read_text(encoding="utf-8") == "untouched"
    assert not (tmp_path / "logit_gap__qwen.jsonl").exists()
    assert not (tmp_path / "logit_gap__qwen__aggregate.json").exists()
    assert not _transaction_debris(tmp_path)


def test_precommit_snapshot_failure_cleans_stages_and_keeps_old_pair(tmp_path, monkeypatch):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen")
    jsonl_path.write_text('{"old":true}\n', encoding="utf-8")
    aggregate_path.write_text('{"old_marker":true}\n', encoding="utf-8")
    real_read_bytes = Path.read_bytes

    def fail_aggregate_snapshot(path):
        if path == aggregate_path:
            raise InjectedTransactionFailure("snapshot failed")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_aggregate_snapshot)
    with pytest.raises(InjectedTransactionFailure, match="snapshot failed"):
        probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    assert jsonl_path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert aggregate_path.read_text(encoding="utf-8") == '{"old_marker":true}\n'
    assert not _transaction_debris(tmp_path)


def test_commit_removes_old_marker_before_jsonl_and_installs_new_marker_last(
    tmp_path, monkeypatch
):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen")
    jsonl_path.write_text('{"old":true}\n', encoding="utf-8")
    aggregate_path.write_text('{"old_marker":true}\n', encoding="utf-8")
    real_replace = probe.os.replace
    observed = []

    def observe_transition(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == jsonl_path and destination_path.suffix == ".bak":
            assert not aggregate_path.exists()
            assert json.loads(jsonl_path.read_text(encoding="utf-8")) == {"old": True}
            observed.append("old-marker-absent-before-jsonl-backup")
        if source_path.suffix == ".tmp" and destination_path == aggregate_path:
            assert not aggregate_path.exists()
            assert [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ] == [{"new": 1}]
            observed.append("new-jsonl-present-before-marker-install")
        return real_replace(source, destination)

    monkeypatch.setattr(probe.os, "replace", observe_transition)
    probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    assert observed == [
        "old-marker-absent-before-jsonl-backup",
        "new-jsonl-present-before-marker-install",
    ]
    probe.validate_output_pair(jsonl_path, aggregate_path)


@pytest.mark.parametrize("failing_transition", [1, 2, 3, 4])
def test_every_backup_or_install_failure_restores_exact_old_pair(
    tmp_path, monkeypatch, failing_transition
):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen")
    old_jsonl = b'{"old":true}\n'
    old_aggregate = b'{"old_marker":true}\n'
    jsonl_path.write_bytes(old_jsonl)
    aggregate_path.write_bytes(old_aggregate)
    unrelated = tmp_path / "keep.bin"
    unrelated.write_bytes(b"keep")
    real_replace = probe.os.replace
    calls = 0

    def fail_transition(source, destination):
        nonlocal calls
        calls += 1
        if calls == failing_transition:
            raise InjectedTransactionFailure(f"transition {calls}")
        return real_replace(source, destination)

    monkeypatch.setattr(probe.os, "replace", fail_transition)
    with pytest.raises(InjectedTransactionFailure, match=f"transition {failing_transition}"):
        probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    assert jsonl_path.read_bytes() == old_jsonl
    assert aggregate_path.read_bytes() == old_aggregate
    assert unrelated.read_bytes() == b"keep"
    assert not _transaction_debris(tmp_path)


def test_rollback_failure_keeps_backups_and_reports_compound_failure(tmp_path, monkeypatch):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen")
    old_jsonl = b'{"old":true}\n'
    old_aggregate = b'{"old_marker":true}\n'
    jsonl_path.write_bytes(old_jsonl)
    aggregate_path.write_bytes(old_aggregate)
    real_replace = probe.os.replace

    def fail_install_and_jsonl_rollback(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".tmp" and destination_path == jsonl_path:
            raise InjectedTransactionFailure("install failed")
        if source_path.suffix == ".bak" and destination_path == jsonl_path:
            raise OSError("rollback failed")
        return real_replace(source, destination)

    monkeypatch.setattr(probe.os, "replace", fail_install_and_jsonl_rollback)
    with pytest.raises(probe.OutputTransactionError, match="rollback failed") as caught:
        probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    assert isinstance(caught.value.primary_error, InjectedTransactionFailure)
    assert not aggregate_path.exists()
    recoverable = _transaction_debris(tmp_path)
    assert any(path.read_bytes() == old_jsonl for path in recoverable)
    assert any(path.read_bytes() == old_aggregate for path in recoverable)


def test_cleanup_failure_cannot_invalidate_a_successful_commit(tmp_path, monkeypatch):
    jsonl_path, aggregate_path = probe.output_paths(tmp_path, "qwen")
    jsonl_path.write_text('{"old":true}\n', encoding="utf-8")
    aggregate_path.write_text('{"old_marker":true}\n', encoding="utf-8")
    real_unlink = Path.unlink

    def fail_backup_cleanup(path, *args, **kwargs):
        if path.suffix == ".bak":
            raise OSError("cleanup failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    probe.write_outputs_atomic(tmp_path, "qwen", [{"new": 1}], {"groups": []})

    marker = probe.validate_output_pair(jsonl_path, aggregate_path)
    assert marker["jsonl_record_count"] == 1
    assert any(path.suffix == ".bak" for path in tmp_path.iterdir())


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


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--revision", "mutable-main", "--confirm-vllm-stopped"], "40 hexadecimal"),
        (["--revision", "a" * 40], "confirm-vllm-stopped"),
        (
            ["--revision", "a" * 40, "--confirm-vllm-stopped", "--batch-size", "0"],
            "batch_size must be positive",
        ),
        (["--revision", "a" * 40, "--confirm-vllm-stopped"], "missing canonical"),
    ],
)
def test_cli_expected_errors_are_argparse_diagnostics_without_loading_or_traceback(
    tmp_path, capsys, extra_args, message
):
    loader_calls = []
    with pytest.raises(SystemExit) as caught:
        probe.main(
            extra_args
            + [
                "--backbone-tag",
                "qwen",
                "--input-root",
                str(tmp_path / "inputs"),
                "--output-root",
                str(tmp_path / "outputs"),
            ],
            model_loader=_forbidden_loader(loader_calls),
        )
    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert "error:" in captured.err
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert len(captured.err) < 2_000
    assert captured.out == ""
    assert loader_calls == []
