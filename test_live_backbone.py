from __future__ import annotations

import json

import pytest

from live_backbone import (
    LiveBackboneError,
    LiveBackboneSettings,
    OpenAICompatibleLADRGAdapter,
)


PINNED_QWEN_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _response(payload: dict, **metadata) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}], **metadata}


class _RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _payload(tokens=("Acme", "Labs")) -> dict:
    return {
        "tokens": list(tokens),
        "base_tags": ["O"] * len(tokens),
        "gated_positions": list(range(len(tokens))),
        "candidate_type_proposals": {index: "ORG" for index in range(len(tokens))},
        "valid_types": ["PER", "LOC", "ORG", "MISC"],
    }


def test_vllm_ror_uses_configured_served_name_and_retains_pinned_evidence():
    transport = _RecordingTransport([_response({"spans": [{"start": 0, "end": 2}]})])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm",
            model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1",
            api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=transport,
    )

    assert adapter.ror_reasoner("span_detection", _payload()) == {
        "spans": [{"start": 0, "end": 2}]
    }
    request = transport.calls[0]
    assert request["timeout"] == 120.0
    assert request["model"] == "Qwen/Qwen3-32B-AWQ"
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == {
        "type": "object",
        "additionalProperties": False,
        "required": ["spans"],
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                },
            }
        },
    }
    assert adapter.provider_metadata()["revision"] == PINNED_QWEN_REVISION
    assert adapter.provider_metadata()["served_model"] == (
        f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    )


def test_deepseek_settings_reject_any_model_except_deepseek_v4_flash():
    with pytest.raises(ValueError, match="deepseek-v4-flash"):
        LiveBackboneSettings(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="test-key",
        )


@pytest.mark.parametrize("revision", [None, "", "main", "v1.0", "0123456"])
def test_vllm_settings_reject_missing_or_mutable_qwen_revisions(revision):
    with pytest.raises(ValueError, match="immutable"):
        LiveBackboneSettings(
            provider="vllm",
            model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1",
            api_key="offline-key",
            revision=revision,
        )


def test_immutable_qwen_revision_is_enforced_in_served_model_evidence():
    settings = LiveBackboneSettings(
        provider="vllm",
        model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1",
        api_key="offline-key",
        revision=PINNED_QWEN_REVISION,
    )

    assert settings.served_model == f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    adapter = OpenAICompatibleLADRGAdapter(settings, transport=_RecordingTransport([]))
    assert adapter.provider_metadata()["served_model"] == settings.served_model


def test_deepseek_v4_flash_ror_enables_thinking():
    transport = _RecordingTransport([_response({"spans": []})])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="deepseek",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            revision="2026-08-01",
        ),
        transport=transport,
    )

    assert adapter.ror_reasoner("span_detection", _payload()) == {"spans": []}
    assert transport.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert adapter.settings.revision == "2026-08-01"


def test_callback_result_retains_independent_response_metadata_without_changing_dict_api():
    transport = _RecordingTransport(
        [
            _response(
                {"spans": [{"start": 0, "end": 2}]},
                model="qwen-served",
                system_fingerprint="fp-span",
                usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            ),
            _response(
                {"types": [{"start": 0, "end": 2, "type": "ORG"}]},
                model="qwen-served",
                system_fingerprint="fp-type",
                usage={"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17},
            ),
        ]
    )
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm",
            model="qwen",
            base_url="http://localhost/v1",
            api_key="x",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=transport,
    )

    span_result = adapter.ror_reasoner("span_detection", _payload())
    type_result = adapter.ror_reasoner(
        "type_assignment", {**_payload(), "spans": [{"start": 0, "end": 2}]}
    )

    assert span_result == {"spans": [{"start": 0, "end": 2}]}
    assert type_result == {"types": [{"start": 0, "end": 2, "type": "ORG"}]}
    assert span_result.provider_metadata == {
        "stage": "ror_span_detection",
        "provider": "vllm",
        "model": "qwen-served",
        "served_model": f"qwen@{PINNED_QWEN_REVISION}",
        "revision": PINNED_QWEN_REVISION,
        "system_fingerprint": "fp-span",
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    assert type_result.provider_metadata["stage"] == "ror_type_assignment"
    assert type_result.provider_metadata["system_fingerprint"] == "fp-type"
    assert type_result.provider_metadata["usage"]["total_tokens"] == 17
    assert span_result.provider_metadata is not type_result.provider_metadata


def test_type_assignment_requires_one_valid_type_for_every_requested_span():
    transport = _RecordingTransport(
        [_response({"types": [{"start": 0, "end": 1, "type": "ORG"}]})]
    )
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(provider="vllm", model="qwen", base_url="http://localhost/v1", api_key="x", revision=PINNED_QWEN_REVISION),
        transport=transport,
    )

    with pytest.raises(LiveBackboneError, match="complete span typing"):
        adapter.ror_reasoner(
            "type_assignment",
            {**_payload(), "spans": [{"start": 0, "end": 1}, {"start": 1, "end": 2}]},
        )


def test_gasd_r_validates_exact_ontology_tags_and_converts_them_to_fixed_scores():
    transport = _RecordingTransport(
        [_response({"reason": "The tokens are a location.", "tags": ["I-LOC", "I-LOC"]})]
    )
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(provider="vllm", model="qwen", base_url="http://localhost/v1", api_key="x", revision=PINNED_QWEN_REVISION),
        transport=transport,
    )

    result = adapter.gasd_reason_decoder(
        {
            "tokens": ["New", "York"],
            "base_tags": ["O", "O"],
            "candidate_paths": [["O", "O"]],
            "ror_proposals": {0: "LOC", 1: "LOC"},
            "ror_reasoning": {"source": "live"},
            "valid_tags": ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"],
            "constraint": "hard_iob2",
        }
    )

    assert result == {
        "reason": "The tokens are a location.",
        "tags": ["I-LOC", "I-LOC"],
        "tag_scores": [{"I-LOC": 2.0}, {"I-LOC": 2.0}],
    }


@pytest.mark.parametrize(
    "response",
    [
        _response({"spans": [{"start": 0, "end": 3}]}),
        _response({"spans": [{"start": True, "end": 1}]}),
        {"choices": [{"message": {"content": "not json"}}]},
        RuntimeError("transport down"),
    ],
)
def test_official_malformed_or_failed_responses_raise_live_backbone_error(response):
    transport = _RecordingTransport([response])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(provider="vllm", model="qwen", base_url="http://localhost/v1", api_key="x", revision=PINNED_QWEN_REVISION),
        transport=transport,
    )

    with pytest.raises(LiveBackboneError):
        adapter.ror_reasoner("span_detection", _payload())
