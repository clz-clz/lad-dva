from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from live_backbone import (
    LiveBackboneError,
    LiveBackboneSettings,
    OpenAICompatibleLADRGAdapter,
)


PINNED_QWEN_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _response(payload: dict, **metadata) -> dict:
    metadata.setdefault("model", f"qwen@{PINNED_QWEN_REVISION}")
    metadata.setdefault("usage", {"prompt_tokens": 2, "completion_tokens": 1})
    finish_reason = metadata.pop("finish_reason", "stop")
    return {
        "choices": [{
            "message": {"content": json.dumps(payload)},
            "finish_reason": finish_reason,
        }],
        **metadata,
    }


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
    served_model = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    transport = _RecordingTransport([
        _response({"spans": [{"start": 0, "end": 2}]}, model=served_model)
    ])
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
    assert request["model"] == served_model
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
    qwen_prompt = "\n".join(message["content"] for message in request["messages"])
    assert '{"spans":[{"start":0,"end":1}]}' not in qwen_prompt
    assert "no additional keys" not in qwen_prompt.lower()
    assert adapter.provider_metadata()["revision"] == PINNED_QWEN_REVISION
    assert adapter.provider_metadata()["served_model"] == (
        f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    )


def test_vllm_rejects_response_from_bare_or_different_served_identity():
    transport = _RecordingTransport([
        _response({"spans": []}, model="Qwen/Qwen3-32B-AWQ")
    ])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=transport,
    )

    with pytest.raises(LiveBackboneError, match="served model identity"):
        adapter.ror_reasoner("span_detection", _payload())


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
    transport = _RecordingTransport([_responses_response({"spans": []})])
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
    assert transport.calls[0]["reasoning"] == {"effort": "high"}
    assert transport.calls[0]["text"]["format"]["type"] == "json_schema"
    assert adapter.settings.revision == "2026-08-01"


def test_deepseek_ror_requests_use_responses_schema_and_ontology_enum():
    transport = _RecordingTransport([
        _responses_response({"spans": [{"start": 0, "end": 1}]}),
        _responses_response({"types": [{"start": 0, "end": 1, "type": "ORG"}]}),
    ])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="deepseek",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
        ),
        transport=transport,
    )

    spans = adapter.ror_reasoner("span_detection", _payload())
    adapter.ror_reasoner("type_assignment", {**_payload(), "spans": spans["spans"]})

    span_request = transport.calls[0]
    type_request = transport.calls[1]
    assert "input" in span_request and "messages" not in span_request
    assert span_request["text"]["format"]["strict"] is True
    assert span_request["text"]["format"]["schema"]["additionalProperties"] is False
    type_schema = type_request["text"]["format"]["schema"]
    assert type_schema["properties"]["types"]["items"]["properties"]["type"]["enum"] == [
        "PER", "LOC", "ORG", "MISC"
    ]


def test_sdk_adapter_supplies_proxy_isolated_http_client_to_factory():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None)),
            responses=SimpleNamespace(create=lambda **_: None),
        )

    OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="deepseek",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
        ),
        client_factory=factory,
    )

    http_client = captured.get("http_client")
    assert http_client is not None
    try:
        assert http_client._trust_env is False
        assert http_client._transport._pool._max_connections == 1000
        assert http_client._transport._pool._max_keepalive_connections == 100
    finally:
        http_client.close()


def test_sdk_vllm_adapter_does_not_require_responses_resource():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None))
        )

    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm",
            model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1",
            api_key="test-key",
            revision=PINNED_QWEN_REVISION,
        ),
        client_factory=factory,
    )

    assert adapter.provider_metadata()["structured_api"] == (
        "chat-completions-json-schema"
    )
    adapter.close()


def test_sdk_adapter_closes_the_client_and_owned_http_transport():
    captured = {}

    class Client:
        def __init__(self, http_client):
            self.http_client = http_client
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: None)
            )
            self.responses = SimpleNamespace(create=lambda **_: None)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.http_client.close()

    def factory(**kwargs):
        captured.update(kwargs)
        captured["client"] = Client(kwargs["http_client"])
        return captured["client"]

    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="deepseek",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
        ),
        client_factory=factory,
    )

    close = getattr(adapter, "close", None)
    assert callable(close)
    close()
    close()
    assert captured["client"].close_calls == 1
    assert captured["http_client"].is_closed


def test_callback_result_retains_independent_response_metadata_without_changing_dict_api():
    transport = _RecordingTransport(
        [
            _response(
                {"spans": [{"start": 0, "end": 2}]},
                model=f"qwen@{PINNED_QWEN_REVISION}",
                system_fingerprint="fp-span",
                usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            ),
            _response(
                {"types": [{"start": 0, "end": 2, "type": "ORG"}]},
                model=f"qwen@{PINNED_QWEN_REVISION}",
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
        "status": "live",
        "provider": "vllm",
        "model": "qwen",
        "served_model": f"qwen@{PINNED_QWEN_REVISION}",
        "revision": PINNED_QWEN_REVISION,
        "response_model": f"qwen@{PINNED_QWEN_REVISION}",
        "structured_api": "chat-completions-json-schema",
        "response_status": "completed",
        "finish_reason": "stop",
        "incomplete_reason": None,
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


def test_gasd_r_requests_schema_constrains_exact_token_count_and_ontology():
    transport = _RecordingTransport([
        _response({"reason": "No entity.", "tags": ["O", "O"]})
    ])
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="qwen", base_url="http://localhost/v1",
            api_key="x", revision=PINNED_QWEN_REVISION,
        ),
        transport=transport,
    )

    adapter.gasd_reason_decoder({
        "tokens": ["No", "entity"],
        "valid_tags": ["O", "B-PER", "I-PER"],
    })

    assert transport.calls[0]["response_format"]["json_schema"]["schema"] == {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "tags"],
        "properties": {
            "reason": {"type": "string"},
            "tags": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "string", "enum": ["B-PER", "I-PER", "O"]},
            },
        },
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


def _responses_response(payload, *, status="completed", model="deepseek-v4-flash",
                        output_text=True, incomplete_reason=None):
    response = {
        "status": status,
        "model": model,
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }
    if output_text:
        response["output_text"] = json.dumps(payload)
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return response


def _deepseek_settings():
    return LiveBackboneSettings(
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        api_key="test-key",
    )


def test_deepseek_structured_request_uses_responses_json_schema_with_exact_69_items():
    transport = _RecordingTransport([
        _responses_response({"tags": ["O"] * 69}),
    ])
    adapter = OpenAICompatibleLADRGAdapter(_deepseek_settings(), transport=transport)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 69,
                "maxItems": 69,
                "items": {"type": "string", "enum": ["O", "B-PER", "I-PER"]},
            },
        },
    }

    result = adapter.structured_requester(
        "coder_path_1",
        {
            "name": "lad_rg_coder_path_1",
            "schema": schema,
            "messages": [
                {"role": "system", "content": "strict"},
                {"role": "user", "content": "label"},
            ],
            "temperature": 1.0,
            "enable_thinking": True,
        },
    )

    assert result == {"tags": ["O"] * 69}
    request = transport.calls[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["input"] == [
        {"role": "system", "content": "strict"},
        {"role": "user", "content": "label"},
    ]
    assert request["temperature"] == 1.0
    assert request["reasoning"] == {"effort": "high"}
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "lad_rg_coder_path_1",
        "strict": True,
        "schema": schema,
    }
    assert result.provider_metadata["structured_api"] == "responses-json-schema"
    assert result.provider_metadata["response_status"] == "completed"
    assert result.provider_metadata["finish_reason"] is None
    assert result.provider_metadata["incomplete_reason"] is None
    assert result.provider_metadata["response_model"] == "deepseek-v4-flash"


@pytest.mark.parametrize(
    "response, match",
    [
        (_responses_response({"tags": ["O"] * 69}, status="incomplete",
                            incomplete_reason="max_output_tokens"), "completed"),
        (_responses_response({"tags": ["O"] * 69}, status="failed"), "completed"),
        (_responses_response({"tags": ["O"] * 69}, model="deepseek-chat"), "model"),
        (_responses_response({"tags": ["O"] * 69}, output_text=False), "output_text"),
        (_responses_response({"tags": ["O"] * 68}), "exactly 69"),
        (_responses_response({"tags": ["O"] * 69 + ["B-PER"]}), "exactly 69"),
        (_responses_response({"tags": ["B-UNKNOWN"] * 69}), "enum"),
        (_responses_response({"tags": ["O"] * 69, "extra": True}), "schema"),
    ],
)
def test_deepseek_structured_request_fails_closed_for_status_identity_and_schema(
    response, match
):
    adapter = OpenAICompatibleLADRGAdapter(
        _deepseek_settings(), transport=_RecordingTransport([response])
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 69,
                "maxItems": 69,
                "items": {"type": "string", "enum": ["O", "B-PER"]},
            },
        },
    }

    with pytest.raises(LiveBackboneError, match=match):
        adapter.structured_requester(
            "capability_probe_69",
            {
                "name": "probe",
                "schema": schema,
                "messages": [{"role": "user", "content": "probe"}],
                "temperature": 0.0,
                "enable_thinking": False,
            },
        )


def test_vllm_structured_request_keeps_chat_completions_strict_schema_route():
    served_model = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    transport = _RecordingTransport([
        _response({"tags": ["O", "B-PER"]}, model=served_model),
    ])
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
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "string", "enum": ["O", "B-PER"]},
            },
        },
    }

    result = adapter.structured_requester(
        "coder_path_1",
        {
            "name": "lad_rg_coder_path_1",
            "schema": schema,
            "messages": [{"role": "user", "content": "label"}],
            "temperature": 1.0,
            "enable_thinking": True,
        },
    )

    request = transport.calls[0]
    assert result == {"tags": ["O", "B-PER"]}
    assert request["messages"] == [{"role": "user", "content": "label"}]
    assert request["temperature"] == 1.0
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "lad_rg_coder_path_1",
            "strict": True,
            "schema": schema,
        },
    }
    assert result.provider_metadata["structured_api"] == "chat-completions-json-schema"


def test_vllm_thinking_structured_request_accepts_69_tag_schema_json_from_reasoning_field():
    served_model = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    expected_tags = ["O"] * 69
    response = _response({"tags": expected_tags}, model=served_model)
    response["choices"][0]["message"] = {
        "content": None,
        "reasoning": json.dumps({"tags": expected_tags}),
    }
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=_RecordingTransport([response]),
    )
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["tags"],
        "properties": {"tags": {
            "type": "array", "minItems": 69, "maxItems": 69,
            "items": {"type": "string", "enum": ["O", "B-PER"]},
        }},
    }

    result = adapter.structured_requester("capability_probe_69", {
        "name": "contextual_capability_probe_69", "schema": schema,
        "messages": [{"role": "user", "content": "label"}],
        "temperature": 0.0, "enable_thinking": True,
    })

    assert result == {"tags": expected_tags}


def test_vllm_thinking_reasoning_json_still_enforces_schema_length():
    served_model = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    response = _response({"tags": ["O"] * 68}, model=served_model)
    response["choices"][0]["message"] = {
        "content": None,
        "reasoning": json.dumps({"tags": ["O"] * 68}),
    }
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=_RecordingTransport([response]),
    )

    with pytest.raises(LiveBackboneError, match="exactly 69 items; received 68"):
        adapter.structured_requester("capability_probe_69", {
            "name": "contextual_capability_probe_69",
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["tags"],
                "properties": {"tags": {
                    "type": "array", "minItems": 69, "maxItems": 69,
                    "items": {"type": "string", "enum": ["O", "B-PER"]},
                }},
            },
            "messages": [{"role": "user", "content": "label"}],
            "temperature": 0.0, "enable_thinking": True,
        })


def test_vllm_non_thinking_request_rejects_reasoning_only_json():
    served_model = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    response = _response({"tags": ["O"]}, model=served_model)
    response["choices"][0]["message"] = {
        "content": None,
        "reasoning": json.dumps({"tags": ["O"]}),
    }
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=_RecordingTransport([response]),
    )

    with pytest.raises(LiveBackboneError, match="content is not a JSON string"):
        adapter.structured_requester("coder_path_1", {
            "name": "lad_rg_coder_path_1",
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["tags"],
                "properties": {"tags": {
                    "type": "array", "minItems": 1, "maxItems": 1,
                    "items": {"type": "string", "enum": ["O"]},
                }},
            },
            "messages": [{"role": "user", "content": "label"}],
            "temperature": 0.0, "enable_thinking": False,
        })


@pytest.mark.parametrize(
    "response, match",
    [
        (_response({"tags": ["O"]}, finish_reason=None), "finish_reason"),
        (_response({"tags": ["O"]}, usage={}), "usage metadata"),
    ],
)
def test_vllm_structured_request_requires_terminal_finish_reason_and_usage(response, match):
    response["model"] = f"Qwen/Qwen3-32B-AWQ@{PINNED_QWEN_REVISION}"
    adapter = OpenAICompatibleLADRGAdapter(
        LiveBackboneSettings(
            provider="vllm", model="Qwen/Qwen3-32B-AWQ",
            base_url="http://127.0.0.1:8000/v1", api_key="offline-key",
            revision=PINNED_QWEN_REVISION,
        ),
        transport=_RecordingTransport([response]),
    )

    with pytest.raises(LiveBackboneError, match=match):
        adapter.structured_requester("verifier", {
            "name": "selectdenoise_verifier",
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["tags"],
                "properties": {"tags": {"type": "array", "minItems": 1,
                                            "maxItems": 1,
                                            "items": {"type": "string", "enum": ["O"]}}},
            },
            "messages": [{"role": "user", "content": "label"}],
            "temperature": 0.0,
            "enable_thinking": True,
        })
