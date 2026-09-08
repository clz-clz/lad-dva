"""Validated live structured-output callbacks for the official LAD-RG graph.

The official graph injects one cached ``structured_requester`` into Coder,
Reviewer, RoR, and GASD-R. Legacy graph paths remain independent of this
adapter and retain their existing fallback behavior.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from official_contract import (
    OFFICIAL_DECODER_CONSTANTS,
    OFFICIAL_PROVIDER_TIMEOUT_SECONDS,
    OFFICIAL_SDK_MAX_RETRIES,
)


class LiveBackboneError(RuntimeError):
    """A provider, response-schema, or live-backbone validation failure."""


class LiveBackboneResult(dict):
    """Dict-compatible callback result carrying immutable per-call evidence."""

    def __init__(self, value: Mapping[str, Any], provider_metadata: Mapping[str, Any]):
        super().__init__(value)
        self.provider_metadata = dict(provider_metadata)


@dataclass(frozen=True)
class LiveBackboneSettings:
    """Configuration for one OpenAI-compatible LAD-RG provider client."""

    provider: str
    model: str
    base_url: str
    api_key: Optional[str]
    revision: Optional[str] = None
    timeout_seconds: float = OFFICIAL_PROVIDER_TIMEOUT_SECONDS
    max_retries: int = OFFICIAL_SDK_MAX_RETRIES

    def __post_init__(self) -> None:
        provider = self.provider.lower()
        if provider not in {"deepseek", "vllm"}:
            raise ValueError("BACKBONE_PROVIDER must be 'deepseek' or 'vllm'")
        if not self.model:
            raise ValueError("BACKBONE_MODEL must not be empty")
        if not self.base_url:
            raise ValueError("BACKBONE_BASE_URL must not be empty")
        if provider == "deepseek" and self.model != "deepseek-v4-flash":
            raise ValueError("DeepSeek LAD-RG requires BACKBONE_MODEL=deepseek-v4-flash")
        if provider == "vllm" and (
            not isinstance(self.revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)
        ):
            raise ValueError("vLLM/Qwen requires an immutable 40-hex BACKBONE_REVISION")
        if self.timeout_seconds != OFFICIAL_PROVIDER_TIMEOUT_SECONDS:
            raise ValueError("live LAD-RG provider timeout must be 120 seconds")
        if not 0 <= self.max_retries <= OFFICIAL_SDK_MAX_RETRIES:
            raise ValueError("live LAD-RG SDK retries must be between 0 and 2")
        object.__setattr__(self, "provider", provider)

    @property
    def served_model(self) -> str:
        """The provider model identifier that binds vLLM requests to a revision."""
        if self.provider == "vllm":
            return f"{self.model}@{self.revision}"
        return self.model

    @property
    def structured_api(self) -> str:
        """The immutable structured-output route used by this provider."""
        return (
            "responses-json-schema"
            if self.provider == "deepseek"
            else "chat-completions-json-schema"
        )

    @classmethod
    def from_env(cls) -> "LiveBackboneSettings":
        provider = os.environ.get("BACKBONE_PROVIDER", "deepseek").lower()
        is_deepseek = provider == "deepseek"
        return cls(
            provider=provider,
            model=os.environ.get(
                "BACKBONE_MODEL",
                "deepseek-v4-flash" if is_deepseek else "Qwen/Qwen3-32B-AWQ",
            ),
            base_url=os.environ.get(
                "BACKBONE_BASE_URL",
                "https://api.deepseek.com/v1" if is_deepseek else "http://127.0.0.1:8000/v1",
            ),
            api_key=os.environ.get("BACKBONE_API_KEY")
            or (os.environ.get("DEEPSEEK_API_KEY") if is_deepseek else None),
            revision=os.environ.get("BACKBONE_REVISION"),
        )


_SPAN_SCHEMA = {
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

def _gasd_schema(token_count: int, valid_tags: set[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "tags"],
        "properties": {
            "reason": {"type": "string"},
            "tags": {
                "type": "array",
                "minItems": token_count,
                "maxItems": token_count,
                "items": {"type": "string", "enum": sorted(valid_tags)},
            },
        },
    }


def _type_schema(valid_types: Sequence[str]) -> dict[str, Any]:
    """Build the strict RoR type schema with the dataset ontology in the enum."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["types"],
        "properties": {
            "types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end", "type"],
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "type": {"type": "string", "enum": list(valid_types)},
                    },
                },
            }
        },
    }


def _validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset used by official structured requests.

    The provider supplies the first schema-enforcement layer.  This small
    local validator is deliberately independent so a provider that silently
    ignores a length or ontology constraint cannot reach the graph.
    """
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            raise LiveBackboneError(f"schema type violation at {path}: expected object")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise LiveBackboneError(f"schema required property missing at {path}: {missing[0]}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise LiveBackboneError(
                    f"schema additional property at {path}: {sorted(extra)[0]}"
                )
        for key, child_schema in properties.items():
            if key in value:
                _validate_json_schema(value[key], child_schema, f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise LiveBackboneError(f"schema type violation at {path}: expected array")
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            raise LiveBackboneError(
                f"schema array at {path} must contain exactly {minimum} items; received {len(value)}"
            )
        if maximum is not None and len(value) > maximum:
            raise LiveBackboneError(
                f"schema array at {path} must contain exactly {maximum} items; received {len(value)}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, f"{path}[{index}]")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise LiveBackboneError(f"schema type violation at {path}: expected string")
    elif expected_type == "integer":
        if type(value) is not int:
            raise LiveBackboneError(f"schema type violation at {path}: expected integer")
    elif expected_type == "number":
        if type(value) not in (int, float):
            raise LiveBackboneError(f"schema type violation at {path}: expected number")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise LiveBackboneError(f"schema enum violation at {path}")
    minimum = schema.get("minimum")
    if minimum is not None and value < minimum:
        raise LiveBackboneError(f"schema minimum violation at {path}")
    maximum = schema.get("maximum")
    if maximum is not None and value > maximum:
        raise LiveBackboneError(f"schema maximum violation at {path}")


class OpenAICompatibleLADRGAdapter:
    """Injectable OpenAI-compatible transport with strict LAD-RG validation.

    ``transport`` receives either the OpenAI ``responses.create`` or
    ``chat.completions.create`` keyword arguments and returns its response.
    Supplying it makes unit tests fully offline; otherwise the official
    OpenAI SDK client is constructed lazily.
    """

    REASON_BONUS = OFFICIAL_DECODER_CONSTANTS["gasd_reason_bonus"]

    def __init__(
        self,
        settings: Optional[LiveBackboneSettings] = None,
        *,
        transport: Optional[Callable[..., Any]] = None,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.settings = settings or LiveBackboneSettings.from_env()
        self._client: Any = None
        self._http_client: Any = None
        self._closed = False
        if transport is not None:
            self._transport = transport
            self._chat_transport = transport
            self._responses_transport = transport
            return
        try:
            from openai import DefaultHttpxClient

            if client_factory is None:
                from openai import OpenAI

                client_factory = OpenAI
            http_client = DefaultHttpxClient(trust_env=False)
            try:
                client = client_factory(
                    api_key=self.settings.api_key,
                    base_url=self.settings.base_url,
                    timeout=self.settings.timeout_seconds,
                    max_retries=self.settings.max_retries,
                    http_client=http_client,
                )
            except Exception:
                http_client.close()
                raise
            self._client = client
            self._http_client = http_client
            self._chat_transport = client.chat.completions.create
            self._responses_transport = (
                client.responses.create
                if self.settings.provider == "deepseek"
                else None
            )
            self._transport = self._chat_transport
        except Exception as exc:  # noqa: BLE001 - converted to the public error type
            raise LiveBackboneError("could not initialize live backbone client") from exc

    def close(self) -> None:
        """Release the owned SDK transport once all experiment cells finish."""
        if self._closed:
            return
        self._closed = True
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            close_client()
        elif self._http_client is not None:
            self._http_client.close()

    def provider_metadata(self) -> dict[str, Any]:
        """Stable, response-independent settings for later evidence callbacks."""
        return {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "served_model": self.settings.served_model,
            "revision": self.settings.revision,
            "base_url": self.settings.base_url,
            "timeout_seconds": self.settings.timeout_seconds,
            "max_retries": self.settings.max_retries,
            "structured_api": self.settings.structured_api,
        }

    def structured_requester(
        self, stage: str, payload: Mapping[str, Any]
    ) -> LiveBackboneResult:
        """Execute one strict structured request and return parsed JSON.

        The callback is intentionally provider-agnostic at the graph boundary:
        the graph supplies a schema, messages, temperature, and thinking flag;
        this adapter selects the provider route and preserves response evidence.
        """
        if self._closed:
            raise LiveBackboneError("live backbone adapter is closed")
        if not isinstance(stage, str) or not stage:
            raise LiveBackboneError("structured request stage must be non-empty")
        if not isinstance(payload, Mapping):
            raise LiveBackboneError("structured request payload must be a mapping")
        name = payload.get("name")
        schema = payload.get("schema")
        messages = payload.get("messages")
        temperature = payload.get("temperature")
        enable_thinking = payload.get("enable_thinking")
        if not isinstance(name, str) or not name:
            raise LiveBackboneError("structured request name must be non-empty")
        if not isinstance(schema, Mapping):
            raise LiveBackboneError("structured request schema must be a mapping")
        if (not isinstance(messages, list)
                or not all(isinstance(message, Mapping) for message in messages)):
            raise LiveBackboneError("structured request messages must be a list of mappings")
        if type(temperature) not in (int, float) or not 0.0 <= float(temperature) <= 2.0:
            raise LiveBackboneError("structured request temperature must be between 0 and 2")
        if type(enable_thinking) is not bool:
            raise LiveBackboneError("structured request enable_thinking must be boolean")

        try:
            if self.settings.provider == "deepseek":
                parsed, response_metadata = self._responses_request(
                    name=name, schema=schema, messages=list(messages),
                    temperature=float(temperature), enable_thinking=enable_thinking,
                )
            else:
                parsed, response_metadata = self._chat_request(
                    name=name, schema=schema, messages=list(messages),
                    temperature=float(temperature), enable_thinking=enable_thinking,
                )
            _validate_json_schema(parsed, schema)
            return LiveBackboneResult(
                parsed, self._stage_metadata(stage, response_metadata)
            )
        except LiveBackboneError:
            raise
        except Exception as exc:  # SDK has already applied its configured retries.
            raise LiveBackboneError("live backbone request failed after SDK retries") from exc

    def ror_reasoner(
        self, stage: str, payload: Mapping[str, Any]
    ) -> LiveBackboneResult:
        """Callback compatible with ``multi_agent_v2.ror_node``."""
        tokens = _tokens(payload)
        valid_types = _valid_types(payload)
        if stage == "span_detection":
            result = self.structured_requester(
                "ror_span_detection",
                {
                    "name": "lad_rg_span_detection",
                    "schema": _SPAN_SCHEMA,
                    "messages": _ror_messages(
                        "span_detection", payload, valid_types,
                        provider=self.settings.provider,
                    ),
                    "temperature": 0.7,
                    "enable_thinking": True,
                },
            )
            return LiveBackboneResult(
                {"spans": _validate_spans(result, len(tokens))},
                result.provider_metadata,
            )
        if stage == "type_assignment":
            requested_spans = _validate_spans({"spans": payload.get("spans")}, len(tokens))
            result = self.structured_requester(
                "ror_type_assignment",
                {
                    "name": "lad_rg_type_assignment",
                    "schema": _type_schema(valid_types),
                    "messages": _ror_messages(
                        "type_assignment", payload, valid_types,
                        provider=self.settings.provider,
                    ),
                    "temperature": 0.7,
                    "enable_thinking": True,
                },
            )
            return LiveBackboneResult(
                {"types": _validate_complete_types(
                    result, requested_spans, valid_types, len(tokens)
                )},
                result.provider_metadata,
            )
        raise LiveBackboneError(f"unsupported RoR stage: {stage!r}")

    def gasd_reason_decoder(self, payload: Mapping[str, Any]) -> LiveBackboneResult:
        """Callback compatible with ``multi_agent_v2.gasd_node``.

        The LLM supplies one ontology-valid tag per token.  Each selected tag
        receives the fixed 2.0 reason score; the existing GASD Viterbi decoder
        remains responsible for enforcing hard IOB2 transition legality.
        """
        tokens = _tokens(payload)
        valid_tags = _valid_tags(payload)
        result = self.structured_requester(
            "gasd_r",
            {
                "name": "lad_rg_gasd_r",
                "schema": _gasd_schema(len(tokens), valid_tags),
                "messages": _gasd_messages(payload, valid_tags),
                "temperature": 0.0,
                "enable_thinking": self.settings.provider == "vllm",
            },
        )
        reason, tags = _validate_gasd(result, len(tokens), valid_tags)
        return LiveBackboneResult(
            {
                "reason": reason,
                "tags": tags,
                "tag_scores": [{tag: self.REASON_BONUS} for tag in tags],
            },
            result.provider_metadata,
        )

    def _stage_metadata(
        self, stage: str, response_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "status": "live",
            "provider": self.settings.provider,
            "model": self.settings.model,
            "served_model": self.settings.served_model,
            "revision": self.settings.revision,
            "response_model": response_metadata.get("model"),
            "system_fingerprint": response_metadata.get("system_fingerprint"),
            "usage": dict(response_metadata.get("usage") or {}),
            "structured_api": response_metadata.get(
                "structured_api", self.settings.structured_api
            ),
            "response_status": response_metadata.get("response_status"),
            "finish_reason": response_metadata.get("finish_reason"),
            "incomplete_reason": response_metadata.get("incomplete_reason"),
        }

    def _responses_request(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        messages: list[dict[str, str]],
        temperature: float,
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.settings.served_model,
            "input": messages,
            "temperature": temperature,
            "timeout": self.settings.timeout_seconds,
            "reasoning": {"effort": "high" if enable_thinking else "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
        }
        try:
            raw = self._responses_transport(**request)
            response_status = _get(raw, "status")
            response_model = _get(raw, "model")
            if response_status != "completed":
                incomplete_details = _mapping_value(_get(raw, "incomplete_details"))
                reason = incomplete_details.get("reason")
                suffix = f" ({reason})" if reason else ""
                raise LiveBackboneError(
                    "official Responses response status must be completed; "
                    f"received {response_status!r}{suffix}"
                )
            if response_model != self.settings.served_model:
                raise LiveBackboneError(
                    "DeepSeek response model identity does not match the immutable request"
                )
            output_text = _get(raw, "output_text")
            if not isinstance(output_text, str):
                raise LiveBackboneError("official Responses response has no output_text")
            parsed = _json_text(output_text, "official Responses response")
            incomplete_details = _mapping_value(_get(raw, "incomplete_details"))
            return parsed, {
                "model": response_model,
                "system_fingerprint": _get(raw, "system_fingerprint"),
                "usage": _mapping_value(_get(raw, "usage")),
                "structured_api": "responses-json-schema",
                "response_status": response_status,
                "finish_reason": None,
                "incomplete_reason": incomplete_details.get("reason"),
            }
        except LiveBackboneError:
            raise
        except Exception as exc:  # SDK has already applied its configured retries.
            raise LiveBackboneError("live backbone request failed after SDK retries") from exc

    def _chat_request(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        messages: list[dict[str, str]],
        temperature: float,
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.settings.served_model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.settings.timeout_seconds,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": dict(schema)},
            },
        }
        if enable_thinking:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        try:
            raw = self._chat_transport(**request)
            response_model = _get(raw, "model")
            if response_model != self.settings.served_model:
                raise LiveBackboneError(
                    "vLLM response served model identity does not match the immutable request"
                )
            choices = _get(raw, "choices")
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
                raise LiveBackboneError("official chat response has no completion choice")
            finish_reason = _get(choices[0], "finish_reason")
            if finish_reason not in (None, "stop"):
                raise LiveBackboneError(
                    f"official chat response has unsupported finish_reason {finish_reason!r}"
                )
            return _json_response(raw), {
                "model": response_model,
                "system_fingerprint": _get(raw, "system_fingerprint"),
                "usage": _mapping_value(_get(raw, "usage")),
                "structured_api": "chat-completions-json-schema",
                "response_status": "completed",
                "finish_reason": finish_reason,
                "incomplete_reason": None,
            }
        except LiveBackboneError:
            raise
        except Exception as exc:  # SDK has already applied its configured retries.
            raise LiveBackboneError("live backbone request failed after SDK retries") from exc

    def _request(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        messages: list[dict[str, str]],
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compatibility shim for callers from the v1 adapter."""
        result = self.structured_requester(
            name,
            {
                "name": name,
                "schema": schema,
                "messages": messages,
                "temperature": 0.7,
                "enable_thinking": enable_thinking,
            },
        )
        return dict(result), dict(result.provider_metadata)


def _tokens(payload: Mapping[str, Any]) -> list[str]:
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise LiveBackboneError("payload tokens must be a list of strings")
    return tokens


def _valid_types(payload: Mapping[str, Any]) -> list[str]:
    valid_types = payload.get("valid_types")
    if not isinstance(valid_types, list) or not valid_types or not all(
        isinstance(entity_type, str) and entity_type for entity_type in valid_types
    ):
        raise LiveBackboneError("payload valid_types must be a non-empty ontology list")
    return valid_types


def _valid_tags(payload: Mapping[str, Any]) -> set[str]:
    tags = payload.get("valid_tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) for tag in tags):
        raise LiveBackboneError("payload valid_tags must be a non-empty ontology tag list")
    return set(tags)


def _json_response(raw: Any) -> dict[str, Any]:
    choices = _get(raw, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise LiveBackboneError("official response has no completion choice")
    content = _get(_get(choices[0], "message"), "content")
    if not isinstance(content, str):
        raise LiveBackboneError("official response content is not a JSON string")
    return _json_text(content, "official response")


def _json_text(content: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LiveBackboneError(f"{label} content is malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise LiveBackboneError(f"{label} must be a JSON object")
    return parsed


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _mapping_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return {}


def _validate_spans(response: Mapping[str, Any], token_count: int) -> list[dict[str, int]]:
    if set(response) != {"spans"} or not isinstance(response.get("spans"), list):
        raise LiveBackboneError("official span response must match {'spans': [{'start': int, 'end': int}]}")
    spans: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for span in response["spans"]:
        if not isinstance(span, Mapping) or set(span) != {"start", "end"}:
            raise LiveBackboneError("official span item has an invalid schema")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= token_count:
            raise LiveBackboneError("official span has invalid token bounds")
        pair = (start, end)
        if pair in seen:
            raise LiveBackboneError("official span response contains a duplicate span")
        seen.add(pair)
        spans.append({"start": start, "end": end})
    return spans


def _validate_complete_types(
    response: Mapping[str, Any],
    requested_spans: list[dict[str, int]],
    valid_types: list[str],
    token_count: int,
) -> list[dict[str, Any]]:
    if set(response) != {"types"} or not isinstance(response.get("types"), list):
        raise LiveBackboneError("official type response must match its required schema")
    requested = {(span["start"], span["end"]) for span in requested_spans}
    typed: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for item in response["types"]:
        if not isinstance(item, Mapping) or set(item) != {"start", "end", "type"}:
            raise LiveBackboneError("official type item has an invalid schema")
        start, end, entity_type = item["start"], item["end"], item["type"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= token_count:
            raise LiveBackboneError("official type item has invalid token bounds")
        if not isinstance(entity_type, str) or entity_type not in valid_types:
            raise LiveBackboneError("official type is outside the dataset ontology")
        span = (start, end)
        if span not in requested or span in seen:
            raise LiveBackboneError("official type response does not match requested spans")
        seen.add(span)
        typed.append({"start": start, "end": end, "type": entity_type})
    if seen != requested:
        raise LiveBackboneError("official type response must provide complete span typing")
    return typed


def _validate_gasd(response: Mapping[str, Any], token_count: int, valid_tags: set[str]) -> tuple[str, list[str]]:
    if set(response) != {"reason", "tags"}:
        raise LiveBackboneError("official GASD-R response has an invalid schema")
    reason, tags = response.get("reason"), response.get("tags")
    if not isinstance(reason, str) or not isinstance(tags, list):
        raise LiveBackboneError("official GASD-R response has invalid field types")
    if len(tags) != token_count:
        raise LiveBackboneError(
            f"official GASD-R response must contain exactly {token_count} tags; "
            f"received {len(tags)}"
        )
    if not all(isinstance(tag, str) and tag in valid_tags for tag in tags):
        raise LiveBackboneError("official GASD-R response uses a tag outside the dataset ontology")
    return reason, list(tags)


def _ror_messages(
    stage: str,
    payload: Mapping[str, Any],
    valid_types: list[str],
    *,
    provider: str,
) -> list[dict[str, str]]:
    json_contract = ""
    if stage == "span_detection":
        task = "Detect only entity spans using zero-based, end-exclusive token offsets."
        deepseek_json_contract = (
            'Required JSON output: {"spans":[{"start":0,"end":1}]}. '
            'The top-level object must contain exactly the key "spans"; every span item must '
            'contain exactly "start" and "end", with no additional keys. Do not include entity '
            'types, token text, names, or explanations. Use {"spans":[]} when no span exists.'
        )
    else:
        task = "Assign exactly one ontology type to every supplied zero-based, end-exclusive span."
        deepseek_json_contract = (
            'Required JSON output: {"types":[{"start":0,"end":1,"type":"PER"}]}. '
            'The top-level object must contain exactly the key "types"; every type item must '
            'contain exactly "start", "end", and "type", with no additional keys. Return every '
            'supplied span exactly once and no other spans.'
        )
    if provider == "deepseek":
        json_contract = deepseek_json_contract
    contract_line = f"\n{json_contract}" if json_contract else ""
    return [
        {
            "role": "system",
            "content": (
                "You are a strict LAD-RG RoR component. Return only the requested JSON object."
                + (f" {json_contract}" if json_contract else "")
            ),
        },
        {
            "role": "user",
            "content": (
                f"{task}{contract_line}\nOntology: {json.dumps(valid_types)}\n"
                f"Payload: {json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"
            ),
        },
    ]


def _gasd_messages(payload: Mapping[str, Any], valid_tags: set[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a strict LAD-RG GASD-R component. Return only the requested JSON object.",
        },
        {
            "role": "user",
            "content": (
                "Return exactly one ontology-valid tag per token. The downstream decoder enforces "
                "hard IOB2 transitions.\n"
                f"Valid tags: {json.dumps(sorted(valid_tags))}\n"
                f"Payload: {json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"
            ),
        },
    ]
