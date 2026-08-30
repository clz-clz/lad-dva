"""Validated live backbone callbacks for the LAD-RG RoR and GASD-R seams.

This module deliberately does not modify the legacy graph.  Callers can inject
``ror_reasoner`` and ``gasd_reason_decoder`` into that graph when strict live
semantics are enabled by a later integration task.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from official_contract import OFFICIAL_DECODER_CONSTANTS


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
    timeout_seconds: float = 120.0
    max_retries: int = 2

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
        if self.timeout_seconds != 120.0:
            raise ValueError("live LAD-RG provider timeout must be 120 seconds")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("live LAD-RG SDK retries must be between 0 and 2")
        object.__setattr__(self, "provider", provider)

    @property
    def served_model(self) -> str:
        """The provider model identifier that binds vLLM requests to a revision."""
        if self.provider == "vllm":
            return f"{self.model}@{self.revision}"
        return self.model

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
                "https://api.deepseek.com" if is_deepseek else "http://127.0.0.1:8000/v1",
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

_TYPE_SCHEMA = {
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
                    "type": {"type": "string"},
                },
            },
        }
    },
}

_GASD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reason", "tags"],
    "properties": {
        "reason": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


class OpenAICompatibleLADRGAdapter:
    """Injectable OpenAI-compatible transport with strict LAD-RG validation.

    ``transport`` receives the OpenAI ``chat.completions.create`` keyword
    arguments and returns its response.  Supplying it makes unit tests fully
    offline; otherwise the official OpenAI SDK client is constructed lazily.
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
        if transport is not None:
            self._transport = transport
            return
        try:
            if client_factory is None:
                from openai import OpenAI

                client_factory = OpenAI
            client = client_factory(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                max_retries=self.settings.max_retries,
            )
            self._transport = client.chat.completions.create
        except Exception as exc:  # noqa: BLE001 - converted to the public error type
            raise LiveBackboneError("could not initialize live backbone client") from exc

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
        }

    def ror_reasoner(self, stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Callback compatible with ``multi_agent_v2.ror_node``."""
        tokens = _tokens(payload)
        valid_types = _valid_types(payload)
        if stage == "span_detection":
            response, response_metadata = self._request(
                name="lad_rg_span_detection",
                schema=_SPAN_SCHEMA,
                messages=_ror_messages("span_detection", payload, valid_types),
                enable_thinking=True,
            )
            return LiveBackboneResult(
                {"spans": _validate_spans(response, len(tokens))},
                self._stage_metadata("ror_span_detection", response_metadata),
            )
        if stage == "type_assignment":
            requested_spans = _validate_spans({"spans": payload.get("spans")}, len(tokens))
            response, response_metadata = self._request(
                name="lad_rg_type_assignment",
                schema=_TYPE_SCHEMA,
                messages=_ror_messages("type_assignment", payload, valid_types),
                enable_thinking=True,
            )
            return LiveBackboneResult(
                {"types": _validate_complete_types(
                    response, requested_spans, valid_types, len(tokens)
                )},
                self._stage_metadata("ror_type_assignment", response_metadata),
            )
        raise LiveBackboneError(f"unsupported RoR stage: {stage!r}")

    def gasd_reason_decoder(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Callback compatible with ``multi_agent_v2.gasd_node``.

        The LLM supplies one ontology-valid tag per token.  Each selected tag
        receives the fixed 2.0 reason score; the existing GASD Viterbi decoder
        remains responsible for enforcing hard IOB2 transition legality.
        """
        tokens = _tokens(payload)
        valid_tags = _valid_tags(payload)
        response, response_metadata = self._request(
            name="lad_rg_gasd_r",
            schema=_GASD_SCHEMA,
            messages=_gasd_messages(payload, valid_tags),
            enable_thinking=self.settings.provider == "vllm",
        )
        reason, tags = _validate_gasd(response, len(tokens), valid_tags)
        return LiveBackboneResult(
            {
                "reason": reason,
                "tags": tags,
                "tag_scores": [{tag: self.REASON_BONUS} for tag in tags],
            },
            self._stage_metadata("gasd_r", response_metadata),
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
        }

    def _request(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        messages: list[dict[str, str]],
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.settings.served_model,
            "messages": messages,
            "timeout": self.settings.timeout_seconds,
        }
        if self.settings.provider == "vllm":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": dict(schema)},
            }
            if enable_thinking:
                request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        else:
            request["response_format"] = {"type": "json_object"}
            if enable_thinking:
                request["extra_body"] = {"thinking": {"type": "enabled"}}
        try:
            raw = self._transport(**request)
            response_model = _get(raw, "model")
            if (self.settings.provider == "vllm"
                    and response_model != self.settings.served_model):
                raise LiveBackboneError(
                    "vLLM response served model identity does not match the immutable request"
                )
            return _json_response(raw), {
                "model": response_model,
                "system_fingerprint": _get(raw, "system_fingerprint"),
                "usage": _mapping_value(_get(raw, "usage")),
            }
        except LiveBackboneError:
            raise
        except Exception as exc:  # SDK has already applied its configured retries.
            raise LiveBackboneError("live backbone request failed after SDK retries") from exc


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
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LiveBackboneError("official response content is malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise LiveBackboneError("official response must be a JSON object")
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


def _ror_messages(stage: str, payload: Mapping[str, Any], valid_types: list[str]) -> list[dict[str, str]]:
    task = (
        "Detect only entity spans using zero-based, end-exclusive token offsets."
        if stage == "span_detection"
        else "Assign exactly one ontology type to every supplied zero-based, end-exclusive span."
    )
    return [
        {"role": "system", "content": "You are a strict LAD-RG RoR component. Return only the requested JSON object."},
        {
            "role": "user",
            "content": (
                f"{task}\nOntology: {json.dumps(valid_types)}\n"
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
