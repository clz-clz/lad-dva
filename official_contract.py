"""Immutable constants shared by official LAD-RG decoding and manifests."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


OFFICIAL_PROVIDER_TIMEOUT_SECONDS = 120.0
OFFICIAL_SDK_MAX_RETRIES = 2
OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES = 2
OFFICIAL_VERIFIER_SEMANTIC_RETRY_POLICY: dict[str, Any] = {
    "max_retries": OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES,
    "max_attempts": OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES + 1,
    "retryable_error": "illegal-iob2-transition-v1",
    "feedback": "first-invalid-transition-v1",
    "no_dfa": True,
    "no_fallback": True,
}


OFFICIAL_DECODER_CONSTANTS: dict[str, Any] = {
    "contract_version": "lad-rg-official-decoder-v2",
    "hard_constraint": "strict-iob2-v1",
    "gasd_beta_omega": 2.0,
    "gasd_gamma_proposal": 1.0,
    "gasd_candidate_scale_g": 1.0,
    "gasd_candidate_scale_r": 0.0,
    "gasd_reason_bonus": 2.0,
    "ror_omega_quantile": 0.6,
    "ror_confidence_threshold": 0.6,
    "voting_entity_boost": 1.0,
    "voting_consensus_ratio": 0.6,
}


def _valid_iob2_transition(previous: str, current: str) -> bool:
    if current == "O" or current.startswith("B-"):
        return True
    return (
        current.startswith("I-")
        and previous.startswith(("B-", "I-"))
        and previous[2:] == current[2:]
    )


def _tag_response_sha256(tags: Sequence[str]) -> str:
    encoded = json.dumps(
        {"tags": list(tags)}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_verifier_semantic_retry_evidence(
    records: Sequence[Mapping[str, Any]], *, exhausted: bool = False,
    interrupted: bool = False,
) -> None:
    """Validate an accepted Verifier call or its bounded semantic retry chain."""
    if exhausted and interrupted:
        raise ValueError("Verifier evidence cannot be exhausted and interrupted")
    retry_fields = (
        "semantic_attempt", "semantic_outcome", "illegal_transition",
        "rejected_tags", "response_sha256",
    )
    if len(records) == 1 and not exhausted and not interrupted:
        if any(
            key in records[0] for key in retry_fields
        ):
            raise ValueError("single-attempt Verifier evidence has retry-only fields")
        return
    expected_exhausted_attempts = OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES + 1
    if exhausted and len(records) != expected_exhausted_attempts:
        raise ValueError("exhausted Verifier evidence must contain every bounded attempt")
    if not exhausted and not 2 <= len(records) <= expected_exhausted_attempts:
        if not interrupted:
            raise ValueError("Verifier semantic retry evidence exceeds the bounded policy")
    if interrupted and not 1 <= len(records) < expected_exhausted_attempts:
        raise ValueError("interrupted Verifier evidence must be a rejected retry prefix")
    for attempt, record in enumerate(records, start=1):
        if record.get("status") != "live" or record.get("semantic_attempt") != attempt:
            raise ValueError("Verifier semantic retry attempts are not contiguous live calls")
        rejected = exhausted or interrupted or attempt < len(records)
        if rejected:
            transition = record.get("illegal_transition")
            tags = record.get("rejected_tags")
            if (
                record.get("semantic_outcome") != "rejected_illegal_iob2"
                or not isinstance(transition, Mapping)
                or set(transition) != {"index", "previous", "current"}
                or type(transition.get("index")) is not int
                or not isinstance(tags, list)
                or not tags
                or any(
                    not isinstance(tag, str)
                    or re.fullmatch(r"(?:O|[BI]-[A-Z][A-Z0-9_]*)", tag) is None
                    for tag in tags
                )
                or not 0 <= transition["index"] < len(tags)
                or not isinstance(transition.get("previous"), str)
                or not isinstance(transition.get("current"), str)
                or not isinstance(record.get("response_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["response_sha256"]) is None
            ):
                raise ValueError("Verifier rejected-attempt provenance is malformed")
            index = transition["index"]
            previous = "O" if index == 0 else tags[index - 1]
            current = tags[index]
            if (
                transition["previous"] != previous
                or transition["current"] != current
                or _valid_iob2_transition(previous, current)
                or record["response_sha256"] != _tag_response_sha256(tags)
            ):
                raise ValueError("Verifier rejected-attempt provenance is unbound")
        elif (
            record.get("semantic_outcome") != "accepted"
            or "illegal_transition" in record
            or "rejected_tags" in record
            or "response_sha256" in record
        ):
            raise ValueError("Verifier semantic retry chain lacks one accepted terminal call")


class VerifierSemanticRetryExhausted(ValueError):
    """Carry credential-free rejected-attempt evidence across atomic cell aborts."""

    def __init__(self, records: Sequence[Mapping[str, Any]],
                 context: Mapping[str, Any] | None = None):
        copied = json.loads(json.dumps(list(records), ensure_ascii=False))
        validate_verifier_semantic_retry_evidence(copied, exhausted=True)
        self.records: tuple[dict[str, Any], ...] = tuple(copied)
        self.context = (
            json.loads(json.dumps(dict(context), ensure_ascii=False))
            if context is not None else None
        )
        super().__init__(
            "official Verifier returned illegal IOB2 sequences after "
            f"{len(self.records)} attempts"
        )


class VerifierSemanticRetryInterrupted(RuntimeError):
    """Carry a rejected semantic prefix when a later nonsemantic call aborts."""

    def __init__(self, records: Sequence[Mapping[str, Any]], cause: Exception):
        copied = json.loads(json.dumps(list(records), ensure_ascii=False))
        validate_verifier_semantic_retry_evidence(copied, interrupted=True)
        self.records: tuple[dict[str, Any], ...] = tuple(copied)
        self.cause_type = type(cause).__name__
        super().__init__(
            "official Verifier retry chain was interrupted by a nonsemantic "
            f"{self.cause_type} after {len(self.records)} rejected attempts"
        )


def official_manifest_decoder_constants(
    request_timeout: float, *, structured_api: str | None = None,
    provider_timeout: float = OFFICIAL_PROVIDER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    constants = {
        "contract_version": OFFICIAL_DECODER_CONSTANTS["contract_version"],
        "hard_constraint": OFFICIAL_DECODER_CONSTANTS["hard_constraint"],
        "provider_timeout_seconds": float(provider_timeout),
        "sdk_max_retries": OFFICIAL_SDK_MAX_RETRIES,
        "runner_request_timeout_seconds": float(request_timeout),
        **{
            key: value for key, value in OFFICIAL_DECODER_CONSTANTS.items()
            if key not in {"contract_version", "hard_constraint"}
        },
    }
    if structured_api is not None:
        constants["structured_api"] = structured_api
    return constants
