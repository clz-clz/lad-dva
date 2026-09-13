"""Immutable constants shared by official LAD-RG decoding and manifests."""

from __future__ import annotations

from typing import Any


OFFICIAL_PROVIDER_TIMEOUT_SECONDS = 120.0
OFFICIAL_SDK_MAX_RETRIES = 2


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
