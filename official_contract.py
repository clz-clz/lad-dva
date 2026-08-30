"""Immutable constants shared by official LAD-RG decoding and manifests."""

from __future__ import annotations

from typing import Any


OFFICIAL_DECODER_CONSTANTS: dict[str, Any] = {
    "contract_version": "lad-rg-official-decoder-v1",
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


def official_manifest_decoder_constants(request_timeout: float) -> dict[str, Any]:
    return {
        "contract_version": OFFICIAL_DECODER_CONSTANTS["contract_version"],
        "hard_constraint": OFFICIAL_DECODER_CONSTANTS["hard_constraint"],
        "provider_timeout_seconds": 120.0,
        "sdk_max_retries": 2,
        "runner_request_timeout_seconds": float(request_timeout),
        **{
            key: value for key, value in OFFICIAL_DECODER_CONSTANTS.items()
            if key not in {"contract_version", "hard_constraint"}
        },
    }
