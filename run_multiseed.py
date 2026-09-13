"""
run_multiseed.py — multi-seed multi-config experiment driver.

Wraps your existing `run_agent_pipeline` from multi_agent_v2.py. Does NOT
modify that pipeline; treats it as a black box.

For each (config, dataset, noise_type, seed):
  1. Reads the per-seed noisy file produced by gen_noisy.py
  2. Runs run_agent_pipeline concurrently (asyncio + semaphore)
  3. Writes predictions to:
        predictions_multiseed/pred_seed{S}__{config}__{dataset}__{noise}.jsonl
  4. Each line: {"tokens": [...], "gold_tags": [...], "pred_tags": [...]}

Resumable: cells whose prediction file already exists are skipped.

Usage:
    python run_multiseed.py                 # full matrix, real pipeline
    python run_multiseed.py --dummy         # offline; mocks run_agent_pipeline
    python run_multiseed.py --configs lad_rg_full lad_rg_gasd_r lad_rg_gasd_both
    python run_multiseed.py --seeds 13 42 2024
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from official_contract import official_manifest_decoder_constants
from request_diagnostics import sentence_context, queue_context, safe_emit as emit
from official_provider_cache import (
    PROVIDER_CACHE_SCHEMA, QWEN_MODEL, QWEN_REVISION, QWEN_SERVED_MODEL,
    read_provider_cell, write_provider_cell,
)
from live_backbone import qwen_enable_thinking_from_environment

# Silence telemetry noise from chroma / langchain
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS    = ["msra", "conll2003", "wnut17", "fewnerd", "ontonotes5"]
NOISE_TYPES = ["BT", "IF", "ATF"]
SEEDS       = [13, 42, 2024]
SAMPLE_SIZE = None  # None = full dataset
MAX_CONCURRENCY = 200     # high throughput for LLM API calls
PER_REQUEST_TIMEOUT = 180  # legacy outer deadline; failure policy controls the outcome
OFFICIAL_REQUEST_TIMEOUT = 3600.0
OFFICIAL_SAMPLE_SIZE = 200
PAID_SMOKE_SIZE = 20
OFFICIAL_NOISE_RATIO = 0.15
# The live provider cap for the confirmed Stage-A execution profile.  Keep
# this separate from the legacy/general-purpose concurrency default above so
# the provider-cache phase cannot silently drift to another launch contract.
OFFICIAL_PROVIDER_MAX_CONCURRENCY = 32
OFFICIAL_MANIFEST_SCHEMA = "lad-rg-official-run-v2"
# Staged Contextual Lattice runs use this immutable provider-evidence schema.
# Provider-cache execution is deliberately isolated from the later offline
# contextual-replay phase; legacy/end-to-end execution remains unchanged.
OFFICIAL_PROVIDER_CACHE_SCHEMA = PROVIDER_CACHE_SCHEMA
OFFICIAL_QWEN_MODEL = "Qwen/Qwen3-32B-AWQ"
PROVIDER_CACHE_DIR = Path("provider_cache")
PROVIDER_CACHE_INDEX_SCHEMA = "selectdenoise-contextual-provider-cache-index-v1"
CONTEXTUAL_REPLAY_PHASE = "contextual-replay"
_OFFICIAL_CONTEXTUAL_STAGES = frozenset({"coder", "reviewer", "verifier"})
DATASET_ENTITY_TYPES = {
    "msra": ["PER", "LOC", "ORG"],
    "conll2003": ["PER", "LOC", "ORG", "MISC"],
    "wnut17": ["PER", "LOC", "ORG", "MISC"],
    "fewnerd": ["PER", "LOC", "ORG", "MISC"],
    "ontonotes5": ["PER", "LOC", "ORG", "MISC"],
}

# Configurations — each key maps to either a pipeline config or a standalone method.
CONFIGURATIONS: Dict[str, dict] = {
    # Locked v1 terminal: existing Coder/Reviewer graph plus the frozen
    # contextual lattice replacement.  The bundle is loaded at inference time
    # from CONTEXTUAL_LATTICE_BUNDLE; no fitting occurs here.
    "selectdenoise_contextual_lattice": {
        "terminal_decoder": "contextual-lattice-v1",
        "deanchor_atf": True,
        "use_verifier": True,
    },
    # SelectDenoise: Coder(+ATF de-anchoring) -> Reviewer(LADS/DEER) -> Verifier
    "selectdenoise_full":        {"deanchor_atf": True,  "use_verifier": True},
    # Explicit rollback name for new runs; historical selectdenoise_full files
    # remain addressable and are never overwritten by the versioned config.
    "selectdenoise_full_legacy": {"deanchor_atf": True,  "use_verifier": True},
    # Ablations to attribute the gain to each lever
    "selectdenoise_no_verifier": {"deanchor_atf": True,  "use_verifier": False},
    "selectdenoise_no_deanchor": {"deanchor_atf": False, "use_verifier": True},
    "selectdenoise_verify_all":  {"deanchor_atf": True,  "use_verifier": True,
                                  "verify_all": True},
    "selectdenoise_vote":        {"deanchor_atf": False, "use_verifier": False},
    # LAD-RG: Coder -> Reviewer/LADS -> RoR proposals -> GASD global decode
    "lad_rg_full":               {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": True, "gasd_potentials": True,
                                  "gasd_variant": "g"},
    "lad_rg_gasd_r":             {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": True, "gasd_potentials": True,
                                  "gasd_variant": "r"},
    "lad_rg_gasd_both":          {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": True, "gasd_potentials": True,
                                  "gasd_variant": "both"},
    "lad_rg_no_lads":            {"terminal_graph": "lad-rg",
                                  "use_lads": False, "use_ror": True,
                                  "use_gasd": True,  "gasd_potentials": False},
    "lad_rg_no_ror":             {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": False,
                                  "use_gasd": True, "gasd_potentials": True},
    "lad_rg_no_gasd":            {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": False, "gasd_potentials": True},
    "lad_rg_lads_ror":           {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": False, "gasd_potentials": True},
    "lad_rg_lads_gasd":          {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": False,
                                  "use_gasd": True, "gasd_potentials": True},
    "lad_rg_ror_gasd":           {"terminal_graph": "lad-rg",
                                  "use_lads": False, "use_ror": True,
                                  "use_gasd": True,  "gasd_potentials": False},
    "lad_rg_coder_only":         {"terminal_graph": "lad-rg",
                                  "use_lads": False, "use_ror": False,
                                  "use_gasd": False, "gasd_potentials": False},
    "lad_rg_no_potentials":      {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": True, "gasd_potentials": False},
    "lad_rg_ror_ungated":        {"terminal_graph": "lad-rg",
                                  "use_lads": True, "use_ror": True,
                                  "use_gasd": True, "gasd_potentials": True,
                                  "ror_ungated": True},
    # Baselines (2024-2025 published & standard)
    "baseline_zero_shot":         {"method": "zero_shot"},
    "baseline_cot_reasoning":     {"method": "cot_reasoning"},
    "baseline_self_refine":       {"method": "self_refine"},
    "baseline_rule_only":         {"method": "rule_only"},
    "baseline_standard_prompting": {"method": "standard_prompting"},
    # Offline-derived post-hoc methods. These are registered here so downstream
    # tooling can refer to them by explicit key, but the runner itself only
    # emits a clear error telling the user which offline utility to use.
    "baseline_zero_shot_sfloor": {
        "offline_only": "apply_structural_floor_to_baselines.py",
        "source_method": "baseline_zero_shot",
    },
    "baseline_cot_reasoning_sfloor": {
        "offline_only": "apply_structural_floor_to_baselines.py",
        "source_method": "baseline_cot_reasoning",
    },
    "baseline_self_refine_sfloor": {
        "offline_only": "apply_structural_floor_to_baselines.py",
        "source_method": "baseline_self_refine",
    },
    "baseline_standard_prompting_sfloor": {
        "offline_only": "apply_structural_floor_to_baselines.py",
        "source_method": "baseline_standard_prompting",
    },
    "oracle_pool_sent": {
        "offline_only": "oracle_gap.py",
        "oracle_variant": "sentence_selection",
        "source_method": "candidate-evidence prediction files",
    },
    "oracle_pool_tok": {
        "offline_only": "oracle_gap.py",
        "oracle_variant": "token_ceiling",
        "source_method": "candidate-evidence prediction files",
    },
}

NOISY_DIR = Path("results_multiseed")
PRED_DIR  = Path("predictions_multiseed")
DEFAULT_CONFIGS = ["selectdenoise_contextual_lattice"]

# Short tag namespacing prediction files per backbone, so a cross-backbone run
# writes new files instead of being skipped by the resume check. Unset (the
# default) reproduces the original, tag-free filenames byte for byte, keeping
# every existing prediction addressable. Set it alongside BACKBONE_MODEL /
# BACKBONE_BASE_URL / BACKBONE_API_KEY (read in multi_agent_v2.py), e.g.
#     BACKBONE_TAG=llama8b python run_multiseed.py --configs selectdenoise_full
# Downstream scripts (aggregate_seeds, derive_ablations, sweep_results) import
# _pred_path, so they follow the same namespace automatically.
BACKBONE_TAG = os.environ.get("BACKBONE_TAG", "").strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _endpoint_origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("BACKBONE_BASE_URL must be an absolute HTTP(S) URL")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}"


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("openai", "langgraph", "langchain-openai", "datasets",
                    "numpy", "seqeval"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _official_settings_from_env(config_names: list[str], environment: Mapping[str, str]):
    """Validate explicit launch variables before constructing the live adapter."""
    required = ["BACKBONE_PROVIDER", "BACKBONE_MODEL", "BACKBONE_BASE_URL", "BACKBONE_TAG"]
    missing = [name for name in required if not str(environment.get(name, "")).strip()]
    provider = str(environment.get("BACKBONE_PROVIDER", "")).strip().lower()
    key = str(environment.get("BACKBONE_API_KEY", "")).strip()
    if provider == "deepseek" and not key:
        key = str(environment.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        missing.append("BACKBONE_API_KEY (or provider-appropriate key)")
    if missing:
        raise ValueError("official mode requires explicit " + ", ".join(missing))
    tag = str(environment["BACKBONE_TAG"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("BACKBONE_TAG must contain only letters, numbers, dot, underscore, or hyphen")
    model = str(environment["BACKBONE_MODEL"]).strip()
    revision = str(environment.get("BACKBONE_REVISION", "")).strip() or None
    if provider == "deepseek":
        if model != "deepseek-v4-flash":
            raise ValueError("DeepSeek official runs require BACKBONE_MODEL=deepseek-v4-flash")
    elif provider == "vllm":
        if model != OFFICIAL_QWEN_MODEL:
            raise ValueError(f"vLLM official runs require BACKBONE_MODEL={OFFICIAL_QWEN_MODEL}")
        if not revision:
            raise ValueError("Qwen official runs require immutable BACKBONE_REVISION")
    from live_backbone import LiveBackboneSettings
    enable_thinking = qwen_enable_thinking_from_environment(environment)
    settings = LiveBackboneSettings(
        provider=provider,
        model=model,
        base_url=str(environment["BACKBONE_BASE_URL"]).strip(),
        api_key=key,
        revision=revision,
        timeout_seconds=(float(environment.get("QWEN_PROVIDER_TIMEOUT_SECONDS", "300"))
                         if provider == 'vllm' else 120.0),
        enable_thinking=enable_thinking,
    )
    _endpoint_origin(settings.base_url)
    _validate_official_settings_for_configs(settings, config_names)
    if (settings.enable_thinking is False
            and any(_official_profile(CONFIGURATIONS[name]) == "contextual-lattice"
                    for name in config_names)
            and "nothink" not in tag.lower()):
        raise ValueError("QWEN_ENABLE_THINKING=false requires a fresh BACKBONE_TAG containing 'nothink'")
    return settings, tag


def _official_profile(config: Mapping[str, Any]) -> str:
    if config.get("terminal_graph") == "lad-rg":
        return "lad-rg"
    if config.get("terminal_decoder") == "contextual-lattice-v1":
        return "contextual-lattice"
    raise ValueError("official mode requires a LAD-RG or contextual-lattice config")


def _validate_official_settings_for_configs(settings, config_names: Sequence[str]) -> None:
    for config_name in config_names:
        config = CONFIGURATIONS.get(config_name)
        if config is None:
            raise ValueError(f"official mode requires a known config, got {config_name!r}")
        _official_profile(config)
    if settings.provider == "vllm" and settings.model != OFFICIAL_QWEN_MODEL:
        raise ValueError(f"vLLM official runs require BACKBONE_MODEL={OFFICIAL_QWEN_MODEL}")
    if settings.provider == "deepseek" and any(
        str(CONFIGURATIONS[name].get("gasd_variant", "g")).lower() in {"r", "both"}
        for name in config_names
    ):
        raise ValueError("DeepSeek official runs reject GASD-R/Both; use vllm/Qwen")
    if any(_official_profile(CONFIGURATIONS[name]) == "contextual-lattice"
           for name in config_names):
        if (settings.provider != "vllm" or settings.model != QWEN_MODEL
                or settings.revision != QWEN_REVISION
                or settings.served_model != QWEN_SERVED_MODEL):
            raise ValueError("official contextual runs require the pinned Qwen served identity")


def _configure_official_request_model(settings) -> str:
    """Bind Coder/Reviewer imports to the official identity and SDK controls."""
    os.environ["BACKBONE_SERVED_MODEL"] = settings.served_model
    os.environ["LAD_RG_OFFICIAL_REQUESTS"] = "1"
    return settings.served_model


def _build_official_manifest(settings, tag: str, git_sha: str,
                             request_timeout: float) -> dict[str, Any]:
    return {
        "schema_version": OFFICIAL_MANIFEST_SCHEMA,
        "git_sha": git_sha,
        "backbone": {
            "provider": settings.provider,
            "model": settings.model,
            "served_model": settings.served_model,
            "revision": settings.revision,
            "structured_api": settings.structured_api,
            "enable_thinking": settings.configured_enable_thinking,
            "thinking_mode": settings.thinking_mode,
            "immutable_revision": settings.revision or settings.model,
            "endpoint_origin": _endpoint_origin(settings.base_url),
            "tag": tag,
        },
        "protocol": {
            "sample_size": OFFICIAL_SAMPLE_SIZE,
            "records_per_cell": OFFICIAL_SAMPLE_SIZE,
            "noise_ratio": OFFICIAL_NOISE_RATIO,
            "seeds": list(SEEDS),
            "datasets": list(DATASETS),
            "noise_types": list(NOISE_TYPES),
        },
        "dependencies": _dependency_versions(),
        "decoder_constants": official_manifest_decoder_constants(
            request_timeout, structured_api=settings.structured_api,
            provider_timeout=settings.timeout_seconds
        ),
    }


def _manifest_path(pred_dir: Path, tag: str) -> Path:
    return pred_dir / f"run_manifest__{tag}.json"


def _ensure_official_manifest(expected: Mapping[str, Any], pred_dir: Path,
                              tag: str) -> Path:
    manifest_path = _manifest_path(pred_dir, tag)
    outputs = list(pred_dir.glob(f"pred_*__{tag}.jsonl")) if pred_dir.exists() else []
    if not manifest_path.exists():
        if outputs:
            raise RuntimeError(f"official resume rejected: missing manifest {manifest_path.name}")
        pred_dir.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(_canonical_json(expected) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
        return manifest_path
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"official manifest is unreadable: {manifest_path}") from exc
    if _canonical_json(actual) != _canonical_json(expected):
        raise RuntimeError(f"official manifest is incompatible: {manifest_path.name}")
    return manifest_path


def _valid_tags(dataset: str) -> set[str]:
    return {"O"} | {f"{prefix}-{kind}" for kind in DATASET_ENTITY_TYPES[dataset]
                    for prefix in ("B", "I")}


def _is_legal_iob2(tags: list[str]) -> bool:
    previous = "O"
    for tag in tags:
        if tag.startswith("I-"):
            kind = tag[2:]
            if previous not in {f"B-{kind}", f"I-{kind}"}:
                return False
        previous = tag
    return True


def _validate_official_prediction(
    path: Path, dataset: str, config_name: str, noise_type: str,
    expected_count: int = OFFICIAL_SAMPLE_SIZE,
    provider_identity: Optional[Mapping[str, Any]] = None,
) -> None:
    rows = _load_noisy(path)
    if len(rows) != expected_count:
        raise RuntimeError(f"official prediction has {len(rows)} records, expected {expected_count}: {path.name}")
    config = CONFIGURATIONS[config_name]
    profile = _official_profile(config)
    expected_variant = str(config.get("gasd_variant", "g")).lower()
    expected_used = "disabled" if not config.get("use_gasd", True) else expected_variant
    valid = _valid_tags(dataset)
    for index, row in enumerate(rows):
        tokens, gold, pred = row.get("tokens"), row.get("gold_tags"), row.get("pred_tags")
        if (not isinstance(tokens, list) or not isinstance(gold, list) or not isinstance(pred, list)
                or not all(isinstance(item, str) for values in (tokens, gold, pred) for item in values)
                or not len(tokens) == len(gold) == len(pred)
                or any(tag not in valid for tag in gold + pred) or not _is_legal_iob2(pred)):
            raise RuntimeError(f"official prediction row {index} is structurally incompatible: {path.name}")
        if profile == "contextual-lattice":
            anchor = row.get("terminal_anchor_tags")
            model_hash = row.get("terminal_model_hash")
            if (not isinstance(anchor, list)
                    or len(anchor) != len(tokens)
                    or any(tag not in valid for tag in anchor)
                    or not _is_legal_iob2(anchor)
                    or not isinstance(model_hash, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", model_hash)
                    or row.get("fallback_used") is not False
                    or not isinstance(row.get("provider_metadata"), dict)):
                raise RuntimeError(f"official contextual row {index} lacks valid terminal evidence: {path.name}")
            if provider_identity is not None:
                _validate_official_provider_evidence(
                    config, row["provider_metadata"], provider_identity,
                    None, noise_type=noise_type,
                )
            continue
        if (row.get("ror_reasoning_source") not in {"not_triggered", "live", "disabled"}
                or row.get("gasd_variant_requested") != expected_variant
                or row.get("gasd_variant_used") != expected_used
                or row.get("fallback_used") is not False
                or not isinstance(row.get("provider_metadata"), dict)):
            raise RuntimeError(f"official prediction row {index} lacks compatible launch evidence: {path.name}")
        if provider_identity is not None:
            _validate_official_provider_evidence(
                config, row["provider_metadata"], provider_identity,
                row["ror_reasoning_source"], noise_type=noise_type,
            )


def _validate_official_provider_evidence(
    config: Mapping[str, Any], evidence: Any, identity: Mapping[str, Any],
    ror_source: str, *, noise_type: str,
) -> None:
    """Require justified stage statuses and immutable provider identity."""
    if not isinstance(evidence, Mapping):
        raise RuntimeError("official provider evidence must be stage-separated")
    if _official_profile(config) == "contextual-lattice":
        _validate_contextual_provider_evidence(evidence, identity, noise_type)
        return
    if set(evidence) != {"coder", "reviewer", "ror", "gasd"}:
        raise RuntimeError("official provider evidence must contain exactly four stages")
    expected_identity = {
        key: identity.get(key)
        for key in ("provider", "model", "served_model", "revision", "structured_api")
    }

    provider = expected_identity["provider"]
    model = expected_identity["model"]
    served_model = expected_identity["served_model"]
    revision = expected_identity["revision"]
    if provider == "vllm":
        if (model != OFFICIAL_QWEN_MODEL
                or not isinstance(revision, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
                or served_model != f"{model}@{revision}"):
            raise RuntimeError("official provider identity is not immutable vLLM/Qwen")
    elif provider == "deepseek":
        if (model != "deepseek-v4-flash" or served_model != model
                or expected_identity["structured_api"] != "responses-json-schema"):
            raise RuntimeError("official provider identity is not exact DeepSeek")
    else:
        raise RuntimeError("official provider identity has an invalid provider")
    if (provider == "vllm"
            and expected_identity["structured_api"] != "chat-completions-json-schema"):
        raise RuntimeError("official provider identity has an invalid vLLM structured API")

    def records(stage: str, allowed_stages: set[str]) -> list[Mapping[str, Any]]:
        value = evidence.get(stage)
        if (not isinstance(value, list) or not value
                or not all(isinstance(record, Mapping) for record in value)):
            raise RuntimeError(f"official provider evidence is missing stage {stage}")
        for record in value:
            if record.get("stage") not in allowed_stages:
                raise RuntimeError(f"official provider evidence has an invalid stage for {stage}")
            if any(record.get(key) != expected for key, expected in expected_identity.items()):
                raise RuntimeError(f"official provider evidence identity mismatch for {stage}")
            status = record.get("status")
            if status == "live":
                if record.get("response_model") != expected_identity["served_model"]:
                    raise RuntimeError(
                        f"official provider response identity mismatch for {stage}"
                    )
                if record.get("response_status") != "completed":
                    raise RuntimeError(
                        f"official provider response status is not completed for {stage}"
                    )
            elif record.get("response_model") is not None:
                raise RuntimeError(f"non-live official evidence cannot claim a response for {stage}")
        return value

    coder_path_indexes = {
        "BT": (1, 2, 5),
        "IF": (1, 2, 5),
        "ATF": (1, 2, 3, 4, 5),
    }.get(noise_type)
    if coder_path_indexes is None:
        raise RuntimeError(f"unsupported official noise type: {noise_type!r}")
    coder_stages = {f"coder_path_{index}" for index in coder_path_indexes}
    coder = records("coder", coder_stages)
    if ({record.get("stage") for record in coder} != coder_stages
            or len(coder) != len(coder_stages)
            or any(record.get("status") != "live" for record in coder)):
        raise RuntimeError("official Coder provider evidence must be live")

    reviewer = records("reviewer", {"reviewer"})
    reviewer_allowed = {"live", "skipped_identical"}
    if not config.get("use_lads", True):
        reviewer_allowed.add("disabled")
    if (len(reviewer) != 1
            or any(record.get("status") not in reviewer_allowed for record in reviewer)):
        raise RuntimeError("official Reviewer provider evidence status is unjustified")

    ror = records("ror", {"ror", "ror_span_detection", "ror_type_assignment"})
    expected_ror_status = (
        "disabled" if not config.get("use_ror", True) else ror_source
    )
    if expected_ror_status == "live":
        ror_stages = [record.get("stage") for record in ror]
        if (any(record.get("status") != "live" for record in ror)
                or ror_stages not in (["ror_span_detection"],
                                      ["ror_span_detection", "ror_type_assignment"])):
            raise RuntimeError("official RoR live claim lacks matching provider evidence")
    elif (len(ror) != 1 or ror[0].get("stage") != "ror"
          or ror[0].get("status") != expected_ror_status):
        raise RuntimeError("official RoR skipped status is inconsistent with configuration")

    gasd = records("gasd", {"gasd", "gasd_r"})
    variant = str(config.get("gasd_variant", "g")).lower()
    expected_gasd_status = (
        "disabled" if not config.get("use_gasd", True)
        else ("live" if variant in {"r", "both"} else "local")
    )
    if expected_gasd_status == "live":
        if (len(gasd) != 1 or gasd[0].get("status") != "live"
                or gasd[0].get("stage") != "gasd_r"):
            raise RuntimeError("official GASD-R live claim lacks matching provider evidence")
    elif (len(gasd) != 1 or gasd[0].get("stage") != "gasd"
          or gasd[0].get("status") != expected_gasd_status):
        raise RuntimeError("official GASD status is inconsistent with configuration")


def _validate_contextual_provider_evidence(
    evidence: Mapping[str, Any], identity: Mapping[str, Any], noise_type: str,
) -> None:
    """Validate the immutable three-stage Contextual Lattice provider contract."""
    if set(evidence) != _OFFICIAL_CONTEXTUAL_STAGES:
        raise RuntimeError("official contextual evidence must contain exactly coder, reviewer, and verifier")
    expected_identity = {
        "provider": "vllm", "model": QWEN_MODEL,
        "served_model": QWEN_SERVED_MODEL, "revision": QWEN_REVISION,
        "structured_api": "chat-completions-json-schema",
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("official contextual provider identity is not pinned Qwen")
    expected_thinking = identity.get("enable_thinking")
    enforce_thinking_identity = type(expected_thinking) is bool
    expected_mode = "thinking" if expected_thinking else "nothink"

    def records(stage: str) -> list[Mapping[str, Any]]:
        value = evidence.get(stage)
        if (not isinstance(value, list) or not value
                or not all(isinstance(record, Mapping) for record in value)):
            raise RuntimeError(f"official contextual evidence is missing {stage}")
        for record in value:
            if record.get("stage") != stage:
                raise RuntimeError(f"official contextual evidence has an invalid {stage} record")
            if any(record.get(key) != expected for key, expected in expected_identity.items()):
                raise RuntimeError(f"official contextual evidence identity mismatch for {stage}")
            if enforce_thinking_identity and (
                    type(record.get("enable_thinking")) is not bool
                    or record.get("enable_thinking") != expected_thinking
                    or record.get("thinking_mode") != expected_mode):
                raise RuntimeError(f"official contextual {stage} thinking-mode identity mismatch")
            if record.get("status") == "live":
                if (record.get("response_model") != QWEN_SERVED_MODEL
                        or record.get("response_status") != "completed"
                        or record.get("finish_reason") != "stop"
                        or not isinstance(record.get("usage"), Mapping)
                        or not record["usage"]):
                    raise RuntimeError(f"official contextual {stage} live evidence is incomplete")
            elif (record.get("response_model") is not None
                  or record.get("response_status") is not None
                  or record.get("finish_reason") is not None):
                raise RuntimeError(f"official contextual {stage} skipped evidence claims a response")
        return value

    expected_coder_count = {"BT": 3, "IF": 3, "ATF": 5}.get(noise_type)
    if expected_coder_count is None:
        raise RuntimeError(f"unsupported official noise type: {noise_type!r}")
    coder = records("coder")
    if len(coder) != expected_coder_count or any(record.get("status") != "live" for record in coder):
        raise RuntimeError("official contextual Coder evidence must be live")

    reviewer = records("reviewer")
    if (len(reviewer) != 1 or reviewer[0].get("status")
            not in {"live", "skipped_identical"}):
        raise RuntimeError("official contextual Reviewer evidence status is unjustified")
    verifier = records("verifier")
    if (len(verifier) != 1 or verifier[0].get("status")
            not in {"live", "skipped_uncontested", "skipped_identical"}):
        raise RuntimeError("official contextual Verifier evidence status is unjustified")


# ---------------------------------------------------------------------------
# Pipeline import (real or mock)
# ---------------------------------------------------------------------------

def _import_pipeline(dummy: bool):
    """Return (default_pipeline, baseline_pipelines_dict).

    default_pipeline: async fn for SelectDenoise and opt-in LAD-RG configurations.
    baseline_pipelines: dict mapping method_name -> async fn for standalone baselines.
    """
    from utils import enforce_iob2_syntax

    # Non-LLM baselines — always available
    async def _baseline_rule_only(tokens, dirty_tags, config=None, dataset_name=None):
        _DS_TYPES = {
            "msra": {"PER", "LOC", "ORG"},
            "conll2003": {"PER", "LOC", "ORG", "MISC"},
            "wnut17": {"PER", "LOC", "ORG", "MISC"},
            "fewnerd": {"PER", "LOC", "ORG", "MISC"},
            "ontonotes5": {"PER", "LOC", "ORG", "MISC"},
        }
        valid = _DS_TYPES.get(dataset_name or "conll2003", {"PER", "LOC", "ORG"})
        return enforce_iob2_syntax(list(dirty_tags), valid_entity_types=valid)

    baseline_pipelines = {
        "rule_only":  _baseline_rule_only,
    }

    if dummy:
        async def _mock_cot_reasoning(tokens, dirty_tags, config=None, dataset_name=None):
            import random as _rnd
            gold = config.get("__gold__", dirty_tags) if config else dirty_tags
            seed = config.get("__seed__", 0) if config else 0
            rate = 0.35
            rng = _rnd.Random(seed)
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            return out

        async def _mock_self_refine(tokens, dirty_tags, config=None, dataset_name=None):
            import random as _rnd
            gold = config.get("__gold__", dirty_tags) if config else dirty_tags
            seed = config.get("__seed__", 0) if config else 0
            rate = 0.45
            rng = _rnd.Random(seed)
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            return out

        async def _mock_zero_shot(tokens, dirty_tags, config=None, dataset_name=None):
            import random as _rnd
            gold = config.get("__gold__", dirty_tags) if config else dirty_tags
            seed = config.get("__seed__", 0) if config else 0
            rate = 0.30  # Zero-shot: no examples, weakest LLM baseline
            rng = _rnd.Random(seed)
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            return out

        baseline_pipelines["cot_reasoning"] = _mock_cot_reasoning
        baseline_pipelines["self_refine"]   = _mock_self_refine
        baseline_pipelines["zero_shot"]     = _mock_zero_shot

        async def _mock_standard_prompting(tokens, dirty_tags, config=None, dataset_name=None):
            import random as _rnd
            gold = config.get("__gold__", dirty_tags) if config else dirty_tags
            seed = config.get("__seed__", 0) if config else 0
            rate = 0.20  # Standard prompting: minimal prompt, no IOB2 rules, weak baseline
            rng = _rnd.Random(seed)
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            return out
        baseline_pipelines["standard_prompting"] = _mock_standard_prompting

        async def mock_pipeline(tokens, dirty_tags, config=None, dataset_name=None):
            import random as _rnd
            gold = config.get("__gold__", dirty_tags) if config else dirty_tags
            rate = 0.6 if (config and config.get("use_dfa")) else 0.3
            rng = _rnd.Random(config.get("__seed__", 0) if config else 0)
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            if config and config.get("__return_candidates__"):
                candidate_paths = [
                    list(dirty_tags[:len(gold)]) + ["O"] * max(0, len(gold) - len(dirty_tags)),
                    list(gold[:len(gold)]),
                ]
                rag_weights = [0.25, 0.75]
                confidence = [
                    0.75 if j < len(gold) and candidate_paths[1][j] != candidate_paths[0][j] else 1.0
                    for j in range(len(gold))
                ]
                return {
                    "pred_tags": list(gold[:len(gold)]),
                    "candidate_paths": candidate_paths,
                    "rag_weights": rag_weights,
                    "confidence": confidence,
                }
            return out
        return mock_pipeline, baseline_pipelines

    try:
        from multi_agent_v2 import (
            run_agent_pipeline,
            baseline_cot_reasoning_pipeline,
            baseline_self_refine_pipeline,
            baseline_zero_shot_pipeline,
            baseline_standard_prompting_pipeline,
        )
        baseline_pipelines["cot_reasoning"]      = baseline_cot_reasoning_pipeline
        baseline_pipelines["self_refine"]        = baseline_self_refine_pipeline
        baseline_pipelines["zero_shot"]          = baseline_zero_shot_pipeline
        baseline_pipelines["standard_prompting"] = baseline_standard_prompting_pipeline
        return run_agent_pipeline, baseline_pipelines
    except ImportError as e:                              # pragma: no cover
        raise ImportError(
            "Could not import run_agent_pipeline from multi_agent_v2.py. "
            "Either run from the directory containing multi_agent_v2.py, "
            "or add it to PYTHONPATH."
        ) from e


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

def _noisy_path(dataset: str, noise: str, seed: int, size: int,
                ratio: float = 0.15) -> Path:
    size_suffix = f"__N{size}" if size else ""
    if abs(ratio - 0.15) < 1e-9:
        return NOISY_DIR / f"noisy_seed{seed}__{noise}__{dataset}{size_suffix}.jsonl"
    rate_pct = int(round(ratio * 100))
    return NOISY_DIR / f"noisy_seed{seed}__{noise}__{dataset}{size_suffix}__r{rate_pct}.jsonl"


def _pred_path(config_name: str, dataset: str, noise: str, seed: int,
               ratio: float = 0.15) -> Path:
    # Same backward-compat scheme as _noisy_path, plus an optional backbone tag
    # appended last (empty by default → byte-identical to the original names).
    # Note the noisy path takes no tag: noise data is backbone-independent.
    tag_suffix = f"__{BACKBONE_TAG}" if BACKBONE_TAG else ""
    if abs(ratio - 0.15) < 1e-9:
        return PRED_DIR / (f"pred_seed{seed}__{config_name}__{dataset}"
                           f"__{noise}{tag_suffix}.jsonl")
    rate_pct = int(round(ratio * 100))
    return PRED_DIR / (f"pred_seed{seed}__{config_name}__{dataset}"
                       f"__{noise}__r{rate_pct}{tag_suffix}.jsonl")


def _load_noisy(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _supports_candidate_evidence(config_name: str, config: dict) -> bool:
    if config.get("terminal_graph") == "lad-rg":
        return True
    return config_name in {
        "selectdenoise_contextual_lattice",
        "selectdenoise_full",
        "selectdenoise_full_legacy",
        "selectdenoise_no_deanchor",
    }


# ---------------------------------------------------------------------------
# Per-cell async runner
# ---------------------------------------------------------------------------


class _CachedAdapterFactory:
    """Own one live adapter for an official process and close it exactly once."""

    def __init__(self, create: Callable[[], Any]) -> None:
        self._create = create
        self._adapter: Any = None
        self._closed = False

    def __call__(self) -> Any:
        if self._closed:
            raise RuntimeError("cached live adapter factory is closed")
        if self._adapter is None:
            self._adapter = self._create()
        return self._adapter

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_adapter = getattr(self._adapter, "close", None)
        if callable(close_adapter):
            close_adapter()


def _provider_cache_cell_path(root: Path, tag: str, config_name: str,
                              dataset: str, noise: str, seed: int) -> Path:
    return Path(root) / tag / f"provider_seed{seed}__{config_name}__{dataset}__{noise}.jsonl"


def _provider_cache_index_path(root: Path, tag: str) -> Path:
    return Path(root) / tag / "index.json"


def _provider_cache_cell_key(config_name: str, dataset: str, noise: str, seed: int) -> str:
    return f"{config_name}__{dataset}__{noise}__seed{seed}"


def _provider_input_digest(row: Mapping[str, Any]) -> str:
    """Hash only the gold-free Stage A input."""
    tokens, dirty_tags = row.get("tokens"), row.get("dirty_tags")
    if (not isinstance(tokens, list) or not isinstance(dirty_tags, list)
            or not all(isinstance(value, str) for value in tokens + dirty_tags)
            or len(tokens) != len(dirty_tags)):
        raise ValueError("provider-cache source row requires aligned tokens and dirty_tags")
    return _sha256_json({"tokens": tokens, "dirty_tags": dirty_tags})


def _provider_cache_manifest(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *,
                             dataset: str, noise: str, seed: int, git_sha: str,
                             bundle_hash: str,
                             enable_thinking: bool = True) -> dict[str, Any]:
    if type(enable_thinking) is not bool:
        raise ValueError("provider-cache thinking identity must be boolean")
    return {
        "schema": PROVIDER_CACHE_SCHEMA,
        "source_digests": {str(index): _provider_input_digest(row) for index, row in enumerate(rows)},
        "git_sha": git_sha, "model_revision": QWEN_REVISION, "bundle_hash": bundle_hash,
        "configuration": {"config": "selectdenoise_contextual_lattice",
                          "terminal_decoder": config.get("terminal_decoder"),
                          "dataset": dataset, "noise": noise, "seed": seed,
                          "enable_thinking": enable_thinking,
                          "thinking_mode": "thinking" if enable_thinking else "nothink"},
    }


def _read_provider_cache_manifest(path: Path) -> Mapping[str, Any]:
    try:
        header = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
    except (IndexError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("provider-cache header is malformed") from exc
    if not isinstance(header, Mapping) or set(header) != {"schema", "manifest"}:
        raise ValueError("provider-cache header is malformed")
    return header["manifest"]


def _update_provider_cache_index(root: Path, tag: str, *, key: str,
                                 cell: Mapping[str, Any]) -> None:
    index_path = _provider_cache_index_path(root, tag)
    index: dict[str, Any] = {"schema": PROVIDER_CACHE_INDEX_SCHEMA, "cells": {}}
    if index_path.exists():
        try:
            candidate = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("provider-cache index is malformed") from exc
        if (not isinstance(candidate, Mapping) or candidate.get("schema") != PROVIDER_CACHE_INDEX_SCHEMA
                or not isinstance(candidate.get("cells"), Mapping)):
            raise ValueError("provider-cache index is malformed")
        index["cells"] = dict(candidate["cells"])
    index["cells"][key] = dict(cell)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(_canonical_json(index) + "\n", encoding="utf-8")
        temporary.replace(index_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_provider_cache_index_cell(root: Path, tag: str, key: str) -> Mapping[str, Any]:
    index_path = _provider_cache_index_path(root, tag)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("provider-cache index is required for resume") from exc
    cells = index.get("cells") if isinstance(index, Mapping) else None
    cell = cells.get(key) if isinstance(cells, Mapping) else None
    if (not isinstance(index, Mapping) or index.get("schema") != PROVIDER_CACHE_INDEX_SCHEMA
            or not isinstance(cell, Mapping)):
        raise ValueError("provider-cache index is missing the cache cell")
    required = {"path", "sha256", "row_count", "git_sha", "model_revision", "provider_fingerprint"}
    if set(cell) != required:
        raise ValueError("provider-cache index cell is malformed")
    return cell


def _validate_replay_cache_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    compatible_git_shas: Sequence[str] = (),
) -> None:
    """Bind replay to semantic identity while preserving producer provenance."""
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("provider-cache replay identity must be a mapping")
    allowed_git_shas = {str(expected.get("git_sha", "")).lower()}
    for value in compatible_git_shas:
        normalized = str(value).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", normalized):
            raise ValueError("provider-cache compatible git_sha is invalid")
        allowed_git_shas.add(normalized)
    if str(actual.get("git_sha", "")).lower() not in allowed_git_shas:
        raise ValueError("provider-cache replay git_sha identity mismatch")
    for field in ("schema", "model_revision", "bundle_hash", "configuration", "source_digests"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"provider-cache replay {field} identity mismatch")


def _validate_replay_source_rows(source_rows: Sequence[Mapping[str, Any]],
                                 cached_rows: Sequence[Mapping[str, Any]]) -> None:
    """Re-read canonical noisy inputs before allowing terminal decoding."""
    if len(source_rows) != len(cached_rows):
        raise ValueError("provider-cache replay source row count mismatch")
    for index, (source, cached) in enumerate(zip(source_rows, cached_rows)):
        if cached.get("row_index") != index:
            raise ValueError(f"provider-cache replay row index mismatch at {index}")
        expected_digest = _provider_input_digest(source)
        if cached.get("input_digest") != expected_digest:
            raise ValueError(f"provider-cache replay input digest mismatch at row {index}")


def _validate_contextual_replay_launch(config_names: Sequence[str], *, size: int | None,
                                      ratios: Sequence[float], max_concurrency: int,
                                      failure_policy: str, dummy: bool,
                                      datasets: Sequence[str] | None = None,
                                      noise_types: Sequence[str] | None = None,
                                      seeds: Sequence[int] | None = None) -> None:
    if dummy or failure_policy != "abort":
        raise ValueError("contextual-replay phase requires real offline replay and failure-policy abort")
    if size != OFFICIAL_SAMPLE_SIZE or len(ratios) != 1 or abs(ratios[0] - OFFICIAL_NOISE_RATIO) >= 1e-12:
        raise ValueError("contextual-replay phase requires size 200 and ratio 0.15")
    if list(config_names) != ["selectdenoise_contextual_lattice"]:
        raise ValueError("contextual-replay phase requires exactly the contextual-lattice config")
    if max_concurrency <= 0:
        raise ValueError("contextual-replay phase requires positive max-concurrency")
    for name, actual, expected in (
        ("datasets", datasets, DATASETS),
        ("noise_types", noise_types, NOISE_TYPES),
        ("seeds", seeds, SEEDS),
    ):
        if actual is not None and list(actual) != list(expected):
            raise ValueError(f"contextual-replay phase requires canonical {name}: {list(expected)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contextual_replay_bundle(bundle_hash: str) -> tuple[Path, Path]:
    """Validate the locked local bundle and its pre-existing embedding cache."""
    from contextual_lattice_runtime import (
        LOCKED_BUNDLE_MANIFEST_HASH, LOCKED_CHECKPOINT_HASH,
        LOCKED_DECODER_FILE_HASH, LOCKED_GATE_FILE_HASH,
        LOCKED_DECODER_MODEL_HASH, LOCKED_SPLIT_HASH,
    )

    if (not isinstance(bundle_hash, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", bundle_hash)
            or bundle_hash.lower() != LOCKED_BUNDLE_MANIFEST_HASH):
        raise ValueError("contextual-replay requires the locked bundle manifest hash")
    bundle_text = os.environ.get("CONTEXTUAL_LATTICE_BUNDLE", "").strip()
    if not bundle_text:
        raise ValueError("contextual-replay requires CONTEXTUAL_LATTICE_BUNDLE")
    bundle_path = Path(bundle_text).expanduser().resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"contextual replay bundle directory not found: {bundle_path}")
    manifest_path = bundle_path / "manifest.json"
    decoder_path = bundle_path / "decoder.pt"
    gate_path = bundle_path / "gate.joblib"
    for path in (manifest_path, decoder_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"contextual replay bundle file not found: {path}")
    if _sha256_file(manifest_path) != LOCKED_BUNDLE_MANIFEST_HASH:
        raise ValueError("contextual replay bundle manifest hash mismatch")
    if _sha256_file(decoder_path) != LOCKED_DECODER_FILE_HASH:
        raise ValueError("contextual replay decoder file hash mismatch")
    if _sha256_file(gate_path) != LOCKED_GATE_FILE_HASH:
        raise ValueError("contextual replay gate file hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("contextual replay bundle manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("contextual replay bundle manifest must be an object")
    checkpoint_value = manifest.get("checkpoint") if isinstance(manifest, Mapping) else None
    checkpoint = (checkpoint_value.get("path") if isinstance(checkpoint_value, Mapping)
                  else checkpoint_value)
    if not isinstance(checkpoint, str):
        raise ValueError("contextual replay bundle checkpoint provenance is missing")
    if manifest.get("checkpoint_hash") != LOCKED_CHECKPOINT_HASH:
        raise ValueError("contextual replay manifest checkpoint hash is not locked")
    if (manifest.get("split_hash") != LOCKED_SPLIT_HASH
            or manifest.get("decoder_model_hash") != LOCKED_DECODER_MODEL_HASH):
        raise ValueError("contextual replay bundle provenance is not locked")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        candidates = [Path.cwd() / checkpoint_path,
                      bundle_path.parents[2] / checkpoint_path]
        checkpoint_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    checkpoint_file = (checkpoint_path / "model.safetensors"
                       if checkpoint_path.is_dir() else checkpoint_path)
    if not checkpoint_file.is_file() or _sha256_file(checkpoint_file) != LOCKED_CHECKPOINT_HASH:
        raise ValueError("contextual replay checkpoint hash mismatch")
    embedding_text = os.environ.get("CONTEXTUAL_LATTICE_ENCODER_CACHE", "").strip()
    embedding_path = (Path(embedding_text).expanduser().resolve()
                      if embedding_text else bundle_path / "embedding_cache")
    if not embedding_path.is_dir():
        raise FileNotFoundError(
            f"contextual replay embedding cache directory not found: {embedding_path}"
        )
    os.environ["CONTEXTUAL_LATTICE_ENCODER_CACHE"] = str(embedding_path)
    return bundle_path, embedding_path


@contextlib.contextmanager
def _offline_contextual_replay_environment():
    """Remove provider credentials and fail closed on socket connection attempts."""
    credential_keys = (
        "BACKBONE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY",
        "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN",
    )
    offline_keys = credential_keys + (
        "BACKBONE_PROVIDER", "BACKBONE_MODEL", "BACKBONE_BASE_URL",
        "BACKBONE_SERVED_MODEL", "BACKBONE_REVISION", "LAD_RG_OFFICIAL_REQUESTS",
        "QWEN_ENABLE_THINKING",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    )
    saved = {key: os.environ.get(key) for key in offline_keys}
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    for key in credential_keys:
        os.environ[key] = ""
    os.environ.update({
        "BACKBONE_PROVIDER": "offline",
        "BACKBONE_MODEL": "offline-contextual-replay",
        "BACKBONE_BASE_URL": "http://127.0.0.1",
        "BACKBONE_SERVED_MODEL": "offline-contextual-replay",
        "BACKBONE_REVISION": "",
        "LAD_RG_OFFICIAL_REQUESTS": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled during contextual replay")

    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.socket.connect = original_connect
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_contextual_replay_prediction(
    path: Path, dataset: str, noise_type: str, expected_sha256: str,
    expected_count: int = OFFICIAL_SAMPLE_SIZE, enable_thinking: bool = True,
) -> None:
    from contextual_lattice_runtime import LOCKED_DECODER_MODEL_HASH

    rows = _load_noisy(path)
    if len(rows) != expected_count:
        raise RuntimeError(f"contextual replay prediction has {len(rows)} records, expected {expected_count}")
    valid = _valid_tags(dataset)
    identity = {
        "provider": "vllm", "model": QWEN_MODEL,
        "served_model": QWEN_SERVED_MODEL, "revision": QWEN_REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": enable_thinking,
        "thinking_mode": "thinking" if enable_thinking else "nothink",
    }
    for index, row in enumerate(rows):
        tokens, gold, pred = row.get("tokens"), row.get("gold_tags"), row.get("pred_tags")
        if (not isinstance(tokens, list) or not isinstance(gold, list) or not isinstance(pred, list)
                or not len(tokens) == len(gold) == len(pred)
                or not all(isinstance(value, str) for values in (tokens, gold, pred) for value in values)
                or any(tag not in valid for values in (gold, pred) for tag in values)
                or not _is_legal_iob2(pred)):
            raise RuntimeError(f"contextual replay row {index} is structurally invalid")
        anchor = row.get("terminal_anchor_tags")
        model_hash = row.get("terminal_model_hash")
        used_anchor = row.get("terminal_used_anchor")
        gain = row.get("terminal_predicted_gain")
        fallback_count = row.get("terminal_fallback_count")
        if (not isinstance(anchor, list) or len(anchor) != len(tokens)
                or any(tag not in valid for tag in anchor) or not _is_legal_iob2(anchor)
                or not isinstance(model_hash, str)
                or model_hash != LOCKED_DECODER_MODEL_HASH
                or not isinstance(used_anchor, bool)
                or isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain))
                or isinstance(fallback_count, bool) or not isinstance(fallback_count, int) or fallback_count < 0
                or row.get("provider_cache_sha256") != expected_sha256
                or row.get("fallback_used") is not False
                or not isinstance(row.get("provider_metadata"), Mapping)):
            raise RuntimeError(f"contextual replay row {index} lacks terminal provenance")
        _validate_contextual_provider_evidence(row["provider_metadata"], identity, noise_type)


def _validate_provider_cache_launch(config_names: Sequence[str], *, size: int | None,
                                    ratios: Sequence[float], max_concurrency: int,
                                    failure_policy: str, dummy: bool,
                                    datasets: Sequence[str] | None = None,
                                    noise_types: Sequence[str] | None = None,
                                    seeds: Sequence[int] | None = None) -> None:
    if dummy or failure_policy != "abort":
        raise ValueError("provider-cache phase requires real requests and failure-policy abort")
    if size != OFFICIAL_SAMPLE_SIZE or len(ratios) != 1 or abs(ratios[0] - OFFICIAL_NOISE_RATIO) >= 1e-12:
        raise ValueError("provider-cache phase requires size 200 and ratio 0.15")
    if max_concurrency != OFFICIAL_PROVIDER_MAX_CONCURRENCY:
        raise ValueError(
            "provider-cache phase requires "
            f"max-concurrency {OFFICIAL_PROVIDER_MAX_CONCURRENCY}"
        )
    if list(config_names) != ["selectdenoise_contextual_lattice"]:
        raise ValueError("provider-cache phase requires exactly the contextual-lattice config")
    for name, actual, expected in (
        ("datasets", datasets, DATASETS),
        ("noise_types", noise_types, NOISE_TYPES),
        ("seeds", seeds, SEEDS),
    ):
        if actual is not None and list(actual) != list(expected):
            raise ValueError(f"provider-cache phase requires canonical {name}: {list(expected)}")


async def _run_provider_cache_cell(
    config_name: str, config: Mapping[str, Any], dataset: str, noise: str, seed: int,
    size: int, pipelines, *, cache_root: Path, cache_tag: str, max_concurrency: int,
    request_timeout: float, adapter_factory: Callable[[], Any], git_sha: str,
    bundle_hash: str, compatible_git_shas: Sequence[str] = (),
) -> dict[str, Any]:
    """Execute only the provider graph and atomically publish one gold-free cell."""
    _validate_provider_cache_launch([config_name], size=size, ratios=[OFFICIAL_NOISE_RATIO],
                                    max_concurrency=max_concurrency, failure_policy="abort", dummy=False)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cache_tag):
        raise ValueError("provider-cache tag is invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_sha):
        raise ValueError("provider-cache requires a 40-hex git SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", bundle_hash):
        raise ValueError("provider-cache requires a 64-hex bundle hash")
    noisy_path = _noisy_path(dataset, noise, seed, size, OFFICIAL_NOISE_RATIO)
    if not noisy_path.exists():
        raise FileNotFoundError(f"official noisy file not found: {noisy_path}")
    rows = _load_noisy(noisy_path)
    if len(rows) != OFFICIAL_SAMPLE_SIZE:
        raise RuntimeError("provider-cache cells require exactly 200 noisy rows")
    cell_path = _provider_cache_cell_path(cache_root, cache_tag, config_name, dataset, noise, seed)
    key = _provider_cache_cell_key(config_name, dataset, noise, seed)
    adapter = adapter_factory()
    identity = adapter.provider_metadata()
    if not isinstance(identity, Mapping):
        raise RuntimeError("provider-cache adapter identity is malformed")
    identity = dict(identity)
    enable_thinking = identity.get("enable_thinking")
    if type(enable_thinking) is not bool:
        # Historical Qwen calls requested thinking when no explicit override
        # was configured; retain that default while making it part of identity.
        enable_thinking = True
        identity["enable_thinking"] = True
        identity["thinking_mode"] = "thinking"
    elif identity.get("thinking_mode") != ("thinking" if enable_thinking else "nothink"):
        raise RuntimeError("provider-cache adapter identity has an invalid thinking-mode binding")
    if not enable_thinking and "nothink" not in cache_tag.lower():
        raise ValueError("no-thinking provider caches require a fresh tag containing 'nothink'")
    manifest = _provider_cache_manifest(
        rows, config, dataset=dataset, noise=noise, seed=seed,
        git_sha=git_sha.lower(), bundle_hash=bundle_hash.lower(),
        enable_thinking=enable_thinking,
    )
    fingerprint = _sha256_json(identity)
    manifest['request_limits'] = {
        'provider_timeout_seconds': identity.get('timeout_seconds'),
        'runner_timeout_seconds': request_timeout,
        'max_concurrency': max_concurrency,
    }
    if cell_path.exists():
        indexed = _read_provider_cache_index_cell(cache_root, cache_tag, key)
        cached_manifest = _read_provider_cache_manifest(cell_path)
        _validate_replay_cache_identity(
            cached_manifest,
            manifest,
            compatible_git_shas=compatible_git_shas,
        )
        if (indexed["path"] != str(cell_path) or indexed["row_count"] != OFFICIAL_SAMPLE_SIZE
                or indexed["git_sha"] != cached_manifest["git_sha"]
                or indexed["model_revision"] != cached_manifest["model_revision"]):
            raise ValueError("provider-cache index identity mismatch; refusing resume")
        cached = read_provider_cell(cell_path, indexed["sha256"], OFFICIAL_SAMPLE_SIZE)
        for record in cached:
            _validate_contextual_provider_evidence(record["provider_metadata"], identity, noise)
        return dict(indexed)

    default_fn, _baseline_fns = pipelines
    sem = asyncio.Semaphore(max_concurrency)
    provider_sem = threading.BoundedSemaphore(max_concurrency)
    provider_requester = getattr(adapter, "structured_requester", None)
    if not callable(provider_requester):
        raise RuntimeError("provider-cache adapter has no structured requester")
    if inspect.iscoroutinefunction(provider_requester):
        raise RuntimeError("provider-cache structured requester must be synchronous")
    provider_aborted = threading.Event()
    first_provider_error = []
    error_lock = threading.Lock()

    def limited_provider_requester(stage: str, payload: Mapping[str, Any]) -> Any:
        queued_at = time.monotonic()
        while not provider_aborted.is_set():
            if provider_sem.acquire(timeout=0.05):
                break
        else:
            raise RuntimeError("provider-cache requests are aborted")
        try:
            if provider_aborted.is_set():
                raise RuntimeError("provider-cache requests are aborted")
            queue_seconds = time.monotonic() - queued_at
            emit('provider_admitted', stage=stage, path_name=payload.get('name'),
                 queue_seconds=queue_seconds)
            with queue_context(queue_seconds):
                result = provider_requester(stage, payload)
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise RuntimeError(
                    "provider-cache structured requester must be synchronous"
                )
            return result
        except BaseException as exc:
            with error_lock:
                if not first_provider_error and not provider_aborted.is_set():
                    first_provider_error.append(exc)
            provider_aborted.set()
            raise
        finally:
            provider_sem.release()

    cached: list[Optional[dict[str, Any]]] = [None] * len(rows)
    provider_config = dict(config)
    provider_config["preterminal_only"] = True

    async def process(index: int, row: Mapping[str, Any]) -> None:
        async with sem:
            tokens, dirty = row["tokens"], row["dirty_tags"]
            runtime_config = dict(provider_config)
            runtime_config.update({"__dataset__": dataset, "__noise_type__": noise,
                                   "official": True, "__return_candidates__": True,
                                   "structured_requester": limited_provider_requester,
                                   "provider_metadata": dict(identity)})
            with sentence_context(dataset=dataset, noise=noise, seed=seed, row_index=index):
                result = await asyncio.wait_for(
                    default_fn(tokens, dirty, runtime_config, dataset_name=dataset), timeout=request_timeout)
            if not isinstance(result, Mapping):
                raise RuntimeError(f"provider-cache sentence {index} returned no stage evidence")
            anchor, evidence = result.get("pred_tags"), result.get("provider_metadata")
            if (not isinstance(anchor, list) or len(anchor) != len(tokens)
                    or any(tag not in _valid_tags(dataset) for tag in anchor) or not _is_legal_iob2(anchor)):
                raise RuntimeError(f"provider-cache sentence {index} returned invalid anchor tags")
            _validate_contextual_provider_evidence(evidence, identity, noise)
            cached[index] = {"row_index": index, "input_digest": _provider_input_digest(row),
                             "anchor_tags": anchor, "candidate_paths": result.get("candidate_paths"),
                             "rag_weights": result.get("rag_weights"), "confidence": result.get("confidence"),
                             "provider_metadata": evidence, "fallback_used": result.get("fallback_used")}

    tasks = [asyncio.create_task(process(index, row)) for index, row in enumerate(rows)]
    try:
        await asyncio.gather(*tasks)
        if any(record is None for record in cached):
            raise RuntimeError("provider-cache cell has an incomplete result buffer")
        digest = write_provider_cell(cell_path, [record for record in cached if record is not None], manifest)
    except BaseException:
        provider_aborted.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        cell_path.with_suffix(cell_path.suffix + ".tmp").unlink(missing_ok=True)
        if first_provider_error:
            raise first_provider_error[0]
        raise
    cell = {"path": str(cell_path), "sha256": digest, "row_count": len(cached),
            "git_sha": manifest["git_sha"], "model_revision": QWEN_REVISION,
            "provider_fingerprint": fingerprint}
    _update_provider_cache_index(cache_root, cache_tag, key=key, cell=cell)
    return cell


async def _run_contextual_replay_cell(
    dataset: str, noise: str, seed: int, size: int, *,
    cache_root: Path, cache_tag: str, bundle_hash: str, git_sha: str,
    replay_fn: Callable[..., Mapping[str, Any]], terminal: Any,
    max_concurrency: int, prediction_path: Optional[Path] = None,
    enable_thinking: bool = True, compatible_git_shas: Sequence[str] = (),
) -> Path:
    """Decode one provider-cache cell locally with no provider adapter."""
    _validate_contextual_replay_launch(
        ["selectdenoise_contextual_lattice"], size=size,
        ratios=[OFFICIAL_NOISE_RATIO], max_concurrency=max_concurrency,
        failure_policy="abort", dummy=False,
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cache_tag):
        raise ValueError("contextual-replay provider-cache tag is invalid")
    if not enable_thinking and "nothink" not in cache_tag.lower():
        raise ValueError("no-thinking replay caches require a fresh tag containing 'nothink'")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_sha):
        raise ValueError("contextual-replay requires a 40-hex Git SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", bundle_hash):
        raise ValueError("contextual-replay requires a 64-hex bundle hash")
    noisy_path = _noisy_path(dataset, noise, seed, size, OFFICIAL_NOISE_RATIO)
    if not noisy_path.exists():
        raise FileNotFoundError(f"official noisy file not found: {noisy_path}")
    source_rows = _load_noisy(noisy_path)
    if len(source_rows) != OFFICIAL_SAMPLE_SIZE:
        raise RuntimeError("contextual-replay cells require exactly 200 noisy rows")
    config = CONFIGURATIONS["selectdenoise_contextual_lattice"]
    expected_manifest = _provider_cache_manifest(
        source_rows, config, dataset=dataset, noise=noise, seed=seed,
        git_sha=git_sha.lower(), bundle_hash=bundle_hash.lower(),
        enable_thinking=enable_thinking,
    )
    key = _provider_cache_cell_key("selectdenoise_contextual_lattice", dataset, noise, seed)
    indexed = _read_provider_cache_index_cell(cache_root, cache_tag, key)
    expected_cell_path = _provider_cache_cell_path(
        cache_root, cache_tag, "selectdenoise_contextual_lattice", dataset, noise, seed,
    )
    if (Path(str(indexed["path"])).resolve() != expected_cell_path.resolve()
            or indexed["row_count"] != OFFICIAL_SAMPLE_SIZE
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(indexed["sha256"]))
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(indexed["provider_fingerprint"]))):
        raise ValueError("contextual-replay cache index identity mismatch")
    cell_path = expected_cell_path
    cached_rows = read_provider_cell(cell_path, str(indexed["sha256"]), OFFICIAL_SAMPLE_SIZE)
    cached_manifest = _read_provider_cache_manifest(cell_path)
    _validate_replay_cache_identity(
        cached_manifest,
        expected_manifest,
        compatible_git_shas=compatible_git_shas,
    )
    if (indexed["git_sha"] != cached_manifest["git_sha"]
            or indexed["model_revision"] != cached_manifest["model_revision"]):
        raise ValueError("contextual-replay cache index identity mismatch")
    _validate_replay_source_rows(source_rows, cached_rows)
    provider_identity = {
        "provider": "vllm", "model": QWEN_MODEL,
        "served_model": QWEN_SERVED_MODEL, "revision": QWEN_REVISION,
        "structured_api": "chat-completions-json-schema",
        "enable_thinking": enable_thinking,
        "thinking_mode": "thinking" if enable_thinking else "nothink",
    }
    for cached in cached_rows:
        _validate_contextual_provider_evidence(
            cached["provider_metadata"], provider_identity, noise,
        )
    cache_sha256 = str(indexed["sha256"]).lower()
    pred_path = Path(prediction_path) if prediction_path is not None else _pred_path(
        "selectdenoise_contextual_lattice", dataset, noise, seed, OFFICIAL_NOISE_RATIO,
    )
    if pred_path.exists() and pred_path.stat().st_size > 0:
        _validate_contextual_replay_prediction(
            pred_path, dataset, noise, cache_sha256,
            enable_thinking=enable_thinking,
        )
        return pred_path

    sem = asyncio.Semaphore(max_concurrency)
    buffer: list[Optional[dict[str, Any]]] = [None] * len(source_rows)
    tmp_path = pred_path.with_suffix(pred_path.suffix + ".tmp")
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    async def process(index: int, source: Mapping[str, Any], cached: Mapping[str, Any]) -> None:
        async with sem:
            if _provider_input_digest(source) != cached["input_digest"]:
                raise ValueError(f"contextual-replay input digest changed at row {index}")
            tokens = source.get("tokens")
            dirty_tags = source.get("dirty_tags")
            gold_tags = source.get("ner_tags")
            if (not isinstance(tokens, list) or not isinstance(dirty_tags, list)
                    or not isinstance(gold_tags, list)):
                raise ValueError(f"contextual-replay source row {index} is malformed")
            result = await asyncio.to_thread(
                replay_fn,
                tokens=tokens,
                dirty_tags=dirty_tags,
                provider_record=cached,
                dataset_name=dataset,
                terminal=terminal,
            )
            if not isinstance(result, Mapping):
                raise RuntimeError(f"contextual-replay row {index} returned no terminal result")
            required = {
                "pred_tags", "terminal_anchor_tags", "terminal_model_hash",
                "terminal_used_anchor", "terminal_predicted_gain",
                "terminal_fallback_count",
            }
            if not required.issubset(result):
                raise RuntimeError(f"contextual-replay row {index} lacks terminal provenance")
            buffer[index] = {
                "tokens": list(tokens), "gold_tags": list(gold_tags),
                "pred_tags": list(result["pred_tags"]),
                "terminal_anchor_tags": list(result["terminal_anchor_tags"]),
                "terminal_model_hash": result["terminal_model_hash"],
                "terminal_used_anchor": result["terminal_used_anchor"],
                "terminal_predicted_gain": result["terminal_predicted_gain"],
                "terminal_fallback_count": result["terminal_fallback_count"],
                "provider_cache_sha256": cache_sha256,
                "provider_metadata": cached["provider_metadata"],
                "fallback_used": False,
            }

    tasks = [asyncio.create_task(process(index, source, cached))
             for index, (source, cached) in enumerate(zip(source_rows, cached_rows))]
    try:
        await asyncio.gather(*tasks)
        if any(record is None for record in buffer):
            raise RuntimeError("contextual-replay cell produced an incomplete result buffer")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _validate_contextual_replay_prediction(
            tmp_path, dataset, noise, cache_sha256,
            enable_thinking=enable_thinking,
        )
        tmp_path.replace(pred_path)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        tmp_path.unlink(missing_ok=True)
        raise
    return pred_path


async def _run_one_cell(config_name: str, config: dict,
                        dataset: str, noise: str, seed: int,
                        size: int, pipelines,
                        *, max_concurrency: int, dummy: bool,
                        ratio: float = 0.15, official: bool = False,
                        failure_policy: str = "abort",
                        request_timeout: float = PER_REQUEST_TIMEOUT,
                        adapter_factory: Optional[Callable[[], Any]] = None,
                        paid_smoke_size: Optional[int] = None,
                        row_indices: Optional[Sequence[int]] = None,
                        prediction_path: Optional[Path] = None):
    """Run one cell and clean only its exact temporary output on abort."""
    pred_p = (Path(prediction_path) if prediction_path is not None
              else _pred_path(config_name, dataset, noise, seed, ratio))
    tmp_p = pred_p.with_suffix(pred_p.suffix + ".tmp")
    try:
        return await _run_one_cell_impl(
            config_name, config, dataset, noise, seed, size, pipelines,
            max_concurrency=max_concurrency, dummy=dummy, ratio=ratio,
            official=official, failure_policy=failure_policy,
            request_timeout=request_timeout, adapter_factory=adapter_factory,
            paid_smoke_size=paid_smoke_size, row_indices=row_indices,
            prediction_path=prediction_path,
        )
    except Exception:
        if failure_policy == "abort":
            tmp_p.unlink(missing_ok=True)
        raise


async def _run_one_cell_impl(config_name: str, config: dict,
                             dataset: str, noise: str, seed: int,
                             size: int, pipelines,
                             *, max_concurrency: int, dummy: bool,
                             ratio: float = 0.15, official: bool = False,
                             failure_policy: str = "abort",
                             request_timeout: float = PER_REQUEST_TIMEOUT,
                             adapter_factory: Optional[Callable[[], Any]] = None,
                             paid_smoke_size: Optional[int] = None,
                             row_indices: Optional[Sequence[int]] = None,
                             prediction_path: Optional[Path] = None):
    if config.get("offline_only"):
        raise RuntimeError(
            f"Config '{config_name}' is offline-only; derive it with "
            f"{config['offline_only']} from {config.get('source_method', 'existing prediction')} files."
        )

    if failure_policy not in {"abort", "dirty"}:
        raise ValueError("failure_policy must be 'abort' or 'dirty'")
    is_paid_smoke = paid_smoke_size is not None
    launch_strict = official or is_paid_smoke
    official_profile = _official_profile(config) if launch_strict else None
    if official and is_paid_smoke:
        raise ValueError("official publication and paid smoke modes are mutually exclusive")
    if official and prediction_path is not None:
        raise ValueError("official predictions must use the canonical tagged namespace")
    if is_paid_smoke:
        contextual_smoke = config.get("terminal_decoder") == "contextual-lattice-v1"
        if (not contextual_smoke and paid_smoke_size != PAID_SMOKE_SIZE):
            raise ValueError(f"paid smoke requires exactly {PAID_SMOKE_SIZE} records")
        if contextual_smoke and (
                not row_indices or paid_smoke_size != len(row_indices)
                or len(set(row_indices)) != len(row_indices)):
            raise ValueError("contextual paid smoke requires unique selected row indices")
        if prediction_path is None:
            raise ValueError("paid smoke requires a dedicated prediction_path")
        if dummy:
            raise ValueError("paid smoke cannot use --dummy")
        if failure_policy != "abort":
            raise ValueError("paid smoke requires failure_policy='abort'")
        if (config.get("terminal_graph") != "lad-rg"
                and config.get("terminal_decoder") != "contextual-lattice-v1"):
            raise ValueError("paid smoke requires a LAD-RG configuration")
        if size != OFFICIAL_SAMPLE_SIZE:
            raise ValueError("paid smoke must read the canonical size 200 noisy cell")
        if abs(ratio - OFFICIAL_NOISE_RATIO) >= 1e-12:
            raise ValueError("paid smoke requires noise ratio 0.15")
    if official:
        if dummy:
            raise ValueError("official mode cannot use --dummy")
        if failure_policy != "abort":
            raise ValueError("official mode requires failure_policy='abort'")
        if size != OFFICIAL_SAMPLE_SIZE or abs(ratio - OFFICIAL_NOISE_RATIO) >= 1e-12:
            raise ValueError("official cells require size 200 and noise ratio 0.15")

    default_fn, baseline_fns = pipelines
    pred_p = (Path(prediction_path) if prediction_path is not None
              else _pred_path(config_name, dataset, noise, seed, ratio))
    tmp_p = pred_p.with_suffix(pred_p.suffix + ".tmp")
    pred_p.parent.mkdir(parents=True, exist_ok=True)
    if is_paid_smoke and pred_p.exists():
        raise FileExistsError(f"paid smoke output already exists: {pred_p}")
    if not launch_strict and pred_p.exists() and pred_p.stat().st_size > 0:
        logging.info(f"[skip] {pred_p.name}")
        return

    noisy_p = _noisy_path(dataset, noise, seed, size, ratio)
    if not noisy_p.exists():
        if launch_strict:
            raise FileNotFoundError(f"official noisy file not found: {noisy_p}")
        logging.warning(f"[miss] noisy file not found: {noisy_p}; "
                        f"run gen_noisy.py first")
        return

    rows = _load_noisy(noisy_p)
    if launch_strict and len(rows) != OFFICIAL_SAMPLE_SIZE:
        raise RuntimeError(
            f"official noisy cell must contain exactly {OFFICIAL_SAMPLE_SIZE} records; "
            f"found {len(rows)} in {noisy_p.name}"
        )
    if is_paid_smoke:
        rows = ([rows[index] for index in row_indices]
                if row_indices is not None else rows[:paid_smoke_size])
    logging.info(f"[run]  {pred_p.name}  ({len(rows)} sentences)")
    t0 = time.time()

    sem = asyncio.Semaphore(max_concurrency)
    # Buffer results IN INPUT ORDER so prediction files line up across methods.
    buffer: List[Optional[dict]] = [None] * len(rows)
    adapter = None
    provider_identity: Mapping[str, Any] = {}
    if launch_strict:
        if adapter_factory is None:
            from live_backbone import OpenAICompatibleLADRGAdapter
            adapter_factory = OpenAICompatibleLADRGAdapter
        adapter = adapter_factory()
        provider_identity = adapter.provider_metadata()
        if official and pred_p.exists() and pred_p.stat().st_size > 0:
            _validate_official_prediction(
                pred_p, dataset, config_name, noise,
                expected_count=OFFICIAL_SAMPLE_SIZE,
                provider_identity=provider_identity,
            )
            logging.info(f"[skip] {pred_p.name}")
            return

    async def _process(i, row):
        async with sem:
            tokens = row["tokens"]
            gold   = row["ner_tags"]
            dirty  = row["dirty_tags"]
            cfg = dict(config)
            cfg["__dataset__"] = dataset   # so the pipeline knows the ontology
            cfg["__noise_type__"] = noise  # so the Coder can adapt strategy per noise
            if dummy:
                cfg["__gold__"] = gold
                cfg["__seed__"] = seed
            if launch_strict:
                cfg.update({
                    "official": True,
                    "structured_requester": getattr(adapter, "structured_requester", None),
                    # Retain the stage-specific adapter attributes for
                    # preflight and compatibility callers; official graph
                    # stages use structured_requester.
                    "ror_reasoner": getattr(adapter, "ror_reasoner", None),
                    "gasd_reason_decoder": getattr(adapter, "gasd_reason_decoder", None),
                    "provider_metadata": dict(provider_identity),
                })
            # Log the Coder candidate pool for the main methods only (oracle /
            # selection analysis); keeps ablation prediction files lean.
            if _supports_candidate_evidence(config_name, config):
                cfg["__return_candidates__"] = True
            extra: dict = {}

            async def _call():
                method = cfg.get("method")
                if method and method in baseline_fns:
                    fn = baseline_fns[method]
                    try:
                        return await fn(tokens, dirty, cfg, dataset_name=dataset)
                    except TypeError:
                        return await fn(tokens, dirty, cfg)
                if launch_strict:
                    return await default_fn(tokens, dirty, cfg, dataset_name=dataset)
                try:
                    return await default_fn(tokens, dirty, cfg, dataset_name=dataset)
                except TypeError:
                    return await default_fn(tokens, dirty, cfg)

            try:
                # Per-request timeout: a single straggler sentence (e.g. very long
                # MSRA input) must not stall the whole cell's async gather.
                with sentence_context(dataset=dataset, noise=noise, seed=seed,
                                      row_index=row_indices[i] if is_paid_smoke and row_indices is not None else i):
                    pred = await asyncio.wait_for(_call(), timeout=request_timeout)
            except asyncio.TimeoutError:
                if failure_policy == "abort" or config.get("terminal_decoder") == "contextual-lattice-v1":
                    raise RuntimeError(
                        f"sentence {i} exceeded the {request_timeout:g}-second request timeout"
                    )
                logging.warning(f"[timeout] sentence {i} exceeded "
                                f"{request_timeout}s; using dirty as fallback")
                pred = list(dirty)
            except Exception as e:                       # noqa: BLE001
                if failure_policy == "abort" or config.get("terminal_decoder") == "contextual-lattice-v1":
                    raise
                logging.error(f"[!] sentence {i} failed ({e!r}); using dirty as fallback")
                pred = list(dirty)
            # Candidate-rich return (dict) → unpack pred_tags + extra fields.
            if isinstance(pred, dict):
                extra = {k: pred[k] for k in
                         ("candidate_paths", "rag_weights", "confidence",
                          "terminal_anchor_tags", "terminal_model_hash", "terminal_used_anchor",
                          "terminal_predicted_gain", "terminal_fallback_count",
                          "ror_reasoning_source", "gasd_variant_requested",
                          "gasd_variant_used", "provider_metadata", "fallback_used")
                         if k in pred}
                pred = pred.get("pred_tags", list(dirty))
            if launch_strict:
                _validate_official_provider_evidence(
                    config, extra.get("provider_metadata"), provider_identity,
                    str(extra.get("ror_reasoning_source")), noise_type=noise,
                )
                common_invalid = (
                    not isinstance(pred, list)
                    or len(pred) != len(gold)
                    or any(tag not in _valid_tags(dataset) for tag in pred)
                    or not _is_legal_iob2(pred)
                    or extra.get("fallback_used") is not False
                    or not isinstance(extra.get("provider_metadata"), dict)
                )
                if official_profile == "contextual-lattice":
                    anchor = extra.get("terminal_anchor_tags")
                    model_hash = extra.get("terminal_model_hash")
                    profile_invalid = (
                        not isinstance(anchor, list) or len(anchor) != len(gold)
                        or any(tag not in _valid_tags(dataset) for tag in anchor)
                        or not _is_legal_iob2(anchor)
                        or not isinstance(model_hash, str)
                        or not re.fullmatch(r"[0-9a-fA-F]{64}", model_hash)
                    )
                else:
                    expected_variant = str(config.get("gasd_variant", "g")).lower()
                    expected_used = "disabled" if not config.get("use_gasd", True) else expected_variant
                    profile_invalid = (
                        extra.get("ror_reasoning_source") not in {"not_triggered", "live", "disabled"}
                        or extra.get("gasd_variant_requested") != expected_variant
                        or extra.get("gasd_variant_used") != expected_used
                    )
                if common_invalid or profile_invalid:
                    raise RuntimeError(f"official sentence {i} lacks valid launch evidence")
            # Length alignment (defensive)
            if not isinstance(pred, list):
                pred = ["O"] * len(gold)
            if len(pred) < len(gold):
                pred = list(pred) + ["O"] * (len(gold) - len(pred))
            elif len(pred) > len(gold):
                pred = pred[:len(gold)]
            rec = {"tokens": tokens, "gold_tags": gold, "pred_tags": pred}
            if extra:
                rec.update(extra)
            buffer[i] = rec

    rate_tag = f"r{int(round(ratio*100))}"
    try:
        if launch_strict:
            tasks = [
                asyncio.create_task(_process(i, row))
                for i, row in enumerate(rows)
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                # Fail the cell as soon as any sentence fails, then drain every
                # task so queued requests cannot continue and no exception is
                # left orphaned. asyncio.run also waits for any already-running
                # to_thread provider call before the adapter is closed.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            try:
                from tqdm.asyncio import tqdm as async_tqdm
                await async_tqdm.gather(
                    *(_process(i, r) for i, r in enumerate(rows)),
                    desc=f"{dataset}/{noise}/seed{seed}/{config_name}/{rate_tag}",
                    leave=False,
                )
            except ImportError:
                await asyncio.gather(*(_process(i, r) for i, r in enumerate(rows)))

        # Write to disk in input order, atomically (via temp file + rename)
        with tmp_p.open("w", encoding="utf-8") as f_out:
            for rec in buffer:
                if rec is None:
                    if launch_strict:
                        raise RuntimeError("official cell produced an incomplete result buffer")
                    continue
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if launch_strict:
            _validate_official_prediction(
                tmp_p, dataset, config_name, noise,
                expected_count=(paid_smoke_size if is_paid_smoke
                                else OFFICIAL_SAMPLE_SIZE),
                provider_identity=provider_identity,
            )
        tmp_p.replace(pred_p)
    except Exception:
        if failure_policy == "abort":
            tmp_p.unlink(missing_ok=True)
        raise

    logging.info(f"[done] {pred_p.name} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",  nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--noise",    nargs="+", default=NOISE_TYPES,
                    choices=NOISE_TYPES)
    ap.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    ap.add_argument("--size",     type=int, default=SAMPLE_SIZE)
    ap.add_argument("--ratios",   nargs="+", type=float, default=[0.15],
                    help="Noise ratios to evaluate. Default: just the legacy 0.15.")
    ap.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--dummy",    action="store_true",
                    help="Use a mock pipeline; no API calls.")
    ap.add_argument("--official", action="store_true",
                    help="Enable fail-closed official LAD-RG launch semantics.")
    ap.add_argument("--failure-policy", choices=("abort", "dirty"), default=None,
                    help="Sentence failure behavior; dirty is legacy opt-in only.")
    ap.add_argument("--request-timeout", type=float, default=None,
                    help="Outer per-sentence timeout in seconds.")
    ap.add_argument("--phase", choices=("end-to-end", "provider-cache", CONTEXTUAL_REPLAY_PHASE), default="end-to-end")
    ap.add_argument("--provider-cache-root", type=Path, default=PROVIDER_CACHE_DIR)
    ap.add_argument("--provider-cache-tag", default=None,
                    help="Immutable provider-cache namespace for staged runs.")
    ap.add_argument(
        "--compatible-provider-cache-git-sha",
        action="append",
        default=[],
        help=("Explicitly allow a provider-cache producer commit whose semantic "
              "pipeline identity has been independently verified as compatible."),
    )
    ap.add_argument("--bundle-hash", help="Pinned 64-hex contextual lattice bundle hash.")
    args = ap.parse_args(argv)
    args.failure_policy = args.failure_policy or "abort"
    args.request_timeout = (
        args.request_timeout if args.request_timeout is not None
        else (OFFICIAL_REQUEST_TIMEOUT if args.official else float(PER_REQUEST_TIMEOUT))
    )
    if args.request_timeout <= 0:
        ap.error("--request-timeout must be positive")
    return args


def _run_contextual_replay_phase(args) -> None:
    """Run the complete local Stage B without constructing a provider adapter."""
    global BACKBONE_TAG
    _validate_contextual_replay_launch(
        args.configs, size=args.size, ratios=args.ratios,
        max_concurrency=args.max_concurrency, failure_policy=args.failure_policy,
        dummy=args.dummy, datasets=args.datasets, noise_types=args.noise,
        seeds=args.seeds,
    )
    if not isinstance(args.bundle_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", args.bundle_hash):
        raise ValueError("contextual-replay requires --bundle-hash with 64 hexadecimal characters")
    tag = str(args.provider_cache_tag or BACKBONE_TAG).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("contextual-replay requires a valid --provider-cache-tag")
    if qwen_enable_thinking_from_environment(os.environ) is not False:
        raise ValueError("contextual-replay requires QWEN_ENABLE_THINKING=false")
    if "nothink" not in tag.lower():
        raise ValueError("contextual-replay requires a fresh tag containing 'nothink'")
    BACKBONE_TAG = tag
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    old_bundle = os.environ.get("CONTEXTUAL_LATTICE_BUNDLE")
    old_embedding = os.environ.get("CONTEXTUAL_LATTICE_ENCODER_CACHE")
    try:
        with _offline_contextual_replay_environment():
            bundle_path, embedding_path = _validate_contextual_replay_bundle(args.bundle_hash)
            os.environ["CONTEXTUAL_LATTICE_BUNDLE"] = str(bundle_path)
            os.environ["CONTEXTUAL_LATTICE_ENCODER_CACHE"] = str(embedding_path)
            from multi_agent_v2 import (
                _load_contextual_lattice_terminal,
                run_contextual_lattice_replay,
            )
            terminal = _load_contextual_lattice_terminal()
            for dataset in args.datasets:
                for noise in args.noise:
                    for seed in args.seeds:
                        asyncio.run(_run_contextual_replay_cell(
                            dataset, noise, seed, args.size,
                            cache_root=args.provider_cache_root,
                            cache_tag=tag, bundle_hash=args.bundle_hash,
                            git_sha=git_sha, replay_fn=run_contextual_lattice_replay,
                            terminal=terminal, max_concurrency=args.max_concurrency,
                            enable_thinking=False,
                            compatible_git_shas=args.compatible_provider_cache_git_sha,
                        ))
    finally:
        if old_bundle is None:
            os.environ.pop("CONTEXTUAL_LATTICE_BUNDLE", None)
        else:
            os.environ["CONTEXTUAL_LATTICE_BUNDLE"] = old_bundle
        if old_embedding is None:
            os.environ.pop("CONTEXTUAL_LATTICE_ENCODER_CACHE", None)
        else:
            os.environ["CONTEXTUAL_LATTICE_ENCODER_CACHE"] = old_embedding


def main(argv=None):
    global BACKBONE_TAG
    args = _parse_args(argv)

    if args.phase == CONTEXTUAL_REPLAY_PHASE:
        _run_contextual_replay_phase(args)
        return

    if args.phase == "provider-cache":
        if not args.official:
            raise ValueError("provider-cache phase requires --official")
        _validate_provider_cache_launch(
            args.configs, size=args.size, ratios=args.ratios,
            max_concurrency=args.max_concurrency, failure_policy=args.failure_policy,
            dummy=args.dummy, datasets=args.datasets, noise_types=args.noise,
            seeds=args.seeds,
        )
        if not isinstance(args.bundle_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", args.bundle_hash):
            raise ValueError("provider-cache phase requires --bundle-hash with 64 hexadecimal characters")

    official_settings = None
    adapter_factory = None
    if args.official:
        if args.dummy:
            raise ValueError("--official cannot be combined with --dummy")
        if args.failure_policy != "abort":
            raise ValueError("official mode does not permit dirty fallback")
        if args.size != OFFICIAL_SAMPLE_SIZE:
            raise ValueError("official mode requires --size 200")
        if len(args.ratios) != 1 or abs(args.ratios[0] - OFFICIAL_NOISE_RATIO) >= 1e-12:
            raise ValueError("official mode requires exactly --ratios 0.15")
        official_settings, BACKBONE_TAG = _official_settings_from_env(args.configs, os.environ)
        _configure_official_request_model(official_settings)
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        if args.phase != "provider-cache":
            manifest = _build_official_manifest(
                official_settings, BACKBONE_TAG, git_sha, args.request_timeout,
            )
            _ensure_official_manifest(manifest, PRED_DIR, BACKBONE_TAG)
        from live_backbone import OpenAICompatibleLADRGAdapter

        adapter_factory = _CachedAdapterFactory(
            lambda: OpenAICompatibleLADRGAdapter(official_settings)
        )

    # Expand default thread pool so MAX_CONCURRENCY LLM calls fit.
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency))
    asyncio.set_event_loop(loop)

    pipeline_fn = _import_pipeline(args.dummy)

    total = (len(args.configs) * len(args.datasets)
             * len(args.noise) * len(args.seeds) * len(args.ratios))
    logging.info(f"running {total} cells "
                 f"({'DUMMY' if args.dummy else 'REAL'} pipeline)")

    n_done = n_err = 0
    try:
        if args.phase == "provider-cache":
            git_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout.strip()
            for ds in args.datasets:
                for nt in args.noise:
                    for seed in args.seeds:
                        asyncio.run(_run_provider_cache_cell(
                            "selectdenoise_contextual_lattice",
                            CONFIGURATIONS["selectdenoise_contextual_lattice"],
                            ds, nt, seed, args.size, pipeline_fn,
                            cache_root=args.provider_cache_root, cache_tag=BACKBONE_TAG,
                            max_concurrency=args.max_concurrency,
                            request_timeout=args.request_timeout,
                            adapter_factory=adapter_factory, git_sha=git_sha,
                            bundle_hash=args.bundle_hash,
                            compatible_git_shas=args.compatible_provider_cache_git_sha,
                        ))
                        n_done += 1
            return
        for cfg_name in args.configs:
            if cfg_name not in CONFIGURATIONS:
                logging.error(f"Unknown config: {cfg_name}; "
                              f"known: {list(CONFIGURATIONS)}")
                continue
            cfg = CONFIGURATIONS[cfg_name]
            for ds in args.datasets:
                for nt in args.noise:
                    for s in args.seeds:
                        for r in args.ratios:
                            try:
                                asyncio.run(_run_one_cell(
                                    cfg_name, cfg, ds, nt, s, args.size,
                                    pipeline_fn,
                                    max_concurrency=args.max_concurrency,
                                    dummy=args.dummy,
                                    ratio=r,
                                    official=args.official,
                                    failure_policy=args.failure_policy,
                                    request_timeout=args.request_timeout,
                                    adapter_factory=adapter_factory,
                                ))
                                n_done += 1
                            except Exception:                    # noqa: BLE001
                                logging.error("[!] cell raised")
                                traceback.print_exc(file=sys.stderr)
                                n_err += 1
                                if args.official:
                                    raise
    finally:
        if adapter_factory is not None:
            adapter_factory.close()

    logging.info(f"summary: {n_done} ok, {n_err} failed")


if __name__ == "__main__":
    main()
