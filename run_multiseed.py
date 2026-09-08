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
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from official_contract import official_manifest_decoder_constants

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
OFFICIAL_REQUEST_TIMEOUT = 600.0
OFFICIAL_SAMPLE_SIZE = 200
PAID_SMOKE_SIZE = 20
OFFICIAL_NOISE_RATIO = 0.15
OFFICIAL_MANIFEST_SCHEMA = "lad-rg-official-run-v2"
OFFICIAL_QWEN_MODEL = "Qwen/Qwen3-32B-AWQ"
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
    settings = LiveBackboneSettings(
        provider=provider,
        model=model,
        base_url=str(environment["BACKBONE_BASE_URL"]).strip(),
        api_key=key,
        revision=revision,
    )
    _endpoint_origin(settings.base_url)
    _validate_official_settings_for_configs(settings, config_names)
    return settings, tag


def _validate_official_settings_for_configs(settings, config_names: Sequence[str]) -> None:
    for config_name in config_names:
        config = CONFIGURATIONS.get(config_name)
        if config is None or config.get("terminal_graph") != "lad-rg":
            raise ValueError(f"official mode requires LAD-RG config, got {config_name!r}")
    if settings.provider == "vllm" and settings.model != OFFICIAL_QWEN_MODEL:
        raise ValueError(f"vLLM official runs require BACKBONE_MODEL={OFFICIAL_QWEN_MODEL}")
    if settings.provider == "deepseek" and any(
        str(CONFIGURATIONS[name].get("gasd_variant", "g")).lower() in {"r", "both"}
        for name in config_names
    ):
        raise ValueError("DeepSeek official runs reject GASD-R/Both; use vllm/Qwen")


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
            request_timeout, structured_api=settings.structured_api
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


async def _run_one_cell(config_name: str, config: dict,
                        dataset: str, noise: str, seed: int,
                        size: int, pipelines,
                        *, max_concurrency: int, dummy: bool,
                        ratio: float = 0.15, official: bool = False,
                        failure_policy: str = "abort",
                        request_timeout: float = PER_REQUEST_TIMEOUT,
                        adapter_factory: Optional[Callable[[], Any]] = None,
                        paid_smoke_size: Optional[int] = None,
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
            paid_smoke_size=paid_smoke_size, prediction_path=prediction_path,
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
    if official and is_paid_smoke:
        raise ValueError("official publication and paid smoke modes are mutually exclusive")
    if official and prediction_path is not None:
        raise ValueError("official predictions must use the canonical tagged namespace")
    if is_paid_smoke:
        if paid_smoke_size != PAID_SMOKE_SIZE:
            raise ValueError(f"paid smoke requires exactly {PAID_SMOKE_SIZE} records")
        if prediction_path is None:
            raise ValueError("paid smoke requires a dedicated prediction_path")
        if dummy:
            raise ValueError("paid smoke cannot use --dummy")
        if failure_policy != "abort":
            raise ValueError("paid smoke requires failure_policy='abort'")
        if config.get("terminal_graph") != "lad-rg":
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
        if config.get("terminal_graph") != "lad-rg":
            raise ValueError("official mode requires a LAD-RG configuration")
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
        rows = rows[:paid_smoke_size]
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
                          "terminal_model_hash", "terminal_used_anchor",
                          "terminal_predicted_gain", "terminal_fallback_count",
                          "ror_reasoning_source", "gasd_variant_requested",
                          "gasd_variant_used", "provider_metadata", "fallback_used")
                         if k in pred}
                pred = pred.get("pred_tags", list(dirty))
            if launch_strict:
                expected_variant = str(config.get("gasd_variant", "g")).lower()
                expected_used = "disabled" if not config.get("use_gasd", True) else expected_variant
                _validate_official_provider_evidence(
                    config, extra.get("provider_metadata"), provider_identity,
                    str(extra.get("ror_reasoning_source")), noise_type=noise,
                )
                if (
                    not isinstance(pred, list)
                    or len(pred) != len(gold)
                    or any(tag not in _valid_tags(dataset) for tag in pred)
                    or not _is_legal_iob2(pred)
                    or extra.get("ror_reasoning_source") not in {"not_triggered", "live", "disabled"}
                    or extra.get("gasd_variant_requested") != expected_variant
                    or extra.get("gasd_variant_used") != expected_used
                    or extra.get("fallback_used") is not False
                    or not isinstance(extra.get("provider_metadata"), dict)
                ):
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
    args = ap.parse_args(argv)
    args.failure_policy = args.failure_policy or "abort"
    args.request_timeout = (
        args.request_timeout if args.request_timeout is not None
        else (OFFICIAL_REQUEST_TIMEOUT if args.official else float(PER_REQUEST_TIMEOUT))
    )
    if args.request_timeout <= 0:
        ap.error("--request-timeout must be positive")
    return args


def main(argv=None):
    global BACKBONE_TAG
    args = _parse_args(argv)

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
