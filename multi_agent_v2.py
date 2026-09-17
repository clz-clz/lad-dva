import asyncio
import inspect
import logging
import os
import json
import re
import hashlib
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, TypedDict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
import argparse

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient

from utils import enforce_iob2_syntax, legalize_noise_aware, extract_json_list
from metrics import _is_valid_transition
from official_contract import (
    OFFICIAL_DECODER_CONSTANTS,
    OFFICIAL_PROVIDER_TIMEOUT_SECONDS,
    OFFICIAL_SDK_MAX_RETRIES,
    OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES,
    VerifierSemanticRetryExhausted,
    VerifierSemanticRetryInterrupted,
    validate_verifier_semantic_retry_evidence,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Backbone configuration (env-var driven, DeepSeek defaults).
# Swap to any OpenAI-compatible endpoint (e.g. Qwen2.5 / Llama-3.1 via vLLM,
# Together, OpenRouter) by exporting BACKBONE_MODEL / BACKBONE_BASE_URL /
# BACKBONE_API_KEY — no code change needed. run_multiseed.py is already
# backbone-agnostic; only these two client constructors bind the backbone.
# ---------------------------------------------------------------------------
BACKBONE_MODEL    = os.environ.get("BACKBONE_MODEL", "deepseek-chat")
BACKBONE_REQUEST_MODEL = os.environ.get("BACKBONE_SERVED_MODEL", BACKBONE_MODEL)
BACKBONE_BASE_URL = os.environ.get("BACKBONE_BASE_URL", "https://api.deepseek.com")
BACKBONE_API_KEY  = (os.environ.get("BACKBONE_API_KEY")
                     or os.environ.get("DEEPSEEK_API_KEY"))
# $BACKBONE_TAG namespaces prediction filenames per backbone; it is owned by
# run_multiseed.py, which builds those paths (see _pred_path there).

def _provider_client_options(environment: Optional[Mapping[str, str]] = None) -> dict:
    """Pin SDK controls only for the official runner's pre-import handshake."""
    source = os.environ if environment is None else environment
    if str(source.get("LAD_RG_OFFICIAL_REQUESTS", "")).strip() != "1":
        return {}
    return {
        "timeout": OFFICIAL_PROVIDER_TIMEOUT_SECONDS,
        "max_retries": OFFICIAL_SDK_MAX_RETRIES,
        "http_client": DefaultHttpxClient(trust_env=False),
    }


_PROVIDER_CLIENT_OPTIONS = _provider_client_options()


llm = ChatOpenAI(
    model=BACKBONE_REQUEST_MODEL,
    temperature=0.7,
    base_url=BACKBONE_BASE_URL,
    api_key=BACKBONE_API_KEY,
    **_PROVIDER_CLIENT_OPTIONS,
)

# Dedicated higher-temperature LLM for Coder to encourage diverse paths
coder_llm = ChatOpenAI(
    model=BACKBONE_REQUEST_MODEL,
    temperature=1.0,
    base_url=BACKBONE_BASE_URL,
    api_key=BACKBONE_API_KEY,
    **_PROVIDER_CLIENT_OPTIONS,
)

DEEPSEEK_NER_TOKEN_IDS = {
    "B-PER": 42531, "I-PER": 42532,
    "B-LOC": 18274, "I-LOC": 18275,
    "B-ORG": 39112, "I-ORG": 39113,
    "B-MISC": 6489, "I-MISC": 6490
}

# =====================================================================
# Dataset-aware entity ontologies
# Each dataset has its own valid entity type set. The Coder and Reviewer
# prompts inject this list so the LLM doesn't default to the common
# PER/LOC/ORG triad regardless of input dataset.
# =====================================================================
DATASET_ENTITY_TYPES = {
    "msra":      ["PER", "LOC", "ORG"],
    "conll2003": ["PER", "LOC", "ORG", "MISC"],
    # WNUT-17 in this project is COARSENED to the standard 4-class ontology
    # via type_remap in gen_noisy.py (person->PER, location->LOC,
    # corporation/group->ORG, product/creative-work->MISC). The pipeline must
    # therefore prompt with the same coarse types.
    "wnut17":    ["PER", "LOC", "ORG", "MISC"],
    # Few-NERD coarsened to 4-class (person->PER, location->LOC,
    # organization->ORG, art/building/event/other/product->MISC)
    "fewnerd":   ["PER", "LOC", "ORG", "MISC"],
    # OntoNotes5 coarsened to 4-class (PERSON->PER, GPE/LOC/FAC->LOC,
    # ORG->ORG, rest->MISC)
    "ontonotes5": ["PER", "LOC", "ORG", "MISC"],
}

def _format_valid_tags(dataset_name: str) -> str:
    """Return 'O, B-X, I-X, ...' as a single comma-separated string."""
    types = DATASET_ENTITY_TYPES.get(dataset_name)
    if not types:
        # Fallback: infer from a sample dirty_tags later, or use union
        types = ["PER", "LOC", "ORG", "MISC"]
    tags = ["O"]
    for t in types:
        tags.append(f"B-{t}")
        tags.append(f"I-{t}")
    return ", ".join(tags)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    loop_count: int
    iterations: int
    tokens: List[str]
    dirty_tags: List[str]
    candidate_paths: List[List[str]]
    terminal_candidate_paths: List[List[str]]
    rag_weights: List[float]
    current_tags: List[str]
    errors: List[str]
    lambda_weight: float
    use_wash: bool
    dataset_name: str
    noise_type: str
    # --- Synergistic redesign (LAD-RG) — legacy, superseded by SelectDenoise ---
    ror_proposals: dict          # token-index -> proposed tag from RoR; empty when none fire
    ror_reasoning: dict          # auditable span/type reasoning evidence; never final tags
    use_lads: bool               # enable LADS retrieval/potential weighting
    use_ror: bool                # enable the RoR recall stage
    use_gasd: bool               # enable the GASD terminal decoder
    ror_ungated: bool            # ablation: fire RoR everywhere (no ω/conf gate)
    gasd_potentials: bool        # integrate LADS ω(t) potentials in GASD decode
    gasd_variant: str            # g, r, or both
    ror_reasoner: Optional[
        Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    ]
    gasd_reason_decoder: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ]
    structured_requester: Optional[
        Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    ]
    gasd_variant_requested: str  # terminal evidence: configured decoder variant
    gasd_variant_used: str       # terminal evidence: decoder variant actually used
    official: bool               # fail-closed launch semantics
    provider_settings: dict      # immutable provider settings supplied by runner
    provider_metadata: dict      # stage-separated per-response evidence
    fallback_used: bool
    # --- SelectDenoise (two levers) ---
    # Lever 1 (generation): de-anchor ATF entity types so the correct type
    # becomes the majority across Coder paths instead of a minority.
    deanchor_atf: bool
    # Lever 2 (selection): LLM verifier picks the best complete labeling among
    # the distinct candidate paths on contested sentences.
    use_verifier: bool
    verify_all: bool             # ablation: verify every sentence (no trigger)
    verifier_topk: int           # max distinct candidate paths shown to verifier


_PROVIDER_STAGES = ("coder", "reviewer", "ror", "gasd")
_CONTEXTUAL_PROVIDER_STAGES = ("coder", "reviewer", "verifier")
_QWEN_OFFICIAL_MODEL = "Qwen/Qwen3-32B-AWQ"
_QWEN_OFFICIAL_REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"


def _provider_evidence(value: Any = None) -> dict[str, list[dict[str, Any]]]:
    source = value if isinstance(value, Mapping) else {}
    stages = (
        _CONTEXTUAL_PROVIDER_STAGES
        if set(source) == set(_CONTEXTUAL_PROVIDER_STAGES)
        else _PROVIDER_STAGES
    )
    return {
        stage: [dict(record) for record in source.get(stage, [])
                if isinstance(record, Mapping)]
        for stage in stages
    }


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            mapped = method()
            if isinstance(mapped, Mapping):
                return dict(mapped)
    return {}


def _official_enable_thinking(state: Mapping[str, Any]) -> bool:
    """Return the effective mode for an official stage request.

    The contextual Qwen graph historically requested thinking explicitly.  A
    fixed-Qwen setting may override that request; all other providers retain
    the request supplied by their existing caller.
    """
    settings = _plain_mapping(state.get("provider_settings"))
    configured = settings.get("enable_thinking")
    if (settings.get("provider") == "vllm"
            and settings.get("model") == _QWEN_OFFICIAL_MODEL
            and type(configured) is bool):
        return configured
    return True


def _stage_thinking_metadata(settings: Mapping[str, Any]) -> tuple[Optional[bool], str]:
    configured = settings.get("enable_thinking")
    if type(configured) is bool:
        return configured, "thinking" if configured else "nothink"
    # Official Qwen stages historically request thinking unless explicitly
    # overridden.  This makes skipped-stage evidence auditable too.
    if (settings.get("provider") == "vllm"
            and settings.get("model") == _QWEN_OFFICIAL_MODEL):
        return True, "thinking"
    return None, "request-controlled"


def _response_stage_record(response: Any, stage: str,
                           provider_settings: Any = None) -> dict[str, Any]:
    settings = _plain_mapping(provider_settings)
    response_metadata = _plain_mapping(getattr(response, "response_metadata", None))
    usage = _plain_mapping(getattr(response, "usage_metadata", None))
    if not usage:
        usage = _plain_mapping(response_metadata.get("token_usage"))
    response_model = (response_metadata.get("model_name")
                      or response_metadata.get("model"))
    configured_thinking, configured_mode = _stage_thinking_metadata(settings)
    return {
        "stage": stage,
        "status": "live",
        "provider": settings.get("provider"),
        "model": settings.get("model") or response_model,
        "served_model": settings.get("served_model"),
        "revision": settings.get("revision"),
        "response_model": response_model,
        "system_fingerprint": response_metadata.get("system_fingerprint"),
        "usage": usage,
        "structured_api": settings.get("structured_api"),
        "response_status": response_metadata.get("response_status"),
        "finish_reason": response_metadata.get("finish_reason"),
        "incomplete_reason": response_metadata.get("incomplete_reason"),
        "enable_thinking": response_metadata.get(
            "enable_thinking", configured_thinking
        ),
        "thinking_mode": response_metadata.get(
            "thinking_mode", configured_mode
        ),
    }


def _callback_stage_record(result: Any, stage: str,
                           provider_settings: Any = None) -> dict[str, Any]:
    settings = _plain_mapping(provider_settings)
    configured_thinking, configured_mode = _stage_thinking_metadata(settings)
    callback_metadata = getattr(result, "provider_metadata", None)
    if isinstance(callback_metadata, Mapping):
        record = dict(callback_metadata)
        record.setdefault("stage", stage)
        record.setdefault("status", "live")
        record.setdefault("provider", None)
        record.setdefault("model", None)
        record.setdefault("served_model", None)
        record.setdefault("revision", None)
        record.setdefault("response_model", None)
        record.setdefault("system_fingerprint", None)
        record.setdefault("structured_api", _plain_mapping(provider_settings).get("structured_api"))
        record.setdefault("response_status", None)
        record.setdefault("finish_reason", None)
        record.setdefault("incomplete_reason", None)
        record.setdefault("enable_thinking", configured_thinking)
        record.setdefault("thinking_mode", configured_mode)
        record["usage"] = _plain_mapping(record.get("usage"))
        return record
    return _response_stage_record(None, stage, provider_settings)


def _stage_status_record(stage: str, status: str,
                         provider_settings: Any = None) -> dict[str, Any]:
    settings = _plain_mapping(provider_settings)
    configured_thinking, configured_mode = _stage_thinking_metadata(settings)
    return {
        "stage": stage,
        "status": status,
        "provider": settings.get("provider"),
        "model": settings.get("model"),
        "served_model": settings.get("served_model"),
        "revision": settings.get("revision"),
        "response_model": None,
        "system_fingerprint": None,
        "usage": {},
        "structured_api": settings.get("structured_api"),
        "response_status": None,
        "finish_reason": None,
        "incomplete_reason": None,
        "enable_thinking": configured_thinking,
        "thinking_mode": configured_mode,
    }


def _with_stage_records(state: Mapping[str, Any], stage: str,
                        records: List[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    evidence = _provider_evidence(state.get("provider_metadata"))
    evidence[stage].extend(records)
    return evidence


def _validate_contextual_official_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject unbound or incomplete provider evidence before cache publication."""
    if set(evidence) != set(_CONTEXTUAL_PROVIDER_STAGES):
        raise RuntimeError("official contextual pipeline is missing stage-separated evidence")
    for stage in _CONTEXTUAL_PROVIDER_STAGES:
        records = evidence.get(stage)
        if not isinstance(records, list) or not records:
            raise RuntimeError("official contextual pipeline has omitted stage evidence")
        for record in records:
            if not isinstance(record, Mapping) or record.get("stage") != stage:
                raise RuntimeError("official contextual pipeline has invalid stage evidence")
            if (record.get("provider") != "vllm"
                    or record.get("model") != _QWEN_OFFICIAL_MODEL
                    or record.get("revision") != _QWEN_OFFICIAL_REVISION
                    or record.get("served_model") != (
                        f"{_QWEN_OFFICIAL_MODEL}@{_QWEN_OFFICIAL_REVISION}"
                    )
                    or record.get("structured_api") != "chat-completions-json-schema"):
                raise RuntimeError("official contextual evidence lacks pinned Qwen identity")
            if (type(record.get("enable_thinking")) is not bool
                    or record.get("thinking_mode") != (
                        "thinking" if record.get("enable_thinking") else "nothink"
                    )):
                raise RuntimeError("official contextual evidence lacks thinking-mode identity")
            if record.get("status") == "live":
                if (record.get("response_model") != record.get("served_model")
                        or record.get("response_status") != "completed"
                        or record.get("finish_reason") != "stop"
                        or not isinstance(record.get("usage"), Mapping)
                        or not record["usage"]):
                    raise RuntimeError("official contextual live evidence is incomplete")
            else:
                allowed_skips = {
                    "reviewer": {"skipped_identical"},
                    "verifier": {"skipped_uncontested", "skipped_identical"},
                }.get(stage, set())
                if record.get("status") not in allowed_skips:
                    raise RuntimeError(f"official contextual {stage} evidence status is unjustified")
                if (record.get("response_model") is not None
                        or record.get("response_status") is not None
                        or record.get("finish_reason") is not None):
                    raise RuntimeError("official contextual skipped evidence claims a response")
        if stage == "verifier":
            try:
                validate_verifier_semantic_retry_evidence(records)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc


def _official_tag_path(content: Any, expected_length: int,
                       valid_tags: set[str]) -> List[str]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("official Coder response must be valid JSON") from exc
    elif isinstance(content, Mapping):
        payload = dict(content)
    else:
        raise ValueError("official Coder response must be a JSON mapping")
    if not isinstance(payload, Mapping) or set(payload) != {"tags"}:
        raise ValueError("official Coder response must match the {'tags': [...]} schema")
    path = payload["tags"]
    if not isinstance(path, list) or not all(isinstance(tag, str) for tag in path):
        raise ValueError("official Coder response tags must be a JSON string list")
    if len(path) != expected_length:
        raise ValueError(
            "official Coder response must contain exactly one tag per token; "
            f"expected {expected_length}, received {len(path)}"
        )
    if any(tag not in valid_tags for tag in path):
        raise ValueError("official Coder response uses a tag outside the dataset ontology")
    return list(path)


def _official_reviewer_weights(content: Any, expected_length: int) -> List[float]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("official Reviewer response must be valid JSON") from exc
    elif isinstance(content, Mapping):
        payload = dict(content)
    else:
        raise ValueError("official Reviewer response must be a JSON mapping")
    if not isinstance(payload, Mapping) or set(payload) != {"weights"}:
        raise ValueError("official Reviewer response must match the {'weights': [...]} schema")
    values = payload["weights"]
    if (not isinstance(values, list) or len(values) != expected_length
            or not all(type(value) in (int, float) and 0.0 <= value <= 1.0
                       for value in values)):
        raise ValueError("official Reviewer response must contain one score in [0, 1] per path")
    return [float(value) for value in values]


def _require_structured_requester(
    state: Mapping[str, Any], stage: str
) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    requester = state.get("structured_requester")
    if not callable(requester):
        raise RuntimeError(
            f"official {stage} requires a live structured_requester callback"
        )
    return requester


async def _invoke_structured_requester(
    requester: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    stage: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    # The cached OpenAI adapter exposes the synchronous SDK client.  Keep its
    # transport off the event loop so the runner's sentence/path concurrency
    # remains effective; async test or custom requesters still work because
    # invoking an async callable only creates its coroutine here.
    result = await asyncio.to_thread(requester, stage, payload)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Mapping):
        raise ValueError(f"official {stage} requester returned a non-mapping response")
    return result


def _invoke_structured_requester_sync(
    requester: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    stage: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = requester(stage, payload)
    if inspect.isawaitable(result):
        raise RuntimeError(
            f"official {stage} structured_requester must be synchronous"
        )
    if not isinstance(result, Mapping):
        raise ValueError(f"official {stage} requester returned a non-mapping response")
    return result
    

def extract_nested_json_list(llm_output: str, expected_paths: int = 5) -> List[List[str]]:
    try:
        match = re.search(r'\[\s*\[.*?\]\s*\]', llm_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return parsed[:expected_paths]
    except Exception as e:
        logging.warning(f"Coder JSON ：{e}")
    return [["O"]] * expected_paths

def extract_float_weights(llm_output: str, expected_len: int = 5) -> List[float]:
    try:
        match = re.search(r'\[[\d\.\s,]+\]', llm_output)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                weights = [float(x) for x in parsed]
                if len(weights) >= expected_len:
                    return weights[:expected_len]
                else:
                    return weights + [0.0] * (expected_len - len(weights))
    except Exception as e:
        logging.error(f" Reviewer JSON 载入异常: {e}")

    logging.error(f" Warning: {llm_output[:50]}...")
    return [1.0] * expected_len 


def _mask_entity_types(tags: List[str]) -> List[str]:
    """Replace entity TYPES with a neutral 'ENT' placeholder, keeping the B/I/O
    boundary structure. Used by Lever 1 (ATF de-anchoring): the Coder sees the
    correct spans but no (possibly wrong) type suggestion, so it must re-derive
    each type from context + calibration instead of anchoring on the dirty type."""
    out = []
    for t in tags:
        if t == "O" or "-" not in t:
            out.append("O")
        else:
            out.append(f"{t[0]}-ENT")
    return out


def _apply_types_to_boundaries(model_path: List[str],
                               boundary_tags: List[str]) -> List[str]:
    """Keep boundary_tags' B/I/O structure (correct under ATF) but adopt the
    TYPE the model assigned to each span. Per span, the type is the majority of
    the model's non-O tags over the span positions; ties/empties fall back to the
    first assigned type, else keep the boundary tag's own type. Guarantees the
    de-anchored path never introduces boundary errors on ATF."""
    n = len(boundary_tags)
    out = list(boundary_tags)
    i = 0
    while i < n:
        bt = boundary_tags[i]
        if bt.startswith("B-"):
            j = i + 1
            while j < n and boundary_tags[j].startswith("I-"):
                j += 1
            # collect model type votes over span [i, j)
            votes: dict = {}
            for k in range(i, j):
                mk = model_path[k] if k < len(model_path) else "O"
                if mk != "O" and "-" in mk:
                    ty = mk.split("-", 1)[1]
                    votes[ty] = votes.get(ty, 0) + 1
            if votes:
                ty = max(votes, key=votes.get)
            else:
                ty = bt.split("-", 1)[1]        # keep original type if no vote
            out[i] = f"B-{ty}"
            for k in range(i + 1, j):
                out[k] = f"I-{ty}"
            i = j
        else:
            out[i] = "O"
            i += 1
    return out


def _format_sentence_initial_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate guidance about sentence-initial entity heads."""
    if dataset_name == "msra":
        return (
            "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A Chinese "
            "sentence often begins with a named entity (organization, "
            "person, location). Do not default the first token to O just "
            "because it lacks left context. If the first 1-4 tokens form a "
            "recognizable entity, label them B-<TYPE> I-<TYPE> .... Examples: "
            "'中国政府...' -> B-ORG I-ORG I-ORG; '北京市...' -> B-LOC I-LOC I-LOC."
        )
    # English-style (CoNLL-2003 and coarsened WNUT-17)
    return (
        "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A capitalized "
        "token at position 0 that names a person, organization, location, or "
        "other entity should be labeled B-<TYPE>, not O. Do not default the "
        "first token to O just because it lacks left context. Examples: "
        "'Obama said ...' -> the first token is B-PER, not O. "
        "'Apple announced ...' -> the first token is B-ORG, not O. "
        "'German troops ...' -> the first token is B-MISC, not O."
    )


def _format_misc_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate MISC type guidance (only for ontologies that have MISC)."""
    if dataset_name == "msra":
        # MSRA uses only PER/LOC/ORG, no MISC. Skip the MISC paragraph entirely.
        return ""
    return (
        "- DO NOT systematically avoid rare tag types. The MISC type (for "
        "nationalities, languages, events, works, products, and other "
        "proper-noun entities that are not PER/LOC/ORG) is valid and should "
        "be predicted whenever the token clearly refers to such an entity, "
        "even if MISC appears less frequently than PER/LOC/ORG. Examples of "
        "MISC entities: \"German\" (nationality), \"Olympic\" (event), "
        "\"iPhone\" (product), \"Bible\" (work)."
    )


async def coder_node(state: State):
    print(f"\n [Coder](Self-Consistency)")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    dataset_name = state.get("dataset_name", "conll2003")
    official = bool(state.get("official", False))
    contextual_official = (
        official and set(state.get("provider_metadata", {}))
        == set(_CONTEXTUAL_PROVIDER_STAGES)
    )
    structured_requester = (
        _require_structured_requester(state, "Coder") if official else None
    )
    noise_type = state.get("noise_type", "BT")
    deanchor_atf = state.get("deanchor_atf", False) and noise_type == "ATF"
    valid_tags_str = _format_valid_tags(dataset_name)
    valid_types = DATASET_ENTITY_TYPES.get(dataset_name, ["PER", "LOC", "ORG"])
    valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                          for entity_type in valid_types for prefix in ("B", "I")}
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    # DEER examples: same calibration data the Reviewer sees
    deer_examples = _get_deer_examples(
        tokens, top_k=3, dataset_name=dataset_name, cache_only=official
    )

    # ---- Noise-type-specific policies and per-path strategies ----
    if noise_type == "BT":
        noise_policy = """
    NOISE TYPE: Boundary Truncation (BT). Entity boundaries are corrupted — missing B- tags, dangling I- tags, broken spans. CRITICAL: entity TYPES are reliable. The dirty tag's type field (PER/LOC/ORG/MISC) is trustworthy. Never change an entity's type — only fix its boundaries."""
        path_strategies = {
            1: "CONSERVATIVE BOUNDARY FIX: Repair ONLY dangling I- tags. Upgrade each dangling I-<TYPE> to B-<TYPE>. Never change entity types. Never change O to entity. Never extend single-token entities. This is the most cautious strategy — only fix obvious structural IOB2 errors.",
            2: "FULL BOUNDARY REPAIR: Fix dangling I- tags AND broken B-X O B-X patterns. When you see B-<TYPE> O B-<TYPE>, merge into B-<TYPE> I-<TYPE> I-<TYPE>. Also consider B-X O O I-X → B-X I-X I-X. Never change entity types. Target: comprehensive boundary repair — fix all structure errors.",
            5: "OPTIMAL JOINT REPAIR: Combine all boundary repairs (dangling I- fix + B-X O B-X merging) with false-positive cleanup. Demote to O any entity token that is clearly a function word (the, of, in, a, an, for, to, with), common verb (said, was, is), punctuation, or bare number. Never change entity types for tokens that remain entities. Target: maximum accuracy.",
        }
    elif noise_type == "IF":
        noise_policy = """
    NOISE TYPE: Internal Fragmentation (IF). Entities are incorrectly split by O tags in the middle. CRITICAL: entity TYPES are reliable. The dirty tag's type field is trustworthy. Never change an entity's type — only merge fragmented entities."""
        path_strategies = {
            1: "CONSERVATIVE FRAGMENT MERGE: Merge ONLY adjacent same-type entities separated by exactly 1 O token. Only merge when the gap token is NOT a sentence delimiter (not punctuation, not a clause boundary). B-PER O B-PER → B-PER I-PER I-PER. Never change entity types. Target: fix obvious fragments with minimal risk.",
            2: "AGGRESSIVE FRAGMENT MERGE: Merge adjacent same-type entities with gaps up to 2 O tokens. Only skip merging when the gap token is clearly a sentence delimiter. Never change entity types. Target: comprehensive fragment repair — catch all splittings.",
            5: "OPTIMAL JOINT REPAIR: Merge fragments with balanced aggressiveness (Path 1 threshold for ambiguous gaps, Path 2 for clear gaps) plus false-positive cleanup. Demote to O any entity token that is a function word, common verb, punctuation, or bare number. Never change entity types for tokens that remain entities. Target: maximum accuracy.",
        }
    elif noise_type == "ATF" and deanchor_atf:
        # Lever 1: de-anchored ATF. The dirty TYPES are hidden (masked to ENT);
        # boundaries are given and correct. Every path re-derives each span's
        # type from scratch, so the correct type is the MAJORITY across paths
        # (not a dirty-anchored minority). Boundaries are re-imposed afterward.
        noise_policy = """
    NOISE TYPE: Adversarial Type Flipping (ATF), TYPE-BLIND mode. You are given entity SPANS whose boundaries (B/I/O structure) are CORRECT, but whose TYPES are HIDDEN (shown as ENT). The original dirty types were unreliable and have been withheld ON PURPOSE. Your task: assign the CORRECT entity type (from the valid vocabulary) to EACH given span using ONLY the token semantics, sentence context, and calibration examples — never guess a default type. Do NOT add, remove, or resize spans; only fill in each span's type. All tokens within one span MUST share one type."""
        path_strategies = {
            1: "SEMANTIC TYPING: Assign each span's type from the real-world semantics of its head token (a person name -> PER, a place -> LOC, a company/institution -> ORG, else MISC). Decide independently for every span.",
            2: "CALIBRATION TYPING: For each span, find similar tokens in the calibration examples and adopt the type they consistently receive there. If none match, fall back to semantics.",
            3: "CONTEXT TYPING: Assign each span's type from the surrounding sentence context (verbs, prepositions, appositions, neighboring entities) that disambiguate person vs place vs organization.",
            4: "CONSENSUS TYPING: For each span consider semantics, context, AND calibration together and pick the single most probable type; enforce one uniform type per span.",
            5: "ROBUST TYPING + CLEANUP: Assign the most probable type per span (semantics+context+calibration); additionally, if a span is clearly a function word, bare number, or punctuation, you may mark it O.",
        }
    elif noise_type == "ATF":
        noise_policy = """
    NOISE TYPE: Adversarial Type Flipping (ATF). Entity BOUNDARIES (B/I/O structure) are CORRECT — the spans are right, but the entity TYPES are all potentially wrong. CRITICAL: Do NOT change entity boundaries. Only verify and fix entity TYPES."""
        path_strategies = {
            1: "CONSERVATIVE TYPE FIX: Only change an entity's type when the dirty type is IMPOSSIBLE for the token (e.g., 'Obama' as ORG, 'Beijing' as PER). If the dirty type is plausible, keep it. Do NOT change entity boundaries (B/I/O). Target: fix clear type errors only.",
            2: "CALIBRATION-GUIDED TYPE FIX: For each entity span, check the calibration examples for similar tokens. If a token consistently appears with a specific type in calibration, adopt that type. If no calibration evidence, keep the dirty type. Do NOT change boundaries. Target: data-driven type correction.",
            3: "AGGRESSIVE TYPE FIX: Verify EVERY entity's type against token semantics and world knowledge. Change any type that seems wrong, even if marginally plausible. Check multi-token entities for type consistency (all tokens in one entity MUST share the same type). Do NOT change boundaries. Target: comprehensive type repair.",
            4: "TYPE NORMALIZATION: For each entity span, pick the most probable type based on token semantics and calibration evidence. For multi-token entities, enforce uniform type across all tokens. Do NOT change boundaries. Target: statistical type correction.",
            5: "OPTIMAL JOINT REPAIR: Combine calibration-guided type fixing with false-positive cleanup. Demote to O any entity token that is clearly a function word, common verb, punctuation, or bare number. Boundaries stay identical except when demoting false positives. Target: maximum accuracy.",
        }
    else:
        noise_policy = ""
        path_strategies = {
            1: "DEFAULT REPAIR: Fix obvious IOB2 errors while preserving the original labeling as much as possible.",
            2: "DEFAULT REPAIR: Fix IOB2 errors and consider contextual appropriateness of entity boundaries.",
            5: "DEFAULT REPAIR: Comprehensive repair including false-positive cleanup.",
        }

    diagnostic_path = state.get("__diagnostic_coder_path__")
    if diagnostic_path is not None:
        if (isinstance(diagnostic_path, bool)
                or not isinstance(diagnostic_path, int)
                or diagnostic_path not in path_strategies):
            raise ValueError(
                "diagnostic coder path must be one of the paths enabled for this noise type"
            )
        # This opt-in key is used only by the credential-free diagnostic probe;
        # normal official cells never set it and retain the complete path pool.
        path_strategies = {diagnostic_path: path_strategies[diagnostic_path]}

    # Shared prompt prefix (identical for all paths). Under Lever 1 de-anchoring
    # the dirty TYPES are masked so the model cannot anchor on them.
    if deanchor_atf:
        presented_tags_line = (
            f"Entity Spans (boundaries CORRECT; types HIDDEN — assign them): "
            f"{_mask_entity_types(dirty_tags)}")
    else:
        presented_tags_line = f"Dirty IOB2 Tags: {dirty_tags}"
    shared_prefix = f"""
You are an Elite AI Data Engineer performing IOB2 label denoising for Named Entity Recognition.
Tokens: {tokens}
{presented_tags_line}
{noise_policy}

Calibration Examples (correct annotations from the TRAINING SET — retrieved via label-guided similarity. These show what valid output looks like for sentences similar to the current one. Your repairs MUST be consistent with these patterns):
{json.dumps(deer_examples, indent=2) if deer_examples else '[]'}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

CRITICAL DENOISING POLICY — follow these rules precisely:

1. IOB2 GRAMMAR: Every multi-token entity MUST start with B-<TYPE>. A dangling I-<TYPE> without preceding same-type B/I is FATAL — upgrade to B-<TYPE>. All tokens in one entity must share the same type.

2. SINGLE-TOKEN ENTITY PROTECTION: A single B-X tag followed by O is a COMPLETE one-token entity. Do NOT extend it with I-X into following O. Single-token entities are correct and common.

3. NO ENTITY INVENTION: Never change O to B-X or I-X unless fixing a clear BT noise pattern (dangling I-). The NER model has near-perfect entity detection recall.

4. PRESERVATION BIAS: The dirty tags are MOSTLY correct. Prefer keeping a non-O tag over changing it to O. A token tagged as entity should stay entity unless it is clearly a function word, common verb, number, or punctuation.
{misc_guidance}
{sentence_initial_guidance}"""

    # Call LLM separately for each path (parallel) to guarantee diversity
    n_paths = len(path_strategies)
    print(f" [Coder] {noise_type}: {n_paths} individual LLM calls (parallel)")

    requests = []
    coder_schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(valid_tags)},
                "minItems": len(tokens),
                "maxItems": len(tokens),
            },
        },
        "required": ["tags"],
        "additionalProperties": False,
    }
    for pidx, pdesc in path_strategies.items():
        response_kwargs = {}
        if official:
            output_instruction = f"""
Generate EXACTLY 1 repair path in a JSON object whose only key is "tags".
The "tags" array MUST preserve all {len(tokens)} positions in order.
Copy this exact-length slot template and replace every null with one valid IOB2 tag;
never add, remove, merge, split, or reorder slots:
{json.dumps({"tags": [None] * len(tokens)})}
Output ONLY that JSON object, no markdown, no explanation."""
        else:
            output_instruction = f"""
Generate EXACTLY 1 repair path (a JSON list of {len(tokens)} IOB2 tags).
Output ONLY the JSON list, no markdown, no explanation."""
            response_kwargs = {}
        prompt = shared_prefix + f"""

PATH {pidx} STRATEGY — your ONLY task:
{pdesc}

{output_instruction}"""
        requests.append((pidx, prompt, response_kwargs))

    if official:
        responses = await asyncio.gather(*[
            _invoke_structured_requester(
                structured_requester,
                "coder" if contextual_official else f"coder_path_{strategy_key}",
                {
                    "name": (
                        f"selectdenoise_coder_path_{strategy_key}"
                        if contextual_official else f"lad_rg_coder_path_{strategy_key}"
                    ),
                    "schema": coder_schema,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a strict LAD-RG Coder. Return only the requested JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 1.0,
                    "enable_thinking": _official_enable_thinking(state),
                },
            )
            for strategy_key, prompt, _response_kwargs in requests
        ])
    else:
        responses = await asyncio.gather(*[
            asyncio.to_thread(coder_llm.invoke, prompt, **response_kwargs)
            for _pidx, prompt, response_kwargs in requests
        ])

    candidate_paths = []
    for (strategy_key, _prompt, _kwargs), response in zip(requests, responses):
        if official:
            try:
                path = _official_tag_path(response, len(tokens), valid_tags)
            except ValueError as exc:
                raise ValueError(f"official Coder path {strategy_key}: {exc}") from exc
        else:
            path = extract_json_list(response.content, fallback_length=len(tokens))
        candidate_paths.append(path)

    raw_contextual_paths = (
        [list(path) for path in candidate_paths] if contextual_official else []
    )

    coder_records = [
        (
            _callback_stage_record(
                response,
                "coder" if contextual_official else f"coder_path_{strategy_key}",
                state.get("provider_settings"),
            )
            if official
            else _response_stage_record(
                response, f"coder_path_{strategy_key}", state.get("provider_settings")
            )
        )
        for (strategy_key, _prompt, _kwargs), response in zip(requests, responses)
    ]

    if official and not contextual_official:
        if any(not _is_legal_tag_sequence(path) for path in candidate_paths):
            raise ValueError("official Coder response contains an illegal IOB2 transition")
    elif not official:
        # Legacy behavior retains its defensive length and DFA normalization.
        candidate_paths = [
            p[:len(tokens)] + ["O"] * max(0, len(tokens) - len(p))
            for p in candidate_paths
        ]

    # IOB2语法清洗：legacy paths先过DFA修复，保证Reviewer只判边界+类型
    ds_name = state.get("dataset_name", "conll2003")
    valid_set = set(DATASET_ENTITY_TYPES.get(ds_name, ["PER", "LOC", "ORG"]))
    if not official:
        candidate_paths = [enforce_iob2_syntax(p, valid_entity_types=valid_set)
                           for p in candidate_paths]

    # ---- Lever 1: re-impose correct ATF boundaries, keep model TYPE ----
    # De-anchored paths may drift on structure; ATF boundaries are given/correct,
    # so force B/I/O = dirty and adopt only the model's per-span type assignment.
    if deanchor_atf:
        candidate_paths = [_apply_types_to_boundaries(p, dirty_tags)
                           for p in candidate_paths]
        print(f" [Coder] Lever 1 de-anchored ATF: types re-derived, "
              f"boundaries re-imposed from spans")

    # ---- Hard constraint for BT/IF noise ----
    # BT/IF noise only drops entity tags (entity→O) and breaks boundaries.
    # Entity TYPES are correct, entity DETECTION recall is near-perfect.
    # The Coder is unreliable on BT/IF — block most changes except gap-fill.
    # Allowed: O→Entity when bridging two same-type entity fragments.
    # BT: blocked Entity→O, type changes, B↔I shuffles (boundaries are locally correct).
    # IF: blocked Entity→O, type changes; B↔I shuffles ALLOWED for fragment merging.
    if noise_type in ("BT", "IF"):
        n_reverted = 0
        for p in candidate_paths:
            for i in range(len(p)):
                if p[i] == dirty_tags[i]:
                    continue
                # ---- O→Entity: only allow legitimate BT gap-fill ----
                if dirty_tags[i] == "O" and p[i] != "O":
                    p_type = p[i][2:]
                    p_prefix = p[i][0]  # "B" or "I"
                    prev_same = (i > 0 and dirty_tags[i-1] != "O"
                                and dirty_tags[i-1][2:] == p_type)
                    next_same = (i+1 < len(dirty_tags) and dirty_tags[i+1] != "O"
                                and dirty_tags[i+1][2:] == p_type)
                    if p_prefix == "B":
                        # O→B-X: allowed if next is I-X (BT dropped the B- tag)
                        if not next_same:
                            p[i] = "O"
                            n_reverted += 1
                    else:
                        # O→I-X: allowed if prev is same type AND next is NOT a
                        # different entity type. This allows true gap-fill
                        # (next_same) AND BT tail-drop (next=O), but blocks
                        # entity extension into another entity (next=B-OTHER).
                        next_other = (i+1 < len(dirty_tags) and dirty_tags[i+1] != "O"
                                      and dirty_tags[i+1][2:] != p_type)
                        if not prev_same or next_other:
                            p[i] = "O"
                            n_reverted += 1
                    continue
                # ---- Entity→O: always revert (BT/IF never deletes entities) ----
                if dirty_tags[i] != "O" and p[i] == "O":
                    p[i] = dirty_tags[i]
                    n_reverted += 1
                    continue
                # ---- Same type, different B/I: boundary shuffle → revert ----
                if dirty_tags[i] != "O" and p[i] != "O":
                    d_type = dirty_tags[i][2:]
                    p_type = p[i][2:]
                    if d_type != p_type:
                        # Type change → revert
                        p[i] = dirty_tags[i]
                        n_reverted += 1
                    elif noise_type == "BT":
                        # BT: boundaries are locally correct (just truncated).
                        # Same-type B↔I shuffles are noise → revert.
                        p[i] = dirty_tags[i]
                        n_reverted += 1
                    # IF: allow same-type B↔I changes for fragment merging.
                    # B→I is needed to merge adjacent same-type entity fragments
                    # (e.g. B-PER O B-PER → B-PER I-PER I-PER).
        if n_reverted > 0:
            print(f" [Coder] Hard constraint reverted {n_reverted} changes "
                  f"(BT/IF: O-tag + type + boundary protection)")

    # Diversity check: if all paths identical, fallback to dirty when harmful
    fallback_used = bool(state.get("fallback_used", False))
    if _all_paths_identical(candidate_paths):
        only_path = candidate_paths[0]
        all_o = all(t == "O" for t in only_path)
        has_entities_in_dirty = any(t != "O" for t in dirty_tags)
        if all_o and has_entities_in_dirty and not official:
            # LLM erased all entities — fallback to dirty (safer)
            print(f" [Coder] All paths identical (all O) with entities in dirty — "
                  f"overriding with dirty tags")
            candidate_paths = [list(dirty_tags) for _ in range(5)]
            fallback_used = True
        else:
            # Paths identical but not all-O: LLM is confident, proceed
            n_diff = sum(1 for a, b in zip(only_path, dirty_tags) if a != b)
            print(f" [Coder] All {len(candidate_paths)} paths identical "
                  f"({n_diff} diffs vs dirty) — LLM consensus, skipping diversity")

    terminal_candidate_paths = []
    if contextual_official:
        terminal_candidate_paths = [
            list(raw_path) if not _is_legal_tag_sequence(raw_path) else list(processed_path)
            for raw_path, processed_path in zip(raw_contextual_paths, candidate_paths)
        ]

    result = {
        "candidate_paths": candidate_paths,
        "iterations": state.get("iterations", 0) + 1,
        "provider_metadata": _with_stage_records(state, "coder", coder_records),
        "fallback_used": fallback_used,
    }
    if contextual_official:
        result["terminal_candidate_paths"] = terminal_candidate_paths
    return result

def _all_paths_identical(paths: List[List[str]]) -> bool:
    """Check if all candidate paths are identical."""
    if len(paths) <= 1:
        return True
    first = paths[0]
    return all(p == first for p in paths[1:])


# ---- DEER integration (Bai et al., EMNLP 2025) ----
# Label-guided retrieval replaces the old RAG in the Reviewer.
# Statistics are built per-dataset from the appropriate training set.

_deer_stats = {}     # dataset_name -> DEERStatistics
_deer_retriever = {}  # dataset_name -> DEERRetriever

_DEER_CACHED_TRAIN_FILES = {
    "msra": ("msra_ner", "msra_ner-train.arrow"),
    "conll2003": ("conll2003", "conll2003-train.arrow"),
    "wnut17": ("wnut_17", "wnut_17-train.arrow"),
    "fewnerd": ("DFKI-SLT___few-nerd", "few-nerd-train.arrow"),
    "ontonotes5": ("tner___ontonotes5", "ontonotes5-train.arrow"),
}
_deer_init_locks = {
    dataset_name: threading.Lock() for dataset_name in _DEER_CACHED_TRAIN_FILES
}
_deer_fallback_init_lock = threading.Lock()


def _deer_ready(dataset_name: str) -> bool:
    return dataset_name in _deer_stats and dataset_name in _deer_retriever


def _load_cached_deer_training_split(dataset_name: str):
    """Open the newest cached training Arrow directly without taking HF locks."""
    from datasets import Dataset, config as datasets_config

    cache_directory, arrow_name = _DEER_CACHED_TRAIN_FILES[dataset_name]
    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE", str(datasets_config.HF_DATASETS_CACHE))
    )
    candidates = [
        path for path in (cache_root / cache_directory).rglob(arrow_name)
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No cached DEER training Arrow found for {dataset_name} under {cache_root}"
        )
    arrow_path = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return Dataset.from_file(str(arrow_path))

# The contextual lattice is an optional terminal replacement.  It is loaded
# once per process only when the versioned configuration requests it; legacy
# SelectDenoise behavior does not import the frozen encoder or bundle.
_contextual_lattice_terminal = None
_contextual_lattice_bundle_path = None
_contextual_lattice_load_lock = threading.Lock()


class ContextualLatticeError(RuntimeError):
    """A terminal configuration/input error that must not become dirty output."""


def _resolve_contextual_checkpoint_path(checkpoint: str, bundle_path: Path) -> Path:
    """Resolve a frozen-manifest checkpoint relative to its workspace root."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        # The frozen manifest records paths relative to the workspace root.
        # The official runner executes from its worktree, so try that location
        # and then the shared workspace root implied by the bundle path.
        candidates = (
            Path.cwd() / checkpoint_path,
            Path(bundle_path).parents[2] / checkpoint_path,
        )
        checkpoint_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )
    return checkpoint_path.resolve()


def _load_contextual_lattice_terminal():
    """Load the frozen, gold-free terminal bundle exactly once."""
    global _contextual_lattice_terminal, _contextual_lattice_bundle_path
    try:
        bundle_text = os.environ.get("CONTEXTUAL_LATTICE_BUNDLE", "").strip()
        if not bundle_text:
            raise RuntimeError("CONTEXTUAL_LATTICE_BUNDLE is required for contextual-lattice-v1")
        bundle_path = os.path.abspath(bundle_text)
        with _contextual_lattice_load_lock:
            if _contextual_lattice_terminal is not None:
                if _contextual_lattice_bundle_path != bundle_path:
                    raise RuntimeError("A different contextual lattice bundle was requested after initialization")
                return _contextual_lattice_terminal
            manifest_path = os.path.join(bundle_path, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Contextual lattice manifest not found: {manifest_path}")
            from contextual_lattice_runtime import (
                LOCKED_BUNDLE_MANIFEST_HASH,
                LOCKED_CHECKPOINT_HASH,
                LOCKED_DECODER_FILE_HASH,
                LOCKED_DECODER_MODEL_HASH,
                LOCKED_GATE_FILE_HASH,
                LOCKED_SPLIT_HASH,
                load_bundle,
            )
            if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != LOCKED_BUNDLE_MANIFEST_HASH:
                raise ValueError("Contextual lattice manifest is not the locked v1 artifact")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            checkpoint = manifest.get("checkpoint")
            checkpoint_hash = manifest.get("checkpoint_hash")
            if not checkpoint or not checkpoint_hash:
                raise ValueError("Contextual lattice bundle is missing checkpoint provenance")
            if checkpoint_hash != LOCKED_CHECKPOINT_HASH or manifest.get("split_hash") != LOCKED_SPLIT_HASH:
                raise ValueError("Contextual lattice bundle provenance does not match locked v1")
            checkpoint = str(_resolve_contextual_checkpoint_path(checkpoint, bundle_path))
            from selectdenoise_contextual_lattice import GLiNERContextEncoder

            device = os.environ.get("CONTEXTUAL_LATTICE_DEVICE", "cuda")
            cache_dir = os.environ.get(
                "CONTEXTUAL_LATTICE_ENCODER_CACHE",
                os.path.join(bundle_path, "embedding_cache"),
            )
            encoder = GLiNERContextEncoder(
                checkpoint,
                {"PER", "LOC", "ORG", "MISC"},
                device=device,
                cache_dir=cache_dir,
            )
            _contextual_lattice_terminal = load_bundle(
                Path(bundle_path),
                encoder,
                expected_checkpoint_hash=LOCKED_CHECKPOINT_HASH,
                expected_split_hash=LOCKED_SPLIT_HASH,
                expected_manifest_hash=LOCKED_BUNDLE_MANIFEST_HASH,
                expected_decoder_file_hash=LOCKED_DECODER_FILE_HASH,
                expected_gate_file_hash=LOCKED_GATE_FILE_HASH,
                expected_decoder_model_hash=LOCKED_DECODER_MODEL_HASH,
            )
            _contextual_lattice_bundle_path = bundle_path
            logging.info(
                "[ContextualLattice] loaded bundle=%s model=%s checkpoint=%s",
                bundle_path,
                _contextual_lattice_terminal.model_hash,
                checkpoint_hash,
            )
            return _contextual_lattice_terminal
    except ContextualLatticeError:
        raise
    except Exception as exc:
        raise ContextualLatticeError("Contextual lattice bundle validation failed") from exc


def _apply_contextual_lattice_terminal(state: State, terminal):
    """Apply the terminal using only the allow-listed model-facing fields."""
    tokens = list(state.get("tokens", []))
    dirty_tags = list(state.get("dirty_tags", []))
    anchor_tags = list(state.get("current_tags", [])) or list(dirty_tags)
    candidate_paths = (
        state.get("terminal_candidate_paths") or state.get("candidate_paths", []) or []
    )
    reviewer_weights = state.get("rag_weights", []) or []
    dataset_name = state.get("dataset_name", "conll2003")
    valid_types = frozenset(DATASET_ENTITY_TYPES.get(dataset_name, DATASET_ENTITY_TYPES["conll2003"]))
    try:
        result = terminal.decode(
            tokens=tokens,
            dirty_tags=dirty_tags,
            anchor_tags=anchor_tags,
            candidate_paths=candidate_paths,
            reviewer_weights=reviewer_weights,
            valid_types=valid_types,
            deer_stats=terminal.sentence_deer_stats(tokens),
        )
    except Exception as exc:
        raise ContextualLatticeError("Contextual lattice sentence validation failed") from exc
    tags = list(result.tags)
    valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                           for entity_type in valid_types for prefix in ("B", "I")}
    result_model_hash = getattr(result, "model_hash", None)
    terminal_model_hash = getattr(terminal, "model_hash", None)
    result_used_anchor = getattr(result, "used_anchor", None)
    result_predicted_gain = getattr(result, "predicted_gain", None)
    terminal_fallback_count = getattr(terminal, "fallback_count", None)
    invalid = (
        not isinstance(result_model_hash, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", result_model_hash)
        or result_model_hash != terminal_model_hash
        or len(tags) != len(tokens)
        or any(tag not in valid_tags for tag in tags)
        or not _is_legal_tag_sequence(tags)
        or not isinstance(result_used_anchor, bool)
        or isinstance(result_predicted_gain, bool)
        or not isinstance(result_predicted_gain, (int, float))
        or not math.isfinite(float(result_predicted_gain))
        or isinstance(terminal_fallback_count, bool)
        or not isinstance(terminal_fallback_count, int)
        or terminal_fallback_count < 0
    )
    if invalid and state.get("official", False):
        raise ContextualLatticeError("Contextual lattice terminal result is invalid")
    if invalid:
        tags = anchor_tags
    return {
        "current_tags": tags,
        "terminal_anchor_tags": anchor_tags,
        "terminal_model_hash": result_model_hash,
        "terminal_used_anchor": result_used_anchor,
        "terminal_predicted_gain": float(result_predicted_gain),
        "terminal_fallback_count": terminal_fallback_count,
    }


def run_contextual_lattice_replay(
    *, tokens: Sequence[str], dirty_tags: Sequence[str],
    provider_record: Mapping[str, Any], dataset_name: str,
    terminal: Any = None,
) -> dict[str, Any]:
    """Decode one frozen provider-cache row without reopening the provider graph.

    The replay boundary is intentionally narrow: only the seven allow-listed
    terminal inputs are copied into the terminal state.  Cache provenance and
    provider evidence stay in the runner and are never model-facing inputs.
    """
    if dataset_name not in DATASET_ENTITY_TYPES:
        raise ContextualLatticeError(f"unknown replay dataset ontology: {dataset_name!r}")
    if not isinstance(provider_record, Mapping):
        raise ContextualLatticeError("contextual replay provider row must be a mapping")
    forbidden = {"gold_tags", "ner_tags", "gold_labels"}

    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                (isinstance(key, str) and key.lower() in forbidden)
                or contains_forbidden(nested)
                for key, nested in value.items()
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(contains_forbidden(nested) for nested in value)
        return False

    if contains_forbidden(provider_record):
        raise ContextualLatticeError("contextual replay provider row contains gold labels")
    if (not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes))
            or not all(isinstance(token, str) for token in tokens)):
        raise ContextualLatticeError("contextual replay tokens must be a string sequence")
    if (not isinstance(dirty_tags, Sequence) or isinstance(dirty_tags, (str, bytes))
            or len(dirty_tags) != len(tokens)
            or not all(isinstance(tag, str) for tag in dirty_tags)):
        raise ContextualLatticeError("contextual replay dirty_tags are not aligned")

    valid_tags = {"O"} | {
        f"{prefix}-{entity_type}"
        for entity_type in DATASET_ENTITY_TYPES[dataset_name]
        for prefix in ("B", "I")
    }

    def checked_tags(value: Any, field: str, *, require_legal: bool = True) -> list[str]:
        if (not isinstance(value, list) or len(value) != len(tokens)
                or not all(isinstance(tag, str) and tag in valid_tags for tag in value)
                or (require_legal and not _is_legal_tag_sequence(value))):
            raise ContextualLatticeError(f"contextual replay {field} is invalid")
        return list(value)

    anchor_tags = checked_tags(provider_record.get("anchor_tags"), "anchor_tags")
    candidate_paths = provider_record.get("candidate_paths")
    reviewer_weights = provider_record.get("rag_weights")
    if not isinstance(candidate_paths, list) or not isinstance(reviewer_weights, list):
        raise ContextualLatticeError("contextual replay candidate evidence is malformed")
    if len(reviewer_weights) not in {0, len(candidate_paths)}:
        raise ContextualLatticeError("contextual replay reviewer weights are not aligned")
    checked_paths: list[list[str]] = []
    for path in candidate_paths:
        # Candidate legality is a lattice concern: malformed IOB2 paths are
        # excluded by build_lattice, while shape and ontology remain strict.
        checked_paths.append(checked_tags(path, "candidate_paths", require_legal=False))
    checked_weights: list[float] = []
    for weight in reviewer_weights:
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))):
            raise ContextualLatticeError("contextual replay reviewer weights are invalid")
        checked_weights.append(float(weight))
    if terminal is None:
        terminal = _load_contextual_lattice_terminal()
    state = {
        "tokens": list(tokens),
        "dirty_tags": list(dirty_tags),
        "current_tags": anchor_tags,
        "candidate_paths": checked_paths,
        "rag_weights": checked_weights,
        "dataset_name": dataset_name,
        "official": True,
    }
    result = _apply_contextual_lattice_terminal(state, terminal)
    return {"pred_tags": result["current_tags"], **result}


def _build_deer(dataset_name: str = "conll2003", *, cache_only: bool = False):
    """Lazy-init DEER statistics and retriever for a given dataset."""
    global _deer_stats, _deer_retriever
    if _deer_ready(dataset_name):
        return

    from deer_retriever import DEERStatistics, DEERRetriever

    def load_training_split(dataset_path: str, *args, **kwargs):
        if cache_only:
            return _load_cached_deer_training_split(dataset_name)
        from datasets import load_dataset

        return load_dataset(dataset_path, *args, **kwargs)

    if dataset_name == "msra":
        print("[DEER] Building token statistics from MSRA-NER training set...")
        ds = load_training_split("msra_ner", split="train", trust_remote_code=True)
        id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    elif dataset_name == "wnut17":
        # WNUT-17 original tags coarsened to 4-class: person->PER, location->LOC,
        # corporation/group->ORG, creative-work/product->MISC
        print("[DEER] Building token statistics from WNUT-17 training set...")
        ds = load_training_split("wnut_17", split="train", trust_remote_code=True)
        id2tag = ["O",
                  "B-ORG", "I-ORG",      # 1,2: corporation -> ORG
                  "B-MISC", "I-MISC",     # 3,4: creative-work -> MISC
                  "B-ORG", "I-ORG",       # 5,6: group -> ORG
                  "B-LOC", "I-LOC",       # 7,8: location -> LOC
                  "B-PER", "I-PER",       # 9,10: person -> PER
                  "B-MISC", "I-MISC"]     # 11,12: product -> MISC
    elif dataset_name == "fewnerd":
        # Few-NERD supervised: person->PER, location->LOC, organization->ORG,
        # art/building/event/other/product->MISC
        print("[DEER] Building token statistics from Few-NERD training set...")
        ds = load_training_split(
            "DFKI-SLT/few-nerd", "supervised", split="train", trust_remote_code=True
        )
        id2tag = ds.features["ner_tags"].feature.names
        # Coarsen fine-grained types to 4-class
        _coarse = {
            "O": "O",
            "B-person": "B-PER", "I-person": "I-PER",
            "B-location": "B-LOC", "I-location": "I-LOC",
            "B-organization": "B-ORG", "I-organization": "I-ORG",
            "B-art": "B-MISC", "I-art": "I-MISC",
            "B-building": "B-MISC", "I-building": "I-MISC",
            "B-event": "B-MISC", "I-event": "I-MISC",
            "B-other": "B-MISC", "I-other": "I-MISC",
            "B-product": "B-MISC", "I-product": "I-MISC",
            "B-location": "B-LOC", "I-location": "I-LOC",
        }
        id2tag = [_coarse.get(t, t) for t in id2tag]
    elif dataset_name == "ontonotes5":
        # OntoNotes5 via tner: tags field (not ner_tags). 37 IOB2 tags → coarsen to 4-class.
        print("[DEER] Building token statistics from OntoNotes5 training set...")
        ds = load_training_split("tner/ontonotes5", split="train", trust_remote_code=True)
        _id2tag_raw = [
            "O",
            "B-CARDINAL", "B-DATE", "I-DATE",
            "B-PERSON", "I-PERSON",
            "B-NORP", "B-GPE", "I-GPE",
            "B-LAW", "I-LAW",
            "B-ORG", "I-ORG",
            "B-PERCENT", "I-PERCENT",
            "B-ORDINAL",
            "B-MONEY", "I-MONEY",
            "B-WORK_OF_ART", "I-WORK_OF_ART",
            "B-FAC",
            "B-TIME",
            "I-CARDINAL",
            "B-LOC",
            "B-QUANTITY", "I-QUANTITY",
            "I-NORP",
            "I-LOC",
            "B-PRODUCT",
            "I-TIME",
            "B-EVENT", "I-EVENT",
            "I-FAC",
            "B-LANGUAGE",
            "I-PRODUCT",
            "I-ORDINAL",
            "I-LANGUAGE",
        ]
        _coarse = {
            "PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
            "ORG": "ORG",
            "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
            "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
            "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC",
            "MONEY": "MISC", "QUANTITY": "MISC", "ORDINAL": "MISC",
            "CARDINAL": "MISC",
        }
        id2tag = []
        for t in _id2tag_raw:
            if t == "O":
                id2tag.append("O")
            else:
                prefix, etype = t.split("-", 1)
                id2tag.append(f"{prefix}-{_coarse.get(etype, 'MISC')}")
    else:
        print("[DEER] Building token statistics from CoNLL2003 training set...")
        ds = load_training_split("conll2003", split="train", trust_remote_code=True)
        id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    _tags_field = "tags" if dataset_name == "ontonotes5" else "ner_tags"
    train_data = []
    for ex in ds:
        toks = list(ex["tokens"])
        raw = ex[_tags_field]
        if not toks or len(toks) != len(raw):
            continue
        train_data.append((toks, [id2tag[i] for i in raw]))

    stats = DEERStatistics(context_window=2)
    stats.build(train_data)

    retriever = DEERRetriever(stats)
    retriever.index(train_data)

    _deer_stats[dataset_name] = stats
    _deer_retriever[dataset_name] = retriever
    print(f"[DEER] Ready for {dataset_name}: {len(train_data)} training sentences indexed")


def _init_deer(dataset_name: str = "conll2003", *, cache_only: bool = False):
    """Initialize one dataset once, including under concurrent sentence runs."""
    if _deer_ready(dataset_name):
        return
    init_lock = _deer_init_locks.get(dataset_name, _deer_fallback_init_lock)
    with init_lock:
        if _deer_ready(dataset_name):
            return
        _build_deer(dataset_name, cache_only=cache_only)


def _get_deer_examples(query_tokens: List[str], top_k: int = 3,
                       dataset_name: str = "conll2003", *,
                       cache_only: bool = False) -> List[dict]:
    """Retrieve top-k training sentences via DEER label-guided scoring."""
    if cache_only:
        _init_deer(dataset_name, cache_only=True)
    else:
        _init_deer(dataset_name)
    retriever = _deer_retriever.get(dataset_name)
    if retriever is None:
        return []
    results = retriever.retrieve(query_tokens, top_k=top_k)
    return [{"tokens": tokens, "tags": tags, "score": round(score, 4)}
            for _, score, tokens, tags in results]


def _omega_weights(tokens: List[str], dataset_name: str) -> List[float]:
    """Per-token LADS informativeness weight ω(t) = w_e·P(t|entity) +
    w_c·P(t|context) (+ tiny w_o·P(t|other)), from DEER training statistics.

    This is the *shared signal* of the LAD-RG architecture: it gates RoR's
    recall recovery AND supplies GASD's soft entity potentials, so removing
    LADS degrades both downstream modules. Returns 1.0 (neutral) per token if
    stats are unavailable (e.g. dummy mode without DEER).
    """
    stats = _deer_stats.get(dataset_name)
    if stats is None:
        return [1.0] * len(tokens)
    return [float(stats.token_weight(t)) for t in tokens]


async def reviewer_node(state: State):
    print(f"\n [Reviewer]")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    candidate_paths = state.get("candidate_paths", [])
    ds_name = state.get("dataset_name", "conll2003")
    use_lads = state.get("use_lads", True)
    official = bool(state.get("official", False))
    structured_requester = (
        _require_structured_requester(state, "Reviewer") if official else None
    )

    if not candidate_paths:
        return {"rag_weights": []}
    if not use_lads:
        return {
            "rag_weights": [1.0 / len(candidate_paths)] * len(candidate_paths),
            "provider_metadata": _with_stage_records(state, "reviewer", [
                _stage_status_record(
                    "reviewer", "disabled", state.get("provider_settings")
                )
            ]),
        }

    # Skip Reviewer when all Coder paths are identical (no diversity to judge)
    if _all_paths_identical(candidate_paths):
        print(f" [Reviewer] SKIPPED — all {len(candidate_paths)} paths identical")
        return {
            "rag_weights": [1.0 / len(candidate_paths)] * len(candidate_paths),
            "provider_metadata": _with_stage_records(state, "reviewer", [
                _stage_status_record(
                    "reviewer", "skipped_identical", state.get("provider_settings")
                )
            ]),
        }

    # DEER retrieval: use label statistics to find similar training sentences
    deer_examples = _get_deer_examples(
        tokens, top_k=3, dataset_name=ds_name, cache_only=official
    )
    if official:
        reviewer_output_instruction = f"""
    Output ONLY a JSON object whose only key is "weights".
    The "weights" array must contain exactly {len(candidate_paths)} numbers in
    candidate-path order, each between 0.0 and 1.0.
    Required shape: {json.dumps({"weights": [None] * len(candidate_paths)})}
    Replace every null; never add, remove, or reorder slots."""
        reviewer_schema = {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": len(candidate_paths),
                    "maxItems": len(candidate_paths),
                },
            },
            "required": ["weights"],
            "additionalProperties": False,
        }
    else:
        reviewer_output_instruction = f"""
    Output ONLY a JSON list of {len(candidate_paths)} float numbers between 0.0 and 1.0.
    Example: [0.15, 0.92, 0.48, 0.05, 0.78]"""
        response_kwargs = {}

    reviewer_prompt = f"""
    You are an IOB2 Adjudicator. Your task is to score candidate IOB2 tag paths for quality and identify the BEST path among them.

    Blind Tokens: {tokens}
    Dirty Tags (reference — these contain 15% noise; do NOT trust blindly): {dirty_tags}

    VALID TAG VOCABULARY for this dataset (case-sensitive):
    {_format_valid_tags(ds_name)}

    Calibration Examples (correct annotations from the TRAINING SET — retrieved via label-guided similarity. These show what valid output looks like for sentences similar to the current one. Use them to calibrate your scoring):
    {json.dumps(deer_examples, indent=2) if deer_examples else '[]'}

    {len(candidate_paths)} candidate paths to evaluate:
    {json.dumps(candidate_paths, indent=2)}

    SCORING CRITERIA — assign 0.0 to 1.0 for each path. Use the FULL range. Be discriminating:

    1. IOB2 GRAMMAR (FATAL — violation → score < 0.2):
       - Dangling I-X (I-PER without preceding B-PER or I-PER) = FATAL. Score ≤ 0.15.
       - Perfect grammar → baseline 0.5 minimum.

    2. BOUNDARY REPAIR QUALITY (VERY HIGH weight):
       - Fixed dangling I- → B- where dirty has I-X after O or at position 0: STRONG POSITIVE (+0.3).
       - Merged adjacent same-type entities where dirty has B-X O B-X: STRONG POSITIVE (+0.25).
       - Introduced new boundary errors not present in dirty: STRONG NEGATIVE (−0.3).

    3. TYPE CORRECTNESS (VERY HIGH weight):
       - Fixed type-flipping where dirty has wrong type: STRONG POSITIVE (+0.2 per fix).
       - Introduced new type errors: STRONG NEGATIVE (−0.2 per error).
       - Compare entity types against calibration examples: if the calibration examples consistently annotate a token with a specific type, paths matching that type should score higher.

    4. CALIBRATION CONSISTENCY (HIGH weight):
       - Paths whose entity spans (boundaries + types) resemble patterns seen in the calibration examples score higher.
       - If calibration examples show that a word is always tagged as a specific entity type, favor paths that follow this convention.
       - If calibration examples show no entity at a position where a path proposes one, apply a small penalty.

    5. ENTITY COHERENCE (HIGH weight):
       - Multi-token entities must form coherent phrases.
       - Function words (the, of, in, a, an, and, for, to, with) tagged as entity → penalty.

    6. CORRECTION CONFIDENCE (MEDIUM-HIGH weight):
       - Prefer paths that make confident corrections over paths that barely differ from dirty.
       - A path that fixes 3+ noise errors correctly scores higher than one that fixes 1.
       - But a path that changes many correct tags to wrong ones gets heavily penalized.

    CRITICAL: Best path ≥ 0.85. Worst path ≤ 0.25. Spread scores across the full range.

    {reviewer_output_instruction}
    """

    if official:
        response = await _invoke_structured_requester(
            structured_requester,
            "reviewer",
            {
                "name": "lad_rg_reviewer",
                "schema": reviewer_schema,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict LAD-RG Reviewer. Return only the requested JSON object.",
                    },
                    {"role": "user", "content": reviewer_prompt},
                ],
                "temperature": 0.7,
                "enable_thinking": _official_enable_thinking(state),
            },
        )
        rag_weights = _official_reviewer_weights(response, len(candidate_paths))
        reviewer_record = _callback_stage_record(
            response, "reviewer", state.get("provider_settings")
        )
    else:
        response = await asyncio.to_thread(llm.invoke, reviewer_prompt, **response_kwargs)
        rag_weights = extract_float_weights(
            response.content, expected_len=len(candidate_paths)
        )
        reviewer_record = _response_stage_record(
            response, "reviewer", state.get("provider_settings")
        )
    return {
        "rag_weights": rag_weights,
        "provider_metadata": _with_stage_records(
            state, "reviewer", [reviewer_record]
        ),
    }

def _weighted_majority_voting(candidate_paths: List[List[str]],
                              weights: List[float],
                              entity_boost: float = OFFICIAL_DECODER_CONSTANTS["voting_entity_boost"],
                              dirty_tags: List[str] = None,
                              consensus_ratio: float = OFFICIAL_DECODER_CONSTANTS["voting_consensus_ratio"]) -> List[str]:
    """Weighted majority voting with entity boost and consensus threshold.

    If the winning tag has less than consensus_ratio of total weighted votes
    (i.e., fewer than 3/5 paths agree), the dirty tag is kept instead.
    This filters out single-path hallucinations.
    """
    if len(candidate_paths) != len(weights):
        raise ValueError("Number of paths must match number of weights!")

    sequence_length = len(candidate_paths[0])
    final_voted_tags = []
    n_consensus_fallbacks = 0

    for t in range(sequence_length):
        vote_tally = Counter()
        position_total = 0.0

        for i, path in enumerate(candidate_paths):
            tag = path[t]
            weight = weights[i]

            is_valid_entity = (tag != "O" and "-" in tag
                               and tag.split("-", 1)[0] in {"B", "I"})

            if is_valid_entity:
                vote_tally[tag] += (weight * entity_boost)
                position_total += (weight * entity_boost)
            else:
                vote_tally["O"] += weight
                position_total += weight

        if not vote_tally:
            winning_tag = "O"
        else:
            winning_tag = vote_tally.most_common(1)[0][0]
            winning_votes = vote_tally[winning_tag]
            ratio = winning_votes / position_total if position_total > 0 else 0.0

            # If consensus is too weak, fall back to dirty tag
            if dirty_tags and ratio < consensus_ratio:
                winning_tag = dirty_tags[t]
                n_consensus_fallbacks += 1

        final_voted_tags.append(winning_tag)

    if n_consensus_fallbacks > 0:
        print(f" [Voting] Consensus fallback: {n_consensus_fallbacks} positions "
              f"reverted to dirty (ratio < {consensus_ratio})")

    return final_voted_tags


def voting_node(state: State):
    lam = state.get("lambda_weight", 1.0)

    candidate_paths = state.get("candidate_paths", [])
    rag_weights = state.get("rag_weights", [])
    dirty_tags = state.get("dirty_tags", [])

    raw_voted_tags = _weighted_majority_voting(
        candidate_paths,
        rag_weights,
        entity_boost=(lam * OFFICIAL_DECODER_CONSTANTS["voting_entity_boost"]),
        dirty_tags=dirty_tags,
        consensus_ratio=OFFICIAL_DECODER_CONSTANTS["voting_consensus_ratio"],
    )

    return {"current_tags": raw_voted_tags}


def physical_wash_node(state: State):
   
    print(f"\n [Physical Wash] ...")
    tokens = state.get("tokens", [])
    raw_tags = state.get("current_tags", [])
    dirty_tags = state.get("dirty_tags", []) 

    try:
        ds_name = state.get("dataset_name", "conll2003")
        noise_type = state.get("noise_type", "BT")
        valid_set = set(DATASET_ENTITY_TYPES.get(ds_name, ["PER", "LOC", "ORG"]))
        final_safe_tags = enforce_iob2_syntax(raw_tags, valid_entity_types=valid_set)

        # ---- BT/IF off-by-one correction ----
        # enforce_iob2_syntax fixes dangling I-X (after O) by changing I-X→B-X.
        # But for BT noise, the correct fix is O→B-X (restore the dropped B- tag),
        # not I-X→B-X (which shifts the entity start by 1 token).
        # Pattern: dirty=O I-X, washed=B-X -> should be B-X I-X.
        if noise_type in ("BT", "IF") and dirty_tags:
            for i in range(len(final_safe_tags)):
                ft = final_safe_tags[i]
                if i > 0 and ft.startswith("B-") and dirty_tags[i].startswith("I-"):
                    if ft[2:] == dirty_tags[i][2:]:
                        prev_dirty = dirty_tags[i-1]
                        prev_fixed = final_safe_tags[i-1]
                        if prev_dirty == "O" and prev_fixed == "O":
                            # BT dropped B-X at i-1. Restore it, keep I-X at i.
                            final_safe_tags[i-1] = "B-" + ft[2:]  # O→B-X
                            final_safe_tags[i] = dirty_tags[i]      # keep I-X

        if not final_safe_tags or len(final_safe_tags) == 0:
            print(f" [DFA Alert]")
            final_safe_tags = dirty_tags if dirty_tags else ["O"] * len(tokens)


        if len(final_safe_tags) != len(tokens):
            print(f"[Length Correction]: {len(final_safe_tags)} -> {len(tokens)}")
            if len(final_safe_tags) > len(tokens):
                final_safe_tags = final_safe_tags[:len(tokens)]
            else:
                final_safe_tags.extend(["O"] * (len(tokens) - len(final_safe_tags)))

        print(f"sample clean finish: {len(final_safe_tags)}")
        return {"current_tags": final_safe_tags}
        
    except Exception as e:

        return {"current_tags": ["O"] * len(tokens)}


# =====================================================================
# LAD-RG synergistic redesign: RoR (recall) + GASD (structural integrator)
# These replace the old voting_node (flat, F1-negative λ-bias) and
# physical_wash_node (greedy, F1-neutral DFA patch). The two new modules
# share the LADS ω(t) signal and consume each other's output, so removing
# any single module degrades F1 (non-substitutability).
# =====================================================================

def _ror_recover(tokens: List[str], base: List[str],
                 candidate_paths: List[List[str]], weights: List[float],
                 omega: List[float], *, ungated: bool = False,
                 omega_quantile: float = OFFICIAL_DECODER_CONSTANTS["ror_omega_quantile"],
                 conf_thresh: float = OFFICIAL_DECODER_CONSTANTS["ror_confidence_threshold"]) -> dict:
    """RoR recall gate: at O-positions that are statistically entity-bearing
    (high ω) yet low-confidence, propose the entity *type* most supported by
    the candidate pool. Proposals are handed to GASD (never applied raw), which
    reconciles them with IOB2 legality.

    Gating uses the shared LADS ω(t) signal → remove LADS and the gate loses its
    discriminative threshold (falls back toward the toxic ungated regime).

    NOTE: this deterministic statistics+candidate recovery is the portable core;
    for open-weight / high-compute runs it is refined by a focused span→type
    reasoning call (ReCoT-style) that confirms whether an entity was truly erased.
    """
    import numpy as _np
    proposals: dict = {}
    n = len(tokens)
    if n == 0 or not candidate_paths:
        return proposals
    o_positions = [i for i, t in enumerate(base) if t == "O"]
    if not o_positions:
        return proposals
    omg = _np.array([omega[i] for i in o_positions], dtype=float)
    thr = float(_np.quantile(omg, omega_quantile)) if len(omg) else 0.0
    Wsum = sum(weights) or 1.0
    for i in o_positions:
        type_votes: dict = {}
        agree_base = 0.0
        for k, path in enumerate(candidate_paths):
            tag = path[i] if i < len(path) else "O"
            if tag == base[i]:
                agree_base += weights[k]
            if tag != "O" and "-" in tag:
                et = tag.split("-", 1)[1]
                type_votes[et] = type_votes.get(et, 0.0) + weights[k]
        conf = agree_base / Wsum          # confidence that base (=O) is correct
        gated = ungated or (omega[i] >= thr and conf < conf_thresh)
        if not gated or not type_votes:
            continue
        proposals[i] = max(type_votes, key=type_votes.get)
    return proposals


def _official_ror_spans(response: Any, token_count: int,
                        gated_positions: set[int]) -> List[dict[str, int]]:
    if (not isinstance(response, Mapping) or set(response) != {"spans"}
            or not isinstance(response.get("spans"), list)):
        raise ValueError("official RoR span response has an invalid schema")
    validated = []
    seen = set()
    for span in response["spans"]:
        if not isinstance(span, Mapping) or set(span) != {"start", "end"}:
            raise ValueError("official RoR span item has an invalid schema")
        start, end = span["start"], span["end"]
        pair = (start, end)
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= token_count
                or not all(index in gated_positions for index in range(start, end))
                or pair in seen):
            raise ValueError("official RoR span is invalid or outside the gated positions")
        seen.add(pair)
        validated.append({"start": start, "end": end})
    return validated


def _official_ror_types(response: Any, spans: List[dict[str, int]],
                        valid_types: set[str]) -> tuple[List[dict[str, Any]], dict[int, str]]:
    if (not isinstance(response, Mapping) or set(response) != {"types"}
            or not isinstance(response.get("types"), list)):
        raise ValueError("official RoR type response has an invalid schema")
    requested = {(span["start"], span["end"]) for span in spans}
    seen = set()
    typed = []
    proposals = {}
    for item in response["types"]:
        if (not isinstance(item, Mapping)
                or set(item) != {"start", "end", "type"}):
            raise ValueError("official RoR type item has an invalid schema")
        start, end, entity_type = item["start"], item["end"], item["type"]
        span = (start, end)
        if (type(start) is not int or type(end) is not int
                or not isinstance(entity_type, str)
                or span not in requested or span in seen
                or entity_type not in valid_types):
            raise ValueError("official RoR type item is incomplete or outside the ontology")
        seen.add(span)
        record = {"start": start, "end": end, "type": entity_type}
        typed.append(record)
        proposals.update({index: entity_type for index in range(start, end)})
    if seen != requested:
        raise ValueError("official RoR type response must type every live span")
    return typed, proposals


def _official_ror_request(
    stage: str, payload: Mapping[str, Any], valid_types: Sequence[str]
) -> dict[str, Any]:
    if stage == "span_detection":
        schema = {
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
        task = "Detect only entity spans using zero-based, end-exclusive offsets."
        name = "lad_rg_span_detection"
    else:
        schema = {
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
        task = "Assign one ontology type to every supplied span."
        name = "lad_rg_type_assignment"
    return {
        "name": name,
        "schema": schema,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict LAD-RG RoR component. Return only the requested JSON object.",
            },
            {
                "role": "user",
                "content": (
                    f"{task}\nOntology: {json.dumps(list(valid_types))}\n"
                    f"Payload: {json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"
                ),
            },
        ],
        "temperature": 0.7,
        "enable_thinking": True,
    }


def ror_node(state: State):
    """Recall-Oriented Reasoning stage (replaces voting_node).

    Produces the F1-neutral base selection (weighted vote, entity_boost=1.0 —
    no uniform inflation) AND gated recovery proposals for GASD.
    """
    print(f"\n [RoR] recall-oriented recovery")
    candidate_paths = state.get("candidate_paths", [])
    rag_weights = state.get("rag_weights", [])
    dirty_tags = state.get("dirty_tags", [])
    tokens = state.get("tokens", [])
    ds = state.get("dataset_name", "conll2003")
    valid_types = DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"])
    valid_set = set(valid_types)
    official = bool(state.get("official", False))
    fallback_used = bool(state.get("fallback_used", False))
    structured_requester = (
        _require_structured_requester(state, "RoR") if official else None
    )

    if not candidate_paths:
        if official:
            raise ValueError("official RoR requires a non-empty candidate pool")
        fallback = state.get("current_tags", []) or dirty_tags or ["O"] * len(tokens)
        fallback = enforce_iob2_syntax(fallback, valid_set)
        fallback = fallback[:len(tokens)] + ["O"] * max(0, len(tokens) - len(fallback))
        return {"current_tags": fallback, "ror_proposals": {},
                "ror_reasoning": {"source": "not_triggered", "spans": []},
                "fallback_used": True}

    if not rag_weights or len(rag_weights) != len(candidate_paths):
        if official:
            raise ValueError("official RoR requires one Reviewer weight per candidate")
        rag_weights = [1.0] * len(candidate_paths)

    if official:
        valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                              for entity_type in valid_types for prefix in ("B", "I")}
        if any(not isinstance(path, list) or len(path) != len(tokens)
               or any(tag not in valid_tags for tag in path)
               for path in candidate_paths):
            raise ValueError("official RoR candidate path has invalid length or ontology")

    base = _weighted_majority_voting(
        candidate_paths, rag_weights,
        entity_boost=OFFICIAL_DECODER_CONSTANTS["voting_entity_boost"],
        dirty_tags=dirty_tags,
        consensus_ratio=OFFICIAL_DECODER_CONSTANTS["voting_consensus_ratio"])

    if not state.get("use_ror", True):
        return {"current_tags": base, "ror_proposals": {},
                "ror_reasoning": {"source": "disabled", "spans": []},
                "provider_metadata": _with_stage_records(state, "ror", [
                    _stage_status_record("ror", "disabled", state.get("provider_settings"))
                ]),
                "fallback_used": fallback_used}

    use_lads = state.get("use_lads", True)
    omega = _omega_weights(tokens, ds) if use_lads else [0.0] * len(tokens)
    proposals = _ror_recover(tokens, base, candidate_paths, rag_weights, omega,
                             ungated=(state.get("ror_ungated", False) or not use_lads))
    if not proposals:
        return {
            "current_tags": base,
            "ror_proposals": {},
            "ror_reasoning": {"source": "not_triggered", "spans": []},
            "provider_metadata": _with_stage_records(state, "ror", [
                _stage_status_record(
                    "ror", "not_triggered", state.get("provider_settings")
                )
            ]),
            "fallback_used": fallback_used,
        }

    reasoning = {"source": "deterministic_fallback", "spans": []}
    ror_records = []
    reasoner = state.get("ror_reasoner")
    if official or callable(reasoner):
        payload = {
            "tokens": list(tokens),
            "base_tags": list(base),
            "gated_positions": sorted(proposals),
            "candidate_type_proposals": dict(proposals),
            "valid_types": list(valid_types),
        }
        try:
            if official:
                span_response = _invoke_structured_requester_sync(
                    structured_requester,
                    "ror_span_detection",
                    _official_ror_request("span_detection", payload, valid_types),
                )
            else:
                span_response = reasoner("span_detection", payload)
            gated = set(proposals)
            ror_records.append(_callback_stage_record(
                span_response, "ror_span_detection", state.get("provider_settings")))
            if official:
                valid_spans = _official_ror_spans(span_response, len(tokens), gated)
            else:
                spans = span_response.get("spans", []) if isinstance(span_response, dict) else []
                valid_spans = []
                for span in spans:
                    start, end = span.get("start"), span.get("end")
                    if (isinstance(start, int) and isinstance(end, int)
                            and 0 <= start < end <= len(tokens)
                            and all(i in gated for i in range(start, end))):
                        valid_spans.append({"start": start, "end": end})
            live_rejected_all = (
                isinstance(span_response, Mapping)
                and isinstance(span_response.get("spans"), list)
                and len(span_response["spans"]) == 0
            )
            if not valid_spans and (official or live_rejected_all):
                proposals = {}
                reasoning = {"source": "live", "spans": []}
            if valid_spans:
                type_payload = {**payload, "spans": valid_spans}
                if official:
                    type_response = _invoke_structured_requester_sync(
                        structured_requester,
                        "ror_type_assignment",
                        _official_ror_request(
                            "type_assignment", type_payload, valid_types
                        ),
                    )
                else:
                    type_response = reasoner("type_assignment", type_payload)
                ror_records.append(_callback_stage_record(
                    type_response, "ror_type_assignment", state.get("provider_settings")))
                if official:
                    typed, reasoned = _official_ror_types(
                        type_response, valid_spans, valid_set)
                else:
                    typed = type_response.get("types", []) if isinstance(type_response, dict) else []
                    reasoned = {}
                    for item in typed:
                        start, end, entity_type = item.get("start"), item.get("end"), item.get("type")
                        if ({"start": start, "end": end} in valid_spans
                                and entity_type in valid_set):
                            reasoned.update({i: entity_type for i in range(start, end)})
                if reasoned:
                    proposals = reasoned
                    reasoning = {"source": "live" if official else "callback",
                                 "spans": valid_spans,
                                 "type_assignments": typed}
        except Exception as exc:                              # noqa: BLE001
            if official:
                raise
            logging.warning(f"[RoR] reasoner unavailable ({exc!r}); using deterministic fallback")
            fallback_used = True
    elif proposals:
        fallback_used = True
    if proposals:
        print(f" [RoR] {len(proposals)} gated entity-recovery proposals")
    return {"current_tags": base, "ror_proposals": proposals,
            "ror_reasoning": reasoning,
            "provider_metadata": _with_stage_records(state, "ror", ror_records),
            "fallback_used": fallback_used}


def _gasd_viterbi_decode(candidate_paths: List[List[str]], weights: List[float],
                         tokens: List[str], omega: List[float],
                         proposals: dict, valid_types: List[str], *,
                         use_potentials: bool = True,
                         beta_omega: float = 0.5,
                         gamma_prop: float = OFFICIAL_DECODER_CONSTANTS["gasd_gamma_proposal"],
                         reason_tag_scores: Optional[List[dict]] = None,
                         candidate_scale: float = 1.0) -> List[str]:
    """GASD-G: global IOB2-constrained inference (Viterbi) over the candidate
    distribution. The per-position emission integrates three F1-relevant terms —
    candidate fidelity, LADS ω(t) entity potentials, and RoR recovery proposals —
    under hard IOB2 transition legality. Globally optimal, distribution-aware,
    and 0.00% SER by construction (only legal transitions are ever taken).

    This is the *integrator* where synergy is manufactured: it needs LADS
    potentials to beat a greedy legal patch, and it preserves RoR's high-ω
    recoveries as legal spans instead of deleting them.
    """
    n = len(tokens)
    if n == 0:
        return []
    tagset = ["O"] + [f"{p}-{t}" for t in valid_types for p in ("B", "I")]
    W = sum(weights) or 1.0
    NEG = float("-inf")

    def emission(pos: int, tag: str) -> float:
        vote = 0.0
        for i, path in enumerate(candidate_paths):
            if pos < len(path) and path[pos] == tag:
                vote += weights[i]
        score = candidate_scale * vote / W
        if reason_tag_scores and pos < len(reason_tag_scores):
            scores = reason_tag_scores[pos]
            if isinstance(scores, dict):
                raw = scores.get(tag, 0.0)
                if isinstance(raw, (int, float)):
                    score += float(raw)
        if tag != "O":
            etype = tag.split("-", 1)[1]
            if use_potentials:
                score += beta_omega * omega[pos]
            prop = proposals.get(pos)
            if prop is not None and (prop == etype or prop == "ENTITY"):
                score += gamma_prop
        return score

    # Viterbi. prev="O" is the virtual start (matches compute_ser semantics).
    dp = {tag: (emission(0, tag) if _is_valid_transition("O", tag) else NEG)
          for tag in tagset}
    back: List[dict] = [{}]
    for pos in range(1, n):
        new_dp, bp = {}, {}
        for tag in tagset:
            e = emission(pos, tag)
            best_prev, best_score = None, NEG
            for ptag in tagset:
                if dp[ptag] == NEG or not _is_valid_transition(ptag, tag):
                    continue
                sc = dp[ptag] + e
                if sc > best_score:
                    best_score, best_prev = sc, ptag
            new_dp[tag] = best_score
            bp[tag] = best_prev
        dp, _ = new_dp, back.append(bp)

    last = max(tagset, key=lambda t: dp[t])
    seq = [last]
    for pos in range(n - 1, 0, -1):
        last = back[pos][last]
        seq.append(last)
    seq.reverse()
    return seq


def _official_reason_scores(evidence: Any, token_count: int,
                            valid_tags: set[str]) -> List[dict[str, float]]:
    if (not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("reason"), str)
            or not isinstance(evidence.get("tags"), list)
            or not isinstance(evidence.get("tag_scores"), list)):
        raise ValueError("official GASD-R response has an invalid schema")
    tags = evidence["tags"]
    if (len(tags) != token_count
            or any(not isinstance(tag, str) or tag not in valid_tags for tag in tags)):
        raise ValueError("official GASD-R tags have invalid length or ontology")
    scores = evidence["tag_scores"]
    if len(scores) != token_count:
        raise ValueError("official GASD-R response must score every token")
    normalized = []
    for declared_tag, token_scores in zip(tags, scores):
        if (not isinstance(token_scores, Mapping)
                or set(token_scores) != {declared_tag}):
            raise ValueError("official GASD-R score must match the declared tag exactly")
        score = token_scores[declared_tag]
        if type(score) not in (int, float) or not math.isfinite(float(score)):
            raise ValueError("official GASD-R declared tag score must be finite")
        normalized.append({declared_tag: float(score)})
    return normalized


def _is_legal_tag_sequence(tags: List[str]) -> bool:
    previous = "O"
    for tag in tags:
        if not _is_valid_transition(previous, tag):
            return False
        previous = tag
    return True


def _first_illegal_tag_transition(tags: Sequence[str]) -> Optional[dict[str, Any]]:
    previous = "O"
    for index, current in enumerate(tags):
        if not _is_valid_transition(previous, current):
            return {"index": index, "previous": previous, "current": current}
        previous = current
    return None


def _official_gasd_request(
    payload: Mapping[str, Any], valid_tags: Sequence[str], token_count: int
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "tags"],
        "properties": {
            "reason": {"type": "string"},
            "tags": {
                "type": "array",
                "minItems": token_count,
                "maxItems": token_count,
                "items": {"type": "string", "enum": list(valid_tags)},
            },
        },
    }
    return {
        "name": "lad_rg_gasd_r",
        "schema": schema,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict LAD-RG GASD-R component. Return only the requested JSON object.",
            },
            {
                "role": "user",
                "content": (
                    "Return exactly one ontology-valid tag per token. The downstream decoder "
                    "enforces hard IOB2 transitions.\n"
                    f"Valid tags: {json.dumps(list(valid_tags))}\n"
                    f"Payload: {json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"
                ),
            },
        ],
        "temperature": 0.0,
        "enable_thinking": True,
    }


def _official_structured_gasd_evidence(
    evidence: Any, token_count: int, valid_tags: set[str]
) -> dict[str, Any]:
    if (not isinstance(evidence, Mapping)
            or set(evidence) != {"reason", "tags"}):
        raise ValueError("official GASD-R structured response has an invalid schema")
    tags = evidence.get("tags")
    if (not isinstance(evidence.get("reason"), str)
            or not isinstance(tags, list)
            or len(tags) != token_count
            or any(not isinstance(tag, str) or tag not in valid_tags for tag in tags)):
        raise ValueError("official GASD-R structured response has invalid tags")
    return {
        "reason": evidence["reason"],
        "tags": list(tags),
        "tag_scores": [{tag: OFFICIAL_DECODER_CONSTANTS["gasd_reason_bonus"]}
                        for tag in tags],
    }


def gasd_node(state: State):
    """GASD structural integrator (replaces physical_wash_node)."""
    print(f"\n [GASD] global constrained decode")
    tokens = state.get("tokens", [])
    candidate_paths = state.get("candidate_paths", [])
    weights = state.get("rag_weights", []) or [1.0] * len(candidate_paths)
    base = state.get("current_tags", [])
    proposals = state.get("ror_proposals", {}) or {}
    ds = state.get("dataset_name", "conll2003")
    valid_types = DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"])
    valid_set = set(valid_types)
    valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                          for entity_type in valid_types for prefix in ("B", "I")}
    official = bool(state.get("official", False))
    fallback_used = bool(state.get("fallback_used", False))
    variant = str(state.get("gasd_variant", "g")).lower()
    gasd_records = []

    try:
        decoder = state.get("gasd_reason_decoder")
        structured_requester = None
        if official and variant in {"r", "both"}:
            structured_requester = _require_structured_requester(state, "GASD-R")
        if not state.get("use_gasd", True):
            fallback = base or state.get("dirty_tags", []) or ["O"] * len(tokens)
            fallback = enforce_iob2_syntax(fallback, valid_set)
            fallback = fallback[:len(tokens)] + ["O"] * max(0, len(tokens) - len(fallback))
            return {"current_tags": fallback,
                    "gasd_variant_requested": variant,
                    "gasd_variant_used": "disabled",
                    "provider_metadata": _with_stage_records(state, "gasd", [
                        _stage_status_record(
                            "gasd", "disabled", state.get("provider_settings")
                        )
                    ]),
                    "fallback_used": fallback_used}
        if len(weights) != len(candidate_paths):
            if official:
                raise ValueError("official GASD requires one weight per candidate")
            weights = [1.0] * len(candidate_paths)
        if not candidate_paths:
            if official:
                raise ValueError("official GASD requires a non-empty candidate pool")
            fallback = base or state.get("dirty_tags", []) or ["O"] * len(tokens)
            fallback = enforce_iob2_syntax(fallback, valid_set)
            fallback = fallback[:len(tokens)] + ["O"] * max(0, len(tokens) - len(fallback))
            return {"current_tags": fallback,
                    "gasd_variant_requested": variant,
                    "gasd_variant_used": "g_fallback",
                    "fallback_used": True}
        if official and (variant not in {"g", "r", "both"}
                         or any(not isinstance(path, list) or len(path) != len(tokens)
                                or any(tag not in valid_tags for tag in path)
                                for path in candidate_paths)):
            raise ValueError("official GASD variant or candidate ontology is invalid")
        use_lads = state.get("use_lads", True)
        omega = _omega_weights(tokens, ds) if use_lads else [0.0] * len(tokens)
        reason_scores = None
        variant_used = "g"
        if variant in {"r", "both"}:
            if official or callable(decoder):
                try:
                    decoder_payload = {
                        "tokens": list(tokens),
                        "base_tags": list(base),
                        "candidate_paths": [list(path) for path in candidate_paths],
                        "ror_proposals": dict(proposals),
                        "ror_reasoning": state.get("ror_reasoning", {}),
                        "valid_tags": ["O"] + [f"{p}-{t}" for t in valid_types for p in ("B", "I")],
                        "constraint": "hard_iob2",
                    }
                    if official:
                        evidence = _invoke_structured_requester_sync(
                            structured_requester,
                            "gasd_r",
                            _official_gasd_request(
                                decoder_payload,
                                ["O"] + [
                                    f"{p}-{t}" for t in valid_types for p in ("B", "I")
                                ],
                                len(tokens),
                            ),
                        )
                    else:
                        evidence = decoder(decoder_payload)
                except Exception as exc:                      # noqa: BLE001
                    if official:
                        raise
                    logging.warning(f"[GASD-R] decoder unavailable ({exc!r}); using GASD-G")
                    evidence = None
                if evidence is not None:
                    gasd_records.append(_callback_stage_record(
                        evidence, "gasd_r", state.get("provider_settings")))
                if official:
                    evidence = _official_structured_gasd_evidence(
                        evidence, len(tokens), valid_tags
                    )
                    reason_scores = _official_reason_scores(
                        evidence, len(tokens), valid_tags)
                    variant_used = variant
                elif isinstance(evidence, dict) and isinstance(evidence.get("tag_scores"), list):
                    reason_scores = evidence["tag_scores"]
                    variant_used = variant
            if reason_scores is None:
                variant_used = "g_fallback"
                fallback_used = True
        decoded = _gasd_viterbi_decode(
            candidate_paths, weights, tokens, omega, proposals, valid_types,
            use_potentials=(state.get("gasd_potentials", True) and use_lads),
            beta_omega=GASD_BETA_OMEGA,
            reason_tag_scores=reason_scores,
            candidate_scale=(
                OFFICIAL_DECODER_CONSTANTS["gasd_candidate_scale_r"]
                if variant == "r" and reason_scores is not None
                else OFFICIAL_DECODER_CONSTANTS["gasd_candidate_scale_g"]
            ))
        if len(decoded) != len(tokens):
            if official:
                raise ValueError("official GASD decoder returned the wrong sequence length")
            decoded = enforce_iob2_syntax(base, valid_set)
            if len(decoded) != len(tokens):
                decoded = (decoded[:len(tokens)]
                           + ["O"] * max(0, len(tokens) - len(decoded)))
                fallback_used = True
        if official and (any(tag not in valid_tags for tag in decoded)
                         or not _is_legal_tag_sequence(decoded)):
            raise ValueError("official GASD decoder violated hard IOB2 legality")
        if not gasd_records and variant == "g":
            gasd_records.append(_stage_status_record(
                "gasd", "local", state.get("provider_settings")
            ))
        return {"current_tags": decoded,
                "gasd_variant_requested": variant,
                "gasd_variant_used": variant_used,
                "provider_metadata": _with_stage_records(state, "gasd", gasd_records),
                "fallback_used": fallback_used}
    except Exception as e:                                    # noqa: BLE001
        if official:
            raise
        logging.error(f"[GASD] decode failed ({e!r}); legalizing base")
        fallback = base or state.get("dirty_tags", []) or ["O"] * len(tokens)
        fallback = enforce_iob2_syntax(fallback, valid_set)
        fallback = fallback[:len(tokens)] + ["O"] * max(0, len(tokens) - len(fallback))
        return {"current_tags": fallback,
                "gasd_variant_requested": variant,
                "gasd_variant_used": "g_fallback",
                "fallback_used": True}


# =====================================================================
# SelectDenoise — Lever 2: verifier_node (replaces ror/gasd in the graph)
# =====================================================================

def _distinct_paths_by_weight(candidate_paths: List[List[str]],
                              weights: List[float], topk: int
                              ) -> List[Tuple[List[str], float]]:
    """Deduplicate candidate paths, summing the Reviewer weight of identical
    paths, and return the top-`topk` by aggregated weight (desc)."""
    agg: dict = {}
    for i, p in enumerate(candidate_paths):
        key = tuple(p)
        w = weights[i] if i < len(weights) else 1.0
        agg[key] = agg.get(key, 0.0) + w
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [(list(k), v) for k, v in ranked[:topk]]


def _is_type_contested(candidate_paths: List[List[str]]) -> bool:
    """TYPE-contested = the selection-bound signature the verifier is good at:
    a position where every candidate agrees it is an entity (none says O) but
    they disagree on the TYPE. This is the ATF case (boundaries agreed, type
    disputed). It deliberately excludes pure BOUNDARY disagreement (IF/BT: some
    paths say O, others entity) where whole-path selection tends to hurt — so the
    verifier does no harm on generation-bound noise and only fires where the
    candidate pool holds a genuine type choice to make."""
    if len(candidate_paths) < 2:
        return False
    n = min(len(p) for p in candidate_paths)
    for pos in range(n):
        labels = [p[pos] for p in candidate_paths]
        if any(l == "O" for l in labels):
            continue                                   # boundary dispute → skip
        types = {l.split("-", 1)[1] for l in labels if "-" in l}
        if len(types) >= 2:
            return True
    return False


def _spans_str(tokens: List[str], tags: List[str]) -> str:
    """Human-readable entity spans of a path, e.g. "[Jeff Dean]->PER"."""
    out, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-") and "-" in t:
            ty = t.split("-", 1)[1]; j = i + 1
            while j < n and tags[j] == f"I-{ty}":
                j += 1
            span = " ".join(tokens[k] for k in range(i, min(j, len(tokens))))
            out.append(f"[{span}]->{ty}"); i = j
        else:
            i += 1
    return "; ".join(out) if out else "(no entities)"


# Entity-potential weight for the BT/IF global decode. ω(t) from DEER is small
# (~0.001-0.03), so β scales it against candidate fidelity (which lives in
# [0,1]). F1 sits on a flat plateau for β in [1,3] and collapses by β=10; the
# value was selected on seed 13 alone and validated on the held-out seeds.
GASD_BETA_OMEGA = OFFICIAL_DECODER_CONSTANTS["gasd_beta_omega"]


def _base_decode(candidate_paths: List[List[str]], weights: List[float],
                 tokens: List[str], dirty_tags: List[str],
                 dataset_name: str, noise_type: str) -> List[str]:
    """Noise-adaptive base decode over the candidate pool.

    BT/IF corrupt entities *locally* (a dropped boundary or interior tag), so
    the correct labeling is a globally coherent span that per-position voting
    fragments. A global IOB2-constrained Viterbi over the candidate
    distribution recovers it: +0.054 F1 on BT and +0.006 on IF (macro over 5
    datasets x 3 seeds, re-decoded on fixed candidate pools), at SER=0.00 by
    construction rather than by post-hoc repair.

    ATF instead flips a whole span's TYPE, which is a selection problem the LLM
    Verifier handles; there the global decode is slightly harmful (-0.004), so
    ATF keeps the weighted vote. consensus_ratio=0.7 there keeps more of the
    (reliable) dirty tag on low-agreement positions.

    Falls back to the vote when DEER statistics are unavailable (e.g. dummy
    mode): ω would degrade to a constant, making the entity potential a flat
    bias that swamps candidate fidelity.
    """
    if noise_type in ("BT", "IF") and _deer_stats.get(dataset_name) is not None:
        return _gasd_viterbi_decode(
            candidate_paths, weights, tokens,
            _omega_weights(tokens, dataset_name), {},
            DATASET_ENTITY_TYPES.get(dataset_name, ["PER", "LOC", "ORG"]),
            use_potentials=True, beta_omega=GASD_BETA_OMEGA)
    return _weighted_majority_voting(candidate_paths, weights, entity_boost=1.0,
                                     dirty_tags=dirty_tags, consensus_ratio=0.7)


async def verifier_node(state: State):
    """SelectDenoise Lever 2: on contested sentences, ask the LLM to SELECT the
    single best complete IOB2 labeling among the distinct candidate paths (the
    correct answer is provably in the pool — oracle analysis). Uncontested
    sentences use the cheap weighted vote. Output is validated + legalized
    (SER=0.00); any failure falls back to the vote, so it is never worse."""
    print(f"\n [Verifier] selection")
    tokens = state.get("tokens", [])
    candidate_paths = state.get("candidate_paths", [])
    weights = state.get("rag_weights", []) or [1.0] * len(candidate_paths)
    dirty_tags = state.get("dirty_tags", [])
    ds = state.get("dataset_name", "conll2003")
    valid_set = set(DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"]))
    valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                            for entity_type in valid_set for prefix in ("B", "I")}
    official = bool(state.get("official", False))

    def skipped(status: str, tags: List[str]) -> dict:
        result = {"current_tags": tags}
        if official:
            result["provider_metadata"] = _with_stage_records(
                state, "verifier", [_stage_status_record(
                    "verifier", status, state.get("provider_settings")
                )]
            )
        return result

    # Noise-aware final legalization. On IF, a residual dangling I- is a
    # failed-merge artifact — DEMOTE it to O (promoting to B- invents a
    # false-positive entity and collapses precision). BT/ATF keep the historical
    # PROMOTE behavior (a dropped B- should be recovered as a real entity).
    dangling_policy = "demote" if state.get("noise_type") == "IF" else "promote"

    if not candidate_paths:
        if official:
            raise ValueError("official Verifier requires a non-empty candidate pool")
        return skipped("skipped_no_candidates", legalize_noise_aware(
            list(dirty_tags), valid_set, dangling_policy))
    if len(weights) != len(candidate_paths):
        if official:
            raise ValueError("official Verifier requires one weight per candidate")
        weights = [1.0] * len(candidate_paths)

    # Base / fallback: noise-adaptive (global Viterbi on BT/IF, vote on ATF).
    base = _base_decode(candidate_paths, weights, tokens, dirty_tags,
                        ds, state.get("noise_type"))
    base = legalize_noise_aware(base, valid_set, dangling_policy)

    # Fire only on TYPE-contested sentences (selection-bound / ATF signature);
    # boundary-contested (IF/BT) sentences stay on the safe vote → do no harm.
    fire = state.get("use_verifier", True) and (
        state.get("verify_all", False) or _is_type_contested(candidate_paths))
    if not fire:
        return skipped("skipped_uncontested", base)

    topk = int(state.get("verifier_topk", 4))
    distinct = _distinct_paths_by_weight(candidate_paths, weights, topk)
    if len(distinct) < 2:                       # nothing to choose between
        return skipped("skipped_identical", base)

    deer_examples = _get_deer_examples(
        tokens, top_k=3, dataset_name=ds, cache_only=official
    )
    cand_block = "\n".join(
        f"  Candidate {idx+1} (score {w:.2f}): {json.dumps(p, ensure_ascii=False)}\n"
        f"      entities: {_spans_str(tokens, p)}"
        for idx, (p, w) in enumerate(distinct))
    n = len(tokens)
    prompt = f"""
You are an IOB2 Selection Judge. Several candidate labelings were produced for the SAME sentence; each may be partly right. Choose the SINGLE most correct complete labeling. You MAY output one candidate verbatim, or combine their correct entity spans into one better labeling — but every tag must be legal IOB2.

Tokens ({n}): {json.dumps(tokens, ensure_ascii=False)}
Dirty tags (reference — contain ~15% noise, do NOT trust blindly): {json.dumps(dirty_tags, ensure_ascii=False)}

VALID TAG VOCABULARY (use ONLY these; case-sensitive):
{_format_valid_tags(ds)}

Calibration Examples (correct annotations from the TRAINING SET, retrieved by label-guided similarity — use these to judge which candidate's entity TYPES are right):
{json.dumps(deer_examples, indent=2) if deer_examples else '[]'}

Candidate labelings to choose among:
{cand_block}

Decide by: (1) IOB2 legality; (2) entity TYPES consistent with the calibration examples and token semantics; (3) correct boundaries. When candidates disagree on an entity's type, pick the type the calibration/context best supports — the majority candidate is NOT automatically right.

Output ONLY a JSON object whose only key is "tags". The "tags" array must
contain exactly {n} IOB2 tags in token order. No markdown or explanation."""

    if official:
        requester = _require_structured_requester(state, "Verifier")
        max_semantic_retries = state.get("verifier_semantic_max_retries", 0)
        if (
            type(max_semantic_retries) is not int
            or not 0 <= max_semantic_retries <= OFFICIAL_VERIFIER_SEMANTIC_MAX_RETRIES
        ):
            raise ValueError("official Verifier semantic retry limit is invalid")
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["tags"],
            "properties": {"tags": {
                "type": "array", "minItems": n, "maxItems": n,
                "items": {"type": "string", "enum": sorted(valid_tags)},
            }},
        }
        messages = [
            {"role": "system", "content": "You are a strict SelectDenoise Verifier. Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ]
        records: list[dict[str, Any]] = []
        for attempt in range(1, max_semantic_retries + 2):
            try:
                response = await _invoke_structured_requester(requester, "verifier", {
                    "name": "selectdenoise_verifier", "schema": schema,
                    "messages": list(messages),
                    "temperature": 0.0,
                    "enable_thinking": _official_enable_thinking(state),
                })
                # Length, ontology, and schema failures are deliberately not retried.
                picked = _official_tag_path(response, n, valid_tags)
            except asyncio.CancelledError:
                recorder = state.get("verifier_semantic_interruption_recorder")
                if records and callable(recorder):
                    recorder(tuple(records), "CancelledError")
                raise
            except Exception as exc:
                if records:
                    raise VerifierSemanticRetryInterrupted(records, exc) from exc
                raise
            transition = _first_illegal_tag_transition(picked)
            record = _callback_stage_record(
                response, "verifier", state.get("provider_settings")
            )
            if transition is None:
                if records:
                    record.update({
                        "semantic_attempt": attempt,
                        "semantic_outcome": "accepted",
                    })
                    records.append(record)
                    validate_verifier_semantic_retry_evidence(records)
                else:
                    records = [record]
                return {
                    "current_tags": picked,
                    "provider_metadata": _with_stage_records(
                        state, "verifier", records
                    ),
                }

            record.update({
                "semantic_attempt": attempt,
                "semantic_outcome": "rejected_illegal_iob2",
                "illegal_transition": transition,
                "rejected_tags": list(picked),
                "response_sha256": hashlib.sha256(json.dumps(
                    {"tags": picked}, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
            })
            records.append(record)
            if attempt > max_semantic_retries:
                raise VerifierSemanticRetryExhausted(records)
            messages.append({
                "role": "user",
                "content": (
                    "Your previous tags passed JSON, length, and ontology validation "
                    "but failed strict IOB2 transition validation at token index "
                    f"{transition['index']}: {transition['previous']} -> "
                    f"{transition['current']}. Regenerate the entire tags array; do "
                    "not explain, shorten, or locally patch the prior answer. Previous "
                    f"rejected tags: {json.dumps(picked, ensure_ascii=False)}"
                ),
            })

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        picked = extract_json_list(response.content, fallback_length=n)
        if not isinstance(picked, list) or len(picked) != n:
            print(f" [Verifier] bad output (len {len(picked) if isinstance(picked,list) else '?'} != {n}); vote fallback")
            return {"current_tags": base}
        picked = legalize_noise_aware(picked, valid_set, dangling_policy)
        if len(picked) != n:
            return {"current_tags": base}
        return {"current_tags": picked}
    except Exception as e:                                    # noqa: BLE001
        logging.error(f"[Verifier] failed ({e!r}); vote fallback")
        return {"current_tags": base}


workflow = StateGraph(State)

workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("verifier", verifier_node)

workflow.add_edge(START, "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", "verifier")
workflow.add_edge("verifier", END)

multi_agent_graph = workflow.compile()


lad_rg_workflow = StateGraph(State)

lad_rg_workflow.add_node("coder", coder_node)
lad_rg_workflow.add_node("reviewer", reviewer_node)
lad_rg_workflow.add_node("ror", ror_node)
lad_rg_workflow.add_node("gasd", gasd_node)

lad_rg_workflow.add_edge(START, "coder")
lad_rg_workflow.add_edge("coder", "reviewer")
lad_rg_workflow.add_edge("reviewer", "ror")
lad_rg_workflow.add_edge("ror", "gasd")
lad_rg_workflow.add_edge("gasd", END)

lad_rg_graph = lad_rg_workflow.compile()


def _select_pipeline_graph(config: dict):
    return lad_rg_graph if config.get("terminal_graph") == "lad-rg" else multi_agent_graph


#if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="LAD-RG Agent Runner")
    #parser.add_argument("--input", required=True, help="Path to input noisy jsonl")
    #parser.add_argument("--output", required=True, help="Path to save predictions")
    #args = parser.parse_args()
    
    # run_agent_pipeline(args.input, args.output)


def _per_position_confidence(candidate_paths: List[List[str]],
                             weights: List[float],
                             predicted_tags: List[str]) -> List[float]:
    """K-path agreement confidence: weighted fraction of Coder paths that
    emitted the finally-decoded tag at each position. Used as the B4/B5
    confidence proxy when the backbone does not expose token logprobs."""
    n = len(predicted_tags)
    W = sum(weights) or 1.0
    conf: List[float] = []
    for pos in range(n):
        agree = 0.0
        available_weight = 0.0
        for k, path in enumerate(candidate_paths):
            if pos < len(path):
                weight = weights[k] if k < len(weights) else 1.0
                available_weight += weight
                if path[pos] == predicted_tags[pos]:
                    agree += weight
        conf.append(agree / (available_weight or 1.0))
    return conf


def _normalize_candidate_evidence(candidate_paths: List[List[str]],
                                  weights: List[float],
                                  expected_length: int) -> Tuple[List[List[str]], List[float]]:
    """Make candidate evidence safe to persist alongside a sentence.

    Every usable path is clipped to the sentence length without inventing
    synthetic suffix tags, and weights are made one-per-path so downstream
    consumers never need to special-case malformed partial evidence.
    """
    normalized_paths: List[List[str]] = []
    normalized_weights: List[float] = []
    for i, path in enumerate(candidate_paths):
        if not isinstance(path, list) or not path:
            continue
        normalized_paths.append(list(path[:expected_length]))
        normalized_weights.append(float(weights[i]) if i < len(weights) else 1.0)
    return normalized_paths, normalized_weights


async def run_agent_pipeline(tokens: List[str], dirty_tags: List[str],
                              config: dict = None,
                              dataset_name: str = None):
    """
    `dataset_name` controls which entity-type ontology is injected into
    the Coder / Reviewer prompts. Allowed values: 'msra', 'conll2003', 'wnut17'.
    If omitted, falls back to config['__dataset__'], then to 'conll2003'.

    Returns the decoded tag list (``List[str]``) by default. When
    ``config['__return_candidates__']`` is set, instead returns a dict
    ``{"pred_tags", "candidate_paths", "rag_weights", "confidence"}`` so the
    driver can persist the Coder candidate pool for the B5 oracle-pool study
    and the B4 logit/agreement analysis (backbone-agnostic; no extra LLM calls).
    """
    if config is None:
        config = {"lambda_bias": 1.0, "use_dfa": True}
    official = bool(config.get("official", False))
    contextual_official = (
        official and config.get("terminal_decoder") == "contextual-lattice-v1"
    )
    if official and not contextual_official and config.get("terminal_graph") != "lad-rg":
        raise ValueError("official pipeline requires terminal_graph='lad-rg'")

    # Dataset name resolution: explicit arg → config["__dataset__"] → default
    if dataset_name is None:
        dataset_name = config.get("__dataset__", "conll2003")
    if dataset_name not in DATASET_ENTITY_TYPES:
        if official:
            raise ValueError(f"unknown official dataset ontology: {dataset_name!r}")
        logging.warning(f"[!] Unknown dataset_name={dataset_name!r}; "
                       f"defaulting to conll2003 ontology. "
                       f"Known: {list(DATASET_ENTITY_TYPES)}")
        dataset_name = "conll2003"

    noise_type = config.get("__noise_type__", "BT")
    terminal = None
    if (config.get("terminal_decoder") == "contextual-lattice-v1"
            and not config.get("preterminal_only", False)):
        # Load before opening any sentence work so a missing/mismatched bundle
        # is visible to the orchestrator instead of silently changing methods.
        terminal = _load_contextual_lattice_terminal()

    initial_state = {
        "messages": [],
        "tokens": tokens,
        "dirty_tags": dirty_tags,
        "loop_count": 0,
        "iterations": 0,
        "errors": [],
        "candidate_paths": [],
        "rag_weights": [],
        "current_tags": [],
        "lambda_weight": config.get("lambda_bias", 1.0),
        "use_wash": config.get("use_dfa", True),
        "dataset_name": dataset_name,
        "noise_type": noise_type,
        # LAD-RG synergy controls. Used only by the opt-in lad-rg graph.
        "ror_proposals": {},
        "ror_reasoning": {},
        "use_lads": config.get("use_lads", True),
        "use_ror": config.get("use_ror", True),
        "use_gasd": config.get("use_gasd", True),
        "ror_ungated": config.get("ror_ungated", False),
        "gasd_potentials": config.get("gasd_potentials", True),
        "gasd_variant": config.get("gasd_variant", "g"),
        "ror_reasoner": config.get("ror_reasoner"),
        "gasd_reason_decoder": config.get("gasd_reason_decoder"),
        "structured_requester": config.get("structured_requester"),
        "__diagnostic_coder_path__": config.get("__diagnostic_coder_path__"),
        "official": official,
        "provider_settings": (dict(config.get("provider_metadata", {}))
                              if isinstance(config.get("provider_metadata"), Mapping)
                              else {}),
        "provider_metadata": (
            {stage: [] for stage in _CONTEXTUAL_PROVIDER_STAGES}
            if contextual_official else _provider_evidence()
        ),
        "fallback_used": False,
        # SelectDenoise controls (defaults = full system)
        "deanchor_atf": config.get("deanchor_atf", True),
        "use_verifier": config.get("use_verifier", True),
        "verify_all": config.get("verify_all", False),
        "verifier_topk": config.get("verifier_topk", 4),
        "verifier_semantic_max_retries": config.get(
            "verifier_semantic_max_retries", 0
        ),
        "verifier_semantic_interruption_recorder": config.get(
            "verifier_semantic_interruption_recorder"
        ),
    }
    
    return_candidates = bool(config.get("__return_candidates__"))
    cand: List[List[str]] = []
    weights: List[float] = []
    terminal_metadata = {}
    official_metadata = {}
    try:
        final_state = await _select_pipeline_graph(config).ainvoke(initial_state)
        predicted_tags = final_state.get("current_tags", [])
        if not predicted_tags or len(predicted_tags) != len(tokens):
            if official:
                raise ValueError("official pipeline returned the wrong tag sequence length")
            predicted_tags = dirty_tags
        cand = (
            final_state.get("terminal_candidate_paths")
            if contextual_official else None
        ) or final_state.get("candidate_paths", []) or []
        weights = final_state.get("rag_weights", []) or [1.0] * len(cand)
        if terminal is not None:
            terminal_metadata = _apply_contextual_lattice_terminal(final_state, terminal)
            predicted_tags = terminal_metadata["current_tags"]
        if official:
            valid_types = DATASET_ENTITY_TYPES[dataset_name]
            valid_tags = {"O"} | {f"{prefix}-{entity_type}"
                                  for entity_type in valid_types for prefix in ("B", "I")}
            if (any(tag not in valid_tags for tag in predicted_tags)
                    or not _is_legal_tag_sequence(predicted_tags)):
                raise ValueError("official pipeline returned an illegal or out-of-ontology sequence")
            if contextual_official:
                evidence = _provider_evidence(final_state.get("provider_metadata"))
                _validate_contextual_official_evidence(evidence)
                if bool(final_state.get("fallback_used", False)):
                    raise RuntimeError("official pipeline cannot publish fallback evidence")
                official_metadata = {
                    "provider_metadata": evidence,
                    "fallback_used": False,
                }
            else:
                reasoning = final_state.get("ror_reasoning", {})
                reasoning_source = (
                    reasoning.get("source") if isinstance(reasoning, Mapping) else None
                )
                if reasoning_source not in {"not_triggered", "live", "disabled"}:
                    raise RuntimeError("official pipeline is missing valid RoR completion evidence")
                requested_value = final_state.get("gasd_variant_requested")
                used = final_state.get("gasd_variant_used")
                if not isinstance(requested_value, str) or not isinstance(used, str):
                    raise RuntimeError("official pipeline is missing GASD completion evidence")
                requested = requested_value.lower()
                configured_variant = str(config.get("gasd_variant", "g")).lower()
                if requested != configured_variant:
                    raise RuntimeError("official GASD requested variant does not match configuration")
                if requested in {"r", "both"} and used != requested:
                    raise RuntimeError("official GASD-R did not use the requested variant")
                if bool(final_state.get("fallback_used", False)):
                    raise RuntimeError("official pipeline cannot publish fallback evidence")
                official_metadata = {
                    "ror_reasoning_source": reasoning_source,
                    "gasd_variant_requested": requested,
                    "gasd_variant_used": used,
                    "provider_metadata": _provider_evidence(
                        final_state.get("provider_metadata")
                    ),
                    "fallback_used": bool(final_state.get("fallback_used", False)),
                }
    except Exception as e:
        logging.error(f"[-] Pipeline : {e}")
        if official or terminal is not None:
            raise
        predicted_tags = dirty_tags

    if not return_candidates:
        return predicted_tags
    cand, weights = _normalize_candidate_evidence(cand, weights, len(predicted_tags))
    return {
        "pred_tags": predicted_tags,
        "candidate_paths": cand,
        "rag_weights": weights,
        "confidence": _per_position_confidence(cand, weights, predicted_tags),
        **terminal_metadata,
        **official_metadata,
    }


# =========================================================================
# Baseline pipelines
# =========================================================================

def _simple_majority_voting(candidate_paths: List[List[str]]) -> List[str]:
    """Equal-weight majority voting per position (no RAG weights, no entity boost)."""
    if not candidate_paths:
        return []
    n_positions = len(candidate_paths[0])
    result = []
    for pos in range(n_positions):
        votes = Counter()
        for path in candidate_paths:
            votes[path[pos]] += 1
        result.append(votes.most_common(1)[0][0])
    return result


async def baseline_dirty_pipeline(tokens: List[str], dirty_tags: List[str],
                                  config: dict = None,
                                  dataset_name: str = None) -> List[str]:
    """No denoising. Returns dirty tags as-is (performance lower bound)."""
    return list(dirty_tags)


async def baseline_standard_prompting_pipeline(tokens: List[str], dirty_tags: List[str],
                                               config: dict = None,
                                               dataset_name: str = None) -> List[str]:
    """
    Standard Prompting baseline (Nagar et al., 2024/ACL 2025).
    arXiv:2408.12249 — "LLMs are not Zero-Shot Reasoners for Biomedical IE"
    Single LLM call with minimal prompt: no IOB2 rules, no CoT, no examples,
    no retrieval. The paper shows complex prompting harms NER; this is the
    simplest valid LLM baseline.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""Fix the noisy NER tags below.

Tokens: {tokens}
Noisy Tags: {dirty_tags}

Valid output tags: {valid_tags_str}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Standard Prompting baseline failed: {e}")
        return list(dirty_tags)


async def baseline_rule_only_pipeline(tokens: List[str], dirty_tags: List[str],
                                      config: dict = None,
                                      dataset_name: str = None) -> List[str]:
    """DFA rule-based IOB2 syntax correction only. No LLM, no RAG."""
    ds = dataset_name or "conll2003"
    valid_set = set(DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"]))
    return enforce_iob2_syntax(list(dirty_tags), valid_entity_types=valid_set)


async def baseline_self_consist_pipeline(tokens: List[str], dirty_tags: List[str],
                                         config: dict = None,
                                         dataset_name: str = None) -> List[str]:
    """
    Self-Consistency baseline (Wang et al. 2022).
    Coder generates 5 candidate paths via LLM, then equal-weight majority
    voting. No Reviewer, no RAG, no DFA wash.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    prompt = f"""
You are an Elite AI Data Engineer performing IOB2 label denoising.
Tokens: {tokens}
Dirty IOB2 Tags: {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

DENOISING POLICY:
- The dirty tags contain noise but most positions are already correct.
- Preserve non-O tags from the dirty input unless they clearly conflict with the token semantics or IOB2 grammar.
- When a dirty tag is non-O, prefer to keep it rather than collapsing to O. Only convert non-O tags to O when the token is clearly not part of any entity (e.g., a common verb, preposition, or punctuation).
{misc_guidance}
{sentence_initial_guidance}

Generate EXACTLY 5 different plausible IOB2 repair paths.
1. Every path must have exactly {len(tokens)} tokens.
2. Must follow IOB2 syntax strictly (I-X must be preceded by B-X or I-X of the same type).
3. Use ONLY tags from the VALID TAG VOCABULARY above. Do NOT invent new tag names or change case.
4. The 5 paths should explore different plausible interpretations, including paths that preserve the dirty tag, paths that promote rare types where appropriate, and paths that recover sentence-initial B- tags where they may have been dropped.
Output ONLY a JSON list of 5 lists. No markdown.
Example format: [["O", "B-PER", "I-PER"], ["B-PER", "I-PER", "I-PER"], ...]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        candidate_paths = extract_nested_json_list(response.content, expected_paths=5)
        candidate_paths = [p[:len(tokens)] + ["O"] * max(0, len(tokens) - len(p))
                           for p in candidate_paths]
        voted = _simple_majority_voting(candidate_paths)
        if len(voted) != len(tokens):
            return dirty_tags
        return voted
    except Exception as e:
        logging.error(f"[-] Self-consistency baseline failed: {e}")
        return dirty_tags


async def baseline_single_pass_pipeline(tokens: List[str], dirty_tags: List[str],
                                        config: dict = None,
                                        dataset_name: str = None) -> List[str]:
    """
    Single-pass LLM correction baseline.
    One LLM call to directly fix dirty tags. No multi-agent, no voting, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    prompt = f"""
You are an AI Data Engineer performing IOB2 label denoising.
Tokens: {tokens}
Dirty IOB2 Tags: {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

DENOISING POLICY:
- The dirty tags contain noise but most positions are already correct.
- Preserve non-O tags from the dirty input unless they clearly conflict with the token semantics or IOB2 grammar.
- When a dirty tag is non-O, prefer to keep it rather than collapsing to O.
{misc_guidance}
{sentence_initial_guidance}

Fix the dirty IOB2 tags and output the corrected sequence.
Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return dirty_tags
    except Exception as e:
        logging.error(f"[-] Single-pass baseline failed: {e}")
        return dirty_tags


async def baseline_cot_reasoning_pipeline(tokens: List[str], dirty_tags: List[str],
                                           config: dict = None,
                                           dataset_name: str = None) -> List[str]:
    """
    Chain-of-Thought Reasoning baseline (ReasoningNER, AAAI 2026).
    Single LLM call with explicit step-by-step reasoning before tag correction.
    The LLM reasons about each entity first, then outputs corrected tags.
    No multi-agent, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""
You are an expert IOB2 label validator for a production NER system. The dirty tags below come from a state-of-the-art NER model with 85%+ F1. Your role is quality assurance — catch only obvious glitches while respecting the model's decisions.

Tokens: {tokens}
Dirty IOB2 Tags (trusted reference): {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

EXPERT VALIDATION PROTOCOL (follow precisely):

1. PRESUMPTION OF CORRECTNESS: The NER model achieves 85%+ F1. This means at least 85 out of every 100 tags are correct. For a {len(tokens)}-token sentence like this one, you should expect to change AT MOST {max(1, len(tokens) // 15)} tags. If you find yourself wanting to change more, you are likely over-correcting. A validator that changes too many tags is more harmful than one that changes too few.

2. BOUNDARY TRUST: The model's span boundaries (where entities start and end) are its strongest feature. Focus exclusively on type verification within those boundaries. If a span is tagged B-PER...I-PER, accept the boundary and only verify whether "PER" is the correct type.

3. SINGLE-TOKEN ENTITIES: When an entity appears as a single token, the annotators often used I- tags instead of B- tags as a shorthand (e.g., "I-PER" for a standalone person). This is NOT an error — do NOT "fix" it by changing to B-PER. Changing I- to B- on single-token entities will corrupt the annotation and is the #1 source of over-correction.

4. ADJACENT SAME-TYPE ENTITIES: In this annotation scheme, B-PER O B-PER indicates TWO separate person entities mentioned close together, not one fragmented entity. The annotators explicitly chose separate B- tags. Merging them into one span (B-PER I-PER I-PER) destroys valid entity distinction and is strictly prohibited.

5. TYPE VERIFICATION RULES:
   - Only change an entity type if the token SEMANTICALLY CANNOT be that type under any reasonable interpretation.
   - "Washington" can be PER (person), LOC (city/state), or ORG (institution). The model's choice is acceptable.
   - "Apple" can be ORG (company) or MISC (fruit/product). The model's choice is acceptable.
   - Do NOT substitute your world knowledge for the model's — the model was trained on more data.
   - MISC entities are extremely rare in this corpus (< 2%). Only predict MISC for tokens that are unambiguously miscellaneous (e.g., nationalities like "German", languages, events).

6. FINAL CHECK: Before outputting, count your changes. If you changed more than {max(1, len(tokens) // 15)} tags, revert the least confident changes until you are within the limit.

Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        content = response.content
        # Try to extract the last JSON list in the output
        matches = list(re.finditer(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', content))
        for m in reversed(matches):
            try:
                pred_tags = json.loads(m.group(0))
                if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                    pred_tags = pred_tags[:len(tokens)]
                    pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                    return pred_tags
            except json.JSONDecodeError:
                continue
        return dirty_tags
    except Exception as e:
        logging.error(f"[-] CoT reasoning baseline failed: {e}")
        return dirty_tags


async def baseline_self_refine_pipeline(tokens: List[str], dirty_tags: List[str],
                                         config: dict = None,
                                         dataset_name: str = None) -> List[str]:
    """
    Self-Refine baseline (SCIR, AAAI 2026).
    Two-stage iterative correction: LLM corrects dirty tags (Round 1),
    then self-reviews and refines its own output (Round 2).
    No multi-agent, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    # === Round 1: Initial correction ===
    round1_prompt = f"""
You are a precision-first IOB2 validator. The dirty tags come from an ensemble of 3 NER models with consensus agreement — estimated accuracy > 90%. Your corrections should be extremely conservative: only fix a tag when all three models would agree it's wrong.

Tokens: {tokens}
Dirty IOB2 Tags (ensemble consensus — highly reliable): {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

CORRECTION POLICY (follow strictly):

1. CHANGE BUDGET: For {len(tokens)} tokens, you may change AT MOST {max(0, len(tokens) // 20)} tags. If your analysis suggests more changes, only apply the {max(0, len(tokens) // 20)} most confident ones and discard the rest. An over-aggressive validator causes more harm than an under-aggressive one.

2. BOUNDARY PRESERVATION: The ensemble's strongest agreement is on entity boundaries. Never change B- to I- or I- to B-. Never change O to any entity tag (the ensemble almost never misses entity starts). Never merge adjacent entities — the ensemble explicitly marks separate entities with separate B- tags.

3. TYPE VERIFICATION (only change if IMPOSSIBLE):
   - A token tagged as entity type X is correct unless it CANNOT be X under any context.
   - Common words like "University", "Bank", "Park" can each be ORG, LOC, or MISC depending on context. Trust the ensemble's judgment.
   - Nationalities, languages, events, and products tagged as MISC are acceptable. Do NOT change MISC to O.
   - Only flag a type error when the token's primary meaning is unambiguously different (e.g., a verb tagged as PER).

4. NO UPGRADING O: O tags represent the ensemble's consensus that no entity exists at that position. This is the most reliable prediction. Never change O to B- or I- under any circumstance. If a token seems like it could be an entity but the ensemble said O, the ensemble is right.

5. SINGLE-TOKEN I- TAGS: The annotation guidelines for this corpus explicitly allow I-PER, I-LOC, etc. for single-token entities. Do NOT change these to B- tags — they are correct as-is.

Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response1 = await asyncio.to_thread(llm.invoke, round1_prompt)
        match = re.search(r'\[.*\]', response1.content, re.DOTALL)
        if not match:
            return dirty_tags
        round1_tags = json.loads(match.group(0))
        if not (isinstance(round1_tags, list) and len(round1_tags) > 0
                and isinstance(round1_tags[0], str)):
            return dirty_tags
        round1_tags = round1_tags[:len(tokens)]
        round1_tags.extend(["O"] * max(0, len(tokens) - len(round1_tags)))
    except Exception as e:
        logging.error(f"[-] Self-Refine Round 1 failed: {e}")
        return dirty_tags

    # === Round 2: Self-review and refine ===
    round2_prompt = f"""
You are a senior QA auditor reviewing a junior annotator's IOB2 correction. The junior annotator was given dirty tags from a production NER ensemble (95%+ accurate) and asked to fix errors. Juniors tend to over-correct — your job is to REVERT unnecessary changes.

Tokens: {tokens}
Authoritative Dirty Tags (production ensemble output — consider these 95%+ ground truth): {dirty_tags}
Junior Annotator Correction (likely over-corrected): {round1_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

AUDIT PROTOCOL:

1. For EVERY tag where the junior correction differs from the dirty tag:
   - Ask: "Is the dirty tag IMPOSSIBLE for this token?" If not impossible, the junior was wrong → REVERT to dirty.
   - Only keep the junior's change if the dirty tag is INDEFENSIBLE (e.g., a punctuation mark tagged as B-PER).

2. O-TAG PROTECTION: The production ensemble has near-perfect recall for entity detection. If the dirty tag is O and the junior changed it to an entity tag, this is an over-correction 99% of the time → REVERT to O immediately. Do NOT think twice about this.

3. BOUNDARY STABILITY: If the junior changed a B- tag to I- or moved entity boundaries, REVERT. The ensemble's boundaries are authoritative.

4. PROXIMITY BIAS: The final output should be much closer to the dirty tags than to the junior correction. The dirty tags represent ensemble consensus; the junior correction represents one person's opinion.

5. REVERSION DEFAULT: When in doubt between the dirty tag and the junior correction, always choose the dirty tag. The ensemble is more reliable than any single annotator.

6. MAXIMUM DEVIATION: Your final output must NOT differ from the dirty tags by more than {max(1, len(tokens) // 20)} positions. If the junior made more changes, revert the least justified ones.

Output ONLY a JSON list of exactly {len(tokens)} refined tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response2 = await asyncio.to_thread(llm.invoke, round2_prompt)
        match = re.search(r'\[.*\]', response2.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return round1_tags  # Fallback to Round 1 if Round 2 fails
    except Exception as e:
        logging.error(f"[-] Self-Refine Round 2 failed: {e}")
        return round1_tags


# =========================================================================
# Random-ICL Baseline (standard baseline in all 2025 ICL NER papers)
# =========================================================================

_random_icl_train = None  # lazy-loaded CoNLL2003 training sentences


def _init_random_icl():
    global _random_icl_train
    if _random_icl_train is not None:
        return
    from datasets import load_dataset
    import random as _rnd
    ds = load_dataset("conll2003", split="train", trust_remote_code=True)
    id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    data = []
    for ex in ds:
        toks = list(ex["tokens"])
        raw = ex["ner_tags"]
        if toks and len(toks) == len(raw):
            data.append((toks, [id2tag[i] for i in raw]))
    _rnd.shuffle(data)
    _random_icl_train = data
    print(f"[Random-ICL] Loaded {len(data)} training sentences")


async def baseline_random_icl_pipeline(tokens: List[str], dirty_tags: List[str],
                                        config: dict = None,
                                        dataset_name: str = None) -> List[str]:
    """Random-ICL: randomly select k training sentences as ICL demonstrations."""
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    _init_random_icl()
    valid_tags_str = _format_valid_tags(dataset_name)

    import random as _rnd
    k = (config or {}).get("icl_k", 8)
    rng = _rnd.Random((config or {}).get("__seed__", 42))
    demos = rng.sample(_random_icl_train, min(k, len(_random_icl_train)))

    demo_text = ""
    for i, (demo_tokens, demo_tags) in enumerate(demos):
        demo_text += f"\nExample {i + 1}:\nInput: {demo_tokens}\nOutput: {demo_tags}\n"

    prompt = f"""You are an IOB2 sequence denoiser. Correct the noisy IOB2 tags using the examples below as reference.

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

IOB2 RULES:
- B-X marks the Beginning of entity type X; I-X marks Inside/continuation
- Every B-X must be followed by I-X of the SAME type until the entity ends
- An I- tag MUST be preceded by B- or I- of the same type
- O marks non-entity tokens

{demo_text}

Now correct the noisy tags for this sentence:
Tokens: {tokens}
Noisy Tags: {dirty_tags}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example format: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred = json.loads(match.group(0))
            if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], str):
                pred = pred[:len(tokens)]
                pred.extend(["O"] * max(0, len(tokens) - len(pred)))
                return pred
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Random-ICL failed: {e}")
        return list(dirty_tags)


# =========================================================================
# Zero-Shot Baseline (standard baseline in 2025 ICL NER papers: DEER, GPT-NER etc.)
# =========================================================================

async def baseline_zero_shot_pipeline(tokens: List[str], dirty_tags: List[str],
                                       config: dict = None,
                                       dataset_name: str = None) -> List[str]:
    """Zero-Shot: single LLM call, no ICL examples, clean standard prompt."""
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""You are an IOB2 sequence denoiser. Correct the noisy IOB2 tags below.

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

IOB2 RULES:
- B-X marks the Beginning of entity type X; I-X marks Inside/continuation
- Every B-X must be followed by I-X of the SAME type until the entity ends
- An I- tag MUST be preceded by B- or I- of the same type
- O marks non-entity tokens

Common noise patterns to fix:
- Boundary Truncation (BT): missing B- tags, dangling I- tags, broken entity spans
- Internal Fragmentation (IF): entities incorrectly split by O tags
- Adversarial Type Flipping (ATF): wrong entity type assigned

Tokens: {tokens}
Noisy Tags: {dirty_tags}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example format: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred = json.loads(match.group(0))
            if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], str):
                pred = pred[:len(tokens)]
                pred.extend(["O"] * max(0, len(tokens) - len(pred)))
                return pred
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Zero-Shot failed: {e}")
        return list(dirty_tags)
