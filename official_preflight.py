"""Read-only launch checks for official LAD-RG experiments.

Static mode validates the repository, canonical noisy-data matrix, DEER
initialization, and tagged prediction manifests.  Live mode is explicitly
selected and validates the configured OpenAI-compatible endpoint and adapter
schemas.  Neither mode creates or modifies prediction artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from live_backbone import LiveBackboneSettings, OpenAICompatibleLADRGAdapter
from run_multiseed import (
    DATASETS,
    NOISE_TYPES,
    OFFICIAL_NOISE_RATIO,
    OFFICIAL_SAMPLE_SIZE,
    SEEDS,
    _build_official_manifest,
    _canonical_json,
    _is_legal_iob2,
    _manifest_path,
    _official_settings_from_env,
    _validate_official_settings_for_configs,
    _validate_official_prediction,
    _valid_tags,
)


def _block(blockers: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    blockers.append({"code": code, "message": message, **details})


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _canonical_noisy_paths(noisy_dir: Path) -> set[Path]:
    return {
        noisy_dir / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
        for dataset in DATASETS for noise in NOISE_TYPES for seed in SEEDS
    }


def _validate_noisy_matrix(noisy_dir: Path, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _canonical_noisy_paths(noisy_dir)
    actual = set(noisy_dir.glob("noisy_seed*.jsonl")) if noisy_dir.is_dir() else set()
    missing = sorted(path.name for path in expected - actual)
    extra = sorted(path.name for path in actual - expected)
    if missing or extra:
        _block(
            blockers, "noisy_file_matrix", "canonical noisy-data matrix is incomplete or has extras",
            missing=missing, extra=extra,
        )

    records = 0
    invalid_records = 0
    for path in sorted(expected & actual):
        dataset = path.name.split("__")[2]
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _block(blockers, "invalid_noisy_file", f"cannot read {path.name}: {exc}")
            continue
        if len(lines) != OFFICIAL_SAMPLE_SIZE:
            _block(
                blockers, "invalid_noisy_count",
                f"{path.name} has {len(lines)} records; expected {OFFICIAL_SAMPLE_SIZE}",
            )
        records += len(lines)
        valid_tags = _valid_tags(dataset)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                tokens = row.get("tokens")
                gold = row.get("ner_tags")
                dirty = row.get("dirty_tags")
                lists = (tokens, gold, dirty)
                valid = (
                    all(isinstance(value, list) for value in lists)
                    and all(isinstance(item, str) for value in lists for item in value)
                    and len(tokens) > 0
                    and len(tokens) == len(gold) == len(dirty)
                    and all(tag in valid_tags for tag in gold + dirty)
                    and _is_legal_iob2(gold)
                )
            except (AttributeError, TypeError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_records += 1
                if invalid_records <= 20:
                    _block(
                        blockers, "invalid_noisy_record",
                        f"{path.name} record {index} violates the official data contract",
                    )
    return {"files": len(actual), "records": records, "invalid_records": invalid_records,
            "missing": missing, "extra": extra}


def _one_deer_check(repo_root: Path, dataset: str, timeout: float) -> dict[str, Any]:
    code = (
        "import json, multi_agent_v2 as m; "
        f"m._init_deer({dataset!r}); "
        f"assert {dataset!r} in m._deer_stats and {dataset!r} in m._deer_retriever; "
        "print(json.dumps({'ok': True}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout:g} seconds"}
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-2000:]}
    return {"ok": True}


def _default_deer_checker(
    datasets: Sequence[str], timeout: float, *, repo_root: Path
) -> dict[str, dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(datasets)) as pool:
        futures = {
            dataset: pool.submit(_one_deer_check, repo_root, dataset, timeout)
            for dataset in datasets
        }
        return {dataset: future.result() for dataset, future in futures.items()}


def _check_tagged_artifacts(
    pred_dir: Path, settings: LiveBackboneSettings, tag: str, expected_sha: str,
    request_timeout: float, blockers: list[dict[str, Any]],
    config_names: Sequence[str],
) -> dict[str, Any]:
    manifest_path = _manifest_path(pred_dir, tag)
    canonical_outputs = {
        f"pred_seed{seed}__{config_name}__{dataset}__{noise}__{tag}.jsonl"
        for config_name in config_names for dataset in DATASETS
        for noise in NOISE_TYPES for seed in SEEDS
    }
    canonical_temps = {name + ".tmp" for name in canonical_outputs}

    def has_selected_tag(path: Path) -> bool:
        parts = path.name.split("__")
        if len(parts) < 5:
            return False
        return any(
            component == tag or component.startswith(f"{tag}.")
            for component in parts[4:]
        )

    tagged_artifacts = sorted(
        path for path in pred_dir.glob("pred_*")
        if pred_dir.is_dir() and has_selected_tag(path)
    ) if pred_dir.is_dir() else []
    noncanonical = [
        path for path in tagged_artifacts
        if path.name not in canonical_outputs and path.name not in canonical_temps
    ]
    if noncanonical:
        _block(
            blockers, "noncanonical_prediction_artifact",
            "tagged prediction artifacts fall outside the selected official namespace",
            files=[path.name for path in noncanonical],
        )
    outputs = [path for path in tagged_artifacts if path.name in canonical_outputs]
    temp_outputs = [path for path in tagged_artifacts if path.name in canonical_temps]
    expected_manifest = _build_official_manifest(settings, tag, expected_sha, request_timeout)
    compatible_manifest = False
    if manifest_path.exists():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            compatible_manifest = _canonical_json(actual) == _canonical_json(expected_manifest)
        except (OSError, json.JSONDecodeError):
            compatible_manifest = False
        if not compatible_manifest:
            _block(blockers, "incompatible_manifest", f"{manifest_path.name} is not canonical/exact")
    elif outputs:
        _block(blockers, "missing_manifest", f"tagged outputs exist without {manifest_path.name}")
    if temp_outputs:
        _block(
            blockers, "stale_temporary_output", "tagged temporary predictions remain",
            files=[path.name for path in temp_outputs],
        )
    if compatible_manifest:
        for path in outputs:
            parts = path.stem.split("__")
            if len(parts) < 5:
                _block(blockers, "incompatible_prediction", f"unrecognized tagged output {path.name}")
                continue
            config_name, dataset, noise_type = parts[1], parts[2], parts[3]
            try:
                _validate_official_prediction(
                    path, dataset, config_name, noise_type,
                    provider_identity={
                        "provider": settings.provider,
                        "model": settings.model,
                        "served_model": settings.served_model,
                        "revision": settings.revision,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - report all blockers as JSON
                _block(blockers, "incompatible_prediction", str(exc), file=path.name)
    return {
        "manifest": manifest_path.name,
        "manifest_exists": manifest_path.exists(),
        "manifest_compatible": compatible_manifest if manifest_path.exists() else None,
        "outputs": len(outputs),
        "temporary_outputs": len(temp_outputs),
    }


def run_static_preflight(
    *, repo_root: Path, expected_sha: str, noisy_dir: Path, pred_dir: Path,
    environment: Mapping[str, str] = os.environ, deer_timeout: float = 300.0,
    deer_checker: Callable[[Sequence[str], float], Mapping[str, Mapping[str, Any]]] | None = None,
    request_timeout: float = 600.0, config_names: Sequence[str] = ("lad_rg_full",),
) -> dict[str, Any]:
    """Run all read-only static checks and return a JSON-serializable report."""
    repo_root, noisy_dir, pred_dir = map(Path, (repo_root, noisy_dir, pred_dir))
    blockers: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    settings: LiveBackboneSettings | None = None
    tag = ""
    try:
        settings, tag = _official_settings_from_env(list(config_names), environment)
        checks["configuration"] = {
            "provider": settings.provider, "model": settings.model,
            "revision": settings.revision, "tag": tag, "configs": list(config_names),
        }
    except Exception as exc:  # noqa: BLE001 - converted into a launch blocker
        _block(blockers, "invalid_configuration", str(exc))

    try:
        actual_sha = _git_output(repo_root, "rev-parse", "HEAD")
        tracked_status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=no")
        checks["git"] = {"actual_sha": actual_sha, "expected_sha": expected_sha,
                         "tracked_clean": not bool(tracked_status)}
        if actual_sha != expected_sha:
            _block(blockers, "git_sha_mismatch", f"HEAD {actual_sha} != expected {expected_sha}")
        if tracked_status:
            _block(blockers, "git_dirty", "tracked worktree changes are present")
    except (OSError, subprocess.CalledProcessError) as exc:
        _block(blockers, "git_check_failed", str(exc))

    checks["noisy_files"] = _validate_noisy_matrix(noisy_dir, blockers)

    try:
        deer_results = (
            deer_checker(DATASETS, deer_timeout)
            if deer_checker is not None
            else _default_deer_checker(DATASETS, deer_timeout, repo_root=repo_root)
        )
    except Exception as exc:  # noqa: BLE001
        deer_results = {name: {"ok": False, "error": str(exc)} for name in DATASETS}
    checks["deer"] = dict(deer_results)
    for dataset in DATASETS:
        result = deer_results.get(dataset, {})
        if result.get("ok") is not True:
            _block(blockers, "deer_initialization_failed", f"DEER failed for {dataset}",
                   dataset=dataset, result=dict(result))

    if settings is not None:
        checks["tagged_artifacts"] = _check_tagged_artifacts(
            pred_dir, settings, tag, expected_sha, request_timeout, blockers,
            config_names,
        )
    else:
        checks["tagged_artifacts"] = {"skipped": True}
    return {"mode": "static", "ok": not blockers, "checks": checks, "blockers": blockers}


def _models_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/models"):
        models_path = path
    elif path.endswith("/v1"):
        models_path = path + "/models"
    else:
        models_path = path + "/v1/models"
    return urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))


def _default_models_fetcher(settings: LiveBackboneSettings) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        _models_url(settings.base_url),
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("/v1/models did not return a data list")
    return models


def run_live_preflight(
    *, settings: LiveBackboneSettings,
    config_names: Sequence[str] = ("lad_rg_full",),
    models_fetcher: Callable[[LiveBackboneSettings], Sequence[Mapping[str, Any]]] = _default_models_fetcher,
    adapter_factory: Callable[[LiveBackboneSettings], OpenAICompatibleLADRGAdapter] = OpenAICompatibleLADRGAdapter,
) -> dict[str, Any]:
    """Run opt-in provider checks through the same strict adapter as official runs."""
    blockers: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    fingerprints: set[str] = set()
    try:
        _validate_official_settings_for_configs(settings, config_names)
        models = list(models_fetcher(settings))
        model_ids = [item.get("id") for item in models if isinstance(item, Mapping)]
        expected_model_id = (
            settings.served_model if settings.provider == "vllm" else settings.model
        )
        if expected_model_id not in model_ids:
            raise ValueError(
                f"configured immutable identity is absent from /v1/models: {expected_model_id}"
            )
        checks["models"] = {"ids": model_ids, "configured_model_present": True}

        adapter = adapter_factory(settings)
        expected_identity = {
            "provider": settings.provider,
            "model": settings.model,
            "served_model": settings.served_model,
            "revision": settings.revision,
        }
        adapter_identity = adapter.provider_metadata()
        if (not isinstance(adapter_identity, Mapping)
                or any(adapter_identity.get(key) != value
                       for key, value in expected_identity.items())):
            raise ValueError("live adapter identity does not match official settings")
        tokens = ["Alice", "met", "Paris"]
        valid_types = ["PER", "LOC", "ORG", "MISC"]
        spans = adapter.ror_reasoner(
            "span_detection", {"tokens": tokens, "valid_types": valid_types, "dirty_tags": ["O"] * 3},
        )
        typed = adapter.ror_reasoner(
            "type_assignment", {"tokens": tokens, "valid_types": valid_types,
                                "dirty_tags": ["O"] * 3, "spans": spans["spans"]},
        )
        checks["ror"] = {"spans": spans["spans"], "types": typed["types"]}
        evidence = [(spans, "ror_span_detection"), (typed, "ror_type_assignment")]
        if settings.provider == "vllm":
            valid_tags = ["O"] + [
                f"{prefix}-{entity_type}" for entity_type in valid_types for prefix in ("B", "I")
            ]
            gasd = adapter.gasd_reason_decoder({
                "tokens": tokens, "valid_tags": valid_tags,
                "candidate_paths": [["O", "O", "O"]], "weights": [1.0],
                "ror_proposals": {}, "constraint": "hard_iob2",
            })
            checks["gasd_r"] = {"tag_count": len(gasd["tags"]), "tags": gasd["tags"]}
            evidence.append((gasd, "gasd_r"))
        for result, expected_stage in evidence:
            metadata = getattr(result, "provider_metadata", {})
            if (not isinstance(metadata, Mapping)
                    or metadata.get("stage") != expected_stage
                    or metadata.get("status") != "live"
                    or metadata.get("response_model") != settings.served_model
                    or any(metadata.get(key) != value
                           for key, value in expected_identity.items())):
                raise ValueError(
                    f"live provider evidence does not match official identity for {expected_stage}"
                )
            fingerprint = metadata.get("system_fingerprint")
            if fingerprint:
                fingerprints.add(str(fingerprint))
        checks["provider_metadata"] = {
            "settings": dict(adapter_identity),
            "system_fingerprints": sorted(fingerprints),
            "fingerprint_supported": bool(fingerprints),
        }
    except Exception as exc:  # noqa: BLE001 - one fail-closed live blocker
        _block(blockers, "live_validation_failed", str(exc))
    return {"mode": "live", "ok": not blockers, "checks": checks, "blockers": blockers}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "live"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-sha")
    parser.add_argument("--noisy-root", type=Path, default=Path("results_multiseed"))
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions_multiseed"))
    parser.add_argument("--deer-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="Runner timeout expected in the exact manifest.")
    parser.add_argument("--configs", nargs="+", default=["lad_rg_full"],
                        help="Official LAD-RG configurations intended for this launch.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "static":
        if not args.expected_sha:
            report = {"mode": "static", "ok": False, "checks": {}, "blockers": [{
                "code": "missing_expected_sha", "message": "--expected-sha is required",
            }]}
        else:
            try:
                report = run_static_preflight(
                    repo_root=args.repo_root, expected_sha=args.expected_sha,
                    noisy_dir=args.noisy_root, pred_dir=args.predictions_root,
                    deer_timeout=args.deer_timeout, request_timeout=args.request_timeout,
                    config_names=args.configs,
                )
            except Exception as exc:  # noqa: BLE001 - preserve JSON-only stdout
                report = {"mode": "static", "ok": False, "checks": {}, "blockers": [{
                    "code": "preflight_internal_error", "message": str(exc),
                }]}
    else:
        try:
            settings, _ = _official_settings_from_env(list(args.configs), os.environ)
            report = run_live_preflight(settings=settings, config_names=args.configs)
        except Exception as exc:  # noqa: BLE001
            report = {"mode": "live", "ok": False, "checks": {}, "blockers": [{
                "code": "invalid_configuration", "message": str(exc),
            }]}
    sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
