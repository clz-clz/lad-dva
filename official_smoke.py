"""Run one paid 20-sentence LAD-RG launch smoke without publishing official output.

The smoke reads the first 20 rows of one canonical N=200 noisy cell and uses
the same fail-closed pipeline/evidence validation as the official runner.  Its
artifacts live under ``predictions_multiseed/smoke`` and are never resumable
official evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from metrics import compute_ser
import run_multiseed as runner


PAID_SMOKE_SIZE = runner.PAID_SMOKE_SIZE
SMOKE_REPORT_SCHEMA = "lad-rg-paid-smoke-v1"
SMOKE_CONFIGS = ("lad_rg_full", "lad_rg_gasd_r", "lad_rg_gasd_both")


def select_contextual_smoke_rows(
    cells: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
) -> list[tuple[str, str, int, int]]:
    """Return reproducible fixed and length-distribution representatives.

    The historical fixed records use zero-based row indices.  Percentiles are
    nearest-rank over the stable input order after sorting by token length.
    """
    selected: set[tuple[str, str, int, int]] = {
        entry for entry in (("msra", "BT", 13, 125), ("msra", "BT", 13, 133),
                            ("msra", "ATF", 13, 2))
        if entry[:3] in cells
    }
    for (dataset, noise, seed), rows in cells.items():
        if len(rows) != runner.OFFICIAL_SAMPLE_SIZE:
            raise ValueError("contextual smoke selection requires canonical 200-row cells")
        ordered = sorted(range(len(rows)), key=lambda index: (len(rows[index].get("tokens", [])), index))
        for percentile in (0.50, 0.95, 1.0):
            rank = max(0, min(len(ordered) - 1, int(percentile * len(ordered)) - 1))
            selected.add((dataset, noise, seed, ordered[rank]))
    return sorted(selected)


def _candidate_evidence_present(row: Mapping[str, Any]) -> bool:
    tokens = row.get("tokens")
    paths = row.get("candidate_paths")
    weights = row.get("rag_weights")
    confidence = row.get("confidence")
    return (
        isinstance(tokens, list)
        and isinstance(paths, list)
        and bool(paths)
        and all(isinstance(path, list) and len(path) == len(tokens) for path in paths)
        and isinstance(weights, list)
        and len(weights) == len(paths)
        and isinstance(confidence, list)
        and len(confidence) == len(tokens)
    )


def _has_live_gasd_r(row: Mapping[str, Any]) -> bool:
    metadata = row.get("provider_metadata")
    if not isinstance(metadata, Mapping):
        return False
    records = metadata.get("gasd")
    return isinstance(records, list) and any(
        isinstance(record, Mapping)
        and record.get("stage") == "gasd_r"
        and record.get("status") == "live"
        for record in records
    )


def assess_smoke_rows(rows: Sequence[Mapping[str, Any]], *, config_name: str) -> dict[str, Any]:
    """Return the explicit authorization gate for a completed paid smoke."""
    records = list(rows)
    blockers: list[dict[str, Any]] = []
    fallback_count = sum(row.get("fallback_used") is not False for row in records)
    candidate_count = sum(_candidate_evidence_present(row) for row in records)
    predictions = [row.get("pred_tags", []) for row in records]
    gasd_ser = compute_ser(predictions)
    config = runner.CONFIGURATIONS.get(config_name, {})
    variant = str(config.get("gasd_variant", "g")).lower()
    gasd_variant_count = sum(
        row.get("gasd_variant_requested") == variant
        and row.get("gasd_variant_used") == variant
        for row in records
    )
    live_gasd_r_count = sum(_has_live_gasd_r(row) for row in records)

    if len(records) != PAID_SMOKE_SIZE:
        blockers.append({
            "code": "record_count",
            "message": f"paid smoke produced {len(records)} records, expected {PAID_SMOKE_SIZE}",
        })
    if fallback_count:
        blockers.append({
            "code": "fallback_used",
            "message": f"{fallback_count} paid-smoke rows used or omitted fallback evidence",
        })
    if candidate_count != len(records):
        blockers.append({
            "code": "candidate_evidence",
            "message": f"candidate evidence is complete for {candidate_count}/{len(records)} rows",
        })
    if gasd_variant_count != len(records):
        blockers.append({
            "code": "gasd_variant",
            "message": (
                f"requested/used GASD-{variant.upper()} evidence is complete for "
                f"{gasd_variant_count}/{len(records)} rows"
            ),
        })
    # ``pred_tags`` is the terminal output of GASD's hard-I/O-B2 Viterbi
    # decoder. Provider GASD-R tags are reason-score inputs to that decoder and
    # are intentionally not treated as an already-decoded tag sequence.
    if gasd_ser != 0.0:
        blockers.append({
            "code": "gasd_ser",
            "message": f"paid-smoke GASD SER is {gasd_ser:.12g}, expected 0",
        })
    if variant in {"r", "both"} and live_gasd_r_count != len(records):
        blockers.append({
            "code": "live_gasd_r",
            "message": f"live GASD-R evidence is complete for {live_gasd_r_count}/{len(records)} rows",
        })
    return {
        "ok": not blockers,
        "record_count": len(records),
        "fallback_count": fallback_count,
        "candidate_evidence_count": candidate_count,
        "gasd_variant_evidence_count": gasd_variant_count,
        "live_gasd_r_count": live_gasd_r_count,
        "gasd_ser": gasd_ser,
        "blockers": blockers,
    }


def _smoke_output_path(
    output_root: Path, *, tag: str, config_name: str, dataset: str,
    noise: str, seed: int,
) -> Path:
    return Path(output_root) / (
        f"smoke_seed{seed}__{config_name}__{dataset}__{noise}__{tag}.jsonl"
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=SMOKE_CONFIGS)
    parser.add_argument("--dataset", required=True, choices=runner.DATASETS)
    parser.add_argument("--noise", required=True, choices=runner.NOISE_TYPES)
    parser.add_argument("--seed", required=True, type=int, choices=runner.SEEDS)
    parser.add_argument("--noisy-root", type=Path, default=Path("results_multiseed"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("predictions_multiseed") / "smoke"
    )
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=runner.OFFICIAL_REQUEST_TIMEOUT)
    args = parser.parse_args(argv)
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def _failure_report(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_REPORT_SCHEMA,
        "ok": False,
        "blockers": [{"code": "smoke_failed", "message": str(exc)}],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings, tag = runner._official_settings_from_env([args.config], os.environ)
        runner._configure_official_request_model(settings)
        runner.BACKBONE_TAG = tag
        runner.NOISY_DIR = args.noisy_root
        output_path = _smoke_output_path(
            args.output_root, tag=tag, config_name=args.config,
            dataset=args.dataset, noise=args.noise, seed=args.seed,
        )
        report_path = output_path.with_suffix(".report.json")
        if output_path.exists() or report_path.exists():
            raise FileExistsError(
                "paid smoke artifacts already exist; select a new BACKBONE_TAG or move them first"
            )

        from live_backbone import OpenAICompatibleLADRGAdapter

        pipelines = runner._import_pipeline(False)
        adapter_factory = runner._CachedAdapterFactory(
            lambda: OpenAICompatibleLADRGAdapter(settings)
        )
        # Pipeline nodes retain historical human-readable print diagnostics.
        # Keep stdout as one machine-readable report by routing those diagnostics
        # to stderr for this operational CLI.
        try:
            with redirect_stdout(sys.stderr):
                asyncio.run(runner._run_one_cell(
                    args.config, runner.CONFIGURATIONS[args.config],
                    args.dataset, args.noise, args.seed, runner.OFFICIAL_SAMPLE_SIZE,
                    pipelines, max_concurrency=args.max_concurrency, dummy=False,
                    ratio=runner.OFFICIAL_NOISE_RATIO, official=False,
                    paid_smoke_size=PAID_SMOKE_SIZE, prediction_path=output_path,
                    failure_policy="abort", request_timeout=args.request_timeout,
                    adapter_factory=adapter_factory,
                ))
        finally:
            adapter_factory.close()
        rows = runner._load_noisy(output_path)
        gate = assess_smoke_rows(rows, config_name=args.config)
        fingerprints = sorted({
            str(record["system_fingerprint"])
            for row in rows
            for records in (
                row.get("provider_metadata", {}).values()
                if isinstance(row.get("provider_metadata"), Mapping) else []
            )
            for record in records
            if isinstance(record, Mapping) and record.get("system_fingerprint")
        })
        report = {
            "schema_version": SMOKE_REPORT_SCHEMA,
            "ok": gate["ok"],
            "git_sha": _git_sha(),
            "backbone": {
                "provider": settings.provider,
                "model": settings.model,
                "served_model": settings.served_model,
                "revision": settings.revision,
                "structured_api": settings.structured_api,
                "endpoint_origin": runner._endpoint_origin(settings.base_url),
                "tag": tag,
                "system_fingerprints": fingerprints,
            },
            "cell": {
                "config": args.config, "dataset": args.dataset,
                "noise": args.noise, "seed": args.seed,
                "canonical_source_size": runner.OFFICIAL_SAMPLE_SIZE,
                "smoke_records": PAID_SMOKE_SIZE,
            },
            "output": {
                "path": str(output_path),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            },
            "gate": gate,
            "blockers": gate["blockers"],
        }
        _write_json_atomic(report_path, report)
    except Exception as exc:  # noqa: BLE001 - CLI must emit one machine-readable failure
        report = _failure_report(exc)
    sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
