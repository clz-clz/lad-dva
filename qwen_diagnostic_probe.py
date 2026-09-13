"""Run one exact no-thinking Qwen Coder-path diagnostic request.

The probe reads one canonical noisy row but sends only its tokens and dirty
tags to the model.  It never writes target labels or a formal prediction.  A
fresh, empty diagnostics directory is required for each invocation so request
IDs and incremental stream evidence cannot be mixed across observations.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import run_multiseed as runner
from request_diagnostics import select_request_evidence, sentence_context


CONFIG_NAME = "selectdenoise_contextual_lattice"
DEFAULT_PATH = 5
DIAGNOSTIC_REPORT_SCHEMA = "qwen-nothink-coder-diagnostic-v1"
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=runner.DATASETS, required=True)
    parser.add_argument("--noise", choices=runner.NOISE_TYPES, required=True)
    parser.add_argument("--seed", choices=runner.SEEDS, type=int, required=True)
    parser.add_argument("--row-index", type=int, required=True,
                        help="Zero-based row index from the canonical N=200 noisy cell.")
    parser.add_argument("--path", type=int, choices=(1, 2, 3, 4, 5),
                        default=DEFAULT_PATH, help="One Coder strategy to call.")
    parser.add_argument("--diagnostic-root", type=Path, required=True,
                        help="Fresh Git-ignored directory for this observation.")
    args = parser.parse_args(argv)
    if not 0 <= args.row_index < runner.OFFICIAL_SAMPLE_SIZE:
        parser.error("--row-index must be between 0 and 199")
    return args


def _fresh_diagnostic_root(root: Path) -> Path:
    root = Path(root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("diagnostic root must be a fresh empty directory")
    else:
        root.mkdir(parents=True)
    return root


def _load_gold_free_input(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    noisy_path = runner._noisy_path(
        args.dataset, args.noise, args.seed,
        runner.OFFICIAL_SAMPLE_SIZE, runner.OFFICIAL_NOISE_RATIO,
    )
    if not noisy_path.exists():
        raise FileNotFoundError(f"canonical noisy file not found: {noisy_path}")
    rows = runner._load_noisy(noisy_path)
    if len(rows) != runner.OFFICIAL_SAMPLE_SIZE:
        raise RuntimeError("diagnostic requires exactly 200 canonical noisy rows")
    row = rows[args.row_index]
    tokens, dirty = row.get("tokens"), row.get("dirty_tags")
    if (not isinstance(tokens, list) or not isinstance(dirty, list)
            or len(tokens) != len(dirty)
            or not all(isinstance(value, str) for value in tokens + dirty)):
        raise RuntimeError("diagnostic source row has invalid tokens or dirty tags")
    # Deliberately return only model-facing, gold-free fields.  The source row
    # may contain ner_tags, but they never enter the probe state or report.
    return list(tokens), list(dirty)


def _diagnostic_report(
    *, args: argparse.Namespace, root: Path, settings: Any,
    request_ids: list[str], status: str, result: Mapping[str, Any] | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    evidence = []
    for request_id in request_ids:
        selected = select_request_evidence(root, request_id)
        evidence.append({
            "request_id": request_id,
            "request_file": selected["request_path"].name
            if selected["request_path"] is not None else None,
            "stream_file": selected["stream_path"].name
            if selected["stream_path"] is not None else None,
            "event_count": len(selected["events"]),
        })
    candidate_paths = result.get("terminal_candidate_paths") if isinstance(result, Mapping) else None
    token_count = (
        len(candidate_paths[0])
        if isinstance(candidate_paths, list) and candidate_paths
        and isinstance(candidate_paths[0], list)
        else None
    )
    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_REPORT_SCHEMA,
        "status": status,
        "dataset": args.dataset,
        "noise": args.noise,
        "seed": args.seed,
        "row_index": args.row_index,
        "coder_path": args.path,
        "token_count": token_count,
        "backbone": {
            "provider": settings.provider,
            "model": settings.model,
            "served_model": settings.served_model,
            "revision": settings.revision,
            "structured_api": settings.structured_api,
            "enable_thinking": settings.configured_enable_thinking,
            "thinking_mode": settings.thinking_mode,
            "timeout_seconds": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        },
        "request_ids": evidence,
        "formal_prediction": False,
    }
    if error_type is not None:
        report["error_type"] = error_type
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = _fresh_diagnostic_root(args.diagnostic_root)
    os.environ["QWEN_DIAGNOSTICS_DIR"] = str(root)
    settings, _tag = runner._official_settings_from_env([CONFIG_NAME], os.environ)
    if settings.configured_enable_thinking is not False:
        raise ValueError("diagnostic requires QWEN_ENABLE_THINKING=false")
    if settings.timeout_seconds != 300.0:
        raise ValueError("diagnostic requires QWEN_PROVIDER_TIMEOUT_SECONDS=300")
    # Diagnosis must make one SDK attempt.  Formal provider-cache execution
    # retains its independent max-retries=2 setting.
    settings = replace(settings, max_retries=0)
    tokens, dirty_tags = _load_gold_free_input(args)

    from live_backbone import OpenAICompatibleLADRGAdapter
    from multi_agent_v2 import coder_node

    adapter = OpenAICompatibleLADRGAdapter(settings)
    state = {
        "official": True,
        "tokens": tokens,
        "dirty_tags": dirty_tags,
        "dataset_name": args.dataset,
        "noise_type": args.noise,
        "structured_requester": adapter.structured_requester,
        "provider_settings": adapter.provider_metadata(),
        "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "verifier")},
        "__diagnostic_coder_path__": args.path,
    }
    request_ids: list[str] = []
    try:
        with sentence_context(
            dataset=args.dataset, noise=args.noise, seed=args.seed,
            row_index=args.row_index,
        ):
            result = asyncio.run(coder_node(state))
        events_path = root / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                request_id = event.get("request_id")
                if (event.get("event") == "request_started"
                        and isinstance(request_id, str)
                        and _REQUEST_ID_RE.fullmatch(request_id)):
                    request_ids.append(request_id)
        # Only the selected path may issue a logical request.
        if len(set(request_ids)) != 1:
            raise RuntimeError(
                f"diagnostic expected exactly one logical request, found {len(set(request_ids))}"
            )
        report = _diagnostic_report(
            args=args, root=root, settings=settings, request_ids=sorted(set(request_ids)),
            status="completed", result=result,
        )
    except Exception as exc:  # noqa: BLE001 - report only the safe exception type
        events_path = root / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                request_id = event.get("request_id")
                if (event.get("event") == "request_started"
                        and isinstance(request_id, str)
                        and _REQUEST_ID_RE.fullmatch(request_id)):
                    request_ids.append(request_id)
        report = _diagnostic_report(
            args=args, root=root, settings=settings, request_ids=sorted(set(request_ids)),
            status="failed", error_type=type(exc).__name__,
        )
        adapter.close() if "adapter" in locals() else None
        (root / "diagnostic-report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
        return 1
    finally:
        if "adapter" in locals():
            adapter.close()
    (root / "diagnostic-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
