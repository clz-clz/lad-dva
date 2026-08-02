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
    python run_multiseed.py --configs lad_dva_full semantic_rag_baseline
    python run_multiseed.py --seeds 13 42 2024
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

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
PER_REQUEST_TIMEOUT = 180  # seconds per sentence; straggler → dirty fallback

# Configurations — each key maps to either a pipeline config or a standalone method.
CONFIGURATIONS: Dict[str, dict] = {
    # SelectDenoise: Coder(+ATF de-anchoring) -> Reviewer(LADS/DEER) -> Verifier
    "selectdenoise_full":        {"deanchor_atf": True,  "use_verifier": True},
    # Ablations to attribute the gain to each lever
    "selectdenoise_no_verifier": {"deanchor_atf": True,  "use_verifier": False},
    "selectdenoise_no_deanchor": {"deanchor_atf": False, "use_verifier": True},
    "selectdenoise_verify_all":  {"deanchor_atf": True,  "use_verifier": True,
                                  "verify_all": True},
    "selectdenoise_vote":        {"deanchor_atf": False, "use_verifier": False},
    # Legacy LAD-RG (graph no longer wires ror/gasd; kept for filename compat)
    "lad_rg_full":               {"use_ror": True,  "gasd_potentials": True},
    "lad_rg_no_ror":             {"use_ror": False, "gasd_potentials": True},
    "lad_rg_no_potentials":      {"use_ror": True,  "gasd_potentials": False},
    "lad_rg_ror_ungated":        {"use_ror": True,  "gasd_potentials": True,
                                  "ror_ungated": True},
    # Baselines (2024-2025 published & standard)
    "baseline_zero_shot":         {"method": "zero_shot"},
    "baseline_cot_reasoning":     {"method": "cot_reasoning"},
    "baseline_self_refine":       {"method": "self_refine"},
    "baseline_rule_only":         {"method": "rule_only"},
    "baseline_standard_prompting": {"method": "standard_prompting"},
}

NOISY_DIR = Path("results_multiseed")
PRED_DIR  = Path("predictions_multiseed")
PRED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline import (real or mock)
# ---------------------------------------------------------------------------

def _import_pipeline(dummy: bool):
    """Return (default_pipeline, baseline_pipelines_dict).

    default_pipeline: async fn for lad_dva_full.
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
    # Same backward-compat scheme as _noisy_path.
    if abs(ratio - 0.15) < 1e-9:
        return PRED_DIR / f"pred_seed{seed}__{config_name}__{dataset}__{noise}.jsonl"
    rate_pct = int(round(ratio * 100))
    return PRED_DIR / f"pred_seed{seed}__{config_name}__{dataset}__{noise}__r{rate_pct}.jsonl"


def _load_noisy(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-cell async runner
# ---------------------------------------------------------------------------

async def _run_one_cell(config_name: str, config: dict,
                        dataset: str, noise: str, seed: int,
                        size: int, pipelines,
                        *, max_concurrency: int, dummy: bool,
                        ratio: float = 0.15):
    default_fn, baseline_fns = pipelines
    pred_p = _pred_path(config_name, dataset, noise, seed, ratio)
    if pred_p.exists() and pred_p.stat().st_size > 0:
        logging.info(f"[skip] {pred_p.name}")
        return

    noisy_p = _noisy_path(dataset, noise, seed, size, ratio)
    if not noisy_p.exists():
        logging.warning(f"[miss] noisy file not found: {noisy_p}; "
                        f"run gen_noisy.py first")
        return

    rows = _load_noisy(noisy_p)
    logging.info(f"[run]  {pred_p.name}  ({len(rows)} sentences)")
    t0 = time.time()

    sem = asyncio.Semaphore(max_concurrency)
    # Buffer results IN INPUT ORDER so prediction files line up across methods.
    buffer: List[Optional[dict]] = [None] * len(rows)

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
            # Log the Coder candidate pool for the main methods only (oracle /
            # selection analysis); keeps ablation prediction files lean.
            if config_name in ("selectdenoise_full", "selectdenoise_no_deanchor",
                               "lad_rg_full"):
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
                try:
                    return await default_fn(tokens, dirty, cfg, dataset_name=dataset)
                except TypeError:
                    return await default_fn(tokens, dirty, cfg)

            try:
                # Per-request timeout: a single straggler sentence (e.g. very long
                # MSRA input) must not stall the whole cell's async gather.
                pred = await asyncio.wait_for(_call(), timeout=PER_REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                logging.warning(f"[timeout] sentence {i} exceeded "
                                f"{PER_REQUEST_TIMEOUT}s; using dirty as fallback")
                pred = list(dirty)
            except Exception as e:                       # noqa: BLE001
                logging.error(f"[!] sentence {i} failed ({e!r}); using dirty as fallback")
                pred = list(dirty)
            # Candidate-rich return (dict) → unpack pred_tags + extra fields.
            if isinstance(pred, dict):
                extra = {k: pred[k] for k in
                         ("candidate_paths", "rag_weights", "confidence")
                         if k in pred}
                pred = pred.get("pred_tags", list(dirty))
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
        from tqdm.asyncio import tqdm as async_tqdm
        await async_tqdm.gather(
            *(_process(i, r) for i, r in enumerate(rows)),
            desc=f"{dataset}/{noise}/seed{seed}/{config_name}/{rate_tag}",
            leave=False,
        )
    except ImportError:
        await asyncio.gather(*(_process(i, r) for i, r in enumerate(rows)))

    # Write to disk in input order, atomically (via temp file + rename)
    tmp_p = pred_p.with_suffix(pred_p.suffix + ".tmp")
    with tmp_p.open("w", encoding="utf-8") as f_out:
        for rec in buffer:
            if rec is None:
                continue
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_p.replace(pred_p)

    logging.info(f"[done] {pred_p.name} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",  nargs="+", default=list(CONFIGURATIONS.keys()))
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
    args = ap.parse_args(argv)

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
                            ))
                            n_done += 1
                        except Exception:                    # noqa: BLE001
                            logging.error("[!] cell raised")
                            traceback.print_exc(file=sys.stderr)
                            n_err += 1

    logging.info(f"summary: {n_done} ok, {n_err} failed")


if __name__ == "__main__":
    main()