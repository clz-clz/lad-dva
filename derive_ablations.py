"""
derive_ablations.py — regenerate SelectDenoise ablation predictions OFFLINE
(zero LLM calls) from the candidate pools already logged by selectdenoise_full
(and, for vote-ATF, by selectdenoise_no_deanchor).

Two ablations are pure functions of an existing candidate pool, so they need no
new API calls:

  * ``no_verifier`` = the pipeline with Lever 2 (LLM Verifier) OFF. When
    ``use_verifier=False`` the shipped ``verifier_node`` returns exactly
        legalize_noise_aware(
            _weighted_majority_voting(cands, weights, entity_boost=1.0,
                                      dirty_tags, consensus_ratio=0.7),
            valid_set, "demote" if IF else "promote")
    which depends only on the logged ``candidate_paths`` / ``rag_weights`` and
    the ``dirty_tags`` from the noisy file. selectdenoise_full logged these in
    every cell, so no_verifier is reproducible for the whole grid.

  * ``vote`` = both levers OFF. On BT/IF de-anchoring is a no-op, so
    vote == no_verifier there (copied). On ATF vote uses the *no_deanchor*
    candidate pool (de-anchor OFF changes the Coder), derived the same way.

The decode is produced by calling the SAME functions the pipeline uses
(imported, not re-implemented), so the output matches the shipped code path.

Usage:
    python derive_ablations.py no_verifier      # all 45 cells (needs full preds)
    python derive_ablations.py vote             # BT/IF copy + ATF from no_deanchor
    python derive_ablations.py all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from multi_agent_v2 import _base_decode, _init_deer, DATASET_ENTITY_TYPES
from utils import legalize_noise_aware
from run_multiseed import DATASETS, NOISE_TYPES, SEEDS, PRED_DIR, _pred_path, _noisy_path


def _load_full_pred(ds: str, nt: str, seed: int) -> Optional[List[dict]]:
    """Rows of selectdenoise_full (tokens, gold_tags, candidate_paths, rag_weights)."""
    p = _pred_path("selectdenoise_full", ds, nt, seed)
    if not (p.exists() and p.stat().st_size > 0):
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def _load_pred(config: str, ds: str, nt: str, seed: int) -> Optional[List[dict]]:
    p = _pred_path(config, ds, nt, seed)
    if not (p.exists() and p.stat().st_size > 0):
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def _load_dirty(ds: str, nt: str, seed: int, size: int = 200) -> Optional[List[List[str]]]:
    p = _noisy_path(ds, nt, seed, size)
    if not (p.exists() and p.stat().st_size > 0):
        return None
    return [json.loads(l)["dirty_tags"] for l in p.open(encoding="utf-8")]


def _decode_no_verifier(row: dict, dirty: List[str], valid_set: set,
                        policy: str, ds: str, nt: str) -> List[str]:
    """Reproduce verifier_node's output when use_verifier=False.

    Calls the pipeline's own `_base_decode`, so this tracks the noise-adaptive
    decode (global Viterbi on BT/IF, weighted vote on ATF) automatically.
    """
    cands = row.get("candidate_paths") or []
    if not cands:                                   # dirty-fallback cell/sentence
        return legalize_noise_aware(list(dirty), valid_set, policy)
    weights = row.get("rag_weights") or [1.0] * len(cands)
    if len(weights) != len(cands):
        weights = [1.0] * len(cands)
    base = _base_decode(cands, weights, row["tokens"], dirty, ds, nt)
    return legalize_noise_aware(base, valid_set, policy)


def _write_pred(config: str, ds: str, nt: str, seed: int,
                rows_out: List[dict]) -> Path:
    out = _pred_path(config, ds, nt, seed)
    with out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def derive_from_candidates(source_config: str, target_config: str,
                           ds: str, nt: str, seed: int) -> Optional[Path]:
    """Vote-decode the logged candidate pool of `source_config` → `target_config`."""
    rows = _load_pred(source_config, ds, nt, seed)
    if rows is None:
        return None
    dirty = _load_dirty(ds, nt, seed)
    if dirty is None or len(dirty) != len(rows):
        print(f"  ! dirty mismatch {ds}/{nt}/{seed} "
              f"(rows={len(rows)}, dirty={None if dirty is None else len(dirty)}) — skip")
        return None
    valid_set = set(DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"]))
    policy = "demote" if nt == "IF" else "promote"
    # The BT/IF branch of _base_decode needs DEER ω(t); without it it silently
    # falls back to the vote, so make sure the statistics are loaded.
    _init_deer(ds)
    out_rows = [{"tokens": r["tokens"], "gold_tags": r["gold_tags"],
                 "pred_tags": _decode_no_verifier(r, dirty[i], valid_set,
                                                  policy, ds, nt)}
                for i, r in enumerate(rows)]
    return _write_pred(target_config, ds, nt, seed, out_rows)


def copy_lean(source_config: str, target_config: str,
              ds: str, nt: str, seed: int) -> Optional[Path]:
    """Copy pred file keeping only tokens/gold_tags/pred_tags (provably-equal config)."""
    rows = _load_pred(source_config, ds, nt, seed)
    if rows is None:
        return None
    out_rows = [{"tokens": r["tokens"], "gold_tags": r["gold_tags"],
                 "pred_tags": r["pred_tags"]} for r in rows]
    return _write_pred(target_config, ds, nt, seed, out_rows)


def do_full_btif() -> None:
    """Re-decode selectdenoise_full's BT/IF cells in place with the current
    `_base_decode`, preserving candidate_paths/rag_weights/confidence.

    Sound because the Verifier is a provable no-op on BT/IF: it fires only on
    TYPE-contested sentences (`_is_type_contested`), and across all 6000 logged
    BT/IF sentences the stored prediction equals the base decode exactly. So
    re-running the base decode reproduces the full pipeline without API calls.
    """
    print("=== re-decoding selectdenoise_full BT/IF in place "
          "(verifier is a no-op there) ===")
    n = 0
    for ds in DATASETS:
        valid_set = set(DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"]))
        _init_deer(ds)
        for nt in ("BT", "IF"):
            policy = "demote" if nt == "IF" else "promote"
            for seed in SEEDS:
                rows = _load_pred("selectdenoise_full", ds, nt, seed)
                dirty = _load_dirty(ds, nt, seed)
                if rows is None or dirty is None or len(dirty) != len(rows):
                    print(f"  -- missing/misaligned {ds}/{nt}/{seed}")
                    continue
                changed = 0
                for i, r in enumerate(rows):
                    new = _decode_no_verifier(r, dirty[i], valid_set, policy, ds, nt)
                    if new != r["pred_tags"]:
                        changed += 1
                    r["pred_tags"] = new          # keep every other field intact
                out = _write_pred("selectdenoise_full", ds, nt, seed, rows)
                n += 1
                print(f"  wrote {out.name}  ({changed}/{len(rows)} sentences changed)")
    print(f"full BT/IF: rewrote {n} cells")


def do_no_verifier() -> None:
    print("=== deriving selectdenoise_no_verifier (all cells, from full candidates) ===")
    n = 0
    for ds in DATASETS:
        for nt in NOISE_TYPES:
            for seed in SEEDS:
                out = derive_from_candidates(
                    "selectdenoise_full", "selectdenoise_no_verifier", ds, nt, seed)
                if out:
                    n += 1
                    print(f"  wrote {out.name}")
                else:
                    print(f"  -- missing full preds for {ds}/{nt}/{seed}")
    print(f"no_verifier: wrote {n} cells")


def do_no_deanchor_btif() -> None:
    """Copy full BT/IF -> no_deanchor (identical config; ATF cells run live separately)."""
    print("=== copying selectdenoise_full BT/IF -> no_deanchor (config-identical) ===")
    n = 0
    for ds in DATASETS:
        for nt in ("BT", "IF"):
            for seed in SEEDS:
                out = copy_lean("selectdenoise_full", "selectdenoise_no_deanchor",
                                ds, nt, seed)
                if out:
                    n += 1
                    print(f"  wrote {out.name}")
                else:
                    print(f"  -- missing full preds for {ds}/{nt}/{seed}")
    print(f"no_deanchor BT/IF: wrote {n} cells (ATF cells must be run live)")


def do_vote() -> None:
    print("=== deriving selectdenoise_vote ===")
    n = 0
    for ds in DATASETS:
        for nt in NOISE_TYPES:
            for seed in SEEDS:
                if nt in ("BT", "IF"):
                    # vote == no_verifier on BT/IF (de-anchor no-op). Copy it.
                    out = copy_lean("selectdenoise_no_verifier",
                                    "selectdenoise_vote", ds, nt, seed)
                else:  # ATF: vote-decode the no_deanchor candidate pool
                    out = derive_from_candidates(
                        "selectdenoise_no_deanchor", "selectdenoise_vote",
                        ds, nt, seed)
                if out:
                    n += 1
                    print(f"  wrote {out.name}")
                else:
                    print(f"  -- missing source for {ds}/{nt}/{seed} "
                          f"({'no_verifier' if nt!='ATF' else 'no_deanchor'})")
    print(f"vote: wrote {n} cells")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["full_btif", "no_verifier",
                                      "no_deanchor_btif", "vote", "all"])
    args = ap.parse_args(argv)
    # Order matters: full_btif first, since the others derive from full.
    if args.which in ("full_btif", "all"):
        do_full_btif()
    if args.which in ("no_verifier", "all"):
        do_no_verifier()
    if args.which in ("no_deanchor_btif", "all"):
        do_no_deanchor_btif()
    if args.which in ("vote", "all"):
        do_vote()
    return 0


if __name__ == "__main__":
    sys.exit(main())
