"""
oracle_gap.py — locate the F1 bottleneck: generation vs selection.

Uses the candidate_paths logged by B0 (no API). For each cell reports:
  dirty        : noisy input F1 (lower bound)
  majority     : equal-weight per-position majority vote of candidates
  lad_rg       : the shipped LAD-RG decode (pred_tags on disk)
  oracle_sent  : per-sentence pick the candidate path with best F1  (selection ceiling)
  oracle_tok   : per-token pick the tag from any candidate matching gold (token ceiling)
  best_path    : single globally-best candidate index, averaged

Big oracle_sent - lad_rg gap  => selection is the bottleneck (better decode helps).
Small gap (oracle_sent ~ lad_rg) => generation is the bottleneck (need better Coder).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List

from metrics import f1_only, compute_prf1

PRED = Path("predictions_multiseed")
NOISY = Path("results_multiseed")


def _oracle_sentence(gold: List[str], cands: List[List[str]]) -> List[str]:
    best, bf = gold, -1.0
    for c in cands:
        if len(c) != len(gold):
            continue
        f = f1_only([gold], [c])
        if f > bf:
            bf, best = f, c
    return best


def _oracle_token(gold: List[str], cands: List[List[str]]) -> List[str]:
    """Per position, emit gold tag if ANY candidate got it right, else the
    majority candidate tag. Loose upper bound (may break IOB2, but f1_only is
    span-strict so illegal picks just don't score)."""
    n = len(gold)
    out = []
    for i in range(n):
        picks = [c[i] for c in cands if i < len(c)]
        if gold[i] in picks:
            out.append(gold[i])
        elif picks:
            out.append(max(set(picks), key=picks.count))
        else:
            out.append("O")
    return out


def _majority(cands: List[List[str]]) -> List[str]:
    if not cands:
        return []
    n = min(len(c) for c in cands)
    return [max({c[i] for c in cands}, key=lambda t: sum(1 for c in cands if c[i] == t))
            for i in range(n)]


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _prediction_file(dataset: str, noise: str, seed: int, method: str, tag: str) -> Path:
    tag_suffix = f"__{tag}" if tag else ""
    return PRED / f"pred_seed{seed}__{method}__{dataset}__{noise}{tag_suffix}.jsonl"


def _noisy_file(dataset: str, noise: str, seed: int) -> Path:
    return NOISY / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"


def _sanitize_candidates(row: dict) -> List[List[str]]:
    gold = row.get("gold_tags") or []
    expected_len = len(gold)
    cleaned: List[List[str]] = []
    for path in row.get("candidate_paths") or []:
        if not isinstance(path, list) or not path:
            continue
        clipped = list(path[:expected_len])
        if len(clipped) < expected_len:
            clipped.extend(["O"] * (expected_len - len(clipped)))
        cleaned.append(clipped)
    return cleaned


def _best_single_path(gold: List[List[str]], rows: List[dict]) -> float:
    max_paths = max((len(_sanitize_candidates(row)) for row in rows), default=0)
    if max_paths == 0:
        return 0.0
    pooled: List[List[str]] = []
    best = 0.0
    for idx in range(max_paths):
        pooled.clear()
        for row in rows:
            cands = _sanitize_candidates(row)
            pooled.append(cands[idx] if idx < len(cands) else list(row["pred_tags"]))
        best = max(best, compute_prf1(gold, pooled)["f1"])
    return best


def analyze_cell(*, dataset: str, noise: str, seed: int,
                 method: str, tag: str = "") -> dict:
    pred_file = _prediction_file(dataset, noise, seed, method, tag)
    noisy_file = _noisy_file(dataset, noise, seed)
    if not pred_file.exists() or not noisy_file.exists():
        return {
            "status": "missing",
            "dataset": dataset,
            "noise": noise,
            "seed": seed,
            "method_name": method,
            "message": f"missing files: pred={pred_file.exists()} noisy={noisy_file.exists()}",
        }

    rows = _load_jsonl(pred_file)
    noisy_rows = _load_jsonl(noisy_file)
    n = min(len(rows), len(noisy_rows))
    rows = rows[:n]
    noisy_rows = noisy_rows[:n]
    if not rows:
        return {
            "status": "missing",
            "dataset": dataset,
            "noise": noise,
            "seed": seed,
            "method_name": method,
            "message": "empty prediction or noisy file",
        }

    gold = [row["gold_tags"] for row in rows]
    pred = [row["pred_tags"] for row in rows]
    dirty = [row["dirty_tags"] for row in noisy_rows]
    maj, or_sent, or_tok = [], [], []
    candidate_rows = 0
    for row in rows:
        cands = _sanitize_candidates(row)
        g = row["gold_tags"]
        if cands:
            candidate_rows += 1
            m = _majority(cands)
            m = m[:len(g)] + ["O"] * max(0, len(g) - len(m))
            maj.append(m)
            or_sent.append(_oracle_sentence(g, cands))
            or_tok.append(_oracle_token(g, cands))
        else:
            maj.append(list(row["pred_tags"]))
            or_sent.append(list(row["pred_tags"]))
            or_tok.append(list(row["pred_tags"]))

    score = lambda paths: compute_prf1(gold, paths)["f1"]
    summary = {
        "status": "ok" if candidate_rows else "no_candidate_evidence",
        "dataset": dataset,
        "noise": noise,
        "seed": seed,
        "method_name": method,
        "n_rows": len(rows),
        "dirty": score(dirty),
        "majority": score(maj),
        "method": score(pred),
        "oracle_sent": score(or_sent),
        "oracle_tok": score(or_tok),
        "best_path": _best_single_path(gold, rows) if candidate_rows else score(pred),
    }
    summary["sel_gap"] = summary["oracle_sent"] - summary["method"]
    if candidate_rows == 0:
        summary["message"] = "Prediction file has no candidate_paths evidence for oracle analysis."
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="conll2003")
    ap.add_argument("--noise", nargs="+", default=["BT", "IF", "ATF"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--method", default="selectdenoise_full",
                    help="prediction config whose logged candidate pool to analyse")
    ap.add_argument("--tag", default=os.environ.get("BACKBONE_TAG", "").strip(),
                    help="backbone tag suffix on prediction filenames "
                         "(defaults to $BACKBONE_TAG; noisy files are untagged)")
    args = ap.parse_args(argv)
    tag_suffix = f"__{args.tag}" if args.tag else ""

    print(f"\n{'cell':<18}{'dirty':>8}{'majority':>9}{'method':>8}"
          f"{'oracleS':>9}{'oracleT':>9}{'bestK':>8}{'selGap':>8}")
    print("-" * 68)
    for nt in args.noise:
        summary = analyze_cell(
            dataset=args.dataset,
            noise=nt,
            seed=args.seed,
            method=args.method,
            tag=args.tag,
        )
        if summary["status"] == "missing":
            print(f"{args.dataset}/{nt:<10} [missing] {summary['message']}")
            continue
        if summary["status"] == "no_candidate_evidence":
            print(f"{args.dataset}/{nt:<10} [no-cands] {summary['message']}")
            continue
        print(f"{args.dataset}/{nt:<10}{summary['dirty']:8.4f}{summary['majority']:9.4f}"
              f"{summary['method']:8.4f}{summary['oracle_sent']:9.4f}"
              f"{summary['oracle_tok']:9.4f}{summary['best_path']:8.4f}"
              f"{summary['sel_gap']:+8.4f}")
    print("-" * 68)
    print("selGap = oracleS - method.  Large => selection-bound; ~0 => generation-bound.")


if __name__ == "__main__":
    main()
