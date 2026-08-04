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
          f"{'oracleS':>9}{'oracleT':>9}{'selGap':>8}")
    print("-" * 68)
    for nt in args.noise:
        pf = (PRED / f"pred_seed{args.seed}__{args.method}__{args.dataset}"
                     f"__{nt}{tag_suffix}.jsonl")
        nf = NOISY / f"noisy_seed{args.seed}__{nt}__{args.dataset}__N200.jsonl"
        if not pf.exists() or not nf.exists():
            print(f"{args.dataset}/{nt:<10} [missing]"); continue
        rows = [json.loads(l) for l in pf.open(encoding="utf-8") if l.strip()]
        noisy = [json.loads(l) for l in nf.open(encoding="utf-8") if l.strip()]
        gold = [r["gold_tags"] for r in rows]
        pred = [r["pred_tags"] for r in rows]
        dirty = [noisy[i]["dirty_tags"] for i in range(len(rows))]
        maj, orS, orT = [], [], []
        for r in rows:
            c = r.get("candidate_paths") or []
            g = r["gold_tags"]
            if c:
                m = _majority(c)
                m = m[:len(g)] + ["O"] * max(0, len(g) - len(m))
                maj.append(m)
                orS.append(_oracle_sentence(g, c))
                orT.append(_oracle_token(g, c))
            else:
                maj.append(r["pred_tags"]); orS.append(r["pred_tags"]); orT.append(r["pred_tags"])
        f = lambda P: compute_prf1(gold, P)["f1"]
        d, mj, lr, os_, ot = f(dirty), f(maj), f(pred), f(orS), f(orT)
        print(f"{args.dataset}/{nt:<10}{d:8.4f}{mj:9.4f}{lr:8.4f}"
              f"{os_:9.4f}{ot:9.4f}{os_-lr:+8.4f}")
    print("-" * 68)
    print("selGap = oracleS - method.  Large => selection-bound; ~0 => generation-bound.")


if __name__ == "__main__":
    main()
