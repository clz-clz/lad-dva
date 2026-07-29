"""
metrics.py — NER metrics + significance testing for IOB2 sequences.

Pure functions. No I/O. No API calls.

  - span-level Precision / Recall / F1 (seqeval, strict matching, IOB2 scheme)
  - Syntax Error Rate (SER) under IOB2 grammar
  - Sentence-level paired bootstrap on F1
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from seqeval.metrics import f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
except ImportError as e:                                  # pragma: no cover
    raise ImportError("Install seqeval >= 1.2.2:  pip install seqeval") from e


# ---------------------------------------------------------------------------
# Span-level metrics
# ---------------------------------------------------------------------------

def compute_prf1(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    """Strict span-level Precision / Recall / F1 in IOB2 mode."""
    kw = dict(mode="strict", scheme=IOB2, zero_division=0)
    return {
        "precision": float(precision_score(gold, pred, **kw)),
        "recall":    float(recall_score(gold, pred,    **kw)),
        "f1":        float(f1_score(gold, pred,        **kw)),
    }


def f1_only(gold: List[List[str]], pred: List[List[str]]) -> float:
    return float(f1_score(gold, pred, mode="strict",
                          scheme=IOB2, zero_division=0))


# ---------------------------------------------------------------------------
# IOB2 Syntax Error Rate
# ---------------------------------------------------------------------------

def _is_valid_transition(prev: str, curr: str) -> bool:
    """
    Invalid IOB2 transitions:
        O   -> I-X            (span must start with B-)
        B-X -> I-Y  (X != Y)
        I-X -> I-Y  (X != Y)
    """
    if not curr.startswith("I-"):
        return True
    curr_t = curr[2:]
    if prev == "O":
        return False
    if prev.startswith(("B-", "I-")):
        return prev[2:] == curr_t
    return False


def compute_ser(pred: List[List[str]]) -> float:
    """Fraction of adjacent-tag transitions that violate IOB2."""
    n_trans = n_bad = 0
    for sent in pred:
        if not sent:
            continue
        prev = "O"
        for tag in sent:
            n_trans += 1
            if not _is_valid_transition(prev, tag):
                n_bad += 1
            prev = tag
    return n_bad / n_trans if n_trans else 0.0


# ---------------------------------------------------------------------------
# Token-class rates (Lazy-Erasure diagnostics)
# ---------------------------------------------------------------------------

def token_class_rates(tags: List[List[str]]) -> Dict[str, float]:
    """Token-level O vs entity coverage.

    Substantiates the *Lazy Erasure* claim (predictions collapse to the
    background O class). Computed over a flat tag sequence set:

      - o_rate       : fraction of tokens tagged ``O``
      - entity_rate  : fraction of tokens tagged with a ``B-``/``I-`` entity
                       (equals ``1 - o_rate`` for well-formed IOB2)

    Compare a method's ``o_rate`` against the gold and the dirty-input
    ``o_rate``: a large positive gap over gold is the signature of the
    recall-collapse failure mode.
    """
    n_tok = n_o = 0
    for sent in tags:
        for tag in sent:
            n_tok += 1
            if tag == "O":
                n_o += 1
    if not n_tok:
        return {"o_rate": 0.0, "entity_rate": 0.0}
    o_rate = n_o / n_tok
    return {"o_rate": float(o_rate), "entity_rate": float(1.0 - o_rate)}


# ---------------------------------------------------------------------------
# Paired bootstrap (sentence-level) — fast vectorized implementation
# ---------------------------------------------------------------------------

def _extract_spans_from_iob2(tags: List[str]) -> set:
    """Return set of (type, start, end_exclusive) spans for an IOB2 sequence."""
    spans = set()
    i, n = 0, len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            t = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{t}":
                j += 1
            spans.add((t, i, j))
            i = j
        else:
            i += 1
    return spans


def _per_sentence_span_counts(gold: List[List[str]],
                              pred: List[List[str]]) -> np.ndarray:
    """
    For each sentence, return [tp, fp, fn] under strict span-level matching.
    Shape: (n_sentences, 3), int64.
    """
    out = np.zeros((len(gold), 3), dtype=np.int64)
    for i, (g, p) in enumerate(zip(gold, pred)):
        gs = _extract_spans_from_iob2(g)
        ps = _extract_spans_from_iob2(p)
        tp = len(gs & ps)
        out[i, 0] = tp
        out[i, 1] = len(ps) - tp     # FP
        out[i, 2] = len(gs) - tp     # FN
    return out


def _micro_f1_from_counts(c: np.ndarray) -> float:
    """Compute micro-F1 from a (3,) or (k,3) summed array [tp, fp, fn]."""
    if c.ndim == 1:
        tp, fp, fn = c
    else:
        tp, fp, fn = c.sum(axis=0)
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def paired_bootstrap_f1(
    gold:   List[List[str]],
    pred_A: List[List[str]],
    pred_B: List[List[str]],
    n_iter: int = 10_000,
    seed:   int = 0,
) -> Dict[str, float]:
    """
    One-sided sentence-level paired bootstrap on micro-F1. Tests H1: F1(A) > F1(B).

    Fast vectorized implementation: precomputes per-sentence (TP, FP, FN) span
    counts once, then bootstrap iterations are pure NumPy sums. Equivalent to
    the seqeval-based formulation for strict span-level micro-F1.

    Returns
    -------
    f1_A, f1_B       point estimates on the full set
    delta            f1_A - f1_B
    p_value          P(delta_resampled <= 0)   — small => A is better
    ci_95_low/high   95% CI on delta
    """
    assert len(gold) == len(pred_A) == len(pred_B), \
        f"Sentence counts must match: {len(gold)}, {len(pred_A)}, {len(pred_B)}"
    n = len(gold)
    if n == 0:
        return {"f1_A": 0.0, "f1_B": 0.0, "delta": 0.0,
                "p_value": 1.0, "ci_95_low": 0.0, "ci_95_high": 0.0}

    # Precompute per-sentence TP/FP/FN once for each method
    counts_A = _per_sentence_span_counts(gold, pred_A)   # (n, 3)
    counts_B = _per_sentence_span_counts(gold, pred_B)   # (n, 3)

    obs_A = _micro_f1_from_counts(counts_A)
    obs_B = _micro_f1_from_counts(counts_B)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        # Sum (TP, FP, FN) over the resampled sentences for each method
        sa = counts_A[idx].sum(axis=0)
        sb = counts_B[idx].sum(axis=0)
        tp_a, fp_a, fn_a = sa
        tp_b, fp_b, fn_b = sb
        denom_a = 2 * tp_a + fp_a + fn_a
        denom_b = 2 * tp_b + fp_b + fn_b
        f1_a = (2 * tp_a) / denom_a if denom_a > 0 else 0.0
        f1_b = (2 * tp_b) / denom_b if denom_b > 0 else 0.0
        deltas[i] = f1_a - f1_b

    return {
        "f1_A":       float(obs_A),
        "f1_B":       float(obs_B),
        "delta":      float(obs_A - obs_B),
        "p_value":    float(np.mean(deltas <= 0)),
        "ci_95_low":  float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
    }


def significance_marker(p: float) -> str:
    if p < 0.001: return r"$^{***}$"
    if p < 0.01:  return r"$^{**}$"
    if p < 0.05:  return r"$^{*}$"
    return ""


if __name__ == "__main__":
    gold = [["B-PER", "I-PER", "O", "B-LOC"],
            ["O", "B-ORG", "I-ORG", "O"]]
    perfect = [list(s) for s in gold]
    half = [["B-PER", "O", "O", "B-LOC"],
            ["O", "O", "I-ORG", "O"]]
    print("F1 perfect:", compute_prf1(gold, perfect)["f1"])
    print("F1 half:   ", compute_prf1(gold, half)["f1"])
    print("SER perfect:", compute_ser(perfect))
    print("SER half:  ", compute_ser(half))
    bs = paired_bootstrap_f1(gold, perfect, half, n_iter=2000)
    print("Bootstrap:", bs)
    print("Marker:", repr(significance_marker(bs["p_value"])))