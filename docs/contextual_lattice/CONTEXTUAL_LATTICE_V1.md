# Anchor-Aware Contextual Lattice v1

Status: locked development candidate (2026-08-21).  The terminal replacement is
the documented default for new cached/future runs on
`codex/selectdenoise-contextual-lattice`.  It is not a formal 13/15-cell winner:
the locked 450-row gate has 12 wins, 1 tie, and 2 losses.  No superiority claim
is made outside those 450 development rows.

## Purpose and scope

The unchanged Coder/Reviewer graph produces the dirty prediction, five cached
Coder paths, and Reviewer weights.  The terminal adapter replaces only the
final commitment.  It retains the existing SelectDenoise output as an anchor,
constructs a typed span lattice from dirty/anchor/Coder paths, scores the legal
span sets with a frozen contextual model, and uses a cross-fitted residual gate
to choose the lattice result or the anchor.  `selectdenoise_full@e1ae1d8` remains
available for historical comparison and rollback.

There are no new provider calls, labels, prompts, or evaluation splits in v1.
The host graph may retain its existing dataset/noise routing, but none of that
metadata is passed to the terminal model.

## Model contract

The terminal accepts the following allow-listed fields only:

```python
ContextualLatticeInput(
    tokens: tuple[str, ...],
    dirty_tags: tuple[str, ...],
    anchor_tags: tuple[str, ...],
    candidate_paths: tuple[tuple[str, ...], ...],
    reviewer_weights: tuple[float, ...],
    valid_types: frozenset[str],
    deer_stats: DEERStatistics,
)
```

The runtime `decode` signature mirrors this list.  Dataset, family, cell, seed,
example identifiers, audit fields, and `__noise_type__` are forbidden model
metadata.  Unknown fields are rejected before deserialization; they are not
serialized, logged, or used as features.

All token/tag sequences must have equal length.  The anchor must be a legal IOB2
sequence in the supplied ontology; an invalid anchor hard-fails bundle use.
Dirty tags are alphabet-checked but may contain a noisy transition.  Candidate
paths with the wrong length or unknown tag type are excluded.  Missing Reviewer
weights are represented by an empty sequence and become uniform support with a
missing-weight feature.  An explicit weight/path length mismatch is rejected.
If terminal decoding fails for a sentence, the adapter returns the anchor,
preserves length and IOB2 legality, increments a fallback counter, and continues
the run.  Bundle, checkpoint, schema, split, margin, and file-hash failures are
startup errors rather than silent substitutions.

## Feature and model procedure

The encoder is the cached GLiNER mDeBERTa encoder, loaded offline and frozen.
First-subtoken pooling is used; sentences longer than the encoder limit use
320-token windows with stride 256 and averaged overlap embeddings.  Each typed
candidate span receives six contextual pools (start, end, mean, max, left, and
right), a 16-dimensional type embedding, and scalar evidence including exact
and overlap support, Reviewer-weight mass, dirty/anchor presence, span geometry,
DEER values, OOV rate, and edit-shape features.  Contextual features are
projected to 256 dimensions and scalar features to 64 dimensions, then scored by
a `336 -> 256 -> 1` GELU MLP with dropout 0.2.

Training uses semi-Markov conditional likelihood over all legal non-overlapping
candidate-span sets.  The target is the reachable exact gold-span set; gold
spans missing from the lattice are recorded as unreachable.  Exact interval
dynamic programming decodes the complete lattice: there is no k-best truncation
and no post-hoc IOB2 repair.

The frozen training settings are seed 731, AdamW, learning rate `1e-3`, weight
decay `1e-4`, 20 epochs, batch size 16 sentence lattices, and gradient clipping
at 1.0.  Encoder embeddings are float32 and keyed by checkpoint hash plus
canonical tokens.  DEER statistics are built from training groups only.

## Leakage-safe data protocol

Rows are grouped by `dataset + canonical(tokens)` across all seeds and families;
all views and seed replicates of a sentence stay together for splitting.  Groups
are salted with `selectdenoise-contextual-lattice-v1-20260820`.  Per dataset,
the first 30 hashed groups are test, the next 30 calibration, and the remainder
training.  Calibration and test use one seed replicate per group (selected by a
second salted hash modulo `[13, 42, 2024]`) and use that same seed for all three
families.  The resulting test gate is exactly 30 examples in each of 15 cells:
450 rows total.  The original protocol contains 9,000 source rows: 6,201 train,
450 calibration, and 450 test rows after the view expansion.  Training weights
give each sentence group total weight one.  Fold-local DEER is computed from the
training groups only.

Predictions and their SHA-256 are written before evaluator metadata is joined.
Existing result directories are never overwritten.  The two replay passes used
the same seed, split, checkpoint, model settings, and margin and produced the
same prediction hash.

## Residual gate

Five-fold sentence-grouped OOF lattice outputs on the training partition train a
fixed `HistGradientBoostingRegressor` for
`(FP+FN)_anchor - (FP+FN)_lattice`.  Its settings are 200 iterations, learning
rate 0.05, 15 leaf nodes, minimum leaf size 20, L2 regularization 1.0, and seed
731.  Features include the lattice/anchor score gap, changed-span counts,
support extrema, predicted span probabilities, sentence length, and entity
density.

The single global margin grid is
`[-0.50, -0.25, -0.10, 0, 0.10, 0.25, 0.50, 0.75, 1.00]`.  Calibration chooses
the highest pooled F1 margin subject to at least 5% alternative coverage and no
sentence-level F1 regression on any retention-null row.  Ties choose the larger
margin.  The frozen choice is `-0.1`.

## Metric definitions

- Strict precision, recall, and F1 are entity-level IOB2 exact-span metrics.
- SER is the fraction of output sequences containing an invalid IOB2 transition.
- Invalid length is a prediction whose tag count differs from its token count.
- An accepted alternative is a sentence on which the residual gate selects the
  lattice hypothesis rather than the anchor.
- Clean rows are rows where the observed dirty tags equal gold.  Clean-change
  rate counts any output difference; clean regression counts strict-F1 losses.
- Bootstrap intervals are paired, stratified, 10,000-resample sentence-cluster
  intervals over the 150 dataset/sentence groups, with the reported two-sided
  95% interval and one-sided lower bound.
- Anchor-inclusive residual lattice-oracle recovery is the achieved pooled F1
  gain divided by the reachable lattice-oracle gain above the anchor.

## Locked calibration and test evidence

The complete machine-readable record is
[`metrics_450.json`](metrics_450.json).  Calibration pooled F1 is
`0.873385013` versus anchor `0.830957230` (Δ `+0.042427783`), with 421/450
accepted alternatives.  There were 105 clean rows and 0 changes or regressions.

The test pooled F1 is `0.848049281` versus anchor `0.787064677` (Δ
`+0.060984605`).  The paired bootstrap mean Δ is `+0.060776609`, with 95% lower
bound `+0.039458164` and upper bound `+0.082582791`.  Cell wins/ties/losses are
`12/1/2`; BT is 5/5, IF is 4/5, and ATF is 3/5.  The worst cell is OntoNotes5
ATF at Δ `−0.007337808`.  Oracle-gap recovery is `86.4856%`.  The gate accepted
422/450 alternatives.  Of 69 clean rows, changes/regressions were 0/0.  SER
and invalid lengths were both zero.  The locked evidence records runtime
`1493.374335` seconds, zero API calls, and `$0` external cost.
The serialized-runtime replay reported zero terminal fallbacks.
An intermediate pre-fix replay is preserved at
`contextual_lattice_run_runtime_reload_20260821`; it differed on two rows
because malformed-path counts were removed before feature construction and is
not part of the locked evidence.  The corrected adapter was replayed twice
against the same 450 rows and produced the locked hash on both passes.

### Full 15-cell strict metrics

Values below are generated from the locked report; `M` is the contextual-lattice
terminal and `A` is the SelectDenoise anchor.

| Dataset | Family | M-P | M-R | M-F1 | A-P | A-R | A-F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conll2003 | BT | 0.975610 | 0.666667 | 0.792079 | 0.928571 | 0.650000 | 0.764706 | +0.027373 |
| conll2003 | IF | 1.000000 | 0.666667 | 0.800000 | 1.000000 | 0.666667 | 0.800000 | +0.000000 |
| conll2003 | ATF | 0.900000 | 0.900000 | 0.900000 | 0.866667 | 0.866667 | 0.866667 | +0.033333 |
| fewnerd | BT | 0.982759 | 0.760000 | 0.857143 | 0.901639 | 0.733333 | 0.808824 | +0.048319 |
| fewnerd | IF | 0.964912 | 0.733333 | 0.833333 | 0.854839 | 0.706667 | 0.773723 | +0.059611 |
| fewnerd | ATF | 0.863014 | 0.840000 | 0.851351 | 0.853333 | 0.853333 | 0.853333 | −0.001982 |
| msra | BT | 0.980769 | 0.871795 | 0.923077 | 0.927273 | 0.871795 | 0.898678 | +0.024399 |
| msra | IF | 0.979167 | 0.803419 | 0.882629 | 0.692308 | 0.692308 | 0.692308 | +0.190321 |
| msra | ATF | 0.940171 | 0.940171 | 0.940171 | 0.876106 | 0.846154 | 0.860870 | +0.079301 |
| ontonotes5 | BT | 0.980769 | 0.671053 | 0.796875 | 0.838710 | 0.684211 | 0.753623 | +0.043252 |
| ontonotes5 | IF | 0.946429 | 0.697368 | 0.803030 | 0.714286 | 0.657895 | 0.684932 | +0.118099 |
| ontonotes5 | ATF | 0.917808 | 0.881579 | 0.899329 | 0.918919 | 0.894737 | 0.906667 | −0.007338 |
| wnut17 | BT | 1.000000 | 0.464286 | 0.634146 | 0.733333 | 0.392857 | 0.511628 | +0.122518 |
| wnut17 | IF | 0.833333 | 0.357143 | 0.500000 | 0.692308 | 0.321429 | 0.439024 | +0.060976 |
| wnut17 | ATF | 0.607143 | 0.607143 | 0.607143 | 0.571429 | 0.571429 | 0.571429 | +0.035714 |

The locked prediction and repeat hashes are both
`54b54b1cd37e24de8fd04d1e43b67cd674e58ef95a6648d071c819f63e763be1`.

## Historical comparison

`selectdenoise_full@e1ae1d8` had historical macro F1 `0.782120`, SER 0, and
complete 5-dataset × 3-family × 3-seed coverage.  That comparison is unpaired:
the sentences, seeds, prompts, candidate paths, Reviewer weights, and verifier
outputs differ.  The 450-row same-fold anchor comparison above is therefore the
authoritative evidence for this locked artifact.

## Rollback and provenance

The replay bundle is under
`unified_experiment_20260816/contextual_lattice_run_lock_bundle_20260821/bundle`.
Its manifest records the checkpoint hash
`2100142f31627531497850659dcb3821c99d5e71c08a8e01a98e4b11ef32a199`, decoder
hash `2b45770a128cc5a716d3c7dbdc75ab059afc1a1b9ed471ea5cf1bd4f2a150e4e`, split
hash `fe4b00d32e655a334e8451e57a49f8d505047b313308bb5d60df0084865d421a`, and
bundle manifest hash
`b24d4d5485e92e5513db246c48ed8d5258bd70b69c31889ae7d393186bda7825`.
Bundle file hashes are retained in the machine-readable frozen manifest.
The runtime also pins the checkpoint, split, manifest, decoder-file, gate-file,
and decoder-state hashes in code before loading the joblib gate; regenerating an
in-bundle hash inventory cannot make a different bundle acceptable.

To use the new terminal, set `CONTEXTUAL_LATTICE_BUNDLE` to the bundle directory
and run configuration `selectdenoise_contextual_lattice`.  To roll back without
changing old prediction files, run `selectdenoise_full_legacy`; it uses the
unchanged SelectDenoise graph and does not load the contextual bundle.

The preserved historical worktree is `selectdenoise-e1ae1d8`; the dirty
`codex/unified-risk-calibrated-snapshot` worktree and all prior contextual-lattice
attempt directories remain untouched.  Source, dependency, split, command-line,
and artifact hashes are recorded in [`frozen_manifest.json`](frozen_manifest.json).
