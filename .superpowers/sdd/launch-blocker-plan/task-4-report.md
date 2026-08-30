# Task 4 report — reproducible Qwen logit-gap probe

## Scope and files

- Added `logit_gap_probe.py` for the pinned `Qwen/Qwen3-32B-AWQ` raw
  next-token probe. Model and tokenizer loading are lazy and injectable.
- Added `test_logit_gap_probe.py` with offline fake-tokenizer/fake-model
  coverage.
- Added this report. `launch-blocker-plan.md` remains untracked, unmodified,
  and must not be committed.

## RED evidence

Command:

```text
D:/py/Anaconda3/python.exe -m pytest -q test_logit_gap_probe.py
```

Observed before adding production code (exit 1):

```text
E   ModuleNotFoundError: No module named 'logit_gap_probe'
ERROR test_logit_gap_probe.py
1 error in 2.78s
```

This was the expected failure: the focused test contract existed and the new
module did not.

## GREEN evidence

Focused command after implementation (exit 0):

```text
.................                                                        [100%]
17 passed in 2.88s
```

Focused plus related official runner/preflight regression subset (exit 0):

```text
.....................................................                    [100%]
53 passed in 16.11s
```

Additional checks at this point:

```text
D:/py/Anaconda3/python.exe -m py_compile logit_gap_probe.py  # exit 0
git diff --check                                             # exit 0
```

The pre-change related baseline was also clean: 36 tests passed in 16.02s.

## Self-review

- Identity is fail-closed: runtime requires the exact model constant, a
  40-hex immutable revision, and a nonempty filename-safe tag.
- Normal import does not import PyTorch or Transformers, construct a model,
  access Hugging Face, touch a GPU, or read inputs. CLI identity validation is
  tested to occur before an injected model loader can run.
- The codebook is deterministic (`0=O`, then B/I in the exact project ontology
  order). Each prompt is rendered by the Qwen chat template with
  `enable_thinking=False`; every code is checked against that exact rendered
  context for prefix stability, one-token continuation length, integer token
  ID, and token-ID uniqueness.
- Source loading accepts exactly the expected matrix by default (5 datasets x
  3 noises x 3 seeds), exactly 200 rows per file, equal nonempty string-list
  lengths, in-ontology gold/dirty tags, and legal gold IOB2. It reads no
  predictions and completes validation before allocating the model.
- Every dirty/gold mismatch is included. Controls are equal-size, unchanged
  gold-O positions from the same file, selected without replacement by a
  SHA-256 rank derived from a fixed seed and stable position identity.
- Batched inference defaults to 8, runs under `torch.inference_mode()`, finds
  each row's actual final non-padding token for left or right padding, and
  gathers only the exact contextual code token IDs.
- Per-position rows include all requested source coordinates/tags, group,
  logits, strongest entity label, delta-z, exact model/revision, both
  fingerprints, and auditable codebooks/token IDs.
- Aggregation groups by dataset/noise/group, records counts and all three
  requested means, and computes a deterministic fixed-seed percentile 95% CI
  from the requested default 10,000 bootstrap resamples.
- Both final artifacts are fully serialized and staged before replacement;
  installation preserves existing outputs for rollback on failure. No output
  is written until all source validation, inference, scoring, and aggregation
  have completed.

## Concerns / deferred live evidence

- Per task constraints, no real Hugging Face access, model download, CUDA model
  load, vLLM interaction, API call, or experiment execution was performed.
  The exact contextual digit tokenization and 32B-AWQ loading therefore remain
  intentionally fail-closed runtime checks for the rented GPU.
- Direct CLI execution additionally requires `--confirm-vllm-stopped`; this is
  a safety acknowledgement and does not attempt to inspect or stop vLLM.
