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

## Review fix round 1

### RED evidence

The inherited partial test edit was preserved and audited. Its first run was
not sufficient product RED evidence because it stopped in the test harness:

```text
........F........                                                        [100%]
1 failed, 16 passed in 4.00s
```

`PaddingTokenizer` had been changed to require `padding_side`, but its existing
call site had not been updated. After correcting that fixture and completing
the behavior-level tests, the unchanged production code failed in the intended
places (exit 1):

```text
............F..FFF.FFFFF....FFFF                                         [100%]
13 failed, 19 passed in 4.02s
```

Those failures reproduced mixed 7/9-code batching at the MSRA boundary,
missing CIs for two logit metrics, absent aggregate commit metadata, staging
leaks, incomplete `BaseException` rollback, cleanup failures escaping after a
valid install, and raw CLI validation exceptions. A later transaction-audit
test separately proved that a pre-commit snapshot failure leaked both staged
files before its fix (exit 1: `1 failed, 32 deselected in 2.45s`).

### GREEN evidence

The completed focused suite now passes (exit 0):

```text
...................................                                      [100%]
35 passed in 3.72s
```

The full synthetic run uses the canonical 45-file by 200-row matrix and the
old batch size of eight that crosses from MSRA into CONLL-2003. It proves that
inference batches stay within one dataset/exact codebook, while the 90 output
records retain canonical dataset/file/group order. Separate left- and
right-padding cases use different contextual token-ID maps per row and assert
the exact gathered logits.

The 10,000-iteration bootstrap test compares all three CIs against a direct
NumPy reference. One derived PCG64 stream supplies each group's resample index
matrix, and each chunk reuses those indices for delta-z, O-logit, and strongest
entity-logit means.

### Transaction rationale

The aggregate JSON is now the commit marker for the exact canonical JSONL. It
records protocol/version, a transaction ID, canonical JSONL name, SHA-256,
byte count, line count, and record count; `validate_output_pair()` verifies the
marker and parses every JSONL record.

Both artifacts are staged in the destination directory before canonical paths
change. The old aggregate marker is moved aside first, followed by the old
JSONL; the new JSONL is installed next and the new aggregate marker last. Thus
a process crash after the old marker is removed but before the new marker is
installed cannot present an uncommitted JSONL as a committed pair. Handled
`BaseException` paths restore the exact prior JSONL and restore its aggregate
marker last. If rollback itself fails, the canonical aggregate remains absent,
recoverable backups/stages are retained, and `OutputTransactionError` reports
both the primary and rollback failures. Cleanup after a validated commit is
best effort, so a cleanup error cannot turn a valid pair into a reported
failure. Fault-injection coverage also verifies unrelated files are untouched.

## Review fix round 2

### RED evidence

With valid canonical inputs and GPU confirmation, injected loader failures
reproduced the uncovered CLI boundary (exit 1):

```text
FF.                                                                      [100%]
2 failed, 1 passed, 35 deselected in 4.00s
```

`ImportError` and `MemoryError` escaped through `main()` rather than producing
argparse diagnostics. The passing case proved that an injected
`AssertionError` already remained visible as a programming error.

### GREEN evidence

After extending only the expected CLI exception tuple, the targeted cases
passed (exit 0):

```text
...                                                                      [100%]
3 passed, 35 deselected in 3.86s
```

The full focused suite then passed (exit 0):

```text
......................................                                   [100%]
38 passed in 5.20s
```

`main()` now converts explicit `ImportError`/`ModuleNotFoundError` and
`MemoryError` model/dependency failures to concise nonzero argparse-style
diagnostics without catching `BaseException` or hiding `AssertionError`.
