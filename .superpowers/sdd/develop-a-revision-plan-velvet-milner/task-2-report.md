# Task 2 Report — A2 P/R/O-rate reporting

## Scope

Implemented the Task 2 brief in `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice` only:

- expose and validate `metrics.token_class_rates(tags)` for empty, all-O, all-entity, and mixed inputs
- aggregate token-class rates alongside strict precision/recall/F1/SER
- preserve existing aggregation JSON keys and `BACKBONE_TAG`-based output naming
- verify the compact LaTeX P/R/F1/SER/O-rate report includes the gold O-rate reference

## RED evidence

Test command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed failure:

```text
FAILED test_aggregate_seeds.py::test_aggregate_cell_tracks_entity_rate_and_gold_o_rate
E       KeyError: 'entity_rate_mean'
```

Interpretation: partial reporting code carried `o_rate` but failed to persist aggregated `entity_rate`.

## GREEN evidence

Re-run command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed result:

```text
......                                                                   [100%]
6 passed in 1.36s
```

## Files changed

- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\aggregate_seeds.py`
  - thread `entity_rate` through per-seed records and aggregate means/stds
  - leave existing `o_rate`, `gold_o_rate`, JSON structure, and `BACKBONE_TAG` filename behavior intact
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\test_aggregate_seeds.py`
  - added real file/data tests for token-class rates and aggregation/reporting behavior

## Tests run

- `D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q`

Coverage in that file:

- `token_class_rates` empty input
- `token_class_rates` all-O input
- `token_class_rates` all-entity input
- `token_class_rates` mixed input
- file-backed `_aggregate_cell(...)` behavior with real JSONL prediction data
- `emit_prf_orate_latex(...)` fields for Precision / Recall / F1 / SER / O-rate plus gold reference

No existing local test file in this worktree covered `aggregate_seeds.py` or `metrics.py`; this new test file is the available changed-file coverage.

## Self-review

- Smallest fix applied: no unrelated pipeline or report refactor
- Existing JSON contract preserved by additive keys only (`entity_rate_mean`, `entity_rate_std`, per-seed `entity_rate`)
- Existing `aggregated.json` vs `aggregated__{BACKBONE_TAG}.json` behavior unchanged
- LaTeX report behavior already satisfied the brief; tests now lock it in

## Concerns

- No functional blockers found for Task 2
- I did not run unrelated full-suite tests because the only available local coverage for the changed files was the new aggregation test file

---

## Review Fix Round 1 — gold O-rate reference independent of method ordering

### Finding

`emit_prf_orate_latex(...)` derived the gold O-rate reference from `methods[0]`. When the first requested method had no cell for a noise type but a later requested method did, the LaTeX gold reference row disappeared even though the gold O-rate was available from existing cells.

### RED evidence

Test command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed failure:

```text
FAILED test_aggregate_seeds.py::test_prf_orate_latex_keeps_gold_reference_with_partial_method_coverage
E       AssertionError
```

The regression test writes a real prediction JSONL file for the second requested method only, requests methods in the order `baseline_self_refine`, `baseline_zero_shot`, and asserts that the concrete LaTeX gold-reference row remains present for `BT`.

### Fix

- kept the existing output contract
- changed gold-reference selection in `aggregate_seeds.py` to derive `gold_o_rate_mean` from any available requested-method cell for the current noise type instead of assuming `methods[0]` has coverage

### GREEN evidence

Re-run command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed result:

```text
.......                                                                  [100%]
7 passed in 1.26s
```

### Files changed in this round

- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\aggregate_seeds.py`
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\test_aggregate_seeds.py`

### Self-review

- regression reproduced with real file-backed aggregation inputs
- fix is localized to gold-reference row selection
- method ordering no longer affects whether the gold row appears

---

## Review Fix Round 2 — deduplicate gold reference by dataset

### Finding

The round-1 `gold_reference()` fix still averaged `gold_o_rate_mean` across every matching method/dataset cell for a noise type. Under uneven coverage, that over-counted datasets covered by more than one requested method. Example: dataset coverage `A,B` from one method plus `A` from another incorrectly averaged `A,B,A` instead of `A,B`.

### RED evidence

Test command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed failure:

```text
FAILED test_aggregate_seeds.py::test_prf_orate_latex_keeps_gold_reference_with_partial_method_coverage
E       AssertionError
```

The extended regression test writes real prediction JSONL files for two datasets with uneven method coverage:

- `baseline_zero_shot`: `msra`, `conll2003`
- `baseline_self_refine`: `msra` only

The correct deduplicated gold reference is `(0.500 + 0.250) / 2 = 0.375`; the buggy implementation over-counted `msra` and failed that assertion.

### Fix

- kept the existing output contract
- changed `gold_reference()` to scan datasets first and select one available `gold_o_rate_mean` per dataset for the noise type, using a stable method choice independent of the requested method order

### GREEN evidence

Re-run command:

```powershell
D:/py/Anaconda3/python.exe -m pytest C:/Users/LinzeChen/AI_Workspace/.worktrees/selectdenoise-contextual-lattice/test_aggregate_seeds.py -q
```

Observed result:

```text
.......                                                                  [100%]
7 passed in 1.32s
```

### Files changed in this round

- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\aggregate_seeds.py`
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\test_aggregate_seeds.py`

### Self-review

- regression remains real file-backed
- gold reference now uses each covered dataset once per noise type
- requested method ordering no longer affects dataset inclusion or weighting
