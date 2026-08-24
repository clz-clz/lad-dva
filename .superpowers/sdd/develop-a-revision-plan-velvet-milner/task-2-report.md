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
