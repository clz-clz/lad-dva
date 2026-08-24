# Task 5 Report — B3/B4/B5 analysis and run registration

## Scope

Implemented Task 5 in `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice` only. No manuscript files were touched. No live API/GPU runs were executed or claimed.

## Requirements addressed

- Registered explicit LAD-RG method keys for:
  - `lad_rg_full`
  - leave-one-out: `lad_rg_no_lads`, `lad_rg_no_ror`, `lad_rg_no_gasd`
  - pairwise: `lad_rg_lads_ror`, `lad_rg_lads_gasd`, `lad_rg_ror_gasd`
  - diagnostic: `lad_rg_no_potentials`, `lad_rg_ror_ungated`
  - pairwise floor case: `lad_rg_coder_only`
- Preserved prior SelectDenoise/contextual-lattice/legacy names.
- Kept prediction files disjoint across backbones via existing `BACKBONE_TAG` pathing.
- Made ablation flags reach runtime state and affect behavior:
  - `use_lads=False` now disables reviewer weighting and LADS omega use
  - `use_gasd=False` now bypasses GASD decode and legalizes the base path
  - `use_ror=False` remains active as the RoR bypass
- Extended candidate-evidence persistence to all LAD-RG variants so offline oracle analysis has the needed inputs.
- Added deterministic offline oracle analysis helpers in `oracle_gap.py`, robust to:
  - missing prediction/noisy files
  - missing `candidate_paths`
  - partial/oversized candidate paths
- Registered structural-floor method keys as explicit offline-only configs that fail clearly with the required offline derivation message instead of acting like runnable live configs.

## RED

Added tests first, then ran:

```powershell
D:/py/Anaconda3/python.exe -m pytest test_lad_rg_graph.py test_backbone_evidence.py test_oracle_gap.py
```

Observed failures before implementation:

- missing config keys: `lad_rg_no_lads`, `lad_rg_no_gasd`, `lad_rg_ror_gasd`
- no runtime support for `use_lads` / `use_gasd`
- `oracle_gap` lacked an offline analysis API (`analyze_cell`)

The initial RED run ended with:

- `6 failed, 12 passed, 1 error`

I corrected one test seam issue (`llm.invoke` monkeypatch on a pydantic client object) without changing the intended behavior under test, then implemented production code.

## GREEN

Re-ran the focused Task 5 suite:

```powershell
D:/py/Anaconda3/python.exe -m pytest test_lad_rg_graph.py test_backbone_evidence.py test_oracle_gap.py
```

Result:

- `18 passed in 9.24s`

## Additional verification

Ran the full local pytest suite:

```powershell
D:/py/Anaconda3/python.exe -m pytest -x
```

Result:

- stopped at `test_contextual_lattice_runtime.py::test_documented_metrics_and_manifest_have_locked_inventory`
- status at stop: `1 failed, 41 passed`

Cause:

- `docs/contextual_lattice/frozen_manifest.json` still pins old hashes for:
  - `multi_agent_v2.py`
  - `run_multiseed.py`

Observed mismatches:

- `multi_agent_v2.py`
  - manifest: `f0e4b9d0b05d8d9fe2c0f675c3d5c9a1dad07c9dabe2d30f056ae6b045fc978f`
  - current: `266dba79c75cac7ac33b006f25369179bff383c7b782e36adcd6f18d5e738ade`
- `run_multiseed.py`
  - manifest: `6bea2a6901538cc3c20b978dfaf40dd7addf7f2e86d6840a3a429bac43618e80`
  - current: `48b28ecfe69741818a0ee68ec2284cd3cf44079eb6155c40646ab192e7e8750a`

I did not update the frozen contextual-lattice inventory in this task because Task 5 was scoped to B3/B4/B5 run registration and offline analysis, not to re-baselining the locked contextual-lattice documentation artifacts.

## Files changed

- `run_multiseed.py`
- `multi_agent_v2.py`
- `oracle_gap.py`
- `aggregate_seeds.py`
- `test_lad_rg_graph.py`
- `test_backbone_evidence.py`
- `test_oracle_gap.py`

## Behavior summary

- Runner now exposes the approved LAD-RG ablation surface directly.
- Offline-only structural-floor methods are explicitly named and fail with clear guidance instead of silently behaving like live run configs.
- Candidate evidence is persisted for all LAD-RG variants.
- `oracle_gap.py` now provides deterministic offline oracle-vs-selected analysis and handles missing/partial evidence defensively.

## Self-review

- Kept the change minimal: no manuscript edits, no live-run scaffolding, no unrelated refactors.
- Reused existing path naming and `BACKBONE_TAG` behavior instead of inventing a second namespacing scheme.
- Avoided dead labels by wiring new config flags into `reviewer_node`, `ror_node`, `gasd_node`, and runner state.
- Left one known branch-level blocker documented: the frozen contextual-lattice manifest needs an explicit refresh if branch policy requires the full suite to pass after source changes.

## Blockers / follow-ups

- Full-suite green is blocked by the locked contextual-lattice manifest hash inventory, not by the new Task 5 logic.
- No live API/GPU experiment results were run; only deterministic/offline tests were executed in this task.

## Fix Round 1/5

### Review findings addressed

1. Added explicit offline oracle-pool registry keys and labels:
   - `oracle_pool_sent`
   - `oracle_pool_tok`
2. Added an early empty-candidate fallback in `ror_node` so an empty pool no longer reaches `_weighted_majority_voting`.
3. Made built-in dummy mode return deterministic candidate evidence when `__return_candidates__` is requested, so LAD-RG dummy runs persist oracle evidence through `_run_one_cell`.

### RED

Command:

```powershell
D:/py/Anaconda3/python.exe -m pytest test_lad_rg_graph.py test_backbone_evidence.py -k "oracle_pool or dummy_mode or empty_candidate or offline_oracle"
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.9.7, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: C:\Users\LinzeChen\AI_Workspace, configfile: pytest.ini
plugins: anyio-4.12.1, langsmith-0.4.37
collected 19 items / 16 deselected / 3 selected

test_lad_rg_graph.py F                                                   [ 33%]
test_backbone_evidence.py FF                                             [100%]

================================== FAILURES ===================================
KeyError: 'oracle_pool_sent'
AssertionError
KeyError: 'oracle_pool_sent'

====================== 3 failed, 16 deselected in 9.69s =======================
```

Interpretation:

- offline oracle config keys were absent
- offline oracle labels were absent
- dummy-mode candidate evidence was not being persisted

### GREEN

Targeted verification:

```powershell
D:/py/Anaconda3/python.exe -m pytest test_lad_rg_graph.py test_backbone_evidence.py -k "oracle_pool or dummy_mode or empty_candidate or offline_oracle or ror_returns_legal"
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.9.7, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: C:\Users\LinzeChen\AI_Workspace, configfile: pytest.ini
plugins: anyio-4.12.1, langsmith-0.4.37
collected 19 items / 15 deselected / 4 selected

test_lad_rg_graph.py ..                                                  [ 50%]
test_backbone_evidence.py ..                                             [100%]

====================== 4 passed, 15 deselected in 9.40s =======================
```

Focused Task 5 suite:

```powershell
D:/py/Anaconda3/python.exe -m pytest test_lad_rg_graph.py test_backbone_evidence.py test_oracle_gap.py
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.9.7, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: C:\Users\LinzeChen\AI_Workspace, configfile: pytest.ini
plugins: anyio-4.12.1, langsmith-0.4.37
collected 22 items

test_lad_rg_graph.py ............                                        [ 54%]
test_backbone_evidence.py .......                                        [ 86%]
test_oracle_gap.py ...                                                   [100%]

============================= 22 passed in 9.73s ==============================
```

### Files changed in fix round

- `run_multiseed.py`
- `multi_agent_v2.py`
- `aggregate_seeds.py`
- `test_lad_rg_graph.py`
- `test_backbone_evidence.py`
