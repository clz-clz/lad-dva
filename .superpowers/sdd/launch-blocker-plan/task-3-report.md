# Task 3 Report: Official LAD-RG Launch Gate

## Status

Implemented and verified offline.

## Scope and constraints

- Worktree: `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice`
- Baseline HEAD: `79a68b9d1da614f5a8549a75826a33ff0b6cdfaa`
- Interpreter: `D:/py/Anaconda3/python.exe`
- Offline/mock-only testing; no provider, Hugging Face, GPU, prediction regeneration, manuscript, or data operations.
- Frozen contextual-lattice hashes are out of scope and will not be updated.

## Initial audit

Although the task handoff said there were no Task 3 changes, the first `git status --short` showed a modified `run_multiseed.py` and untracked `launch-blocker-plan.md`, `test_official_preflight.py`, and `test_run_multiseed_official.py`. These files were preserved and treated as partial prior-attempt work. No reset or discard was performed.

`rg --files` could not start because the packaged `rg.exe` returned `Access denied`; PowerShell `Get-ChildItem` and `Select-String` were used as the safe fallback.

## TDD evidence

### RED

1. Initial preflight seam:

   `D:/py/Anaconda3/python.exe -m pytest -q test_official_preflight.py test_run_multiseed_official.py`

   Exit 1 during collection: `ModuleNotFoundError: No module named 'official_preflight'`.

2. Existing runner seam:

   `D:/py/Anaconda3/python.exe -m pytest -q test_run_multiseed_official.py`

   Exit 1: `3 failed, 2 passed in 0.52s`. Failures were the missing `_parse_args` API and missing `official`/failure-control parameters on `_run_one_cell`.

3. `/v1/models` and disabled-GASD semantics:

   `D:/py/Anaconda3/python.exe -m pytest -q test_official_preflight.py::test_models_url_always_targets_openai_v1_models_endpoint test_run_multiseed_official.py::test_official_gasd_disabled_cell_accepts_disabled_variant_evidence`

   Exit 1: `2 failed in 0.53s`. The URL was `/models` rather than `/v1/models`, and the runner rejected valid `gasd_variant_used="disabled"` evidence.

4. Optional fingerprint and JSON-only CLI failure handling:

   `D:/py/Anaconda3/python.exe -m pytest -q test_official_preflight.py::test_live_preflight_records_unsupported_fingerprint_without_blocking test_official_preflight.py::test_static_cli_converts_internal_failure_to_json_only`

   Exit 1: `2 failed in 0.45s`. Missing fingerprints incorrectly blocked vLLM, and `--request-timeout`/internal JSON error conversion were absent.

5. Intended-config validation in static preflight:

   `D:/py/Anaconda3/python.exe -m pytest -q test_official_preflight.py::test_static_preflight_validates_the_intended_official_configs`

   Exit 1: `1 failed in 1.11s` because `config_names` was not yet exposed.

### GREEN

- Initial runner/preflight implementation: `9 passed in 3.60s`.
- `/v1/models` and disabled GASD fixes plus full focused files: `13 passed in 2.58s` (two selectors were intentionally repeated in that command).
- Fingerprint/JSON-only fixes plus full focused files: `15 passed in 2.82s` (two selectors repeated).
- Intended-config fix plus focused files: `15 passed in 5.70s` (one selector repeated); the two files contain 14 distinct tests.

## Files changed

- `run_multiseed.py`: official CLI controls, explicit environment/model validation, canonical manifest creation/compatibility, one adapter per cell, fail-closed evidence validation, configurable timeout/failure policy, and exact temp cleanup.
- `official_preflight.py`: read-only static/live preflight APIs and CLI, configurable noisy/prediction roots, canonical data validation, bounded isolated DEER subprocess checks, tagged artifact checks, `/v1/models`, RoR schemas, Qwen GASD-R, and provider evidence.
- `test_run_multiseed_official.py`: offline runner/manifest/adapter/failure/ablation tests.
- `test_official_preflight.py`: offline static/live/CLI tests using temporary repositories and mocked transports.
- `.superpowers/sdd/launch-blocker-plan/task-3-report.md`: this report (force-added because `.superpowers/sdd/.gitignore` ignores SDD reports).

## Verification commands and output

- `D:/py/Anaconda3/python.exe -m pytest -q test_official_preflight.py test_run_multiseed_official.py test_live_backbone.py test_lad_rg_graph.py test_backbone_evidence.py`
  - Exit 0: `84 passed in 13.61s`.
- `D:/py/Anaconda3/python.exe -m py_compile run_multiseed.py official_preflight.py live_backbone.py multi_agent_v2.py`
  - Exit 0, no output.
- `D:/py/Anaconda3/python.exe -c "import json, run_multiseed as r; ..."`
  - Exit 0: `{"official": ["abort", 600.0], "legacy": ["abort", 180.0]}`.
- `D:/py/Anaconda3/python.exe official_preflight.py static`
  - Exit 1 with JSON-only stdout: `{"blockers": [{"code": "missing_expected_sha", ...}], "checks": {}, "mode": "static", "ok": false}`.
- `git diff --check`
  - Exit 0; only the existing Git line-ending warning was emitted.

Fresh final verification before staging:

- Related focused pytest command above: exit 0, `85 passed in 14.89s`.
- `py_compile` command above: exit 0, no output.
- `git diff --check`: exit 0; only Git's existing LF-to-CRLF warning for `run_multiseed.py`.

Required commit subject: `feat: add official LAD-RG launch preflight`.

## Self-review

- Requirements were checked line by line against the Task 3 brief.
- Official mode rejects dummy/dirty operation, non-LAD-RG configs, wrong N/ratio, missing explicit settings, mismatched provider/model, mutable Qwen revisions, and DeepSeek GASD-R/Both.
- The manifest is deterministic canonical JSON and includes Git SHA, provider/model/revision/tag, sanitized endpoint origin, fixed N=200/0.15 matrix, dependency versions, and decoder constants. Existing tagged outputs cannot resume without an exact compatible manifest.
- Official sentence results must retain exact length/ontology/legal IOB2 and launch evidence; no fallback or defensive padding is allowed. GASD-disabled ablations explicitly use `disabled`.
- Static preflight is read-only, checks only tracked worktree dirtiness as required, validates all 45 canonical files/9,000 records, and captures DEER subprocess output. Live checks are opt-in and transport-injectable.
- Temp cleanup targets only the exact cell `.jsonl.tmp`; unrelated temporary artifacts are preserved.
- `git diff --check`, focused regressions, and compilation were clean. No frozen contextual-lattice hash was readjusted.

## Concerns

- The worktree baseline contradicted the handoff by containing partial Task 3 artifacts. They are being preserved and completed rather than discarded.
- No frozen contextual-lattice hash has been modified.
- The pre-existing untracked root `launch-blocker-plan.md` is outside this Task 3 commit and remains untouched.
- Multi-agent reviewer tools were unavailable in this session, so the requesting-code-review skill could not dispatch an independent reviewer; a manual requirement/diff audit was performed instead.
- Real DEER/Hugging Face initialization, provider credentials, `/v1/models`, GPU, and predictions were intentionally not exercised. Those checks remain launch-time responsibilities of static/live preflight.
- The full repository pytest suite was not run because the user requested focused mocked/no-network verification; 84 related offline tests were run.

## Review fix round 1

### RED evidence

- Required focused suite: `1 failed, 93 passed in 14.99s`; the Qwen live-preflight length blocker reported only a generic one-tag-per-token error instead of the exact expected count.
- Added contract regressions: `4 failed, 1 passed in 0.47s`, exposing resume input-count bypass, shallow persisted-stage validation, missing served-model manifest identity, and direct live-preflight acceptance of a non-official vLLM model.
- Selected-tag namespace regression: `1 failed in 1.56s`, proving a longer unrelated backbone tag was incorrectly treated as the selected launch tag.

### GREEN evidence

- Exact Qwen GASD-R length diagnostic: `1 passed in 0.02s`.
- New resume/evidence/identity/artifact regressions: `7 passed in 1.31s`.
- Noisy-load lifecycle cleanup: `1 passed in 0.04s`.
- Final required focused suite: `99 passed in 13.95s`.
- Required `py_compile`: exit 0, no output.
- `git diff --check`: exit 0; only existing LF-to-CRLF warnings.
- Repository-wide diagnostic with `-x`: `1 failed, 43 passed, 4 warnings in 13.72s`; the sole failure is the intentionally untouched frozen contextual-lattice source-hash inventory in `test_contextual_lattice_runtime.py` after `multi_agent_v2.py` changed. Frozen hash updates remain out of Task 3 scope.

### Files changed

- `official_contract.py`: centralized serialized output-affecting LAD-RG decoder constants and contract/hard-constraint versions.
- `live_backbone.py`: immutable served-model requests/responses, exact evidence fields, centralized reason bonus, and exact GASD-R length diagnostics.
- `multi_agent_v2.py`: immutable Coder/Reviewer request identity, explicit stage status records, and centralized voting/RoR/GASD constants.
- `run_multiseed.py`: exact-200 resume/input/output gates, full-lifecycle exact-temp cleanup, strict stage/identity evidence validation, served identity in manifests, selected-config validation, and corrected failure-policy semantics.
- `official_preflight.py`: selected-config live validation, exact `/v1/models` and adapter/response evidence checks, and canonical selected-tag artifact coordinates without colliding with unrelated tags.
- `test_backbone_evidence.py`, `test_lad_rg_graph.py`, `test_live_backbone.py`, `test_official_preflight.py`, `test_run_multiseed_official.py`: offline regressions for all eight findings.
- `.superpowers/sdd/launch-blocker-plan/task-3-report.md`: this review-fix evidence.

## Review fix round 2

### Rationale

- The approved Coder policy is unchanged: BT and IF intentionally execute strategy paths 1, 2, and 5, while ATF executes paths 1 through 5. No path 3/4 prompts were added to BT/IF and no Coder prompt or repair behavior was altered.
- Coder evidence now uses the actual `path_strategies` keys, so BT/IF records are exactly `coder_path_1`, `coder_path_2`, and `coder_path_5`; ATF records remain `coder_path_1` through `coder_path_5`.
- Official validation now receives the cell noise type and requires the exact corresponding live stage set, rejecting duplicates, missing/extra paths, non-live paths, and unsupported noise types.
- Selected-tag artifact discovery now examines the structural tag/tail components of every `pred_*` name and recognizes the exact tag with any extension or extra component. Dotted tags remain valid, while longer hyphenated tags remain outside the selected namespace.

### RED evidence

- Coder producer/validator regressions: `8 failed, 14 deselected in 8.92s`. The old validator did not accept a cell noise type and the real BT/IF producer still mislabeled strategy 5 as `coder_path_3`.
- Selected-tag artifact regressions: `2 failed, 12 deselected in 1.61s`. Exact-tag `.tmp`/arbitrary-extension artifacts were omitted, including for a dotted tag.

### GREEN evidence

- Coder producer/validator regressions: `8 passed, 14 deselected in 8.72s`.
- Selected-tag artifact regressions: `2 passed, 12 deselected in 1.21s`.
- Final required focused suite: `108 passed in 14.12s`.
- Required `py_compile`: exit 0, no output.
- `git diff --check`: exit 0; only Git's existing LF-to-CRLF warnings were emitted.

### Files changed

- `multi_agent_v2.py`: names Coder evidence from actual path-strategy keys without changing BT/IF's intentional three-path algorithm.
- `run_multiseed.py`: threads cell noise through resume, per-sentence, and completed-output validation and enforces exact noise-specific Coder evidence.
- `official_preflight.py`: structurally recognizes exact selected-tag artifacts across arbitrary extensions/extra tail components and passes artifact noise into output validation.
- `test_run_multiseed_official.py`: mocked real-Coder-to-validator coverage for BT, IF, and ATF plus strict invalid-stage/noise regressions.
- `test_official_preflight.py`: arbitrary-extension, extra-component, dotted-tag, and longer-unrelated-tag regressions.
- `.superpowers/sdd/launch-blocker-plan/task-3-report.md`: this round-2 evidence and rationale.

### Concerns

- No new implementation concern was found in the scoped diff. Live providers, Hugging Face, GPU execution, and prediction generation remained intentionally unexercised per the task constraints.
