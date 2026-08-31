# Task 5 Report — official launch gate, test isolation, and operations

## Scope and implementation

- Replaced every direct live `_init_deer()` use in `test_selectdenoise.py` with
  deterministic in-memory statistics/retrieval fixtures. The fixture preloads
  the only test dataset and fails if a test attempts a different live dataset.
- Added `test_lad_rg_live_integration.py`. It is guarded by
  `RUN_LAD_RG_LIVE_TESTS=1`; normal pytest skips it before any provider call.
- Added `official_smoke.py` and a narrow strict-smoke seam in
  `run_multiseed.py`. The smoke always reads a complete canonical N=200 cell,
  deterministically selects its first 20 rows, reuses one live adapter, applies
  the official fail-closed row/evidence checks, and writes under
  `predictions_multiseed/smoke/` without creating or resuming an official
  manifest. The official runner still requires N=200 and ratio 0.15.
- The smoke report independently gates exact record count, zero fallback,
  complete candidate evidence, requested/used GASD variant, live GASD-R for
  R/Both, and zero SER on GASD's terminal hard-Viterbi output. Pipeline chatter
  is routed to stderr so stdout remains one JSON report.
- Added `docs/LAD_RG_LAUNCH_GATE.md` with the exact DeepSeek and Qwen variables,
  immutable Qwen revision/served identity, vLLM command, SSH tunnel, static and
  live preflight, opt-in integration check, one 20-row paid smoke per backbone,
  official N=200 commands, and the post-vLLM B4 logit probe.
- Updated `README.md` and
  `docs/contextual_lattice/AVAILABLE_ARTIFACTS.md` to distinguish implementation
  readiness from completed B3/B4 evidence.
- Preserved `docs/contextual_lattice/frozen_manifest.json` byte-for-byte. Its
  `source_sha256` map remains the historical lock inventory. The runtime test
  now checks byte identity only for the three genuinely frozen contextual-
  lattice runtime sources while retaining and validating the full historical
  inventory.

## TDD evidence

RED checks observed before implementation/fixes:

- `test_run_multiseed_official.py -k paid_smoke`: 2 failures because the strict
  smoke seam did not exist.
- `test_official_smoke.py`: collection error because `official_smoke.py` did
  not exist.
- `test_launch_gate_docs.py`: 2 failures because the launch guide/link and
  explicit artifact boundary did not exist.
- The frozen-inventory test failed while hashing mutable post-lock runner and
  pipeline files against their historical hashes.
- The smoke stdout regression failed with `JSONDecodeError` when simulated
  pipeline chatter preceded the JSON report.
- The independent GASD-variant gate regression failed because a
  `g_fallback` row was initially accepted by the report assessor.

GREEN checks after implementation:

- Task-focused set: `53 passed, 1 skipped`.
- `test_official_smoke.py`: `6 passed`.
- Frozen inventory focused test: `1 passed`.
- SelectDenoise isolation: `4 passed` with no dataset access.
- Documentation contract: `2 passed`.

## Final verification

All commands used `D:/py/Anaconda3/python.exe`.

- Complete offline suite: `203 passed, 1 skipped, 4 warnings in 20.90s`.
  The skipped test is the explicitly opt-in live integration test. The four
  warnings are existing contextual-lattice legality/core-count diagnostics.
- Required compilation:
  `python -m py_compile run_multiseed.py multi_agent_v2.py live_backbone.py official_preflight.py official_smoke.py logit_gap_probe.py` — exit 0.
- Existing isolated dummy smoke:
  `test_backbone_evidence.py::test_dummy_mode_persists_candidate_evidence_for_lad_rg_configs`
  — `1 passed in 8.65s`.
- `git diff --check` — exit 0 (Windows line-ending notices only).
- Independent review found two smoke-report questions. Requested/used GASD
  validation was added with a regression test; GASD SER was confirmed to refer
  to the terminal hard-Viterbi output, not provider score-input tags. Re-review
  result: no actionable findings.

## Boundary

No API, Hugging Face download, GPU/model load, prediction regeneration,
manuscript edit, or experimental run occurred. The live integration test was
not enabled. The paid-smoke and official commands are documented and tested
with mocks only. B3 synergy and B4 cross-backbone/logit evidence remain absent.

Required commit subject: `docs: document official LAD-RG launch gate`.
