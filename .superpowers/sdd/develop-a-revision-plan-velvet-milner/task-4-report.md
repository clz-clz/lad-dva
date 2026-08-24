## Task 4 Report - B1/B2 LAD-RG graph wiring

Implemented in `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice` only.

### Summary

- Added an opt-in compiled `lad_rg_graph` wired as `coder -> reviewer -> ror -> gasd`.
- Left the existing `multi_agent_graph` wired as `coder -> reviewer -> verifier` for SelectDenoise and contextual-lattice configurations.
- Added explicit `terminal_graph: "lad-rg"` only to the `lad_rg_*` configurations.
- Updated `run_agent_pipeline()` to select the LAD-RG graph only when that explicit config flag is present.
- Kept RoR as proposal-only output: it returns `current_tags` as the base vote and `ror_proposals` for GASD.
- Kept GASD deterministic and provider-free; it uses candidate fidelity, optional omega potentials, and RoR proposal bonuses under IOB2-constrained Viterbi decoding.
- Hardened GASD no-candidate/error fallback to legalize base/dirty tags and normalize length.

### RED Evidence

Command:

```powershell
D:/py/Anaconda3/python.exe -m pytest -q test_lad_rg_graph.py
```

Result before implementation:

```text
3 failed, 4 passed in 10.05s
```

Expected failures:

- `test_lad_rg_configurations_select_lad_rg_graph_and_preserve_selectdenoise_configs` failed with `KeyError: 'terminal_graph'`.
- `test_run_agent_pipeline_uses_lad_rg_graph_only_for_lad_rg_config` failed because `multi_agent_v2` had no `lad_rg_graph`.
- `test_gasd_decodes_legal_iob2_from_candidates_potentials_and_proposals` exposed that the test expectation was too strict for legal global decoding from an illegal `I-LOC` candidate; I corrected it to assert the legal constrained output at the proposed token.

### GREEN Evidence

Focused Task 4 tests:

```powershell
D:/py/Anaconda3/python.exe -m pytest -q test_lad_rg_graph.py
```

```text
7 passed in 9.32s
```

Regression slice:

```powershell
D:/py/Anaconda3/python.exe -m pytest -q test_lad_rg_graph.py test_backbone_evidence.py test_contextual_lattice_runtime.py::test_versioned_configuration_exposes_contextual_default_and_legacy_rollback test_contextual_lattice_runtime.py::test_mocked_host_pipeline_invokes_terminal_without_provider_call
```

```text
13 passed in 9.82s
```

Diff whitespace check:

```powershell
git diff --check
```

```text
exit 0; only existing Windows line-ending warnings for multi_agent_v2.py and run_multiseed.py
```

### Files Changed

- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\multi_agent_v2.py`
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\run_multiseed.py`
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\test_lad_rg_graph.py`
- `C:\Users\LinzeChen\AI_Workspace\.worktrees\selectdenoise-contextual-lattice\.superpowers\sdd\develop-a-revision-plan-velvet-milner\task-4-report.md`

No manuscript files were touched.

### Test Coverage Added

- Configuration selection: only `lad_rg_*` configs opt into `terminal_graph: "lad-rg"`; SelectDenoise/contextual-lattice configs remain unchanged.
- Pipeline graph selection: `run_agent_pipeline()` invokes `lad_rg_graph` only for the explicit LAD-RG graph marker.
- RoR gate behavior: omega/confidence-gated proposals fire only at high-omega low-confidence positions.
- Ungated diagnostic: `ror_ungated=True` bypasses the omega/confidence gate.
- Proposal routing: RoR proposals are not applied as raw final tags.
- GASD legal decoding: constrained Viterbi emits legal IOB2 under candidate fidelity, omega, and proposal bonuses.
- `gasd_potentials=False`: removes only the omega term; proposal bonuses still apply.
- GASD fallback: no-candidate and decode-error paths return legal, length-normalized fallback tags.

### Self-Review

- The LAD-RG selector is explicit and conservative. Existing configs that only happen to carry `use_ror` or `gasd_potentials` do not switch graphs.
- Contextual-lattice behavior remains terminal-only after the existing SelectDenoise graph; the mocked contextual host-adapter regression still passes.
- RoR remains deterministic and proposal-only. It still computes base tags, but final LAD-RG output comes from GASD.
- GASD now uses the shared `GASD_BETA_OMEGA = 2.0` constant in the LAD-RG graph path, matching the existing base-decode constant.
- The no-provider-call behavior for RoR/GASD was preserved; the new tests call those helpers directly without any LLM mocks.

### Concerns

- I did not run live provider-backed graph execution; tests use deterministic helper calls and mocked graph invocations to avoid API calls.
- I did not run the full repository test suite because prior task reports note that `test_selectdenoise.py` can hang during cached Hugging Face dataset loading in this environment.
- The new graph wiring is covered structurally through graph selection and helper behavior, but there is no end-to-end provider-backed LAD-RG sentence test.
