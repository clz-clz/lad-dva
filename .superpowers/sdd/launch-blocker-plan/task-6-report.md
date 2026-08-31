# Task 6 Report — whole-branch launch-blocker review

## Integrated review findings

The first independent whole-branch review of `efd9d53..5a33878` found two P1
cross-task contract gaps that focused task reviews had missed:

1. The immutable manifest recorded a 120-second provider timeout and two SDK
   retries, but the LangChain Coder and Reviewer clients still used their
   legacy defaults.
2. Static DEER preflight inherited normal Hugging Face networking and provider
   credentials, so a missing local dataset cache could download or reach an
   external endpoint instead of failing closed.

## Fixes

- Centralized `OFFICIAL_PROVIDER_TIMEOUT_SECONDS=120.0` and
  `OFFICIAL_SDK_MAX_RETRIES=2` in `official_contract.py`; the manifest, live
  adapter, Coder, and Reviewer now consume the same constants.
- Added the runner's pre-import `LAD_RG_OFFICIAL_REQUESTS=1` handshake. It pins
  both official ChatOpenAI clients to timeout 120.0 / retries 2, while a direct
  regression confirms legacy construction receives no override.
- Made each DEER subprocess Hugging Face cache-only by forcing datasets, hub,
  and Transformers offline modes. Real provider credentials and dotenv loading
  are removed. Import-time LangChain clients receive only a non-secret dummy
  key, dummy model, and unreachable `127.0.0.1:9` endpoint.
- Added fail-closed tests for DEER timeout and missing offline cache.
- Updated the launch guide to state the cache-materialization prerequisite and
  the provider timeout/retry contract.

## TDD and verification

RED:

- Three focused regressions initially failed: the official handshake flag was
  absent, Coder/Reviewer exposed `None` timeout/retry settings, and the DEER
  child lacked offline environment controls.
- A credential-free import check then exposed LangChain's import-time key
  requirement; this led to the inert loopback identity rather than restoring
  any real secret.

GREEN (all commands used `D:/py/Anaconda3/python.exe`):

- Provider/offline focused regressions: `3 passed`.
- Focused launch suite after initial fixes: `65 passed`.
- Final DEER/provider/docs set: `20 passed`.
- Complete offline suite: `207 passed, 1 skipped, 4 warnings in 30.77s`.
  The skip is the explicitly opt-in live provider test; warnings are existing
  contextual-lattice diagnostics.
- Required `py_compile`: exit 0.
- Existing dummy candidate-evidence smoke: `1 passed in 8.57s`.
- `git diff --check`: exit 0 (Windows line-ending notices only).
- Independent fix re-review: no findings.

## Boundary

No API, Hugging Face download, GPU/model load, prediction generation,
experimental run, or manuscript edit occurred. Dataset-cache absence is now a
static blocker rather than an implicit preparation action.
