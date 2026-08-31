# Available Artifacts and Verification Boundary

This clean tracked snapshot includes the locked contextual-lattice code,
`metrics_450.json`, and `frozen_manifest.json`, but it does not include the
manuscript-only files `acl_latex.tex`, `custom.bib`, or `f1.pdf`. It also does
not include fresh live-run outputs produced by provider-backed or GPU-backed
re-execution beyond the already tracked locked bundle and prediction hashes.
In particular, the B3 synergy/non-substitutability grid and B4 cross-backbone,
logit, and GASD-R live evidence are unavailable. The offline seams and tests do
not constitute experimental results for either study.

The launch adapter, strict runner, preflight, paid-smoke seam, and logit probe
establish implementation readiness; they are not completed B3/B4 evidence.
This launch-safety implementation performed no real provider or GPU run and
generated no fresh experimental prediction, logit, or aggregate artifact.
The manifest's full `source_sha256` map remains the historical lock snapshot;
only the three frozen contextual-lattice runtime sources are expected to stay
byte-identical as later runner, pipeline-integration, analysis, and test code
evolves.

Verification in this worktree is therefore limited to tracked source/docs
consistency, locked hash inventory, and offline tests that do not require live
API or GPU execution. No manuscript edits, bibliography reconciliation, figure
refresh, response-letter claims, or new experimental results should be asserted
unless those artifacts are later added as tracked files.
