# Qwen3-32B-AWQ Contextual Lattice launch gate

This is the release gate for the staged Qwen Contextual Lattice run. It is a
runbook and a static contract; it is not evidence that a provider or GPU run
has completed. Until the paid-run gate is explicitly authorized, every command
in the live sections is documentation only. Static mode makes no provider call.

## Immutable identities

The only official model identity is:

```text
model:        Qwen/Qwen3-32B-AWQ
revision:     0499c3ac83fdef8810b907a23894ba91e95eddd8
served name:  Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8
route:        chat-completions-json-schema
```

The formal Stage A profile is an independent no-thinking namespace. Set
`QWEN_ENABLE_THINKING=false` for the fixed Qwen model. The client sends
`chat_template_kwargs.enable_thinking=false` explicitly on every Coder,
Reviewer, Verifier, and preflight request, and the effective value is recorded
in request evidence, manifests, and provider-cache identity. The tag must
contain `nothink`; never resume or copy a thinking-mode cache. If the variable
is unset, historical caller-controlled behavior remains unchanged. The flag is
ignored for DeepSeek.

Resolve the model at that immutable revision and record a SHA-256 hash for
every model file in the launch manifest. Do not use a moving branch, a
different served name, or an unpinned download. The contextual bundle must
also pass its locked checkpoint, split, manifest, decoder-file, gate-file, and
decoder-model hashes before the first replay row.

The service environment must contain `vllm==0.28.0` and
`accelerate==1.14.0`; record the output of:

```bash
python -m pip check
python -c "import vllm, accelerate; print(vllm.__version__, accelerate.__version__)"
```

The commands above belong on the prepared A800 host after approval. They are
not part of the Windows static check and must not be used to bypass the
budget gate.

## Single A800 service

Run one A800-80G service only. Keep vLLM loopback-only and expose it to the
Windows experiment host through an SSH tunnel. The launch arguments are
locked exactly as follows:

```bash
vllm serve Qwen/Qwen3-32B-AWQ \
  --revision 0499c3ac83fdef8810b907a23894ba91e95eddd8 \
  --served-model-name Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8 \
  --host 127.0.0.1 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --quantization awq --dtype half \
  --reasoning-parser qwen3 \
  --structured-outputs-config.enable_in_reasoning=True \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32
```

From the Windows host, use a loopback tunnel such as:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@a800-host
```

With the tunnel up, query `/v1/models` using the private key and require one
and only one model whose `id` is the exact served name above. A different ID,
an extra model, a non-loopback listener, or a failed health response blocks
the run.

Record these fields in the immutable launch manifest before smoke execution:

| Evidence | Required value |
|---|---|
| model revision | `0499c3ac83fdef8810b907a23894ba91e95eddd8` |
| model file hashes | SHA-256 map for all resolved files |
| vLLM / accelerate | `0.28.0` / `1.14.0` |
| CUDA | driver/runtime version |
| GPU name / GPU memory | `A800-80G` and reported memory |
| server arguments | exact command-line vector above |
| `/v1/models` | exactly the pinned served identity |
| Git and bundle | final Git SHA and locked contextual bundle hashes |
| provider fingerprint | response system fingerprint(s), if exposed |

## Windows static gate

Use the mandated interpreter from the experiment host:

```powershell
D:/py/Anaconda3/python.exe -m pytest -q
D:/py/Anaconda3/python.exe -m py_compile official_provider_cache.py live_backbone.py multi_agent_v2.py run_multiseed.py official_preflight.py official_smoke.py qwen_budget_executor.py qwen_full_audit.py
D:/py/Anaconda3/python.exe official_preflight.py static `
  --repo-root . --expected-sha $env:EXPECTED_GIT_SHA `
  --configs selectdenoise_contextual_lattice `
  --request-timeout 7200
git diff --check
```

Static mode may inspect local DEER/Hugging Face caches and manifests, but it
must not instantiate a live provider adapter or call `/v1/models`. Missing
caches, missing noisy cells, a dirty tracked worktree, a Git-SHA mismatch, or
an incompatible manifest is a blocker. Do not replace a failed static check
with a live call.

The contextual bundle and embedding cache are local frozen artifacts. Set
`CONTEXTUAL_LATTICE_BUNDLE` and
`CONTEXTUAL_LATTICE_ENCODER_CACHE` to existing paths before static replay
validation; replay must reject a missing or newly-created embedding path.

## Smoke and cost gate

Before the full provider-cache matrix, run only the approved contextual smoke
selection: the fixed `msra/BT/seed13` rows 125 and 133, the fixed
`msra/ATF/seed13` row 2, and the median/p95/max representative rows across
all 15 seed-13 cells. The smoke must pass the exact Qwen identity, strict
JSON-schema lengths and ontology, stage evidence, no fallback, and the
contextual terminal provenance checks.

Measure the smoke wall time, request throughput, and observed Coder/Reviewer/
Verifier trigger counts. Estimate Stage A hours from those measurements and
the complete 45-cell request schedule, multiply the measured rental estimate
by 1.25, and obtain an explicit rental budget before the first full provider
request. The full protocol is five datasets × three noise types × three seeds
× 200 records: 45 cells and 9,000 rows. A missing budget is a hard stop.

## Staged execution and evidence boundary

### Timeout recovery on a replacement host

Use an independently verified SSH identity and a distinct local tunnel port
when moving to a cloned instance. Verify the service key against `/v1/models`;
an SSH password change does not imply a service-key change. Do not run the
same experiment concurrently on the old and replacement instances.

For the formal no-thinking Qwen continuation, use
`QWEN_PROVIDER_TIMEOUT_SECONDS=600` and `--request-timeout 7200`. Retain the
32-request provider cap and at most two SDK retries. Provider timeout, runner
timeout, and concurrency are recorded as provenance; they are not semantic
cache conflicts. Model revision, no-thinking mode, contextual configuration,
bundle hash, and every input digest remain strict identity checks. The runner's
budget executor chains the new continuation ledger to the immutable old ledger
SHA, so a later phase cannot reacquire the full budget.

Set `QWEN_DIAGNOSTICS_DIR` to a **local Git-ignored directory** for opt-in
evidence. Each logical request gets a UUID and SHA-256 payload digest; JSONL
events record dataset/noise/seed/original zero-based row, stage/path, queue
wait, SDK attempt index, timing, exception-type chain, and available usage.
Unknown timeout usage is not estimated. Allowlisted model-facing request
bodies are captured separately without credentials or target gold labels.
Do not publish these bodies. A diagnostic sink error must not replace the
provider exception or cause another HTTP attempt.

Run the no-thinking diagnostic for zero-based row 193, Coder Path 5, then row
2, in separate request namespaces with SDK retries disabled. Select evidence
by the exact diagnostic directory and full request ID. A streaming observation
may run for at most 600 seconds and is diagnostic only, never a formal
prediction; chunks are persisted incrementally so content before a cutoff is
available. Do not truncate outputs, skip samples, relax validation, add DFA,
or mutate limits during a run. Preserve the first provider exception when
aborting.

Before restoring Stage A, require the fixed 47-row smoke to pass. Commit and
push the verified code, then continue the original formal tag
`qwen32b-contextual-nothink-v1`; never copy diagnostic cells into the formal
cache. Confirm cumulative cost and the replacement host's hourly
rate against the approved total budget, or label a conservative estimate
explicitly. The executor must enforce a hard deadline derived from remaining
budget, including diagnosis and reruns. Stopping requests does not itself
stop instance rental billing.

Stage A is the only paid phase. After the smoke and budget approval, set the
original formal tag, set the independent no-thinking profile, and run only
`selectdenoise_contextual_lattice` with size 200,
ratio 0.15, max client concurrency 32, and failure policy `abort`:

```powershell
$env:QWEN_ENABLE_THINKING = "false"
$env:QWEN_PROVIDER_TIMEOUT_SECONDS = "600"
$env:BACKBONE_TAG = "qwen32b-contextual-nothink-v1"

D:/py/Anaconda3/python.exe qwen_budget_executor.py `
  --total-yuan 110 --spent-yuan $env:SPENT_YUAN `
  --hourly-rate-yuan $env:HOURLY_RATE_YUAN `
  --billing-basis $env:BILLING_BASIS `
  --ledger "$env:QWEN_DIAGNOSTICS_DIR/continuation-budget-ledger.json" `
  --previous-ledger "$env:QWEN_DIAGNOSTICS_DIR/006-n100-budget-ledger.json" `
  --report "$env:QWEN_DIAGNOSTICS_DIR/budget-launch.json" -- `
  --official --phase provider-cache `
  --configs selectdenoise_contextual_lattice `
  --size 200 --ratios 0.15 --max-concurrency 32 `
  --request-timeout 7200 `
  --bundle-hash $env:CONTEXTUAL_BUNDLE_HASH `
  --provider-cache-root provider_cache `
  --reuse-provider-cache-tag qwen32b-contextual-nothink-s13-n100-v1 `
  --compatible-provider-cache-git-sha $env:EXISTING_FULL_PRODUCER_SHA `
  --compatible-provider-cache-git-sha $env:REDUCED_PREFIX_PRODUCER_SHA
```

The eight valid N200 cells in the target tag are verified and skipped without
requests. For each other seed-13 cell, a valid N100 source prefix is held only
in memory while rows 100–199 run; the target is published atomically only when
all 200 rows pass. A missing seed-13 prefix is a hard pre-request failure;
seed 42/2024 cells have no approved prefix and run all 200 rows. A source SHA,
source tag, selected range, and source producer Git SHA are retained in the
final manifest. Stop vLLM immediately after the provider-cache index is complete.
Validate all 45 cache cells and exactly 200 rows per cell before proceeding.

Stage B is local and must not receive provider credentials. It re-reads each
canonical noisy row, checks the input digest and cache SHA, validates the
locked bundle and embedding path, then decodes only from the seven locked
terminal inputs. Run it with the explicit cache namespace:

```powershell
D:/py/Anaconda3/python.exe run_multiseed.py `
  --phase contextual-replay `
  --configs selectdenoise_contextual_lattice `
  --size 200 --ratios 0.15 `
  --provider-cache-root provider_cache `
  --provider-cache-tag $env:BACKBONE_TAG `
  --bundle-hash $env:CONTEXTUAL_BUNDLE_HASH `
  --compatible-provider-cache-git-sha $env:EXISTING_FULL_PRODUCER_SHA
```

The final prediction rows must retain the provider-cache SHA,
`terminal_anchor_tags`, `terminal_model_hash`, `terminal_used_anchor`,
`terminal_predicted_gain`, and `terminal_fallback_count`. Validate 45 final
prediction cells with exactly 200 rows each before aggregation. Any digest,
identity, bundle, ontology, length, IOB2, provenance, or fallback failure
aborts the run. Do not pad, truncate, reorder, retry schemas, or fall back to
another method in official mode.

Aggregate only the formal method/tag, then run the exact full audit:

```powershell
$env:BACKBONE_TAG = "qwen32b-contextual-nothink-v1"
D:/py/Anaconda3/python.exe aggregate_seeds.py `
  --methods selectdenoise_contextual_lattice `
  --reference selectdenoise_contextual_lattice

D:/py/Anaconda3/python.exe qwen_full_audit.py `
  --noisy-root results_multiseed `
  --provider-cache-root provider_cache `
  --predictions-root predictions_multiseed `
  --bundle-hash $env:CONTEXTUAL_BUNDLE_HASH `
  --producer-git-sha $env:EXPECTED_GIT_SHA `
  --compatible-provider-cache-git-sha $env:EXISTING_FULL_PRODUCER_SHA `
  --compatible-provider-cache-git-sha $env:REDUCED_PREFIX_PRODUCER_SHA `
  --existing-full-producer-git-sha $env:EXISTING_FULL_PRODUCER_SHA `
  --expected-existing-full-cells 8 --expected-prefix-cells 12
```

The audit fails unless the exact tagged 45-cell cache and prediction matrices
contain 9,000 rows, every cache SHA/input digest/provider identity passes, all
predictions match the ontology and strict IOB2, no fallback/temporary file or
provider gold/credential field exists, reuse counts are exactly 8 and 12, and
`aggregated__qwen32b-contextual-nothink-v1.json` is complete and newer than
every prediction.

Historical Qwen LAD-RG manifests, smokes, logit-gap outputs, and prediction
files are historical evidence. This run writes a new tag and must not
overwrite them. No new LAD-RG-R/Both ablation is authorized by this plan.
