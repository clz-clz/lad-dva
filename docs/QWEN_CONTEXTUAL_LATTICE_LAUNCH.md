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
D:/py/Anaconda3/python.exe -m py_compile official_provider_cache.py live_backbone.py multi_agent_v2.py run_multiseed.py official_preflight.py official_smoke.py
D:/py/Anaconda3/python.exe official_preflight.py static `
  --repo-root . --expected-sha $env:EXPECTED_GIT_SHA `
  --configs selectdenoise_contextual_lattice
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

Stage A is the only paid phase. After the smoke and budget approval, set a
fresh tag and run only `selectdenoise_contextual_lattice` with size 200,
ratio 0.15, max client concurrency 20, and failure policy `abort`:

```powershell
D:/py/Anaconda3/python.exe run_multiseed.py `
  --official --phase provider-cache `
  --configs selectdenoise_contextual_lattice `
  --size 200 --ratios 0.15 --max-concurrency 20 `
  --bundle-hash $env:CONTEXTUAL_BUNDLE_HASH `
  --provider-cache-root provider_cache
```

The provider cache is immutable, gold-free, atomic, SHA-checked, and resumable
only when its identity matches the manifest. Stop vLLM immediately after the
provider-cache index is complete. Validate all 45 cache cells and exactly 200
rows per cell before proceeding.

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
  --bundle-hash $env:CONTEXTUAL_BUNDLE_HASH
```

The final prediction rows must retain the provider-cache SHA,
`terminal_anchor_tags`, `terminal_model_hash`, `terminal_used_anchor`,
`terminal_predicted_gain`, and `terminal_fallback_count`. Validate 45 final
prediction cells with exactly 200 rows each before aggregation. Any digest,
identity, bundle, ontology, length, IOB2, provenance, or fallback failure
aborts the run. Do not pad, truncate, reorder, retry schemas, or fall back to
another method in official mode.

Historical Qwen LAD-RG manifests, smokes, logit-gap outputs, and prediction
files are historical evidence. This run writes a new tag and must not
overwrite them. No new LAD-RG-R/Both ablation is authorized by this plan.
