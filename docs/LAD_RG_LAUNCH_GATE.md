# LAD-RG Official Launch Gate

This is the operational contract for the approved N=200 LAD-RG run. The
implementation is fail-closed, but no command in this document should be run
against a paid provider or rented GPU until credentials and experiment
execution are authorized. Run commands from the repository root with
`D:/py/Anaconda3/python.exe` on the Windows experiment host.

Passing offline tests means the implementation is ready to test. It is not B3
or B4 experimental evidence. A full run is authorized only after static and
live preflight pass and both paid-smoke reports have `"ok": true`.

## 1. Common preparation

The tracked worktree must be clean, and `results_multiseed` must contain the
exact 45-file matrix: five datasets × three noise types × three seeds, every
file named as a canonical `N200` file and containing exactly 200 rows.

```powershell
$expectedSha = (git rev-parse HEAD).Trim()
D:/py/Anaconda3/python.exe -m pytest -q
D:/py/Anaconda3/python.exe -m py_compile run_multiseed.py multi_agent_v2.py live_backbone.py official_preflight.py official_smoke.py logit_gap_probe.py
```

Normal pytest is offline. The one live integration test is skipped unless
`RUN_LAD_RG_LIVE_TESTS=1` is explicitly set.

## 2. DeepSeek-V4-Flash gate

Set every identity value explicitly. Use a fresh `BACKBONE_TAG` for a new
official evidence namespace; never reuse a tag for different settings or Git
state.

```powershell
$env:BACKBONE_PROVIDER = "deepseek"
$env:BACKBONE_MODEL = "deepseek-v4-flash"
$env:BACKBONE_BASE_URL = "https://api.deepseek.com/v1"
$env:BACKBONE_API_KEY = "<deepseek-api-key>"
$env:BACKBONE_TAG = "deepseek-v4-flash-<run-id>"
Remove-Item Env:BACKBONE_REVISION -ErrorAction SilentlyContinue
```

Run static preflight first. It validates Git state, all 45 N=200 files, their
ontology and IOB2 invariants, isolated bounded DEER initialization for all five
datasets, and the selected tagged artifact namespace. It writes no prediction.

```powershell
D:/py/Anaconda3/python.exe official_preflight.py static --expected-sha $expectedSha --configs lad_rg_full
D:/py/Anaconda3/python.exe official_preflight.py live --configs lad_rg_full
$env:RUN_LAD_RG_LIVE_TESTS = "1"
$env:LAD_RG_LIVE_CONFIGS = "lad_rg_full"
D:/py/Anaconda3/python.exe -m pytest -q test_lad_rg_live_integration.py
Remove-Item Env:RUN_LAD_RG_LIVE_TESTS,Env:LAD_RG_LIVE_CONFIGS -ErrorAction SilentlyContinue
```

DeepSeek is GASD-G only. GASD-R/Both are Qwen-only and the runner rejects them
for DeepSeek before any provider call.

Run one paid 20-sentence cell. The command reads the first 20 records from the
canonical `conll2003/BT/seed13/N200` source, applies official fail-closed row
validation, and writes only under `predictions_multiseed/smoke/`; it creates no
official run manifest and cannot be resumed as official evidence.

```powershell
D:/py/Anaconda3/python.exe official_smoke.py --config lad_rg_full --dataset conll2003 --noise BT --seed 13 --max-concurrency 5
```

Do not start the N=200 matrix unless the emitted report has `"ok": true`, zero
fallbacks, candidate evidence for all 20 rows, and GASD SER 0.

```powershell
D:/py/Anaconda3/python.exe run_multiseed.py --official --size 200 --ratios 0.15 --configs lad_rg_full --datasets msra conll2003 wnut17 fewnerd ontonotes5 --noise BT IF ATF --seeds 13 42 2024 --max-concurrency 200
```

## 3. Rented Qwen3-32B-AWQ gate

Resolve the exact Hugging Face commit before launch. `BACKBONE_REVISION` must
be a 40-hex commit, not `main`, a branch, a tag, or a short hash. On the rented
Linux GPU host, launch the already-installed vLLM from that immutable revision
and serve the same synthetic identity the client records:

```bash
export BACKBONE_REVISION="<40-hex-qwen-commit>"
export VLLM_API_KEY="<private-vllm-key>"
vllm serve Qwen/Qwen3-32B-AWQ \
  --revision "$BACKBONE_REVISION" \
  --served-model-name "Qwen/Qwen3-32B-AWQ@${BACKBONE_REVISION}" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --quantization awq \
  --dtype half \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95
```

Keep vLLM bound to loopback. On the Windows experiment host, open the SSH
tunnel in a dedicated terminal and leave it running:

```powershell
ssh -N -L 8000:127.0.0.1:8000 <gpu-user>@<gpu-host>
```

In a second Windows terminal, configure the matching client identity. The API
key must equal the remote `VLLM_API_KEY`.

```powershell
$env:BACKBONE_PROVIDER = "vllm"
$env:BACKBONE_MODEL = "Qwen/Qwen3-32B-AWQ"
$env:BACKBONE_REVISION = "<same-40-hex-qwen-commit>"
$env:BACKBONE_BASE_URL = "http://127.0.0.1:8000/v1"
$env:BACKBONE_API_KEY = "<same-private-vllm-key>"
$env:BACKBONE_TAG = "qwen3-32b-awq-<run-id>"
```

Run static preflight, then exercise the endpoint schemas and real GASD-R:

```powershell
$expectedSha = (git rev-parse HEAD).Trim()
D:/py/Anaconda3/python.exe official_preflight.py static --expected-sha $expectedSha --configs lad_rg_full lad_rg_gasd_r lad_rg_gasd_both
D:/py/Anaconda3/python.exe official_preflight.py live --configs lad_rg_gasd_r
$env:RUN_LAD_RG_LIVE_TESTS = "1"
$env:LAD_RG_LIVE_CONFIGS = "lad_rg_gasd_r"
D:/py/Anaconda3/python.exe -m pytest -q test_lad_rg_live_integration.py
Remove-Item Env:RUN_LAD_RG_LIVE_TESTS,Env:LAD_RG_LIVE_CONFIGS -ErrorAction SilentlyContinue
D:/py/Anaconda3/python.exe official_smoke.py --config lad_rg_gasd_r --dataset conll2003 --noise BT --seed 13 --max-concurrency 5
```

The Qwen smoke is a blocker unless every row has `gasd_variant_used: "r"`,
live `gasd_r` provider evidence, no fallback, candidate evidence, and GASD SER
0. After both backbone smoke reports pass, run the official Qwen configurations:

```powershell
D:/py/Anaconda3/python.exe run_multiseed.py --official --size 200 --ratios 0.15 --configs lad_rg_full lad_rg_gasd_r lad_rg_gasd_both --datasets msra conll2003 wnut17 fewnerd ontonotes5 --noise BT IF ATF --seeds 13 42 2024 --max-concurrency 20
```

Official output is resumable only when
`predictions_multiseed/run_manifest__<BACKBONE_TAG>.json` exactly matches the
current Git SHA, provider identity, endpoint origin, dependencies, N=200
protocol, and decoder constants. A mismatch is a hard stop, not a reason to
edit the manifest or reuse stale predictions.

## 4. B4 raw-logit probe

Run the probe on the rented GPU only after the official calls are complete.
Stop vLLM cleanly in its server terminal (normally `Ctrl+C`) and verify that no
vLLM process still owns model memory. Keep the same checkout, canonical noisy
matrix, model ID, `BACKBONE_REVISION`, and `BACKBONE_TAG`. Re-export the same
Qwen run tag if this is a new GPU shell, inspect GPU ownership, and then run:

```bash
export BACKBONE_TAG="<same-qwen-backbone-tag>"
nvidia-smi
python logit_gap_probe.py --confirm-vllm-stopped --revision "$BACKBONE_REVISION" --backbone-tag "$BACKBONE_TAG" --input-root results_multiseed --output-root predictions_multiseed --batch-size 8 --bootstrap-iterations 10000
```

Invoke the GPU host's pinned Python environment if `python` is not already that
interpreter. The committed outputs are
`predictions_multiseed/logit_gap__<BACKBONE_TAG>.jsonl` and
`predictions_multiseed/logit_gap__<BACKBONE_TAG>__aggregate.json`; the aggregate
is installed last and commits the exact JSONL hash and record count.

## 5. Authorization boundary

Static preflight, live preflight, and paid smoke are gates, not substitutes for
the full B3/B4 experiment matrix. This implementation change performed no real
API call, GPU run, prediction refresh, or manuscript edit. B3 synergy and B4
cross-backbone/logit evidence remain unavailable until separately authorized
runs produce complete, validated artifacts.
