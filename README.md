# Multi-seed runner for SelectDenoise and LAD-RG

Drop these five files alongside your existing `multi_agent_v2.py`,
`rag_voting_engine.py`, `utils.py`, and `chroma_db/`. Then:

```bash
# 0. Dependencies (one-time)
pip install datasets seqeval numpy tqdm

# 1. Set your API key
export DEEPSEEK_API_KEY=sk-...

# 2. Generate per-seed noisy data (3 seeds × 3 datasets × 3 noise types = 27 files)
python gen_noisy.py
# Writes: results_multiseed/noisy_seed{S}__{NT}__{ds}__N2000.jsonl

# 3. Run the experiments (multi-seed, multi-config, multi-dataset, multi-noise)
python run_multiseed.py
# Writes: predictions_multiseed/pred_seed{S}__{config}__{ds}__{NT}.jsonl
# Resumable — kill and restart at any time.

# 4. Aggregate and emit the new Table 1
python aggregate_seeds.py > new_table1.tex
# Reads predictions_multiseed/, computes mean±std and paired bootstrap,
# writes predictions_multiseed/aggregated.json, prints LaTeX to stdout.
```

That's the entire flow. No code editing required.

## Files

| File | Purpose | Edit? |
|---|---|---|
| `noise_injector.py` | Entity-level BT/IF/ATF, matching `ablation_runner4.py` | No |
| `metrics.py` | Span F1, IOB2 SER, paired bootstrap | No |
| `gen_noisy.py` | HuggingFace loader + per-seed noise generation | No |
| `run_multiseed.py` | Multi-seed orchestrator; wraps `run_agent_pipeline` | Maybe — see below |
| `aggregate_seeds.py` | Read predictions → LaTeX Table 1 | No |

## What gets written

### Noisy data files (`results_multiseed/`)

One file per `(dataset, noise_type, seed)`:

```
noisy_seed13__BT__msra__N2000.jsonl
noisy_seed13__BT__conll2003__N2000.jsonl
...
```

Each line:
```json
{"tokens": ["..."], "ner_tags": ["B-PER", "I-PER", "O"], "dirty_tags": ["O", "I-PER", "O"]}
```

Field names match what `multi_agent_v2.run_agent_pipeline` expects.

### Prediction files (`predictions_multiseed/`)

One file per `(config, dataset, noise_type, seed)`:

```
pred_seed13__lad_dva_full__msra__BT.jsonl
pred_seed13__semantic_rag_baseline__msra__BT.jsonl
...
```

Each line:
```json
{"tokens": ["..."], "gold_tags": ["B-PER", "I-PER", "O"], "pred_tags": ["B-PER", "I-PER", "O"]}
```

### Aggregated JSON (`predictions_multiseed/aggregated.json`)

```json
{
  "lad_dva_full|msra|BT": {
    "n_seeds": 3,
    "f1_mean": 0.6312, "f1_std": 0.0091,
    "p_mean":  0.7104, "p_std":  0.0156,
    "r_mean":  0.5670, "r_std":  0.0123,
    "ser_mean": 0.0,   "ser_std": 0.0,
    "per_seed": [...],
    "bootstrap_vs_ref": {
      "f1_A": 0.6312, "f1_B": 0.6210, "delta": 0.0102,
      "p_value": 0.018, "ci_95_low": 0.001, "ci_95_high": 0.018
    }
  },
  ...
}
```

## Configuration

The legacy LAD-DVA-style examples below are preserved for historical file and
rollback-name continuity. For new denoising runs, `run_multiseed.py` defaults
to the locked contextual-lattice terminal via
`selectdenoise_contextual_lattice`; the pre-terminal SelectDenoise graph remains
addressable through the explicit `selectdenoise_full_legacy` rollback name, and
the modern LAD-RG ablations stay opt-in under the `lad_rg_*` configuration
keys.

```python
CONFIGURATIONS = {
    "lad_dva_full":           {"lambda_bias": 1.0, "use_dfa": True,  "use_topology_rag": True},
    "semantic_rag_baseline": {"lambda_bias": 1.0, "use_dfa": False, "use_topology_rag": False},
    # uncomment to add ablations:
    # "wo_topology_rag":     {"lambda_bias": 1.0, "use_dfa": True,  "use_topology_rag": False},
    # "wo_dfa_wash":         {"lambda_bias": 1.0, "use_dfa": False, "use_topology_rag": True},
    # "late_fusion_bias":    {"lambda_bias": 1.35, "use_dfa": True, "use_topology_rag": True},
}
```

Adding all five gives you the full ablation table (5 methods × 3 datasets × 3 noise × 3 seeds = 135 cells).

## Smoke test (offline, no API calls)

```bash
python gen_noisy.py --dummy --size 30          # 27 tiny noisy files
python run_multiseed.py --dummy                 # mock pipeline; 27 default cells in seconds
python aggregate_seeds.py > smoke_table.tex     # confirm table format
```

This verifies all four scripts work and your output paths are right *before*
you spend any DeepSeek API quota.

## Locked contextual-lattice terminal

New future runs default to the versioned `selectdenoise_contextual_lattice`
configuration when `CONTEXTUAL_LATTICE_BUNDLE` points at the frozen runtime
bundle.  The terminal is an offline, exact interval decoder with an anchor
fallback; it makes no additional provider calls.  The reversible
`selectdenoise_full_legacy` configuration preserves the historical graph.

The complete procedure, allow-list, provenance, and locked 450-row metrics are
documented in
[`docs/contextual_lattice/CONTEXTUAL_LATTICE_V1.md`](docs/contextual_lattice/CONTEXTUAL_LATTICE_V1.md),
with machine-readable records in
[`metrics_450.json`](docs/contextual_lattice/metrics_450.json) and
[`frozen_manifest.json`](docs/contextual_lattice/frozen_manifest.json), plus a
snapshot-boundary note in
[`AVAILABLE_ARTIFACTS.md`](docs/contextual_lattice/AVAILABLE_ARTIFACTS.md). The
artifact is a development candidate only: its locked gate is 12/15 positive
cells, so the formal 13/15 requirement remains unmet.

## Cost estimate (real run)

For a representative two-method run
(`selectdenoise_contextual_lattice` plus `selectdenoise_full_legacy`) across
3 datasets × 3 noise × 3 seeds × 2000 samples × 5 candidates:
- **Coder calls**: 2 × 3 × 3 × 3 × 2000 = **108,000** calls returning 5 candidates each
- **Reviewer calls**: 10,800 (one per sentence)
- **Total**: ~21,600 DeepSeek API calls

At ~$0.0001 / 1K input tokens for DeepSeek-V2 chat, this is ballpark **a few dollars**. Adding the three ablations multiplies this by 2.5×.

If your budget is tight: drop `--seeds` to `13 42` (2 seeds still gives std), or `--size 100`.

## Statistical conventions

* **Seeds** vary the noise realization (which 15% of entities get corrupted). Across seeds, the LLM also has its own sampling noise (`temperature=0.7`). The reported std captures both, which is what reviewers want.
* **Mean ± std**: sample std (`ddof=1`). With 3 seeds this is intentionally wide; don't overclaim 0.01-F1 gaps.
* **Paired bootstrap**: sentence-level resampling (10K iterations), predictions concatenated across seeds before resampling. Markers: `*` p<0.05, `**` p<0.01, `***` p<0.001 vs. `semantic_rag_baseline` (configurable with `--reference`).
* **DFA Wash SER**: deterministic and parameter-free, so should always be `0.00%` when `use_dfa=True`. If you see non-zero SER for a DFA-enabled config, your `enforce_iob2_syntax` is letting something through.

## Common gotchas

1. **`gen_noisy.py` failing on a dataset**: HuggingFace may require `trust_remote_code=True` or a specific `revision`. Edit `_HF_CONFIG` in `gen_noisy.py` if the default paths fail; the most common fix is adding `"hf_revision": "main"`.

2. **MSRA tag mapping**: HuggingFace's `msra_ner` uses tag IDs that I mapped to `["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]`. Double-check against your local pipeline — if you've been using a different ordering, edit `_HF_CONFIG["msra"]["id2tag"]`.

3. **`run_multiseed.py` says "noisy file not found"**: run `gen_noisy.py` first. The two are decoupled so you can regenerate noise without re-running the pipeline (and vice versa).

4. **`run_multiseed.py` import error**: it imports `from multi_agent_v2 import run_agent_pipeline`. Run from the directory containing your `multi_agent_v2.py`, or `export PYTHONPATH=/path/to/that/dir`.

5. **The pipeline accepts a `seed`?** Your current `run_agent_pipeline(tokens, dirty_tags, config)` does not have a seed parameter, so all randomness across seeds comes from (a) different noise realizations (seed-dependent) and (b) the LLM's temperature=0.7 sampling (uncontrolled). This is fine for std error bars — that's the genuine variance you want to capture. If you want tighter control, modify `multi_agent_v2.ChatOpenAI` to accept `model_kwargs={"seed": ...}` and thread the seed through `run_agent_pipeline`; the orchestrator already passes it via `config["__seed__"]` in dummy mode and you can plumb it in real mode the same way.
