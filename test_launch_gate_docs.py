import re
from pathlib import Path


ROOT = Path(__file__).parent


def test_launch_guide_covers_every_authorized_gate_and_command():
    guide = (ROOT / "docs" / "LAD_RG_LAUNCH_GATE.md").read_text(encoding="utf-8")

    for variable in (
        "BACKBONE_PROVIDER", "BACKBONE_MODEL", "BACKBONE_BASE_URL",
        "BACKBONE_API_KEY", "BACKBONE_TAG", "BACKBONE_REVISION",
        "RUN_LAD_RG_LIVE_TESTS", "LAD_RG_LIVE_CONFIGS",
    ):
        assert variable in guide
    for command in (
        "official_preflight.py static", "official_preflight.py live",
        "official_smoke.py --config lad_rg_full",
        "official_smoke.py --config lad_rg_gasd_r",
        "run_multiseed.py --official --size 200",
        "logit_gap_probe.py --confirm-vllm-stopped",
        "--served-model-name",
        "Qwen/Qwen3-32B-AWQ@",
        "ssh -N -L",
    ):
        assert command in guide
    assert "DeepSeek" in guide and "GASD-R/Both" in guide and "Qwen-only" in guide
    assert "cache-only" in guide
    assert "missing cache" in guide.lower()
    assert not re.search(r"--official[^\n]*--size\s+20\b", guide)


def test_docs_make_readiness_evidence_boundary_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    boundary = (
        ROOT / "docs" / "contextual_lattice" / "AVAILABLE_ARTIFACTS.md"
    ).read_text(encoding="utf-8")

    assert "docs/LAD_RG_LAUNCH_GATE.md" in readme
    assert "implementation readiness" in boundary.lower()
    assert "not completed B3/B4 evidence" in boundary
    assert "no real provider or gpu run" in boundary.lower()


def test_qwen_contextual_launch_guide_pins_service_and_cost_gates():
    guide = (ROOT / "docs" / "QWEN_CONTEXTUAL_LATTICE_LAUNCH.md").read_text(encoding="utf-8")

    for value in (
        "Qwen/Qwen3-32B-AWQ",
        "0499c3ac83fdef8810b907a23894ba91e95eddd8",
        "vllm==0.28.0",
        "accelerate==1.14.0",
        "python -m pip check",
        "--host 127.0.0.1 --port 8000",
        "--quantization awq --dtype half",
        "--reasoning-parser qwen3",
        "--structured-outputs-config.enable_in_reasoning=True",
        "--max-model-len 32768",
        "--gpu-memory-utilization 0.90",
        "--max-num-seqs 32",
        "/v1/models",
        "ssh -N -L",
        "model file hashes",
        "CUDA",
        "GPU name",
        "GPU memory",
        "provider-cache",
        "contextual-replay",
        "45 cells",
        "9,000 rows",
        "1.25",
        "explicit rental budget",
        "no provider call",
    ):
        assert value in guide
    assert "Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8" in guide
    assert "stop vllm" in guide.lower()
    assert "historical" in guide.lower()
