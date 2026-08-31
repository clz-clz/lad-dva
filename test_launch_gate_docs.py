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
