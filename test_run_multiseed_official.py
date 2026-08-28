import asyncio
import json
from pathlib import Path

import pytest

import run_multiseed
from live_backbone import LiveBackboneSettings


def _env(provider="vllm"):
    values = {
        "BACKBONE_PROVIDER": provider,
        "BACKBONE_MODEL": "Qwen/Qwen3-32B-AWQ",
        "BACKBONE_BASE_URL": "http://127.0.0.1:8000/v1?secret=no",
        "BACKBONE_API_KEY": "test-key",
        "BACKBONE_TAG": "qwen32b-r1",
        "BACKBONE_REVISION": "a" * 40,
    }
    if provider == "deepseek":
        values["BACKBONE_MODEL"] = "deepseek-v4-flash"
        values["BACKBONE_BASE_URL"] = "https://api.deepseek.com/v1"
        values.pop("BACKBONE_REVISION")
    return values


def _write_noisy(path: Path, count=1):
    row = {
        "tokens": ["Alice", "works"],
        "ner_tags": ["B-PER", "O"],
        "dirty_tags": ["B-PER", "O"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8")


def test_cli_defaults_abort_and_require_explicit_legacy_dirty():
    official = run_multiseed._parse_args(["--official", "--size", "200"])
    legacy = run_multiseed._parse_args([])
    explicit_dirty = run_multiseed._parse_args(["--failure-policy", "dirty"])

    assert (official.failure_policy, official.request_timeout) == ("abort", 600.0)
    assert (legacy.failure_policy, legacy.request_timeout) == ("abort", 180.0)
    assert explicit_dirty.failure_policy == "dirty"


def test_official_environment_is_explicit_and_rejects_deepseek_gasd_r():
    with pytest.raises(ValueError, match="BACKBONE_TAG"):
        run_multiseed._official_settings_from_env(["lad_rg_full"], {})
    with pytest.raises(ValueError, match="GASD-R/Both"):
        run_multiseed._official_settings_from_env(["lad_rg_gasd_r"], _env("deepseek"))


def test_manifest_is_canonical_sanitized_and_required_for_resume(tmp_path):
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://user:pass@127.0.0.1:8000/v1?token=x",
        api_key="secret", revision="a" * 40,
    )
    manifest = run_multiseed._build_official_manifest(
        settings, "qwen32b-r1", "1" * 40, 600.0
    )
    assert manifest["backbone"]["endpoint_origin"] == "http://127.0.0.1:8000"
    assert "secret" not in run_multiseed._canonical_json(manifest)

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    stale = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen32b-r1.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing manifest"):
        run_multiseed._ensure_official_manifest(manifest, pred_dir, "qwen32b-r1")


def test_official_cell_reuses_one_adapter_and_persists_launch_evidence(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen32b-r1")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=2)

    adapters = []

    class Adapter:
        def __init__(self):
            adapters.append(self)

        def ror_reasoner(self, stage, payload):
            raise AssertionError("not called by fake pipeline")

        def gasd_reason_decoder(self, payload):
            raise AssertionError("not called by fake pipeline")

        def provider_metadata(self):
            return {"provider": "vllm", "model": "Qwen/Qwen3-32B-AWQ", "revision": "a" * 40}

    async def pipeline(tokens, dirty, config, dataset_name=None):
        assert config["official"] is True
        assert config["ror_reasoner"].__self__ is adapters[0]
        assert config["gasd_reason_decoder"].__self__ is adapters[0]
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "r",
            "gasd_variant_used": "r",
            "provider_metadata": {"coder": [], "reviewer": [], "ror": [], "gasd": []},
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=2,
        dummy=False, ratio=0.15, official=True, failure_policy="abort",
        request_timeout=10.0, adapter_factory=Adapter,
    ))

    assert len(adapters) == 1
    rows = run_multiseed._load_noisy(
        pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen32b-r1.jsonl"
    )
    assert len(rows) == 2
    assert rows[0]["gasd_variant_used"] == "r"
    assert rows[0]["fallback_used"] is False


def test_official_failure_removes_only_exact_cell_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "tag")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl")
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__tag.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("partial", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_full__msra__BT__tag.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(lambda: {})

    async def broken(*args, **kwargs):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (broken, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort",
            request_timeout=10.0, adapter_factory=Adapter,
        ))

    assert not target.exists()
    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_official_gasd_disabled_cell_accepts_disabled_variant_evidence(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen32b-g-off")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl")

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(lambda: {"provider": "vllm"})

    async def pipeline(tokens, dirty, config, dataset_name=None):
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "disabled",
            "provider_metadata": {"coder": [], "reviewer": [], "ror": [], "gasd": []},
            "fallback_used": False,
        }

    asyncio.run(run_multiseed._run_one_cell(
        "lad_rg_no_gasd", run_multiseed.CONFIGURATIONS["lad_rg_no_gasd"],
        "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=1,
        dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
        adapter_factory=Adapter,
    ))

    output = pred_dir / "pred_seed13__lad_rg_no_gasd__msra__BT__qwen32b-g-off.jsonl"
    assert run_multiseed._load_noisy(output)[0]["gasd_variant_used"] == "disabled"
