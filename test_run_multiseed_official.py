import asyncio
import json
from pathlib import Path

import pytest

import multi_agent_v2
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


def _identity():
    revision = "a" * 40
    return {
        "provider": "vllm",
        "model": "Qwen/Qwen3-32B-AWQ",
        "served_model": f"Qwen/Qwen3-32B-AWQ@{revision}",
        "revision": revision,
    }


def _live_record(stage):
    identity = _identity()
    return {
        "stage": stage,
        "status": "live",
        **identity,
        "response_model": identity["served_model"],
        "system_fingerprint": "fp-test",
        "usage": {},
    }


def _status_record(stage, status):
    return {
        "stage": stage,
        "status": status,
        **_identity(),
        "response_model": None,
        "system_fingerprint": None,
        "usage": {},
    }


def _full_evidence(*, ror_status="not_triggered", gasd_status="live",
                   coder_stages=(1, 2, 5)):
    return {
        "coder": [_live_record(f"coder_path_{index}") for index in coder_stages],
        "reviewer": [_live_record("reviewer")],
        "ror": [_status_record("ror", ror_status)],
        "gasd": ([_live_record("gasd_r")] if gasd_status == "live"
                 else [_status_record("gasd", gasd_status)]),
    }


def _write_noisy(path: Path, count=1):
    row = {
        "tokens": ["Alice", "works"],
        "ner_tags": ["B-PER", "O"],
        "dirty_tags": ["B-PER", "O"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8")


def _write_official_prediction(path: Path, *, evidence=None, count=200):
    row = {
        "tokens": ["Alice", "works"],
        "gold_tags": ["B-PER", "O"],
        "pred_tags": ["B-PER", "O"],
        "ror_reasoning_source": "not_triggered",
        "gasd_variant_requested": "r",
        "gasd_variant_used": "r",
        "provider_metadata": evidence or _full_evidence(),
        "fallback_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("noise_type", "expected_stages"),
    [
        ("BT", ["coder_path_1", "coder_path_2", "coder_path_5"]),
        ("IF", ["coder_path_1", "coder_path_2", "coder_path_5"]),
        ("ATF", [f"coder_path_{index}" for index in range(1, 6)]),
    ],
)
def test_real_coder_evidence_validates_exact_noise_stage_set(
    monkeypatch, noise_type, expected_stages
):
    identity = _identity()

    class Response:
        content = '["B-PER", "O"]'
        response_metadata = {
            "model_name": identity["served_model"],
            "system_fingerprint": "fp-test",
            "token_usage": {},
        }
        usage_metadata = None

    class CoderLLM:
        @staticmethod
        def invoke(_prompt):
            return Response()

    monkeypatch.setattr(multi_agent_v2, "coder_llm", CoderLLM())
    monkeypatch.setattr(multi_agent_v2, "_get_deer_examples", lambda *args, **kwargs: [])
    result = asyncio.run(multi_agent_v2.coder_node({
        "tokens": ["Alice", "works"],
        "dirty_tags": ["B-PER", "O"],
        "dataset_name": "msra",
        "noise_type": noise_type,
        "official": True,
        "provider_settings": identity,
        "provider_metadata": {},
    }))
    evidence = _full_evidence()
    evidence["coder"] = result["provider_metadata"]["coder"]

    run_multiseed._validate_official_provider_evidence(
        run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
        evidence,
        identity,
        "not_triggered",
        noise_type=noise_type,
    )

    assert [record["stage"] for record in evidence["coder"]] == expected_stages


@pytest.mark.parametrize(
    "coder_records",
    [
        [_live_record("coder_path_1"), _live_record("coder_path_2")],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _live_record("coder_path_5"), _live_record("coder_path_5"),
        ],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _live_record("coder_path_3"), _live_record("coder_path_5"),
        ],
        [
            _live_record("coder_path_1"), _live_record("coder_path_2"),
            _status_record("coder_path_5", "skipped"),
        ],
    ],
    ids=("missing", "duplicate", "extra", "non-live"),
)
def test_official_coder_evidence_rejects_invalid_bt_stage_sets(coder_records):
    evidence = _full_evidence()
    evidence["coder"] = coder_records

    with pytest.raises(RuntimeError, match="[Cc]oder"):
        run_multiseed._validate_official_provider_evidence(
            run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            evidence,
            _identity(),
            "not_triggered",
            noise_type="BT",
        )


def test_official_coder_evidence_rejects_unsupported_noise_type():
    with pytest.raises(RuntimeError, match="unsupported official noise type"):
        run_multiseed._validate_official_provider_evidence(
            run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            _full_evidence(),
            _identity(),
            "not_triggered",
            noise_type="OTHER",
        )


@pytest.mark.parametrize("record_count", [199, 201])
def test_official_cell_rejects_non_200_input_before_adapter_or_calls(
    tmp_path, monkeypatch, record_count
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-count")
    _write_noisy(
        noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=record_count
    )
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-count.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")

    def adapter_factory():
        raise AssertionError("adapter must not be constructed for a non-200 cell")

    async def pipeline(*args, **kwargs):
        raise AssertionError("pipeline must not run for a non-200 cell")

    with pytest.raises(RuntimeError, match="exactly 200"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort",
            request_timeout=10.0, adapter_factory=adapter_factory,
        ))

    assert not target_tmp.exists()


def test_official_adapter_construction_failure_cleans_exact_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-adapter-fail")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-adapter-fail.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="adapter failed"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(RuntimeError("adapter failed")),
        ))

    assert not target_tmp.exists()


def test_official_noisy_load_failure_cleans_only_exact_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-load-fail")
    noisy_path = noisy_dir / "noisy_seed13__BT__msra__N200.jsonl"
    noisy_path.parent.mkdir(parents=True)
    noisy_path.write_text("not-json\n", encoding="utf-8")
    pred_dir.mkdir()
    target = pred_dir / "pred_seed13__lad_rg_full__msra__BT__qwen-load-fail.jsonl"
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_full__msra__BT__qwen-load-fail.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_full", run_multiseed.CONFIGURATIONS["lad_rg_full"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("adapter must not be constructed after a noisy-load failure")
            ),
        ))

    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_official_resume_rejects_non_200_input_before_adapter(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-resume-count")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=199)
    _write_official_prediction(
        pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen-resume-count.jsonl"
    )

    with pytest.raises(RuntimeError, match="exactly 200"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("adapter must not be constructed for a non-200 resume")
            ),
        ))


def test_official_resume_rejects_mislabelled_stage_evidence_and_cleans_exact_temp(
    tmp_path, monkeypatch
):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-resume-evidence")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
    evidence = _full_evidence()
    evidence["coder"][0]["stage"] = "gasd_r"
    target = pred_dir / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen-resume-evidence.jsonl"
    _write_official_prediction(target, evidence=evidence)
    target_tmp = target.with_suffix(".jsonl.tmp")
    target_tmp.write_text("stale", encoding="utf-8")
    other_tmp = pred_dir / "pred_seed42__lad_rg_gasd_r__msra__BT__qwen-resume-evidence.jsonl.tmp"
    other_tmp.write_text("keep", encoding="utf-8")

    class Adapter:
        provider_metadata = staticmethod(_identity)

    with pytest.raises(RuntimeError, match="stage"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (lambda *args: None, {}), max_concurrency=1,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=Adapter,
        ))

    assert not target_tmp.exists()
    assert other_tmp.read_text(encoding="utf-8") == "keep"


def test_cli_defaults_abort_and_require_explicit_legacy_dirty():
    official = run_multiseed._parse_args(["--official", "--size", "200"])
    legacy = run_multiseed._parse_args([])
    explicit_dirty = run_multiseed._parse_args(["--failure-policy", "dirty"])

    assert (official.failure_policy, official.request_timeout) == ("abort", 600.0)
    assert (legacy.failure_policy, legacy.request_timeout) == ("abort", 180.0)
    assert explicit_dirty.failure_policy == "dirty"


def test_official_request_model_configuration_uses_immutable_served_name(monkeypatch):
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision="a" * 40,
    )
    monkeypatch.delenv("BACKBONE_SERVED_MODEL", raising=False)

    assert run_multiseed._configure_official_request_model(settings) == settings.served_model
    assert run_multiseed.os.environ["BACKBONE_SERVED_MODEL"] == settings.served_model


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
    assert manifest["backbone"]["served_model"] == settings.served_model
    assert "secret" not in run_multiseed._canonical_json(manifest)
    assert manifest["decoder_constants"] == {
        "contract_version": "lad-rg-official-decoder-v1",
        "hard_constraint": "strict-iob2-v1",
        "provider_timeout_seconds": 120.0,
        "sdk_max_retries": 2,
        "runner_request_timeout_seconds": 600.0,
        "gasd_beta_omega": 2.0,
        "gasd_gamma_proposal": 1.0,
        "gasd_candidate_scale_g": 1.0,
        "gasd_candidate_scale_r": 0.0,
        "gasd_reason_bonus": 2.0,
        "ror_omega_quantile": 0.6,
        "ror_confidence_threshold": 0.6,
        "voting_entity_boost": 1.0,
        "voting_consensus_ratio": 0.6,
    }

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
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    adapters = []

    class Adapter:
        def __init__(self):
            adapters.append(self)

        def ror_reasoner(self, stage, payload):
            raise AssertionError("not called by fake pipeline")

        def gasd_reason_decoder(self, payload):
            raise AssertionError("not called by fake pipeline")

        def provider_metadata(self):
            return _identity()

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
            "provider_metadata": _full_evidence(),
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
    assert len(rows) == 200
    assert rows[0]["gasd_variant_used"] == "r"
    assert rows[0]["fallback_used"] is False


def test_official_failure_removes_only_exact_cell_temp(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "tag")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)
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


def test_official_cell_rejects_empty_stage_evidence(tmp_path, monkeypatch):
    from tqdm.asyncio import tqdm as async_tqdm

    async def forbidden_progress_gather(*args, **kwargs):
        raise AssertionError("official failures must not use orphan-prone tqdm gather")

    monkeypatch.setattr(async_tqdm, "gather", forbidden_progress_gather)
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen-empty-evidence")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(_identity)

    async def pipeline(tokens, dirty, config, dataset_name=None):
        return {
            "pred_tags": list(dirty), "candidate_paths": [list(dirty)],
            "rag_weights": [1.0], "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "r", "gasd_variant_used": "r",
            "provider_metadata": {stage: [] for stage in ("coder", "reviewer", "ror", "gasd")},
            "fallback_used": False,
        }

    with pytest.raises(RuntimeError, match="provider evidence"):
        asyncio.run(run_multiseed._run_one_cell(
            "lad_rg_gasd_r", run_multiseed.CONFIGURATIONS["lad_rg_gasd_r"],
            "msra", "BT", 13, 200, (pipeline, {}), max_concurrency=5,
            dummy=False, official=True, failure_policy="abort", request_timeout=10.0,
            adapter_factory=Adapter,
        ))


def test_official_gasd_disabled_cell_accepts_disabled_variant_evidence(tmp_path, monkeypatch):
    noisy_dir = tmp_path / "noisy"
    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(run_multiseed, "NOISY_DIR", noisy_dir)
    monkeypatch.setattr(run_multiseed, "PRED_DIR", pred_dir)
    monkeypatch.setattr(run_multiseed, "BACKBONE_TAG", "qwen32b-g-off")
    _write_noisy(noisy_dir / "noisy_seed13__BT__msra__N200.jsonl", count=200)

    class Adapter:
        ror_reasoner = staticmethod(lambda *_: {})
        gasd_reason_decoder = staticmethod(lambda *_: {})
        provider_metadata = staticmethod(_identity)

    async def pipeline(tokens, dirty, config, dataset_name=None):
        return {
            "pred_tags": list(dirty),
            "candidate_paths": [list(dirty)],
            "rag_weights": [1.0],
            "confidence": [1.0] * len(dirty),
            "ror_reasoning_source": "not_triggered",
            "gasd_variant_requested": "g",
            "gasd_variant_used": "disabled",
            "provider_metadata": _full_evidence(gasd_status="disabled"),
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
