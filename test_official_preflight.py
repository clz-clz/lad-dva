import json
import subprocess
from pathlib import Path

import official_preflight
from live_backbone import LiveBackboneResult, LiveBackboneSettings, OpenAICompatibleLADRGAdapter


DATASETS = ["msra", "conll2003", "wnut17", "fewnerd", "ontonotes5"]
NOISE = ["BT", "IF", "ATF"]
SEEDS = [13, 42, 2024]
QWEN_REVISION = "a" * 40
QWEN_SERVED_MODEL = f"Qwen/Qwen3-32B-AWQ@{QWEN_REVISION}"


def _env():
    return {
        "BACKBONE_PROVIDER": "vllm",
        "BACKBONE_MODEL": "Qwen/Qwen3-32B-AWQ",
        "BACKBONE_BASE_URL": "http://127.0.0.1:8000/v1",
        "BACKBONE_API_KEY": "test-key",
        "BACKBONE_TAG": "qwen32b-r1",
        "BACKBONE_REVISION": QWEN_REVISION,
    }


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    return repo, sha


def _noisy_fixture(root: Path):
    row3 = {"tokens": ["Li", "Ming"], "ner_tags": ["B-PER", "I-PER"],
            "dirty_tags": ["B-PER", "O"]}
    row4 = {"tokens": ["New", "York"], "ner_tags": ["B-LOC", "I-LOC"],
            "dirty_tags": ["B-LOC", "O"]}
    root.mkdir()
    for dataset in DATASETS:
        row = row3 if dataset == "msra" else row4
        for noise in NOISE:
            for seed in SEEDS:
                path = root / f"noisy_seed{seed}__{noise}__{dataset}__N200.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for _ in range(200)), encoding="utf-8")


def test_deer_check_is_cache_only_and_has_no_provider_credentials(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setenv("BACKBONE_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setattr(official_preflight.subprocess, "run", fake_run)

    result = official_preflight._one_deer_check(tmp_path, "msra", 3.0)

    assert result == {"ok": True}
    child_env = captured["environment"]
    assert child_env["HF_DATASETS_OFFLINE"] == "1"
    assert child_env["HF_HUB_OFFLINE"] == "1"
    assert child_env["TRANSFORMERS_OFFLINE"] == "1"
    assert child_env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert child_env["BACKBONE_API_KEY"] == "offline-deer-preflight"
    assert child_env["BACKBONE_MODEL"] == "offline-deer-preflight"
    assert child_env["BACKBONE_BASE_URL"] == "http://127.0.0.1:9/v1"
    for secret_name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert secret_name not in child_env


def test_deer_check_fails_closed_on_timeout(tmp_path, monkeypatch):
    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(official_preflight.subprocess, "run", time_out)

    assert official_preflight._one_deer_check(tmp_path, "fewnerd", 2.5) == {
        "ok": False, "error": "timed out after 2.5 seconds",
    }


def test_deer_check_fails_closed_when_offline_cache_is_missing(tmp_path, monkeypatch):
    def cache_miss(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="offline cache entry not found"
        )

    monkeypatch.setattr(official_preflight.subprocess, "run", cache_miss)

    assert official_preflight._one_deer_check(tmp_path, "wnut17", 3.0) == {
        "ok": False, "error": "offline cache entry not found",
    }


def test_static_preflight_accepts_complete_fixture_and_writes_no_predictions(tmp_path):
    repo, sha = _repo(tmp_path)
    noisy = tmp_path / "noisy"
    pred = tmp_path / "pred"
    pred.mkdir()
    _noisy_fixture(noisy)

    report = official_preflight.run_static_preflight(
        repo_root=repo, expected_sha=sha, noisy_dir=noisy, pred_dir=pred,
        environment=_env(), deer_timeout=3.0,
        deer_checker=lambda datasets, timeout: {
            name: {"ok": True, "duration_seconds": 0.01} for name in datasets
        },
    )

    assert report["ok"] is True
    assert report["checks"]["noisy_files"]["files"] == 45
    assert report["checks"]["noisy_files"]["records"] == 9000
    assert list(pred.iterdir()) == []


def test_static_preflight_blocks_dirty_tracked_tree_and_illegal_gold(tmp_path):
    repo, sha = _repo(tmp_path)
    noisy = tmp_path / "noisy"
    pred = tmp_path / "pred"
    pred.mkdir()
    _noisy_fixture(noisy)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    bad = noisy / "noisy_seed13__BT__msra__N200.jsonl"
    rows = bad.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["ner_tags"] = ["I-PER", "O"]
    rows[0] = json.dumps(first)
    bad.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = official_preflight.run_static_preflight(
        repo_root=repo, expected_sha=sha, noisy_dir=noisy, pred_dir=pred,
        environment=_env(), deer_timeout=3.0,
        deer_checker=lambda datasets, timeout: {name: {"ok": True} for name in datasets},
    )

    assert report["ok"] is False
    codes = {item["code"] for item in report["blockers"]}
    assert {"git_dirty", "invalid_noisy_record"} <= codes


def test_static_preflight_validates_the_intended_official_configs(tmp_path):
    repo, sha = _repo(tmp_path)
    noisy = tmp_path / "noisy"
    pred = tmp_path / "pred"
    pred.mkdir()
    _noisy_fixture(noisy)
    environment = {
        "BACKBONE_PROVIDER": "deepseek",
        "BACKBONE_MODEL": "deepseek-v4-flash",
        "BACKBONE_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY": "test-key",
        "BACKBONE_TAG": "deepseek-r-invalid",
    }

    report = official_preflight.run_static_preflight(
        repo_root=repo, expected_sha=sha, noisy_dir=noisy, pred_dir=pred,
        environment=environment, config_names=["lad_rg_gasd_r"], deer_timeout=3.0,
        deer_checker=lambda datasets, timeout: {name: {"ok": True} for name in datasets},
    )

    assert report["ok"] is False
    assert any(item["code"] == "invalid_configuration" for item in report["blockers"])


def test_static_preflight_rejects_off_protocol_tagged_artifacts(tmp_path):
    repo, sha = _repo(tmp_path)
    noisy = tmp_path / "noisy"
    pred = tmp_path / "pred"
    pred.mkdir()
    _noisy_fixture(noisy)
    artifacts = [
        pred / "pred_seed99__lad_rg_full__msra__BT__qwen32b-r1.jsonl",
        pred / "pred_seed13__lad_rg_full__msra__XX__qwen32b-r1.jsonl",
        pred / "pred_seed13__lad_rg_gasd_r__msra__BT__qwen32b-r1.jsonl",
        pred / "pred_seed13__lad_rg_full__msra__BT__extra__qwen32b-r1.jsonl",
        pred / "pred_seed99__lad_rg_full__msra__BT__qwen32b-r1.tmp",
        pred / "pred_seed13__lad_rg_full__msra__BT__extra__qwen32b-r1.backup",
        pred / "pred_seed13__lad_rg_full__msra__BT__qwen32b-r1__extra.bin",
    ]
    for artifact in artifacts:
        artifact.write_text("{}\n", encoding="utf-8")
    unrelated_tag = pred / "pred_seed13__lad_rg_full__msra__BT__qwen32b-r1-other.jsonl"
    unrelated_tag.write_text("{}\n", encoding="utf-8")

    report = official_preflight.run_static_preflight(
        repo_root=repo, expected_sha=sha, noisy_dir=noisy, pred_dir=pred,
        environment=_env(), config_names=["lad_rg_full"], deer_timeout=3.0,
        deer_checker=lambda datasets, timeout: {name: {"ok": True} for name in datasets},
    )

    assert report["ok"] is False
    rejected = next(
        item for item in report["blockers"]
        if item["code"] == "noncanonical_prediction_artifact"
    )
    assert set(rejected["files"]) == {artifact.name for artifact in artifacts}
    assert unrelated_tag.name not in rejected["files"]


def test_tagged_artifact_scan_keeps_dotted_tag_exact_and_ignores_longer_tag(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    tag = "qwen.32b-r1"
    exact_tag = pred / f"pred_seed99__lad_rg_full__msra__BT__{tag}.tmp"
    exact_tag.write_text("{}\n", encoding="utf-8")
    longer_tag = pred / f"pred_seed99__lad_rg_full__msra__BT__{tag}-other.tmp"
    longer_tag.write_text("{}\n", encoding="utf-8")
    blockers = []
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    official_preflight._check_tagged_artifacts(
        pred, settings, tag, "1" * 40, 600.0, blockers, ["lad_rg_full"]
    )

    rejected = next(
        item for item in blockers
        if item["code"] == "noncanonical_prediction_artifact"
    )
    assert rejected["files"] == [exact_tag.name]
    assert longer_tag.name not in rejected["files"]


def _response(payload, fingerprint="fp-test"):
    return {
        "model": QWEN_SERVED_MODEL,
        "system_fingerprint": fingerprint,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }


def test_live_preflight_uses_real_adapter_validation_with_mocked_transport():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    def transport(**request):
        name = request["response_format"]["json_schema"]["name"]
        if name == "lad_rg_span_detection":
            return _response({"spans": [{"start": 0, "end": 1}]})
        if name == "lad_rg_type_assignment":
            return _response({"types": [{"start": 0, "end": 1, "type": "PER"}]})
        return _response({"reason": "consistent", "tags": ["B-PER", "O", "B-LOC"]})

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": QWEN_SERVED_MODEL}],
        adapter_factory=lambda configured: OpenAICompatibleLADRGAdapter(
            configured, transport=transport
        ),
    )

    assert report["ok"] is True
    assert report["checks"]["gasd_r"]["tag_count"] == 3
    assert report["checks"]["provider_metadata"]["system_fingerprints"] == ["fp-test"]


def test_live_preflight_blocks_qwen_gasd_length_error():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    def transport(**request):
        name = request["response_format"]["json_schema"]["name"]
        if name == "lad_rg_span_detection":
            return _response({"spans": []})
        if name == "lad_rg_type_assignment":
            return _response({"types": []})
        return _response({"reason": "short", "tags": ["O"]})

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": settings.served_model}],
        adapter_factory=lambda configured: OpenAICompatibleLADRGAdapter(
            configured, transport=transport
        ),
    )

    assert report["ok"] is False
    assert any(
        item["code"] == "live_validation_failed"
        and "exactly 3 items" in item["message"]
        for item in report["blockers"]
    )


def test_models_url_always_targets_openai_v1_models_endpoint():
    assert official_preflight._models_url("https://api.deepseek.com") == (
        "https://api.deepseek.com/v1/models"
    )
    assert official_preflight._models_url("http://127.0.0.1:8000/v1/") == (
        "http://127.0.0.1:8000/v1/models"
    )


def test_live_preflight_records_unsupported_fingerprint_without_blocking():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    def transport(**request):
        name = request["response_format"]["json_schema"]["name"]
        if name == "lad_rg_span_detection":
            return _response({"spans": []}, fingerprint=None)
        if name == "lad_rg_type_assignment":
            return _response({"types": []}, fingerprint=None)
        return _response({"reason": "none", "tags": ["O", "O", "O"]}, fingerprint=None)

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": settings.served_model}],
        adapter_factory=lambda configured: OpenAICompatibleLADRGAdapter(
            configured, transport=transport
        ),
    )

    assert report["ok"] is True
    assert report["checks"]["provider_metadata"]["fingerprint_supported"] is False


def test_live_preflight_rejects_bare_vllm_model_advertisement():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": settings.model}],
        adapter_factory=lambda configured: (_ for _ in ()).throw(
            AssertionError("adapter must not be created for a bare model listing")
        ),
    )

    assert report["ok"] is False
    assert any(
        item["code"] == "live_validation_failed"
        and "absent from /v1/models" in item["message"]
        for item in report["blockers"]
    )


def test_live_preflight_rejects_non_official_vllm_model_before_network():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Other-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: (_ for _ in ()).throw(
            AssertionError("network check must not run for a non-official model")
        ),
    )

    assert report["ok"] is False
    assert any("Qwen/Qwen3-32B-AWQ" in item["message"] for item in report["blockers"])


def test_live_preflight_rejects_adapter_identity_before_schema_calls():
    settings = LiveBackboneSettings(
        provider="vllm", model="Qwen/Qwen3-32B-AWQ",
        base_url="http://127.0.0.1:8000/v1", api_key="test",
        revision=QWEN_REVISION,
    )

    class WrongIdentityAdapter:
        def __init__(self, configured):
            self.configured = configured

        def provider_metadata(self):
            return {
                "provider": "vllm", "model": self.configured.model,
                "served_model": self.configured.model,
                "revision": self.configured.revision,
            }

        def ror_reasoner(self, *args):
            raise AssertionError("schema calls must not run")

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": settings.served_model}],
        adapter_factory=WrongIdentityAdapter,
    )

    assert report["ok"] is False
    assert any("adapter identity" in item["message"] for item in report["blockers"])


def test_live_preflight_rejects_deepseek_reason_configs_before_network():
    settings = LiveBackboneSettings(
        provider="deepseek", model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1", api_key="test",
    )

    report = official_preflight.run_live_preflight(
        settings=settings, config_names=["lad_rg_gasd_both"],
        models_fetcher=lambda _: (_ for _ in ()).throw(
            AssertionError("network check must not run for an invalid config/provider pair")
        ),
    )

    assert report["ok"] is False
    assert any("GASD-R/Both" in item["message"] for item in report["blockers"])


def test_deepseek_live_preflight_probes_69_item_responses_schema():
    settings = LiveBackboneSettings(
        provider="deepseek", model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1", api_key="test",
    )
    calls = []
    identity = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "served_model": "deepseek-v4-flash",
        "revision": None,
        "structured_api": "responses-json-schema",
    }

    class Adapter:
        def __init__(self, configured):
            assert configured is settings

        def provider_metadata(self):
            return dict(identity)

        def structured_requester(self, stage, payload):
            calls.append((stage, payload))
            return LiveBackboneResult(
                {"tags": ["O"] * 69},
                {
                    **identity,
                    "stage": stage,
                    "status": "live",
                    "response_model": settings.served_model,
                    "response_status": "completed",
                    "finish_reason": None,
                    "incomplete_reason": None,
                    "system_fingerprint": None,
                    "usage": {},
                },
            )

        def ror_reasoner(self, stage, payload):
            return LiveBackboneResult(
                {"spans": []} if stage == "span_detection" else {"types": []},
                {
                    **identity,
                    "stage": "ror_span_detection" if stage == "span_detection" else "ror_type_assignment",
                    "status": "live",
                    "response_model": settings.served_model,
                    "response_status": "completed",
                    "finish_reason": None,
                    "incomplete_reason": None,
                    "system_fingerprint": None,
                    "usage": {},
                },
            )

        def close(self):
            pass

    report = official_preflight.run_live_preflight(
        settings=settings,
        models_fetcher=lambda _: [{"id": settings.served_model}],
        adapter_factory=Adapter,
    )

    assert report["ok"] is True
    assert calls and calls[0][0] == "capability_probe_69"
    probe_schema = calls[0][1]["schema"]["properties"]["tags"]
    assert probe_schema["minItems"] == probe_schema["maxItems"] == 69
    assert set(probe_schema["items"]["enum"]) == {
        "O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG",
        "B-MISC", "I-MISC",
    }


def test_static_cli_converts_internal_failure_to_json_only(capsys, monkeypatch, tmp_path):
    def broken(**kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(official_preflight, "run_static_preflight", broken)
    exit_code = official_preflight.main([
        "static", "--expected-sha", "1" * 40,
        "--repo-root", str(tmp_path), "--noisy-root", str(tmp_path / "noisy"),
        "--predictions-root", str(tmp_path / "pred"), "--request-timeout", "777",
    ])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["blockers"][0]["code"] == "preflight_internal_error"
