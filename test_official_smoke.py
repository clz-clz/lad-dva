import json

import official_smoke


def _identity(provider="vllm"):
    revision = "a" * 40 if provider == "vllm" else None
    model = "Qwen/Qwen3-32B-AWQ" if provider == "vllm" else "deepseek-v4-flash"
    served_model = f"{model}@{revision}" if revision else model
    return {
        "provider": provider,
        "model": model,
        "served_model": served_model,
        "revision": revision,
    }


def _row(*, variant="r", candidates=True, fallback=False, legal=True):
    identity = _identity()
    tags = ["B-PER", "O"] if legal else ["I-PER", "O"]
    record = {
        "stage": "gasd_r",
        "status": "live",
        **identity,
        "response_model": identity["served_model"],
        "system_fingerprint": "fp-test",
        "usage": {},
    }
    row = {
        "tokens": ["Alice", "works"],
        "gold_tags": ["B-PER", "O"],
        "pred_tags": tags,
        "ror_reasoning_source": "not_triggered",
        "gasd_variant_requested": variant,
        "gasd_variant_used": variant,
        "provider_metadata": {"gasd": [record]},
        "fallback_used": fallback,
    }
    if candidates:
        row.update({
            "candidate_paths": [["B-PER", "O"]],
            "rag_weights": [1.0],
            "confidence": [1.0, 1.0],
        })
    return row


def test_smoke_gate_requires_20_rows_candidates_live_gasd_r_and_zero_ser():
    report = official_smoke.assess_smoke_rows(
        [_row() for _ in range(official_smoke.PAID_SMOKE_SIZE)],
        config_name="lad_rg_gasd_r",
    )

    assert report["ok"] is True
    assert report["fallback_count"] == 0
    assert report["candidate_evidence_count"] == official_smoke.PAID_SMOKE_SIZE
    assert report["live_gasd_r_count"] == official_smoke.PAID_SMOKE_SIZE
    assert report["gasd_ser"] == 0.0


def test_smoke_gate_reports_every_launch_blocker():
    rows = [_row() for _ in range(official_smoke.PAID_SMOKE_SIZE - 1)]
    rows[0] = _row(candidates=False, fallback=True, legal=False)

    report = official_smoke.assess_smoke_rows(rows, config_name="lad_rg_gasd_r")

    assert report["ok"] is False
    codes = {item["code"] for item in report["blockers"]}
    assert codes == {"record_count", "fallback_used", "candidate_evidence", "gasd_ser"}


def test_deepseek_g_smoke_does_not_require_live_gasd_r():
    rows = [_row(variant="g") for _ in range(official_smoke.PAID_SMOKE_SIZE)]
    for row in rows:
        row["provider_metadata"]["gasd"] = [{
            "stage": "gasd", "status": "local", **_identity("deepseek"),
            "response_model": None, "system_fingerprint": None, "usage": {},
        }]

    report = official_smoke.assess_smoke_rows(rows, config_name="lad_rg_full")

    assert report["ok"] is True
    assert report["live_gasd_r_count"] == 0


def test_smoke_gate_independently_rejects_wrong_gasd_variant():
    rows = [_row() for _ in range(official_smoke.PAID_SMOKE_SIZE)]
    rows[0]["gasd_variant_used"] = "g_fallback"

    report = official_smoke.assess_smoke_rows(
        rows, config_name="lad_rg_gasd_r"
    )

    assert report["ok"] is False
    assert report["gasd_variant_evidence_count"] == official_smoke.PAID_SMOKE_SIZE - 1
    assert {item["code"] for item in report["blockers"]} == {"gasd_variant"}


def test_missing_environment_fails_before_any_live_work(capsys, monkeypatch, tmp_path):
    for name in (
        "BACKBONE_PROVIDER", "BACKBONE_MODEL", "BACKBONE_BASE_URL",
        "BACKBONE_API_KEY", "DEEPSEEK_API_KEY", "BACKBONE_TAG",
        "BACKBONE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)

    code = official_smoke.main([
        "--config", "lad_rg_full", "--dataset", "msra", "--noise", "BT",
        "--seed", "13", "--output-root", str(tmp_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["blockers"][0]["code"] == "smoke_failed"
    assert not list(tmp_path.iterdir())


def test_smoke_cli_keeps_stdout_machine_readable(capsys, monkeypatch, tmp_path):
    environment = {
        "BACKBONE_PROVIDER": "deepseek",
        "BACKBONE_MODEL": "deepseek-v4-flash",
        "BACKBONE_BASE_URL": "https://api.deepseek.com/v1",
        "BACKBONE_API_KEY": "offline-key",
        "BACKBONE_TAG": "deepseek-output-test",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("BACKBONE_REVISION", raising=False)
    monkeypatch.setenv("BACKBONE_SERVED_MODEL", "")
    monkeypatch.setenv("LAD_RG_OFFICIAL_REQUESTS", "0")
    monkeypatch.setattr(official_smoke.runner, "_import_pipeline", lambda _dummy: object())
    monkeypatch.setattr(official_smoke, "_git_sha", lambda: "1" * 40)

    async def fake_run(*args, **kwargs):
        print("pipeline chatter")
        rows = [_row(variant="g") for _ in range(official_smoke.PAID_SMOKE_SIZE)]
        output = kwargs["prediction_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    monkeypatch.setattr(official_smoke.runner, "_run_one_cell", fake_run)

    code = official_smoke.main([
        "--config", "lad_rg_full", "--dataset", "msra", "--noise", "BT",
        "--seed", "13", "--output-root", str(tmp_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["ok"] is True
    assert "pipeline chatter" not in captured.out
    assert "pipeline chatter" in captured.err


def test_contextual_smoke_selector_keeps_fixed_and_length_representative_rows():
    cells = {}
    for dataset in official_smoke.runner.DATASETS:
        for noise in official_smoke.runner.NOISE_TYPES:
            cells[(dataset, noise, 13)] = [
                {"tokens": ["x"] * length} for length in range(1, 201)
            ]

    selected = official_smoke.select_contextual_smoke_rows(cells)

    assert {("msra", "BT", 13, 125), ("msra", "BT", 13, 133),
            ("msra", "ATF", 13, 2)} <= set(selected)
    for key in cells:
        representatives = [entry for entry in selected if entry[:3] == key]
        assert {entry[3] for entry in representatives} >= {99, 189, 199}


def test_contextual_smoke_main_wires_selected_rows_into_production_runner(
    capsys, monkeypatch, tmp_path,
):
    environment = {
        "BACKBONE_PROVIDER": "vllm",
        "BACKBONE_MODEL": "Qwen/Qwen3-32B-AWQ",
        "BACKBONE_BASE_URL": "http://127.0.0.1:8000/v1",
        "BACKBONE_API_KEY": "offline-key",
        "BACKBONE_TAG": "qwen-contextual-smoke-test",
        "BACKBONE_REVISION": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(official_smoke, "_contextual_smoke_plan", lambda: [
        ("msra", "BT", 13, 125), ("msra", "BT", 13, 199),
    ])
    monkeypatch.setattr(official_smoke.runner, "_import_pipeline", lambda _dummy: object())
    monkeypatch.setattr(official_smoke, "_git_sha", lambda: "1" * 40)

    class Factory:
        def close(self):
            pass

    monkeypatch.setattr(official_smoke.runner, "_CachedAdapterFactory", lambda _factory: Factory())
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(kwargs["row_indices"])
        row = {
            "tokens": ["Alice", "works"], "gold_tags": ["B-PER", "O"],
            "pred_tags": ["B-PER", "O"], "fallback_used": False,
            "candidate_paths": [["B-PER", "O"]], "rag_weights": [1.0],
            "confidence": [1.0, 1.0],
        }
        output = kwargs["prediction_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row) + "\n" for _ in kwargs["row_indices"]),
            encoding="utf-8",
        )

    monkeypatch.setattr(official_smoke.runner, "_run_one_cell", fake_run)

    code = official_smoke.main([
        "--config", "selectdenoise_contextual_lattice", "--output-root", str(tmp_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert calls == [[125, 199]]
    assert payload["cell"]["smoke_records"] == 2
