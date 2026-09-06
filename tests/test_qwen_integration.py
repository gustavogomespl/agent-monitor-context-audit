"""Independent CPU regressions for backend dispatch, resume and GPU accounting."""

import json
from pathlib import Path

import pytest

from context_audit.runtime_models import AuditConfig, QwenConfig


def test_qwen_config_is_exact_model_but_templates_do_not_authorize_live():
    config = AuditConfig(
        provider="qwen_local",
        qwen=QwenConfig(),
        monitor_model="Qwen/Qwen3.8-27B",
        summarizer_model="Qwen/Qwen3.8-27B",
    )
    with pytest.raises(ValueError, match="revision"):
        config.qwen.validate_live()
    with pytest.raises(ValueError, match="both roles"):
        AuditConfig(provider="qwen_local", qwen=QwenConfig(), monitor_model="other")


@pytest.mark.parametrize(
    "url", ["https://example.org", "http://127.0.0.1:8000/v1", "http://user@localhost:8000"]
)
def test_remote_or_ambiguous_qwen_endpoints_rejected(url):
    with pytest.raises(ValueError, match="loopback"):
        QwenConfig(base_url=url)


def test_cli_returns_failure_for_partial_live_run(monkeypatch, capsys):
    import context_audit.cli as cli
    import context_audit.dataset as dataset
    import context_audit.runner as runner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "independent-not-a-key")
    monkeypatch.setattr(runner, "load_config", lambda path: AuditConfig())
    monkeypatch.setattr(dataset, "load_dataset", lambda *args: ([], []))
    monkeypatch.setattr(dataset, "load_canaries", lambda *args: ["synthetic notice"])
    monkeypatch.setattr(cli, "_dataset_manifest", lambda cfg: {})
    monkeypatch.setattr(
        runner, "run_experiment", lambda *args, **kw: {"status": "partial_or_failed"}
    )
    assert cli.main(["run", "--config", "unused", "--live", "--max-cost-usd", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partial_or_failed"


def test_resume_does_not_replace_complete_rows_with_replayed_prefix(tmp_path):
    from context_audit.runner import merge_score_rows

    old = [
        dict(transcript_id="t1", condition=c, repetition=0, value=i)
        for i, c in enumerate(("full", "head_tail", "free_summary", "structured_summary"))
    ]
    updated = merge_score_rows(old, [dict(old[0], value=99)])
    assert len(updated) == 4
    assert updated[0]["value"] == 99
    assert updated[1:] == old[1:]


def test_gpu_overhead_included_without_breaking_legacy_rows():
    from test_metrics import numeric_rows

    from context_audit.metrics import validate_rows

    rows = numeric_rows()
    assert validate_rows(rows).infrastructure_cost_usd.eq(0).all()
    for row in rows:
        row["infrastructure_cost_usd"] = 0.25
        row["cost_usd"] += 0.25
    assert validate_rows(rows).infrastructure_cost_usd.sum() == 0.25 * len(rows)
    rows[0]["cost_usd"] -= 0.25
    with pytest.raises(ValueError, match="cost"):
        validate_rows(rows)


def test_private_run_path_accepts_ignored_drive_symlink(tmp_path, monkeypatch):
    import subprocess

    from context_audit.runner import require_private_path

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("runs/private\n")
    (repo / "runs").mkdir()
    durable = tmp_path / "durable"
    durable.mkdir()
    (repo / "runs/private").symlink_to(durable, target_is_directory=True)
    monkeypatch.chdir(repo)
    require_private_path(Path("runs/private/pilot"), Path("runs/private"))
    with pytest.raises(ValueError):
        require_private_path(Path("elsewhere/private"), Path("runs/private"))


def test_qwen_runner_four_conditions_retries_and_resume_over_http(tmp_path, monkeypatch):
    import httpx
    import pandas as pd

    import context_audit.dataset as dataset
    import context_audit.qwen_provider as qwen_module
    import context_audit.runner as runner
    from context_audit.schemas import EvaluationLabel

    model = "Qwen/Qwen3.8-27B"
    cfg = AuditConfig(
        provider="qwen_local",
        qwen=QwenConfig(model_revision="a" * 40, gpu_hourly_rate_usd=2),
        monitor_model=model,
        summarizer_model=model,
        pilot_pairs=None,
        run_dir=str(tmp_path / "private"),
        repetitions=2,
    )
    source = dataset.parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"),
        "t_" + "a" * 24,
        data_origin="synthetic_fixture",
    )
    inputs = [source, source.model_copy(update={"transcript_id": "t_" + "b" * 24})]
    labels = {
        t.transcript_id: EvaluationLabel(
            transcript_id=t.transcript_id,
            label=i,
            scenario_id="s_" + "1" * 24,
            family_id="f_" + "1" * 24,
            split="development",
        )
        for i, t in enumerate(inputs)
    }
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: (inputs, list(labels.values())))
    monkeypatch.setattr(runner, "code_state", lambda: {"commit": "1" * 40, "hash": "code"})
    generated = []
    first_invalid = True

    def count(payload):
        if "messages" in payload:
            return 10 + sum((len(m["content"].encode()) + 3) // 4 for m in payload["messages"])
        return (len(payload["prompt"].encode()) + 3) // 4

    def serve(request):
        nonlocal first_invalid
        payload = json.loads(request.content) if request.content else {}
        if request.url.path == "/version":
            result = {"version": cfg.qwen.vllm_version}
        elif request.url.path == "/v1/models":
            result = {"data": [{"id": model, "max_model_len": cfg.qwen.max_model_len}]}
        elif request.url.path == "/tokenize":
            tokens = count(payload)
            result = {
                "count": tokens,
                "tokens": list(range(tokens)),
                "max_model_len": cfg.qwen.max_model_len,
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            assert "tools" not in payload
            assert payload["chat_template_kwargs"] == {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
            generated.append(payload)
            system = payload["messages"][0]["content"]
            if "suspicion_score" in system:
                if first_invalid:
                    text, first_invalid = "invalid independent fixture", False
                else:
                    text = json.dumps(
                        dict(
                            suspicion_score=60,
                            brief_justification="Observed event.",
                            evidence_event_ids=[],
                            uncertainties=[],
                        )
                    )
            elif "environment_and_state" in system:
                text = json.dumps(
                    dict(
                        environment_and_state=[],
                        actions_and_observed_results=["Observed action [E0001]."],
                        important_identifiers=[],
                        contradictions_or_missing_information=[],
                        source_event_ids=["E0001"],
                    )
                )
            else:
                text = "Observed a visible action [E0001]."
            tokens = count(payload)
            result = dict(
                model=model,
                choices=[
                    dict(
                        message=dict(role="assistant", content=text),
                        finish_reason="stop",
                    )
                ],
                usage=dict(prompt_tokens=tokens, completion_tokens=40, total_tokens=tokens + 40),
            )
        return httpx.Response(200, json=result)

    real_client = httpx.Client
    monkeypatch.setattr(
        qwen_module.httpx,
        "Client",
        lambda **kw: real_client(
            **kw,
            transport=httpx.MockTransport(serve),
        ),
    )
    args = (
        cfg,
        inputs,
        labels,
        {"dataset_commit": "1" * 40},
        {"code_hash": "code", "prompts": {}},
        2,
        ["Independent synthetic notice"],
        {},
        {},
    )
    first = runner._run_locked(*args)
    assert first["successful_rows"] == first["expected_rows"] == 16
    assert len(generated) == 25  # Four monitor + two summary calls, two repeats, one repair.
    scores = pd.read_csv(Path(cfg.run_dir) / "scores.csv")
    second = runner._run_locked(*args)
    assert second["generation_requests"] == first["generation_requests"]
    assert len(generated) == 25
    assert pd.read_csv(Path(cfg.run_dir) / "scores.csv").equals(scores)
