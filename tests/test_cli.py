import json

from context_audit.cli import main


def test_live_without_opt_in_fails_before_loading_real_data(capsys):
    assert main(["run", "--config", "missing.yaml", "--max-cost-usd", "2"]) == 2
    assert "--live" in capsys.readouterr().err


def test_live_without_key_has_no_fixture_fallback(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("context_audit.cli.load_dotenv", lambda **_: None)
    assert main(["run", "--config", "configs/pilot.yaml", "--live", "--max-cost-usd", "2"]) == 2
    assert "ANTHROPIC_API_KEY is missing" in capsys.readouterr().err


def test_demo_makes_no_api_calls_and_does_not_export_scores(capsys):
    assert main(["demo"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data_origin"] == "synthetic_fixture"
    assert result["generation_calls"] == 0
    assert not result["empirical_scores_generated"]
    assert set(result["measured_tokens"]) == {
        "full",
        "head_tail",
        "free_summary",
        "structured_summary",
    }
