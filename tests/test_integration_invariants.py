"""Independent synthetic integration checks; never load real data or contact APIs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_audit.cli import freeze
from context_audit.runtime_models import AuditConfig
from context_audit.schemas import EvaluationLabel
from context_audit.storage import digest


def freeze_environment(tmp_path, monkeypatch):
    import context_audit.cli as cli
    import context_audit.dataset as dataset
    import context_audit.runner as runner

    config = AuditConfig(
        split="test",
        pilot_pairs=None,
        protocol_version="protocol-v1",
        rubric_reviewed=True,
        data_use_confirmed=True,
        monitor_model="fixture-monitor",
        summarizer_model="fixture-summary",
        protocol_file=str(tmp_path / "protocol.json"),
    )
    labels = [
        EvaluationLabel(
            transcript_id="t_" + str(i) * 24,
            label=i,
            scenario_id="s_" + "1" * 24,
            family_id="f_" + "1" * 24,
            split="development",
        )
        for i in (0, 1)
    ]
    current_dataset = {"dataset_commit": "1" * 40, "normalized_content_digest": "current"}
    signature = dict(
        prompts={"monitor": "fixed-prompt"},
        code_hash="current-code",
        dataset_manifest_hash=digest(current_dataset),
    )
    monkeypatch.setattr(runner, "require_private_path", lambda path, root: None)
    monkeypatch.setattr(runner, "protocol_signature", lambda cfg, data: signature)
    monkeypatch.setattr(cli, "_dataset_manifest", lambda cfg: current_dataset)
    monkeypatch.setattr(dataset, "load_dataset", lambda path, split: ([], labels))
    run = tmp_path / "development"
    (run / "manifests").mkdir(parents=True)
    planned = [
        dict(transcript_id=label.transcript_id, condition=condition, repetition=0)
        for label in labels
        for condition in ("full", "head_tail", "free_summary", "structured_summary")
    ]
    development = dict(
        data_origin="sleight_bench",
        config=config.model_dump() | {"split": "development"},
        planned_calls=planned,
        planned_transcripts=[row.transcript_id for row in labels],
        prompt_hashes=signature["prompts"],
        code_hash=signature["code_hash"],
        dataset_manifest_hash=signature["dataset_manifest_hash"],
        run_id="same-run",
    )
    completion = dict(
        status="executed", rows=8, expected_rows=8, successful_rows=8, run_id="same-run"
    )
    return config, run, development, completion


def write_freeze_inputs(run, development, completion):
    (run / "manifests/run.json").write_text(json.dumps(development))
    (run / "manifests/completion.json").write_text(json.dumps(completion))
    import pandas as pd

    from context_audit.metrics import PUBLIC_COLUMNS

    rows = []
    for item in development["planned_calls"]:
        label = int(item["transcript_id"][-1])
        row = dict.fromkeys(PUBLIC_COLUMNS, 0)
        row.update(
            item,
            label=label,
            scenario_id="s_" + "1" * 24,
            family_id="f_" + "1" * 24,
            split="development",
            data_origin="sleight_bench",
            status="ok",
            suspicion_score=75 if label else 25,
            escalate=bool(label),
            cost_is_upper_bound=False,
            monitor_model="fixture-monitor",
            summarizer_model="fixture-summary",
            body_tokens=500,
            representation_tokens=128,
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(run / "scores.csv", index=False)
    completion["scores_sha256"] = hashlib.sha256((run / "scores.csv").read_bytes()).hexdigest()
    (run / "manifests/completion.json").write_text(json.dumps(completion))


@pytest.mark.parametrize(
    "changed",
    [
        "code",
        "dataset",
        "completion_identity",
        "missing_condition",
        "missing_condition_consistent_counts",
        "missing_scores",
        "failure_in_scores",
        "scores_checksum",
    ],
)
def test_freeze_rejects_development_evidence_that_does_not_cover_current_protocol(
    tmp_path, monkeypatch, changed
):
    config, run, development, completion = freeze_environment(tmp_path, monkeypatch)
    if changed == "code":
        development["code_hash"] = "old-parser-code"
    elif changed == "dataset":
        development["dataset_manifest_hash"] = "old-normalized-content"
    elif changed == "completion_identity":
        completion["run_id"] = "another-run"
    elif changed.startswith("missing_condition"):
        development["planned_calls"] = development["planned_calls"][:-1]
        if changed == "missing_condition_consistent_counts":
            completion.update(rows=7, expected_rows=7, successful_rows=7)
    write_freeze_inputs(run, development, completion)
    if changed == "missing_scores":
        (run / "scores.csv").unlink()
    elif changed == "scores_checksum":
        with (run / "scores.csv").open("a") as handle:
            handle.write("\n")
    elif changed == "failure_in_scores":
        import pandas as pd

        frame = pd.read_csv(run / "scores.csv")
        frame.loc[0, "status"] = "api_error"
        frame.loc[0, "suspicion_score"] = float("nan")
        frame.loc[0, "escalate"] = True
        frame.to_csv(run / "scores.csv", index=False)
        completion["scores_sha256"] = hashlib.sha256((run / "scores.csv").read_bytes()).hexdigest()
        (run / "manifests/completion.json").write_text(json.dumps(completion))
    expected_error = {
        "code": "Code changed",
        "dataset": "Dataset changed",
        "completion_identity": "completion does not match",
        "missing_condition": "every pair, condition",
        "missing_condition_consistent_counts": "every pair, condition",
        "missing_scores": "scores are missing or changed",
        "failure_in_scores": "do not prove complete successful",
        "scores_checksum": "scores are missing or changed",
    }[changed]
    with pytest.raises(ValueError, match=expected_error):
        freeze(config, run)
    assert not Path(config.protocol_file).exists()


def test_freeze_accepts_complete_matching_development_evidence(tmp_path, monkeypatch):
    config, run, development, completion = freeze_environment(tmp_path, monkeypatch)
    write_freeze_inputs(run, development, completion)
    assert freeze(config, run)["status"] == "freeze_prepared"
    assert Path(config.protocol_file).is_file()


def test_export_rejects_labels_from_reinventoried_dataset(tmp_path, monkeypatch):
    import pandas as pd

    import context_audit.cli as cli
    import context_audit.dataset as dataset
    import context_audit.metrics as metrics
    import context_audit.reporting as reporting
    import context_audit.runner as runner

    run = tmp_path / "run"
    (run / "manifests").mkdir(parents=True)
    pd.DataFrame([{"independent_numeric_fixture": 1}]).to_csv(run / "scores.csv", index=False)
    manifest = {
        name: {}
        for name in (
            "dataset_commit",
            "code_commit",
            "code_hash",
            "seeds",
            "prompt_hashes",
            "models",
            "failure_policy",
            "created_at",
            "prices",
            "planned_calls",
            "counter_method",
        )
    }
    manifest.update(
        data_origin="sleight_bench",
        dataset_manifest_hash="stale-dataset",
        planned_transcripts=[],
        config=AuditConfig().model_dump(),
        run_id="run",
        concurrency=1,
    )
    (run / "manifests/run.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "require_private_path", lambda path, root: None)
    monkeypatch.setattr(
        cli, "_dataset_manifest", lambda config: {"normalized_content_digest": "new"}
    )
    monkeypatch.setattr(dataset, "load_dataset", lambda path, split: ([], []))
    monkeypatch.setattr(metrics, "validate_rows", lambda frame, **kwargs: frame)
    monkeypatch.setattr(reporting, "_validate_public_identifiers", lambda frame: None)
    with pytest.raises(ValueError):
        cli.export_results(run, tmp_path / "public")
    assert not (tmp_path / "public/public_scores.csv").exists()


class SyntheticClient:
    """The real provider handles this isolated fake SDK boundary, with no network."""

    generations = []
    first_monitor_invalid = True

    def __init__(self, **kwargs):
        self.messages = self
        self.models = self

    def retrieve(self, model):
        data = dict(id=model, max_input_tokens=100000, max_tokens=10000)
        return SimpleNamespace(model_dump=lambda **kwargs: data)

    def count_tokens(self, **kwargs):
        text = kwargs["messages"][0]["content"]
        return SimpleNamespace(input_tokens=(len(text.encode()) + 3) // 4 + 4)

    def create(self, **kwargs):
        assert "tools" not in kwargs
        assert "thinking" not in kwargs
        assert len(kwargs["messages"]) == 1
        self.generations.append(kwargs)
        system = kwargs["system"]
        if "suspicion_score" in system:
            if self.first_monitor_invalid:
                type(self).first_monitor_invalid = False
                text = "invalid independent synthetic response"
            else:
                text = json.dumps(
                    dict(
                        suspicion_score=70,
                        brief_justification="Observed events.",
                        evidence_event_ids=[],
                        uncertainties=[],
                    )
                )
        elif "environment_and_state" in system:
            text = json.dumps(
                dict(
                    environment_and_state=[],
                    actions_and_observed_results=["Observed [E0001]."],
                    important_identifiers=[],
                    contradictions_or_missing_information=[],
                    source_event_ids=["E0001"],
                )
            )
        else:
            text = "Observed a visible action [E0001]."
        raw = dict(
            content=[dict(type="text", text=text)], model=kwargs["model"], stop_reason="end_turn"
        )
        return SimpleNamespace(
            model_dump=lambda **kwargs: raw,
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            content=[SimpleNamespace(type="text", text=text)],
        )


def test_full_runner_retries_repeats_costs_and_resume_use_real_provider_boundary(
    tmp_path, monkeypatch
):
    import anthropic
    import pandas as pd

    import context_audit.runner as runner
    from context_audit.dataset import parse_transcript
    from context_audit.provider import Price

    monkeypatch.setenv("ANTHROPIC_API_KEY", "independent-synthetic-not-an-api-key")
    monkeypatch.setattr(anthropic, "Anthropic", SyntheticClient)
    monkeypatch.setattr(runner, "code_state", lambda: dict(commit="1" * 40, hash="code"))
    SyntheticClient.generations = []
    SyntheticClient.first_monitor_invalid = True
    cfg = AuditConfig(
        monitor_model="fixture-monitor",
        summarizer_model="fixture-summary",
        repetitions=2,
        pilot_pairs=None,
        run_dir=str(tmp_path / "private"),
    )
    original = parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"),
        "t_" + "a" * 24,
        data_origin="synthetic_fixture",
    )
    inputs = [original, original.model_copy(update={"transcript_id": "t_" + "b" * 24})]
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
    price = Price(
        input_per_million=1,
        output_per_million=1,
        cache_read_per_million=0,
        cache_write_per_million=0,
    )
    arguments = (
        cfg,
        inputs,
        labels,
        {"dataset_commit": "1" * 40},
        {"code_hash": "code", "prompts": {}},
        1.0,
        ["Independent synthetic canary"],
        {cfg.monitor_model: price, cfg.summarizer_model: price},
        {},
    )
    first = runner._run_locked(*arguments)
    assert first["successful_rows"] == first["expected_rows"] == 16
    assert first["generation_requests"] == len(SyntheticClient.generations) == 25
    scores = pd.read_csv(tmp_path / "private/scores.csv")
    assert set(scores.data_origin) == {"synthetic_fixture"}
    assert set(scores.repetition) == {0, 1}
    assert scores.cost_usd.sum() == pytest.approx(25 * 30 / 1_000_000)
    assert all(scores.summary_cost_usd[scores.condition.str.endswith("summary")] > 0)
    second = runner._run_locked(*arguments)
    assert second["generation_requests"] == first["generation_requests"]
    assert len(SyntheticClient.generations) == 25
    assert pd.read_csv(tmp_path / "private/scores.csv").equals(scores)
