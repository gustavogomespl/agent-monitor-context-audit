"""Independent synthetic checks for visible-evidence monitor decoding."""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from context_audit.runtime_models import MonitorDecision


def monitor_schema(visible_ids=None):
    from context_audit.monitor_schema import monitor_response_schema

    return monitor_response_schema({"E0002", "E0004"} if visible_ids is None else visible_ids)


def decision(**changes):
    return {
        "suspicion_score": 25.5,
        "brief_justification": "Independent toy inventory checked.",
        "evidence_event_ids": ["E0002"],
        "uncertainties": ["Toy count remains unknown."],
        **changes,
    }


@pytest.mark.parametrize("score", [0, 0.001, 25.5, 99.999, 100])
def test_schema_accepts_continuous_scores_and_visible_evidence(score):
    schema = monitor_schema()
    Draft202012Validator.check_schema(schema)
    body = decision(suspicion_score=score)
    Draft202012Validator(schema).validate(body)
    assert MonitorDecision.model_validate(body).suspicion_score == score


@pytest.mark.parametrize("score", [-0.001, 100.001, "25", True, None, [], {}])
def test_schema_rejects_out_of_range_and_non_numeric_scores(score):
    assert not Draft202012Validator(monitor_schema()).is_valid(decision(suspicion_score=score))


@pytest.mark.parametrize("references", [
    ["E0003"], ["E9999"], ["E2"], ["e0002"], ["E0002", "E0003"],
    [2], [None], [True], "E0002", None,
])
def test_schema_rejects_invisible_or_malformed_evidence(references):
    # E0003 can be present in a full history but absent from this compressed view.
    assert not Draft202012Validator(monitor_schema()).is_valid(
        decision(evidence_event_ids=references)
    )


@pytest.mark.parametrize("references", [[], ["E0004"], ["E0004", "E0002"] * 5])
def test_schema_preserves_optional_evidence_and_repeated_visible_references(references):
    body = decision(evidence_event_ids=references)
    Draft202012Validator(monitor_schema()).validate(body)
    assert MonitorDecision.model_validate(body).evidence_event_ids == references


@pytest.mark.parametrize("field", list(decision()))
def test_schema_requires_every_monitor_field(field):
    body = decision()
    del body[field]
    assert not Draft202012Validator(monitor_schema()).is_valid(body)


@pytest.mark.parametrize("field", ["label", "metadata", "thinking", "tools", "status"])
def test_schema_rejects_extra_fields(field):
    assert not Draft202012Validator(monitor_schema()).is_valid(decision(**{field: "unused"}))


@pytest.mark.parametrize("changes", [
    {"brief_justification": 5}, {"brief_justification": None},
    {"uncertainties": "unknown"}, {"uncertainties": [None]},
    {"uncertainties": [5]}, {"uncertainties": None},
])
def test_schema_rejects_invalid_text_field_types(changes):
    assert not Draft202012Validator(monitor_schema()).is_valid(decision(**changes))


@pytest.mark.parametrize("text", [
    'Toy "quote", backslash \\, newline\nand Unicode café 😀.',
    "", "x" * 3001,
], ids=["escaped_text", "empty_text", "overlong_text"])
def test_string_grammar_preserves_escaped_text_and_defers_length_to_local_validation(text):
    body = decision(brief_justification=text, uncertainties=[text])
    Draft202012Validator(monitor_schema()).validate(json.loads(json.dumps(body)))
    if not 1 <= len(text) <= 3000:
        with pytest.raises(ValidationError):
            MonitorDecision.model_validate(body)
    else:
        assert MonitorDecision.model_validate(body).brief_justification == text


def test_empty_visibility_allows_only_empty_evidence_with_valid_schema():
    schema = monitor_schema(set())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(decision(evidence_event_ids=[]))
    assert not validator.is_valid(decision(evidence_event_ids=["E0002"]))
    assert schema["properties"]["evidence_event_ids"]["maxItems"] == 0
    assert "enum" not in schema["properties"]["evidence_event_ids"]["items"]


def test_schema_is_deterministic_fresh_and_omits_decoder_string_bounds():
    visible_ids = {"E0004", "E0002"}
    schema = monitor_schema(visible_ids)
    assert schema["properties"]["evidence_event_ids"]["items"]["enum"] == ["E0002", "E0004"]
    assert schema["properties"]["suspicion_score"] == {
        "type": "number", "minimum": 0, "maximum": 100,
    }
    serialized = json.dumps(schema)
    for unsupported in ("pattern", "minLength", "maxLength"):
        assert unsupported not in serialized
    schema["required"].clear()
    schema["properties"]["evidence_event_ids"]["items"]["enum"].clear()
    assert visible_ids == {"E0002", "E0004"}
    assert monitor_schema(visible_ids) == monitor_schema({"E0002", "E0004"})
    assert len(monitor_schema(visible_ids)["required"]) == 4
    assert monitor_schema(visible_ids)["properties"]["evidence_event_ids"]["items"]["enum"]


@pytest.mark.parametrize("visible_ids", [{"E2"}, {"e0002"}, {"E0002", "Ebad"}, {1}, {None}])
def test_schema_builder_rejects_invalid_visible_id_inputs(visible_ids):
    with pytest.raises(ValueError, match="visible event IDs"):
        monitor_schema(visible_ids)
