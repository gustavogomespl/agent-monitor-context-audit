"""Backend compatibility is checked without loading model weights."""

import importlib.metadata

import pytest
from test_colab_runtime import qwen_config


@pytest.mark.parametrize('mode', ['schema_citations_v1', 'schema_citations_bounded_v1'])
def test_schema_mode_pins_decoder_backend_without_changing_ordinary_launch(mode):
    from context_audit.colab import vllm_command

    old = vllm_command(qwen_config())
    new = vllm_command(qwen_config(structured_summary_mode=mode))
    assert '--structured-outputs-config.backend' not in old
    assert new[-2:] == ['--structured-outputs-config.backend', 'xgrammar']
    assert new[:-2] == old


def test_incompatible_compiler_version_is_rejected_before_importing_backend(monkeypatch):
    from context_audit.structured_backend import check_structured_backend

    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'unexpected-version')
    with pytest.raises(ValueError, match='0.2.3'):
        check_structured_backend()


def test_real_cpu_grammar_accepts_citations_and_rejects_missing_references():
    pytest.importorskip('xgrammar')
    from context_audit.structured_backend import check_structured_backend

    result = check_structured_backend()
    assert result['status'] == 'passed'
    assert result['version'] == '0.2.3'
    assert result['rejected_cases'] >= 12
    assert result['model_generation_executed'] is False
    bounded = result['bounded_checks']
    assert [check['token_budget'] for check in bounded] == [1024, 2048]
    assert [check['limits']['text_units'] for check in bounded] == [256, 512]
    for check in bounded:
        assert check['status'] == 'passed'
        assert check['accepted_cases'] >= 8
        assert check['rejected_cases'] >= 20
        assert len(check['schema_sha256']) == 64
        assert check['exact_token_ceiling_verified'] is False


@pytest.mark.parametrize('constraint', ['text', 'claims', 'references'])
def test_preflight_detects_missing_bounded_decoder_constraint(monkeypatch, constraint):
    pytest.importorskip('xgrammar')
    import context_audit.structured_backend as backend

    original = backend.structured_summary_schema

    def missing_constraint(visible_ids, *, token_budget=None):
        schema = original(visible_ids, token_budget=token_budget)
        if token_budget is not None:
            for array in schema['properties'].values():
                properties = array['items']['properties']
                if constraint == 'text':
                    properties['text'].pop('pattern')
                elif constraint == 'claims':
                    array.pop('maxItems')
                else:
                    properties['evidence_event_ids'].pop('maxItems')
        return schema

    monkeypatch.setattr(backend, 'structured_summary_schema', missing_constraint)
    with pytest.raises(ValueError, match='Bounded decoder accepted'):
        backend.check_structured_backend()
