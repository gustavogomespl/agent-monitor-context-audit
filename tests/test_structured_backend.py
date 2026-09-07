"""Backend compatibility is checked without loading model weights."""

import importlib.metadata

import pytest
from test_colab_runtime import qwen_config


def test_schema_mode_pins_decoder_backend_without_changing_ordinary_launch():
    from context_audit.colab import vllm_command

    old = vllm_command(qwen_config())
    new = vllm_command(qwen_config(structured_summary_mode='schema_citations_v1'))
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
