"""Synthetic HTTP checks for constrained decoding and separate cache identities."""

import copy
import json

import httpx
import pytest
from test_qwen_provider import MODEL, generate, setup_provider

from context_audit.runtime_models import AuditConfig, QwenConfig
from context_audit.structured_summary import CLAIM_FIELDS, structured_summary_schema


def synthetic_schema():
    return structured_summary_schema({'E0003', 'E0004'})


def test_schema_is_sent_and_recorded_without_becoming_prompt_text(tmp_path, monkeypatch):
    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch)
    schema = synthetic_schema()
    untouched = copy.deepcopy(schema)
    key, result = generate(provider, structured_schema=schema)
    sent = [p for r, p in requests if r.url.path == '/v1/chat/completions'][0]
    tokenized = [p for r, p in requests if r.url.path == '/tokenize'][0]
    assert sent['structured_outputs'] == {'json': untouched}
    assert 'structured_outputs' not in tokenized
    assert sent['messages'] == tokenized['messages']
    assert schema == untouched
    assert 'tools' not in sent and sent['chat_template_kwargs']['enable_thinking'] is False
    assert 'independent-source-notice' not in json.dumps(sent)
    assert store.get('requests', key)['request']['structured_outputs'] == {'json': schema}
    assert key in ledger.settled and result['status'] == 'ok'


def test_schema_changes_do_not_reuse_unconstrained_or_other_schema_cache(tmp_path, monkeypatch):
    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch)
    ordinary, _ = generate(provider)
    schema = synthetic_schema()
    first, _ = generate(provider, structured_schema=schema)
    repeated, _ = generate(provider, structured_schema=schema)
    changed = copy.deepcopy(schema)
    changed['properties'][CLAIM_FIELDS[0]]['maxItems'] = 2
    different, _ = generate(provider, structured_schema=changed)
    assert len({ordinary, first, different}) == 3
    assert repeated == first
    assert len(ledger.amounts) == 3
    assert sum(r.url.path == '/v1/chat/completions' for r, _ in requests) == 3


def test_schema_rejection_is_not_retried_without_constraints(tmp_path, monkeypatch):
    def reject(request, payload):
        if request.url.path == '/v1/chat/completions':
            assert payload['structured_outputs']['json'] == synthetic_schema()
            return httpx.Response(400, json={'error': {'message': 'Synthetic schema rejection'}})

    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch, handler=reject)
    _, result = generate(provider, structured_schema=synthetic_schema())
    assert result['status'] == 'api_error'
    assert result['http_status'] == 400
    assert len(ledger.amounts) == 1
    assert sum(r.url.path == '/v1/chat/completions' for r, _ in requests) == 1


def test_schema_mode_is_explicit_and_only_supported_for_qwen():
    assert AuditConfig().structured_summary_mode == 'prompt'
    with pytest.raises(ValueError, match='Qwen|qwen'):
        AuditConfig(structured_summary_mode='schema_citations_v1')
    config = AuditConfig(provider='qwen_local', qwen=QwenConfig(),
                         monitor_model=MODEL, summarizer_model=MODEL,
                         structured_summary_mode='schema_citations_v1')
    assert config.model_dump()['structured_summary_mode'] == 'schema_citations_v1'
    assert config.max_attempts == 2 and config.token_maximum == 1024
