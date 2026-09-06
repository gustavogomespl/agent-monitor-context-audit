import pytest

from context_audit.storage import PrivateStore, digest


def test_content_cache_invalidation_and_atomic_roundtrip(tmp_path):
    store = PrivateStore(tmp_path / "private")
    payload = dict(content="visible", model="model-a", prompt="v1", condition="full", repetition=0)
    key = digest(payload)
    store.put("calls", key, {"status": "ok", "value": 3})
    assert store.get("calls", key)["value"] == 3
    for field, replacement in [
        ("model", "model-b"),
        ("prompt", "v2"),
        ("repetition", 1),
        ("condition", "head_tail"),
        ("content", "changed"),
    ]:
        assert store.get("calls", digest({**payload, field: replacement})) is None


def test_truncated_journal_is_not_silently_ignored(tmp_path):
    store = PrivateStore(tmp_path)
    store.append("ledger", {"reserved": 2})
    assert store.read_journal("ledger") == [{"reserved": 2}]
    (tmp_path / "ledger.jsonl").open("a").write("{broken")
    with pytest.raises(ValueError, match="journal"):
        store.read_journal("ledger")


def test_store_rejects_path_escape(tmp_path):
    store = PrivateStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("../outside", "id", {})
