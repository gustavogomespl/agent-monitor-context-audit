import pytest

from context_audit.dataset import DatasetError, assign_splits, opaque_id, validate_splits
from context_audit.schemas import EvaluationLabel


def labels():
    return [
        EvaluationLabel(
            transcript_id=f"t_{i * 2 + y:024x}",
            label=y,
            scenario_id=f"s_{i:024x}",
            family_id=f"f_{i // 2:024x}",
            split="unassigned",
        )
        for i in range(8)
        for y in (0, 1)
    ]


def test_family_pairs_remain_together_and_split_is_seeded():
    first = assign_splits(labels(), seed=71, development_pairs=2)
    assert first == assign_splits(list(reversed(labels())), seed=71, development_pairs=2)
    assert sum(x.split == "development" for x in first) == 4
    validate_splits(first)
    assert all(len({x.split for x in first if x.family_id == f"f_{i:024x}"}) == 1 for i in range(4))


def test_family_overlap_or_incomplete_pairs_rejected():
    rows = assign_splits(labels(), seed=71, development_pairs=2)
    target = rows[0].family_id
    bad = [x.model_copy(update={"split": "test"}) if x == rows[0] else x for x in rows]
    same = [x for x in rows if x.family_id == target and x != rows[0]][0]
    bad[0] = rows[0].model_copy(update={"split": "development" if same.split == "test" else "test"})
    with pytest.raises(DatasetError):
        validate_splits(bad)
    with pytest.raises(DatasetError):
        validate_splits(rows[:-1])


def test_ids_need_secret_and_do_not_encode_label_or_source_name():
    assert opaque_id(b"a" * 32, "transcript", "folder/a.jsonl").startswith("t_")
    assert opaque_id(b"a" * 32, "transcript", "folder/a.jsonl") != opaque_id(
        b"b" * 32, "transcript", "folder/a.jsonl"
    )
    with pytest.raises(ValueError):
        opaque_id(b"", "transcript", "anything")
