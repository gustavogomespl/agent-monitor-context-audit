"""CPU-only verification of the pinned citation decoder before model startup."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json

from context_audit.structured_summary import CLAIM_FIELDS, structured_summary_schema

XGRAMMAR_VERSION = "0.2.3"


def check_structured_backend() -> dict:
    """Compile the actual schema and test complete strings using independent toy inputs."""
    version = importlib.metadata.version("xgrammar")
    if version != XGRAMMAR_VERSION:
        raise ValueError(f"Citation decoding requires xgrammar=={XGRAMMAR_VERSION}")
    import xgrammar as xgr

    schema = structured_summary_schema({"E0003", "E0004"})
    grammar = xgr.Grammar.from_json_schema(schema)
    # A small byte vocabulary is enough to validate grammar matching, without any
    # model tokenizer download, GPU use or generation. Colab uses the same compiler.
    vocabulary = [chr(value) for value in range(32, 127)] + ["<eos>"]
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[len(vocabulary) - 1])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(grammar)

    def accepts(body):
        matcher = xgr.GrammarMatcher(compiled)
        text = json.dumps(body, ensure_ascii=True)
        return matcher.accept_string(text) and matcher.is_completed()

    cited = {"text": "Independent toy observation.", "evidence_event_ids": ["E0003"]}
    valid = {field: [copy.deepcopy(cited)] for field in CLAIM_FIELDS}
    if not accepts(valid):
        raise ValueError("Citation decoder rejected a valid synthetic summary")
    escaped = copy.deepcopy(valid)
    escaped[CLAIM_FIELDS[0]][0]["text"] = 'Toy "quote", newline\nand unicode: café.'
    if not accepts(escaped):
        raise ValueError("Citation decoder rejected valid escaped claim text")
    rejected = 0
    for field in CLAIM_FIELDS:
        for invalid in (
            {"text": "No references field."},
            {"text": "Empty references.", "evidence_event_ids": []},
            {"text": "Unknown reference.", "evidence_event_ids": ["E9999"]},
            {"evidence_event_ids": ["E0003"]},
            {**cited, "label": "synthetic forbidden field"},
        ):
            # A cited neighbor must never satisfy another item's missing citation.
            for pair in ([invalid, cited], [cited, invalid]):
                example = copy.deepcopy(valid)
                example[field] = pair
                if accepts(example):
                    raise ValueError("Citation decoder accepted an invalid synthetic item")
                rejected += 1
    missing = copy.deepcopy(valid)
    missing.pop(CLAIM_FIELDS[-1])
    if accepts(missing):
        raise ValueError("Citation decoder accepted a missing summary field")
    rejected += 1
    return {
        "status": "passed", "backend": "xgrammar", "version": version,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "accepted_cases": 2, "rejected_cases": rejected, "model_generation_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(check_structured_backend(), sort_keys=True))
