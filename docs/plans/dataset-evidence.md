# Dataset implementation evidence — 2026-09-05

The original supplied plan, `research_plan.md`, `AGENTS.md`, and the official
README, FORMAT specification and LICENSE were read before parser implementation.
The official checkout was obtained from GitHub and pinned at
`218c58315cc01ff0dc5a100e906c27d82d259521`. The reviewed official `decrypt.py`
performs Fernet decryption and local filesystem writes; only this documented
decryptor was executed. No transcript commands, dataset verifiers, upstream
experiments, or paid model calls were run.

The README's headline counts are not the inventory at this pin. Actual evidence:

| Item | Count |
|---|---:|
| Source directories / categories | 44 / 12 |
| Attack / benign / additional variant JSONL files | 44 / 43 / 1 |
| Multi-session directories excluded / logical cases | 6 / 3 |
| Unpaired cases excluded | 1 |
| Whole pairs excluded for missing required `cwd` | 2 |
| Eligible pairs / transcripts | 35 / 70 |
| Related families / families with two pairs | 33 / 2 |
| Development pairs / test pairs | 8 / 27 |

The six session-related directories were identified through stage/part file
relations plus metadata/description markers. Two multi-session cases have
identical initial requests and remain in the same excluded family. Eligible
related variants are grouped through explicit sidecar references. No annotation,
path name, label, metadata scenario, outcome or benchmark summary is read by
`parse_transcript()` or rendered into model prompts. Metadata influences only
evaluator grouping and exclusions. Human review of grouping remains a pre-freeze
requirement; the automatic rule is not a proof of independence.

The normalization preserves literal initial human text exactly once, its initial cwd in
the common uncompressed header, visible
event order, cwd, call/result relationships, visible tool arguments and error
flags. It excludes actual thinking blocks, retains later human messages, and
keeps user-role tool results explicitly typed as tool results. Empty tool-result
strings are valid. Literal tool-argument keys named `thinking` or `signature`
remain ordinary visible data; they are not interpreted as hidden reasoning.
Canaries are preserved in private envelopes and excluded from `TranscriptInput`.
Real normalized input loading checked 70 inputs with zero source-canary matches.

Source identity is checked through the official Git remote, exact commit and
pristine tracked files. Decrypted source content is compared byte-for-byte to
the pinned encrypted payload; missing, changed or extra relevant source content
fails inventory. Private normalized hashes and evaluator-label hashes are
validated at load. A sanitized aggregate input digest binds the public inventory
to normalized content for later protocol freeze.

Current eligible normalized-body UTF-8 byte sizes are minimum 2,135, median
14,010, maximum 370,352. These are not tokens. Official model token counting
and full-context acceptance are still pending explicit model selection and
provider access. No empirical performance claim is supported yet.

Validation used 25 independent synthetic tests for normalization, malformed
format rejection, hidden-thinking exclusion, tool result authorization boundary,
metadata/directory-name invariance, strict evaluator-field rejection, HMAC IDs,
family-preserving splits, whole-case exclusions and source/artifact integrity.
`uv run pytest tests/test_dataset.py tests/test_splits.py
tests/test_no_label_leakage.py -q` passed 25 tests. Dataset-owned Python paths
pass Ruff. The first required project-wide Ruff invocation found five issues
in concurrently edited, other-owned files; these were reported to the lead
agent for the final project-wide check. Sandbox `uv` attempts hit a macOS system
configuration panic; the same local checks succeeded with approved execution
outside the sandbox.

All source content, mappings, canaries, decryption logs and normalized envelopes
remain beneath Git-ignored `data/private/`. Public manifests contain aggregate
or opaque numeric metadata only. No commits or pushes were made by this worker.

An independent integration review added eleven checks in
`tests/test_integration_invariants.py`: freeze rejects changed code, changed
normalized data, mismatched completion identity and incomplete condition schedules;
matching development evidence freezes successfully; export rejects a stale dataset.
The complete runner was exercised through the actual provider implementation with
only its SDK client replaced by an independent synthetic client. Two fixture
transcripts, four conditions and two repetitions produced 16 valid rows through
25 generation requests including one invalid-JSON retry. Numeric costs include
summaries, monitors and retry. Resumption made zero additional generation requests
and preserved the exact score table. Those fixture scores stay in test temporary
storage and are not empirical results.
