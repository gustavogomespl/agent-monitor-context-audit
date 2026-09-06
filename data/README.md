# Benchmark data

The real source is `safety-research/sleight-bench`, pinned to
`218c58315cc01ff0dc5a100e906c27d82d259521`. Its code license is MIT;
the upstream anti-contamination notices prohibit inclusion in training corpora.
This project keeps all benchmark content and annotations private regardless of
redistribution rights. API use requires separate explicit data-use confirmation.

Install acquisition support with `uv sync --extra data`, then use
`uv run context-audit acquire` and `uv run context-audit inventory`.
Acquisition clones the official source into `data/private/upstream`, validates
the reviewed commit and decryptor, and follows the README's documented local
Fernet procedure. It never runs trajectory commands, verifiers, evaluation
scripts, or provider calls. Decryption logs remain private. Inventory checks
decrypted bytes against the pinned encrypted payloads before reading them.

`manifests/inventory.json` contains aggregate counts, exclusion reasons under
opaque IDs, and a content digest. `manifests/split.csv` contains only opaque
transcript, scenario, family and split identities; labels are evaluator-only.
The actual pin has 44 source directories in 12 categories, 44 attack files,
43 benign files and one extra variant. This differs from the upstream README's
nominal counts. Six directories belong to three multi-session cases and are
excluded with their benigns. One case lacks a paired benign. Two further pairs
contain visible events without the required `cwd` and are excluded whole.
The eligible collection has 35 pairs in 33 related families: 8 development pairs
and 27 test pairs under seed 20260905. No model scores have been generated.

The source-name-to-opaque-ID mapping, HMAC key, normalized canary envelopes,
source hashes and evaluator labels live only in `private/`, which is ignored
by Git. Preserve this directory securely to reproduce the same opaque IDs.
A new private key creates a new identity mapping, even with the same split seed.
Normalized files carry source canaries outside the `TranscriptInput` object.
`load_dataset()` verifies labels and normalized content integrity, then returns
model inputs and evaluator labels as separate lists. `load_canaries()` is for
private artifact storage and must never be added to model messages.

Family grouping takes the transitive closure of stage/session file relationships,
identical normalized initial requests or scenario annotations, and explicit
sidecar references to other cases. This conservative automated grouping needs
human review before protocol freeze; shared tasks do not establish semantic
independence, and absence of a detected link does not prove independence.

Sizes in the current inventory are UTF-8 bytes, not provider token counts.
The largest eligible normalized body is 370,352 bytes. Official model counts
and context-window acceptance remain pending explicit model selection and
provider preflight. `full` must never be silently truncated.

Independent teaching examples are in `tests/fixtures/`. They are not modified
benchmark cases, and outputs from them must retain `data_origin=synthetic_fixture`.
