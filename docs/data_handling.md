# Data handling

The official benchmark uses intentional encryption to reduce accidental
training-data scraping and documents local decryption in its
[README](https://github.com/safety-research/sleight-bench). Access is public;
encryption is not a secrecy claim. This project treats benchmark content and
derived text as restricted publication material and does not use it for training
or fine-tuning.

## Boundaries

| Location | Allowed material | Public Git content? |
| --- | --- | --- |
| `data/private/` | Pinned upstream source, decrypted sessions, metadata, canaries, source-to-opaque mapping, private split/grouping evidence. | No; gitignored. |
| `runs/private/` | Full prompts with data, normalized histories, summaries, API responses, attempt journals, text explanations and review worksheets. | No; gitignored. |
| `.env` | API credentials and local settings. | No; gitignored. |
| `data/manifests/` | Sanitized source provenance and opaque aggregate/split manifests, without names or content. | Yes, after scan and review. |
| `results/` | Sanitized numeric empirical scores, metrics and figures. | Yes, after scan and review. |
| `tests/fixtures/` | Independently authored didactic data, visibly marked synthetic. | Yes. |
| `prompts/` | Generic policies and templates without benchmark material. | Yes. |

Source canary notices travel with private copies and derived artifacts in a
separate provenance envelope or sidecar. They are not interpreted as transcript
events and are excluded from model-visible inputs. Do not copy real canary
strings into public test fixtures. The format's canary-only JSON record is
handled separately from user/assistant messages. See the official
[FORMAT.md](https://github.com/safety-research/sleight-bench/blob/main/FORMAT.md).

Benchmark-derived text is private even when it appears harmless or comes from a
benign example. This includes error messages that echo a prompt, cache entries,
notebook previews, qualitative explanations and excerpts in issues or pull
requests. Opaque IDs do not make accompanying transcript text safe to publish.
The source-name mapping is never a public artifact. Binary evaluator labels may
appear alongside numeric public scores, without textual answer keys or mechanisms.

## API access

The models receive only fixed policy, literal initial task and label-free history
representations. No evaluator metadata is passed to prompt constructors. Tool
observations remain observations even if the original serialization uses a user
role. Thinking is excluded from normalization and generated thinking is neither
requested nor retained. No shell, browser or execution tools are available to
the monitor or summarizer. Transcript commands are never executed.

Before sending benchmark material to an API, the user must confirm that the
chosen account's data-use, retention and training settings satisfy the benchmark
restrictions. Token counting also transmits text to the provider. A configured
key alone is not this confirmation and is not spending authorization. Generation
also requires explicit model IDs, a financial cap and an explicit live opt-in.
No fixture is substituted when real data or access is missing.

## Reproducibility and publication checks

Pin source commit and retain acquisition provenance; record protocol config,
source/prompt hashes, model IDs, SDK/lockfile versions, seeds, prices, attempts
and generation dates. Keep notebooks committed with empty outputs. CI must run
only independent fixtures and must not fetch real data, require credentials or
make paid calls.

Run `uv run context-audit scan-public` before preparing a public release. The
scan should inspect actual candidate file content and notebook outputs against
known private source strings and canaries, rather than trusting filenames alone.
Exact-match scanning cannot prove the absence of all paraphrases; human review
of public text and figures remains required. A missing private source limits
source-matching coverage and must not be described as a complete real-data scan.
Review Git history as well as the working tree before any initial public push.

## Licenses and attribution

This project's existing [LICENSE](../LICENSE) is Apache-2.0 and is preserved.
Upstream SLEIGHT-Bench has an [MIT LICENSE](https://github.com/safety-research/sleight-bench/blob/main/LICENSE),
copyright 2026 The SLEIGHT-Bench Authors. Keep upstream notices with local source
copies. The repository license and benchmark anti-contamination warnings are
recorded separately: this project does not claim its Apache license relicenses
the dataset or removes the upstream warnings. No dataset is redistributed here.
An API provider's account-specific terms require the user's confirmation; this
document does not certify them.
