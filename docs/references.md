# References and verification record

Checked against the linked primary sources on **2026-09-05**. These checks verify
source identity and the scope used in [related_work.md](related_work.md); they are
not independent reproductions of the papers. Versioned arXiv links identify the
versions inspected. An arXiv listing alone does not establish peer review.

| Plan ID | Verified source | Version / date inspected |
| --- | --- | --- |
| R1 | Elle Najt, Colin Toft, Tyler Tracy, Fabien Roger, Joe Benton. [SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors](https://arxiv.org/abs/2605.16626v2). | v2, 2026-05-19; [full text](https://arxiv.org/html/2605.16626v2), including appendices J–K |
| R2 | [SLEIGHT-Bench official README](https://github.com/safety-research/sleight-bench/blob/218c58315cc01ff0dc5a100e906c27d82d259521/README.md). | Commit `218c58315cc01ff0dc5a100e906c27d82d259521`, acquired 2026-09-05. |
| R3 | [SLEIGHT-Bench FORMAT.md](https://github.com/safety-research/sleight-bench/blob/218c58315cc01ff0dc5a100e906c27d82d259521/FORMAT.md). | Same pinned commit; parser contract inspected 2026-09-05. |
| R4 | [experiment_paper.yaml](https://github.com/safety-research/sleight-bench/blob/218c58315cc01ff0dc5a100e906c27d82d259521/experiment_paper.yaml) and [LICENSE](https://github.com/safety-research/sleight-bench/blob/218c58315cc01ff0dc5a100e906c27d82d259521/LICENSE). | Same pinned commit. MIT copyright 2026 The SLEIGHT-Bench Authors. |
| R5 | Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger. [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942v5). | v5, 2024-07-23; first submitted 2023-12-12. |
| R6 | Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov. [The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management](https://arxiv.org/abs/2508.21433v3). | v3, 2025-10-27. |
| R7 | Shiyang Chen. [Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents](https://arxiv.org/abs/2606.22528v2). | v2, 2026-06-27. |
| R8 | Saber Zerhoudi, Jelena Mitrovic, Michael Granitzer. [The Compaction Cliff in Long-Running AI Agent Memory](https://arxiv.org/abs/2608.22752v1). | v1, 2026-08-24. |
| R9 | Yinghan Hou, Zongyou Yang. [Control Under Compression: Reliability Frontiers for Tool-Using Agents](https://arxiv.org/abs/2608.01056v1). | v1, 2026-08-02. |
| R10 | Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172v3). | v3, 2023-11-20. |
| R11 | Anthropic, [CLI, SDKs, and libraries](https://platform.claude.com/docs/en/cli-sdks-libraries/overview). | Live official documentation inspected 2026-09-05. |
| R12 | Anthropic, [Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting). | Live official documentation inspected 2026-09-05. |

All twelve references in the supplied plan resolved to the expected sources.
Bibliographic verification does not validate their empirical conclusions or imply
that this short review exhausts related work.

Additional engineering sources:

- Anthropic, [Pricing](https://platform.claude.com/docs/en/about-claude/pricing),
  inspected 2026-09-05. Rates depend on the model and billing category. Refresh at
  execution time and record currency, per-million-token units, date, source URL,
  input/output and applicable cache rates. No monetary experiment estimate is
  asserted before models, token counts and prices are configured.
- Factory Research, [Evaluating Context Compression for AI Agents](https://factory.com/news/evaluating-compression),
  2025-12-16. This is a vendor engineering report using continuation-related
  probes and an LLM judge; it is not equivalent to this monitor evaluation or an
  independently replicated academic comparison.

The official token-counting endpoint returns an estimate; provider-added tokens
can differ from billable content. Record estimated counts and response usage
separately, and count against the actual selected model. The official SDK provides
request retries, so application accounting must explicitly control those retries.
See [token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
and [SDK overview](https://platform.claude.com/docs/en/cli-sdks-libraries/overview).
