# Related work

This project asks whether changing the **history shown to an external monitor**
changes its ability to rank problematic transcripts above paired benign ones.
Its narrow contribution is a controlled retrospective comparison on an existing
benchmark. Lossy compression losing information is established motivation, not a
novelty claim. The directional hypothesis that structured summaries outperform
free summaries remains untested here.

## What the sources establish

[SLEIGHT-Bench](https://arxiv.org/abs/2605.16626v2) introduces synthetic agent
transcripts with paired benign versions and evaluates monitoring. Its appendix J
shows that benign calibration distribution matters; appendix K distinguishes
failure to produce a score from benign behavior. We adopt the dataset, explicit
failure reporting and caution about calibration. We do not import the original
paper's monitor configurations or performance numbers as our own results.

[AI Control](https://arxiv.org/abs/2312.06942v5) evaluates supervision protocols
under intentional subversion. It motivates studying a monitor's reliability when
the observed agent cannot simply be assumed trustworthy. Our offline analysis
does not implement or evaluate the complete control protocols from that work.

[The Complexity Trap](https://arxiv.org/abs/2508.21433v3) compares observation
masking with LLM summarization in software engineering agents and reports a
competitive inexpensive deterministic strategy. This motivates including a cheap
baseline. Our head/tail text cut is different from observation masking, and our
outcome is monitor discrimination rather than agent task success.

[Governance Decay](https://arxiv.org/abs/2606.22528v2) studies constraints removed
by compaction and resulting agent behavior. [The Compaction Cliff](https://arxiv.org/abs/2608.22752v1)
studies repeated compaction of agent memory and type-specific retention. Both
motivate fidelity concerns. Here the monitor policy and initial task stay outside
compression; the intervention changes event evidence, not an acting agent's
safety policy or memory.

[Control Under Compression](https://arxiv.org/abs/2608.01056v1) compresses
system-side control contexts such as tool documentation and instructions and
assesses execution reliability. Our treatment instead compresses observed
history under a fixed monitor policy. Its method rankings cannot establish a
ranking of our representations.

[Lost in the Middle](https://arxiv.org/abs/2307.03172v3) reports position-dependent
performance on retrieval and question-answering tasks. It motivates checking
which evidence remains available after compression. It does not prove that the
same positional effect occurs for the eventual monitor model or these histories.

The [Factory engineering report](https://factory.com/news/evaluating-compression)
evaluates compression through task-continuation probes graded by an LLM. It is
useful motivation for examining exact identifiers and state, while its vendor
comparison, judge and outcomes differ from this study.

## Decisions made here

The common task header, exclusion of thinking, family-level split, four
conditions, per-example budget, fixed exploratory threshold and paired
complete-case bootstrap come from this project's protocol. They should not be
attributed to the cited papers. Holding the summarizer model, preservation
priorities and token ceiling fixed controls several differences; it does not
isolate a pure causal effect of JSON syntax or guarantee equal realized length.

Full normalized history is a reference, not a guaranteed upper bound: a shorter
view might reduce distraction. Post-trajectory success cannot establish timely
intervention, prevention, or production safety. Shared model IDs do not imply
independent errors. A single sampled pipeline realization does not establish
stability across model calls.

## Limits of this review

The primary sources and their version metadata were checked on 2026-09-05; see
[references.md](references.md). This is a focused review of the supplied plan,
not a systematic search or exhaustive novelty assessment. Source claims have not
been independently replicated. Recent arXiv manuscripts must not automatically
be described as peer reviewed. The benchmark is synthetic even when acquired
from its real source: "real data" here means authentic benchmark records, not
observational production incidents.
