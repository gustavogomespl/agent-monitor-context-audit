# Agent Monitor Context Audit: findings and execution status

**Status: experiment not executed.** No empirical monitoring scores, AUROCs,
performance intervals, qualitative findings or measured generation costs exist
for this project. A working pipeline or synthetic demonstration cannot establish
that structured summaries preserve monitoring signal better than prose summaries.

## Research question

Does a structured summary preserve discrimination between problematic and paired
benign transcripts better than prose under the same models and per-example
ceiling? The planned primary outcome is structured minus free-summary AUROC,
with a paired family bootstrap. Full normalized history and deterministic
head/tail text are secondary references. Negative and uncertain results remain
valid. See [methodology](../docs/methodology.md).

## Verified dataset inventory

The source checkout was acquired and inventoried locally on 2026-09-05 at commit
`218c58315cc01ff0dc5a100e906c27d82d259521`. Current aggregate evidence appears in
[inventory.json](../data/manifests/inventory.json) and the
[opaque split](../data/manifests/split.csv):

| Quantity | Observed count |
| --- | ---: |
| Source directories / categories | 44 / 12 |
| All JSONL files | 88 |
| Attack / paired benign transcript files | 44 / 43 |
| Multisession exclusions | 6 directories comprising 3 cases |
| Unpaired / malformed pair exclusions | 1 / 2 |
| Eligible pairs / transcripts | 35 / 70 |
| Eligible families | 33 |
| Development / test pairs, seed `20260905` | 8 / 27 |

One additional JSONL variant explains why file counts do not define case counts.
Two malformed pairs fail the required cwd contract and were excluded before
scoring. These observations supersede the plan's README-based 37-pair expectation
without changing source data. Family grouping uses structural relations,
identical tasks/scenarios and explicit sidecar references. Two retained families
each contain two pairs; human review of possible missed relationships is pending.

Normalized eligible bodies range from 2,135 to 370,352 UTF-8 bytes, with median
14,010 bytes. **These are byte sizes, not token counts.** Model-specific counts
and full-context feasibility require model choice and authorized provider
counting. At the current 27-pair test size, 54 transcripts imply at most 324
ordinary generation calls before identity shortcuts, retries or repeats. This is
not a dollar estimate or a completed execution.

The [source verification record](../docs/references.md) checks the twelve cited
references against primary sources. Dataset and engineering evidence are
separate from empirical model performance. Independent fixtures exercise the
technical contracts and do not enter public empirical score tables.

## Outstanding empirical work

| Milestone | Required evidence still outstanding |
| --- | --- |
| M1 model feasibility | Explicit model IDs, official token counts and model-context acceptance; human family review. |
| M3 real development pilot | Authorized models/account and 3–5 development pairs with traceable private generation records, lengths, failures and costs. |
| M4 protocol freeze | Completed full development run, author review, immutable protocol signature and `protocol-v1` commit/tag. |
| M5 main experiment | All planned test conditions and an incremental ledger, including reasons for missing cases. |
| M6 empirical analysis | Real sanitized scores, coverage/intervals, figures, private evidence review and reviewed conclusions. |
| Publication | Content review and explicit user authorization to publish. |

Live generation requires explicit monitor/summarizer IDs, a financial cap,
confirmation that the account's data use satisfies benchmark restrictions, and
live opt-in. No paid generation calls have been made. The configuration leaves
models and review confirmations unset; data or fixtures cannot bypass these
gates. A full completion audit is in
[completion-audit.md](../docs/plans/completion-audit.md).

## Reporting after the authorized run

Record source/code revisions, protocol identity, dates, requested/returned model
IDs, planned/observed pairs, exclusions, repetitions and price provenance.
Report condition AUROCs with intervals, the primary paired difference, common
complete-pair N and coverage, every failure category, valid-score/end-to-end
TPR/FPR at 50, balanced-accuracy components, discordances, realized lengths,
generation latency and costs including summaries and retries.

Figures must label models, split, N and coverage: AUROC intervals; TPR/FPR at the
fixed threshold; cost versus performance. Use the public run manifest to verify
planned coverage, including entirely missing pairs. Separate measured charges
from conservative upper bounds for unknown billing. Recorded request latency
excludes counting/preflight and local processing overhead. Caps accumulate per
run directory; separate stages need explicit budget allocations.

The reproduction notebook recalculates sanitized numeric real data without API
calls. No placeholder performance values belong in tables or figures. The
machine-generated `results/findings.md` supplements this authored report; an
empty-results placeholder must not replace the verified inventory here.

Review prespecified disagreement/control cases privately with event references,
reviewer confidence and the methodology rubric. Report only supported patterns;
do not publish benchmark excerpts, annotations or source-name mappings.

## Interpretation limits and assistance

The records are authentic acquisitions of a **synthetic benchmark**, not
production incident observations. The study is small, retrospective,
model-specific and conditional on technical coverage. Equal ceilings do not
ensure equal lengths; one pipeline realization does not establish stability.
Full history is not a guaranteed upper bound. Post-trajectory detection cannot
demonstrate prevention. Few benign scenarios cannot support a low-FPR safety
guarantee, and disagreement-selected reviews do not estimate error prevalence.

AI assistance and pending author review are disclosed in
[decisions.md](../docs/decisions.md). No empirical conclusion has been reached.
