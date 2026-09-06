# Empirical results

**Experiment not executed.** `public_scores.csv` contains the validated schema
header and no result rows. `metrics.json` explicitly reports that no experiment
has been executed. No performance numbers or figures are fabricated.

After an authorized real run, export the sanitized numeric scores and adjacent
`run_manifest.json`, then use `context-audit analyze` or
`notebooks/02_reproduce_results.ipynb`. The run manifest supplies planned units,
so omitted pairs cannot disappear from the coverage denominator. The generated
`findings.md`, three figures and blank opaque qualitative worksheet can be
recomputed on CPU without API access.

Synthetic fixtures never belong in this empirical-results directory. Complete
qualitative reviews, source event text, justifications and API output stay private.
Public artifacts require a content scan and explicit publication authorization.
