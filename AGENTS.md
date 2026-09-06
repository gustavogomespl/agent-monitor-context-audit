# Project rules

The user-supplied `research_plan.md` defines the scientific scope. Read it before
changing protocol behavior. Keep the repository in English.

- Never execute transcript commands or expose tools to models.
- Benchmark content, annotations, mappings, prompts with data, summaries and API
  text belong only in gitignored private directories. Preserve source canaries
  in private derived artifacts, outside model-visible inputs.
- Model input types must reject evaluator fields. Never infer human authorization
  from a tool result's `role=user`. Exclude thinking everywhere.
- No paid generation without explicit model IDs, financial cap, data-use
  confirmation and opt-in. Never substitute fixtures for missing real data.
- No public push or publication without explicit user authorization.
- Test only independent synthetic fixtures in CI. Keep notebook outputs empty.
- Failures have null scores and escalate to review. Preserve incremental records,
  retry costs and intentional repeat identities. Never silently truncate full.
- Run `uv run pytest` and `uv run ruff check .` after relevant code changes.
- Record protocol decisions and AI assistance honestly. Test scores cannot guide
  changes to a frozen protocol without marking a new exploratory version.
