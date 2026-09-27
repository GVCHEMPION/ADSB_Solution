# CLAUDE.md

## What this is

Solution to the Avito DS Bootcamp test task: **candidate generation for services search**.
Given a short search query, return up to 50 `item_id` from a 189k-ad corpus.
Metric: **Recall@50** (order inside the 50 does not matter). Full spec: `tasks/task.md` (Russian).

## Data (`dataset/`, gitignored, ~670 MB)

| File | Rows | Notes |
|---|---|---|
| `train.parquet` | 497 673 | one row = (query, chosen item) pair; `search_*` + `item_*` columns |
| `benchmark_queries.parquet` | 2 452 | `query_id` + `search_*` columns, unlabeled |
| `benchmark_items.parquet` | 189 212 | `item_id` + `item_*` columns — the corpus to search |

Key columns: `search_query`, `search_location_id`, `search_category`, `search_infm_params_text`,
`item_title_raw`, `item_description_raw`, `item_infm_params_text`, `item_category_id`, `item_microcat_id`,
`item_price`, `item_rating`, `item_location_id`, `item_latitude/longitude`.

## Output contract (`answer.csv`)

UTF-8 CSV, comma-separated, exactly two columns `query_id,answer`, no index column.
`answer` = up to 50 space-separated `item_id`, no duplicates within a row.
IDs are 16-char strings, case-sensitive, copied verbatim from the parquet.
One row per `query_id` in `benchmark_queries.parquet` — no gaps, no extras, no repeats.

## Constraints from the task

- Must run locally at the reviewer's machine — **no external API calls** at inference.
- Open-source models/libs are fine, but must be named in the solution write-up.
- No hardcoding answers per `query_id`, no reconstructing the test labels.
- The task grades explained code over bare code — but see the no-comments rule below.

## Environment & commands

uv-managed (`pyproject.toml`, `uv.lock`); always run via `uv run` — global Python lacks the deps.

```bash
uv run python src/experiment.py <method>           # Recall@50/200/1000 on 2-fold holdout, cached in cache/
uv run python src/experiment.py <method> --bench   # writes answer.csv
uv run python src/eval.py answer.csv               # contract check — run before every submission
```

Methods live in `METHODS` (`src/experiment.py`); tune weights on fold0, report fold1.
Scratch grids go in the scratchpad; accepted results and rejected ideas go to README.

## Code rules

**No comments and no docstrings in code** (`.py` and notebook code cells) — user's explicit rule.
All explanation goes to `README.md` or notebook markdown cells; when adding/changing code, update
the matching README section. Enforced by `scripts/check_no_comments.py` (ruff/mypy can't express it).

```bash
uv run ruff format . && uv run ruff check .
uv run mypy                                   # strict
uv run python scripts/check_no_comments.py
uv run python tests/test_no_comments.py
```

## Gotchas

- Only ~5% of train items are in `benchmark_items` → validation corpus is built from train items
  (`src/eval.py::make_holdout`), sized to 189 212 to match the real corpus difficulty.
- `item_infm_params_text` averages ~970 chars of boilerplate; only its head is useful.
- Free RAM is tight: read parquet column subsets, avoid `item_description_raw` unless needed.
