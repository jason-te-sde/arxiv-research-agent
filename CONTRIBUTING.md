# Contributing

## Setup

```bash
uv sync --extra dev --extra eval
source .venv/bin/activate
```

## Before a pull request

```bash
bash scripts/preflight.sh
```

It runs what CI runs: the hygiene check, `ruff check`, `ruff format --check`,
`mypy` (strict, over `src` and `tests`) and the full test suite, including the
`slow` tests that load the real embedder and cross-encoder (downloaded on
first use). Warnings fail the suite.

## Rules

- Every behaviour change comes with a test that failed before the change.
- Tests that need a chat model use `tests/agent/scripted.py`; no test calls a
  network service or a real LLM.
- Repository text is English, and commits and files carry no generated-by
  lines (`scripts/check-hygiene.sh` enforces both).
- A change that can move a reported number (chunking, cleaning, prompts,
  retrieval) is tuned on `eval/questions/dev.jsonl` only, and the new
  numbers come from re-running the commands in `docs/evaluation.md`.
