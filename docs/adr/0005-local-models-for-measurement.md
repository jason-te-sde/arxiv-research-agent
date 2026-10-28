# 5. Published numbers are measured with local models

## Context

The project supports Claude (`langchain-anthropic`) and local models
(Ollama). No Claude API key was available for measurement.

## Decision

The agent is `qwen3:8b` with thinking off; the RAGAS judge is
`llama3.1:8b`, a different model family so the judge is not grading its own
outputs. Both run greedy (temperature 0) with a fixed seed. Each run records
the Ollama digest of every model it used.

The Claude path is implemented and unit-tested (request shape: no
`temperature`, `top_p` or `top_k`, which current models reject; server-side
refusal fallbacks enabled) but has not been run against the live API.

## Consequences

- Anyone with a 16 GB machine can reproduce every number without an API key.
- An 8B judge is noisier than a frontier-model judge; the unscored counts
  are reported next to every mean.
- Numbers for Claude would differ and are not claimed.

## Rejected

- Reporting numbers for a model that was never run.
