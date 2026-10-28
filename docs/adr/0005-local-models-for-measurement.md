# 5. Published numbers are measured with local models

## Context

The project supports Claude (`langchain-anthropic`) and local models
(Ollama). No Claude API key was available for measurement.

## Decision

The agent is `qwen3:4b` with thinking off; the RAGAS judge is
`llama3.1:8b`, a different model family so the judge is not grading its own
outputs. The agent was `qwen3:8b` until measurement showed it holding 7.8 GB
(5.0 GB of weights, about 2.4 GB of KV cache for a 16K window) on a 16 GB
machine; the owner chose the 4B model, accepting different numbers. During
evaluation the agent model is unloaded before the judge loads, so the two are
never resident together, and the context window is a setting sized from the
recorded prompt sizes. Both run greedy (temperature 0) with a fixed seed. Each run records
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
