# 1. A hand-written ReAct loop over LangChain parts

## Context

The agent has to stop after three of its own model calls, fall back to direct
retrieval when it does, check every citation against what the tools returned, and do
all of that the same way on every run so it can be tested exactly.
LangChain 1.x offers `create_agent`, a prebuilt loop on LangGraph.

## Decision

`agent/loop.py` is a short loop of our own (about 150 lines of logic). It
uses LangChain for everything else: the chat model interface and its
Anthropic and Ollama implementations, `StructuredTool`, the recursive text
splitter and the Chroma vector store.

## Consequences

- The budget, the four fallback reasons (`budget_exhausted`, `empty_answer`,
  `ungrounded_answer`, `model_error`, the last covering provider refusals),
  the rule that an answer given before any retrieval is sent back once, and
  the citation check are plain code,
  tested with a scripted chat model that replays exact replies
  (`tests/agent/test_loop.py`).
- The full call count per question is at most three agent calls plus one
  fallback answer call; `summarize_paper` and `compare_papers` each make one
  more model call inside the tool. Tool calls requested on the last agent
  call are not run, since nothing would read their results.
- Every run ends in exactly one `Final` event, and memory is updated before
  it is emitted, so a client that stops reading at `Final` loses nothing.
- Cost: no LangGraph checkpointing, interrupts or human-in-the-loop. Adding
  them would mean adopting LangGraph rather than extending this loop.

## Rejected

- `create_agent`: the iteration limit and recovery behaviour live inside
  the graph; asserting "exactly three calls, then one fallback call" would
  test LangGraph's internals rather than this project's rules.
