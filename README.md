# arxiv-research-agent

A research assistant that answers questions about arXiv papers with cited,
retrieval-grounded answers. It ingests PDFs into a vector index, retrieves in
two stages (embedding search, then a cross-encoder), and runs a ReAct agent
with four tools that streams its work and its answer over Server-Sent Events.
Every citation in an answer is checked against the passages the agent
actually retrieved.

The shipped corpus is 65 papers on LLM agents, tool use, planning and
retrieval augmentation, pinned to exact arXiv versions.

## Measured results

On the 30-question test set, with `qwen3:4b` as the agent and
`llama3.1:8b` as the judge, both local. Every number comes from a run
committed under [eval/runs](eval/runs);
[docs/evaluation.md](docs/evaluation.md) has the method, the intervals and
what the numbers do not show, and `arxiv-agent eval dashboard` charts them.

| Retrieval | MRR@5 | Hit@5 | Recall@5 |
| --- | --- | --- | --- |
| dense top 5 | 0.486 | 0.633 | 0.600 |
| dense top 20, reranked to top 5 | **0.650** | 0.667 | 0.633 |

Reranking adds 0.164 MRR@5, with a 95% interval of +0.039 to +0.297.

| Answers (RAGAS) | context precision | context recall | faithfulness | answer relevancy |
| --- | --- | --- | --- | --- |
| agent | 0.804 | 0.768 | 0.746 | 0.662 |

**Known defects.** The `qwen3:4b` tag is a build that always thinks, so the
model's reasoning comes out as part of every answer, and 4 of the 30 test
answers ran out of output tokens before reaching one. On 11 questions the
agent searched the wrong paper. The answer scores above include both;
[docs/evaluation.md](docs/evaluation.md#known-defects) has the details.

## How it works

- **Ingestion.** pypdf text, cleaned (ligatures, hyphenation, figure-font
  garbage, everything from the bibliography on), split recursively into
  chunks of 512 tokens of the embedder's own tokenizer, title included, with
  64 overlap. Every chunk is embedded with its paper's title.
- **Retrieval.** all-MiniLM-L6-v2 over Chroma returns 20 candidates;
  ms-marco-MiniLM-L-6-v2 reranks them and the top 5 are kept.
- **Agent.** A ReAct loop over LangChain chat models with four tools:
  `search_papers`, `summarize_paper`, `fetch_paper` (from arXiv) and
  `compare_papers`. It has three model calls and must retrieve before it
  answers (an answer from the model's own memory is sent back once). If it
  has not answered from passages by then, or the model fails, it searches
  with the question itself and answers from those passages in one call, or
  returns the passages if no model is reachable. Conversation memory keeps ten turns and summarizes older ones.
- **Service.** FastAPI: `POST /query` streams `step`, `token` and `final`
  events; `POST /ingest` takes arXiv ids, `POST /ingest/pdf` a PDF;
  `GET /dashboard` shows the evaluation runs.

Design decisions and what they cost are in [docs/adr](docs/adr); the
module map and the invariants each test pins are in
[docs/architecture.md](docs/architecture.md).

## Quick start

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and
[Ollama](https://ollama.com) (or an Anthropic API key, see
[configuration](docs/configuration.md)).

```bash
uv sync
ollama pull qwen3:4b
uv run arxiv-agent ingest --corpus corpus/agent-papers.toml
uv run arxiv-agent ask "How does Reflexion turn failures into feedback?"
uv run arxiv-agent serve        # http://127.0.0.1:8000
```

The first ingest downloads the 65 PDFs (about 300 MB) from arXiv, three
seconds apart as arXiv asks; indexing them took about two minutes on an M2.

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Compare ReAct and ReWOO on token use"}'
```

## Development

```bash
uv sync --extra dev --extra eval
bash scripts/preflight.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The service has no authentication;
read [SECURITY.md](SECURITY.md) before exposing it.

## License

MIT. The papers are not redistributed: `arxiv-agent ingest` downloads them
from arXiv, each under its own license.
