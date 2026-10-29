# Architecture

```
                 ingestion                                    question answering
 corpus/*.toml ─┐                                    question ──► ResearchAgent (agent/loop.py)
 arXiv ids ─────┼─► ArxivClient ─► PDF ─► extract_text ─► chunk_paper        │  ≤ 3 agent calls + 1 fallback
 uploaded PDF ──┘   (3 s throttle)      (clean, cut bib)  (512 tokens)       ▼
                                                              │       tools (agent/tools.py)
                                                              ▼        search_papers ─────┐
                                              PaperStore (Chroma, cosine)  summarize_paper │
                                              MiniLM-L6 embeddings     ◄── compare_papers  │
                                                              │            fetch_paper ────┘
                                                              ▼
                                    TwoStageRetriever: dense top-20 ─► cross-encoder ─► top-5
```

## Modules

| Package | Responsibility |
| --- | --- |
| `ingest/` | arXiv API and downloads (`arxiv_client`), the pinned corpus (`corpus`), PDF text and cleaning (`pdf`), token-bounded chunking (`chunking`), PDF to stored chunks (`pipeline`) |
| `retrieval/` | Chroma store (`store`), embedder (`embeddings`), cross-encoder (`rerank`), dense and two-stage retrievers (`retriever`) |
| `agent/` | the ReAct loop and its fallback (`loop`), the four tools (`tools`), prompts, citation extraction and verification (`citations`), conversation memory and sessions (`memory`), run events (`events`), model construction (`models`) |
| `api/` | FastAPI app with SSE streaming (`app`) |
| `eval/` | question sets (`dataset`), model-free relevance (`relevance`), retrieval metrics, answer runs, RAGAS judging, run metadata, the dashboard |
| `runtime.py` | wires settings into a store, retriever, agent and ingestor |
| `cli.py` | `arxiv-agent ingest / ask / papers / serve / eval` |

## One question, step by step

1. The system prompt lists every stored paper (`id: title`), the call budget
   and, in a continuing session, a summary of turns older than the last ten.
2. Each model call streams. Text arrives as `token` events. A call that ends
   with tool calls is a reasoning step: each tool runs, its result goes back
   to the model as a tool message, and `step` events report it. A tool error
   (unknown tool, bad arguments, a paper that is not ingested) is returned to
   the model as text and costs a call, not the run.
3. A call that ends with text only is the answer, provided some tool has
   returned passages in this run; an answer given before any retrieval is
   sent back once with a correction ("search first"). Citations in it
   (`[2210.03629v3#12]`) are checked against the chunks the tools returned
   in this run; the rest are reported as unverified.
4. If three calls pass without a grounded answer, the answer is empty, the
   model fails, or the provider refuses, the run searches with the question itself
   and makes one answer call over those passages. If no model is reachable,
   the passages themselves are the answer.
5. The turn is added to memory, then exactly one `final` event is emitted.

## Invariants and where they are enforced

| Invariant | Enforced by |
| --- | --- |
| A run ends in exactly one `Final` | `ResearchAgent.run`; `tests/agent/test_loop.py::final` asserts it in every loop test |
| At most `max_iterations` agent calls, then one fallback call; tools requested on the last call are not run | `test_exhausted_budget_falls_back_to_direct_retrieval`, `test_tools_requested_on_the_last_call_are_not_run` |
| No answer is accepted before something was retrieved | `test_an_ungrounded_answer_gets_one_correction`, `test_answers_that_stay_ungrounded_fall_back` |
| An evaluation run stops on a model failure instead of recording it | `test_a_model_outage_stops_the_run_instead_of_being_recorded` |
| A citation counts only if a tool returned that chunk | `verify_citations`; `test_citation_the_tools_never_returned_is_unverified` |
| Re-ingesting a paper replaces its chunks | `PaperStore.replace_paper`; `test_replacing_a_paper_does_not_duplicate_it` (fails without the delete) |
| Chroma search equals exact search | `test_chroma_search_equals_brute_force` (fails with a narrow HNSW beam) |
| No chunk exceeds its token budget | `test_every_chunk_fits_the_budget_with_its_title_and_special_tokens`, and against the real tokenizer |
| arXiv requests are at least 3 s apart, also from concurrent callers | `test_requests_are_spaced_by_the_minimum_interval`, `test_concurrent_requests_are_still_spaced` |
| Evaluation runs cannot change the corpus | `Runtime.from_settings(allow_fetch=False)` |

## What is deliberately simple

- Sessions live in process memory (LRU, 1000 sessions). Two concurrent
  queries on one session id are not serialized.
- Tool calls within one model reply run one after another.
- Scaling past a few hundred papers would mean batch ingestion and a server
  vector database; nothing here assumes Chroma beyond `PaperStore`.
