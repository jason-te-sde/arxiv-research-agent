# Configuration

Everything is set through environment variables; nothing is read from files.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARXIV_AGENT_DATA_DIR` | `data` | PDFs (`pdfs/`) and the Chroma index (`chroma/`) |
| `ARXIV_AGENT_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `ARXIV_AGENT_AGENT_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` / `claude-opus-5-5` | model that runs the agent |
| `ARXIV_AGENT_JUDGE_MODEL` | `llama3.1:8b` / `claude-opus-5-5` | RAGAS judge |
| `ARXIV_AGENT_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `ARXIV_AGENT_OLLAMA_NUM_CTX` | `16384` | Ollama context window in tokens; reserved up front as KV cache, so it sets memory use |
| `ARXIV_AGENT_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedder |
| `ARXIV_AGENT_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker |
| `ARXIV_AGENT_CHUNK_SIZE` | `512` | chunk budget in embedder tokens, title and special tokens included |
| `ARXIV_AGENT_CHUNK_OVERLAP` | `64` | overlap in embedder tokens |
| `ARXIV_AGENT_K_CANDIDATES` | `20` | dense candidates passed to the reranker |
| `ARXIV_AGENT_K_FINAL` | `5` | passages returned per search |
| `ARXIV_AGENT_MAX_ITERATIONS` | `3` | model calls before falling back to direct retrieval |

With `ARXIV_AGENT_PROVIDER=anthropic`, the Anthropic SDK reads its key from
`ANTHROPIC_API_KEY`. Requests carry no sampling parameters and opt into
server-side refusal fallbacks.

Changing the embedder or the chunking selects a different Chroma collection;
run `arxiv-agent ingest` again for it.
