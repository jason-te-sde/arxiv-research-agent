# Changelog

## 0.1.0 (unreleased)

- Ingestion of arXiv papers and local PDFs: pypdf extraction with cleaning,
  token-bounded chunking, a pinned 65-paper corpus on LLM agents.
- Two-stage retrieval: MiniLM dense search over Chroma, cross-encoder rerank.
- A ReAct agent with search, summarize, fetch and compare tools, a
  three-call budget with a direct-retrieval fallback, verified citations and
  conversation memory.
- FastAPI service with SSE streaming; `arxiv-agent` CLI.
- Evaluation: model-free retrieval metrics from evidence quotes, RAGAS
  answer metrics with a local judge, run metadata and an HTML dashboard.
