# Security

Report vulnerabilities privately through GitHub's
[security advisory form](https://github.com/jason-te-sde/arxiv-research-agent/security/advisories/new),
not in a public issue.

## What the service exposes

`arxiv-agent serve` binds to 127.0.0.1 by default and has no authentication.
Anyone who can reach it can:

- make the server download PDFs from arxiv.org (`POST /ingest`, and the
  agent's `fetch_paper` tool);
- upload PDFs up to 50 MB, which are parsed by pypdf (`POST /ingest/pdf`);
- spend model calls on your Ollama server or Anthropic key (`POST /query`).

Do not expose it beyond localhost without an authenticating proxy in front.

## Untrusted content

Paper text reaches the model as tool results. A PDF can contain text that
tries to instruct the model; the agent's tools can only search, summarize and
fetch from arXiv, so the worst outcome is a wrong or unhelpful answer, but
answers should be read as model output.
