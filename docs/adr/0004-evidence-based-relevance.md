# 4. Retrieval relevance judged by evidence quotes, not by a model

## Context

MRR needs a relevance judgment for every retrieved chunk. An LLM judge
would make the number depend on the judge and vary between runs.

## Decision

Every question carries verbatim quotes of the passages that answer it. A
retrieved chunk is relevant when it comes from the quoted paper and its
longest common substring with a quote covers at least 60% of the quote,
after folding case, whitespace, curly quotes and ligatures. Quotes are
40-400 characters.

Chunk overlap alone does not guarantee that a quote straddling a boundary
has 60% in one chunk: the recursive splitter carries over only whole pieces
shorter than the overlap, so two chunks cut between long paragraphs share
nothing. What guarantees it is `arxiv-agent eval check-dataset`, which
refuses a set in which any quote is missing from its paper or has no stored
chunk that counts as relevant.

## Consequences

- Retrieval numbers are exact and repeatable on the same index.
- A chunk that answers the question in other words than the quote counts as
  irrelevant, so MRR is a lower bound on what a human would judge.

## Rejected

- An LLM relevance judge: non-deterministic, and its errors would be
  indistinguishable from retrieval errors.
