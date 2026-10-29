# 3. Chunk length counted in the embedder's tokens, 512 by default

## Context

all-MiniLM-L6-v2 embeds at most 256 word pieces and silently drops the
rest. The cross-encoder reads up to 512. A chunk size counted in characters
or in another model's tokens says nothing about what either model sees.

## Decision

Chunk budgets are counted with the embedder's own tokenizer and cover the
whole model input: the title prefix, the chunk body and the two special
tokens. The default is 512 with 64 overlap, chosen on the dev set over
256/32 (two-stage MRR@5 0.750 vs 0.608, dense 0.520 vs 0.483; 10 questions)
by a rule fixed before the runs: take the higher dev two-stage MRR.

## Consequences

- Dense retrieval matches on the first half of each chunk; the reranker
  reads all of it. The dev result says that trade is worth it here; with 10
  dev questions it is a weak signal and is reported as such.
- Every chunk embeds its paper's title, so a passage that never names the
  paper still carries that context.
- One Chroma collection per (embedder, size, overlap): both configurations
  stay side by side and never mix.
- Overlap applies where the splitter breaks inside a paragraph; between
  long paragraphs consecutive chunks may share nothing, because only whole
  pieces shorter than the overlap are carried over.

## Rejected

- 256/32 (fits the embedder window exactly): lower dev MRR in both modes.
