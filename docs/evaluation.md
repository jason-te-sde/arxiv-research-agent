# Evaluation

Every number in the README comes from a run committed under `eval/runs/`,
produced by the commands below on a clean checkout of the commit the run
records. This page says how they are produced, what they mean, and what they
do not cover.

## Setup the numbers depend on

| | |
| --- | --- |
| Corpus | `corpus/agent-papers.toml`: 65 papers on LLM agents, tool use, planning and retrieval augmentation, pinned to exact arXiv versions; titles checked against arXiv on ingestion |
| Text | pypdf, cleaned; everything from the bibliography on is dropped (ADR 2) |
| Chunks | 512 embedder tokens including the title prefix, 64 overlap (ADR 3) |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2`, normalized, cosine |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, 20 candidates in, 5 out |
| Agent | `qwen3:4b` on Ollama, which resolves to Qwen3-4B-Thinking-2507 (digest `359d7dd4bcda`); asked for thinking off, which this build does not honor ([known defects](#known-defects)); temperature 0.7, top-p 0.8, top-k 20, seed 0, output capped at 2048 tokens, 16384-token context; at most 3 agent calls plus one fallback call |
| Judge | `llama3.1:8b` on Ollama, temperature 0, seed 0, output capped at 2048 tokens; a different model family from the agent, so it is not grading its own writing |
| Machine | Apple M2, 16 GB, CPU for embeddings and reranking, Ollama on the same machine |

The agent is not greedy on purpose. These are the values Qwen publishes for
non-thinking use, which is what the agent asks for, and Qwen warns against
greedy decoding; the first greedy run here looped on one question for over
four hours until the output cap was added. The seed is fixed, but no second full run checked that answers
repeat, and on other hardware they will differ. Retrieval numbers involve
no sampling and repeat exactly on the same index.

## Question sets

`eval/questions/test.jsonl` (30 questions) is the reported set;
`eval/questions/dev.jsonl` (10) was used for every choice that has a knob.
Both were drafted from the extracted text of 24 of the 65 papers, split by a
seeded random draw stratified by kind, and the test set was committed on its
own before any measurement ran on it (`git log -- eval/questions/test.jsonl`).

| | test | dev |
| --- | --- | --- |
| single-paper | 18 | 6 |
| two-paper | 6 | 2 |
| numeric answer | 6 | 2 |
| question does not name the method | 11 | 3 |

Each question carries verbatim evidence quotes; `arxiv-agent eval
check-dataset` confirms every quote occurs in its paper and in a stored chunk.

## Reproducing

```bash
uv sync --extra eval
ollama pull qwen3:4b-instruct-2507-q4_K_M && ollama pull llama3.1:8b
arxiv-agent ingest --corpus corpus/agent-papers.toml
arxiv-agent eval check-dataset --set test
arxiv-agent eval retrieval --set test            # MRR@5, Hit@5, Recall@5, dense and two-stage
arxiv-agent eval answers --set test              # runs the agent on 30 questions
arxiv-agent eval judge eval/runs/<run>-answers-test
arxiv-agent eval dashboard                       # docs/eval/dashboard.html
```

## Retrieval metrics

A retrieved chunk is relevant if it holds at least 60% of an evidence quote
from the right paper (ADR 4); no model is involved, so the numbers repeat
exactly on the same index.

- **MRR@5**: mean over questions of 1/rank of the first relevant chunk in the
  top 5, or 0 if there is none.
- **Hit@5**: share of questions with a relevant chunk in the top 5.
- **Recall@5**: share of evidence quotes found in the top 5, averaged over
  questions (two-paper questions have two quotes).

Retrieval is scored on the question exactly as written, which is what the
agent's fallback does; the agent itself usually searches with its own,
shorter queries.

## Answer metrics (RAGAS 0.4.3)

| Metric | RAGAS class | Question it answers |
| --- | --- | --- |
| context precision | `LLMContextPrecisionWithReference` | are the passages the agent retrieved useful for the reference answer, with useful ones ranked first |
| context recall | `LLMContextRecall` | do the retrieved passages contain what the reference answer says |
| faithfulness | `Faithfulness` | is every claim in the answer supported by the retrieved passages |
| answer relevancy | `ResponseRelevancy` | does the answer address the question (embedding similarity of generated questions) |

Retrieved context means every passage any tool returned during the run, in
order. Citation markers are removed from answers before judging. A value the
judge could not produce is counted as unscored and left out of the mean,
never averaged in as zero; the counts are reported next to the means. An
answer given without retrieving anything scores zero on the three context
metrics.

## Results

### Retrieval, test set

Run `eval/runs/20261029T041634Z-retrieval-test`, 30 questions.

| | MRR@5 | Hit@5 | Recall@5 |
| --- | --- | --- | --- |
| dense top 5 | 0.486 | 0.633 | 0.600 |
| dense top 20, reranked to top 5 | **0.650** | 0.667 | 0.633 |

Reranking adds 0.164 to MRR@5, with a 95% interval of +0.039 to +0.297
(paired bootstrap over the 30 questions, as `arxiv-agent eval dashboard`
computes it). The gain is real but its size is uncertain. On the 10 dev
questions the same comparison gives +0.230, with an interval of −0.050 to
+0.490 that ten questions cannot separate from zero.

The reranker raises the first relevant passage on 8 questions, lowers it on
2 (B07 from rank 1 to 2; A09 from rank 3 out of the top 5) and leaves 20
unchanged. Most of the gain is in ordering: one more question gets a
relevant passage into the top 5.

Ten questions have no relevant passage in the reranked top 5. Five of them
are among the 11 questions that do not name the method, and three are
two-paper questions (of 6), where both papers' passages have to compete for
five slots. This run does not record whether the missing passage was among
the 20 candidates, so it cannot say how much of the remaining error is the
reranker's and how much the embedder's.

### Answers, test set

Run `eval/runs/20261029T041722Z-answers-test`: the agent at commit `3bca58f`
on the 30 test questions, judged by `llama3.1:8b`. Intervals are percentile
bootstrap over questions, as the dashboard computes them.

| metric | mean | 95% interval | scored |
| --- | --- | --- | --- |
| context precision | 0.804 | 0.677 to 0.919 | 30 of 30 |
| context recall | 0.768 | 0.626 to 0.892 | 30 of 30 |
| faithfulness | 0.746 | 0.618 to 0.859 | 28 of 30 |
| answer relevancy | 0.662 | 0.509 to 0.802 | 30 of 30 |

The judge timed out on two faithfulness scores (A04, C08); they are left out
of the mean, not counted as zero.

27 questions were answered by the agent within its three calls, using 2.6
calls on average. 3 fell back: two reached the budget and one answered
without retrieving a second time. 27 answers carry at least one verified
citation; 2 citations, in A09 and B02, named passages the agent never
retrieved and were flagged. Median latency was 276 s and the 95th
percentile 503 s, with swap in use. The longest prompt was 12,051 tokens, so
with the 2,048-token output cap nothing was cut from the 16,384-token
context.

Every metric is below its dev value, as expected of a set nothing was tuned
on, and further below than the dev intervals would suggest. Most of the gap
is in the two defects below. On the 19 questions where the agent retrieved
from a right paper, the means are 0.949, 0.905, 0.813 (17 scored) and 0.774;
that split was chosen after reading the answers, so it explains the gap
rather than measuring the system.

### Chunk size, dev set

Chosen on the dev set before the test set was used (ADR 3); runs
`20261028T202521Z-retrieval-dev` (256) and `20261028T202538Z-retrieval-dev`
(512).

| chunk / overlap | dense MRR@5 | two-stage MRR@5 | two-stage Hit@5 | two-stage Recall@5 |
| --- | --- | --- | --- | --- |
| 256 / 32 | 0.483 | 0.608 | 0.80 | 0.70 |
| 512 / 64 | 0.520 | 0.750 | 0.80 | 0.75 |

Ten questions is small, and the two-stage difference comes from three of
them: B01 found at rank 1 instead of missed, D04 moved from rank 3 to 1, and
A07 fell from rank 4 out of the top 5. 512 was kept on that evidence and
because the reranker reads up to 512 tokens, so it sees whole passages; the
embedder reads only the first 256 of them (ADR 3).

### Answers, dev set

Run `eval/runs/20261029T023406Z-answers-dev`. The dev set is where the
prompts and the grounding rule were tuned, so these numbers are optimistic
by construction and are shown only next to the test numbers, never instead
of them.

| context precision | context recall | faithfulness | answer relevancy |
| --- | --- | --- | --- |
| 0.960 | 1.000 | 0.880 | 0.887 |

All 10 answered with citations, none unverified; 2 reached the three-call
budget and were answered by the fallback. The rule that sends an answer
without retrieval back once was added after an earlier dev run with
`qwen3:8b` answered 5 of 10 questions from the model's own memory.

## Known defects

Found by reading the test answers after the run. The first two are defects
in the agent, not fixed in the code the numbers above describe; fixing
either changes every answer number, so a fix needs a new dev run and a new
test run. The third is about the judge.

### The model's reasoning is part of every answer

The `qwen3:4b` tag on Ollama has the same digest as
`qwen3:4b-thinking-2507-q4_K_M`, a build that always thinks: its prompt
template opens every reply with `<think>`. The agent asks for thinking off
(`reasoning=False` in `agent/models.py`). With this build that does not stop
the thinking; it only stops Ollama from separating it, so the reasoning
arrives as answer text.

- All 40 answers, dev and test, open with the model deliberating ("Okay,
  let's break this down...").
- 25 of the 30 test answers then close with `</think>` and give a short
  final answer after it.
- 4 (A05, A08, C01, C08) stop mid-sentence: thinking and answer share the
  2,048-token output cap, and the cap came first. One more (C07) ends
  without the closing tag.

The judge therefore reads the deliberation plus the answer. Faithfulness
checks claims the model made while thinking aloud, guesses included. Answer
relevancy generates questions from that text: 8 test answers score exactly
0, which RAGAS assigns when it reads a response as noncommittal; no dev
answer did. Citation counts include citations made while thinking.

Two fixes would work: the instruct build (`qwen3:4b-instruct-2507`), which
does not think and is the build the sampling values above are published
for, or this build with thinking on, so that Ollama separates it, and a much
larger output cap.

### The agent often searches the wrong paper

`search_papers` takes an optional `arxiv_id` to search one paper, and the
agent picks ids from the list of ingested papers in its prompt. On 11 of the
30 test questions, every passage it retrieved came from a paper other than
the one the question is about. C01 and C08 searched HuggingGPT
(`2303.17580v4`) for Reflexion (`2303.11366v4`); D07 searched AgentTuning
(`2310.12823v2`) for Self-RAG (`2310.11511v1`). In C01 and D07 the model's
reasoning notices the mismatch, and neither searches again.

The question as written, searched with no filter, does better: it puts a
relevant passage in the top 5 for 20 questions, while the agent reached any
passage of a right paper, a weaker test, on 19.

### The judge credits passages that cannot hold the answer

A09 asks for an accuracy reported only in the Plan-and-Solve paper (76.7%)
and B10 for a count reported only in HuggingGPT (130 requests). The agent
retrieved from other papers only (Chain-of-Thought and AgentBench), and the
judge still scored context recall 1.0 on both. Over the 11 wrong-paper
questions, context recall averages 0.53 and context precision 0.55. Some of
those passages are from related papers and may help, but A09 and B10 show a
lenient judge, and no human grading measured how often it is.

## What these numbers do not show

- **Small sets.** 30 test questions: one question moves Hit@5 by 0.033 and a
  RAGAS mean by up to 0.033. Differences of a few points are not findings.
- **How the questions were written.** Every question and its evidence
  quotes were written with the source passage in view, so their wording is
  closer to the papers than a stranger's would be. 11 of the 30 avoid naming
  the method to push against that; it is still a bias toward retrieval
  succeeding.
- **One small judge.** RAGAS scores come from a local 8B judge, with no
  check against human grading, and it is lenient on context in at least two
  cases (above). Read the per-question scores in `scores.jsonl`, not only
  the means.
- **What "context" is.** Context metrics read every passage any tool
  returned during the run, in order, including passages the answer did not
  use.
- **The Claude provider.** The Anthropic adapter is covered by unit tests
  only; no reported number comes from it.
- **Latency.** Measured on a 16 GB laptop running the model, the embedder and
  the reranker together, with swap in use. It describes this machine, not the
  design.
