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
| Agent | `qwen3:4b-instruct-2507-q4_K_M` on Ollama (Qwen3-4B-Instruct-2507, digest `0edcdef34593`), which does not think; the first test run used `qwen3:4b`, a thinking build ([known defects](#known-defects)); temperature 0.7, top-p 0.8, top-k 20, seed 0, output capped at 2048 tokens, 16384-token context; at most 3 agent calls plus one fallback call |
| Judge | `llama3.1:8b` on Ollama, temperature 0, seed 0, output capped at 2048 tokens; a different model family from the agent, so it is not grading its own writing |
| Machine | Apple M2, 16 GB, CPU for embeddings and reranking, Ollama on the same machine |

The agent is not greedy on purpose. These are the values Qwen publishes for
this instruct build, and Qwen warns against greedy decoding; the first
greedy run here, with an earlier build, looped on one question for over
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

## The test set was run twice

The first test run (`20261029T041722Z-answers-test`, commit `3bca58f`)
found two defects in the agent, by reading its answers: the model's
reasoning was part of every answer, and on 11 of the 30 questions every
passage the agent retrieved came from a paper other than the one asked
about ([known defects](#known-defects)). Both were fixed (commits `d5d7bdb` and
`2b57107`), the second with a choice made on the dev set
([ADR 6](adr/0006-a-named-paper-is-favored-not-required.md)), and the test
set was run once more (`20261029T133547Z-answers-test`, commit `e1d76e0`).
Both runs are committed; the README reports the second.

The second run is not a clean held-out measurement: the fixes were made
because of what the first run's answers showed. Neither fix targets a
question (one changes the model build, the other how a tool treats a paper
id), nothing was changed because of the second run's answers, and the test
set was not run a third time. Since `e1d76e0` the source has changed only in
a comment and in the messages `fetch_paper` returns, which evaluation never
shows because it runs with fetching disabled.

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

Run `eval/runs/20261029T133547Z-answers-test`: the agent at commit
`e1d76e0` on the 30 test questions, judged by `llama3.1:8b`. Intervals are
percentile bootstrap over questions, as the dashboard computes them.

| metric | mean | 95% interval | scored |
| --- | --- | --- | --- |
| context precision | 0.914 | 0.850 to 0.967 | 30 of 30 |
| context recall | 0.809 | 0.702 to 0.904 | 29 of 30 |
| faithfulness | 0.775 | 0.674 to 0.872 | 29 of 30 |
| answer relevancy | 0.693 | 0.565 to 0.807 | 30 of 30 |

The judge timed out on one score (A10, context recall) and returned output
it could not parse for another (C10, faithfulness); both are left out of
the mean, not counted as zero.

All 30 questions were answered by the agent within its three calls (2.9 on
average); none fell back. 28 answers carry at least one verified citation
and none cites a passage the agent did not retrieve; A05 and C03 cite
nothing. Median latency was 33 s and the 95th percentile 61 s. The longest
prompt was 7,037 tokens. No answer contains a think tag or opens with
deliberation, and answers average 728 characters.

Against the first test run, question by question (paired bootstrap over the
questions both runs scored):

| metric | change | 95% interval | questions |
| --- | --- | --- | --- |
| context precision | +0.110 | +0.002 to +0.226 | 30 |
| context recall | +0.016 | −0.107 to +0.158 | 29 |
| faithfulness | +0.043 | −0.090 to +0.183 | 27 |
| answer relevancy | +0.031 | −0.139 to +0.207 | 30 |

Only context precision rose by more than 30 questions can separate from
zero: the agent now retrieves from a right paper on 29 questions instead of
19. The other three changes are within the noise of this set and this
judge, although the answers changed in kind: they no longer carry the
model's deliberation, and 5 score zero on answer relevancy instead of 8.
A05, the one question whose passages all came from other papers, named no
paper in its search; the library-wide search missed ReWOO, and the answer
says the passages do not hold what was asked.

### Answers, test set, first run (before the fixes)

Run `eval/runs/20261029T041722Z-answers-test`: commit `3bca58f`, agent
`qwen3:4b` (the thinking build), judged by `llama3.1:8b`.

| metric | mean | 95% interval | scored |
| --- | --- | --- | --- |
| context precision | 0.804 | 0.677 to 0.919 | 30 of 30 |
| context recall | 0.768 | 0.626 to 0.892 | 30 of 30 |
| faithfulness | 0.746 | 0.618 to 0.859 | 28 of 30 |
| answer relevancy | 0.662 | 0.509 to 0.802 | 30 of 30 |

The judge timed out on two faithfulness scores (A04, C08). 27 questions
were answered within three calls (2.6 on average) and 3 fell back: two
reached the budget and one answered without retrieving a second time. 27
answers carry a verified citation; 2 citations, in A09 and B02, named
passages the agent never retrieved and were flagged. Median latency was
276 s and the 95th percentile 503 s, with swap in use and the model writing
its reasoning before every answer (4,815 characters on average). The
longest prompt was 12,051 tokens.

On the 19 questions where the agent retrieved from a right paper, the means
were 0.949, 0.905, 0.813 (17 scored) and 0.774. That split was chosen after
reading the answers, so it explains the gap to dev rather than measuring
the system.

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

The dev set is where the prompts, the grounding rule and the search fix
were chosen, so its numbers are optimistic by construction and are shown
only next to the test numbers, never instead of them.

| run | agent | context precision | context recall | faithfulness | answer relevancy |
| --- | --- | --- | --- | --- | --- |
| `20261029T023406Z-answers-dev` | `qwen3:4b` (thinking build), search restricted to the named paper | 0.960 | 1.000 | 0.880 | 0.887 |
| `20261029T125029Z-answers-dev` | instruct build, search favors the named paper | 0.913 | 0.833 | 0.879 | 0.775 |

Between them, `20261029T124034Z-answers-dev` (instruct build, search still
restricted) has answers and no scores; it is the run ADR 6 replays.

The newer run scores lower although its answers read better: every question
retrieves from a right paper (2 of 10 did not in the older run), no answer
falls back, and none carries reasoning. Read per question, the drops do not
follow answer quality. A02's answer restates the reference answer nearly
word for word and its answer relevancy fell from 1.00 to 0.41, and the
older run scored context recall 1.0 on the two questions where it retrieved
only other papers. On ten questions this is the judge's noise; nothing was
changed because of it.

The rule that sends an answer without retrieval back once was added after
an earlier dev run with `qwen3:8b` answered 5 of 10 questions from the
model's own memory.

## Known defects

The first two were found by reading the first test run's answers and are
fixed; they are described here because that run's numbers include them.
The other two are open.

### The model's reasoning was part of every answer (fixed)

The `qwen3:4b` tag on Ollama has the same digest as
`qwen3:4b-thinking-2507-q4_K_M`, a build that always thinks: its prompt
template opens every reply with `<think>`. The agent asked for thinking off
(`reasoning=False` in `agent/models.py`). With this build that did not stop
the thinking; it only stopped Ollama from separating it, so the reasoning
arrived as answer text.

- All 40 answers, dev and test, opened with the model deliberating ("Okay,
  let's break this down...").
- 25 of the 30 test answers then closed with `</think>` and gave a short
  final answer after it.
- 4 (A05, A08, C01, C08) stopped mid-sentence: thinking and answer shared
  the 2,048-token output cap, and the cap came first. One more (C07) ended
  without the closing tag.

The judge therefore read the deliberation plus the answer. Faithfulness
checked claims the model made while thinking aloud, guesses included, and
answer relevancy generated questions from that text: 8 test answers scored
exactly 0, which RAGAS assigns when it reads a response as noncommittal.
Citation counts included citations made while thinking.

Fixed in `d5d7bdb` by the instruct build (`qwen3:4b-instruct-2507-q4_K_M`),
which does not think and is the build the sampling values above are
published for. Every answer run now counts answers that carry a think tag
(`think_tags` in its summary; a lower bound, since an answer cut off by the
output cap carries none), and every run since, two on dev and one on
test, counts none.

### The agent searched the wrong paper (fixed)

`search_papers` takes an optional `arxiv_id`, and it used to search inside
that paper only. The agent picks ids from the list of ingested papers in
its prompt. On 11 of the 30 questions of the first test run, every passage
it retrieved came from a paper other than the one the question is about.
C01 and C08 searched HuggingGPT (`2303.17580v4`) for Reflexion
(`2303.11366v4`); D07 searched AgentTuning (`2310.12823v2`) for Self-RAG
(`2310.11511v1`). In C01 and D07 the model's reasoning noticed the
mismatch, and neither searched again.

The instruct build made it worse: 5 of the 10 dev questions searched only a
wrong paper (2 with the thinking build), with queries that named the right one ("Tree of Thoughts",
"WebArena") next to the id of a neighbor in the list. Fixed in `2b57107`
([ADR 6](adr/0006-a-named-paper-is-favored-not-required.md)): a search that
names a paper now reranks that paper's candidates together with the whole
library's, and an unknown id searches the whole library. In the second test
run, 1 question of 30 retrieved only other papers (A05, whose search named
no paper).

### The judge credits passages that cannot hold the answer (open)

In the first test run, A09 asked for an accuracy reported only in the
Plan-and-Solve paper (76.7%) and B10 for a count reported only in
HuggingGPT (130 requests). The agent retrieved from other papers only
(Chain-of-Thought and AgentBench), and the judge still scored context
recall 1.0 on both. In the second test run, 12 questions retrieved no
passage that contains an evidence quote, and the judge scored context
recall 1.0 on 7 of them. A passage can state the answer in other words
than the quote (ADR 4), so that is not a count of lenient scores, but
context recall here should not be read as "the evidence was retrieved". No
human grading measured how lenient the judge is.

### A search that names the right paper can miss it (open)

Since ADR 6, naming a paper only adds its candidates to the library's, so a
generic query can return no passage of it. On dev this never cost a hit.
Reserving the named paper's best passage the last place in the 5 tied on
dev and was not adopted, because the second test run had already measured
the current behavior. `fetch_paper` points the model at `summarize_paper`,
which reads only the one paper.

## What these numbers do not show

- **Small sets.** 30 test questions: one question moves Hit@5 by 0.033 and a
  RAGAS mean by up to 0.033. Differences of a few points are not findings.
- **How the questions were written.** Every question and its evidence
  quotes were written with the source passage in view, so their wording is
  closer to the papers than a stranger's would be. 11 of the 30 avoid naming
  the method to push against that; it is still a bias toward retrieval
  succeeding.
- **A test set seen twice.** The reported test run follows fixes prompted
  by reading the first run's answers
  ([above](#the-test-set-was-run-twice)).
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
  the reranker together, with swap in use during the first test run. It
  describes this machine, not the design.
