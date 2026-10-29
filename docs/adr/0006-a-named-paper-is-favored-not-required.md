# 6. A search that names a paper favors it instead of searching only it

## Context

`search_papers` takes an optional `arxiv_id`, and it used to search inside
that paper only. The agent copies the id from the list of 65 papers in its
prompt, and a 4B model copies it wrong. With the thinking build of
`qwen3:4b`, 11 of the 30 test questions retrieved passages only from papers
other than the one asked about
([evaluation](../evaluation.md#known-defects)). With the instruct build, 5
of the 10 dev questions did. Their queries named the right paper ("Tree of
Thoughts", "WebArena", "CodeAct") while the id belonged to a neighbor in the
list: `2307.16789v2` (ToolLLM) for WebArena's `2307.13854v4`. One more
passed an id that does not exist. A search restricted to the wrong paper can
return only passages that cannot hold the answer, and the agent's three
calls leave no room to notice and search again.

## Decision

With an `arxiv_id` that names an ingested paper, `search_papers` runs the
two-stage search twice, inside that paper and over the whole library, and
returns the 5 best of both by reranker score. Both lists come from one
query and one reranker, so their scores compare, and the top 5 of the two
top-5 lists is the top 5 of all 40 candidates. Ties go to the named paper;
beyond that it gets no advantage, so nothing guarantees it a place in the
5. An id that is not ingested, or is not an id, searches the whole library
and says so. `summarize_paper` and `compare_papers` still read only the
papers they name, and `fetch_paper` points at `summarize_paper` for reading
the paper it added.

Chosen on the dev set by replaying the search calls of the instruct
model's dev run (`20261029T124034Z-answers-dev`) and judging each top 5
against the evidence quotes ([ADR 4](0004-evidence-based-relevance.md));
`scripts/replay_paper_filter.py` prints this table:

| `arxiv_id` handling | calls | MRR@5 per call | questions with an evidence passage in a top 5 |
| --- | --- | --- | --- |
| search only that paper (before) | 9 | 0.278 | 3 of 10 |
| ignore it | 10 | 0.708 | 9 of 10 |
| favor that paper (this) | 10 | 0.708 | 9 of 10 |
| favor it, and give its best passage the last place when none made the 5 | 10 | 0.708 | 9 of 10 |

The id that does not exist ended in an error before, so that call searched
nothing. The last three tie on these calls. Favoring was kept over
ignoring because its candidates are everything the unfiltered search sees
plus the named paper's own best 20, so a passage of that paper that the
library-wide search ranks below its 20 candidates can still be returned. It
is not a targeted search.

## Consequences

- Every search that names an ingested paper costs a second dense search
  and 20 more reranked candidates; a wrong id costs nothing beyond that.
- A right id can lose every passage of its own paper to better-scoring ones
  from other papers, most likely with a generic query such as "main
  results". The three dev questions whose calls named the right paper
  reached an evidence passage under every option, but a reader who fetched
  one paper and asks about it is better served by `summarize_paper`.
- The replay holds the agent's queries fixed. With the new tool the model
  may search differently; the dev and test runs made after this change
  measure the whole agent.
- Ten questions: one question moves a count by one.

## Rejected

- Filtering by title instead of id: the queries already carry the paper's
  name, and for a question that does not name its method the model guesses
  a paper either way.
- Dropping the `arxiv_id` argument: tied on dev, and the named paper's own
  candidates could no longer compete.
- Reserving the named paper's best passage the last place: tied on dev, and
  it would keep at least one passage of a paper a model names correctly, at
  the cost of one slot in five when the id is wrong. It was weighed after
  the second test run had measured favoring, and adopting it would need
  another test run; it is left open.
