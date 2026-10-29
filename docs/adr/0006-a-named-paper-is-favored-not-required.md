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
top-5 lists is the top 5 of all 40 candidates. Ties go to the named paper.
An id that is not ingested, or is not an id, searches the whole library
and says so. `summarize_paper` and `compare_papers` still read only the
papers they name.

Chosen on the dev set by replaying the search calls of the instruct
model's dev run (`20261029T124034Z-answers-dev`) and judging each top 5
against the evidence quotes ([ADR 4](0004-evidence-based-relevance.md)):

| `arxiv_id` handling | calls | MRR@5 per call | questions with an evidence passage in a top 5 |
| --- | --- | --- | --- |
| search only that paper (before) | 9 | 0.278 | 3 of 10 |
| ignore it | 10 | 0.708 | 9 of 10 |
| favor that paper (this) | 10 | 0.708 | 9 of 10 |

The id that does not exist ended in an error before, so that call searched
nothing. Ignoring the id and favoring the paper tie on these calls.
Favoring is kept because its candidates are everything the unfiltered
search sees plus the named paper's own best 20, so a model that names the
right paper keeps a targeted search.

## Consequences

- A wrong id costs a second dense search and 20 more reranked candidates,
  not the answer.
- A right id can lose a passage of its own paper to a better-scoring one
  from another paper. The three dev questions whose calls named the right
  paper reached an evidence passage under every option.
- The replay holds the agent's queries fixed. With the new tool the model
  may search differently; the dev and test runs made after this change
  measure the whole agent.
- Ten questions: one question moves a count by one.

## Rejected

- Filtering by title instead of id: the queries already carry the paper's
  name, and for a question that does not name its method the model guesses
  a paper either way.
- Dropping the `arxiv_id` argument: tied on dev, and it takes the targeted
  search away from a model that picks ids correctly.
