# Question sets

`dev.jsonl` (10 items) is for tuning: chunk size, prompts, anything with a
knob. `test.jsonl` (30 items) is the reported set. It was committed on its
own, before any measurement used it, and no setting has been chosen by
looking at its scores.

## One item

```json
{"id": "t01",
 "question": "What two kinds of output does ReAct ask the model to interleave?",
 "reference_answer": "Verbal reasoning traces and task-specific actions.",
 "evidence": [{"arxiv_id": "2210.03629v3",
               "quote": "generate both reasoning traces and task-specific actions in an interleaved manner"}],
 "kind": "single",
 "names_method": true}
```

- `evidence` quotes the passages that answer the question, copied verbatim
  from the paper's extracted text (`arxiv-agent eval check-dataset` fails if
  a quote is not there, or if no stored chunk holds it). Quotes are 40-400
  characters.
- `kind`: `single` (one paper), `multi` (needs two papers; one quote from
  each), or `numeric` (the answer is a reported number).
- `names_method`: whether the question names the method or paper. Questions
  that do not are harder for retrieval and are kept at about half.

## How relevance is judged

A retrieved chunk is relevant to an item when it comes from an evidence
paper and its longest common substring with that evidence quote covers at
least 60% of the quote, after case, whitespace, quote-mark and ligature
folding. No model is involved, so retrieval scores are exact and repeatable.

## Who wrote the questions

The developer, by reading each paper's extracted text. The questions are
therefore phrased by someone who has seen the source passage, which makes
them easier for retrieval than questions from a reader who has not; the
reported numbers carry that bias.
