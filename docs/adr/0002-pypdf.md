# 2. pypdf for text extraction

## Context

PyMuPDF extracts PDF text well but is licensed AGPL-3.0; this project is MIT.

## Decision

Use `pypdf` (BSD-3-Clause). On the ReAct paper both libraries produced the
same text for the passage compared; pypdf took 0.39 s against PyMuPDF's
0.09 s, which does not matter at 65 papers.

Extraction is followed by cleaning (`ingest/pdf.py`): ligatures folded,
line-break hyphenation undone, the arXiv side stamp, bare page numbers and
lines of figure-font garbage dropped, paragraphs rebuilt from layout lines,
and everything from the bibliography on removed.

## Consequences

- The bibliography is found by what follows its heading (a year within 2000
  characters), not by where it sits: long appendices put it as early as 14%
  into the text. Appendices after the bibliography are dropped with it, so
  facts that appear only in an appendix cannot be retrieved.
- pypdf warns that it lacks `fontTools` for some embedded CFF fonts (the
  warnings named fonts such as MyriadPro, common in figures). Whether
  installing it changes any body text was not measured.

## Rejected

- PyMuPDF: AGPL.
- pdfminer.six / pdfplumber: slower, and no better on these papers' main text.
