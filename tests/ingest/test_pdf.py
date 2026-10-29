from pathlib import Path

import pytest

from arxiv_agent.ingest.pdf import (
    EmptyPdfError,
    PaperText,
    PdfExtractionError,
    clean_page,
    extract_text,
    strip_references,
)
from tests.pdfgen import make_pdf

LONG = "This line is long enough to count as an ordinary wrapped line of body text"


def test_line_break_hyphenation_is_removed() -> None:
    cleaned = clean_page("we let the model reason-\ning about it")
    assert cleaned == "we let the model reasoning about it"


def test_hyphen_before_an_uppercase_word_is_kept() -> None:
    assert clean_page("the GPT-\n4 model") == "the GPT-4 model"


def test_wrapped_lines_join_with_a_space() -> None:
    assert clean_page(f"{LONG}\n{LONG}") == f"{LONG} {LONG}"


def test_blank_line_keeps_a_paragraph_break() -> None:
    assert clean_page(f"{LONG}\n\n{LONG}") == f"{LONG}\n\n{LONG}"


def test_short_line_ending_a_sentence_ends_the_paragraph() -> None:
    text = f"{LONG}\n{LONG}\nShort final line.\n{LONG}"
    assert clean_page(text) == f"{LONG} {LONG} Short final line.\n\n{LONG}"


def test_arxiv_stamp_and_page_numbers_are_dropped() -> None:
    text = f"arXiv:2210.03629v3  [cs.CL]  10 Mar 2023\n{LONG}\n12"
    assert clean_page(text) == LONG


def test_strip_references_cuts_at_the_last_heading() -> None:
    body = "x " * 400
    text = f"{body}\nReferences\n[1] A. Author. A paper. 2020."
    assert strip_references(text) == f"{body}".rstrip()


def test_strip_references_accepts_numbered_and_uppercase_headings() -> None:
    body = "x " * 400
    assert "Author" not in strip_references(f"{body}\n7 REFERENCES\n[1] Author. 2021.")
    assert "Author" not in strip_references(f"{body}\nBibliography\n[1] Author (2019)")


def test_strip_references_leaves_text_without_a_heading() -> None:
    text = "We cite related work in the references below, inline."
    assert strip_references(text) == text


def test_strip_references_ignores_a_heading_with_no_bibliography_after_it() -> None:
    text = "Contents\nReferences\n" + "body " * 400
    assert strip_references(text) == text


def test_strip_references_cuts_early_when_a_long_appendix_follows() -> None:
    body = "body " * 100
    appendix = "appendix " * 2000
    text = f"{body}\nReferences\n[1] A. Author. A paper. 2020.\n{appendix}"
    assert strip_references(text) == body.rstrip()


def test_page_of_maps_offsets_across_a_boundary() -> None:
    paper = PaperText(text="aaaa\n\nbbbb", page_starts=(0, 6))
    assert paper.page_of(0) == 1
    assert paper.page_of(5) == 1
    assert paper.page_of(6) == 2
    assert paper.page_of(9) == 2


def test_extract_text_reads_pages_in_order(tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(make_pdf([["Alpha page one."], ["Beta page two."]]))
    paper = extract_text(pdf)
    assert paper.text.index("Alpha") < paper.text.index("Beta")
    assert paper.page_of(paper.text.index("Beta")) == 2


def test_pdf_without_text_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(make_pdf([[], []]))
    with pytest.raises(EmptyPdfError):
        extract_text(pdf)


def test_unreadable_file_raises_extraction_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(PdfExtractionError):
        extract_text(bad)


def test_ligatures_are_normalized() -> None:
    assert clean_page("task-speciﬁc actions") == "task-specific actions"


def test_lines_of_font_garbage_are_dropped() -> None:
    garbage = "$FW\x03\x19 \x03*R\x03WR\x03FRXQWHUWRS\x03\x16"
    assert clean_page(f"{LONG}\n{garbage}\n{LONG}") == f"{LONG} {LONG}"


def test_stray_control_characters_are_removed() -> None:
    assert clean_page("a\x0cb normal text") == "ab normal text"


def test_a_long_first_author_list_still_reads_as_a_bibliography() -> None:
    body = "body " * 100
    authors = ", ".join(f"Author Number{n}" for n in range(45))
    text = f"{body}\nREFERENCES\n{authors}. Do as I can, not as I say. 2022.\n" + "x " * 2000
    assert strip_references(text) == body.rstrip()
