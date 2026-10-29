import itertools

import pytest

from arxiv_agent.documents import Chunk, PaperMeta
from arxiv_agent.ingest.chunking import SPECIAL_TOKENS, chunk_paper, hf_token_length
from arxiv_agent.ingest.pdf import PaperText

META = PaperMeta(
    arxiv_id="2210.03629v3",
    title="ReAct: Synergizing Reasoning and Acting",
    authors=("Shunyu Yao",),
    published="2022-10-06",
    abstract="",
    pdf_url="https://arxiv.org/pdf/2210.03629v3",
)


def words(text: str) -> int:
    return len(text.split())


def paper(n_paragraphs: int = 12, words_per: int = 40) -> PaperText:
    paras = [" ".join(f"p{p}w{w}" for w in range(words_per)) + "." for p in range(n_paragraphs)]
    text = "\n\n".join(paras)
    starts = (0, text.index("p6w0")) if n_paragraphs > 6 else (0,)
    return PaperText(text=text, page_starts=starts)


def test_every_chunk_fits_the_budget_with_its_title_and_special_tokens() -> None:
    chunks = chunk_paper(paper(), META, words, size=60, overlap=10)
    assert chunks
    assert all(words(c.embed_text) + SPECIAL_TOKENS <= 60 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_paper(paper(n_paragraphs=1, words_per=200), META, words, size=60, overlap=10)
    assert len(chunks) > 2
    for a, b in itertools.pairwise(chunks):
        assert set(a.text.split()) & set(b.text.split())


def test_no_overlap_when_overlap_is_zero() -> None:
    chunks = chunk_paper(paper(n_paragraphs=1, words_per=200), META, words, size=60, overlap=0)
    for a, b in itertools.pairwise(chunks):
        assert not set(a.text.split()) & set(b.text.split())


def test_ids_are_contiguous_and_prefixed_by_the_arxiv_id() -> None:
    chunks = chunk_paper(paper(), META, words, size=60, overlap=10)
    assert [c.id for c in chunks] == [f"2210.03629v3#{i}" for i in range(len(chunks))]
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_page_is_where_the_chunk_starts() -> None:
    chunks = chunk_paper(paper(), META, words, size=60, overlap=0)
    first_on_two = next(c for c in chunks if "p6w0" in c.text)
    assert chunks[0].page == 1
    assert first_on_two.page == 2


def test_all_text_is_covered() -> None:
    p = paper()
    chunks = chunk_paper(p, META, words, size=60, overlap=0)
    assert set(" ".join(c.text for c in chunks).split()) == set(p.text.split())


def test_empty_paper_has_no_chunks() -> None:
    assert chunk_paper(PaperText(text="", page_starts=(0,)), META, words, 60, 10) == []


def test_title_that_leaves_no_room_is_rejected() -> None:
    with pytest.raises(ValueError, match="title"):
        chunk_paper(paper(), META, words, size=10, overlap=4)


def test_embed_text_prefixes_the_title() -> None:
    c = Chunk(id="x#0", arxiv_id="x", title="T", page=1, index=0, text="body")
    assert c.embed_text == "T\nbody"


@pytest.mark.slow
def test_real_tokenizer_budget_holds() -> None:
    length = hf_token_length("sentence-transformers/all-MiniLM-L6-v2")
    text = "Reasoning traces help the model induce and update action plans. " * 200
    chunks = chunk_paper(PaperText(text=text, page_starts=(0,)), META, length, 256, 32)
    assert len(chunks) > 3
    assert all(length(c.embed_text) + SPECIAL_TOKENS <= 256 for c in chunks)
