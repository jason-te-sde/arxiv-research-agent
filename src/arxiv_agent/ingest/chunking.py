"""Token-bounded recursive chunking.

Lengths are counted with the embedding model's own tokenizer, and the budget
covers everything the embedder sees: the title prefix, the chunk body and the
model's special tokens. all-MiniLM-L6-v2 reads at most 256 tokens; anything
past that is silently truncated, so a chunk that overflows the budget has an
end that dense retrieval never sees.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from arxiv_agent.documents import Chunk, PaperMeta
from arxiv_agent.ingest.pdf import PaperText

LengthFunction = Callable[[str], int]

# [CLS] and [SEP], added by BERT-family sentence encoders around every input.
SPECIAL_TOKENS = 2

# Paragraph, line, sentence, word, character: the splitter tries them in order.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def hf_token_length(model_name: str) -> LengthFunction:
    """Token counter for ``model_name``'s tokenizer, without special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    return length


def chunk_paper(
    paper: PaperText,
    meta: PaperMeta,
    length: LengthFunction,
    size: int,
    overlap: int,
) -> list[Chunk]:
    if not paper.text.strip():
        return []
    body_size = size - SPECIAL_TOKENS - length(f"{meta.title}\n")
    if body_size <= overlap:
        raise ValueError(
            f"title of {meta.arxiv_id} leaves {body_size} tokens of a {size}-token budget, "
            f"not more than the {overlap}-token overlap"
        )
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        keep_separator="end",
        chunk_size=body_size,
        chunk_overlap=overlap,
        length_function=length,
        add_start_index=True,
    )
    documents = splitter.create_documents([paper.text])
    return [
        Chunk(
            id=f"{meta.arxiv_id}#{i}",
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            page=paper.page_of(int(doc.metadata["start_index"])),
            index=i,
            text=doc.page_content,
        )
        for i, doc in enumerate(documents)
    ]
