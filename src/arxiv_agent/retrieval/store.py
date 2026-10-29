"""Chroma-backed chunk storage, through LangChain's Chroma vector store.

The stored document is the chunk's ``embed_text`` (title, newline, body), so
the vector Chroma keeps is exactly the one the embedder computed from it; the
body is recovered by dropping the first line.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from arxiv_agent.documents import Chunk
from arxiv_agent.retrieval.hits import Hit

# HNSW is approximate; a search beam this wide makes it exact at this corpus
# size (tests/retrieval/test_retriever.py holds it to brute force).
_EF = 256


@dataclass(frozen=True)
class PaperRow:
    arxiv_id: str
    title: str
    chunks: int


def collection_name(embed_model: str, size: int, overlap: int) -> str:
    """One collection per embedder and chunking, so configurations never mix."""
    model = re.sub(r"[^a-z0-9]+", "-", embed_model.rsplit("/", 1)[-1].lower()).strip("-")
    return f"papers-{model}-{size}-{overlap}"


class PaperStore:
    def __init__(self, persist_dir: Path | None, embeddings: Embeddings, collection: str) -> None:
        settings = ChromaSettings(anonymized_telemetry=False)
        client = (
            chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            if persist_dir is not None
            else chromadb.EphemeralClient(settings=settings)
        )
        self._db = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            client=client,
            collection_configuration={
                "hnsw": {"space": "cosine", "ef_construction": _EF, "ef_search": _EF}
            },
        )

    def replace_paper(self, chunks: Sequence[Chunk]) -> int:
        """Store one paper's chunks, removing whatever was stored for it before."""
        if not chunks:
            return 0
        arxiv_ids = {c.arxiv_id for c in chunks}
        if len(arxiv_ids) != 1:
            raise ValueError(f"replace_paper takes one paper's chunks, got {sorted(arxiv_ids)}")
        self.delete_paper(chunks[0].arxiv_id)
        self._db.add_texts(
            texts=[c.embed_text for c in chunks],
            metadatas=[
                {"arxiv_id": c.arxiv_id, "title": c.title, "page": c.page, "index": c.index}
                for c in chunks
            ],
            ids=[c.id for c in chunks],
        )
        return len(chunks)

    def delete_paper(self, arxiv_id: str) -> None:
        existing = self._db.get(where={"arxiv_id": arxiv_id}, include=[])["ids"]
        if existing:
            self._db.delete(ids=existing)

    def has_paper(self, arxiv_id: str) -> bool:
        return bool(self._db.get(where={"arxiv_id": arxiv_id}, limit=1, include=[])["ids"])

    def papers(self) -> list[PaperRow]:
        metadatas = self._db.get(include=["metadatas"])["metadatas"]
        counts = Counter(str(m["arxiv_id"]) for m in metadatas)
        titles = {str(m["arxiv_id"]): str(m["title"]) for m in metadatas}
        return [PaperRow(i, titles[i], counts[i]) for i in sorted(counts)]

    def chunks_for(self, arxiv_id: str) -> list[Chunk]:
        got = self._db.get(where={"arxiv_id": arxiv_id}, include=["documents", "metadatas"])
        chunks = [
            _to_chunk(i, d, m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda c: c.index)

    def dense_search(self, query: str, k: int, arxiv_id: str | None = None) -> list[Hit]:
        results = self._db.similarity_search_with_score(
            query, k=k, filter={"arxiv_id": arxiv_id} if arxiv_id else None
        )
        # Chroma reports cosine distance; similarity is 1 - distance.
        return [Hit(_from_document(doc), 1.0 - float(dist)) for doc, dist in results]

    def all_embeddings(self) -> tuple[list[Chunk], np.ndarray]:
        got = self._db.get(include=["documents", "metadatas", "embeddings"])
        chunks = [
            _to_chunk(i, d, m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]
        return chunks, np.asarray(got["embeddings"], dtype=np.float32)


def _to_chunk(chunk_id: str, document: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        id=chunk_id,
        arxiv_id=str(meta["arxiv_id"]),
        title=str(meta["title"]),
        page=int(meta["page"]),
        index=int(meta["index"]),
        text=document.partition("\n")[2],
    )


def _from_document(doc: Document) -> Chunk:
    meta = doc.metadata
    return _to_chunk(f"{meta['arxiv_id']}#{meta['index']}", doc.page_content, meta)
