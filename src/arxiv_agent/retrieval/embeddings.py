from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


def make_embeddings(model_name: str) -> Embeddings:
    """A sentence-transformers embedder on CPU with unit-length vectors (cosine = dot)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
