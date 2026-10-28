"""Wiring: settings in, a ready store, retriever, agent and ingestor out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from arxiv_agent.agent.loop import ResearchAgent, summarizer_for
from arxiv_agent.agent.memory import ConversationMemory, SessionStore
from arxiv_agent.agent.models import make_chat_model
from arxiv_agent.agent.tools import ToolContext, build_tools
from arxiv_agent.config import Settings
from arxiv_agent.ingest.arxiv_client import ArxivClient, make_http_client
from arxiv_agent.ingest.chunking import hf_token_length
from arxiv_agent.ingest.pipeline import Ingestor
from arxiv_agent.retrieval.embeddings import make_embeddings
from arxiv_agent.retrieval.rerank import CrossEncoderReranker
from arxiv_agent.retrieval.retriever import DenseRetriever, Retriever, TwoStageRetriever
from arxiv_agent.retrieval.store import PaperStore, collection_name

RetrievalMode = Literal["dense", "two-stage"]

_MEMORY_WINDOW_TURNS = 10


def open_store(settings: Settings) -> PaperStore:
    return PaperStore(
        settings.chroma_dir,
        make_embeddings(settings.embed_model),
        collection_name(settings.embed_model, settings.chunk_size, settings.chunk_overlap),
    )


def make_retriever(settings: Settings, store: PaperStore, mode: RetrievalMode) -> Retriever:
    if mode == "dense":
        return DenseRetriever(store, k=settings.k_final)
    return TwoStageRetriever(
        store,
        CrossEncoderReranker(settings.rerank_model),
        k_candidates=settings.k_candidates,
        k=settings.k_final,
    )


def make_ingestor(settings: Settings, store: PaperStore, client: ArxivClient | None) -> Ingestor:
    return Ingestor(
        store,
        hf_token_length(settings.embed_model),
        settings.chunk_size,
        settings.chunk_overlap,
        client=client,
        pdf_dir=settings.pdf_dir,
    )


@dataclass
class Runtime:
    store: PaperStore
    agent: ResearchAgent
    ingestor: Ingestor | None
    sessions: SessionStore
    pdf_dir: Path
    runs_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings, runs_dir: Path = Path("eval/runs")) -> Runtime:
        store = open_store(settings)
        retriever = make_retriever(settings, store, "two-stage")
        client = ArxivClient(make_http_client())
        ingestor = make_ingestor(settings, store, client)
        model = make_chat_model(settings, "agent")
        agent = ResearchAgent(
            model=model,
            tool_factory=build_tools,
            ctx_factory=lambda: ToolContext(retriever, store, model, ingestor, client),
            retriever=retriever,
            max_iterations=settings.max_iterations,
        )
        summarize = summarizer_for(model)
        sessions = SessionStore(lambda: ConversationMemory(_MEMORY_WINDOW_TURNS, summarize))
        return cls(store, agent, ingestor, sessions, settings.pdf_dir, runs_dir)
