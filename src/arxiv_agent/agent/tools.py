"""The agent's four tools, bound to one run's ``ToolContext``.

Each tool returns text for the model. Failures the model can act on (a paper
that is not ingested, a missing argument) come back as text too, so the agent
can recover within its budget instead of the run ending.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

from arxiv_agent.agent import prompts
from arxiv_agent.documents import Chunk
from arxiv_agent.ingest.arxiv_client import (
    ArxivClient,
    ArxivError,
    base_id,
    normalize_arxiv_id,
)
from arxiv_agent.ingest.pipeline import Ingestor
from arxiv_agent.retrieval.hits import Hit
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.retrieval.store import PaperStore

# summarize_paper reads the opening chunks (abstract, introduction) plus the
# chunks that best match a results-and-limitations query, up to this many.
_SUMMARY_OPENING = 3
_SUMMARY_CHUNKS = 8
_COMPARE_PER_PAPER = 3
_COMPARE_MAX_PAPERS = 4


@dataclass
class ToolContext:
    """Dependencies plus what one run has retrieved so far."""

    retriever: Retriever
    store: PaperStore
    model: BaseChatModel
    ingestor: Ingestor | None = None
    client: ArxivClient | None = None
    seen: dict[str, Chunk] = field(default_factory=dict)
    contexts: list[str] = field(default_factory=list)

    def record(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            if c.id not in self.seen:
                self.seen[c.id] = c
                self.contexts.append(c.embed_text)

    def resolve(self, raw: str) -> str | None:
        """The stored id for ``raw`` (versioned or not), or None if it is not ingested."""
        try:
            wanted = normalize_arxiv_id(raw)
        except ValueError:
            wanted = raw.strip()
        stored = [p.arxiv_id for p in self.store.papers()]
        if wanted in stored:
            return wanted
        return next((s for s in stored if base_id(s) == wanted), None)


def format_hits(hits: Sequence[Hit]) -> str:
    return format_chunks([h.chunk for h in hits])


def format_chunks(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.title}, p.{c.page}\n{c.text}" for c in chunks)


def _not_ingested(arxiv_id: str) -> str:
    return f"Paper {arxiv_id} is not ingested. Call fetch_paper with this arxiv_id to add it."


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    async def search_papers(query: str, arxiv_id: str | None = None) -> str:
        """Search the ingested papers for passages relevant to a query.

        Pass arxiv_id to search inside one paper only.
        """
        paper: str | None = None
        if arxiv_id:
            paper = ctx.resolve(arxiv_id)
            if paper is None:
                return _not_ingested(arxiv_id)
        hits = await asyncio.to_thread(ctx.retriever.search, query, paper)
        if not hits:
            return "No passages found."
        ctx.record([h.chunk for h in hits])
        return format_hits(hits)

    async def summarize_paper(arxiv_id: str) -> str:
        """Summarize one ingested paper: problem, method, results and limitations."""
        paper = ctx.resolve(arxiv_id)
        if paper is None:
            return _not_ingested(arxiv_id)
        chunks = await asyncio.to_thread(ctx.store.chunks_for, paper)
        hits = await asyncio.to_thread(
            ctx.retriever.search, "main results, contributions and limitations", paper
        )
        picked = {c.id: c for c in chunks[:_SUMMARY_OPENING]}
        for h in hits:
            if len(picked) >= _SUMMARY_CHUNKS:
                break
            picked.setdefault(h.chunk.id, h.chunk)
        ordered = sorted(picked.values(), key=lambda c: c.index)
        ctx.record(ordered)
        prompt = prompts.SUMMARIZE.format(
            title=ordered[0].title, arxiv_id=paper, passages=format_chunks(ordered)
        )
        reply = await ctx.model.ainvoke([HumanMessage(prompt)])
        return reply.text

    async def fetch_paper(arxiv_id: str | None = None, query: str | None = None) -> str:
        """Add a paper from arXiv to the library, by arxiv_id or by a search query.

        With a query, the best match that is not ingested yet is added.
        """
        if ctx.ingestor is None or ctx.client is None:
            return "Fetching from arXiv is disabled in this deployment."
        if not arxiv_id and not query:
            return "fetch_paper needs an arxiv_id or a query."
        if arxiv_id:
            stored = ctx.resolve(arxiv_id)
            if stored is not None:
                title = next(p.title for p in ctx.store.papers() if p.arxiv_id == stored)
                return f"{stored} ({title}) is already ingested; use search_papers on it."
            return await _ingest(ctx.ingestor, arxiv_id)
        client = ctx.client
        try:
            found = await asyncio.to_thread(client.search, query or "", 5)
        except (ArxivError, ValueError) as exc:
            return f"arXiv search failed: {exc}"
        if not found:
            return f"arXiv has no results for {query!r}."
        listing = "\n".join(f"- {m.arxiv_id}: {m.title} ({m.published})" for m in found)
        new = next((m for m in found if ctx.resolve(m.arxiv_id) is None), None)
        if new is None:
            return f"Every match is already ingested:\n{listing}"
        return f"{await _ingest(ctx.ingestor, new.arxiv_id)}\n\narXiv matches:\n{listing}"

    async def compare_papers(arxiv_ids: list[str], aspect: str) -> str:
        """Compare two to four ingested papers on one aspect, such as their method or results."""
        if len(arxiv_ids) < 2:
            return "compare_papers needs at least two arxiv_ids."
        if len(arxiv_ids) > _COMPARE_MAX_PAPERS:
            return f"compare_papers takes at most {_COMPARE_MAX_PAPERS} papers."
        resolved = {raw: ctx.resolve(raw) for raw in arxiv_ids}
        missing = [raw for raw, stored in resolved.items() if stored is None]
        if missing:
            return " ".join(_not_ingested(m) for m in missing)
        sections = []
        for stored in resolved.values():
            hits = await asyncio.to_thread(ctx.retriever.search, aspect, stored)
            hits = hits[:_COMPARE_PER_PAPER]
            ctx.record([h.chunk for h in hits])
            title = hits[0].chunk.title if hits else stored
            sections.append(f"Paper {stored} ({title}):\n{format_hits(hits)}")
        prompt = prompts.COMPARE.format(aspect=aspect, passages="\n\n".join(sections))
        reply = await ctx.model.ainvoke([HumanMessage(prompt)])
        return reply.text

    return [
        StructuredTool.from_function(
            coroutine=fn, description=inspect.cleandoc(fn.__doc__ or fn.__name__)
        )
        for fn in (search_papers, summarize_paper, fetch_paper, compare_papers)
    ]


async def _ingest(ingestor: Ingestor, arxiv_id: str) -> str:
    [result] = await asyncio.to_thread(ingestor.ingest_arxiv, [arxiv_id])
    if result.error:
        return f"Could not ingest {arxiv_id}: {result.error}"
    return (
        f"Ingested {result.arxiv_id} ({result.title}), {result.chunks} passages. "
        f"Use search_papers with arxiv_id={result.arxiv_id} to read it."
    )
