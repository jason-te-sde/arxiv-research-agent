"""The agent's four tools, bound to one run's ``ToolContext``.

Each tool returns text for the model. Failures the model can act on (a paper
that is not ingested, a missing argument) come back as text too, so the agent
can recover within its budget instead of the run ending.
"""

from __future__ import annotations

import asyncio
import inspect
import re
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
_LOCAL_ID = re.compile(r"local-[0-9a-f]{12}")


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
        """The stored id for ``raw``, or None if that paper is not ingested.

        Accepts what a model is likely to pass: a bare or versioned arXiv id, a
        chunk id (``2210.03629v3#12``), another version of a stored paper (which
        resolves to the stored one), or an upload id. Raises ValueError when
        ``raw`` is not a paper id at all, such as a title.
        """
        candidate = re.sub(r"#\d+$", "", raw.strip())
        if not _LOCAL_ID.fullmatch(candidate):
            candidate = normalize_arxiv_id(candidate)
        if self.store.has_paper(candidate):
            return candidate
        base = base_id(candidate)
        return next((p.arxiv_id for p in self.store.papers() if base_id(p.arxiv_id) == base), None)


def format_hits(hits: Sequence[Hit]) -> str:
    return format_chunks([h.chunk for h in hits])


def format_chunks(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.title}, p.{c.page}\n{c.text}" for c in chunks)


def _not_ingested(arxiv_id: str) -> str:
    return f"Paper {arxiv_id} is not ingested. Call fetch_paper with this arxiv_id to add it."


def _not_an_id(raw: str) -> str:
    return (
        f"{raw!r} is not an arXiv id. Use an id from the paper list, such as 2210.03629v3, "
        "or call search_papers without arxiv_id."
    )


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    async def lookup(raw: str) -> tuple[str | None, str]:
        """(stored id, "") or (None, the message to give the model instead)."""
        try:
            stored = await asyncio.to_thread(ctx.resolve, raw)
        except ValueError:
            return None, _not_an_id(raw)
        return (stored, "") if stored else (None, _not_ingested(raw))

    async def search_papers(query: str, arxiv_id: str | None = None) -> str:
        """Search the ingested papers for passages relevant to a query.

        Pass arxiv_id to search inside one paper only.
        """
        paper: str | None = None
        if arxiv_id:
            paper, problem = await lookup(arxiv_id)
            if paper is None:
                return problem
        hits = await asyncio.to_thread(ctx.retriever.search, query, paper)
        if not hits:
            return "No passages found."
        ctx.record([h.chunk for h in hits])
        return format_hits(hits)

    async def summarize_paper(arxiv_id: str) -> str:
        """Summarize one ingested paper: problem, method, results and limitations."""
        paper, problem = await lookup(arxiv_id)
        if paper is None:
            return problem
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
            try:
                stored = await asyncio.to_thread(ctx.resolve, arxiv_id)
            except ValueError:
                return _not_an_id(arxiv_id)
            if stored is not None:
                return f"{stored} is already ingested; use search_papers on it."
            return await _ingest(ctx.ingestor, arxiv_id)
        client = ctx.client
        try:
            found = await asyncio.to_thread(client.search, query or "", 5)
        except (ArxivError, ValueError) as exc:
            return f"arXiv search failed: {exc}"
        if not found:
            return f"arXiv has no results for {query!r}."
        listing = "\n".join(f"- {m.arxiv_id}: {m.title} ({m.published})" for m in found)
        stored_bases = {base_id(p.arxiv_id) for p in await asyncio.to_thread(ctx.store.papers)}
        new = next((m for m in found if base_id(m.arxiv_id) not in stored_bases), None)
        if new is None:
            return f"Every match is already ingested:\n{listing}"
        return f"{await _ingest(ctx.ingestor, new.arxiv_id)}\n\narXiv matches:\n{listing}"

    async def compare_papers(arxiv_ids: list[str], aspect: str) -> str:
        """Compare two to four ingested papers on one aspect, such as their method or results."""
        if len(arxiv_ids) < 2:
            return "compare_papers needs at least two arxiv_ids."
        if len(arxiv_ids) > _COMPARE_MAX_PAPERS:
            return f"compare_papers takes at most {_COMPARE_MAX_PAPERS} papers."
        resolved: dict[str, str] = {}
        for raw in arxiv_ids:
            stored, problem = await lookup(raw)
            if stored is None:
                return problem
            resolved[raw] = stored
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
