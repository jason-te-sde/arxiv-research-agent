"""A read-only MCP server over the library's retrieval, for any MCP client.

Three tools: ``search_papers`` (the two-stage retrieval the agent uses),
``get_passage`` and ``list_papers``. None of them writes, reaches the network
or calls a language model. The server speaks stdio only; the SDK is the
optional ``mcp`` extra, so the base install does not need it.
"""

from __future__ import annotations

import re
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from arxiv_agent import __version__
from arxiv_agent.documents import Chunk
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.retrieval.store import PaperStore

MAX_QUERY_CHARS = 1000
MAX_K = 20
DEFAULT_K = 5
MAX_CHUNK_ID_CHARS = 128

_VERSION = re.compile(r"v\d+$")
_CHUNK_ID = re.compile(r"[A-Za-z0-9._-]+#\d+")

_READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


class Passage(BaseModel):
    chunk_id: str = Field(description="Cite as [chunk_id]; pass it to get_passage")
    paper_id: str = Field(description="arXiv id without the version")
    version: str | None = Field(description="arXiv version such as 'v3'; null for uploads")
    title: str
    page: int = Field(description="1-based page where the passage starts")
    score: float = Field(description="Cross-encoder logit: higher is better, not a probability")
    text: str


class PassageDetail(BaseModel):
    chunk_id: str
    paper_id: str
    version: str | None
    title: str
    page: int
    text: str
    citation: str = Field(description="'[chunk_id] title, p.N', as the agent cites it")


class PaperEntry(BaseModel):
    paper_id: str
    version: str | None
    title: str
    chunks: int


def _split_id(arxiv_id: str) -> tuple[str, str | None]:
    match = _VERSION.search(arxiv_id)
    if match is None:
        return arxiv_id, None
    return arxiv_id[: match.start()], match.group()


def citation(chunk: Chunk) -> str:
    return f"[{chunk.id}] {chunk.title}, p.{chunk.page}"


def create_server(store: PaperStore, retriever: Retriever) -> MCPServer:
    """Wire the three tools to one store and one retriever.

    ``retriever`` must return at least ``MAX_K`` hits for a broad query (the
    default two-stage retriever is built that way in ``arxiv-agent mcp``).
    """
    server = MCPServer(
        "arxiv-research-agent",
        instructions=(
            "Read-only search over a library of arXiv papers. search_papers returns "
            "passages; cite them by chunk_id, and use get_passage to re-read one."
        ),
        version=__version__,
    )

    @server.tool(annotations=_READ_ONLY, structured_output=True)
    def search_papers(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_QUERY_CHARS,
                description="What to look for, in natural language",
            ),
        ],
        k: Annotated[
            int, Field(ge=1, le=MAX_K, description="How many passages to return")
        ] = DEFAULT_K,
    ) -> list[Passage]:
        """Search the library: embedding search, then a cross-encoder rerank."""
        if not query.strip():
            raise ToolError("query must not be blank")
        passages = []
        for hit in retriever.search(query)[:k]:
            paper_id, version = _split_id(hit.chunk.arxiv_id)
            passages.append(
                Passage(
                    chunk_id=hit.chunk.id,
                    paper_id=paper_id,
                    version=version,
                    title=hit.chunk.title,
                    page=hit.chunk.page,
                    score=hit.score,
                    text=hit.chunk.text,
                )
            )
        return passages

    @server.tool(annotations=_READ_ONLY, structured_output=True)
    def get_passage(
        chunk_id: Annotated[
            str,
            Field(
                min_length=3,
                max_length=MAX_CHUNK_ID_CHARS,
                description="A chunk_id from search_papers, such as 2210.03629v3#12",
            ),
        ],
    ) -> PassageDetail:
        """Return one passage's text and its citation."""
        if not _CHUNK_ID.fullmatch(chunk_id):
            raise ToolError(f"{chunk_id!r} is not a chunk id; expected '<paper id>#<number>'")
        found = store.chunk(chunk_id)
        if found is None:
            raise ToolError(f"no passage {chunk_id!r} in the library")
        paper_id, version = _split_id(found.arxiv_id)
        return PassageDetail(
            chunk_id=found.id,
            paper_id=paper_id,
            version=version,
            title=found.title,
            page=found.page,
            text=found.text,
            citation=citation(found),
        )

    @server.tool(annotations=_READ_ONLY, structured_output=True)
    def list_papers() -> list[PaperEntry]:
        """List every paper in the library, with how many passages each has."""
        entries = []
        for row in store.papers():
            paper_id, version = _split_id(row.arxiv_id)
            entries.append(
                PaperEntry(paper_id=paper_id, version=version, title=row.title, chunks=row.chunks)
            )
        return entries

    return server
