"""The HTTP service: ingestion endpoints and a streaming query endpoint.

``POST /query`` answers over Server-Sent Events: ``step`` for each tool call
starting and finishing, ``token`` for model text, then one ``final`` (answer,
verified citations, whether the run fell back) or one ``error``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette import EventSourceResponse

from arxiv_agent import __version__
from arxiv_agent.agent.events import Final, Token
from arxiv_agent.ingest.arxiv_client import normalize_arxiv_id
from arxiv_agent.ingest.pipeline import IngestResult, local_meta
from arxiv_agent.runtime import Runtime

log = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_KEEPALIVE_SECONDS = 15


class QueryRequest(BaseModel):
    question: str = Field(max_length=4000)
    # It is echoed in a response header, which must stay ASCII.
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is empty")
        return value.strip()


class IngestRequest(BaseModel):
    arxiv_ids: list[str] = Field(min_length=1, max_length=20)

    @field_validator("arxiv_ids")
    @classmethod
    def _valid_ids(cls, values: list[str]) -> list[str]:
        return [normalize_arxiv_id(v) for v in values]


def _result(r: IngestResult) -> dict[str, Any]:
    return {
        "arxiv_id": r.arxiv_id,
        "title": r.title,
        "chunks": r.chunks,
        "error": r.error,
        "skipped": r.skipped,
    }


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="arxiv-research-agent", version=__version__)

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "papers": len(runtime.store.papers())}

    @app.get("/papers")
    def papers() -> list[dict[str, Any]]:
        return [
            {"arxiv_id": p.arxiv_id, "title": p.title, "chunks": p.chunks}
            for p in runtime.store.papers()
        ]

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        from arxiv_agent.eval.dashboard import render_dashboard

        runs = sorted(p for p in runtime.runs_dir.glob("*") if p.is_dir())
        return render_dashboard(runs)

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, Any]:
        if runtime.ingestor is None:
            raise HTTPException(503, "ingestion is disabled")
        results = runtime.ingestor.ingest_arxiv(request.arxiv_ids)
        return {"results": [_result(r) for r in results]}

    @app.post("/ingest/pdf")
    async def ingest_pdf(file: UploadFile, title: Annotated[str, Form()]) -> dict[str, Any]:
        if runtime.ingestor is None:
            raise HTTPException(503, "ingestion is disabled")
        if not title.strip():
            raise HTTPException(422, "title is empty")
        data = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF is larger than 50 MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "upload is not a PDF")
        meta = local_meta(data, title)
        runtime.pdf_dir.mkdir(parents=True, exist_ok=True)
        path = runtime.pdf_dir / f"{meta.arxiv_id}.pdf"
        path.write_bytes(data)
        result = await run_in_threadpool(runtime.ingestor.ingest_pdf, path, meta)
        if result.error:
            raise HTTPException(422, result.error)
        return _result(result)

    @app.post("/query")
    async def query(request: QueryRequest) -> EventSourceResponse:
        session_id = request.session_id or uuid.uuid4().hex
        memory = runtime.sessions.get(session_id)

        async def events() -> AsyncIterator[dict[str, str]]:
            try:
                async for event in runtime.agent.run(request.question, memory):
                    data = event.to_dict()
                    if isinstance(event, Final):
                        data["session_id"] = session_id
                        name = "final"
                    else:
                        name = "token" if isinstance(event, Token) else "step"
                    yield {"event": name, "data": json.dumps(data)}
            except Exception:
                # The run broke outside the agent's own fallbacks; details stay in the log.
                log.exception("query failed")
                message = {"message": "The agent failed unexpectedly; see the server log."}
                yield {"event": "error", "data": json.dumps(message)}

        return EventSourceResponse(
            events(), headers={"X-Session-Id": session_id}, ping=_KEEPALIVE_SECONDS
        )

    return app
