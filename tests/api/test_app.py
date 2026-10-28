import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from arxiv_agent.agent.events import AgentEvent
from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.agent.memory import ConversationMemory, SessionStore
from arxiv_agent.agent.tools import ToolContext, build_tools
from arxiv_agent.api.app import create_app
from arxiv_agent.ingest.arxiv_client import ArxivClient
from arxiv_agent.ingest.pipeline import Ingestor
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from arxiv_agent.runtime import Runtime
from tests.agent.scripted import ScriptedChatModel, call, say
from tests.fakes import HashingEmbeddings, chunk
from tests.pdfgen import make_pdf

REACT = "2210.03629v3"
FEED = (Path(__file__).parent.parent / "fixtures" / "arxiv_query.xml").read_bytes()


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


def make_runtime(tmp_path: Path, script: list[Any]) -> tuple[Runtime, ScriptedChatModel]:
    store = PaperStore(tmp_path / "chroma", HashingEmbeddings(), "api")
    store.replace_paper([chunk(REACT, 0, "react reasoning and acting", title="ReAct")])
    model = ScriptedChatModel(script=script)
    retriever = DenseRetriever(store, k=2)

    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(200, content=FEED)
        return httpx.Response(200, content=make_pdf([["Toolformer teaches itself tools."]]))

    client = ArxivClient(httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
    ingestor = Ingestor(store, lambda t: len(t.split()), 40, 5, client, tmp_path / "pdfs")
    agent = ResearchAgent(
        model=model,
        tool_factory=build_tools,
        ctx_factory=lambda: ToolContext(retriever, store, model, ingestor, client),
        retriever=retriever,
    )
    runtime = Runtime(
        store=store,
        agent=agent,
        ingestor=ingestor,
        sessions=SessionStore(lambda: ConversationMemory(10, None)),
        pdf_dir=tmp_path / "pdfs",
        runs_dir=tmp_path / "runs",
    )
    return runtime, model


def test_health_reports_the_paper_count(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).get("/healthz")
    assert response.json() == {"status": "ok", "papers": 1}


def test_papers_lists_what_is_stored(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    papers = TestClient(create_app(runtime)).get("/papers").json()
    assert papers == [{"arxiv_id": REACT, "title": "ReAct", "chunks": 1}]


def test_query_streams_steps_tokens_and_one_final(tmp_path: Path) -> None:
    runtime, _ = make_runtime(
        tmp_path, [call("search_papers", query="react"), say(f"It acts [{REACT}#0].")]
    )
    response = TestClient(create_app(runtime)).post("/query", json={"question": "What is ReAct?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [n for n, _ in events]
    assert names[0] == "step"
    assert "token" in names
    assert names[-1] == "final"
    assert names.count("final") == 1
    final = events[-1][1]
    assert final["citations"][0]["id"] == f"{REACT}#0"
    assert final["session_id"] == response.headers["x-session-id"]


def test_session_id_carries_memory_between_queries(tmp_path: Path) -> None:
    runtime, model = make_runtime(
        tmp_path,
        [
            call("search_papers", query="react"),
            say("First."),
            call("search_papers", query="react"),
            say("Second."),
        ],
    )
    client = TestClient(create_app(runtime))
    first = client.post("/query", json={"question": "one"})
    session = first.headers["x-session-id"]
    client.post("/query", json={"question": "two", "session_id": session})
    assert [m.text for m in model.inputs[2][1:]] == ["one", "First.", "two"]


def test_empty_question_is_rejected(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    assert (
        TestClient(create_app(runtime)).post("/query", json={"question": "  "}).status_code == 422
    )


def test_agent_crash_mid_stream_ends_with_an_error_event(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    nothing: list[AgentEvent] = []

    async def explode(
        question: str, memory: ConversationMemory | None = None
    ) -> AsyncIterator[AgentEvent]:
        for event in nothing:
            yield event
        raise RuntimeError("boom")

    runtime.agent.run = explode  # type: ignore[method-assign]
    response = TestClient(create_app(runtime)).post("/query", json={"question": "q"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert "boom" not in events[-1][1]["message"]


def test_ingest_by_arxiv_id(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).post("/ingest", json={"arxiv_ids": ["2302.04761v1"]})
    assert response.status_code == 200
    [result] = response.json()["results"]
    assert result["arxiv_id"] == "2302.04761v1"
    assert result["error"] is None
    assert runtime.store.has_paper("2302.04761v1")


def test_ingest_rejects_a_malformed_id(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).post("/ingest", json={"arxiv_ids": ["react"]})
    assert response.status_code == 422


def test_ingest_pdf_upload(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    pdf = make_pdf([["Self-reflection helps agents learn from mistakes."] * 5])
    response = TestClient(create_app(runtime)).post(
        "/ingest/pdf",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"title": "My Notes"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["arxiv_id"].startswith("local-")
    assert result["chunks"] >= 1
    assert runtime.store.has_paper(result["arxiv_id"])


def test_upload_that_is_not_a_pdf_is_rejected(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).post(
        "/ingest/pdf",
        files={"file": ("x.pdf", b"<html></html>", "application/pdf")},
        data={"title": "x"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("title", ["", "   "])
def test_upload_needs_a_title(tmp_path: Path, title: str) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).post(
        "/ingest/pdf",
        files={"file": ("x.pdf", make_pdf([["text"]]), "application/pdf")},
        data={"title": title},
    )
    assert response.status_code == 422


@pytest.mark.slow
def test_the_real_runtime_wires_up(tmp_path: Path) -> None:
    from arxiv_agent.config import Settings

    settings = Settings.from_env({"ARXIV_AGENT_DATA_DIR": str(tmp_path)})
    runtime = Runtime.from_settings(settings, runs_dir=tmp_path / "runs")
    response = TestClient(create_app(runtime)).get("/healthz")
    assert response.json() == {"status": "ok", "papers": 0}


def test_dashboard_renders_the_runs(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path, [])
    response = TestClient(create_app(runtime)).get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "No evaluation runs yet" in response.text


@pytest.mark.slow
def test_an_evaluation_runtime_cannot_fetch_papers(tmp_path: Path) -> None:
    from arxiv_agent.config import Settings

    settings = Settings.from_env({"ARXIV_AGENT_DATA_DIR": str(tmp_path)})
    runtime = Runtime.from_settings(settings, runs_dir=tmp_path / "runs", allow_fetch=False)
    assert runtime.ingestor is None
