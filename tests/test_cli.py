import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from arxiv_agent import __version__
from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.agent.tools import ToolContext, build_tools
from arxiv_agent.cli import cmd_ask, cmd_ingest_corpus, main
from arxiv_agent.ingest.pipeline import IngestResult
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.agent.scripted import ScriptedChatModel, call, say
from tests.fakes import HashingEmbeddings, chunk

MANIFEST = Path(__file__).resolve().parent.parent / "corpus" / "agent-papers.toml"


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_help_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("ingest", "ask", "papers", "serve"):
        assert command in out


def test_ingest_needs_a_source() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest"])
    assert exc.value.code == 2


def test_ingest_pdf_needs_a_title() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--pdf", "paper.pdf"])
    assert exc.value.code == 2


class RecordingIngestor:
    def __init__(self, fail: Sequence[str] = ()) -> None:
        self.calls: list[tuple[list[str], dict[str, str], bool]] = []
        self.fail = set(fail)

    def ingest_arxiv(
        self,
        ids: Sequence[str],
        expected_titles: Mapping[str, str] | None = None,
        *,
        skip_existing: bool = False,
    ) -> list[IngestResult]:
        self.calls.append((list(ids), dict(expected_titles or {}), skip_existing))
        return [IngestResult(i, "T", 0, error="boom" if i in self.fail else None) for i in ids]


def test_ingest_corpus_passes_ids_and_titles_from_the_manifest() -> None:
    ingestor = RecordingIngestor()
    code = cmd_ingest_corpus(ingestor, MANIFEST, skip_existing=True, out=io.StringIO())
    assert code == 0
    ids, titles, skip = ingestor.calls[0]
    assert len(ids) >= 50
    assert titles["2210.03629v3"].startswith("ReAct")
    assert skip


def test_ingest_corpus_fails_when_any_paper_fails() -> None:
    out = io.StringIO()
    code = cmd_ingest_corpus(RecordingIngestor(fail=["2210.03629v3"]), MANIFEST, False, out)
    assert code == 1
    assert "2210.03629v3" in out.getvalue()
    assert "1 failed" in out.getvalue()


async def test_ask_prints_the_answer_and_its_sources(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "c", HashingEmbeddings(), "cli")
    store.replace_paper([chunk("2210.03629v3", 0, "react acts", title="ReAct", page=2)])
    model = ScriptedChatModel(
        script=[call("search_papers", query="react"), say("It acts [2210.03629v3#0].")]
    )
    retriever = DenseRetriever(store)
    agent = ResearchAgent(
        model, build_tools, lambda: ToolContext(retriever, store, model), retriever
    )
    out, err = io.StringIO(), io.StringIO()
    await cmd_ask(agent, "What is ReAct?", out, err)
    assert "It acts [2210.03629v3#0]." in out.getvalue()
    assert "[2210.03629v3#0] ReAct, p.2" in out.getvalue()
    assert "search_papers" in err.getvalue()


def test_eval_has_its_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--help"])
    out = capsys.readouterr().out
    assert "check-dataset" in out
    assert "retrieval" in out


def test_eval_retrieval_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["eval", "retrieval", "--mode", "sparse"])
    assert exc.value.code == 2


def test_eval_lists_answer_judge_and_dashboard(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--help"])
    out = capsys.readouterr().out
    for command in ("answers", "judge", "dashboard"):
        assert command in out
