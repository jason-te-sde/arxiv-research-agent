"""The ``arxiv-agent`` command line."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

from arxiv_agent import __version__
from arxiv_agent.agent.events import Final, Token, ToolStarted
from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.config import Settings
from arxiv_agent.eval.dataset import EvalItem
from arxiv_agent.ingest.corpus import load_corpus
from arxiv_agent.ingest.pipeline import IngestResult


class ArxivIngestor(Protocol):
    def ingest_arxiv(
        self,
        ids: Sequence[str],
        expected_titles: Mapping[str, str] | None = None,
        *,
        skip_existing: bool = False,
    ) -> list[IngestResult]: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-agent",
        description="Ask research questions over arXiv papers. Settings come from "
        "ARXIV_AGENT_* environment variables (see docs/configuration.md).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    ingest = commands.add_parser("ingest", help="add papers to the library")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="a corpus manifest (corpus/*.toml)")
    source.add_argument("--id", nargs="+", dest="ids", metavar="ARXIV_ID", help="arXiv ids")
    source.add_argument("--pdf", type=Path, help="a local PDF (needs --title)")
    ingest.add_argument("--title", help="title of the local PDF")
    ingest.add_argument(
        "--skip-existing", action="store_true", help="leave papers already stored alone"
    )

    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")

    commands.add_parser("papers", help="list the stored papers")

    evaluate = commands.add_parser("eval", help="evaluate retrieval and answers")
    evals = evaluate.add_subparsers(dest="eval_command", metavar="EVAL_COMMAND", required=True)
    check = evals.add_parser("check-dataset", help="check every evidence quote is retrievable")
    retrieval = evals.add_parser("retrieval", help="MRR@5, hit rate and recall per mode")
    retrieval.add_argument("--mode", choices=["dense", "two-stage", "both"], default="both")
    answers = evals.add_parser("answers", help="run the agent over a question set")
    answers.add_argument("--resume", type=Path, help="continue an interrupted run directory")
    for sub in (check, retrieval, answers):
        sub.add_argument("--set", choices=["dev", "test"], default="test", dest="question_set")
    judge = evals.add_parser("judge", help="score an answers run with RAGAS")
    judge.add_argument("run_dir", type=Path)
    dashboard = evals.add_parser("dashboard", help="write the HTML dashboard of all runs")
    dashboard.add_argument("--out", type=Path, default=Path("docs/eval/dashboard.html"))

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest" and args.pdf is not None and not args.title:
        parser.error("--pdf needs --title")
    settings = Settings.from_env(os.environ if env is None else env)
    return _dispatch(args, settings)


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover
    # Wiring only: every branch builds real models; the commands it calls are tested.
    from arxiv_agent import runtime

    if args.command == "papers":
        for p in runtime.open_store(settings).papers():
            print(f"{p.arxiv_id}  {p.chunks:4d}  {p.title}")
        return 0
    if args.command == "serve":
        import uvicorn

        from arxiv_agent.api.app import create_app

        uvicorn.run(
            create_app(runtime.Runtime.from_settings(settings)), host=args.host, port=args.port
        )
        return 0
    if args.command == "ask":
        agent = runtime.Runtime.from_settings(settings).agent
        asyncio.run(cmd_ask(agent, args.question, sys.stdout, sys.stderr))
        return 0
    if args.command == "ingest":
        from arxiv_agent.ingest.arxiv_client import ArxivClient, make_http_client
        from arxiv_agent.ingest.pipeline import local_meta

        store = runtime.open_store(settings)
        ingestor = runtime.make_ingestor(settings, store, ArxivClient(make_http_client()))
        if args.corpus:
            return cmd_ingest_corpus(ingestor, args.corpus, args.skip_existing, sys.stdout)
        if args.ids:
            results = ingestor.ingest_arxiv(args.ids, skip_existing=args.skip_existing)
            return _report(results, sys.stdout)
        meta = local_meta(args.pdf.read_bytes(), args.title)
        return _report([ingestor.ingest_pdf(args.pdf, meta)], sys.stdout)
    if args.command == "eval":
        return _dispatch_eval(args, settings)
    raise AssertionError(f"unhandled command {args.command}")


_QUESTIONS = Path("eval/questions")
_CORPUS = Path("corpus/agent-papers.toml")
_RUNS = Path("eval/runs")


def _dispatch_eval(args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover
    from arxiv_agent import runtime
    from arxiv_agent.eval.commands import cmd_check_dataset, cmd_eval_retrieval
    from arxiv_agent.eval.dataset import load_items
    from arxiv_agent.eval.runs import new_run_dir, run_meta
    from arxiv_agent.ingest.pdf import extract_text
    from arxiv_agent.retrieval.retriever import Retriever

    if args.eval_command == "dashboard":
        from arxiv_agent.eval.dashboard import render_dashboard

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_dashboard(sorted(p for p in _RUNS.glob("*") if p.is_dir())))
        print(f"wrote {args.out}")
        return 0
    if args.eval_command == "judge":
        return _judge(args.run_dir, settings)
    dataset = _QUESTIONS / f"{args.question_set}.jsonl"
    items = load_items(dataset)
    if args.eval_command == "answers":
        return _answers(args, settings, dataset, items)
    store = runtime.open_store(settings)
    if args.eval_command == "check-dataset":
        papers = {ev.arxiv_id for item in items for ev in item.evidence}
        texts = {
            p: extract_text(settings.pdf_dir / f"{p}.pdf").text
            for p in sorted(papers)
            if (settings.pdf_dir / f"{p}.pdf").exists()
        }
        return cmd_check_dataset(items, texts, store, sys.stdout)
    if args.eval_command == "retrieval":
        modes: list[runtime.RetrievalMode] = (
            ["dense", "two-stage"] if args.mode == "both" else [args.mode]
        )
        retrievers: dict[str, Retriever] = {
            m: runtime.make_retriever(settings, store, m) for m in modes
        }
        run_dir = new_run_dir(_RUNS, f"retrieval-{args.question_set}")
        meta = run_meta(settings, dataset, _CORPUS)
        cmd_eval_retrieval(retrievers, items, run_dir, meta, sys.stdout, k=settings.k_final)
        print(f"wrote {run_dir / 'retrieval.json'}")
        return 0
    raise AssertionError(f"unhandled eval command {args.eval_command}")


def _answers(
    args: argparse.Namespace, settings: Settings, dataset: Path, items: Sequence[EvalItem]
) -> int:  # pragma: no cover - runs the real agent
    from arxiv_agent.eval.commands import cmd_eval_answers
    from arxiv_agent.eval.runs import new_run_dir, ollama_digests, ollama_unload, run_meta
    from arxiv_agent.runtime import Runtime

    run_dir = args.resume or new_run_dir(_RUNS, f"answers-{args.question_set}")
    meta = run_meta(settings, dataset, _CORPUS) | {
        "provider": settings.provider,
        "agent_model": settings.agent_model,
        "max_iterations": settings.max_iterations,
    }
    if settings.provider == "ollama":
        meta["ollama_digests"] = ollama_digests(settings.ollama_url, [settings.agent_model])
    agent = Runtime.from_settings(settings, allow_fetch=False).agent
    meta["ollama_num_ctx"] = settings.ollama_num_ctx
    try:
        cmd_eval_answers(agent, items, run_dir, meta, sys.stdout)
    finally:
        if settings.provider == "ollama":
            ollama_unload(settings.ollama_url, settings.agent_model)
    print(f"wrote {run_dir / 'answers.jsonl'}")
    return 0


def _judge(run_dir: Path, settings: Settings) -> int:  # pragma: no cover - runs the judge
    import json

    from arxiv_agent.agent.models import make_chat_model
    from arxiv_agent.eval import ragas_eval
    from arxiv_agent.eval.commands import cmd_eval_judge
    from arxiv_agent.eval.dataset import load_items
    from arxiv_agent.eval.runs import ollama_digests, ollama_unload
    from arxiv_agent.retrieval.embeddings import make_embeddings

    meta = json.loads((run_dir / "meta.json").read_text())
    items = load_items(_QUESTIONS / meta["dataset"])
    llm = make_chat_model(settings, "judge")
    embeddings = make_embeddings(settings.embed_model)
    judge_meta: dict[str, object] = {"judge_model": settings.judge_model}
    if settings.provider == "ollama":
        judge_meta["judge_digest"] = ollama_digests(settings.ollama_url, [settings.judge_model])
        # Only one 8B model resident at a time on a 16 GB machine.
        ollama_unload(settings.ollama_url, settings.agent_model)

    def judge(records: Sequence[Any], by_id: Mapping[str, Any]) -> list[Any]:
        return ragas_eval.judge(records, by_id, llm, embeddings)

    try:
        cmd_eval_judge(run_dir, items, judge, judge_meta, sys.stdout)
    finally:
        if settings.provider == "ollama":
            ollama_unload(settings.ollama_url, settings.judge_model)
    print(f"wrote {run_dir / 'summary.json'}")
    return 0


def cmd_ingest_corpus(
    ingestor: ArxivIngestor, corpus: Path, skip_existing: bool, out: TextIO
) -> int:
    entries = load_corpus(corpus)
    results = ingestor.ingest_arxiv(
        [e.id for e in entries],
        expected_titles={e.id: e.title for e in entries},
        skip_existing=skip_existing,
    )
    return _report(results, out)


def _report(results: Sequence[IngestResult], out: TextIO) -> int:
    for r in results:
        if r.error:
            status = f"FAILED: {r.error}"
        elif r.skipped:
            status = "already stored"
        else:
            status = f"{r.chunks} chunks"
        print(f"{r.arxiv_id}  {status}  {r.title}", file=out)
    failed = sum(1 for r in results if r.error)
    print(f"{len(results) - failed} ingested or present, {failed} failed", file=out)
    return 1 if failed else 0


async def cmd_ask(agent: ResearchAgent, question: str, out: TextIO, err: TextIO) -> None:
    async for event in agent.run(question):
        if isinstance(event, ToolStarted):
            print(f"-> {event.name} {event.args}", file=err)
        elif isinstance(event, Token):
            pass  # the answer is printed whole below, after the reasoning steps
        elif isinstance(event, Final):
            if event.fallback:
                print(f"(fell back to direct retrieval: {event.reason})", file=err)
            print(event.answer, file=out)
            if event.citations:
                print("\nSources:", file=out)
                for c in event.citations:
                    print(f"  [{c.id}] {c.title}, p.{c.page}", file=out)
            if event.unverified_citations:
                unverified = ", ".join(event.unverified_citations)
                print(f"\nUnverified citations: {unverified}", file=out)
