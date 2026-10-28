import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from arxiv_agent.config import Settings
from arxiv_agent.eval.commands import cmd_check_dataset, cmd_eval_retrieval
from arxiv_agent.eval.dataset import EvalItem, Evidence
from arxiv_agent.eval.runs import new_run_dir, run_meta
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.fakes import HashingEmbeddings, chunk

QUOTE = "reasoning traces and task-specific actions in an interleaved manner"
ITEM = EvalItem(
    "t01", "react reasoning actions", "a", (Evidence("2210.03629v3", QUOTE),), "single", True
)


def store_with(tmp_path: Path, text: str) -> PaperStore:
    s = PaperStore(tmp_path / "c", HashingEmbeddings(), "cmd")
    s.replace_paper([chunk("2210.03629v3", 0, text)])
    return s


def test_check_dataset_passes_when_every_quote_is_reachable(tmp_path: Path) -> None:
    text = f"LLMs generate {QUOTE}."
    out = io.StringIO()
    code = cmd_check_dataset([ITEM], {"2210.03629v3": text}, store_with(tmp_path, text), out)
    assert code == 0
    assert "1 items, 0 problems" in out.getvalue()


def test_check_dataset_fails_when_no_chunk_holds_the_quote(tmp_path: Path) -> None:
    text = f"LLMs generate {QUOTE}."
    out = io.StringIO()
    code = cmd_check_dataset([ITEM], {"2210.03629v3": text}, store_with(tmp_path, "other"), out)
    assert code == 1
    assert "no stored chunk" in out.getvalue()


def test_eval_retrieval_writes_every_mode(tmp_path: Path) -> None:
    store = store_with(tmp_path, QUOTE)
    run_dir = tmp_path / "run"
    out = io.StringIO()
    cmd_eval_retrieval(
        {"dense": DenseRetriever(store), "two-stage": DenseRetriever(store)},
        [ITEM],
        run_dir,
        {"git_sha": "abc"},
        out,
    )
    data = json.loads((run_dir / "retrieval.json").read_text())
    assert data["meta"] == {"git_sha": "abc"}
    assert data["modes"]["dense"]["mrr"] == 1.0
    assert data["modes"]["two-stage"]["items"][0]["rank"] == 1
    assert "dense" in out.getvalue()


def test_run_dirs_are_timestamped_and_labelled(tmp_path: Path) -> None:
    path = new_run_dir(tmp_path, "retrieval-test")
    assert path.parent == tmp_path
    assert path.name.endswith("-retrieval-test")
    assert path.is_dir()


def test_run_meta_records_what_the_numbers_depend_on(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("{}\n")
    corpus = tmp_path / "corpus.toml"
    corpus.write_text("")
    meta = run_meta(Settings.from_env({}), dataset, corpus)
    for key in (
        "git_sha",
        "dataset_sha256",
        "corpus_sha256",
        "embed_model",
        "rerank_model",
        "chunk_size",
        "chunk_overlap",
        "k_candidates",
        "k_final",
        "started_at",
    ):
        assert key in meta


def test_eval_answers_writes_meta_and_answers(tmp_path: Path) -> None:
    from arxiv_agent.agent.loop import ResearchAgent
    from arxiv_agent.agent.tools import ToolContext, build_tools
    from arxiv_agent.eval.commands import cmd_eval_answers
    from tests.agent.scripted import ScriptedChatModel, say

    store = store_with(tmp_path, QUOTE)
    model = ScriptedChatModel(script=[say("An answer.")])
    retriever = DenseRetriever(store)
    agent = ResearchAgent(
        model, build_tools, lambda: ToolContext(retriever, store, model), retriever
    )
    run_dir = tmp_path / "run"
    cmd_eval_answers(agent, [ITEM], run_dir, {"agent_model": "m"}, io.StringIO())
    assert json.loads((run_dir / "meta.json").read_text()) == {"agent_model": "m"}
    assert json.loads((run_dir / "answers.jsonl").read_text())["answer"] == "An answer."


def test_eval_judge_writes_scores_and_summary(tmp_path: Path) -> None:
    from arxiv_agent.eval.answer_eval import AnswerRecord
    from arxiv_agent.eval.commands import cmd_eval_judge
    from arxiv_agent.eval.ragas_eval import ScoreRecord

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"agent_model": "m"}))
    record = AnswerRecord("t01", "q", "a", ("c",), (), (), (), False, None, 1, 2.0)
    (run_dir / "answers.jsonl").write_text(record.to_json() + "\n")

    def fake_judge(
        records: Sequence[AnswerRecord], items: Mapping[str, EvalItem]
    ) -> list[ScoreRecord]:
        assert [r.id for r in records] == ["t01"]
        assert set(items) == {"t01"}
        return [ScoreRecord("t01", {"faithfulness": 0.5})]

    out = io.StringIO()
    cmd_eval_judge(run_dir, [ITEM], fake_judge, {"judge_model": "j"}, out)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["meta"]["agent_model"] == "m"
    assert summary["meta"]["judge_model"] == "j"
    assert summary["ragas"]["mean"]["faithfulness"] == 0.5
    assert json.loads((run_dir / "scores.jsonl").read_text())["scores"] == {"faithfulness": 0.5}
    assert "faithfulness" in out.getvalue()
