import io
import json
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
