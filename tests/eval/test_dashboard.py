import json
from pathlib import Path

from arxiv_agent.eval.dashboard import render_dashboard
from arxiv_agent.eval.stats import bootstrap_ci


def retrieval_run(root: Path, name: str, mrr_dense: float, mrr_two: float) -> Path:
    run = root / name
    run.mkdir(parents=True)
    mode = {"k": 5, "hit_rate": 0.8, "recall": 0.7, "items": [{"id": "t01", "rank": 1}]}
    (run / "retrieval.json").write_text(
        json.dumps(
            {
                "meta": {
                    "git_sha": "abcdef123",
                    "dataset": "test.jsonl",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "started_at": "2026-10-28T20:00:00Z",
                },
                "modes": {
                    "dense": {**mode, "mrr": mrr_dense},
                    "two-stage": {**mode, "mrr": mrr_two},
                },
            }
        )
    )
    return run


def answer_run(
    root: Path,
    question: str,
    name: str = "20261028T210000Z-answers-test",
    dataset: str = "test.jsonl",
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    summary = {
        "meta": {
            "git_sha": "abcdef123",
            "dataset": dataset,
            "agent_model": "qwen3:8b",
            "judge_model": "llama3.1:8b",
            "started_at": "2026-10-28T21:00:00Z",
        },
        "ragas": {
            "n": 1,
            "mean": {
                "context_precision": 0.82,
                "context_recall": 0.7,
                "faithfulness": 0.78,
                "answer_relevancy": 0.9,
            },
            "unscored": {
                "context_precision": 0,
                "context_recall": 0,
                "faithfulness": 0,
                "answer_relevancy": 0,
            },
        },
        "operations": {
            "n": 1,
            "fallback_rate": 0.0,
            "mean_iterations": 2.0,
            "latency_p50_s": 30.0,
            "latency_p95_s": 30.0,
            "cited_rate": 1.0,
            "unverified_citations": 0,
            "fallback_reasons": {},
        },
        "items": [
            {
                "id": "t01",
                "question": question,
                "fallback": False,
                "reason": None,
                "iterations": 2,
                "latency_s": 30.0,
                "citations": 1,
                "scores": {
                    "context_precision": 1.0,
                    "context_recall": 0.5,
                    "faithfulness": None,
                    "answer_relevancy": 0.9,
                },
            }
        ],
    }
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def test_no_runs_still_renders_a_page() -> None:
    html = render_dashboard([])
    assert html.startswith("<!doctype html>")
    assert "No evaluation runs yet" in html


def test_retrieval_run_shows_both_modes_and_their_scores(tmp_path: Path) -> None:
    html = render_dashboard(
        [retrieval_run(tmp_path, "20261028T200000Z-retrieval-test", 0.61, 0.75)]
    )
    assert "dense" in html and "two-stage" in html
    assert "0.610" in html and "0.750" in html
    assert "abcdef1" in html


def test_latest_retrieval_run_per_set_is_the_one_charted(tmp_path: Path) -> None:
    old = retrieval_run(tmp_path, "20261028T190000Z-retrieval-test", 0.11, 0.22)
    new = retrieval_run(tmp_path, "20261028T200000Z-retrieval-test", 0.33, 0.44)
    html = render_dashboard([old, new])
    chart = html.split('id="retrieval"')[1].split("</section>")[0]
    assert "0.330" in chart
    assert "0.110" not in chart


def test_answer_run_shows_metrics_operations_and_questions(tmp_path: Path) -> None:
    html = render_dashboard([answer_run(tmp_path, "What is ReAct?")])
    for text in ("context_precision", "faithfulness", "0.820", "qwen3:8b", "What is ReAct?"):
        assert text in html
    assert "unscored" in html


def test_question_text_is_escaped(tmp_path: Path) -> None:
    html = render_dashboard([answer_run(tmp_path, "<script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_page_defines_a_dark_mode(tmp_path: Path) -> None:
    assert "prefers-color-scheme: dark" in render_dashboard([])


def test_answer_headline_prefers_the_test_set_over_a_newer_dev_run(tmp_path: Path) -> None:
    test = answer_run(tmp_path, "test question")
    dev = answer_run(tmp_path, "dev question", "20261028T230000Z-answers-dev", "dev.jsonl")
    html = render_dashboard([test, dev])
    headline = html.split('id="answers"')[1].split("</section>")[0]
    assert "test.jsonl" in headline


def test_the_calls_statistic_says_what_it_counts(tmp_path: Path) -> None:
    html = render_dashboard([answer_run(tmp_path, "q")])
    assert "agent calls per question" in html


def with_ranks(run: Path, dense: list[int | None], two_stage: list[int | None]) -> Path:
    data = json.loads((run / "retrieval.json").read_text())
    for mode, ranks in (("dense", dense), ("two-stage", two_stage)):
        data["modes"][mode]["items"] = [{"id": f"t{i}", "rank": r} for i, r in enumerate(ranks)]
    (run / "retrieval.json").write_text(json.dumps(data))
    return run


def test_the_reranking_gain_has_a_paired_interval(tmp_path: Path) -> None:
    run = retrieval_run(tmp_path, "20261028T200000Z-retrieval-test", 0.5, 1.0)
    html = render_dashboard([with_ranks(run, [None, None, 1, 1], [1, 1, 1, 1])])
    low, high = bootstrap_ci([1.0, 1.0, 0.0, 0.0])
    section = html.split('id="retrieval"')[1].split("</section>")[0]
    assert "+0.500" in section
    assert f"{low:+.3f} to {high:+.3f}" in section
    assert "4 questions" in section


def test_ragas_means_have_intervals_over_the_scored_questions(tmp_path: Path) -> None:
    run = answer_run(tmp_path, "q")
    summary = json.loads((run / "summary.json").read_text())
    item = summary["items"][0]
    summary["items"] = [
        {**item, "id": f"t{i}", "scores": {**item["scores"], "context_precision": v}}
        for i, v in enumerate([1.0, 0.5, None])
    ]
    (run / "summary.json").write_text(json.dumps(summary))
    html = render_dashboard([run])
    low, high = bootstrap_ci([1.0, 0.5])
    section = html.split('id="answers"')[1].split("</section>")[0]
    assert f"{low:.3f} to {high:.3f}" in section
    assert "2 of 3" in section  # context_precision: one question unscored
    assert "0 of 3" in section  # faithfulness: unscored everywhere, so no interval


def test_a_judge_timeout_stored_as_nan_counts_as_unscored(tmp_path: Path) -> None:
    run = answer_run(tmp_path, "q")
    summary = json.loads((run / "summary.json").read_text())
    item = summary["items"][0]
    summary["items"] = [
        {**item, "id": f"t{i}", "scores": {**item["scores"], "context_precision": v}}
        for i, v in enumerate([1.0, 0.5, float("nan")])
    ]
    (run / "summary.json").write_text(json.dumps(summary))  # writes NaN, as the judge does
    html = render_dashboard([run])
    low, high = bootstrap_ci([1.0, 0.5])
    section = html.split('id="answers"')[1].split("</section>")[0]
    assert f"{low:.3f} to {high:.3f}" in section
    assert "2 of 3" in section
    assert "nan" not in section.lower()
