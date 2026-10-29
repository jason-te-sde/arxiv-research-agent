import math

from arxiv_agent.eval.answer_eval import AnswerRecord
from arxiv_agent.eval.dataset import EvalItem, Evidence
from arxiv_agent.eval.ragas_eval import METRICS, ScoreRecord, summarize, to_sample

ITEM = EvalItem(
    "t01", "What is ReAct?", "Reasoning plus acting.", (Evidence("x", "q" * 40),), "single", True
)


def record(answer: str, contexts: tuple[str, ...]) -> AnswerRecord:
    return AnswerRecord("t01", "What is ReAct?", answer, contexts, (), (), (), False, None, 2, 1.0)


def test_sample_maps_the_record_and_strips_citation_markers() -> None:
    sample = to_sample(record("It acts [2210.03629v3#4]. And reasons [a#1; b#2].", ("ctx",)), ITEM)
    assert sample == {
        "user_input": "What is ReAct?",
        "response": "It acts. And reasons.",
        "retrieved_contexts": ["ctx"],
        "reference": "Reasoning plus acting.",
    }


def test_metrics_are_the_four_dimensions() -> None:
    assert METRICS == ("context_precision", "context_recall", "faithfulness", "answer_relevancy")


def test_summary_averages_and_counts_unscored_values_separately() -> None:
    scores = [
        ScoreRecord(
            "a",
            {
                "context_precision": 1.0,
                "context_recall": 1.0,
                "faithfulness": 0.5,
                "answer_relevancy": 0.9,
            },
        ),
        ScoreRecord(
            "b",
            {
                "context_precision": 0.0,
                "context_recall": None,
                "faithfulness": math.nan,
                "answer_relevancy": 0.7,
            },
        ),
    ]
    summary = summarize(scores)
    assert summary["n"] == 2
    assert summary["mean"]["context_precision"] == 0.5
    assert summary["mean"]["faithfulness"] == 0.5
    assert summary["mean"]["answer_relevancy"] == 0.8
    assert summary["unscored"] == {
        "context_precision": 0,
        "context_recall": 1,
        "faithfulness": 1,
        "answer_relevancy": 0,
    }


def test_summary_of_nothing_is_empty_not_a_crash() -> None:
    assert summarize([])["mean"] == dict.fromkeys(METRICS)
