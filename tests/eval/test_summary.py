import httpx

from arxiv_agent.eval.answer_eval import AnswerRecord
from arxiv_agent.eval.ragas_eval import ScoreRecord
from arxiv_agent.eval.runs import build_summary, ollama_digests


def test_summary_joins_answers_scores_and_meta() -> None:
    record = AnswerRecord(
        "t01", "q", "a", ("c",), ("x#1",), (), ("search_papers",), False, None, 2, 9.0
    )
    scores = [
        ScoreRecord(
            "t01",
            {
                "context_precision": 1.0,
                "context_recall": 0.5,
                "faithfulness": 0.75,
                "answer_relevancy": 0.9,
            },
        )
    ]
    summary = build_summary({"dataset": "test.jsonl"}, [record], scores)
    assert summary["meta"]["dataset"] == "test.jsonl"
    assert summary["ragas"]["mean"]["faithfulness"] == 0.75
    assert summary["operations"]["n"] == 1
    assert summary["items"] == [
        {
            "id": "t01",
            "question": "q",
            "fallback": False,
            "reason": None,
            "iterations": 2,
            "latency_s": 9.0,
            "citations": 1,
            "scores": scores[0].scores,
        }
    ]


def test_ollama_digests_come_from_the_tags_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:8b", "digest": "500a1f067a9f"},
                    {"name": "other:1b", "digest": "x"},
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    got = ollama_digests("http://localhost:11434", ["qwen3:8b", "llama3.1:8b"], http)
    assert got == {"qwen3:8b": "500a1f067a9f", "llama3.1:8b": "not installed"}


def test_ollama_digests_survive_a_server_that_is_down() -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    assert ollama_digests("http://x", ["qwen3:8b"], http) == {"qwen3:8b": "unavailable"}
