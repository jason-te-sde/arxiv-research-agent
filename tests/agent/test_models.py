import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from arxiv_agent.agent.models import make_chat_model
from arxiv_agent.config import Settings


def test_ollama_agent_model_is_deterministic_and_does_not_think() -> None:
    model = make_chat_model(Settings.from_env({}), "agent")
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:8b"
    assert model.reasoning is False
    assert model.temperature == 0
    assert model.seed == 0
    assert model.num_ctx == Settings.from_env({}).ollama_num_ctx


def test_ollama_judge_uses_the_judge_model() -> None:
    model = make_chat_model(Settings.from_env({}), "judge")
    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.1:8b"


def test_anthropic_request_sends_no_sampling_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = make_chat_model(Settings.from_env({"ARXIV_AGENT_PROVIDER": "anthropic"}), "agent")
    assert isinstance(model, ChatAnthropic)
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["model"] == "claude-opus-5-5"
    assert not {"temperature", "top_p", "top_k", "tool_choice"} & payload.keys()
    assert payload["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in payload["betas"]
