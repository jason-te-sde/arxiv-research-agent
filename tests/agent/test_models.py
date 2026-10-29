import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from arxiv_agent.agent.models import make_chat_model
from arxiv_agent.config import Settings


def test_ollama_agent_samples_as_qwen_recommends_with_a_fixed_seed() -> None:
    # Qwen3's model card: non-thinking mode wants temperature 0.7, top_p 0.8,
    # top_k 20, and warns that greedy decoding causes endless repetitions.
    model = make_chat_model(Settings.from_env({}), "agent")
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:4b"
    assert model.reasoning is False
    assert (model.temperature, model.top_p, model.top_k) == (0.7, 0.8, 20)
    assert model.seed == 0
    assert model.num_ctx == Settings.from_env({}).ollama_num_ctx


@pytest.mark.parametrize("role", ["agent", "judge"])
def test_ollama_output_is_capped_so_a_repetition_loop_ends(role: str) -> None:
    model = make_chat_model(Settings.from_env({}), role)  # type: ignore[arg-type]
    assert isinstance(model, ChatOllama)
    assert model.num_predict == 2048


def test_ollama_judge_decodes_greedily() -> None:
    model = make_chat_model(Settings.from_env({}), "judge")
    assert isinstance(model, ChatOllama)
    assert model.temperature == 0
    assert model.seed == 0


def test_ollama_calls_time_out_instead_of_hanging() -> None:
    model = make_chat_model(Settings.from_env({}), "agent")
    assert isinstance(model, ChatOllama)
    assert (model.client_kwargs or {}).get("timeout") == 600


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
