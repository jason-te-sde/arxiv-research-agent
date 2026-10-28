"""Chat model construction for the two providers."""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from arxiv_agent.config import Settings

Role = Literal["agent", "judge"]

# Three tool rounds of five ~256-token passages, the paper catalog and the
# conversation fit comfortably; Ollama's own default window is far smaller.
_OLLAMA_CONTEXT = 16384
_ANTHROPIC_MAX_TOKENS = 16000
# On a safety decline the API reruns the request on a fallback model it picks.
_ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_chat_model(settings: Settings, role: Role) -> BaseChatModel:
    name = settings.agent_model if role == "agent" else settings.judge_model
    if settings.provider == "ollama":
        # Greedy decoding with a fixed seed so evaluation runs repeat; thinking off
        # because the ReAct loop is the reasoning, and it would cost minutes per run.
        return ChatOllama(
            model=name,
            base_url=settings.ollama_url,
            reasoning=False,
            temperature=0,
            seed=0,
            num_ctx=_OLLAMA_CONTEXT,
        )
    # No temperature, top_p or top_k: current Claude models reject them with a 400.
    return ChatAnthropic(  # type: ignore[call-arg]
        model=name,
        max_tokens=_ANTHROPIC_MAX_TOKENS,
        betas=[_ANTHROPIC_FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )
