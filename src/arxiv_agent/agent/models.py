"""Chat model construction for the two providers."""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from arxiv_agent.config import Settings

Role = Literal["agent", "judge"]

# A hung Ollama would otherwise hang /query forever. Generous: a long prompt on a
# CPU-bound laptop model can take minutes to evaluate.
_OLLAMA_TIMEOUT_S = 600
# A read timeout cannot stop a stream that keeps producing tokens; this can.
_OLLAMA_MAX_OUTPUT_TOKENS = 2048
_ANTHROPIC_MAX_TOKENS = 16000
# On a safety decline the API reruns the request on a fallback model it picks.
_ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_chat_model(settings: Settings, role: Role) -> BaseChatModel:
    name = settings.agent_model if role == "agent" else settings.judge_model
    if settings.provider == "ollama":
        # The agent samples as Qwen3's model card prescribes for non-thinking mode;
        # greedy decoding sent it into a repetition loop that ran for hours. The
        # judge decodes greedily for stable scores. A fixed seed keeps both
        # repeatable on one machine, and the output cap ends any loop.
        agent = role == "agent"
        return ChatOllama(
            model=name,
            base_url=settings.ollama_url,
            reasoning=False,  # the ReAct loop is the reasoning; thinking costs minutes
            seed=0,
            num_predict=_OLLAMA_MAX_OUTPUT_TOKENS,
            # Reserved up front as KV cache, so it is sized from measured prompts.
            num_ctx=settings.ollama_num_ctx,
            client_kwargs={"timeout": _OLLAMA_TIMEOUT_S},
            temperature=0.7 if agent else 0.0,
            top_p=0.8 if agent else None,
            top_k=20 if agent else None,
        )
    # No temperature, top_p or top_k: current Claude models reject them with a 400.
    return ChatAnthropic(  # type: ignore[call-arg]
        model=name,
        max_tokens=_ANTHROPIC_MAX_TOKENS,
        betas=[_ANTHROPIC_FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )
