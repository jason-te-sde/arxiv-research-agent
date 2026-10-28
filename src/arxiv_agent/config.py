"""Runtime settings, read once from ``ARXIV_AGENT_*`` environment variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Provider = Literal["ollama", "anthropic"]

_PREFIX = "ARXIV_AGENT_"

# The models the published numbers were measured with (docs/evaluation.md).
_MODEL_DEFAULTS: dict[Provider, tuple[str, str]] = {
    "ollama": ("qwen3:8b", "llama3.1:8b"),
    "anthropic": ("claude-opus-5-5", "claude-opus-5-5"),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embed_model: str
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    provider: Provider
    agent_model: str
    judge_model: str
    ollama_url: str
    ollama_num_ctx: int = 16384
    max_iterations: int = 3
    k_candidates: int = 20
    k_final: int = 5

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk size ({self.chunk_size})"
            )
        if self.k_final > self.k_candidates:
            raise ValueError(
                f"k_final ({self.k_final}) cannot exceed k_candidates ({self.k_candidates})"
            )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        def get(name: str, default: str) -> str:
            return env.get(_PREFIX + name, default)

        def get_int(name: str, default: int) -> int:
            raw = env.get(_PREFIX + name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"{_PREFIX}{name} must be an integer, got {raw!r}") from None

        provider = get("PROVIDER", "ollama")
        if provider not in _MODEL_DEFAULTS:
            raise ValueError(
                f"unknown provider {provider!r}; expected one of {list(_MODEL_DEFAULTS)}"
            )
        agent_default, judge_default = _MODEL_DEFAULTS[provider]
        return cls(
            data_dir=Path(get("DATA_DIR", "data")),
            embed_model=get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            rerank_model=get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            chunk_size=get_int("CHUNK_SIZE", 512),
            chunk_overlap=get_int("CHUNK_OVERLAP", 64),
            provider=provider,
            agent_model=get("AGENT_MODEL", agent_default),
            judge_model=get("JUDGE_MODEL", judge_default),
            ollama_url=get("OLLAMA_URL", "http://localhost:11434"),
            ollama_num_ctx=get_int("OLLAMA_NUM_CTX", 16384),
            max_iterations=get_int("MAX_ITERATIONS", 3),
            k_candidates=get_int("K_CANDIDATES", 20),
            k_final=get_int("K_FINAL", 5),
        )
