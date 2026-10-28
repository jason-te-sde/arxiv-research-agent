from pathlib import Path

import pytest

from arxiv_agent.config import Settings


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.data_dir == Path("data")
    assert s.chroma_dir == Path("data/chroma")
    assert s.pdf_dir == Path("data/pdfs")
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Chosen on the dev set (docs/evaluation.md, chunk-size ablation).
    assert (s.chunk_size, s.chunk_overlap) == (512, 64)
    assert s.provider == "ollama"
    assert s.agent_model == "qwen3:8b"
    assert s.judge_model == "llama3.1:8b"
    assert s.ollama_url == "http://localhost:11434"
    assert s.max_iterations == 3
    assert s.k_candidates == 20
    assert s.k_final == 5


def test_env_override() -> None:
    s = Settings.from_env({"ARXIV_AGENT_CHUNK_SIZE": "256", "ARXIV_AGENT_CHUNK_OVERLAP": "32"})
    assert (s.chunk_size, s.chunk_overlap) == (256, 32)


def test_data_dir_moves_derived_paths() -> None:
    s = Settings.from_env({"ARXIV_AGENT_DATA_DIR": "/var/lib/agent"})
    assert s.chroma_dir == Path("/var/lib/agent/chroma")
    assert s.pdf_dir == Path("/var/lib/agent/pdfs")


def test_anthropic_provider_has_its_own_model_defaults() -> None:
    s = Settings.from_env({"ARXIV_AGENT_PROVIDER": "anthropic"})
    assert s.agent_model.startswith("claude-")
    assert s.judge_model.startswith("claude-")


def test_rejects_overlap_ge_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Settings.from_env({"ARXIV_AGENT_CHUNK_SIZE": "64", "ARXIV_AGENT_CHUNK_OVERLAP": "64"})


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        Settings.from_env({"ARXIV_AGENT_PROVIDER": "openai"})


def test_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="ARXIV_AGENT_K_FINAL"):
        Settings.from_env({"ARXIV_AGENT_K_FINAL": "five"})


def test_rejects_final_k_above_candidates() -> None:
    with pytest.raises(ValueError, match="k_final"):
        Settings.from_env({"ARXIV_AGENT_K_FINAL": "30"})


def test_ollama_context_window_is_configurable() -> None:
    assert Settings.from_env({"ARXIV_AGENT_OLLAMA_NUM_CTX": "8192"}).ollama_num_ctx == 8192
