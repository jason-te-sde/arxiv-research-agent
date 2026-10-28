"""Evaluation run directories and the metadata that makes a run reproducible."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from arxiv_agent import __version__
from arxiv_agent.config import Settings
from arxiv_agent.eval.answer_eval import AnswerRecord, operational_stats
from arxiv_agent.eval.dataset import file_hash
from arxiv_agent.eval.ragas_eval import ScoreRecord, summarize


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _git(*args: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def run_meta(settings: Settings, dataset: Path, corpus: Path) -> dict[str, Any]:
    return {
        "version": __version__,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown"),
        "dataset": dataset.name,
        "dataset_sha256": file_hash(dataset),
        "corpus_sha256": file_hash(corpus),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "k_candidates": settings.k_candidates,
        "k_final": settings.k_final,
        "started_at": utc_now(),
    }


def ollama_digests(
    base_url: str, names: Sequence[str], http: httpx.Client | None = None
) -> dict[str, str]:
    """The exact model build behind each Ollama tag, so a run names what it ran."""
    client = http or httpx.Client(timeout=10.0)
    try:
        response = client.get(f"{base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        installed = {m["name"]: m["digest"] for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError, KeyError):
        return dict.fromkeys(names, "unavailable")
    return {name: installed.get(name, "not installed") for name in names}


def unavailable_models(digests: Mapping[str, str]) -> list[str]:
    return [name for name, digest in digests.items() if digest in ("unavailable", "not installed")]


def ollama_unload(base_url: str, model: str, http: httpx.Client | None = None) -> bool:
    """Free a model's memory now instead of after Ollama's keep-alive.

    The agent and the judge are each about 8 GB with their context windows;
    on a 16 GB machine they must not be resident together.
    """
    client = http or httpx.Client(timeout=30.0)
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": 0}
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def build_summary(
    meta: Mapping[str, Any], records: Sequence[AnswerRecord], scores: Sequence[ScoreRecord]
) -> dict[str, Any]:
    by_id = {s.id: s.scores for s in scores}
    return {
        "meta": dict(meta),
        "ragas": summarize(scores),
        "operations": operational_stats(records),
        "items": [
            {
                "id": r.id,
                "question": r.question,
                "fallback": r.fallback,
                "reason": r.reason,
                "iterations": r.iterations,
                "latency_s": r.latency_s,
                "citations": len(r.citations),
                "scores": by_id.get(r.id, {}),
            }
            for r in records
        ],
    }
