"""Evaluation run directories and the metadata that makes a run reproducible."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arxiv_agent import __version__
from arxiv_agent.config import Settings
from arxiv_agent.eval.dataset import file_hash


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
