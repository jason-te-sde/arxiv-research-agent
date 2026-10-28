"""The ``arxiv-agent`` command line."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from arxiv_agent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-agent",
        description="Ask research questions over arXiv papers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover
