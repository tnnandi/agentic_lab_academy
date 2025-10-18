"""CLI entry point for the Academy-powered Agentic Lab."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    from pathlib import Path as _Path
    import sys as _sys

    _base = str(_Path(__file__).resolve().parent)
    if _base not in _sys.path:
        _sys.path.append(_base)
    from workflows.orchestrator import run_workflow
else:
    from .workflows.orchestrator import run_workflow



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic Lab workflow using Academy agents.")
    parser.add_argument("--topic", required=True, help="Specify the research topic.")
    parser.add_argument("--pdfs", nargs="+", help="One or more PDF files to include in the research.")
    parser.add_argument("--links", nargs="+", help="One or more URLs to include in the research.")
    parser.add_argument(
        "--files_dir",
        help="Path to a directory containing files to analyse and summarise for the agents.",
    )
    parser.add_argument(
        "--quick_search",
        action="store_true",
        help="Carry out a quick DuckDuckGo search instead of the full multi-agent workflow.",
    )
    parser.add_argument(
        "--mode",
        choices=["research_only", "code_only", "both"],
        default="both",
        help="Choose whether to generate research, code, or both (default).",
    )
    parser.add_argument(
        "--conda_env",
        help="Optional path to a conda environment for code execution (e.g., /path/to/env).",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Silence informational logging from the agents.",
    )
    parser.set_defaults(verbose=True)
    return parser


def _validate_paths(paths: Sequence[str] | None, kind: str) -> list[str]:
    if not paths:
        return []
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{kind} not found: {', '.join(missing)}")
    return list(paths)


async def _run_from_args(args: argparse.Namespace) -> None:
    pdfs = _validate_paths(args.pdfs, "PDF file") if args.pdfs else []
    if args.files_dir and not Path(args.files_dir).exists():
        raise FileNotFoundError(f"Files directory not found: {args.files_dir}")

    await run_workflow(
        topic=args.topic,
        mode=args.mode,
        quick_search=args.quick_search,
        pdfs=pdfs,
        links=args.links or [],
        files_dir=args.files_dir,
        conda_env=args.conda_env,
        verbose=True,
        # verbose=args.verbose,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_run_from_args(args))


if __name__ == "__main__":  # pragma: no cover
    main()
