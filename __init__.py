"""Academy-powered reimplementation of the Agentic Lab framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentic_lab_academy")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
