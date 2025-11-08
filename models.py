"""Shared dataclasses exchanged between Academy agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

__all__ = [
    "SerializableDataclass",
    "PlanResult",
    "ResearchArtifact",
    "CodeArtifact",
    "ExecutionResult",
    "CritiqueBundle",
]


@dataclass(slots=True)
class SerializableDataclass:
    """Mixin providing helpers to convert dataclasses to and from plain dicts."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass(slots=True)
class PlanResult(SerializableDataclass):
    plan: str
    reasoning: Optional[str] = None


@dataclass(slots=True)
class ResearchArtifact(SerializableDataclass):
    content: str
    iteration: int


@dataclass(slots=True)
class CodeArtifact(SerializableDataclass):
    code: str
    iteration: int


@dataclass(slots=True)
class ExecutionResult(SerializableDataclass):
    success: bool
    stdout: str
    stderr: str
    error_type: Optional[str] = None
    packages_installed: Optional[list[str]] = None
    reasoning: Optional[str] = None


@dataclass(slots=True)
class CritiqueBundle(SerializableDataclass):
    document_feedback: Optional[str]
    code_feedback: Optional[str]
    summary: Optional[str]
    executor_feedback: Optional[str] = None
