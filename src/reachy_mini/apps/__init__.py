"""Metadata about apps."""

from dataclasses import dataclass, field
from enum import Enum


class SourceKind(str, Enum):
    """Kinds of app source."""

    HF_SPACE = "hf_space"
    INSTALLED = "installed"


@dataclass
class AppInfo:
    """Metadata about an app."""

    name: str
    source_kind: SourceKind
    description: str = ""
    url: str | None = None
    extra: dict = field(default_factory=dict)
