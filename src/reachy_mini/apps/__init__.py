"""Metadata about apps."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class SourceKind(str, Enum):
    """Kinds of app source."""

    HF_SPACE = "hf_space"
    DASHBOARD_SELECTION = "dashboard_selection"
    LOCAL = "local"
    INSTALLED = "installed"


@dataclass
class AppInfo:
    """Metadata about an app."""

    name: str
    source_kind: SourceKind
    description: str = ""
    url: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
