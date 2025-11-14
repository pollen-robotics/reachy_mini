"""Metadata about apps."""

from dataclasses import field
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class SourceKind(str, Enum):
    """Kinds of app source."""

    HF_SPACE = "hf_space"
    DASHBOARD_SELECTION = "dashboard_selection"
    LOCAL = "local"
    INSTALLED = "installed"


class AppInfo(BaseModel):
    """Metadata about an app."""

    name: str
    source_kind: SourceKind
    description: str = ""
    url: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
