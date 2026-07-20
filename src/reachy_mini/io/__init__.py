"""IO module."""

from typing import TYPE_CHECKING

from .ws_client import WSClient

if TYPE_CHECKING:
    from .ws_server import WSServer

__all__ = [
    "WSClient",
    "WSServer",
]


def __getattr__(name: str) -> object:
    """Load daemon-only IO classes on first access."""
    if name == "WSServer":
        from .ws_server import WSServer

        globals()[name] = WSServer
        return WSServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List public lazy exports without importing them."""
    return sorted(set(globals()) | set(__all__))
