"""IO module.

``WSServer`` is exposed lazily (PEP 562): importing it eagerly pulls the whole
daemon backend (vision.face_tracking, onnxruntime, fastapi, ...) into every
client app, which costs ~1s of startup on the wireless robot. Only the daemon
process needs it.
"""

from typing import TYPE_CHECKING

from .ws_client import WSClient

if TYPE_CHECKING:
    from .ws_server import WSServer

__all__ = [
    "WSClient",
    "WSServer",
]


def __getattr__(name: str) -> type:
    """Lazily resolve ``WSServer`` to keep daemon-only deps out of client apps."""
    if name == "WSServer":
        from .ws_server import WSServer

        return WSServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
