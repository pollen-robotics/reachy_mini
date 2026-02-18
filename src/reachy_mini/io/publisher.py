"""Transport-agnostic publisher for backend state streaming.

This module defines a simple Publisher class that wraps a callback function,
allowing backends to call .put(data) without knowing the underlying transport
(WebSocket, etc.).
"""

from typing import Callable

from pydantic import BaseModel


class Publisher:
    """Wraps a callback so backends can keep calling .put(data)."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        """Initialize the publisher with a callback function.

        Args:
            callback: A function that accepts a string and delivers it to the transport.

        """
        self._callback = callback

    def put(self, data: BaseModel | str | bytes) -> None:
        """Publish data through the transport.

        Args:
            data: The data to publish — a Pydantic model, a JSON string, or bytes.

        """
        if isinstance(data, BaseModel):
            self._callback(data.model_dump_json())
        elif isinstance(data, bytes):
            self._callback(data.decode("utf-8"))
        else:
            self._callback(data)
