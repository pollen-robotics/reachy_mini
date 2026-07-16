"""JSON-RPC 2.0 envelope for the app control surface.

This is the *one* wire format for driving and observing daemon apps over any
transport (WebRTC DataChannel, ``/ws/sdk`` WebSocket, or an app's local
``/rpc`` WebSocket). It is shared, on purpose, between the daemon relay and the
apps (which import it from the installed ``reachy_mini`` package), so there is a
single source of truth for the contract.

Every frame is a pydantic model, and inbound frames are parsed through a
:class:`~pydantic.TypeAdapter` union (:data:`rpc_adapter`) exactly like the
legacy protocol's ``command_adapter`` / ``server_msg_adapter`` — so there is no
hand-rolled ``json.loads`` + ``dict.get`` shape-sniffing anywhere.

Message families, per JSON-RPC 2.0:

* **Request** (:class:`RpcRequest`) —
  ``{"jsonrpc":"2.0","id":<token>,"method":str,"params":{...}}``. Carries an
  ``id``; expects exactly one response. With ``id`` absent it is a
  **notification** (one-way, never answered); this is how events
  (``conversation.phase``/``turn``/``transcript`` ...) are pushed to every
  connected client.
* **Response** — success (:class:`RpcSuccess`)
  ``{"jsonrpc":"2.0","id":<token>,"result":{...}}`` or failure
  (:class:`RpcErrorResponse`) ``{"jsonrpc":"2.0","id":<token>,"error":{...}}``.

**One deliberate deviation from the base spec** (matches the conversation API
design doc): the stable, machine-branchable string error code lives in
``error.data.reason``. ``error.code`` stays a JSON-RPC integer.

Method names are namespaced by a ``<namespace>.<verb>`` convention. The
namespace decides routing in the daemon relay: ``apps.*`` is handled by the
daemon itself; anything else is relayed to the running app's ``/rpc``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

JSONRPC_VERSION = "2.0"

# An id is a client-chosen token echoed back verbatim. Spec allows string,
# number or null; we accept str | int and keep it opaque.
RpcId = Union[str, int]


# ------------------------------------------------------------------
# Standard JSON-RPC error codes (integers). The UI branches on the string
# `reason` in error.data, not on these — they exist for spec compliance.
# ------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# -32000..-32099 is the implementation-defined server-error range.
SERVER_ERROR = -32000


# ------------------------------------------------------------------
# Wire models. Every frame the relay/app/server emits is one of these, so the
# shape is defined once and validated by pydantic rather than assembled by hand.
# ------------------------------------------------------------------


class RpcErrorObj(BaseModel):
    """The JSON-RPC ``error`` object. ``data`` carries the stable ``reason``."""

    model_config = ConfigDict(extra="ignore")

    code: int
    message: str
    data: dict[str, Any] = {}


class RpcRequest(BaseModel):
    """A JSON-RPC request or notification (``id`` absent => notification)."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: str = JSONRPC_VERSION
    id: Optional[RpcId] = None
    method: str
    params: dict[str, Any] = {}

    @property
    def is_notification(self) -> bool:
        """True when this frame carries no ``id`` (one-way, no response)."""
        return self.id is None

    @property
    def namespace(self) -> str:
        """The part before the first dot (``conversation.say`` -> ``conversation``)."""
        return self.method.split(".", 1)[0]


class RpcSuccess(BaseModel):
    """A JSON-RPC success response (``id`` echoed, ``result`` payload)."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: str = JSONRPC_VERSION
    id: Optional[RpcId]
    result: Any


class RpcErrorResponse(BaseModel):
    """A JSON-RPC error response (``id`` may be null per the spec)."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: str = JSONRPC_VERSION
    id: Optional[RpcId]
    error: RpcErrorObj


class RpcNotification(BaseModel):
    """A one-way JSON-RPC notification / event (no ``id``, never answered)."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: str = JSONRPC_VERSION
    method: str
    params: dict[str, Any] = {}


# Any frame that can arrive on the relay<->app / SDK boundary. Ordered
# most-specific-first: success/error responses carry ``result``/``error``
# (which requests never do), so the union disambiguates on required fields.
AnyRpcInbound = Union[RpcSuccess, RpcErrorResponse, RpcRequest]
rpc_adapter: TypeAdapter[AnyRpcInbound] = TypeAdapter(AnyRpcInbound)


class JsonRpcError(Exception):
    """A JSON-RPC error a handler can raise to fail one request.

    ``reason`` is the stable string the client branches on (goes into
    ``error.data.reason``); ``code`` is the JSON-RPC integer; ``data`` merges
    into ``error.data`` alongside ``reason``.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        code: int = SERVER_ERROR,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Build an error carrying a JSON-RPC ``code`` and stable ``reason``."""
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.code = code
        self.data = data or {}

    def to_error_model(self) -> RpcErrorObj:
        """Render the JSON-RPC ``error`` object as a model."""
        return RpcErrorObj(
            code=self.code,
            message=self.message,
            data={**self.data, "reason": self.reason},
        )

    def to_error_obj(self) -> dict[str, Any]:
        """Render the JSON-RPC ``error`` object as a plain dict."""
        return self.to_error_model().model_dump()


def looks_like_jsonrpc(obj: Any) -> bool:
    """Return whether a decoded JSON value is a JSON-RPC frame.

    Used at the transport boundary to tell the new JSON-RPC surface apart from
    the legacy ``{"type": ...}`` command protocol on the same channel.
    """
    return isinstance(obj, dict) and obj.get("jsonrpc") == JSONRPC_VERSION


def _error_from_validation(exc: ValidationError, what: str) -> JsonRpcError:
    """Map a pydantic error to the right JSON-RPC error.

    Uses pydantic-core's ``json_invalid`` marker to keep the spec distinction
    between a JSON syntax error (``parse_error`` / -32700) and a
    well-formed-but-wrong-shape frame (``invalid_request`` / -32600), without a
    separate ``json.loads`` pass — validation and decoding are one Rust step.
    """
    if any(err.get("type") == "json_invalid" for err in exc.errors()):
        return JsonRpcError(
            f"parse error: {exc}", reason="parse_error", code=PARSE_ERROR
        )
    return JsonRpcError(
        f"invalid {what}: {exc}", reason="invalid_request", code=INVALID_REQUEST
    )


def parse_request(raw: Union[str, bytes, dict[str, Any]]) -> RpcRequest:
    """Parse a raw frame into an :class:`RpcRequest`.

    Raises :class:`JsonRpcError` (``INVALID_REQUEST``/``PARSE_ERROR``) so the
    caller can turn it straight into an error response.
    """
    try:
        if isinstance(raw, dict):
            return RpcRequest.model_validate(raw)
        return RpcRequest.model_validate_json(raw)
    except ValidationError as e:
        raise _error_from_validation(e, "request") from e


def parse_inbound(raw: Union[str, bytes, dict[str, Any]]) -> AnyRpcInbound:
    """Parse any inbound frame into the right model (request/success/error).

    Used on the relay<->app boundary to classify a frame (a correlated
    response vs. an app-pushed notification) without shape-sniffing raw dicts.
    Raises :class:`JsonRpcError` on malformed input.
    """
    try:
        if isinstance(raw, dict):
            return rpc_adapter.validate_python(raw)
        return rpc_adapter.validate_json(raw)
    except ValidationError as e:
        raise _error_from_validation(e, "frame") from e


def make_result(id: Optional[RpcId], result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success response object."""
    return RpcSuccess(id=id, result=result).model_dump()


def make_error(
    id: Optional[RpcId],
    *,
    message: str,
    reason: str,
    code: int = SERVER_ERROR,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a JSON-RPC error response object (string ``reason`` in ``data``)."""
    err = JsonRpcError(message, reason=reason, code=code, data=data)
    return RpcErrorResponse(id=id, error=err.to_error_model()).model_dump()


def error_from_exc(id: Optional[RpcId], exc: BaseException) -> dict[str, Any]:
    """Build an error response from any exception.

    A :class:`JsonRpcError` keeps its code/reason/data; anything else becomes a
    generic ``internal_error``.
    """
    if isinstance(exc, JsonRpcError):
        return RpcErrorResponse(id=id, error=exc.to_error_model()).model_dump()
    return make_error(
        id,
        message=str(exc) or exc.__class__.__name__,
        reason="internal_error",
        code=INTERNAL_ERROR,
    )


def make_notification(
    method: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Build a one-way JSON-RPC notification (event) object."""
    return RpcNotification(method=method, params=params or {}).model_dump()
