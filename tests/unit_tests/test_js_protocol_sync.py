"""Verify that the JS browser SDK sends commands matching protocol.py.

Parses _sendCommand({...}) calls from js/reachy-mini.js and checks each one
against the Pydantic command models defined in protocol.py.  This catches
field renames, missing required fields, and unknown type values at CI time.
"""

import re
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from reachy_mini.io.protocol import AnyCommand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_SDK_PATH = REPO_ROOT / "js" / "reachy-mini.js"


def _get_protocol_commands() -> dict[str, dict[str, bool]]:
    """Return {type_value: {field_name: required, ...}} for every command model.

    The ``type`` field itself is excluded — it's the discriminator, not a
    payload field.
    """
    result: dict[str, dict[str, bool]] = {}
    # AnyCommand is Annotated[Union[...], Field(discriminator=...)]; the first
    # arg of the Annotated wrapper is the bare Union.
    union = get_args(AnyCommand)[0]
    for model in get_args(union):
        assert issubclass(model, BaseModel)
        type_value = model.model_fields["type"].default
        fields: dict[str, bool] = {}
        for name, info in model.model_fields.items():
            if name == "type":
                continue
            required = info.is_required()
            fields[name] = required
        result[type_value] = fields
    return result


def _strip_nested(text: str) -> str:
    """Remove content inside nested parentheses, brackets, and strings.

    This leaves only the top-level object keys and commas, so that
    ``rpyToMatrix(roll, pitch, yaw).flat()`` becomes ``rpyToMatrix.flat``
    and ``[degToRad(x), degToRad(y)]`` becomes ``[]``.
    """
    result: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            # Skip string literal
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        elif depth == 0:
            result.append(ch)
        i += 1
    return "".join(result)


def _parse_js_commands(js_source: str) -> list[dict[str, set[str]]]:
    """Extract ``{type, fields}`` from each ``_sendCommand({...})`` call.

    Returns a list of dicts with keys ``type`` (str) and ``fields`` (set of
    field names excluding ``type``).
    """
    results: list[dict[str, set[str]]] = []
    # Match _sendCommand({ ... }) — the content between the braces.
    for m in re.finditer(r"_sendCommand\(\{([^}]+)\}\)", js_source):
        body = m.group(1)
        # Extract the "type" value before stripping (it's in a string).
        type_match = re.search(r'type:\s*"([^"]+)"', body)
        if not type_match:
            continue
        type_value = type_match.group(1)
        # Strip nested expressions so only top-level keys remain.
        stripped = _strip_nested(body)
        # Split by commas and identify keys.
        keys: set[str] = set()
        for part in stripped.split(","):
            part = part.strip()
            # "key: value" form
            colon_match = re.match(r"(\w+)\s*:", part)
            if colon_match:
                keys.add(colon_match.group(1))
            # ES6 shorthand: bare identifier (e.g. "file" in { type: ..., file })
            elif re.fullmatch(r"\w+", part):
                keys.add(part)
        keys.discard("type")
        results.append({"type": type_value, "fields": keys})
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_js_command_types_exist_in_protocol() -> None:
    """Every type value sent by the JS SDK must be a valid protocol command."""
    protocol = _get_protocol_commands()
    js_source = JS_SDK_PATH.read_text()
    js_cmds = _parse_js_commands(js_source)

    assert js_cmds, "Failed to parse any _sendCommand calls from JS SDK"

    for cmd in js_cmds:
        assert cmd["type"] in protocol, (
            f'JS SDK sends type="{cmd["type"]}" which does not exist in protocol.py. '
            f"Valid types: {sorted(protocol)}"
        )


def test_js_command_fields_match_protocol() -> None:
    """Payload field names sent by the JS SDK must match the protocol model."""
    protocol = _get_protocol_commands()
    js_source = JS_SDK_PATH.read_text()
    js_cmds = _parse_js_commands(js_source)

    for cmd in js_cmds:
        type_value = cmd["type"]
        if type_value not in protocol:
            continue  # caught by test_js_command_types_exist_in_protocol
        expected_fields = set(protocol[type_value])
        extra = cmd["fields"] - expected_fields
        assert not extra, (
            f'JS SDK sends unknown fields {extra} for type="{type_value}". '
            f"Protocol expects: {expected_fields or '(no payload fields)'}"
        )


def test_js_command_required_fields_present() -> None:
    """All required protocol fields must be present in the JS command."""
    protocol = _get_protocol_commands()
    js_source = JS_SDK_PATH.read_text()
    js_cmds = _parse_js_commands(js_source)

    for cmd in js_cmds:
        type_value = cmd["type"]
        if type_value not in protocol:
            continue
        required = {k for k, v in protocol[type_value].items() if v}
        missing = required - cmd["fields"]
        assert not missing, (
            f'JS SDK is missing required fields {missing} for type="{type_value}". '
            f"Protocol requires: {required}"
        )
