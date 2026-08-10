import json
from dataclasses import dataclass
from datetime import datetime

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.sources.hf_space import (
    _build_app_info,
    _coerce_space_data,
    _coerce_space_list,
    _get_card_data,
    _get_string,
    _normalize_space_data,
    _to_plain_json,
)


@dataclass
class _FakeSibling:
    rfilename: str


def test_to_plain_json_normalizes_hf_objects() -> None:
    # Mirrors a SpaceInfo.__dict__: nested SDK objects + datetimes that the raw
    # HfApi path would otherwise leave non-serializable in AppInfo.extra.
    raw = {
        "id": "owner/app",
        "siblings": [_FakeSibling("app/__main__.py")],
        "created_at": datetime(2024, 1, 2, 3, 4, 5),
    }

    plain = _to_plain_json(raw)

    # Result must be pure JSON (no exception) and match the HTTP API shape.
    json.dumps(plain)  # must not raise
    assert plain["siblings"][0]["rfilename"] == "app/__main__.py"
    assert plain["created_at"] == "2024-01-02T03:04:05"


def test_coerce_space_data_stringifies_keys() -> None:
    # Non-string keys become strings; values pass through untouched.
    assert _coerce_space_data({1: "a", "b": 2}) == {"1": "a", "b": 2}


def test_coerce_space_data_rejects_non_dict() -> None:
    assert _coerce_space_data(["not", "a", "dict"]) is None
    assert _coerce_space_data(None) is None


def test_coerce_space_list_keeps_only_dicts() -> None:
    # Dict items survive (with stringified keys); non-dict items are dropped.
    raw = [{"id": "a"}, "skip", 42, {"id": "b"}]
    assert _coerce_space_list(raw) == [{"id": "a"}, {"id": "b"}]


def test_coerce_space_list_rejects_non_list() -> None:
    assert _coerce_space_list({"id": "a"}) == []


def test_get_string_returns_string_values() -> None:
    assert _get_string({"id": "owner/app"}, "id") == "owner/app"


def test_get_string_none_for_missing_or_non_string() -> None:
    assert _get_string({"id": 123}, "id") is None
    assert _get_string({}, "id") is None


def test_get_card_data_returns_dict() -> None:
    card = {"short_description": "hi"}
    assert _get_card_data({"cardData": card}) == card


def test_get_card_data_defaults_to_empty_dict() -> None:
    assert _get_card_data({"cardData": "not a dict"}) == {}
    assert _get_card_data({}) == {}


def test_normalize_space_data_renames_snake_case_keys() -> None:
    # HfApi snake_case fields are moved to the app-store camelCase names.
    normalized = _normalize_space_data(
        {
            "id": "owner/app",
            "created_at": "2024-01-02",
            "last_modified": "2024-03-04",
            "card_data": {"short_description": "desc"},
        }
    )

    assert normalized == {
        "id": "owner/app",
        "createdAt": "2024-01-02",
        "lastModified": "2024-03-04",
        "cardData": {"short_description": "desc"},
    }


def test_normalize_space_data_keeps_existing_camel_case() -> None:
    # Existing camelCase values win; snake_case duplicates are dropped.
    normalized = _normalize_space_data(
        {
            "createdAt": "keep",
            "created_at": "drop",
            "lastModified": "keep",
            "last_modified": "drop",
            "cardData": {"short_description": "keep"},
            "card_data": {"short_description": "drop"},
        }
    )

    assert normalized == {
        "createdAt": "keep",
        "lastModified": "keep",
        "cardData": {"short_description": "keep"},
    }


def test_build_app_info_none_for_missing_input() -> None:
    assert _build_app_info(None) is None


def test_build_app_info_none_without_id() -> None:
    assert _build_app_info({"cardData": {"short_description": "x"}}) is None


def test_build_app_info_builds_full_metadata() -> None:
    app = _build_app_info(
        {"id": "owner/my-app", "cardData": {"short_description": "A nice app"}}
    )

    assert app is not None
    assert app.name == "my-app"
    assert app.description == "A nice app"
    assert app.url == "https://huggingface.co/spaces/owner/my-app"
    assert app.source_kind == SourceKind.HF_SPACE
    assert app.extra["id"] == "owner/my-app"
    assert isinstance(app, AppInfo)


def test_build_app_info_defaults_empty_description() -> None:
    # No cardData short_description -> description falls back to "".
    app = _build_app_info({"id": "owner/app"})

    assert app is not None
    assert app.description == ""
