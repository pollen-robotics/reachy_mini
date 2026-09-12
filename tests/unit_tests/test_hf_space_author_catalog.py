"""Tests for author-scoped HF Spaces catalog merge (issue #1374)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini.apps.sources import hf_space


def _space(space_id: str, likes: int = 0) -> SimpleNamespace:
    return SimpleNamespace(id=space_id, likes=likes, cardData={})


def test_merge_space_payloads_keeps_first_id() -> None:
    """First occurrence of a Space id wins during merge."""
    first = {"id": "user/app", "likes": 1}
    second = {"id": "user/app", "likes": 99}
    other = {"id": "org/other", "likes": 2}
    merged = hf_space._merge_space_payloads([first], [second, other])
    assert merged == [first, other]


def test_list_all_spaces_without_token_skips_author_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No token means only the likes-sorted catalog query runs."""
    api = MagicMock()
    api.list_spaces.return_value = [_space("org/popular", likes=10)]
    monkeypatch.setattr(hf_space, "HfApi", lambda: api)

    payloads = hf_space._list_all_spaces_with_hf_api(None)

    assert [p["id"] for p in payloads] == ["org/popular"]
    api.list_spaces.assert_called_once()
    assert "author" not in api.list_spaces.call_args.kwargs
    api.whoami.assert_not_called()


def test_list_all_spaces_merges_author_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Author Spaces are merged in and private-only apps are kept."""
    api = MagicMock()

    def list_spaces(**kwargs: Any) -> list[SimpleNamespace]:
        if "author" in kwargs:
            return [_space("me/private"), _space("me/also-public")]
        return [_space("me/also-public", likes=5), _space("org/popular", likes=9)]

    api.list_spaces.side_effect = list_spaces
    api.whoami.return_value = {"name": "me"}
    monkeypatch.setattr(hf_space, "HfApi", lambda: api)

    payloads = hf_space._list_all_spaces_with_hf_api("hf_token")

    assert [p["id"] for p in payloads] == [
        "me/private",
        "me/also-public",
        "org/popular",
    ]
    assert api.list_spaces.call_count == 2
    api.whoami.assert_called_once_with(token="hf_token")


def test_list_all_spaces_survives_whoami_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whoami failure must not drop the public catalog."""
    api = MagicMock()
    api.list_spaces.return_value = [_space("org/popular", likes=3)]
    api.whoami.side_effect = RuntimeError("offline")
    monkeypatch.setattr(hf_space, "HfApi", lambda: api)

    payloads = hf_space._list_all_spaces_with_hf_api("hf_token")

    assert [p["id"] for p in payloads] == ["org/popular"]
