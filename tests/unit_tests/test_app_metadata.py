import json
from dataclasses import dataclass
from datetime import datetime

from reachy_mini.apps.sources.hf_space import _to_plain_json


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
