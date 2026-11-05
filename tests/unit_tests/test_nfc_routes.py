"""Tests for the REST NFC router (``/api/nfc``).

The reader itself is covered in ``test_nfc.py`` and the RPC mirror in
``test_jsonrpc_relay.py``; these check the HTTP side of the shared rules in
``daemon.nfc_switch``: what answers without a reader, and what answers 503.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.daemon.app.dependencies import get_nfc_reader
from reachy_mini.daemon.app.routers import nfc
from reachy_mini.nfc import (
    DISABLED_ERROR,
    NfcDump,
    NfcStatus,
    NfcTag,
    NfcWriteResult,
)


class FakeReader:
    """Stands in for ``NfcReader``: no serial port, canned answers."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.writes: list[Any] = []
        self.erases: list[Any] = []

    def get_status(self) -> NfcStatus:
        return NfcStatus(
            connected=True,
            enabled=self.enabled,
            chip_detected=True,
            driver_available=True,
        )

    def get_tag(self) -> NfcTag:
        return NfcTag(present=True, uid="04A1B2C3", content="badge42")

    def is_enabled(self) -> bool:
        return self.enabled

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def write(self, request: Any) -> NfcWriteResult:
        self.writes.append(request)
        return NfcWriteResult(success=True)

    def erase(self, request: Any) -> NfcWriteResult:
        self.erases.append(request)
        return NfcWriteResult(success=False, error="NO_TAG")

    def dump(self) -> NfcDump:
        return NfcDump(present=True, uid="04A1B2C3", size=4, hex="deadbeef")


def make_client(reader: FakeReader | None) -> TestClient:
    """Bare app with just the NFC router, serving ``reader``."""
    app = FastAPI()
    app.include_router(nfc.router)
    app.dependency_overrides[get_nfc_reader] = lambda: reader
    return TestClient(app)


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record the switches saved to the daemon config, without touching disk."""
    calls: list[bool] = []

    def save(enabled: bool) -> bool:
        calls.append(enabled)
        return True

    monkeypatch.setattr("reachy_mini.daemon.startup_app_config.set_nfc_enabled", save)
    return calls


def test_status_and_tag_are_served_from_the_reader() -> None:
    client = make_client(FakeReader())
    assert client.get("/nfc/status").json()["connected"] is True
    assert client.get("/nfc/tag").json()["uid"] == "04A1B2C3"


def test_status_and_tag_answer_without_a_reader() -> None:
    # --no-nfc: the polled routes still answer, with an empty state.
    client = make_client(None)
    status = client.get("/nfc/status")
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert status.json()["error"] == DISABLED_ERROR
    tag = client.get("/nfc/tag")
    assert tag.status_code == 200
    assert tag.json()["present"] is False


@pytest.mark.parametrize("reader", [None, FakeReader(enabled=False)])
def test_tag_operations_answer_503_while_disabled(reader: FakeReader | None) -> None:
    client = make_client(reader)
    for response in (
        client.post("/nfc/write", json={"text": "hi"}),
        client.post("/nfc/erase", json={}),
        client.get("/nfc/dump"),
    ):
        assert response.status_code == 503
        assert response.json()["detail"] == DISABLED_ERROR


def test_write_erase_and_dump_reach_the_reader() -> None:
    reader = FakeReader()
    client = make_client(reader)
    write = client.post("/nfc/write", json={"uri": "https://huggingface.co"})
    assert write.json()["success"] is True
    assert reader.writes[0].uri == "https://huggingface.co"
    erase = client.post("/nfc/erase", json={"full": True})
    assert erase.json()["error"] == "NO_TAG"
    assert reader.erases[0].full is True
    assert client.get("/nfc/dump").json()["hex"] == "deadbeef"


def test_write_rejects_bad_params() -> None:
    client = make_client(FakeReader())
    response = client.post("/nfc/write", json={"text": "a", "uri": "b"})
    assert response.status_code == 422


def test_switch_off_then_on(saved: list[bool]) -> None:
    reader = FakeReader()
    client = make_client(reader)
    off = client.post("/nfc/enabled", json={"enabled": False})
    assert off.json()["enabled"] is False
    assert off.json()["setting_saved"] is True
    assert client.post("/nfc/write", json={"text": "hi"}).status_code == 503
    on = client.post("/nfc/enabled", json={"enabled": True})
    assert on.json()["enabled"] is True
    assert saved == [False, True]


def test_switch_answers_503_without_a_reader(saved: list[bool]) -> None:
    client = make_client(None)
    response = client.post("/nfc/enabled", json={"enabled": True})
    assert response.status_code == 503
    assert saved == []
