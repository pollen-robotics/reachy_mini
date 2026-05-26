"""Tests for the REST audio-config router.

Hits the real ReSpeaker USB board, so each test is gated behind
`@pytest.mark.audio` like the helpers in `test_audio_control_utils.py`.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.daemon.app.routers import audio_config
from reachy_mini.media.audio_control_utils import init_respeaker_usb


@pytest.fixture
def client() -> TestClient:
    """Bare FastAPI test client with just the audio-config router mounted."""
    app = FastAPI()
    app.include_router(audio_config.router)
    return TestClient(app)


@pytest.mark.audio
def test_read_audio_parameter_round_trip(client: TestClient) -> None:
    """The REST route should return a JSON-decoded view of a board parameter."""
    response = client.get("/audio/config/parameter/AUDIO_MGR_MIC_GAIN")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "AUDIO_MGR_MIC_GAIN"
    assert isinstance(body["values"], list)
    assert len(body["values"]) >= 1


@pytest.mark.audio
def test_apply_audio_config_identity_write(client: TestClient) -> None:
    """Identity write (read → apply → verify) must succeed on the real board."""
    respeaker = init_respeaker_usb()
    assert respeaker is not None, "Reachy Mini Audio board is required."
    try:
        original = respeaker.read_values("PP_MIN_NS")
    finally:
        respeaker.close()
    assert original is not None

    response = client.post(
        "/audio/config/apply",
        json={
            "config": [{"name": "PP_MIN_NS", "values": list(original)}],
            "verify": True,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"applied": True}


def test_read_unknown_parameter_returns_404(client: TestClient) -> None:
    """Unknown parameter name must not crash; 404 surfaces it cleanly.

    Runs without the @pytest.mark.audio gate because `init_respeaker_usb()`
    returning None on a board-less machine is also a valid outcome — the
    test passes in either case (404 from the name lookup or 503 from the
    missing board).
    """
    response = client.get("/audio/config/parameter/DEFINITELY_NOT_A_PARAM")
    assert response.status_code in (404, 503)
