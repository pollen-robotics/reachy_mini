"""Real webrtcbin configuration, with a local bin replacing remote signaling.

No pipeline enters PLAYING and no capture/network source is used. These tests
verify configuration and property readback, not live jitter or media quality.
"""

from typing import Any

import pytest

from reachy_mini.media.webrtc_client_gstreamer import (
    ENV_WEBRTCBIN_LATENCY_MS,
    Gst,
    GstWebRTCClient,
)


pytestmark = pytest.mark.webrtc


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch):
    # AudioBase otherwise probes for a physical ReSpeaker during construction.
    from reachy_mini.media import audio_doa

    monkeypatch.setattr(audio_doa, "init_respeaker_usb", lambda: None)
    Gst.init(None)
    if Gst.ElementFactory.find("webrtcbin") is None:
        pytest.fail("real webrtcbin plugin is required for this runtime test")
    clients = []

    def make(latency: Any = None, name: str = "webrtcbin0"):
        source = Gst.Bin.new("offline_source")
        element = Gst.ElementFactory.make("webrtcbin", name)
        assert element is not None
        source.add(element)
        # Only signaling and physical USB discovery are replaced. The constructor, GLib loop,
        # Gst pipeline/appsinks, latency resolver and setter all run normally.
        monkeypatch.setattr(GstWebRTCClient, "_configure_webrtcsrc", lambda *a: source)
        client = GstWebRTCClient(webrtcbin_latency_ms=latency)
        clients.append(client)
        return client, element

    yield make
    for client in clients:
        client.close()
        client._loop.quit()
        client._thread_bus_calls.join(timeout=2)
        assert not client._thread_bus_calls.is_alive()
        client._bus_record.remove_watch()


@pytest.mark.parametrize(
    "env,explicit,expected",
    [
        (None, None, 10),
        ("", None, 10),
        ("  ", None, 10),
        ("100", None, 100),
        (" 200 ", None, 200),
        ("200", 0, 0),
        ("200", 50, 50),
        ("invalid", 10, 10),
    ],
)
@pytest.mark.parametrize("name", ["webrtcbin0", "alternate_webrtcbin"])
def test_application_latency_precedence_real_element(
    monkeypatch, client_factory, env, explicit, expected, name
):
    if env is None:
        monkeypatch.delenv(ENV_WEBRTCBIN_LATENCY_MS, raising=False)
    else:
        monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, env)
    client, element = client_factory(explicit, name)
    assert client.webrtcbin_latency_ms == expected
    assert element.get_property("latency") == 200
    client._configure_webrtcbin(client._webrtcsrc)
    assert element.get_property("latency") == expected
    assert client._pipeline_record.get_state(0).state == Gst.State.NULL
    print(f"env={env!r} explicit={explicit!r} name={name} readback={expected}")


@pytest.mark.parametrize("value", [0, 10, 100, 200, 2**32 - 1])
def test_real_property_limits_and_supported_values(client_factory, value):
    client, element = client_factory(value)
    spec = element.find_property("latency")
    assert (spec.value_type.name, spec.minimum, spec.maximum, spec.default_value) == (
        "guint",
        0,
        2**32 - 1,
        200,
    )
    client._configure_webrtcbin(client._webrtcsrc)
    assert element.get_property("latency") == value
    print(f"supported latency={value} readback={element.get_property('latency')}")


@pytest.mark.parametrize("value", [-1, 2**32, 1.5, True, "bad"])
def test_invalid_explicit_rejected_at_constructor(client_factory, value):
    with pytest.raises(ValueError):
        client_factory(value)


@pytest.mark.parametrize("value", ["-1", str(2**32), "1.5", "bad"])
def test_invalid_env_rejected_at_constructor(monkeypatch, client_factory, value):
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, value)
    with pytest.raises(ValueError):
        client_factory()


def test_invalid_constructor_can_be_finalized(monkeypatch):
    from reachy_mini.media import audio_doa

    monkeypatch.setattr(audio_doa, "init_respeaker_usb", lambda: None)
    client = GstWebRTCClient.__new__(GstWebRTCClient)
    with pytest.raises(ValueError):
        client.__init__(webrtcbin_latency_ms=-1)
    # Python also invokes __del__ when __init__ raises, before _loop exists.
    client.__del__()
