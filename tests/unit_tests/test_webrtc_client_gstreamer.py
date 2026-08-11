"""Unit tests for :class:`GstWebRTCClient`.

Requires the gst-plugins-rs `webrtcsrc` element (from `libgstrswebrtc.so`), so
the client constructs for real; the module is skipped where it's absent. Real
negotiation / frame flow needs a live peer (covered by `test_webrtc_loopback`
and the on-robot `wireless` path); here we unit-test the hardware-free surface
on a real instance: the daemon REST helpers, the signal callbacks, and small
helpers.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import reachy_mini.media.webrtc_client_gstreamer as mod  # noqa: E402
from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient  # noqa: E402

Gst.init([])
# Missing-plugin handling (skip vs fail) is centralised in conftest's
# pytest_collection_modifyitems, keyed on whether `-m webrtc` was selected.
pytestmark = pytest.mark.webrtc


@pytest.fixture
def client():
    """A real GstWebRTCClient (webrtcsrc from the loaded .so).

    Construction configures webrtcsrc's signaller but does not connect — the
    websocket handshake only happens on ``open()`` (PLAYING) — so a bogus
    signaling host is fine for these no-peer tests.
    """
    c = GstWebRTCClient(peer_id="p", signaling_host="h", signaling_port=8443)
    yield c
    c._loop.quit()


# ---------------------------------------------------------------------------
# Daemon REST helpers (mock `requests`)
# ---------------------------------------------------------------------------


def test_play_sound_without_daemon_url_makes_no_request(client, monkeypatch) -> None:
    req = MagicMock()
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = ""
    client.play_sound("wake_up.wav")
    req.post.assert_not_called()


def test_play_sound_posts_to_daemon(client, monkeypatch) -> None:
    req = MagicMock()
    req.post.return_value.ok = True
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = "http://d:8080"
    client.play_sound("wake_up.wav")  # not a local file -> sent as-is
    req.post.assert_called_once()
    url, kwargs = req.post.call_args.args[0], req.post.call_args.kwargs
    assert url == "http://d:8080/api/media/play_sound"
    assert kwargs["json"] == {"file": "wake_up.wav"}


def test_upload_sound_returns_daemon_path(client, monkeypatch, tmp_path) -> None:
    f = tmp_path / "beep.wav"
    f.write_bytes(b"RIFF....")
    req = MagicMock()
    req.post.return_value.json.return_value = {"path": "/daemon/tmp/beep.wav"}
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = "http://d:8080"
    assert client.upload_sound(str(f)) == "/daemon/tmp/beep.wav"
    assert req.post.call_args.args[0] == "http://d:8080/api/media/sounds/upload"


def test_upload_sound_missing_file_raises(client) -> None:
    with pytest.raises(FileNotFoundError):
        client.upload_sound("/does/not/exist.wav")


def test_list_sounds_empty_without_daemon_url(client) -> None:
    client.daemon_url = ""
    assert client.list_sounds() == []


def test_list_sounds_returns_files(client, monkeypatch) -> None:
    req = MagicMock()
    req.get.return_value.json.return_value = {"files": ["a.wav", "b.wav"]}
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = "http://d:8080"
    assert client.list_sounds() == ["a.wav", "b.wav"]


def test_delete_sound_returns_false_without_daemon_url(client) -> None:
    client.daemon_url = ""
    assert client.delete_sound("a.wav") is False


def test_delete_sound_reports_ok(client, monkeypatch) -> None:
    req = MagicMock()
    req.delete.return_value.ok = True
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = "http://d:8080"
    assert client.delete_sound("a.wav") is True
    assert req.delete.call_args.args[0] == "http://d:8080/api/media/sounds/a.wav"


def test_stop_playing_resets_pts_and_stops_daemon_sound(client, monkeypatch) -> None:
    req = MagicMock()
    monkeypatch.setattr(mod, "_requests", req)
    client.daemon_url = "http://d:8080"
    client.stop_playing()
    assert client._appsrc_pts == -1
    assert req.post.call_args.args[0] == "http://d:8080/api/media/stop_sound"


def test_clear_player_without_send_chain_warns_and_flushes_daemon(
    client, monkeypatch
) -> None:
    req = MagicMock()
    monkeypatch.setattr(mod, "_requests", req)
    client._appsrc = None  # send chain not up -> warn, still flush daemon
    client.daemon_url = "http://d:8080"
    client.clear_player()
    assert req.post.call_args.args[0] == "http://d:8080/api/media/clear_incoming_audio"


# ---------------------------------------------------------------------------
# Signal callbacks
# ---------------------------------------------------------------------------


def test_on_deep_element_added_captures_webrtcbin(client) -> None:
    element = MagicMock()
    element.get_factory.return_value.get_name.return_value = "webrtcbin"
    client._on_deep_element_added(MagicMock(), MagicMock(), element)
    assert client._webrtcbin is element
    element.connect.assert_called_once()


def test_on_deep_element_added_ignores_other_elements(client) -> None:
    element = MagicMock()
    element.get_factory.return_value.get_name.return_value = "queue"
    client._on_deep_element_added(MagicMock(), MagicMock(), element)
    assert client._webrtcbin is None


def test_on_new_transceiver_sendrecv_when_no_caps(client) -> None:
    tr = MagicMock()
    tr.get_property.return_value = None  # no codec-preferences
    client._on_new_transceiver(MagicMock(), tr)
    tr.set_property.assert_called_once_with("direction", 4)


def test_on_new_transceiver_sendrecv_for_audio(client) -> None:
    caps = MagicMock()
    caps.get_size.return_value = 1
    caps.get_structure.return_value.get_string.return_value = "audio"
    tr = MagicMock()
    tr.get_property.return_value = caps
    client._on_new_transceiver(MagicMock(), tr)
    tr.set_property.assert_called_once_with("direction", 4)


def test_on_new_transceiver_skips_known_video(client) -> None:
    caps = MagicMock()
    caps.get_size.return_value = 1
    caps.get_structure.return_value.get_string.return_value = "video"
    tr = MagicMock()
    tr.get_property.return_value = caps
    client._on_new_transceiver(MagicMock(), tr)
    tr.set_property.assert_not_called()


def test_on_bus_message_latency_recalculates(client) -> None:
    pipeline = MagicMock()
    msg = Gst.Message.new_latency(None)
    assert client._on_bus_message(MagicMock(), msg, pipeline) is True
    pipeline.recalculate_latency.assert_called_once()


def test_on_bus_message_ignores_appsrc_not_negotiated(client) -> None:
    class _Err:
        def __str__(self) -> str:
            return "not-negotiated"

    msg = MagicMock()
    msg.type = Gst.MessageType.ERROR
    msg.parse_error.return_value = (_Err(), "debug")
    msg.src.get_factory.return_value.get_name.return_value = "appsrc"
    assert client._on_bus_message(MagicMock(), msg, MagicMock()) is True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_iterate_gst_yields_ok_and_resyncs(client) -> None:
    class _Iter:
        def __init__(self, seq):
            self._seq = list(seq)
            self.resyncs = 0

        def next(self):
            return self._seq.pop(0)

        def resync(self):
            self.resyncs += 1

    it = _Iter(
        [
            (Gst.IteratorResult.OK, "e1"),
            (Gst.IteratorResult.RESYNC, None),
            (Gst.IteratorResult.OK, "e2"),
            (Gst.IteratorResult.DONE, None),
        ]
    )
    assert list(client._iterate_gst(it)) == ["e1", "e2"]
    assert it.resyncs == 1


def test_apply_resolution_raises_while_playing(client) -> None:
    client._pipeline_record.get_state = MagicMock(
        return_value=MagicMock(state=Gst.State.PLAYING)
    )
    with pytest.raises(RuntimeError):
        client._apply_resolution(client.camera_specs.default_resolution)


# ---------------------------------------------------------------------------
# _configure_webrtcsrc (real method, exercised directly)
# ---------------------------------------------------------------------------


def test_configure_webrtcsrc_raises_without_plugin(monkeypatch) -> None:
    monkeypatch.setattr(mod.Gst.ElementFactory, "make", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="webrtc rust plugin"):
        GstWebRTCClient._configure_webrtcsrc(types.SimpleNamespace(), "h", 8443, "p")


def test_configure_webrtcsrc_sets_signaller_properties(client, monkeypatch) -> None:
    source = MagicMock()
    monkeypatch.setattr(mod.Gst.ElementFactory, "make", lambda *a, **k: source)
    result = client._configure_webrtcsrc("host", 9000, "peer42")
    assert result is source
    source.connect.assert_called_once()  # pad-added
    signaller = source.get_property.return_value
    signaller.set_property.assert_any_call("producer-peer-id", "peer42")
    signaller.set_property.assert_any_call("uri", "ws://host:9000")
