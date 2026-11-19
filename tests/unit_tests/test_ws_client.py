"""Tests for WSClient's loopback-aware WebSocket compression setting.

permessage-deflate buys nothing when the SDK client and the daemon run on
the same host (no network bandwidth to save) but still costs CPU at the
daemon's 50 Hz state-stream rate. WSClient disables compression when
connecting to a loopback host and leaves the library default (deflate)
untouched for any other host.
"""

import threading
import time
from collections.abc import Callable, Iterator
from unittest.mock import ANY, MagicMock, patch

import pytest
import websockets.exceptions
from websockets.sync.server import ServerConnection, serve

import reachy_mini.io.ws_client as ws_client

from reachy_mini.io.protocol import (
    DaemonState,
    DaemonStatus,
    GotoTaskRequest,
    HeadPoseMsg,
    JointPositionsMsg,
    SetTargetCmd,
)
from reachy_mini.io.ws_client import WSClient, _is_loopback_host
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import InterpolationTechnique


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.0.0.2",
        "::1",
        "[::1]",
        "LOCALHOST",
        "Localhost",
        " 127.0.0.1 ",
    ],
)
def test_is_loopback_host_true_for_loopback_addresses(host: str) -> None:
    """Loopback hostnames/IPs are recognized regardless of case/whitespace."""
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["192.168.1.42", "reachy.local", "example.com", "0.0.0.0", "[not-an-ip]", ""],
)
def test_is_loopback_host_false_for_non_loopback_addresses(host: str) -> None:
    """Non-loopback hosts (including the all-interfaces address) are not matched."""
    assert _is_loopback_host(host) is False


def _make_fake_ws() -> MagicMock:
    """Build a ws_sync.ClientConnection stand-in that stays open until closed.

    Like a real open line, iterating on it waits: an iterator that ended
    at once would look like a dropped line and start a reconnection.
    """
    closed = threading.Event()

    def wait_until_closed() -> Iterator[str]:
        closed.wait()
        yield from ()

    fake_ws = MagicMock()
    fake_ws.__iter__.side_effect = wait_until_closed
    fake_ws.close.side_effect = closed.set
    return fake_ws


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_disables_compression_for_localhost(mock_connect: MagicMock) -> None:
    """Connecting to localhost passes compression=None to the sync client."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="localhost", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with(
        "ws://127.0.0.1:1234/ws/sdk", compression=None, additional_headers=ANY
    )


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", " Localhost "])
@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_localhost_is_pinned_to_ipv4(mock_connect: MagicMock, host: str) -> None:
    """localhost -> 127.0.0.1, so Windows doesn't wait ~2 s on a refused ::1 first.

    ``host`` is what the HTTP calls (media requests, daemon URL) are built from.
    """
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host=host, port=1234)
    client.disconnect()

    assert client.host == "127.0.0.1"
    assert mock_connect.call_args.args[0] == "ws://127.0.0.1:1234/ws/sdk"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "192.168.1.50", "reachy.local"])
@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_other_hosts_are_left_untouched(mock_connect: MagicMock, host: str) -> None:
    """Only the localhost name is rewritten: explicit addresses and names are kept."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host=host, port=1234)
    client.disconnect()

    assert client.host == host


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_disables_compression_for_127_0_0_1(mock_connect: MagicMock) -> None:
    """Connecting to 127.0.0.1 also passes compression=None."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="127.0.0.1", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with(
        "ws://127.0.0.1:1234/ws/sdk", compression=None, additional_headers=ANY
    )


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_keeps_default_compression_for_remote_host(
    mock_connect: MagicMock,
) -> None:
    """Connecting to a non-loopback host leaves the library default (deflate)."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="192.168.1.50", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with(
        "ws://192.168.1.50:1234/ws/sdk", compression="deflate", additional_headers=ANY
    )


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_name_check_keeps_status_for_media_setup(mock_connect: MagicMock) -> None:
    """The robot-name check must leave the status for the media setup that follows.

    Otherwise ReachyMini(robot_name=...) waits for the daemon's next 1 Hz
    status publish before it returns.
    """
    mock_connect.return_value = _make_fake_ws()
    client = WSClient(host="localhost", port=1234)
    client._dispatch(
        DaemonStatus(
            robot_name="lite",
            state=DaemonState.RUNNING,
            wireless_version=False,
            desktop_app_daemon=False,
            simulation_enabled=False,
            mockup_sim_enabled=False,
            backend_status=None,
        )
    )
    mini = ReachyMini.__new__(ReachyMini)
    mini.robot_name = "lite"

    with patch.object(mini, "_connect_single", return_value=client):
        assert mini._connect_and_verify_name("localhost", 1234, 1.0) is client

    assert client.get_status(timeout=0.1).robot_name == "lite"
    client.disconnect()


# A closed socket raises either child of ConnectionClosed, depending on how
# the daemon hung up: both must reach the app as ConnectionError.
CLOSED_SOCKET_ERRORS = [
    websockets.exceptions.ConnectionClosedError(None, None),
    websockets.exceptions.ConnectionClosedOK(None, None),
]


def _client_whose_send_fails(
    mock_connect: MagicMock, error: websockets.exceptions.ConnectionClosed
) -> WSClient:
    """Build a client that still believes it is alive, on a socket already closed."""
    fake_ws = _make_fake_ws()
    fake_ws.send.side_effect = error
    mock_connect.return_value = fake_ws
    client = WSClient(host="localhost", port=1234)
    client._is_alive = True
    return client


@pytest.mark.parametrize("error", CLOSED_SOCKET_ERRORS)
@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_send_command_on_closed_socket_raises_connection_error(
    mock_connect: MagicMock, error: websockets.exceptions.ConnectionClosed
) -> None:
    """send_command raises the documented ConnectionError, keeping the cause."""
    client = _client_whose_send_fails(mock_connect, error)

    with pytest.raises(ConnectionError) as exc_info:
        client.send_command(SetTargetCmd(head=[0.0] * 16))

    assert exc_info.value.__cause__ is error
    client.disconnect()


@pytest.mark.parametrize("error", CLOSED_SOCKET_ERRORS)
@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_send_task_request_on_closed_socket_forgets_the_task(
    mock_connect: MagicMock, error: websockets.exceptions.ConnectionClosed
) -> None:
    """send_task_request raises ConnectionError and leaves no task behind."""
    client = _client_whose_send_fails(mock_connect, error)
    request = GotoTaskRequest(
        head=None,
        antennas=[0.0, 0.0],
        duration=1.0,
        method=InterpolationTechnique.MIN_JERK,
        body_yaw=None,
    )

    with pytest.raises(ConnectionError) as exc_info:
        client.send_task_request(request)

    assert exc_info.value.__cause__ is error
    assert client.tasks == {}
    client.disconnect()


GOTO_REQUEST = GotoTaskRequest(
    head=None,
    antennas=[0.0, 0.0],
    duration=1.0,
    method=InterpolationTechnique.MIN_JERK,
    body_yaw=None,
)


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_waiting_task_fails_at_once_when_the_line_drops(
    mock_connect: MagicMock,
) -> None:
    """A task waiting when the line drops raises ConnectionError, not TimeoutError."""
    mock_connect.side_effect = lambda *args, **kwargs: _make_fake_ws()
    client = WSClient(host="localhost", port=1234)
    client._is_alive = True
    task_uid = client.send_task_request(GOTO_REQUEST)

    assert client._ws is not None
    client._ws.close()  # the daemon hangs up, not us

    with pytest.raises(ConnectionError):
        client.wait_for_task_completion(task_uid, timeout=2.0)
    client.disconnect()


@patch("reachy_mini.io.ws_client.RECONNECT_FIRST_DELAY_S", 0.01)
@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_disconnect_fails_waiting_tasks_and_does_not_reconnect(
    mock_connect: MagicMock,
) -> None:
    """After disconnect(), waiting tasks raise ConnectionError and nothing reconnects."""
    mock_connect.side_effect = lambda *args, **kwargs: _make_fake_ws()
    client = WSClient(host="localhost", port=1234)
    client._is_alive = True
    task_uid = client.send_task_request(GOTO_REQUEST)

    client.disconnect()

    with pytest.raises(ConnectionError):
        client.wait_for_task_completion(task_uid, timeout=2.0)
    assert client._recv_thread is not None
    client._recv_thread.join(timeout=2.0)
    assert not client._recv_thread.is_alive()
    assert mock_connect.call_count == 1


def _wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll a condition until it holds or the timeout ends."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@patch("reachy_mini.io.ws_client.RECONNECT_FIRST_DELAY_S", 0.01)
def test_client_reconnects_when_the_server_hangs_up() -> None:
    """A real server closes the line: the client dials again and commands get through."""
    joints = JointPositionsMsg(
        head_joint_positions=[0.0] * 7, antennas_joint_positions=[0.0, 0.0]
    ).model_dump_json()
    head_pose = HeadPoseMsg(
        head_pose=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).model_dump_json()
    connections: list[ServerConnection] = []
    received: list[tuple[int, str]] = []

    def daemon(connection: ServerConnection) -> None:
        """Stream state like the daemon, and record what the client sends."""
        connections.append(connection)
        index = len(connections)
        try:
            while True:
                connection.send(joints)
                connection.send(head_pose)
                try:
                    received.append((index, str(connection.recv(timeout=0.02))))
                except TimeoutError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass

    with serve(daemon, "127.0.0.1", 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        client = WSClient(host="127.0.0.1", port=port)
        client.wait_for_connection(timeout=2.0)

        connections[0].close()  # the daemon hangs up
        assert _wait_for(lambda: len(connections) == 2)

        # Like an app: a send may still raise ConnectionError until the
        # client is on the new line, so retry until one gets through.
        def command_reached_new_line() -> bool:
            try:
                client.send_command(SetTargetCmd(head=[0.0] * 16))
            except ConnectionError:
                return False
            return any(index == 2 for index, _ in received)

        assert _wait_for(command_reached_new_line)
        client.disconnect()
        server.shutdown()
