from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io import Client, Server

from test_cmd import assert_command_equal, random_command


def test_client_server_io():
    s = Server()
    s.start()

    cmd = s.get_latest_command()
    assert_command_equal(cmd, ReachyMiniCommand().default())

    c = Client()

    new_cmd = random_command()

    s.command_received_event().clear()
    c.send_command(new_cmd)
    s.command_received_event().wait()

    recv_cmd = s.get_latest_command()
    assert_command_equal(recv_cmd, new_cmd)

    recv_cmd = s.get_latest_command()
    assert_command_equal(recv_cmd, new_cmd)

    s.stop()
