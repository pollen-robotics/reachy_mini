from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io import Client, Server

from test_cmd import assert_command_equal, random_command


def test_client_server_io():
    try:
        s = Server()
        s.start()

        cmd = s.get_latest_command()
        assert_command_equal(cmd, ReachyMiniCommand().default())

        c = Client()

        new_cmd = random_command()

        s.command_received_event().clear()
        c.send_command(new_cmd)
        assert s.command_received_event().wait(timeout=0.1)

        recv_cmd = s.get_latest_command()
        assert_command_equal(recv_cmd, new_cmd)

        recv_cmd = s.get_latest_command()
        assert_command_equal(recv_cmd, new_cmd)

        s.stop()
    except Exception as e:
        s.stop()
        raise e


if __name__ == "__main__":
    test_client_server_io()
    print("Test passed successfully.")
