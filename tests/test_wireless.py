import pytest

from reachy_mini.reachy_mini import ReachyMini


@pytest.mark.wireless
def test_daemon_wireless_client_disconnection() -> None:
    with ReachyMini(media_backend="no_media", localhost_only=False) as mini:
        status = mini.client.get_status()
        assert status['state'] == "running"
        assert status['wireless_version'] is True
        assert not status['simulation_enabled']
        assert status['error'] is None
        assert status['backend_status']['motor_control_mode'] == "enabled"
        assert status['backend_status']['error'] is None
        assert isinstance(status['wlan_ip'], str)
        assert status['wlan_ip'].count('.') == 3
        assert all(0 <= int(part) <= 255 for part in status['wlan_ip'].split('.') if part.isdigit())
