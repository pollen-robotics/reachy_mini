# Telemetry

The daemon sends anonymous usage data to help us see how Reachy Mini is used and where it breaks. It is **on by default** and you can turn it off at any time (see below).

## What is sent

Every event carries:

- a random robot id, created when telemetry is enabled and deleted when it is disabled
- the form factor (`wireless`, `lite`, or `mujoco` / `mockup` for the simulators)
- the daemon version and how it was installed (`pypi`, `git`, `editable`)
- the OS and its version (the ReachyMiniOS image version on Reachy Mini Wireless; macOS, Windows or Linux and their version on the computer running the daemon for Reachy Mini Lite), and the CPU architecture

Events:

| Event | Extra data |
|---|---|
| `daemon_started` | time to ready, whether the previous run exited cleanly |
| `daily_heartbeat` | uptime |
| `robot_app_started` / `robot_app_stopped` | app id, duration, exit code, crashed |
| `robot_app_installed` / `robot_app_removed` | app id, success |
| `update_result` | previous and new version, success, error type |
| `client_connected` | client kind (desktop, mobile, web, Python SDK), its version and OS |

The app id is the Hugging Face Space id for public apps only. Private or local apps are reported as `private`.

## What is never sent

IP addresses, location, Wi-Fi names, hostname, robot name, Hugging Face account or token, hardware serial or id, logs, conversation content, camera images or audio.

## Where it goes

Events are sent to [PostHog](https://posthog.com) EU Cloud (servers in the European Union), with IP-based location disabled and no person profiles.

## Turning it off

The simplest way is the **Share anonymous usage data** switch: in the desktop app's settings, or under **About & diagnostics** in the mobile app. It applies right away, no restart needed.

You can also set `telemetry_enabled` to `false` in the daemon config file, then restart the daemon:

- Reachy Mini Wireless: `/home/pollen/.config/reachy_mini/daemon_config.json`
- Reachy Mini Lite: `daemon_config.json` in the `reachy_mini` folder of your user config directory (`~/.config/reachy_mini/` on Linux, `~/Library/Application Support/reachy_mini/` on macOS, `%LOCALAPPDATA%\reachy_mini\reachy_mini\` on Windows)

```json
{
  "telemetry_enabled": false
}
```

The key is written with its default value the first time the daemon starts. If the file can't be parsed or the value isn't `true` or `false`, telemetry stays off.

Or, without a restart, from the daemon API:

```bash
curl http://reachy-mini.local:8000/api/telemetry
curl -X PUT http://reachy-mini.local:8000/api/telemetry \
  -H "Content-Type: application/json" -d '{"enabled": false}'
```

Use `localhost` instead of `reachy-mini.local` for Reachy Mini Lite. Disabling deletes the robot id and drops any events not yet sent.

Any of these environment variables disables telemetry whatever the setting: `REACHY_MINI_TELEMETRY=0`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`.

The implementation is in [`src/reachy_mini/daemon/telemetry.py`](https://github.com/pollen-robotics/reachy_mini/blob/main/src/reachy_mini/daemon/telemetry.py). Questions: contact@pollen-robotics.com.
