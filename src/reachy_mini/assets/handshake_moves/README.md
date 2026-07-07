# Handshake emotion move

`excited.json` (+ `excited.wav`) is the single fixed "excited" move played by
the torque-ON emotion antenna-button code (left, right, left, right external;
see `daemon/backend/secret_handshake.py` and the design spec
`docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md`).

It is bundled here so it ships in the wheel and works offline (no network
fetch at play time). Loaded via `RecordedMove` and played with a short
`initial_goto_duration` so the robot eases into the move's start pose.

Provenance: `enthusiastic1` from `pollen-robotics/reachy-mini-emotions-library`
(HF dataset, snapshot 152e84b8f46b88c4b52dd34bbef6975637366177), ~2.73 s,
"A movement to celebrate incredible news." Renamed to `excited` for its role
here. To swap the move, drop in another library move's JSON (+ optional WAV)
under this name.
