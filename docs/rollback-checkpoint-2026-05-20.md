# Rollback checkpoint -- 2026-05-20

Last commit on `feature/daemon-side-move-upload` before starting
**Option E** (daemon-side audio playback for guaranteed sync). If
Option E doesn't pan out and we want to keep streaming audio from the
browser, this is what the branch should look like.

## Verified working at this point

- Fire-and-forget upload of moves over the data channel
  (upload_move_start / chunk / finish, no per-chunk acks).
- play_uploaded_move runs Backend.play_move on the uploaded slot,
  broadcasts a started event and a finished/cancelled/error event.
- cancel_move flips _move_cancelled, play_move exits cleanly.
- Optional gzip+base64 encoding on the upload chunks
  (UploadMoveStartCmd.encoding).
- Linters + Pytest green on this commit on CI
  (https://github.com/pollen-robotics/reachy_mini/actions).
- Validated end-to-end against the wireless robot via
  /Users/remi/reachy_mini_apps/marionette-experimental/tests/scripts/wireless_e2e.py.
- AUDIO is still played from the browser via the existing WebRTC
  audio sender path. Sync is hand-tunable but per-platform; this is
  what Option E aims to fix.

## To roll back the daemon

```bash
cd /Users/remi/reachy_mini_apps/reachy_mini
git checkout feature/daemon-side-move-upload
git reset --hard e955e28e
git push --force-with-lease origin feature/daemon-side-move-upload
# Then re-deploy on the robot:
curl -X POST "http://reachy-mini.local:8000/update/start-from-ref?git_ref=feature/daemon-side-move-upload"
```

Matching browser checkpoint: see
[`/Users/remi/reachy_mini_apps/marionette-experimental/docs/rollback-checkpoint-2026-05-20.md`](file:///Users/remi/reachy_mini_apps/marionette-experimental/docs/rollback-checkpoint-2026-05-20.md).

## Commit reference

- `feature/daemon-side-move-upload` @ **e955e28e** -- "lint: split docstring summary lines (D205)"
- Companion JS app: `marionette-experimental` `main` @ **1aed024**
