#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Test reachy_mini startup on the CM4 (Wireless robot).
#
# Steps:
#   1. Push current branch to GitHub
#   2. Update both venvs on CM4 from the branch
#   3. Restart daemon + start backend
#   4. Wait for Zenoh to be up
#   5. Sync marionette code
#   6. Start marionette
#   7. Capture BOOT timing lines
#
# Usage:
#   ./benchmarks/test_cm4.sh                    # default
#   ./benchmarks/test_cm4.sh --skip-sdk-update  # only sync marionette + restart app
#   ./benchmarks/test_cm4.sh --runs 3           # run 3 times (default: 1)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="reachy-mini.local"
USER="pollen"
REMOTE="${USER}@${HOST}"
BRANCH="893-load-apps-faster"
DAEMON_API="http://127.0.0.1:8000"
SKIP_SDK_UPDATE=false
RUNS=1
MARIONETTE_DIR=""

# Parse args
for arg in "$@"; do
    case "$arg" in
        --skip-sdk-update) SKIP_SDK_UPDATE=true ;;
        --runs=*) RUNS="${arg#*=}" ;;
        --runs) shift_next=true ;;
        *) if [ "${shift_next:-}" = true ]; then RUNS="$arg"; shift_next=false; fi ;;
    esac
done

# Find marionette directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MARIONETTE_DIR="$(cd "$REPO_DIR/../marionette/marionette" && pwd 2>/dev/null)" || true
if [ -z "$MARIONETTE_DIR" ] || [ ! -d "$MARIONETTE_DIR" ]; then
    echo "WARNING: marionette/marionette/ not found, skipping marionette sync"
    MARIONETTE_DIR=""
fi

echo "=== CM4 Startup Test ==="
echo "Branch: $BRANCH"
echo "Runs: $RUNS"
echo "Skip SDK update: $SKIP_SDK_UPDATE"
echo ""

# ── 1. Push to GitHub ──────────────────────────────────────────────
if [ "$SKIP_SDK_UPDATE" = false ]; then
    echo "==> Pushing $BRANCH to GitHub..."
    cd "$REPO_DIR"
    git push origin "$BRANCH" 2>&1 | tail -3
    COMMIT=$(git rev-parse --short HEAD)
    echo "    Commit: $COMMIT"
    echo ""

    # ── 2. Update both venvs ───────────────────────────────────────
    echo "==> Updating daemon venv on CM4..."
    ssh "$REMOTE" "/venvs/mini_daemon/bin/pip install --upgrade --force-reinstall \
        'reachy_mini[wireless-version] @ git+https://github.com/pollen-robotics/reachy_mini.git@${BRANCH}' \
        2>&1 | tail -3"
    echo ""

    echo "==> Updating apps venv on CM4..."
    ssh "$REMOTE" "/venvs/apps_venv/bin/pip install --upgrade --force-reinstall \
        'reachy-mini @ git+https://github.com/pollen-robotics/reachy_mini.git@${BRANCH}' \
        2>&1 | tail -3"
    echo ""

    echo "==> Clearing pip cache..."
    ssh "$REMOTE" "/venvs/mini_daemon/bin/pip cache purge 2>/dev/null; /venvs/apps_venv/bin/pip cache purge 2>/dev/null" || true
    echo ""
fi

# ── 3. Sync marionette code ───────────────────────────────────────
if [ -n "$MARIONETTE_DIR" ]; then
    echo "==> Syncing marionette to CM4..."
    rsync -az --delete \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "$MARIONETTE_DIR/" \
        "${REMOTE}:/venvs/apps_venv/lib/python3.12/site-packages/marionette/"
    echo "    Done."
    echo ""
fi

# ── 4. Restart daemon ─────────────────────────────────────────────
echo "==> Restarting daemon..."
ssh "$REMOTE" "sudo systemctl restart reachy-mini-daemon"
sleep 10

# Wait for HTTP API to be up
echo "==> Waiting for daemon HTTP API..."
for i in $(seq 1 30); do
    if ssh "$REMOTE" "curl -sf ${DAEMON_API}/health-check >/dev/null 2>&1"; then
        echo "    API up after ${i}s"
        break
    fi
    sleep 1
done

# ── 5. Start backend ──────────────────────────────────────────────
echo "==> Starting daemon backend..."
ssh "$REMOTE" "curl -sf -X POST '${DAEMON_API}/api/daemon/start?wake_up=false' >/dev/null 2>&1"

# Wait for Zenoh (port 7447)
echo "==> Waiting for Zenoh..."
for i in $(seq 1 60); do
    if ssh "$REMOTE" "ss -tlnp 2>/dev/null | grep -q 7447"; then
        echo "    Zenoh up after ${i}s"
        break
    fi
    sleep 1
done

# Extra wait for backend to settle (motor init, etc.)
echo "==> Waiting for backend to settle..."
sleep 10

# ── 6. Run tests ──────────────────────────────────────────────────
for run in $(seq 1 "$RUNS"); do
    echo ""
    echo "=== Run $run/$RUNS ==="

    # Stop any running app
    ssh "$REMOTE" "curl -sf -X POST '${DAEMON_API}/api/apps/stop-current-app' >/dev/null 2>&1" || true
    sleep 2

    # Start marionette
    echo "==> Starting marionette..."
    ssh "$REMOTE" "curl -sf -X POST '${DAEMON_API}/api/apps/start-app/marionette' >/dev/null 2>&1"

    # Wait for boot to complete
    sleep 20

    # Capture BOOT lines
    echo "==> BOOT timing:"
    ssh "$REMOTE" "journalctl -u reachy-mini-daemon --no-hostname -o cat --since '-25s' | grep BOOT | grep -v '^2'" | while read -r line; do
        echo "    $line"
    done
done

echo ""
echo "=== Done ==="
