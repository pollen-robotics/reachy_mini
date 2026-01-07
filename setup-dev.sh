#!/bin/bash
set -e

# setup-dev.sh
# Automates the setup of the Reachy Mini development environment on macOS.
# Ensures all dependencies are installed and the Tauri sidecar is rebuilt with local changes.

echo "=== Reachy Mini Development Setup (macOS) ==="

# 1. System Dependencies
echo "[1/4] Checking system dependencies..."
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

echo "Installing/Updating libraries (libsndfile, portaudio, ffmpeg, swig)..."
brew install libsndfile portaudio ffmpeg swig

# 2. Python Dependencies (for local repo)
echo "[2/4] Syncing Python dependencies (uv)..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install --upgrade pip
    pip install uv
fi

# Ensure we are in the repo root
REPO_ROOT="$(pwd)"
if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    echo "❌ Please run this script from the root of the reachy_mini repository."
    exit 1
fi

echo "Syncing dev and kinematics extras (skipping wireless-version)..."
uv sync --extra dev --extra placo_kinematics --extra mujoco

# 3. Rebuild Sidecar
echo "[3/4] Rebuilding Tauri Sidecar with local source..."
cd reachy-mini-desktop-app

# Set REACHY_MINI_SOURCE to the absolute path of the repo root
# This tells the build script to install the package from THIS directory
export REACHY_MINI_SOURCE="$REPO_ROOT"

echo "Building sidecar (this may take a moment)..."
# We need to ensure we run the build script from the desktop app directory context
./scripts/build/build-sidecar-unix.sh

# 4. Final Instructions
echo "=== Setup Complete! ==="
echo "✅ Local environment ready."
echo "✅ Tauri sidecar rebuilt with 'Placo' kinematics engine support."
echo ""
echo "To start the app:"
echo "  cd reachy-mini-desktop-app"
echo "  yarn tauri:dev"
