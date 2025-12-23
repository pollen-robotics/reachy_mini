"""Logs router for Reachy Mini Daemon API.

This module provides a WebSocket endpoint to stream journalctl logs for the daemon service.
"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/logs")
logger = logging.getLogger(__name__)


@router.websocket("/ws/daemon")
async def websocket_daemon_logs(websocket: WebSocket) -> None:
    """WebSocket endpoint to stream journalctl logs for reachy-mini-daemon service in real time."""
    await websocket.accept()

    process = None
    try:
        # Start journalctl subprocess to stream daemon logs
        process = await asyncio.create_subprocess_exec(
            "journalctl",
            "-u", "reachy-mini-daemon",
            "-f",  # follow mode (tail)
            "-n", "100",  # start with last 100 lines
            "--since", "boot",  # only from current boot
            "--output", "short-iso",  # ISO timestamp format
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Stream lines to WebSocket
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            # Send line to client
            decoded_line = line.decode().strip()
            if decoded_line:  # Only send non-empty lines
                await websocket.send_text(decoded_line)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except FileNotFoundError:
        # journalctl not available
        error_msg = "ERROR: journalctl command not found. This feature requires systemd."
        logger.error(error_msg)
        try:
            await websocket.send_text(error_msg)
        except Exception:
            pass
    except Exception as e:
        error_msg = f"ERROR: Failed to stream logs: {str(e)}"
        logger.error(error_msg)
        try:
            await websocket.send_text(error_msg)
        except Exception:
            pass
    finally:
        # Terminate journalctl process if still running
        if process and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating journalctl process: {e}")

        # Close WebSocket connection
        try:
            await websocket.close()
        except Exception:
            pass
