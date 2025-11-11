#!/usr/bin/env python3
"""Reachy Mini Control Center - Gradio Web UI

A web-based control panel for Reachy Mini with extension support.
Inspired by Automatic1111's Stable Diffusion WebUI architecture.
"""

import gradio as gr
import logging
import argparse
import signal
import atexit
import threading
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.daemon_client import DaemonClient
from core.extension_manager import ExtensionManager
from core.ui_builder import UIBuilder
from core.state_manager import StateManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class ReachyControlCenter:
    """Main control center application."""

    def __init__(
        self,
        daemon_url: str = "http://localhost:8100",
        extensions_dir: str = "extensions",
        builtin_dir: str = "extensions-builtin"
    ):
        """Initialize control center.

        Args:
            daemon_url: URL of Reachy Mini daemon
            extensions_dir: Path to user extensions directory
            builtin_dir: Path to built-in extensions directory
        """
        self.daemon_url = daemon_url
        self.daemon_client = DaemonClient(daemon_url)
        self.extension_manager = ExtensionManager(
            extensions_dir=Path(extensions_dir),
            builtin_dir=Path(builtin_dir),
            daemon_client=self.daemon_client
        )
        self.ui_builder = UIBuilder(
            extension_manager=self.extension_manager,
            daemon_client=self.daemon_client
        )
        self.state_manager = StateManager()

        # Connection monitoring
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self._shutdown_flag = False

        # Discover extensions on init
        self.extension_manager.discover_extensions()

    def create_ui(self) -> gr.Blocks:
        """Create the main Gradio interface.

        Returns:
            gr.Blocks demo object
        """
        with gr.Blocks(
            analytics_enabled=False,
            title="Reachy Control Center",
            theme=gr.themes.Soft()
        ) as demo:

            # Header
            gr.Markdown("# Reachy Mini Control Center")

            # Connection status bar
            with gr.Row():
                connection_status = gr.Textbox(
                    label="Daemon Connection",
                    value="Checking...",
                    interactive=False,
                    scale=1
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=0, size="sm")

            gr.Markdown("---")

            # Main tabs
            with gr.Tabs() as tabs:

                # Manual Control Tab (Built-in)
                with gr.Tab("Manual Control"):
                    self._build_manual_control_tab()

                # Move Library Tab (Built-in)
                with gr.Tab("Move Library"):
                    self._build_move_library_tab()

                # Extension tabs (auto-generated)
                for extension in self.extension_manager.get_enabled_extensions():
                    if not extension.is_builtin:
                        self.ui_builder.build_extension_tab(extension)

                # Settings Tab (Built-in)
                with gr.Tab("Settings"):
                    self._build_settings_tab()

            # Wire up connection refresh
            refresh_btn.click(
                fn=self.check_connection,
                inputs=[],
                outputs=[connection_status]
            )

            # Auto-check connection on load and start monitoring
            def on_load():
                self.start_monitoring()
                return self.check_connection()

            demo.load(
                fn=on_load,
                inputs=[],
                outputs=[connection_status]
            )

            # Cleanup on unload
            demo.unload(fn=self.cleanup_all)

        return demo

    def _build_manual_control_tab(self) -> None:
        """Build manual control tab."""
        # Get saved state
        saved_state = self.state_manager.get_manual_control_state()

        with gr.Row():
            # Left: Video stream
            with gr.Column(scale=2):
                gr.Markdown("## Live View")

                # MuJoCo 3D viewer via MJPEG stream
                video_stream = gr.HTML(f"""
                    <div style="text-align:center;">
                        <img
                            src="{self.daemon_client.get_camera_stream_url()}"
                            style="width:100%; max-height:600px; object-fit:contain; border-radius:8px; border:1px solid #ccc;"
                            alt="Robot Camera Stream"
                            onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22640%22 height=%22480%22><rect width=%22100%25%22 height=%22100%25%22 fill=%22%23f0f0f0%22/><text x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 fill=%22%23999%22>Camera Not Available</text></svg>';"
                        />
                    </div>
                """)

            # Right: Controls
            with gr.Column(scale=1):
                gr.Markdown("## Position Control")

                with gr.Accordion("Head Position", open=True):
                    head_x = gr.Slider(-50, 50, value=saved_state.get("head_x", 0), step=1, label="X (mm)")
                    head_y = gr.Slider(-60, 60, value=saved_state.get("head_y", 0), step=1, label="Y (mm)")
                    head_z = gr.Slider(-30, 80, value=saved_state.get("head_z", 0), step=1, label="Z (mm)")

                with gr.Accordion("Head Rotation", open=True):
                    head_yaw = gr.Slider(-90, 90, value=saved_state.get("head_yaw", 0), step=5, label="Yaw (°)")
                    head_pitch = gr.Slider(-45, 45, value=saved_state.get("head_pitch", 0), step=5, label="Pitch (°)")
                    head_roll = gr.Slider(-45, 45, value=saved_state.get("head_roll", 0), step=5, label="Roll (°)")

                with gr.Accordion("Antennas", open=False):
                    left_antenna = gr.Slider(-3, 3, value=saved_state.get("left_antenna", 0), step=0.1, label="Left Antenna")
                    right_antenna = gr.Slider(-3, 3, value=saved_state.get("right_antenna", 0), step=0.1, label="Right Antenna")

                duration = gr.Slider(0.1, 3.0, value=saved_state.get("duration", 0.5), step=0.1, label="Duration (s)")

                move_btn = gr.Button("Move to Position", variant="primary")
                stop_btn = gr.Button("Emergency Stop", variant="stop")

                status = gr.Textbox(label="Status", interactive=False)

                # Wire up controls
                move_btn.click(
                    fn=self.move_to_position,
                    inputs=[head_x, head_y, head_z, head_yaw, head_pitch, head_roll,
                           left_antenna, right_antenna, duration],
                    outputs=[status]
                )

                stop_btn.click(
                    fn=self.emergency_stop,
                    inputs=[],
                    outputs=[status]
                )

                # Save state when sliders change
                def save_slider_state(x, y, z, yaw, pitch, roll, l_ant, r_ant, dur):
                    self.state_manager.set_manual_control_state(
                        head_x=x, head_y=y, head_z=z,
                        head_yaw=yaw, head_pitch=pitch, head_roll=roll,
                        left_antenna=l_ant, right_antenna=r_ant,
                        duration=dur
                    )

                for slider in [head_x, head_y, head_z, head_yaw, head_pitch, head_roll,
                              left_antenna, right_antenna, duration]:
                    slider.change(
                        fn=save_slider_state,
                        inputs=[head_x, head_y, head_z, head_yaw, head_pitch, head_roll,
                               left_antenna, right_antenna, duration],
                        outputs=[]
                    )

    def _build_move_library_tab(self) -> None:
        """Build move library tab."""
        gr.Markdown("## Pre-Recorded Moves")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Dances")
                dance_moves = self.daemon_client.list_available_moves(
                    "pollen-robotics/reachy-mini-dances-library"
                )
                dance_radio = gr.Radio(
                    choices=dance_moves,
                    label="Select Dance",
                    value=dance_moves[0] if dance_moves else None
                )

            with gr.Column():
                gr.Markdown("### Emotions")
                emotion_moves = self.daemon_client.list_available_moves(
                    "pollen-robotics/reachy-mini-emotions-library"
                )
                emotion_radio = gr.Radio(
                    choices=emotion_moves,
                    label="Select Emotion",
                    value=emotion_moves[0] if emotion_moves else None
                )

        execute_btn = gr.Button("Execute Move", variant="primary")
        stop_btn = gr.Button("Stop Move", variant="stop")
        move_status = gr.Textbox(label="Status", interactive=False)

        # Wire up controls
        execute_btn.click(
            fn=lambda dance, emotion: self.execute_move(dance or emotion, dance is not None),
            inputs=[dance_radio, emotion_radio],
            outputs=[move_status]
        )

        stop_btn.click(
            fn=lambda: self.daemon_client.stop_move() and "✓ Move stopped" or "✗ Failed to stop",
            inputs=[],
            outputs=[move_status]
        )

    def _build_settings_tab(self) -> None:
        """Build settings tab."""
        gr.Markdown("## Settings")

        with gr.Accordion("Connection Settings", open=True):
            daemon_url_input = gr.Textbox(
                label="Daemon URL",
                value=self.daemon_url,
                interactive=True
            )
            test_btn = gr.Button("Test Connection")
            test_result = gr.Textbox(label="Result", interactive=False)

            test_btn.click(
                fn=lambda url: self._test_connection(url),
                inputs=[daemon_url_input],
                outputs=[test_result]
            )

        with gr.Accordion("Extensions", open=True):
            gr.Markdown("### Installed Extensions")

            extensions_list = []
            for ext in self.extension_manager.extensions:
                ext_name = ext.manifest.get("extension", {}).get("name", ext.name)
                ext_type = "Built-in" if ext.is_builtin else "User"
                extensions_list.append(f"- **{ext_name}** ({ext_type})")

            if extensions_list:
                gr.Markdown("\n".join(extensions_list))
            else:
                gr.Markdown("*No extensions installed*")

    # Callback methods

    def check_connection(self) -> str:
        """Check daemon connection status."""
        if self.daemon_client.check_connection():
            status = self.daemon_client.get_status()
            if status:
                return f"✓ Connected to daemon | Backend: {status.get('backend', 'unknown')}"
            return "✓ Connected"
        return "✗ Disconnected - Start daemon first"

    def move_to_position(
        self,
        x: float, y: float, z: float,
        yaw: float, pitch: float, roll: float,
        left_ant: float, right_ant: float,
        duration: float
    ) -> str:
        """Move robot to specified position."""
        import math

        success = self.daemon_client.set_target(
            head_x=x,
            head_y=y,
            head_z=z,
            head_yaw=math.radians(yaw),
            head_pitch=math.radians(pitch),
            head_roll=math.radians(roll),
            left_antenna=left_ant,
            right_antenna=right_ant,
            duration=duration
        )

        if success:
            return "✓ Move command sent"
        return "✗ Failed to send move command"

    def emergency_stop(self) -> str:
        """Emergency stop all robot movement."""
        if self.daemon_client.stop_move():
            return "✓ Emergency stop activated"
        return "✗ Failed to stop"

    def execute_move(self, move_name: str, is_dance: bool) -> str:
        """Execute a pre-recorded move."""
        if not move_name:
            return "✗ No move selected"

        dataset = (
            "pollen-robotics/reachy-mini-dances-library"
            if is_dance
            else "pollen-robotics/reachy-mini-emotions-library"
        )

        uuid = self.daemon_client.play_move(dataset, move_name)
        if uuid:
            return f"✓ Playing {move_name} (UUID: {uuid[:8]}...)"
        return f"✗ Failed to play {move_name}"

    def _test_connection(self, url: str) -> str:
        """Test connection to daemon at given URL."""
        test_client = DaemonClient(url)
        if test_client.check_connection():
            status = test_client.get_status()
            if status:
                return f"✓ Connection successful | Backend: {status.get('backend', 'unknown')}"
            return "✓ Connection successful"
        return "✗ Connection failed"

    def _connection_monitor_loop(self) -> None:
        """Background thread to monitor daemon connection."""
        logger.info("Connection monitor started")
        while not self._monitor_stop.is_set():
            try:
                if not self.daemon_client.check_connection():
                    logger.warning("Daemon connection lost")
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)
        logger.info("Connection monitor stopped")

    def start_monitoring(self) -> None:
        """Start connection monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(
                target=self._connection_monitor_loop,
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started connection monitoring")

    def stop_monitoring(self) -> None:
        """Stop connection monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=2)
            logger.info("Stopped connection monitoring")

    def cleanup_all(self) -> None:
        """Cleanup all resources before shutdown."""
        if self._shutdown_flag:
            return  # Already cleaning up

        self._shutdown_flag = True
        logger.info("Starting cleanup...")

        try:
            # Stop connection monitoring
            self.stop_monitoring()

            # Stop all extensions
            logger.info("Stopping extensions...")
            for extension in self.extension_manager.get_enabled_extensions():
                try:
                    self.extension_manager.stop_extension(extension.name)
                except Exception as e:
                    logger.error(f"Failed to stop extension {extension.name}: {e}")

            # Stop daemon if we started it
            if self.daemon_client.is_daemon_running():
                logger.info("Stopping daemon...")
                self.daemon_client.stop_daemon()

            # Save final state
            logger.info("Saving state...")
            self.state_manager.cleanup()

            logger.info("Cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reachy Control Center")
    parser.add_argument(
        "--daemon-url",
        default="http://localhost:8100",
        help="URL of Reachy Mini daemon (default: http://localhost:8100)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run control center on (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Reachy Control Center...")
    logger.info(f"Daemon URL: {args.daemon_url}")

    # Create control center
    app = ReachyControlCenter(daemon_url=args.daemon_url)

    # Register cleanup handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating cleanup...")
        app.cleanup_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(app.cleanup_all)

    # Build UI
    demo = app.create_ui()

    # Launch
    logger.info(f"Launching on port {args.port}...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            inbrowser=True
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        app.cleanup_all()
    except Exception as e:
        logger.error(f"Error during launch: {e}")
        app.cleanup_all()
        raise


if __name__ == "__main__":
    main()
