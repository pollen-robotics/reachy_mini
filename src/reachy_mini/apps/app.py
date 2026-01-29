"""Reachy Mini Application Base Class.

This module provides a base class for creating Reachy Mini applications.
It includes methods for running the application, stopping it gracefully,
and creating a new app project with a specified name and path.

It uses Jinja2 templates to generate the necessary files for the app project.
"""

import argparse
import importlib
import logging
import threading
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reachy_mini.reachy_mini import ReachyMini


class ReachyMiniApp(ABC):
    """Base class for Reachy Mini applications."""

    custom_app_url: str | None = None
    dont_start_webserver: bool = False
    request_media_backend: str | None = None

    def __init__(self, running_on_wireless: bool = False) -> None:
        """Initialize the Reachy Mini app."""
        self.stop_event = threading.Event()
        self.error: str = ""
        self.logger = logging.getLogger("reachy_mini.app")

        # Detect if daemon is available on localhost
        # If yes, use localhost connection. If no, use multicast scouting for remote daemon.
        self.daemon_on_localhost = self._check_daemon_on_localhost()
        self.logger.info(f"Daemon on localhost: {self.daemon_on_localhost}")

        # Media backend is now auto-detected by ReachyMini, just use "default"
        self.media_backend = (
            self.request_media_backend
            if self.request_media_backend is not None
            else "default"
        )

        self.settings_app: FastAPI | None = None
        if self.custom_app_url is not None and not self.dont_start_webserver:
            self.settings_app = FastAPI()

            static_dir = self._get_instance_path().parent / "static"
            if static_dir.exists():
                self.settings_app.mount(
                    "/static", StaticFiles(directory=static_dir), name="static"
                )

                index_file = static_dir / "index.html"
                if index_file.exists():

                    @self.settings_app.get("/")
                    async def index() -> FileResponse:
                        """Serve the settings app index page."""
                        return FileResponse(index_file)

    @staticmethod
    def _check_daemon_on_localhost(port: int = 8000, timeout: float = 0.5) -> bool:
        """Check if daemon is reachable on localhost.

        Args:
            port: Port to check (default: 8000)
            timeout: Connection timeout in seconds

        Returns:
            True if daemon responds on localhost, False otherwise

        """
        import socket

        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    def wrapped_run(self, *args: Any, **kwargs: Any) -> None:
        """Wrap the run method with Reachy Mini context management."""
        settings_app_t: threading.Thread | None = None
        if self.settings_app is not None:
            import uvicorn

            assert self.custom_app_url is not None
            url = urlparse(self.custom_app_url)
            assert url.hostname is not None and url.port is not None

            config = uvicorn.Config(
                self.settings_app,
                host=url.hostname,
                port=url.port,
            )
            server = uvicorn.Server(config)

            def _server_run() -> None:
                """Run the settings FastAPI app."""
                t = threading.Thread(target=server.run)
                t.start()
                self.stop_event.wait()
                server.should_exit = True
                t.join()

            settings_app_t = threading.Thread(target=_server_run)
            settings_app_t.start()

        try:
            self.logger.info("Starting Reachy Mini app...")
            self.logger.info(f"Using media backend: {self.media_backend}")
            self.logger.info(f"Daemon on localhost: {self.daemon_on_localhost}")

            # Force the connection mode based on daemon location detection
            connection_mode: Literal["localhost_only", "network"] = (
                "localhost_only" if self.daemon_on_localhost else "network"
            )

            with ReachyMini(
                media_backend=self.media_backend,
                connection_mode=connection_mode,
                *args,
                **kwargs,  # type: ignore
            ) as reachy_mini:
                self.run(reachy_mini, self.stop_event)
        except Exception:
            self.error = traceback.format_exc()
            raise
        finally:
            if settings_app_t is not None:
                self.stop_event.set()
                settings_app_t.join()

    @abstractmethod
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the main logic of the app.

        Args:
            reachy_mini (ReachyMini): The Reachy Mini instance to interact with.
            stop_event (threading.Event): An event that can be set to stop the app gracefully.

        """
        pass

    def stop(self) -> None:
        """Stop the app gracefully."""
        self.stop_event.set()
        print("App is stopping...")

    def _get_instance_path(self) -> Path:
        """Get the file path of the app instance."""
        module_name = type(self).__module__
        mod = importlib.import_module(module_name)
        assert mod.__file__ is not None

        return Path(mod.__file__).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="App creation and publishing assistant for Reachy Mini."
    )
    # create/check/publish
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    create_parser = subparsers.add_parser("create", help="Create a new app project")
    create_parser.add_argument(
        "--template",
        type=str,
        choices=["default", "conversation"],
        default="default",
        help="Template to use: 'default' (blank app) or 'conversation' (fork conversation app)",
    )
    create_parser.add_argument(
        "app_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the app to create.",
    )
    create_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path where the app project will be created.",
    )
    create_parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Publish the app to Hugging Face Spaces immediately after creation.",
    )
    create_parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the space private (default is public). Only used with --publish.",
    )

    check_parser = subparsers.add_parser("check", help="Check an existing app project")
    check_parser.add_argument(
        "app_path",
        type=str,
        nargs="?",
        default=None,
        help="Local path to the app to check.",
    )

    publish_parser = subparsers.add_parser(
        "publish", help="Publish the app to the Reachy Mini app store"
    )
    publish_parser.add_argument(
        "app_path",
        type=str,
        nargs="?",
        default=None,
        help="Local path to the app to publish.",
    )
    publish_parser.add_argument(
        "commit_message",
        type=str,
        nargs="?",
        default=None,
        help="Commit message for the app publish.",
    )
    publish_parser.add_argument(
        "--official",
        action="store_true",
        required=False,
        default=False,
        help="Request to publish the app as an official Reachy Mini app.",
    )
    publish_parser.add_argument(
        "--nocheck",
        action="store_true",
        required=False,
        default=False,
        help="Don't run checks before publishing the app.",
    )
    privacy_group = publish_parser.add_mutually_exclusive_group()
    privacy_group.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face Space private.",
    )
    privacy_group.add_argument(
        "--public",
        action="store_true",
        help="Make the Hugging Face Space public.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the app assistant."""
    from rich.console import Console

    from . import assistant

    args = parse_args()
    console = Console()
    if args.command == "create":
        if args.template == "conversation":
            from reachy_mini.apps.fork_conversation import create_from_conversation_app

            created_path = create_from_conversation_app(console, args.app_name, args.path)
        else:
            created_path = assistant.create(console, app_name=args.app_name, app_path=args.path)

        if args.publish and created_path:
            console.print("\nPublishing to Hugging Face Spaces...", style="bold blue")
            assistant.publish(
                console,
                app_path=str(created_path),
                commit_message="Initial commit",
                official=False,
                no_check=False,
                private=args.private,
            )
    elif args.command == "check":
        assistant.check(console, app_path=args.app_path)
    elif args.command == "publish":
        # Determine privacy: --private → True, --public → False, neither → None (prompts)
        if args.private:
            private = True
        elif args.public:
            private = False
        else:
            private = None
        assistant.publish(
            console,
            app_path=args.app_path,
            commit_message=args.commit_message,
            official=args.official,
            no_check=args.nocheck,
            private=private,
        )


if __name__ == "__main__":
    main()
