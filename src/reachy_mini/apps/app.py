"""Reachy Mini Application Base Class.

This module provides a base class for creating Reachy Mini applications.
It includes methods for running the application, stopping it gracefully,
and creating a new app project with a specified name and path.

It uses Jinja2 templates to generate the necessary files for the app project.
"""

import argparse
import os
import threading
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import questionary
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from reachy_mini.reachy_mini import ReachyMini


class ReachyMiniApp(ABC):
    """Base class for Reachy Mini applications."""

    custom_app_url: str | None = None

    def __init__(self) -> None:
        """Initialize the Reachy Mini app."""
        self.stop_event = threading.Event()
        self.error: str = ""

    def wrapped_run(self, *args: Any, **kwargs: Any) -> None:
        """Wrap the run method with Reachy Mini context management."""
        try:
            with ReachyMini(*args, **kwargs) as reachy_mini:
                self.run(reachy_mini, self.stop_event)
        except Exception:
            self.error = traceback.format_exc()
            raise

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


def create_gui(console: Console, app_name: str | None, app_path: Path | None):
    """Create a new Reachy Mini app project using a GUI."""
    if app_name is None:
        # 1) App name
        console.print("$ What is the name of your app ?")
        app_name = questionary.text(
            ">",
            default="",
            validate=lambda text: bool(text.strip()) or "App name cannot be empty.",
        ).ask()

        if app_name is None:
            console.print("[red]Aborted.[/red]")
            return
        app_name = app_name.strip()

    # 2) Language
    console.print("\n$ Choose the language of your app")
    language = questionary.select(
        ">",
        choices=["python", "js"],
        default="python",
    ).ask()
    if language is None:
        console.print("[red]Aborted.[/red]")
        return

    # js is not supported yet
    if language != "python":
        console.print("[red]Currently only Python apps are supported. Aborted.[/red]")
        return

    if app_path is None:
        # 3) App path
        console.print("\n$ Where do you want to create your app project ?")
        app_path = questionary.path(
            ">",
            default="",
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            return
        app_path = Path(app_path).expanduser().resolve()

    return app_name, language, app_path


def create(console: Console, app_name: str, app_path: Path) -> None:
    """Create a new Reachy Mini app project with the given name at the specified path.

    Args:
        console (Console): The console object for printing messages.
        app_name (str): The name of the app to create.
        app_path (Path): The directory where the app project will be created.

    """
    app_name, language, app_path = create_gui(console, app_name, app_path)
    TEMPLATE_DIR = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    def render_template(filename: str, context: Dict[str, str]) -> str:
        template = env.get_template(filename)
        return template.render(context)

    base_path = Path(app_path).resolve() / app_name
    if base_path.exists():
        console.print(f"❌ Folder {base_path} already exists.", style="bold red")
        return

    module_name = app_name.replace("-", "_")
    class_name = "".join(word.capitalize() for word in module_name.split("_"))
    class_name_display = " ".join(word.capitalize() for word in module_name.split("_"))

    base_path.mkdir()
    (base_path / module_name).mkdir()

    # Generate files
    context = {
        "app_name": app_name,
        "package_name": app_name,
        "module_name": module_name,
        "class_name": class_name,
        "class_name_display": class_name_display,
    }

    (base_path / module_name / "__init__.py").touch()
    (base_path / module_name / "main.py").write_text(
        render_template("main.py.j2", context)
    )
    (base_path / "pyproject.toml").write_text(
        render_template("pyproject.toml.j2", context)
    )
    (base_path / "README.md").write_text(render_template("README.md.j2", context))

    (base_path / "index.html").write_text(render_template("index.html.j2", context))
    (base_path / "style.css").write_text(render_template("style.css.j2", context))
    (base_path / ".gitignore").write_text(render_template(".gitignore.j2", context))

    console.print(f"✅ Created app '{app_name}' in {base_path}/", style="bold green")


def check(console: Console, app_path: str) -> None:
    """Check an existing Reachy Mini app project.

    Args:
        console (Console): The console object for printing messages.
        app_path (str): Local path to the app to check.

    """
    if not os.path.exists(app_path):
        console.print(f"[red]App path {app_path} does not exist.[/red]")
        return
    # Placeholder for checking logic
    print(f"Checking app at path '{app_path}'")
    pass


def publish(console: Console, app_path: str, commit_message: str) -> None:
    """Publish the app to the Reachy Mini app store.

    Args:
        console (Console): The console object for printing messages.
        app_path (str): Local path to the app to publish.
        commit_message (str): Commit message for the app publish.

    """
    import huggingface_hub as hf

    if app_path is None:
        console.print("\n$ What is the local path to the app you want to publish?")
        app_path = questionary.path(
            ">",
            default="",
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            return
        app_path = Path(app_path).expanduser().resolve()
    if not os.path.exists(app_path):
        console.print(f"[red]App path {app_path} does not exist.[/red]")
        return
    if not hf.get_token():
        console.print(
            "[red]You need to be logged in to Hugging Face to publish an app.[/red]"
        )
        # Do you want to login now (will run hf auth login)
        if questionary.confirm("Do you want to login now?").ask():
            console.print("Generate a token at https://huggingface.co/settings/tokens")
            hf.login()
        else:
            console.print("[red]Aborted.[/red]")
            return

    username = hf.whoami()["name"]
    repo_path = f"{username}/{Path(app_path).name}"
    repo_url = f"https://huggingface.co/spaces/{repo_path}"

    if hf.repo_exists(repo_path, repo_type="space"):
        os.system(f"cd {app_path} && git pull {repo_url} main")
        console.print("App already exists on Hugging Face Spaces. Updating...")
        commit_message = questionary.text(
            "\n$ Enter a commit message for the update:",
            default="Update app",
        ).ask()
        if commit_message is None:
            console.print("[red]Aborted.[/red]")
            return
        os.system(
            f"cd {app_path} && git add . && git commit -m '{commit_message}' && git push"
        )
        console.print("✅ App updated successfully.")
    else:
        console.print("Do you want your space to be created private or public?")
        privacy = questionary.select(
            ">",
            choices=["private", "public"],
            default="public",
        ).ask()

        # console.print(f"Publishing app at path '{app_path}'")
        hf.create_repo(
            repo_path,
            repo_type="space",
            private=(privacy == "private"),
            exist_ok=False,
            space_sdk="static",
        )
        os.system(
            f"cd {app_path} && git init && git remote add space {repo_url} && git add . && git commit -m 'Initial commit' && git push --set-upstream -f space main:main"
        )

    # print("✅ App published successfully.")


def parse_args():
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

    return parser.parse_args()


def main() -> None:
    """Entry point for the app assistant."""
    args = parse_args()
    console = Console()
    if args.command == "create":
        create(console, app_name=args.app_name, app_path=args.path)
    elif args.command == "check":
        check(console, app_path=args.app_path)
    elif args.command == "publish":
        publish(console, app_path=args.app_path, commit_message=args.commit_message)


if __name__ == "__main__":
    main()
