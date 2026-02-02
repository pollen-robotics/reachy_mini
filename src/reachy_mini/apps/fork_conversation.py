"""Fork conversation app as a standalone customized app."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import questionary
import toml
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from .assistant import validate_app_name, validate_location_and_git_repo

CONVERSATION_APP_REPO = (
    "https://github.com/pollen-robotics/reachy_mini_conversation_app"
)
CONVERSATION_APP_PACKAGE = "reachy_mini_conversation_app"
CONVERSATION_TEMPLATE_DIR = Path(__file__).parent / "templates" / "fork_conversation"

_CLEANUP_DIRS = [".github", ".idea", ".vscode"]
_CLEANUP_FILES = ["uv.lock", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]


def create_from_conversation_app(
    console: Console,
    app_name: str | None,
    app_path: Path | None,
) -> Path:
    """Create a new app by forking the conversation app."""
    # Get user input
    app_name, app_path, profile_name = _fork_cli(console, app_name, app_path)

    target_path = app_path / app_name
    if target_path.exists():
        console.print(f"❌ Folder {target_path} already exists.", style="bold red")
        exit()

    # Clone and customize
    console.print(f"\nCloning conversation app from {CONVERSATION_APP_REPO}...")
    _clone_repo(console, target_path)

    console.print(f"Renaming package to {app_name}...")
    _rename_package(console, target_path, app_name)

    console.print("Customizing pyproject.toml...")
    _update_pyproject(console, target_path, app_name)

    console.print("Setting locked profile in config.py...")
    _update_config(console, target_path, app_name, profile_name)

    console.print("Creating README.md...")
    display_name = " ".join(word.capitalize() for word in app_name.split("_"))
    _update_readme(console, target_path, app_name, display_name, profile_name)

    console.print("Creating landing page (index.html, style.css)...")
    _create_landing_page(target_path, app_name, display_name)

    console.print("Creating simplified static files...")
    _create_static_files(target_path, app_name, display_name)

    console.print(f"Creating profile folder: profiles/{profile_name}/")
    _create_profile(console, target_path, app_name, profile_name)

    console.print("Updating .gitignore...")
    _update_gitignore(target_path, app_name)

    console.print("Cleaning up unnecessary files...")
    _cleanup(target_path, app_name, profile_name)

    console.print("Initializing fresh git repository...")
    _init_git(target_path)

    console.print(f"\n✅ Created '{app_name}' in {target_path}/", style="bold green")
    console.print(f"   - Profile locked to: {profile_name}", style="dim")
    console.print(f"   - Profile folder created: profiles/{profile_name}/", style="dim")

    console.print("\nFiles to customize:", style="bold yellow")
    console.print(
        f"  profiles/{profile_name}/instructions.txt  - System prompt for the assistant"
    )
    console.print(
        f"  profiles/{profile_name}/tools.txt         - Available tools (or 'all')"
    )

    console.print("\nNext steps:", style="bold")
    console.print(f"  cd {target_path}")
    console.print("  pip install -e .")
    console.print("  reachy-mini-app-assistant check .")
    console.print("")
    console.print("To test your app locally:", style="bold")
    console.print(f"  python src/{app_name}/main.py --gradio")
    console.print("  # Then open: http://127.0.0.1:7861/")

    return target_path


def _fork_cli(
    console: Console, app_name: str | None, app_path: Path | None
) -> tuple[str, Path, str]:
    """Prompt user for app name, path, and derive profile name."""
    if app_name is None:
        console.print("$ What name do you want for your app?")
        app_name = questionary.text(">", validate=validate_app_name).ask()
        if app_name is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_name = app_name.strip().lower()

    # Force underscores
    app_name = app_name.replace("-", "_")

    if app_path is None:
        console.print("\n$ Where do you want to create it?")
        app_path_str = questionary.path(
            ">", validate=validate_location_and_git_repo
        ).ask()
        if app_path_str is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_path = Path(app_path_str).expanduser().resolve()

    profile_name = f"_{app_name}_locked_profile"
    return app_name, app_path, profile_name


def _clone_repo(console: Console, target_path: Path) -> None:
    """Clone conversation app repo and remove its git history."""
    # Check git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("❌ Git is not installed or not in PATH.", style="bold red")
        exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_clone = Path(tmpdir) / "clone"

        # Shallow clone
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "-b",
                "develop",
                CONVERSATION_APP_REPO,
                str(tmp_clone),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(
                f"❌ Failed to clone repository: {result.stderr}", style="bold red"
            )
            exit(1)

        # Remove .git folder
        git_dir = tmp_clone / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

        # Copy to target
        shutil.copytree(tmp_clone, target_path)


def _to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _rename_package(console: Console, app_path: Path, app_name: str) -> None:
    """Rename the package folder, update all imports, and rename the main class."""
    old_package_dir = app_path / "src" / CONVERSATION_APP_PACKAGE
    new_package_dir = app_path / "src" / app_name

    if not old_package_dir.exists():
        console.print(
            f"❌ Package folder not found at {old_package_dir}",
            style="bold red",
        )
        exit(1)

    # Rename the package folder
    old_package_dir.rename(new_package_dir)

    # Calculate new class name
    new_class_name = _to_pascal_case(app_name)

    # Find and replace all imports and class name in .py files (src and tests)
    py_dirs = [new_package_dir, app_path / "tests"]
    for py_dir in py_dirs:
        if not py_dir.exists():
            continue
        for py_file in py_dir.rglob("*.py"):
            content = py_file.read_text()
            # Replace imports like "from reachy_mini_conversation_app" and "import reachy_mini_conversation_app"
            new_content = re.sub(
                rf"\b{CONVERSATION_APP_PACKAGE}\b",
                app_name,
                content,
            )
            # Replace the main class name
            new_content = new_content.replace("ReachyMiniConversationApp", new_class_name)
            if new_content != content:
                py_file.write_text(new_content)


def _update_pyproject(console: Console, app_path: Path, app_name: str) -> None:
    """Update pyproject.toml with new app name and entry points."""
    pyproject_path = app_path / "pyproject.toml"

    if not pyproject_path.exists():
        console.print(
            f"❌ pyproject.toml not found at {pyproject_path}",
            style="bold red",
        )
        console.print(
            "The conversation app structure may have changed. Please report this issue.",
            style="dim",
        )
        exit(1)

    with open(pyproject_path, "r") as f:
        data = toml.load(f)

    # Update project name
    data["project"]["name"] = app_name

    # Update scripts:
    script_name = app_name.replace("_", "-")
    class_name = _to_pascal_case(app_name)
    data["project"]["scripts"] = {script_name: f"{app_name}.main:main"}

    # Update entry points
    data["project"]["entry-points"] = {
        "reachy_mini_apps": {app_name: f"{app_name}.main:{class_name}"}
    }

    # Update package-data key from reachy_mini_conversation_app to app_name
    if "tool" in data and "setuptools" in data["tool"]:
        package_data = data["tool"]["setuptools"].get("package-data", {})
        if CONVERSATION_APP_PACKAGE in package_data:
            package_data[app_name] = package_data.pop(CONVERSATION_APP_PACKAGE)

    # Update isort known-local-folder
    if "tool" in data and "ruff" in data["tool"]:
        isort_config = data["tool"]["ruff"].get("lint", {}).get("isort", {})
        if "known-local-folder" in isort_config:
            isort_config["known-local-folder"] = [
                app_name if folder == CONVERSATION_APP_PACKAGE else folder
                for folder in isort_config["known-local-folder"]
            ]

    with open(pyproject_path, "w") as f:
        toml.dump(data, f)


def _update_config(
    console: Console, app_path: Path, app_name: str, profile_name: str
) -> None:
    """Set LOCKED_PROFILE constant in config.py."""
    config_path = app_path / "src" / app_name / "config.py"

    if not config_path.exists():
        console.print(
            f"❌ config.py not found at {config_path}",
            style="bold red",
        )
        console.print(
            "The conversation app structure may have changed. Please report this issue.",
            style="dim",
        )
        exit(1)

    content = config_path.read_text()

    # Replace the LOCKED_PROFILE line
    old_line = "LOCKED_PROFILE: str | None = None"
    new_line = f'LOCKED_PROFILE: str | None = "{profile_name}"'

    if old_line not in content:
        console.print(
            "❌ Could not find LOCKED_PROFILE line in config.py",
            style="bold red",
        )
        console.print(
            f"Expected: {old_line}",
            style="dim",
        )
        console.print(
            "The conversation app structure may have changed. Please report this issue.",
            style="dim",
        )
        exit(1)

    content = content.replace(old_line, new_line)
    config_path.write_text(content)


def _update_readme(
    console: Console, app_path: Path, app_name: str, display_name: str, profile_name: str
) -> None:
    """Rename old README to README_OLD.md and create new README from template."""
    readme_path = app_path / "README.md"
    readme_old_path = app_path / "README_OLD.md"

    if readme_path.exists():
        readme_path.rename(readme_old_path)
        console.print("   Renamed README.md to README_OLD.md", style="dim")

    env = Environment(loader=FileSystemLoader(CONVERSATION_TEMPLATE_DIR))
    template = env.get_template("README.md.j2")
    readme_path.write_text(
        template.render(app_name=app_name, display_name=display_name, profile_name=profile_name)
    )


def _create_landing_page(app_path: Path, app_name: str, display_name: str) -> None:
    """Create landing page (index.html, style.css) from templates."""
    env = Environment(loader=FileSystemLoader(CONVERSATION_TEMPLATE_DIR))

    context = {"app_name": app_name, "display_name": display_name}

    # Render and write index.html
    index_template = env.get_template("index.html.j2")
    (app_path / "index.html").write_text(index_template.render(context))

    # Render and write style.css
    style_template = env.get_template("style.css.j2")
    (app_path / "style.css").write_text(style_template.render(context))


def _create_static_files(app_path: Path, app_name: str, display_name: str) -> None:
    """Replace conversation app static files with simplified API-key-only version."""
    template_dir = CONVERSATION_TEMPLATE_DIR / "static"
    env = Environment(loader=FileSystemLoader(template_dir))

    context = {"display_name": display_name}

    static_dir = app_path / "src" / app_name / "static"

    # Render and write simplified static files
    (static_dir / "index.html").write_text(
        env.get_template("index.html.j2").render(context)
    )
    (static_dir / "main.js").write_text(env.get_template("main.js.j2").render(context))
    (static_dir / "style.css").write_text(
        env.get_template("style.css.j2").render(context)
    )


def _create_profile(
    console: Console, app_path: Path, app_name: str, profile_name: str
) -> None:
    """Create profile folder with template files."""
    new_profile_dir = app_path / "src" / app_name / "profiles" / profile_name
    new_profile_dir.mkdir(parents=True, exist_ok=True)

    # Render all .j2 templates from profile template directory
    template_dir = CONVERSATION_TEMPLATE_DIR / "profile"
    env = Environment(loader=FileSystemLoader(template_dir))
    context = {"profile_name": profile_name, "app_name": app_name}

    for src_file in template_dir.iterdir():
        if src_file.is_file() and src_file.suffix == ".j2":
            # Remove .j2 extension for output filename
            output_name = src_file.stem
            template = env.get_template(src_file.name)
            (new_profile_dir / output_name).write_text(template.render(context))

    console.print("   Created profile with template files:", style="dim")
    for f in sorted(new_profile_dir.iterdir()):
        console.print(f"     - {f.name}", style="dim")


def _update_gitignore(app_path: Path, app_name: str) -> None:
    """Update .gitignore to use new app name in paths."""
    gitignore_path = app_path / ".gitignore"
    if not gitignore_path.exists():
        return

    content = gitignore_path.read_text()
    new_content = content.replace(
        f"src/{CONVERSATION_APP_PACKAGE}/",
        f"src/{app_name}/",
    )
    if new_content != content:
        gitignore_path.write_text(new_content)


def _cleanup(app_path: Path, app_name: str, profile_name: str) -> None:
    """Remove files not needed in the forked app."""
    for dir_name in _CLEANUP_DIRS:
        dir_path = app_path / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)

    for file_name in _CLEANUP_FILES:
        file_path = app_path / file_name
        if file_path.exists():
            file_path.unlink()

    # Remove all profile folders except the new locked profile
    profiles_dir = app_path / "src" / app_name / "profiles"
    if profiles_dir.exists():
        for profile_dir in profiles_dir.iterdir():
            if profile_dir.is_dir() and profile_dir.name != profile_name:
                shutil.rmtree(profile_dir)


def _init_git(app_path: Path) -> None:
    """Initialize a fresh git repository."""
    subprocess.run(
        ["git", "init"],
        cwd=app_path,
        capture_output=True,
        check=True,
    )
