"""Reachy Mini app assistant functions."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import questionary
import toml
import yaml
from huggingface_hub import CommitOperationAdd, HfApi, get_repo_discussions, whoami
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


def validate_app_name(text: str) -> bool | str:
    """Validate the app name."""
    if not text.strip():
        return "App name cannot be empty."
    if " " in text:
        return "App name cannot contain spaces."
    if "-" in text:
        return "App name cannot contain dashes ('-'). Please use underscores ('_') instead."
    if "/" in text or "\\" in text:
        return "App name cannot contain slashes or backslashes ('/' or '\\')."
    if "*" in text or "?" in text or "." in text:
        return "App name cannot contain wildcard characters ('*', '?', or '.')."
    return True


def is_git_repo(path: Path) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def validate_location(text: str) -> bool | str:
    """Validate the location where to create the app project."""
    path = Path(text).expanduser().resolve()
    if not os.path.exists(path):
        return f"The path {path} does not exist."

    return True


def validate_location_and_git_repo(text: str) -> bool | str:
    """Validate the location where to create the app project, ensuring it's not in a git repo."""
    path = Path(text).expanduser().resolve()
    if not os.path.exists(path):
        return f"The path {path} does not exist."
    if is_git_repo(path):
        return f"The path {path} is already inside a git repository."

    return True


def create_cli(
    console: Console, app_name: str | None, app_path: Path | None
) -> tuple[str, str, Path]:
    """Create a new Reachy Mini app project using a CLI."""
    interactive = app_name is None or app_path is None

    if app_name is None:
        # 1) App name
        console.print("$ What is the name of your app ?")
        app_name = questionary.text(
            ">",
            default="",
            validate=validate_app_name,
        ).ask()

        if app_name is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_name = app_name.strip().lower()

    # 2) Language - only prompt interactively, default to python
    if interactive:
        console.print("\n$ Choose the language of your app")
        language = questionary.select(
            ">",
            choices=["python", "js"],
            default="python",
        ).ask()
        if language is None:
            console.print("[red]Aborted.[/red]")
            exit()

        # js is not supported yet
        if language != "python":
            console.print("[red]Currently only Python apps are supported. Aborted.[/red]")
            exit()
    else:
        language = "python"

    if app_path is None:
        # 3) App path
        console.print("\n$ Where do you want to create your app project ?")
        app_path = questionary.path(
            ">",
            default="",
            validate=validate_location_and_git_repo,
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_path = Path(app_path).expanduser().resolve()
        if is_git_repo(app_path):
            console.print(
                f"[red] The path {app_path} is already inside a git repository. "
                "Please choose another path. Aborted.[/red]"
            )
            exit()

    return app_name, language, app_path


def create(console: Console, app_name: str, app_path: Path) -> Path:
    """Create a new Reachy Mini app project with the given name at the specified path.

    Args:
        console (Console): The console object for printing messages.
        app_name (str): The name of the app to create.
        app_path (Path): The directory where the app project will be created.

    Returns:
        Path: The path to the created app project.

    """
    app_name, language, app_path = create_cli(console, app_name, app_path)
    app_name = app_name.replace(
        "-", "_"
    )  # Force underscores in app name (belt and suspenders)

    TEMPLATE_DIR = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    def render_template(filename: str, context: Dict[str, str]) -> str:
        template = env.get_template(filename)
        return template.render(context)

    base_path = Path(app_path).resolve() / app_name
    if base_path.exists():
        console.print(f"❌ Folder {base_path} already exists.", style="bold red")
        exit()

    module_name = app_name
    entrypoint_name = app_name.replace("-", "_")
    class_name = "".join(word.capitalize() for word in module_name.split("_"))
    class_name_display = " ".join(word.capitalize() for word in module_name.split("_"))

    base_path.mkdir()
    (base_path / module_name).mkdir()
    (base_path / module_name / "static").mkdir()

    # Generate files
    context = {
        "app_name": app_name,
        "package_name": app_name,
        "module_name": module_name,
        "class_name": class_name,
        "class_name_display": class_name_display,
        "entrypoint_name": entrypoint_name,
    }

    (base_path / module_name / "__init__.py").touch()
    (base_path / module_name / "main.py").write_text(
        render_template("main.py.j2", context)
    )
    (base_path / module_name / "static" / "index.html").write_text(
        render_template("static/index.html.j2", context)
    )
    (base_path / module_name / "static" / "style.css").write_text(
        render_template("static/style.css.j2", context)
    )
    (base_path / module_name / "static" / "main.js").write_text(
        render_template("static/main.js.j2", context)
    )

    (base_path / "pyproject.toml").write_text(
        render_template("pyproject.toml.j2", context)
    )
    (base_path / "README.md").write_text(render_template("README.md.j2", context))

    (base_path / "index.html").write_text(render_template("index.html.j2", context))
    (base_path / "style.css").write_text(render_template("style.css.j2", context))
    (base_path / ".gitignore").write_text(render_template("gitignore.j2", context))

    # TODO assets dir with a .gif ?

    console.print(f"✅ Created app '{app_name}' in {base_path}/", style="bold green")
    return base_path


def install_app_with_progress(
    console: Console, pip_executable: str, app_path: Path
) -> None:
    """Install the app in a temporary virtual environment with a progress spinner."""
    console.print("Installing the app in the temporary virtual environment...")

    # Start pip in the background, discard its output
    process = subprocess.Popen(
        [
            pip_executable,
            "install",
            "-q",  # quiet
            str(app_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Installing dependencies...", start=True)

        # Keep the spinner running while pip is working
        while process.poll() is None:
            time.sleep(0.1)

        # Mark task as finished
        progress.update(task_id, description="Installation finished")

    # Handle exit code like check=True would
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def check(console: Console, app_path: str) -> None:
    """Check an existing Reachy Mini app project.

    Args:
        console (Console): The console object for printing messages.
        app_path (str): Local path to the app to check.

    """
    if app_path is None:
        console.print("\n$ What is the local path to the app you want to check?")
        app_path = questionary.path(
            ">",
            default="",
            validate=validate_location,
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_path = Path(app_path).expanduser().resolve()

    if not os.path.exists(app_path):
        console.print(f"[red]App path {app_path} does not exist.[/red]")
        exit()

    abs_app_path = Path(app_path).resolve()

    # Check if there is a pyproject.toml file in the root of the app
    pyproject_file = abs_app_path / "pyproject.toml"
    if not pyproject_file.exists():
        console.print("❌ pyproject.toml is missing", style="bold red")
        console.print(
            "Make sure you are providing the path to the root of the app. This is the folder that contains pyproject.toml.",
            style="bold blue",
        )
        sys.exit(1)

    # Extract app name
    with open(pyproject_file, "r", encoding="utf-8") as f:
        pyproject_content = toml.load(f)
        project = pyproject_content.get("project", {})
        app_name = project.get("name", None)
        if app_name is None:
            console.print(
                "❌ Project name is missing in pyproject.toml", style="bold red"
            )
            sys.exit(1)

    entrypoint_name = app_name

    # Extract pkg_name from entry point (allows package name != project name)
    entry_points = (
        pyproject_content.get("project", {})
        .get("entry-points", {})
        .get("reachy_mini_apps", {})
    )
    app_name_normalized = app_name.replace("-", "_")
    ep_value = entry_points.get(app_name) or entry_points.get(app_name_normalized)
    if ep_value and ":" in ep_value:
        pkg_name = ep_value.split(":")[0].split(".")[0]
    else:
        pkg_name = app_name.replace("-", "_")  # fallback to old behavior

    class_name = "".join(word.capitalize() for word in pkg_name.split("_"))

    console.print(f"\tExpected package name: {pkg_name}", style="dim")
    console.print(f"\tExpected class name: {class_name}", style="dim")
    console.print(f"\tExpected entrypoint name: {entrypoint_name}\n", style="dim")

    # Check that:
    # - index.html, style.css exist in the root of the app

    if not os.path.exists(os.path.join(abs_app_path, "index.html")):
        console.print("❌ index.html is missing", style="bold red")
        sys.exit(1)

    if not os.path.exists(os.path.join(abs_app_path, "style.css")):
        console.print("❌ style.css is missing", style="bold red")
        sys.exit(1)
    console.print("✅ index.html and style.css exist in the root of the app.")

    # Check both flat and src layout
    pkg_path = abs_app_path / pkg_name
    if not (pkg_path / "__init__.py").exists():
        pkg_path = abs_app_path / "src" / pkg_name
        if not (pkg_path / "__init__.py").exists():
            console.print(f"❌ Package folder '{pkg_name}' not found", style="bold red")
            console.print(f"   Checked: {abs_app_path / pkg_name}/", style="dim")
            console.print(f"   Checked: {abs_app_path / 'src' / pkg_name}/", style="dim")
            sys.exit(1)
    console.print(f"✅ Package '{pkg_name}' found at {pkg_path.relative_to(abs_app_path)}/")

    if "entry-points" not in pyproject_content["project"]:
        console.print(
            "❌ pyproject.toml is missing the entry-points section",
            style="bold red",
        )
        sys.exit(1)

    entry_points = pyproject_content["project"]["entry-points"]

    if "reachy_mini_apps" not in entry_points:
        console.print(
            "❌ pyproject.toml is missing the reachy_mini_apps entry-points section",
            style="bold red",
        )
        sys.exit(1)

    ep = entry_points["reachy_mini_apps"]
    for k, v in ep.items():
        console.print(f'Found entrypoint: {k} = "{v}"', style="dim")
        if k == entrypoint_name and v == f"{pkg_name}.main:{class_name}":
            console.print(
                "✅ pyproject.toml contains the correct entrypoint for the app."
            )
            break
    else:
        console.print(
            f'❌ pyproject.toml is missing the entrypoint for the app: {entrypoint_name} = "{pkg_name}.main:{class_name}"',
            style="bold red",
        )
        sys.exit(1)


    # - main.py exists
    main_file = pkg_path / "main.py"
    if not main_file.exists():
        console.print("❌ main.py is missing", style="bold red")
        sys.exit(1)
    console.print(f"✅ {pkg_path.relative_to(abs_app_path)}/main.py exists.")

    # - <app_name>/main.py contains a class named <AppName> that inherits from ReachyMiniApp
    with open(main_file, "r") as f:
        main_content = f.read()
    if f"class {class_name}(ReachyMiniApp)" not in str(main_content):
        console.print(
            f"❌ main.py is missing the class {class_name} that inherits from ReachyMiniApp",
            style="bold red",
        )
        sys.exit(1)
    console.print(
        f"✅ main.py contains the class {class_name} that inherits from ReachyMiniApp."
    )

    # - README.md exists in the root of the app
    if not os.path.exists(os.path.join(abs_app_path, "README.md")):
        console.print("❌ README.md is missing", style="bold red")
        sys.exit(1)
    console.print("✅ README.md exists in the root of the app.")

    def parse_readme(file_path: str) -> Any:
        """Parse the metadata section of the README.md file."""
        with open(file_path, "r") as f:
            lines = f.readlines()

        in_metadata = False
        metadata = ""
        for line in lines:
            line = line.strip()
            if line == "---":
                if not in_metadata:
                    in_metadata = True
                else:
                    break
            elif in_metadata:
                metadata += line + "\n"

        try:
            metadata = yaml.safe_load(metadata)
        except yaml.YAMLError as e:
            console.print(f"❌ Error parsing YAML metadata: {e}", style="bold red")
            sys.exit(1)

        return metadata

    #   - README.md contains at least a title and the tags "reachy_mini" and "reachy_mini_{python/js}_app"
    readme_metadata = parse_readme(os.path.join(abs_app_path, "README.md"))
    if readme_metadata is None or len(readme_metadata) == 0:
        console.print("❌ README.md is missing metadata section.", style="bold red")
        sys.exit(1)
    if "title" not in readme_metadata.keys():
        console.print(
            "❌ README.md is missing the title key in metadata.", style="bold red"
        )
        sys.exit(1)
    if readme_metadata["title"] == "":
        console.print("❌ README.md title cannot be empty.", style="bold red")
        sys.exit(1)

    if "tags" not in readme_metadata.keys():
        console.print(
            "❌ README.md is missing the tags key in metadata.", style="bold red"
        )
        sys.exit(1)

    if "reachy_mini" not in readme_metadata["tags"]:
        console.print(
            '❌ README.md must contain the "reachy_mini" tag', style="bold red"
        )
        sys.exit(1)

    if (
        "reachy_mini_python_app" not in readme_metadata["tags"]
        and "reachy_mini_js_app" not in readme_metadata["tags"]
    ):
        console.print(
            '❌ README.md must contain either the "reachy_mini_python_app" or "reachy_mini_js_app" tag',
            style="bold red",
        )
        sys.exit(1)

    console.print("✅ README.md contains the required metadata.")
    # - <app_name>/main.py exists

    # Now, create a temporary python venv in a temp dir, `pip install . the app, check that it works and that the entrypoint is registered
    with tempfile.TemporaryDirectory() as tmpdir:
        # change dir to tmpdir
        os.chdir(tmpdir)

        console.print(
            f"\nCreating a temporary virtual environment to test the app... (tmp dir: {tmpdir})"
        )
        venv_path = os.path.join(tmpdir, "venv")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

        pip_executable = os.path.join(
            venv_path,
            "Scripts" if os.name == "nt" else "bin",
            "pip",
        )
        python_executable = os.path.join(
            venv_path,
            "Scripts" if os.name == "nt" else "bin",
            "python",
        )

        install_app_with_progress(console, pip_executable, abs_app_path)

        console.print("Checking that the app entrypoint is registered...")

        # use from importlib.metadata import entry_points

        check_script = (
            f"from importlib.metadata import entry_points; "
            f"eps = entry_points(group='reachy_mini_apps'); "
            f"app_names = [ep.name for ep in eps]; "
            f"import sys; "
            f"sys.exit(0) if '{app_name}' in app_names else sys.exit(1)"
        )
        if (
            subprocess.run(
                [python_executable, "-c", check_script],
                # capture_output=True,
                text=True,
            ).returncode
            != 0
        ):
            console.print(
                f"❌ App '{app_name}' entrypoint is not registered correctly.",
                style="bold red",
            )
            sys.exit(1)
        console.print("✅ App entrypoint is registered correctly.")

        # Now try to uninstall the app and check that it uninstalls correctly
        console.print("Uninstalling the app from the temporary virtual environment...")
        subprocess.run(
            [pip_executable, "uninstall", "-y", app_name],
            check=True,
        )

        if (
            subprocess.run(
                [python_executable, "-c", check_script],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        ):
            console.print(
                f"❌ App '{app_name}' was not uninstalled correctly.",
                style="bold red",
            )
            sys.exit(1)

        console.print("✅ App installation and uninstallation tests passed.")

    console.print(f"\n✅ App '{app_name}' passed all checks!", style="bold green")


def request_app_addition(new_app_repo_id: str) -> bool:
    """Request to add the new app to the official Reachy Mini app store."""
    api = HfApi()

    repo_id = "pollen-robotics/reachy-mini-official-app-store"
    file_path = "app-list.json"

    # 0. Detect current HF user
    user = whoami()["name"]

    # 1. Check if there is already an open PR by this user for this app
    #    (we used commit_message=f"Add {new_app_repo_id} to app-list.json",
    #     which becomes the PR title)
    existing_prs = get_repo_discussions(
        repo_id=repo_id,
        repo_type="dataset",
        author=user,
        discussion_type="pull_request",
        discussion_status="open",
    )

    for pr in existing_prs:
        if new_app_repo_id in pr.title:
            print(
                f"An open PR already exists for {new_app_repo_id} by {user}: "
                f"https://huggingface.co/{repo_id}/discussions/{pr.num}"
            )
            return False

    # 2. Download current file from the dataset repo
    local_downloaded = api.hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
    )

    with open(local_downloaded, "r") as f:
        app_list = json.load(f)

    # 3. Modify JSON (append if not already present)
    if new_app_repo_id not in app_list:
        app_list.append(new_app_repo_id)
    else:
        print(f"{new_app_repo_id} is already in the app list.")
        # You might still want to continue and create the PR, or early-return here.
        return False

    # 4. Save updated JSON to a temporary path
    with tempfile.TemporaryDirectory() as tmpdir:
        updated_path = os.path.join(tmpdir, file_path)
        os.makedirs(os.path.dirname(updated_path), exist_ok=True)
        with open(updated_path, "w") as f:
            json.dump(app_list, f, indent=4)
            f.write("\n")

        # 5. Commit with create_pr=True
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {new_app_repo_id} to app-list.json",
            commit_description=(
                f"Append `{new_app_repo_id}` to the list of Reachy Mini apps."
            ),
            operations=[
                CommitOperationAdd(
                    path_in_repo=file_path,
                    path_or_fileobj=updated_path,
                )
            ],
            create_pr=True,
        )

    print("Commit URL:", commit_info.commit_url)
    print("PR URL:", commit_info.pr_url)  # None if no PR was opened
    return True


def try_to_push(console: Console, _app_path: Path) -> bool:
    """Try to push changes to the remote repository."""
    console.print("Pushing changes to the remote repository ...", style="bold blue")
    push_result = subprocess.run(
        f"cd {_app_path} && git push",
        shell=True,
        capture_output=True,
        text=True,
    )
    if push_result.returncode != 0:
        console.print(
            f"[red]Failed to push changes to the remote repository: {push_result.stderr}[/red]"
        )
        return False
    return True


def publish(
    console: Console,
    app_path: str,
    commit_message: str,
    official: bool = False,
    no_check: bool = False,
    private: bool | None = None,
) -> None:
    """Publish the app to the Reachy Mini app store.

    Args:
        console (Console): The console object for printing messages.
        app_path (str): Local path to the app to publish.
        commit_message (str): Commit message for the app publish.
        official (bool): Request to publish the app as an official Reachy Mini app.
        no_check (bool): Don't run checks before publishing the app.
        private (bool | None): If True, make private. If False, make public. If None, prompt.

    """
    import huggingface_hub as hf

    if app_path is None:
        console.print("\n$ What is the local path to the app you want to publish?")
        app_path = questionary.path(
            ">",
            default="",
            validate=validate_location,
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            exit()
        name_of_repo = Path(app_path).name
        if name_of_repo == "reachy_mini":
            console.print(
                "[red] Safeguard : You may have selected reachy_mini repo as your app. Aborted.[/red]"
            )
            exit()
    app_path = Path(app_path).expanduser().resolve()  # type: ignore
    if not os.path.exists(app_path):
        console.print(f"[red]App path {app_path} does not exist.[/red]")
        sys.exit()

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
            exit()

    username = hf.whoami()["name"]
    repo_path = f"{username}/{Path(app_path).name}"
    repo_url = f"https://huggingface.co/spaces/{repo_path}"

    if hf.repo_exists(repo_path, repo_type="space"):
        if official:
            # ask for confirmation
            if not questionary.confirm(
                "Are you sure you want to ask to publish this app as an official Reachy Mini app?"
            ).ask():
                console.print("[red]Aborted.[/red]")
                exit()

            worked = request_app_addition(repo_path)
            if worked:
                console.print(
                    "\nYou have requested to publish your app as an official Reachy Mini app."
                )
                console.print(
                    "The Pollen and Hugging Face teams will review your app. Thank you for your contribution!"
                )
            exit()

        console.print("App already exists on Hugging Face Spaces.", style="bold blue")
        os.system(f"cd {app_path} && git pull {repo_url} main")

        status_output = (
            subprocess.check_output(
                f"cd {app_path} && git status --porcelain", shell=True
            )
            .decode("utf-8")
            .strip()
        )

        if status_output == "":
            console.print(
                "✅ No changes to commit.",
                style="bold green",
            )
            push_anyway = questionary.confirm(
                "Do you want to try to push anyway?"
            ).ask()
            if not push_anyway:
                console.print("[red]Aborted.[/red]")
                exit()
            else:
                console.print("Trying to push anyway...")
                pushed = try_to_push(console, Path(app_path))
            exit()

        if no_check:
            console.print(
                "⚠️ Skipping checks as per --nocheck flag.",
                style="bold yellow",
            )
        else:
            console.print(f"\n🔎 Running checks on the app at {app_path}/...")
            check(console, str(app_path))

        if commit_message is None:
            commit_message = questionary.text(
                "\n$ Enter a commit message for the update:",
                default="Update app",
            ).ask()
            if commit_message is None:
                console.print("[red]Aborted.[/red]")
                exit()

        # commit local changes
        console.print("Committing changes locally ...", style="bold blue")
        os.system(f"cd {app_path} && git add . && git commit -m '{commit_message}'")

        # && git push HEAD:main"

        pushed = try_to_push(console, Path(app_path))
        if not pushed:
            exit()

        console.print("✅ App updated successfully.")
    else:
        if no_check:
            console.print(
                "⚠️ Skipping checks as per --nocheck flag.",
                style="bold yellow",
            )
        else:
            console.print(f"\n🔎 Running checks on the app at {app_path}/...")
            check(console, str(app_path))

        # Determine privacy setting
        if private is None:
            console.print("Do you want your space to be created private or public?")
            privacy = questionary.select(
                ">",
                choices=["private", "public"],
                default="public",
            ).ask()
            is_private = (privacy == "private")
        else:
            is_private = private

        hf.create_repo(
            repo_path,
            repo_type="space",
            private=is_private,
            exist_ok=False,
            space_sdk="static",
        )
        os.system(
            f"cd {app_path} && git init && git remote add space {repo_url} && git add . && git commit -m 'Initial commit' && git push --set-upstream -f space HEAD:main"
        )

        console.print("✅ App published successfully.", style="bold green")

        if official:
            # ask for confirmation
            if not questionary.confirm(
                "Are you sure you want to ask to publish this app as an official Reachy Mini app?"
            ).ask():
                console.print("[red]Aborted.[/red]")
                exit()

            worked = request_app_addition(repo_path)
            if worked:
                console.print(
                    "\nYou have requested to publish your app as an official Reachy Mini app."
                )
                console.print(
                    "The Pollen and Hugging Face teams will review your app. Thank you for your contribution!"
                )
