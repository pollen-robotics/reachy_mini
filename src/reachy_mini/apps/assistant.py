"""Reachy Mini app assistant functions."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import questionary
import yaml
from huggingface_hub import CommitOperationAdd, HfApi, get_repo_discussions, whoami
from jinja2 import Environment, FileSystemLoader
from rich.console import Console


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


def create_cli(
    console: Console, app_name: str | None, app_path: Path | None
) -> tuple[str, str, Path]:
    """Create a new Reachy Mini app project using a CLI."""

    def validate_app_name(text: str) -> bool | str:
        if not text.strip():
            return "App name cannot be empty."
        if " " in text:
            return "App name cannot contain spaces."
        if "-" in text:
            return "App name cannot contain dashes ('-'). Please use underscores ('_') instead."
        return True

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

    # 2) Language
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

    if app_path is None:
        # 3) App path
        console.print("\n$ Where do you want to create your app project ?")
        app_path = questionary.path(
            ">",
            default="",
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


def create(console: Console, app_name: str, app_path: Path) -> None:
    """Create a new Reachy Mini app project with the given name at the specified path.

    Args:
        console (Console): The console object for printing messages.
        app_name (str): The name of the app to create.
        app_path (Path): The directory where the app project will be created.

    """
    app_name, language, app_path = create_cli(console, app_name, app_path)
    TEMPLATE_DIR = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    def render_template(filename: str, context: Dict[str, str]) -> str:
        template = env.get_template(filename)
        return template.render(context)

    base_path = Path(app_path).resolve() / app_name
    if base_path.exists():
        console.print(f"❌ Folder {base_path} already exists.", style="bold red")
        exit()

    module_name = app_name.replace("-", "_")
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
        ).ask()
        if app_path is None:
            console.print("[red]Aborted.[/red]")
            exit()
        app_path = Path(app_path).expanduser().resolve()

    if not os.path.exists(app_path):
        console.print(f"[red]App path {app_path} does not exist.[/red]")
        exit()

    app_path = str(app_path).rstrip("/")
    app_name = os.path.basename(app_path)

    print(f"Checking {app_name} at path '{app_path}'")

    # Check that:
    # - index.html, style.css exist in the root of the app

    if not os.path.exists(os.path.join(app_path, "index.html")):
        console.print("❌ index.html is missing", style="bold red")
        sys.exit(1)

    if not os.path.exists(os.path.join(app_path, "style.css")):
        console.print("❌ style.css is missing", style="bold red")
        sys.exit(1)
    console.print("✅ index.html and style.css exist in the root of the app.")
    # - pyproject.toml exists in the root of the app
    if not os.path.exists(os.path.join(app_path, "pyproject.toml")):
        console.print("❌ pyproject.toml is missing", style="bold red")
        sys.exit(1)
    console.print("✅ pyproject.toml exists in the root of the app.")
    #   - pyproject.toml contains the entrypoint
    # [project.entry-points."reachy_mini_apps"]
    # test = "<app_name>.main:<AppName>"
    with open(os.path.join(app_path, "pyproject.toml"), "r") as f:
        pyproject_content = f.read()

    if not '[project.entry-points."reachy_mini_apps"]' in pyproject_content:
        console.print(
            '❌ pyproject.toml is missing the [project.entry-points."reachy_mini_apps"] section',
            style="bold red",
        )
        sys.exit(1)

    if (
        f'{app_name} = "{app_name}.main:{"".join(word.capitalize() for word in app_name.replace("-", "_").split("_"))}"'
        not in pyproject_content
    ):
        console.print(
            f'❌ pyproject.toml is missing the entrypoint for the app: {app_name} = "{app_name}.main:{"".join(word.capitalize() for word in app_name.replace("-", "_").split("_"))}"',
            style="bold red",
        )
        sys.exit(1)

    console.print("✅ pyproject.toml contains the entrypoint section.")
    # - <app_name>/__init__.py exists
    app_name = os.path.basename(app_path)

    if not os.path.exists(os.path.join(app_path, app_name, "__init__.py")):
        console.print("❌ __init__.py is missing", style="bold red")
        sys.exit(1)

    console.print(f"✅ {app_name}/__init__.py exists.")

    # - README.md exists in the root of the app
    if not os.path.exists(os.path.join(app_path, "README.md")):
        console.print("❌ README.md is missing", style="bold red")
        sys.exit(1)
    console.print("✅ README.md exists in the root of the app.")

    def parse_readme(file_path):
        #     ---
        #     title: Test
        #     emoji: 👋
        #     colorFrom: red
        #     colorTo: blue
        #     sdk: static
        #     pinned: false
        #     short_description: Write your description here
        #     tags:
        #     - reachy_mini
        #     - reachy_mini_python_app
        #     other_stuff : abc
        #     ---

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
    readme_metadata = parse_readme(os.path.join(app_path, "README.md"))
    if len(readme_metadata) == 0:
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

    if not os.path.exists(os.path.join(app_path, app_name, "main.py")):
        console.print("❌ main.py is missing", style="bold red")
        sys.exit(1)
    console.print(f"✅ {app_name}/main.py exists.")

    # - <app_name>/main.py contains a class named <AppName> that inherits from ReachyMiniApp
    with open(os.path.join(app_path, app_name, "main.py"), "r") as f:
        main_content = f.read()
    class_name = "".join(word.capitalize() for word in app_name.replace("-", "_").split("_"))
    if f"class {class_name}(ReachyMiniApp)" not in main_content:
        console.print(
            f"❌ main.py is missing the class {class_name} that inherits from ReachyMiniApp",
            style="bold red",
        )
        sys.exit(1) 
    console.print(f"✅ main.py contains the class {class_name} that inherits from ReachyMiniApp.")

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


def publish(
    console: Console, app_path: str, commit_message: str, official: bool = False
) -> None:
    """Publish the app to the Reachy Mini app store.

    Args:
        console (Console): The console object for printing messages.
        app_path (str): Local path to the app to publish.
        commit_message (str): Commit message for the app publish.
        official (bool): Request to publish the app as an official Reachy Mini app.

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
            exit()
        name_of_repo = Path(app_path).name
        if name_of_repo == "reachy_mini":
            console.print(
                "[red] Safeguard : You may have selected reachy_mini repo as your app. Aborted.[/red]"
            )
            exit()
        app_path = Path(app_path).expanduser().resolve()
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
        os.system(f"cd {app_path} && git pull {repo_url} main")
        console.print("App already exists on Hugging Face Spaces. Updating...")
        commit_message = questionary.text(
            "\n$ Enter a commit message for the update:",
            default="Update app",
        ).ask()
        if commit_message is None:
            console.print("[red]Aborted.[/red]")
            exit()
        os.system(
            f"cd {app_path} && git add . && git commit -m '{commit_message}' && git push HEAD:main"
        )
        console.print("✅ App updated successfully.")
    else:
        console.print("Do you want your space to be created private or public?")
        privacy = questionary.select(
            ">",
            choices=["private", "public"],
            default="public",
        ).ask()

        hf.create_repo(
            repo_path,
            repo_type="space",
            private=(privacy == "private"),
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
