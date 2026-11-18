"""Tests for the generated conversation application."""

import io
import shutil
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

from reachy_mini.apps import assistant, fork_conversation
from reachy_mini.apps.assistant import validate_app_name

# Shaped like the conversation app's pyproject.toml: every name the fork renames.
_CONVERSATION_PYPROJECT = """\
[project]
name = "reachy_mini_conversation_app"
version = "0.0.0"

[project.scripts]
reachy-mini-conversation-app = "reachy_mini_conversation_app.main:main"

# Lets the daemon find the app.
[project.entry-points."reachy_mini_apps"]
reachy_mini_conversation_app = "reachy_mini_conversation_app.main:ReachyMiniConversationApp"

[tool.setuptools.package-data]
reachy_mini_conversation_app = ["static/*"]

[tool.ruff.lint.isort]
known-local-folder = ["reachy_mini_conversation_app"]
"""


def test_conversation_template_uses_current_profile_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated app should use one root profile document and shared tools."""
    checkout_path = tmp_path / "conversation_app"
    package_path = checkout_path / "src" / "reachy_mini_conversation_app"
    (package_path / "static").mkdir(parents=True)
    (package_path / "tools").mkdir()
    (package_path / "config.py").write_text("LOCKED_PROFILE: str | None = None\n")
    (package_path / "tools" / "sweep_look.py").write_text(
        "class SweepLook:\n    pass\n"
    )
    default_profile_path = checkout_path / "profiles" / "default"
    default_profile_path.mkdir(parents=True)
    (default_profile_path / "profile.md").write_text(
        "+++\nschema_version = 1\ndefault_tools = []\n+++\n\nDefault profile.\n"
    )
    (checkout_path / "README.md").write_text("Original README\n")
    (checkout_path / "pyproject.toml").write_text(_CONVERSATION_PYPROJECT)

    def copy_checkout(_console: Console, target_path: Path) -> None:
        shutil.copytree(checkout_path, target_path)

    monkeypatch.setattr(fork_conversation, "_clone_repo", copy_checkout)
    monkeypatch.setattr(fork_conversation, "_init_git", Mock())
    output = io.StringIO()

    generated_path = fork_conversation.create_from_conversation_app(
        Console(file=output, color_system=None, width=200),
        "my_conversation",
        tmp_path,
    )

    profile_name = "_my_conversation_locked_profile"
    profile_path = generated_path / "profiles" / profile_name / "profile.md"
    profile_content = profile_path.read_text()
    metadata = tomllib.loads(profile_content.split("+++", maxsplit=2)[1])

    assert metadata["schema_version"] == 1
    assert metadata["default_tools"] == [
        "dance",
        "stop_dance",
        "play_emotion",
        "stop_emotion",
        "sweep_look",
    ]
    assert not (generated_path / "profiles" / "default").exists()
    assert not list(generated_path.rglob("instructions.txt"))
    assert not list(generated_path.rglob("tools.txt"))
    assert (
        generated_path / "src" / "my_conversation" / "tools" / "sweep_look.py"
    ).is_file()
    assert (
        f"profiles/{profile_name}/profile.md"
        in (generated_path / "README.md").read_text()
    )
    assert f"profiles/{profile_name}/profile.md" in output.getvalue()
    assert "python src/my_conversation/main.py --ui" in output.getvalue()

    pyproject_text = (generated_path / "pyproject.toml").read_text()
    pyproject = tomllib.loads(pyproject_text)
    assert pyproject["project"]["name"] == "my_conversation"
    assert pyproject["project"]["version"] == "0.0.0"
    assert pyproject["project"]["scripts"] == {
        "my-conversation": "my_conversation.main:main"
    }
    assert pyproject["project"]["entry-points"] == {
        "reachy_mini_apps": {"my_conversation": "my_conversation.main:MyConversation"}
    }
    assert pyproject["tool"]["setuptools"]["package-data"] == {
        "my_conversation": ["static/*"]
    }
    assert pyproject["tool"]["ruff"]["lint"]["isort"]["known-local-folder"] == [
        "my_conversation"
    ]
    assert "# Lets the daemon find the app." in pyproject_text
    assert "conversation_app" not in pyproject_text
    assert "conversation-app" not in pyproject_text


def test_update_pyproject_stops_when_the_rename_misses(tmp_path: Path) -> None:
    """An unexpected pyproject.toml is left untouched and the fork stops."""
    original = '[project]\nname = "reachy_mini_conversation_app"\nversion = "0.0.0"\n'
    (tmp_path / "pyproject.toml").write_text(original)

    with pytest.raises(SystemExit):
        fork_conversation._update_pyproject(
            Console(file=io.StringIO()), tmp_path, "my_app"
        )

    assert (tmp_path / "pyproject.toml").read_text() == original


def test_update_pyproject_reports_what_the_rename_missed(tmp_path: Path) -> None:
    """The failure message names the field that did not come out as expected."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "reachy_mini_conversation_app"\nversion = "0.0.0"\n'
    )
    output = io.StringIO()

    with pytest.raises(SystemExit):
        fork_conversation._update_pyproject(
            Console(file=output, color_system=None, width=200), tmp_path, "my_app"
        )

    assert "scripts.my-app: expected 'my_app.main:main', got None" in output.getvalue()


def test_update_pyproject_only_renames_whole_names(tmp_path: Path) -> None:
    """URLs, descriptions and dependencies that contain the name are kept."""
    kept_lines = [
        'description = "Talk with reachy_mini_conversation_app"',
        'dependencies = ["reachy_mini_conversation_app_tools>=1"]',
        'Homepage = "https://github.com/pollen-robotics/reachy_mini_conversation_app"',
        'Space = "https://huggingface.co/spaces/pollen/reachy-mini-conversation-app"',
        'Base = "reachy_mini_conversation_app.main:ReachyMiniConversationAppBase"',
    ]
    original = _CONVERSATION_PYPROJECT.replace(
        'version = "0.0.0"\n',
        'version = "0.0.0"\n' + "\n".join(kept_lines[:2]) + "\n",
    ) + ("\n[project.urls]\n" + "\n".join(kept_lines[2:4]) + "\n")
    original = original.replace(
        '[tool.setuptools.package-data]\n',
        "[tool.other]\n" + kept_lines[4] + "\n\n[tool.setuptools.package-data]\n",
    )
    (tmp_path / "pyproject.toml").write_text(original)

    fork_conversation._update_pyproject(
        Console(file=io.StringIO()), tmp_path, "my_app"
    )

    text = (tmp_path / "pyproject.toml").read_text()
    for line in kept_lines[:4]:
        assert line in text
    # The module part of a reference is renamed, a longer class name is not.
    assert 'Base = "my_app.main:ReachyMiniConversationAppBase"' in text
    pyproject = tomllib.loads(text)
    assert pyproject["project"]["name"] == "my_app"
    assert pyproject["project"]["scripts"] == {"my-app": "my_app.main:main"}
    assert pyproject["project"]["entry-points"] == {
        "reachy_mini_apps": {"my_app": "my_app.main:MyApp"}
    }
    assert pyproject["tool"]["setuptools"]["package-data"] == {"my_app": ["static/*"]}
    assert pyproject["tool"]["ruff"]["lint"]["isort"]["known-local-folder"] == [
        "my_app"
    ]


@pytest.mark.parametrize("name", ["café_bot", "1app", 'my"app', "my+app", "my\app"])
def test_validate_app_name_rejects_non_identifiers(name: str) -> None:
    """Names that cannot be a package name or a bare TOML key are refused."""
    assert validate_app_name(name) is not True


@pytest.mark.parametrize("name", ["my_app", "My_App", "_app", "app2"])
def test_validate_app_name_accepts_identifiers(name: str) -> None:
    """ASCII identifiers are valid app names."""
    assert validate_app_name(name) is True


def test_fork_rejects_invalid_command_line_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name given on the command line is validated before anything is cloned."""
    clone = Mock()
    monkeypatch.setattr(fork_conversation, "_clone_repo", clone)

    with pytest.raises(SystemExit):
        fork_conversation.create_from_conversation_app(
            Console(file=io.StringIO()), "café-bot", tmp_path
        )

    clone.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_create_rejects_invalid_command_line_name(tmp_path: Path) -> None:
    """The blank-app template validates command-line names too."""
    with pytest.raises(SystemExit):
        assistant.create(Console(file=io.StringIO()), "café-bot", tmp_path)

    assert not list(tmp_path.iterdir())


def test_failed_fork_removes_the_partial_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork that stops half-way does not leave a folder blocking the next try."""
    checkout_path = tmp_path / "checkout"
    (checkout_path / "src" / "reachy_mini_conversation_app").mkdir(parents=True)
    (checkout_path / "pyproject.toml").write_text(
        '[project]\nname = "something_else"\n'
    )

    def copy_checkout(_console: Console, target_path: Path) -> None:
        shutil.copytree(checkout_path, target_path)

    monkeypatch.setattr(fork_conversation, "_clone_repo", copy_checkout)
    apps_path = tmp_path / "apps"
    apps_path.mkdir()

    with pytest.raises(SystemExit):
        fork_conversation.create_from_conversation_app(
            Console(file=io.StringIO()), "my_app", apps_path
        )

    assert not (apps_path / "my_app").exists()
