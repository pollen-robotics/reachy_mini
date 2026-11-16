"""Tests for the generated conversation application."""

import io
import shutil
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

from reachy_mini.apps import fork_conversation

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
