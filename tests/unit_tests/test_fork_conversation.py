"""Tests for the generated conversation application."""

import io
import shutil
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

from reachy_mini.apps import fork_conversation


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
    (checkout_path / "pyproject.toml").write_text(
        '[project]\nname = "reachy_mini_conversation_app"\nversion = "0.0.0"\n'
    )

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
