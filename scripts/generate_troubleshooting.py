"""Generate troubleshooting file content from JSON sections."""

from typing import List, Tuple

from qa_utils import (
    ROOT,
    TROUBLESHOOTING_CONFIG,
    find_sections,
    load_json_items,
    render_troubleshooting_item,
    replace_section_block,
)

TROUBLESHOOTING_FILE = ROOT / "docs" / "source" / "troubleshooting.mdx"


def load_section(folder_name: str, section_name: str) -> list[dict]:
    """Load a troubleshooting section from JSON files."""
    base_dir = TROUBLESHOOTING_CONFIG.data_dir / folder_name
    if not base_dir.exists():
        msg = (
            "Folder "
            f"'{folder_name}' not found under TROUBLESHOOTING_DATA_DIR: {base_dir}"
        )
        raise FileNotFoundError(msg)

    target_name = f"{section_name}.json"
    matches = list(base_dir.rglob(target_name))

    if not matches:
        msg = (
            f"JSON for section '{section_name}' in folder '{folder_name}' "
            f"not found under: {base_dir}"
        )
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = (
            f"Multiple JSON files named '{target_name}' found under "
            f"'{base_dir}':\n" + "\n".join(f"- {m}" for m in matches)
        )
        raise RuntimeError(msg)

    json_path = matches[0]
    items = load_json_items(json_path)

    answers_dir_candidate = json_path.parent / "answers"
    if answers_dir_candidate.exists():
        answers_dir = answers_dir_candidate
    else:
        answers_dir = TROUBLESHOOTING_CONFIG.default_answers_dir

    for it in items:
        it["_answers_dir"] = answers_dir

    return items


def render_section(folder_name: str, section_name: str) -> str:
    """Render all troubleshooting items for a section as HTML."""
    items = load_section(folder_name, section_name)
    rendered_items = [
        render_troubleshooting_item(item, TROUBLESHOOTING_CONFIG) for item in items
    ]
    return "\n\n".join(rendered_items) + "\n"


def main() -> None:
    """Run the troubleshooting section rendering script."""
    if not TROUBLESHOOTING_FILE.exists():
        msg = f"Troubleshooting file not found: {TROUBLESHOOTING_FILE}"
        raise FileNotFoundError(msg)

    content = TROUBLESHOOTING_FILE.read_text(encoding="utf-8")

    sections: List[Tuple[str, str]] = find_sections(
        content,
        TROUBLESHOOTING_CONFIG.section_block_label,
    )
    if not sections:
        msg = (
            f"No troubleshooting section found in {TROUBLESHOOTING_FILE}. "
            "Use markers like "
            "<!-- TROUBLESHOOTING:folder_name:section-name:start -->."
        )
        raise RuntimeError(msg)

    for folder_name, section_name in sections:
        block = render_section(folder_name, section_name)
        content = replace_section_block(
            content,
            TROUBLESHOOTING_CONFIG.section_block_label,
            folder_name,
            section_name,
            block,
        )

    TROUBLESHOOTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TROUBLESHOOTING_FILE.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
