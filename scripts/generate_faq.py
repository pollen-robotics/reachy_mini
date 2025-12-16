"""Generate FAQ file content from JSON sections."""

from typing import List, Tuple

from qa_utils import (
    FAQ_CONFIG,
    ROOT,
    find_sections,
    load_json_items,
    render_faq_item,
    replace_section_block,
)

FAQ_FILE = ROOT / "docs" / "source" / "troubleshooting_faq.md"


def load_section(folder_name: str, section_name: str) -> list[dict]:
    """Load a FAQ section from JSON files."""
    base_dir = FAQ_CONFIG.data_dir / folder_name
    if not base_dir.exists():
        msg = f"Folder '{folder_name}' not found under FAQ_DATA_DIR: {base_dir}"
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
        answers_dir = FAQ_CONFIG.default_answers_dir

    for it in items:
        it["_answers_dir"] = answers_dir

    return items


def render_section(folder_name: str, section_name: str) -> str:
    """Render all FAQ items for a section as HTML."""
    items = load_section(folder_name, section_name)
    rendered_items = [render_faq_item(item, FAQ_CONFIG) for item in items]
    return "\n\n".join(rendered_items) + "\n"


def main() -> None:
    """Run the FAQ section rendering script."""
    if not FAQ_FILE.exists():
        msg = f"FAQ file not found: {FAQ_FILE}"
        raise FileNotFoundError(msg)

    content = FAQ_FILE.read_text(encoding="utf-8")

    sections: List[Tuple[str, str]] = find_sections(
        content,
        FAQ_CONFIG.section_block_label,
    )
    if not sections:
        msg = (
            f"No FAQ section found in {FAQ_FILE}. "
            "Use markers like <!-- FAQ:folder_name:section-name:start -->."
        )
        raise RuntimeError(msg)

    for folder_name, section_name in sections:
        block = render_section(folder_name, section_name)
        content = replace_section_block(
            content,
            FAQ_CONFIG.section_block_label,
            folder_name,
            section_name,
            block,
        )

    FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAQ_FILE.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
