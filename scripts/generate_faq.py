"""Generate FAQ file content from JSON sections."""

import json
import pathlib
import re
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAQ_DATA_DIR = ROOT / "docs" / "faq"
FAQ_FILE = ROOT / "docs" / "source" / "faq.mdx"


def load_section(folder_name: str, section_name: str) -> List[Dict[str, Any]]:
    """Load a FAQ section from JSON files."""
    base_dir = FAQ_DATA_DIR / folder_name
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
        # You can change this to pick the first if you prefer implicit behavior.
        msg = (
            f"Multiple JSON files named '{target_name}' found under "
            f"'{base_dir}':\n" + "\n".join(f"- {m}" for m in matches)
        )
        raise RuntimeError(msg)

    json_path = matches[0]
    with json_path.open("r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    answers_dir = json_path.parent / "answers"
    for it in items:
        # Internal metadata: where to look for answers for this item.
        it["_answers_dir"] = answers_dir

    return items


def load_answer_text(item: Dict[str, Any]) -> str:
    """Load the answer text for a FAQ item."""
    answer_file = item.get("answer_file")
    if not answer_file:
        msg = f"Missing 'answer_file' for question: {item.get('question')}"
        raise KeyError(msg)

    answers_dir = item.get("_answers_dir")
    if not answers_dir:
        msg = (
            "Missing internal '_answers_dir' for item with question: "
            f"{item.get('question')}"
        )
        raise KeyError(msg)

    answer_path = pathlib.Path(answers_dir) / answer_file
    if not answer_path.exists():
        msg = f"Answer file not found for '{item.get('question')}': {answer_path}"
        raise FileNotFoundError(msg)

    with answer_path.open("r", encoding="utf-8") as f:
        # Strip only trailing newlines.
        return f.read().rstrip()


def render_item(item: Dict[str, Any]) -> str:
    """Render a FAQ item as an HTML details block."""
    question = item["question"]
    tags = item.get("tags", [])
    answer = load_answer_text(item)
    source = item.get("source")

    # HTML for tags.
    tags_html_parts: List[str] = []
    for tag in tags:
        tags_html_parts.append(
            f"""
  <span
    style="
      display: inline-block;
      padding: 2px 10px;
      margin: 2px 4px;
      background: rgba(59, 176, 209, 0.1);
      color: var(--primary);
      border-radius: 12px;
      font-size: 11px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    "
  >
    {tag}
  </span>"""
        )
    tags_html = "".join(tags_html_parts)

    source_html = ""
    if source:
        source_html = f'\n<p style="color:grey"><i>Source: {source}.</i></p>\n'

    block = f"""<details>
<summary><b>{question}</b><br>
<div
  style="
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-xs);
  "
>
  Tags:
  {tags_html}
</div></summary>

{answer}
{source_html}</details><br>"""

    return block


def render_section(folder_name: str, section_name: str) -> str:
    """Render all FAQ items for a section as HTML."""
    items = load_section(folder_name, section_name)
    rendered_items = [render_item(item) for item in items]
    # Blank line between questions.
    return "\n\n".join(rendered_items) + "\n"


def replace_section(
    content: str,
    folder_name: str,
    section_name: str,
    new_block: str,
) -> str:
    """Replace a FAQ section block in the content.

    The block is delimited by:

    <!-- FAQ:folder_name:section_name:start -->
    ...
    <!-- FAQ:folder_name:section_name:end -->
    """
    pattern = re.compile(
        (
            rf"(<!-- FAQ:{re.escape(folder_name)}:{re.escape(section_name)}:start -->)"
            r"(.*?)"
            rf"(<!-- FAQ:{re.escape(folder_name)}:{re.escape(section_name)}:end -->)"
        ),
        re.DOTALL,
    )
    replacement = rf"\1\n\n{new_block}\n\3"
    content, n = pattern.subn(replacement, content)
    if n == 0:
        msg = f"No block for section '{folder_name}:{section_name}' found in {FAQ_FILE}"
        raise ValueError(msg)
    return content


def find_sections(content: str) -> List[Tuple[str, str]]:
    """Find all FAQ section markers in the content.

    Markers have the form: <!-- FAQ:folder_name:section-name:start -->.
    """
    pattern = re.compile(r"<!-- FAQ:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->")
    # Return a sorted list of (folder_name, section_name) pairs.
    return sorted(set(pattern.findall(content)))


def main() -> None:
    """Run the FAQ section rendering script."""
    if not FAQ_FILE.exists():
        msg = f"FAQ file not found: {FAQ_FILE}"
        raise FileNotFoundError(msg)

    with FAQ_FILE.open("r", encoding="utf-8") as f:
        content = f.read()

    sections = find_sections(content)
    if not sections:
        msg = (
            f"No FAQ section found in {FAQ_FILE}. "
            "Use markers like <!-- FAQ:folder_name:section-name:start -->."
        )
        raise RuntimeError(msg)

    for folder_name, section_name in sections:
        block = render_section(folder_name, section_name)
        content = replace_section(content, folder_name, section_name, block)

    FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FAQ_FILE.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
