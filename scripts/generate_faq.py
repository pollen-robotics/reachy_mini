import json
import pathlib
import re
from typing import List, Dict, Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAQ_DATA_DIR = ROOT / "docs" / "faq"
FAQ_ANSWERS_DIR = FAQ_DATA_DIR / "answers"
FAQ_FILE = ROOT / "docs" / "source" / "faq.mdx"


def load_section(section_name: str) -> List[Dict[str, Any]]:
    json_path = FAQ_DATA_DIR / f"{section_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON for section '{section_name}' not found: {json_path}"
        )
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_answer_text(item: Dict[str, Any]) -> str:
    answer_file = item.get("answer_file")
    if not answer_file:
        raise KeyError(f"Missing 'answer_file' for question: {item.get('question')}")

    answer_path = FAQ_ANSWERS_DIR / answer_file
    if not answer_path.exists():
        raise FileNotFoundError(
            f"Answer file not found for '{item.get('question')}': {answer_path}"
        )

    with answer_path.open("r", encoding="utf-8") as f:
        return f.read().rstrip()  # just strip trailing newlines


def render_item(item: Dict[str, Any]) -> str:
    question = item["question"]
    tags = item.get("tags", [])
    answer = load_answer_text(item)
    source = item.get("source")

    # HTML for tags
    tags_html_parts = []
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


def render_section(section_name: str) -> str:
    items = load_section(section_name)
    rendered_items = [render_item(item) for item in items]
    # blank line between questions
    return "\n\n".join(rendered_items) + "\n"


def replace_section(content: str, section_name: str, new_block: str) -> str:
    """
    Replace the FAQ section markers:
    <!-- FAQ:section_name:start -->
    ...
    <!-- FAQ:section_name:end -->
    with the new content.
    """
    pattern = re.compile(
        rf"(<!-- FAQ:{re.escape(section_name)}:start -->)(.*?)(<!-- FAQ:{re.escape(section_name)}:end -->)",
        re.DOTALL,
    )
    replacement = rf"\1\n\n{new_block}\n\3"
    (content, n) = pattern.subn(replacement, content)
    if n == 0:
        raise ValueError(f"No block for section '{section_name}' found in {FAQ_FILE}")
    return content


def find_sections(content: str) -> List[str]:
    """
    Find all section names in markers:
    <!-- FAQ:section_name:start -->
    """
    pattern = re.compile(r"<!-- FAQ:([a-zA-Z0-9_-]+):start -->")
    return sorted(set(pattern.findall(content)))


def main() -> None:
    if not FAQ_FILE.exists():
        raise FileNotFoundError(f"FAQ file not found: {FAQ_FILE}")

    with FAQ_FILE.open("r", encoding="utf-8") as f:
        content = f.read()

    sections = find_sections(content)
    if not sections:
        raise RuntimeError(
            f"No FAQ section found in {FAQ_FILE}. "
            "Use markers like <!-- FAQ:section-name:start -->."
        )

    for section in sections:
        block = render_section(section)
        content = replace_section(content, section, block)

    FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FAQ_FILE.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
