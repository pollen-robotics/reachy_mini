import json
import pathlib
import re
from typing import List, Dict, Any, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAQ_DATA_DIR = ROOT / "docs" / "faq"
FAQ_FILE = ROOT / "docs" / "source" / "faq.mdx"


def load_section(folder_name: str, section_name: str) -> List[Dict[str, Any]]:
    """
    Load a section from JSON by searching in docs/faq/<folder_name> and all its children
    for a file named <section_name>.json.

    Also attaches an internal "_answers_dir" to each item:
    json_path.parent / "answers"
    """
    base_dir = FAQ_DATA_DIR / folder_name
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Folder '{folder_name}' not found under FAQ_DATA_DIR: {base_dir}"
        )

    target_name = f"{section_name}.json"
    matches = list(base_dir.rglob(target_name))

    if not matches:
        raise FileNotFoundError(
            f"JSON for section '{section_name}' in folder '{folder_name}' "
            f"not found under: {base_dir}"
        )
    if len(matches) > 1:
        # You can change this to pick the first if you prefer implicit behavior
        raise RuntimeError(
            f"Multiple JSON files named '{target_name}' found under '{base_dir}':\n"
            + "\n".join(f"- {m}" for m in matches)
        )

    json_path = matches[0]
    with json_path.open("r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    answers_dir = json_path.parent / "answers"
    for it in items:
        # internal metadata: where to look for answers for this item
        it["_answers_dir"] = answers_dir

    return items


def load_answer_text(item: Dict[str, Any]) -> str:
    answer_file = item.get("answer_file")
    if not answer_file:
        raise KeyError(f"Missing 'answer_file' for question: {item.get('question')}")

    answers_dir = item.get("_answers_dir")
    if not answers_dir:
        raise KeyError(
            f"Missing internal '_answers_dir' for item with question: {item.get('question')}"
        )

    answer_path = pathlib.Path(answers_dir) / answer_file
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


def render_section(folder_name: str, section_name: str) -> str:
    items = load_section(folder_name, section_name)
    rendered_items = [render_item(item) for item in items]
    # blank line between questions
    return "\n\n".join(rendered_items) + "\n"


def replace_section(
    content: str, folder_name: str, section_name: str, new_block: str
) -> str:
    """
    Replace the FAQ section markers:

    <!-- FAQ:folder_name:section_name:start -->
    ...
    <!-- FAQ:folder_name:section_name:end -->

    with the new content.
    """
    pattern = re.compile(
        rf"(<!-- FAQ:{re.escape(folder_name)}:{re.escape(section_name)}:start -->)"
        r"(.*?)"
        rf"(<!-- FAQ:{re.escape(folder_name)}:{re.escape(section_name)}:end -->)",
        re.DOTALL,
    )
    replacement = rf"\1\n\n{new_block}\n\3"
    (content, n) = pattern.subn(replacement, content)
    if n == 0:
        raise ValueError(
            f"No block for section '{folder_name}:{section_name}' found in {FAQ_FILE}"
        )
    return content


def find_sections(content: str) -> List[Tuple[str, str]]:
    """
    Find all (folder_name, section_name) pairs in markers:

    <!-- FAQ:folder_name:section-name:start -->
    """
    pattern = re.compile(r"<!-- FAQ:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->")
    # returns list of (folder_name, section_name)
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
            "Use markers like <!-- FAQ:folder_name:section-name:start -->."
        )

    for folder_name, section_name in sections:
        block = render_section(folder_name, section_name)
        content = replace_section(content, folder_name, section_name, block)

    FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FAQ_FILE.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
