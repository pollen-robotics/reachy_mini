"""Generate FAQ sections in documentation from JSON definitions."""

import json
import pathlib
import re
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAQ_DATA_DIR = ROOT / "docs" / "faq"
FAQ_ANSWERS_DIR = FAQ_DATA_DIR / "answers"
DOCS_SOURCE_DIR = ROOT / "docs" / "source"


def load_all_items() -> List[Dict[str, Any]]:
    """Load all FAQ items from JSON files."""
    all_items: List[Dict[str, Any]] = []
    for json_path in FAQ_DATA_DIR.glob("*.json"):
        section_name = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            try:
                items = json.load(f)
            except json.JSONDecodeError as e:
                msg = f"JSON error in {json_path}: {e}"
                raise RuntimeError(msg) from e
        for it in items:
            it.setdefault("_section", section_name)
        all_items.extend(items)
    return all_items


def load_answer_text(item: Dict[str, Any]) -> str:
    """Load the answer text for a FAQ item."""
    answer_file = item.get("answer_file")
    if not answer_file:
        msg = f"FAQ item missing 'answer_file' for question: {item.get('question')}"
        raise KeyError(msg)

    answer_path = FAQ_ANSWERS_DIR / answer_file
    if not answer_path.exists():
        msg = f"Answer file not found for '{item.get('question')}': {answer_path}"
        raise FileNotFoundError(msg)

    with answer_path.open("r", encoding="utf-8") as f:
        # Remove only trailing newline characters.
        return f.read().rstrip()


def render_item(item: Dict[str, Any]) -> str:
    """Render a FAQ item as an HTML details block."""
    question = item["question"]
    tags = item.get("tags", [])
    answer = load_answer_text(item)
    source = item.get("source")

    # Tags
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
{source_html}</details>"""

    return block


def item_has_any_tag(item: Dict[str, Any], wanted_tags: List[str]) -> bool:
    """Return True if the item has at least one of the wanted tags."""
    item_tags = [str(t).strip().lower() for t in item.get("tags", [])]
    wanted_tags_normalized = [t.strip().lower() for t in wanted_tags if t.strip()]
    return any(t in item_tags for t in wanted_tags_normalized)


def render_by_tags(tags_expr: str) -> str:
    """Render all items that have at least one of the requested tags.

    Tags expression examples: "SDK" or "SDK,ASSEMBLY".
    """
    tags = [t.strip() for t in tags_expr.split(",") if t.strip()]
    if not tags:
        msg = f"Empty tags expression: '{tags_expr}'"
        raise ValueError(msg)

    all_items = load_all_items()
    matching: List[Dict[str, Any]] = [
        it for it in all_items if item_has_any_tag(it, tags)
    ]

    # Dédoublonnage par (question, answer_file)
    seen: set[Tuple[str, str]] = set()
    unique_items: List[Dict[str, Any]] = []
    for it in matching:
        key = (it.get("question", ""), it.get("answer_file", ""))
        if key not in seen:
            seen.add(key)
            unique_items.append(it)

    if not unique_items:
        return f"> No questions found for tags: {', '.join(tags)}\n"

    rendered_items = [render_item(item) for item in unique_items]
    return "\n\n".join(rendered_items) + "\n"


def find_tags_placeholders(content: str) -> List[str]:
    """Return tag expressions as they appear in the content.

    Examples: "SDK", "SDK,ASSEMBLY".
    """
    pattern = re.compile(r"<!-- FAQ-TAGS:([^:]+):start -->")
    matches = pattern.findall(content)
    # Keep the expression as is (including spaces) for searching
    # but trim it for the unique list.
    return sorted({m.strip() for m in matches})


def replace_tag_block(content: str, tags_expr: str, new_block: str) -> str:
    """Replace the FAQ block that matches the given tags expression.

    The block is delimited by:

    <!-- FAQ-TAGS:tags_expr:start -->
    ...
    <!-- FAQ-TAGS:tags_expr:end -->
    """
    # Use the expression as written in the file (including spaces)
    # to match exactly.
    escaped_expr = re.escape(tags_expr)
    pattern = re.compile(
        (
            rf"(<!-- FAQ-TAGS:{escaped_expr}:start -->)"
            r"(.*?)"
            rf"(<!-- FAQ-TAGS:{escaped_expr}:end -->)"
        ),
        re.DOTALL,
    )
    replacement = r"\1\n\n" + new_block + r"\n\3"
    content, n = pattern.subn(replacement, content)
    if n == 0:
        msg = f"No FAQ-TAGS block found for '{tags_expr}' in the file."
        raise ValueError(msg)
    return content


def process_file(path: pathlib.Path) -> bool:
    """Process a Markdown file and update its FAQ blocks.

    Returns True if the content was modified.
    """
    original = path.read_text(encoding="utf-8")
    content = original

    # Search for placeholders with a broad regex to retrieve
    # exactly the tag expression as it appears.
    raw_matches = re.findall(r"<!-- FAQ-TAGS:([^:]+):start -->", content)
    # Keep the EXACT expressions (without strip) for replacements,
    # but deduplicate them gently.
    seen_exprs: List[str] = []
    for raw in raw_matches:
        if raw not in seen_exprs:
            seen_exprs.append(raw)

    if not seen_exprs:
        # Nothing to do in this file.
        return False

    for raw_expr in seen_exprs:
        expr_for_render = raw_expr.strip()
        block = render_by_tags(expr_for_render)
        content = replace_tag_block(content, raw_expr, block)

    if content != original:
        path.write_text(content, encoding="utf-8")
        return True
    return False


def main() -> None:
    """Run the FAQ tags injection script."""
    if not DOCS_SOURCE_DIR.exists():
        msg = f"Directory docs/source not found: {DOCS_SOURCE_DIR}"
        raise FileNotFoundError(msg)

    any_changed = False
    for path in DOCS_SOURCE_DIR.rglob("*"):
        if path.suffix.lower() in {".md", ".mdx"} and path.is_file():
            changed = process_file(path)
            if changed:
                print(f"[inject_faq_tags] Updated: {path.relative_to(ROOT)}")
                any_changed = True

    if not any_changed:
        print("[inject_faq_tags] No files modified.")


if __name__ == "__main__":
    main()
