"""Provide shared utilities for FAQ generation scripts."""

import json
import pathlib
from typing import Any, Dict, Iterable, List

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAQ_DATA_DIR = ROOT / "docs" / "faq"
FAQ_ANSWERS_DIR = FAQ_DATA_DIR / "answers"
DOCS_SOURCE_DIR = ROOT / "docs" / "source"


def iter_faq_json_files() -> Iterable[pathlib.Path]:
    """Iterate over all FAQ JSON files under docs/faq recursively."""
    return FAQ_DATA_DIR.rglob("*.json")


def load_json_items(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load FAQ items from a JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        try:
            items: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"JSON error in {json_path}: {e}"
            raise RuntimeError(msg) from e
    return items


def load_all_items() -> List[Dict[str, Any]]:
    """Load all FAQ items from all JSON files under docs/faq."""
    all_items: List[Dict[str, Any]] = []
    for json_path in iter_faq_json_files():
        section_name = json_path.stem
        items = load_json_items(json_path)

        # Prefer a local "answers" directory next to the JSON file.
        answers_dir_candidate = json_path.parent / "answers"
        if answers_dir_candidate.exists():
            answers_dir = answers_dir_candidate
        else:
            answers_dir = FAQ_ANSWERS_DIR

        for it in items:
            it.setdefault("_section", section_name)
            it.setdefault("_answers_dir", answers_dir)

        all_items.extend(items)

    return all_items


def load_answer_text(item: Dict[str, Any]) -> str:
    """Load the answer text for a FAQ item."""
    answer_file = item.get("answer_file")
    if not answer_file:
        msg = f"Missing 'answer_file' for question: {item.get('question')}"
        raise KeyError(msg)

    answers_dir = item.get("_answers_dir") or FAQ_ANSWERS_DIR
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


def item_has_any_tag(item: Dict[str, Any], wanted_tags: List[str]) -> bool:
    """Return True if the item has at least one of the wanted tags."""
    item_tags = [str(t).strip().lower() for t in item.get("tags", [])]
    wanted_tags_normalized = [t.strip().lower() for t in wanted_tags if t.strip()]
    return any(t in item_tags for t in wanted_tags_normalized)


__all__ = [
    "ROOT",
    "FAQ_DATA_DIR",
    "FAQ_ANSWERS_DIR",
    "DOCS_SOURCE_DIR",
    "iter_faq_json_files",
    "load_json_items",
    "load_all_items",
    "load_answer_text",
    "render_item",
    "item_has_any_tag",
]
