"""Generate FAQ sections in documentation from JSON definitions."""

import pathlib
import re
from typing import Any, Dict, List, Tuple

from faq_utils import DOCS_SOURCE_DIR, item_has_any_tag, load_all_items, render_item


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

    # Dédoublonnage par (question, answer_file).
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
                print(
                    f"[inject_faq_tags] Updated: {path.relative_to(DOCS_SOURCE_DIR.parent)}"
                )
                any_changed = True

    if not any_changed:
        print("[inject_faq_tags] No files modified.")


if __name__ == "__main__":
    main()
