"""Inject FAQ blocks based on tag expressions in Markdown files."""

import pathlib
from typing import List

from qa_utils import (
    DOCS_SOURCE_DIR,
    FAQ_CONFIG,
    find_tags_placeholders,
    render_faq_item,
    render_items_by_tags,
    replace_tags_block,
)


def process_file(path: pathlib.Path) -> bool:
    """Process a Markdown file and update its FAQ blocks."""
    original = path.read_text(encoding="utf-8")
    content = original

    raw_exprs: List[str] = find_tags_placeholders(
        content,
        FAQ_CONFIG.tags_block_label,
    )
    if not raw_exprs:
        return False

    for expr in raw_exprs:
        block = render_items_by_tags(FAQ_CONFIG, expr, render_faq_item)
        content = replace_tags_block(
            content,
            FAQ_CONFIG.tags_block_label,
            expr,
            block,
        )

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
