"""Clean FAQ and FAQ-TAGS blocks in Markdown documentation files."""

import argparse
import pathlib
from typing import Iterable

from qa_utils import (
    DOCS_SOURCE_DIR,
    FAQ_CONFIG,
    ROOT,
    clean_qa_markers,
    iter_markdown_files,
)


def process_file(path: pathlib.Path) -> bool:
    """Clean a Markdown file for FAQ markers.

    Returns True if the file was modified.
    """
    original = path.read_text(encoding="utf-8")
    cleaned = clean_qa_markers(original, FAQ_CONFIG)
    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
        print(f"[clean_faq_blocks] cleaned: {path.relative_to(ROOT)}")
        return True
    return False


def main() -> None:
    """Run the FAQ blocks cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean the content between FAQ/FAQ-TAGS blocks.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Markdown (.md/.mdx) files to clean (relative or absolute paths). "
            "If none are provided, it will scan the docs/source/ directory."
        ),
    )
    args = parser.parse_args()

    provided_paths: Iterable[pathlib.Path] = (
        [ROOT / f for f in args.files] if args.files else []
    )

    any_changed = False
    for path in iter_markdown_files(provided_paths, DOCS_SOURCE_DIR):
        if process_file(path):
            any_changed = True

    if not any_changed:
        print("[clean_faq_blocks] no changes (already clean).")


if __name__ == "__main__":
    main()
