"""Clean FAQ and FAQ-TAGS blocks in Markdown documentation files."""

import argparse
import pathlib
import re
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_SOURCE_DIR = ROOT / "docs" / "source"

# Old section blocks:
# <!-- FAQ:section-name:start --> ... <!-- FAQ:section-name:end -->
SECTION_PATTERN_OLD = re.compile(
    r"(?P<start><!-- FAQ:([a-zA-Z0-9_-]+):start -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- FAQ:\2:end -->)",
    re.DOTALL,
)

# New section blocks with folder:
# <!-- FAQ:folder:section-name:start --> ... <!-- FAQ:folder:section-name:end -->
SECTION_PATTERN_NEW = re.compile(
    r"(?P<start><!-- FAQ:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- FAQ:\2:\3:end -->)",
    re.DOTALL,
)

# Blocks by tags:
# <!-- FAQ-TAGS:expr:start --> ... <!-- FAQ-TAGS:expr:end -->
TAGS_PATTERN = re.compile(
    r"(?P<start><!-- FAQ-TAGS:([^:]+):start -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- FAQ-TAGS:\2:end -->)",
    re.DOTALL,
)


def _keep_markers_only(match: re.Match) -> str:
    """Return start marker, two newlines, and end marker."""
    return f"{match.group('start')}\n\n{match.group('end')}"


def clean_content(content: str) -> str:
    """Clean FAQ markers content from the given text.

    This deletes everything between the markers:

    - <!-- FAQ:section:start --> ... <!-- FAQ:section:end -->
    - <!-- FAQ:folder:section:start --> ... <!-- FAQ:folder:section:end -->
    - <!-- FAQ-TAGS:expr:start --> ... <!-- FAQ-TAGS:expr:end -->

    leaving only the markers, with an empty line between the two.

    The implementation runs in one pass per block type (no while loop).
    """
    if "<!-- FAQ" not in content:
        return content

    # Clean new folder+section markers first, then old ones, then tag markers.
    content = SECTION_PATTERN_NEW.sub(_keep_markers_only, content)
    content = SECTION_PATTERN_OLD.sub(_keep_markers_only, content)
    content = TAGS_PATTERN.sub(_keep_markers_only, content)
    return content


def process_file(path: pathlib.Path) -> bool:
    """Clean a Markdown file.

    Returns True if the file was modified.
    """
    original = path.read_text(encoding="utf-8")
    cleaned = clean_content(original)
    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
        print(f"[clean_faq_blocks] cleaned: {path.relative_to(ROOT)}")
        return True
    return False


def iter_target_files(paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    """Yield Markdown files to clean.

    If paths are provided as arguments, only those are processed.
    Otherwise, all docs/source/*.md(x) files are scanned.
    """
    if paths:
        for p in paths:
            p = p.resolve()
            if p.is_file() and p.suffix.lower() in {".md", ".mdx"}:
                yield p
        return

    # Default mode: all docs/source.
    if not DOCS_SOURCE_DIR.exists():
        msg = f"Directory docs/source not found: {DOCS_SOURCE_DIR}"
        raise FileNotFoundError(msg)

    for path in DOCS_SOURCE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}:
            yield path


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

    provided_paths = [ROOT / f for f in args.files] if args.files else []

    any_changed = False
    for path in iter_target_files(provided_paths):
        if process_file(path):
            any_changed = True

    if not any_changed:
        print("[clean_faq_blocks] no changes (already clean).")


if __name__ == "__main__":
    main()
