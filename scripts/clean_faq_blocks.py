import argparse
import pathlib
import re
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_SOURCE_DIR = ROOT / "docs" / "source"

# Blocks by section:
# <!-- FAQ:section-name:start --> ... <!-- FAQ:section-name:end -->
SECTION_PATTERN = re.compile(
    r"(<!-- FAQ:([a-zA-Z0-9_-]+):start -->)(.*?)(<!-- FAQ:\2:end -->)",
    re.DOTALL,
)

# Blocks by tags:
# <!-- FAQ-TAGS:expr:start --> ... <!-- FAQ-TAGS:expr:end -->
TAGS_PATTERN = re.compile(
    r"(<!-- FAQ-TAGS:([^:]+):start -->)(.*?)(<!-- FAQ-TAGS:\2:end -->)",
    re.DOTALL,
)


def clean_content(content: str) -> str:
    """
    Delete everything between the markers:
      - <!-- FAQ:xxx:start --> ... <!-- FAQ:xxx:end -->
      - <!-- FAQ-TAGS:expr:start --> ... <!-- FAQ-TAGS:expr:end -->

    leaving only the markers, with an empty line between the two.

    Implementation in ONE PASS per block type (no while loop).
    """
    # Si pas de marqueur, on retourne direct
    if "<!-- FAQ" not in content:
        return content

    content = SECTION_PATTERN.sub(r"\1\n\n\4", content)
    content = TAGS_PATTERN.sub(r"\1\n\n\4", content)
    return content


def process_file(path: pathlib.Path) -> bool:
    """
    Clean a .md/.mdx file.
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
    """
    If paths are provided as arguments, only those are processed.
    Otherwise, it scans all docs/source/*.md(x).
    """
    if paths:
        for p in paths:
            p = p.resolve()
            if p.is_file() and p.suffix.lower() in {".md", ".mdx"}:
                yield p
        return

    # Default mode: all docs/source
    if not DOCS_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Directory docs/source not found: {DOCS_SOURCE_DIR}")

    for path in DOCS_SOURCE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean the content between FAQ/FAQ-TAGS blocks."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Markdown (.md/.mdx) files to clean (relative or absolute paths). "
        "If none are provided, it will scan the docs/source/ directory.",
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
