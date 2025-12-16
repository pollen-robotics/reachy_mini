"""Provide shared utilities for FAQ and troubleshooting generation scripts."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Pattern, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_SOURCE_DIR = ROOT / "docs" / "source"


@dataclass(frozen=True)
class QaConfig:
    """Configuration for a Q&A-like content family."""

    name: str  # e.g. "faq" or "troubleshooting"
    data_dir: pathlib.Path  # e.g. docs/faq or docs/troubleshooting
    default_answers_dir: pathlib.Path  # fallback for answers
    section_block_label: str  # e.g. "FAQ" or "TROUBLESHOOTING"
    tags_block_label: str  # e.g. "FAQ-TAGS" or "TROUBLESHOOTING-TAGS"


FAQ_CONFIG = QaConfig(
    name="faq",
    data_dir=ROOT / "docs" / "faq",
    default_answers_dir=ROOT / "docs" / "faq" / "answers",
    section_block_label="FAQ",
    tags_block_label="FAQ-TAGS",
)


# ---------------------------------------------------------------------------
# JSON and items loading
# ---------------------------------------------------------------------------


def iter_json_files(config: QaConfig) -> Iterable[pathlib.Path]:
    """Iterate over all JSON files for a given config."""
    return config.data_dir.rglob("*.json")


def load_json_items(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load items from a JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        try:
            items: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"JSON error in {json_path}: {e}"
            raise RuntimeError(msg) from e
    return items


def load_all_items(config: QaConfig) -> List[Dict[str, Any]]:
    """Load all items for a given config from all JSON files."""
    all_items: List[Dict[str, Any]] = []
    for json_path in iter_json_files(config):
        section_name = json_path.stem
        items = load_json_items(json_path)

        # Prefer a local "answers" directory next to the JSON file.
        answers_dir_candidate = json_path.parent / "answers"
        if answers_dir_candidate.exists():
            answers_dir = answers_dir_candidate
        else:
            answers_dir = config.default_answers_dir

        for it in items:
            it.setdefault("_section", section_name)
            it.setdefault("_answers_dir", answers_dir)

        all_items.extend(items)

    return all_items


def load_answer_text(item: Dict[str, Any], config: QaConfig) -> str:
    """Load the answer text for an item."""
    answer_file = item.get("answer_file")
    if not answer_file:
        msg = f"Missing 'answer_file' for question: {item.get('question')}"
        raise KeyError(msg)

    answers_dir = item.get("_answers_dir") or config.default_answers_dir
    answer_path = pathlib.Path(answers_dir) / answer_file
    if not answer_path.exists():
        msg = f"Answer file not found for '{item.get('question')}': {answer_path}"
        raise FileNotFoundError(msg)

    with answer_path.open("r", encoding="utf-8") as f:
        # Strip only trailing newlines.
        return f.read().rstrip()


def item_has_any_tag(item: Dict[str, Any], wanted_tags: List[str]) -> bool:
    """Return True if the item has at least one of the wanted tags."""
    item_tags = [str(t).strip().lower() for t in item.get("tags", [])]
    wanted_tags_normalized = [t.strip().lower() for t in wanted_tags if t.strip()]
    return any(t in item_tags for t in wanted_tags_normalized)


# ---------------------------------------------------------------------------
# Rendering templates
# ---------------------------------------------------------------------------


def render_faq_item(item: Dict[str, Any], config: QaConfig = FAQ_CONFIG) -> str:
    """Render a FAQ item as an HTML details block."""
    question = item["question"]
    tags = item.get("tags", [])
    answer = load_answer_text(item, config)
    source = item.get("source")

    if tags:
        tags_html = " ".join(
            f'<kbd style="'
            "display:inline-block;"
            "padding:2px 10px;"
            "margin:2px 4px;"
            "background:rgba(59,176,209,0.1);"
            "color:#3bb0d1;"
            "border-radius:12px;"
            "font-size:11px;"
            "font-weight:500;"
            "text-transform:uppercase;"
            "letter-spacing:0.5px;"
            "border:none;"
            '">'
            f"{tag}"
            "</kbd>"
            for tag in tags
        )
        tags_block = f"Tags: {tags_html}"
    else:
        tags_block = ""

    source_html = ""
    if source:
        source_html = f'\n<p style="color:grey"><i>Source: {source}.</i></p>\n'

    block = f"""<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>{question}</strong><br>{tags_block}</summary>

{answer}

</details><br>
"""

    return block


# ---------------------------------------------------------------------------
# Generic helpers for sections and tags blocks
# ---------------------------------------------------------------------------


def build_section_pattern(
    label: str,
) -> Pattern[str]:
    """Build a compiled regex pattern for section markers for a label.

    Markers have the form: <!-- LABEL:folder:section:start -->.
    """
    return re.compile(
        rf"(<!-- {label}:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->)"
        r"(.*?)"
        rf"(<!-- {label}:\2:\3:end -->)",
        re.DOTALL,
    )


def find_sections(content: str, label: str) -> List[Tuple[str, str]]:
    """Find all (folder_name, section_name) pairs for a given section label."""
    pattern = re.compile(
        rf"<!-- {label}:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->",
    )
    return sorted(set(pattern.findall(content)))


def replace_section_block(
    content: str,
    label: str,
    folder_name: str,
    section_name: str,
    new_block: str,
) -> str:
    """Replace a section block for the given label, folder, and section."""
    escaped_folder = re.escape(folder_name)
    escaped_section = re.escape(section_name)
    pattern = re.compile(
        (
            rf"(<!-- {label}:{escaped_folder}:{escaped_section}:start -->)"
            r"(.*?)"
            rf"(<!-- {label}:{escaped_folder}:{escaped_section}:end -->)"
        ),
        re.DOTALL,
    )
    replacement = rf"\1\n\n{new_block}\n\3"
    content, n = pattern.subn(replacement, content)
    if n == 0:
        msg = (
            f"No block for section '{folder_name}:{section_name}' with label "
            f"'{label}' found in content."
        )
        raise ValueError(msg)
    return content


def build_tags_pattern(label: str) -> Pattern[str]:
    """Build a compiled regex pattern for tags markers for a label.

    Markers have the form: <!-- LABEL:expr:start -->.
    """
    return re.compile(
        rf"(?P<start><!-- {label}:([^:]+):start -->)"
        r"(?P<body>.*?)"
        rf"(?P<end><!-- {label}:\2:end -->)",
        re.DOTALL,
    )


def find_tags_placeholders(content: str, label: str) -> List[str]:
    """Return tag expressions for a given label as they appear in content."""
    pattern = re.compile(rf"<!-- {label}:([^:]+):start -->")
    matches = pattern.findall(content)
    return sorted({m.strip() for m in matches})


def replace_tags_block(
    content: str,
    label: str,
    tags_expr: str,
    new_block: str,
) -> str:
    """Replace a tags block for the given label and tag expression."""
    escaped_expr = re.escape(tags_expr)
    pattern = re.compile(
        (
            rf"(<!-- {label}:{escaped_expr}:start -->)"
            r"(.*?)"
            rf"(<!-- {label}:{escaped_expr}:end -->)"
        ),
        re.DOTALL,
    )
    replacement = r"\1\n\n" + new_block + r"\n\3"
    content, n = pattern.subn(replacement, content)
    if n == 0:
        msg = f"No {label} block found for '{tags_expr}' in the file."
        raise ValueError(msg)
    return content


def render_items_by_tags(
    config: QaConfig,
    tags_expr: str,
    render_fn: Callable[[Dict[str, Any], QaConfig], str],
) -> str:
    """Render all items for a config that match at least one requested tag."""
    tags = [t.strip() for t in tags_expr.split(",") if t.strip()]
    if not tags:
        msg = f"Empty tags expression: '{tags_expr}'"
        raise ValueError(msg)

    all_items = load_all_items(config)
    matching: List[Dict[str, Any]] = [
        it for it in all_items if item_has_any_tag(it, tags)
    ]

    seen: set[Tuple[str, str]] = set()
    unique_items: List[Dict[str, Any]] = []
    for it in matching:
        key = (it.get("question", ""), it.get("answer_file", ""))
        if key not in seen:
            seen.add(key)
            unique_items.append(it)

    if not unique_items:
        return f"> No questions found for tags: {', '.join(tags)}\n"

    rendered_items = [render_fn(item, config) for item in unique_items]
    return "\n\n".join(rendered_items) + "\n"


def _keep_markers_only(match: re.Match) -> str:
    """Return start marker, two newlines, and end marker."""
    return f"{match.group('start')}\n\n{match.group('end')}"


def clean_qa_markers(content: str, config: QaConfig) -> str:
    """Clean section and tag markers for a given Q&A config.

    This deletes everything between the markers:

    - <!-- LABEL:section:start --> ... <!-- LABEL:section:end -->
    - <!-- LABEL:folder:section:start --> ... <!-- LABEL:folder:section:end -->
    - <!-- TAG_LABEL:expr:start --> ... <!-- TAG_LABEL:expr:end -->

    leaving only the markers, with an empty line between the two.

    The implementation runs in one pass per block type (no while loop).
    """
    section_label = config.section_block_label
    tags_label = config.tags_block_label

    if f"<!-- {section_label}" not in content and f"<!-- {tags_label}" not in content:
        return content

    # Old section blocks: <!-- LABEL:section-name:start --> ... <!-- LABEL:section-name:end -->
    section_pattern_old = re.compile(
        rf"(?P<start><!-- {section_label}:([a-zA-Z0-9_-]+):start -->)"
        r"(?P<body>.*?)"
        rf"(?P<end><!-- {section_label}:\2:end -->)",
        re.DOTALL,
    )

    # New section blocks with folder:
    # <!-- LABEL:folder:section-name:start --> ... <!-- LABEL:folder:section-name:end -->
    section_pattern_new = re.compile(
        rf"(?P<start><!-- {section_label}:([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+):start -->)"
        r"(?P<body>.*?)"
        rf"(?P<end><!-- {section_label}:\2:\3:end -->)",
        re.DOTALL,
    )

    # Blocks by tags:
    # <!-- TAG_LABEL:expr:start --> ... <!-- TAG_LABEL:expr:end -->
    tags_pattern = re.compile(
        rf"(?P<start><!-- {tags_label}:([^:]+):start -->)"
        r"(?P<body>.*?)"
        rf"(?P<end><!-- {tags_label}:\2:end -->)",
        re.DOTALL,
    )

    # Clean new folder+section markers first, then old ones, then tag markers.
    content = section_pattern_new.sub(_keep_markers_only, content)
    content = section_pattern_old.sub(_keep_markers_only, content)
    content = tags_pattern.sub(_keep_markers_only, content)
    return content


def iter_markdown_files(
    paths: Iterable[pathlib.Path],
    docs_source_dir: pathlib.Path = DOCS_SOURCE_DIR,
) -> Iterable[pathlib.Path]:
    """Yield Markdown files to process.

    If paths are provided, only those .md/.mdx files are yielded.
    Otherwise, all docs/source/*.md(x) files are scanned.
    """
    if paths:
        for p in paths:
            p = p.resolve()
            if p.is_file() and p.suffix.lower() in {".md", ".mdx"}:
                yield p
        return

    if not docs_source_dir.exists():
        msg = f"Directory docs/source not found: {docs_source_dir}"
        raise FileNotFoundError(msg)

    for path in docs_source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}:
            yield path
