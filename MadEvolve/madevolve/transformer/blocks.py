"""
Evolve-block utilities for targeted code evolution.

When code contains EVOLVE-BLOCK markers, only the content between
the markers is mutable. Infrastructure code outside the markers is
protected from mutation. If no markers are present, the entire file
is treated as mutable (backward compatible).
"""

import re
from typing import Tuple

from madevolve.common.constants import (
    EVOLVE_BLOCK_START_PATTERN,
    EVOLVE_BLOCK_END_PATTERN,
)


def has_evolve_blocks(code: str) -> bool:
    """Check whether code contains evolve-block markers."""
    return (
        re.search(EVOLVE_BLOCK_START_PATTERN, code) is not None
        and re.search(EVOLVE_BLOCK_END_PATTERN, code) is not None
    )


def split_code_regions(code: str) -> Tuple[str, str, str]:
    """
    Split code into (prefix, mutable, suffix) around evolve-block markers.

    The prefix includes everything up to and including the EVOLVE-BLOCK-START
    marker line. The suffix includes the EVOLVE-BLOCK-END marker line and
    everything after it. The mutable region is the content between the markers.

    If no markers are present, returns ("", code, "").
    """
    if not has_evolve_blocks(code):
        return "", code, ""

    lines = code.split("\n")

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if re.search(EVOLVE_BLOCK_START_PATTERN, line) and start_idx is None:
            start_idx = i
        if re.search(EVOLVE_BLOCK_END_PATTERN, line):
            end_idx = i

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return "", code, ""

    prefix = "\n".join(lines[: start_idx + 1])
    mutable = "\n".join(lines[start_idx + 1 : end_idx])
    suffix = "\n".join(lines[end_idx:])

    return prefix, mutable, suffix


def extract_mutable_content(code: str) -> str:
    """
    Return only the mutable block content, or the full code if no markers.
    """
    _, mutable, _ = split_code_regions(code)
    return mutable


def replace_mutable_content(full_code: str, new_mutable: str) -> str:
    """
    Swap the mutable region in *full_code* with *new_mutable*.

    Auto-detects indentation from the original mutable content and re-indents
    the LLM output to match. This prevents LLM-generated code from breaking
    when the EVOLVE-BLOCK is inside a class body or other indented context.

    If full_code has no markers, returns new_mutable unchanged.
    """
    prefix, orig_mutable, suffix = split_code_regions(full_code)
    if not prefix and not suffix:
        return new_mutable

    # Detect expected indentation from original mutable content.
    # Use the minimum indentation of non-empty lines as the base.
    expected_indent = ""
    for line in orig_mutable.split("\n"):
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            indent = line[:len(line) - len(stripped)]
            if not expected_indent or len(indent) < len(expected_indent):
                expected_indent = indent
    # Fallback: use indentation of the EVOLVE-BLOCK-START marker line
    if not expected_indent:
        marker_line = prefix.rstrip().split("\n")[-1]
        expected_indent = marker_line[:len(marker_line) - len(marker_line.lstrip())]

    # Re-indent LLM output: detect and normalize indentation
    new_lines = new_mutable.split("\n")
    while new_lines and not new_lines[0].strip():
        new_lines.pop(0)
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()

    # Detect the minimum indentation across all non-empty lines in the LLM output
    llm_min_indent = None
    for line in new_lines:
        stripped = line.lstrip()
        if stripped:
            indent_len = len(line) - len(stripped)
            if llm_min_indent is None or indent_len < llm_min_indent:
                llm_min_indent = indent_len
    llm_min_indent = llm_min_indent or 0

    # Re-indent: strip llm_min_indent from each line, then add expected_indent
    reindented = []
    for line in new_lines:
        if not line.strip():
            reindented.append("")
        else:
            stripped = line.lstrip()
            actual_indent = len(line) - len(stripped)
            relative_indent = actual_indent - llm_min_indent
            reindented.append(expected_indent + " " * relative_indent + stripped)

    return prefix + "\n" + "\n".join(reindented) + "\n" + suffix


def format_code_for_prompt(code: str, show_full: bool) -> str:
    """
    Return full code (with markers visible) or only the mutable block.

    Args:
        code: Source code, possibly with evolve-block markers.
        show_full: If True, return the entire code. If False, return only
                   the mutable content (or full code if no markers).
    """
    if show_full or not has_evolve_blocks(code):
        return code
    return extract_mutable_content(code)
