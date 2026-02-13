"""Regex-based content detection for LLM response text.

Moved from scripts/tools/find_tables.py for reuse.
"""

import re

_LATEX_REGEX = re.compile(
    r"""(?s)(?:(?<!\\)\$\$(?:(?!\$\$).)*?(?<!\\)\$\$|(?<!\\)\$(?:[^$\\]|\\.)*?(?:\\(?:frac|sqrt|sum|int|lim|prod|alpha|beta|gamma|to|le|ge|cdot|pm|times|ldots|infty)|[_^]\{[^}]+\}|[A-Za-z]\s*[_^])(?:[^$\\]|\\.)*?(?<!\\)\$|\\\((?:[^\\]|\\.)*?\\\)|\\\[(?:[^\\]|\\.)*?\\\]|\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|cases|pmatrix|bmatrix|matrix)\}.*?\\end\{\1\}|\\(?:frac|sqrt|sum|int|lim|prod|mathbb|mathbf|mathrm|left|right)\b)"""
)

_TABLE_REGEX = re.compile(r"(?m)^\|.*\|\s*\n\|[-:\s|]+\|\s*\n(?:\|.*\|\s*\n?)*")


def contains_latex(text: str) -> bool:
    """Check if text contains LaTeX math notation."""
    return bool(_LATEX_REGEX.search(text))


def contains_table(text: str) -> bool:
    """Check if text contains a Markdown table."""
    return _TABLE_REGEX.search(text) is not None
