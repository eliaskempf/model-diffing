"""Load prompt templates from JSON files in the prompts/ directory."""

import json
from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


@cache
def load_prompts(module_path: str) -> dict[str, str]:
    """Load prompt templates for a module.

    Args:
        module_path: Slash-separated path relative to prompts/, e.g. 'evaluation/judge'.

    Returns:
        Dict mapping template names to their string values.
    """
    path = _PROMPTS_DIR / f"{module_path}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
