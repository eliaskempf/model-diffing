"""Helpers for parsing JSON from LLM text output.

Consolidates duplicated patterns from pipeline and evaluation scripts.
"""

import json
import re


def strip_json_fences(text: str) -> str:
    """Remove ```json / ``` markdown fences from LLM output."""
    return re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()


def parse_llm_json(text: str) -> dict | list:
    """Strip JSON fences and parse the result.

    Raises json.JSONDecodeError if the stripped text is not valid JSON.
    """
    return json.loads(strip_json_fences(text))


def resolve_model_labels(
    results: list[dict],
    model_a_name: str,
    model_b_name: str,
) -> list[dict]:
    """Replace 'Model A'/'A' and 'Model B'/'B' labels with real model names.

    Mutates and returns the input list. Each dict is expected to have a
    'model' key with value like 'Model A', 'A', 'Model B', or 'B'.
    """
    for item in results:
        model = item.get("model", "")
        if model in ("Model A", "A"):
            item["model_name"] = model_a_name
        elif model in ("Model B", "B"):
            item["model_name"] = model_b_name
        else:
            item["model_name"] = "unclear"
    return results
