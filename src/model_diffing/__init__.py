"""model-diffing: LLM behavioral comparison through hypothesis generation and evaluation."""

from model_diffing.content_detection import (
    contains_latex,
    contains_table,
)
from model_diffing.parsing import parse_llm_json, resolve_model_labels, strip_json_fences
from model_diffing.utils import ResponseDict, batch_iterable

__all__ = [
    "ResponseDict",
    "batch_iterable",
    "contains_latex",
    "contains_table",
    "parse_llm_json",
    "resolve_model_labels",
    "strip_json_fences",
]
