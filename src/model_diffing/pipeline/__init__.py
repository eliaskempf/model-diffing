"""Pipeline entry points for the model-diffing framework.

Uses lazy imports (PEP 562) to avoid pulling in heavy dependencies
(torch, umap, sklearn) when importing submodules directly.
"""

import importlib

_LAZY_IMPORTS = {
    "cluster_sentence_embeddings": "model_diffing.pipeline.cluster_sentence_embeddings",
    "compare_llm_responses": "model_diffing.pipeline.llm_diffing",
    "compute_sentence_embeddings": "model_diffing.pipeline.compute_sentence_embeddings",
    "generate_responses": "model_diffing.pipeline.generate",
    "llm_aggregation": "model_diffing.pipeline.llm_aggregation",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
