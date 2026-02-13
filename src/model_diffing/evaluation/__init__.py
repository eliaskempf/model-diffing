"""Evaluation entry points for the model-diffing framework.

Uses lazy imports (PEP 562) to avoid pulling in heavy dependencies
(torch, safetytooling) when importing submodules directly.
"""

import importlib

_LAZY_IMPORTS = {
    # evaluate_hypotheses.py — pure-Python scoring, no heavy deps
    "evaluate_all": "model_diffing.evaluation.evaluate_hypotheses",
    "evaluate_experiment": "model_diffing.evaluation.evaluate_hypotheses",
    "evaluate_hypothesis": "model_diffing.evaluation.evaluate_hypotheses",
    "ExperimentResults": "model_diffing.evaluation.evaluate_hypotheses",
    "HypothesisResult": "model_diffing.evaluation.evaluate_hypotheses",
    "get_direction": "model_diffing.evaluation.evaluate_hypotheses",
    "compute_accuracy": "model_diffing.evaluation.evaluate_hypotheses",
    "compute_frequency": "model_diffing.evaluation.evaluate_hypotheses",
    "load_judging_results": "model_diffing.evaluation.evaluate_hypotheses",
    "load_autorater_scores": "model_diffing.evaluation.evaluate_hypotheses",
    # judge_batched.py — requires CachedModelWrapper
    "judge_hypotheses_batched": "model_diffing.evaluation.judge_batched",
    # compute_judge_variance.py
    "compute_variance": "model_diffing.evaluation.compute_judge_variance",
    "compute_variance_stats": "model_diffing.evaluation.compute_judge_variance",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
