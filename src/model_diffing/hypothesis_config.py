"""
Centralized configuration for hypothesis evaluation experiments.

Loads experiment definitions from a YAML config file. Provides helper
functions for querying experiments by key, name, or method, and derived
data structures (DATASETS, etc.) used by plotting and evaluation scripts.
"""

from pathlib import Path
from typing import NotRequired, TypedDict

import yaml


class ExperimentConfig(TypedDict):
    """Configuration for a single experiment."""

    name: str
    method: str
    key: str
    train_file: str
    test_file: str
    interestingness_file: NotRequired[str]
    abstraction_file: NotRequired[str]


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "experiments.yaml")


def load_config(config_path: str | None = None) -> dict:
    """Load and return the raw YAML config as a dict."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_from_config(config: dict) -> tuple:
    """Build all derived data structures from a loaded config dict."""
    display_names = config["display_names"]

    experiments: list[ExperimentConfig] = []
    for exp in config["experiments"]:
        ec = ExperimentConfig(
            name=exp["name"],
            method=exp["method"],
            key=exp["key"],
            train_file=exp["train_file"],
            test_file=exp["test_file"],
        )
        if "interestingness_file" in exp:
            ec["interestingness_file"] = exp["interestingness_file"]
        if "abstraction_file" in exp:
            ec["abstraction_file"] = exp["abstraction_file"]
        experiments.append(ec)

    datasets = []
    for ds in config.get("datasets", []):
        datasets.append((ds["name"], display_names[ds["name"]], ds["llm_key"], ds["sae_key"]))

    return experiments, datasets


# Load defaults at module level
_config = load_config()
EXPERIMENTS, DATASETS = _build_from_config(_config)


def get_experiment_by_key(key: str, experiments: list[ExperimentConfig] | None = None) -> ExperimentConfig | None:
    """Get experiment configuration by its key (e.g., 'gemini_llm')."""
    for exp in experiments or EXPERIMENTS:
        if exp["key"] == key:
            return exp
    return None


def get_experiments_by_name(name: str, experiments: list[ExperimentConfig] | None = None) -> list[ExperimentConfig]:
    """Get all experiments for a given name (e.g., 'gemini' returns both llm and sae)."""
    return [exp for exp in (experiments or EXPERIMENTS) if exp["name"] == name]


def get_experiments_by_method(method: str, experiments: list[ExperimentConfig] | None = None) -> list[ExperimentConfig]:
    """Get all experiments for a given method ('llm' or 'sae')."""
    return [exp for exp in (experiments or EXPERIMENTS) if exp["method"] == method]
