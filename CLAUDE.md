# CLAUDE.md

## Project Overview

**model-diffing** is a framework for comparing LLM behaviors through hypothesis generation, evaluation, and visualization. It implements an end-to-end pipeline: generate responses from two models, diff them with an LLM, embed/cluster differences into testable hypotheses, evaluate those hypotheses, and visualize results.

## Structure

- `src/model_diffing/` — Installable Python library (core data structures, model wrappers, utilities)
- `src/model_diffing/prompts/` — Externalized prompt templates (JSON), loaded by `model_diffing.prompts`
- `configs/` — Experiment configuration (YAML), loaded by `model_diffing.hypothesis_config`
- `scripts/` — Executable scripts organized by function:
  - `run_diffing.py` — Main diffing pipeline runner
  - `run_eval.py` — Evaluation pipeline runner
  - `plotting/` — Visualization scripts
  - `gender/` — Gender assumption experiment scripts
  - `tools/` — Standalone utilities (hypothesis explorer, format converters, etc.)

## Running

```bash
uv sync                                          # Install base dependencies
uv sync --extra dev                              # Include dev tools (pytest, jupyter)
uv sync --extra safety                           # Include safetytooling (file-based response caching)
uv run python scripts/run_diffing.py --help      # Diffing pipeline
uv run python scripts/run_eval.py --help         # Evaluation pipeline
```

## Key Classes

- `ModelWrapper` (`model_diffing.model`) — Wraps HF models and OpenRouter API with auto-detection. Supports `force_openrouter=True` to skip API check and `openrouter_num_threads` for semaphore concurrency control.
- `CachedModelWrapper` (`model_diffing.model_cached`) — OpenRouter wrapper with semaphore concurrency control. Adds file-based response caching when safetytooling is installed; falls back to parent's httpx implementation without it.
- `ResponseDict` (`model_diffing.utils`) — Hash-indexed conversation-response store
- `WildChat`, `MixtureDataset` (`model_diffing.data`) — Dataset loaders

## Linting & Formatting

```bash
uv run ruff format .        # Auto-format
uv run ruff check --fix .   # Lint with auto-fix
uv run ruff check .         # Lint (check only)
```

Configured in `pyproject.toml` with line-length=120, target py312, rules: F, E, W, I, UP, B, SIM, RUF.

## Optional Dependencies

| Extra | What it includes | When you need it |
|-------|-----------------|------------------|
| `dev` | ruff, ipykernel, pytest | Development, formatting, testing |
| `safety` | safetytooling (GitHub) | File-based response caching in CachedModelWrapper |

## Important Notes

- `ModelWrapper` auto-detects OpenRouter models via API check. Use `force_hf=True` or `force_openrouter=True` to skip the check.
- `CachedModelWrapper` works without safetytooling (loses caching, keeps semaphore concurrency control). Install safetytooling for file-based response caching: `uv sync --extra safety`.

