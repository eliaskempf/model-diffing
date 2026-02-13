# Simple LLM Baselines are Competitive for Model Diffing

Official code for our paper ["Simple LLM Baselines are Competitive for Model Diffing"](https://arxiv.org/abs/2602.10371).

If you find this work useful, please consider citing our paper:
```bibtex
@article{kempf2026simple,
    title={Simple LLM Baselines are Competitive for Model Diffing},
    author={Kempf, Elias and Schrodi, Simon and Cywi{\'n}ski, Bartosz and Brox, Thomas and Nanda, Neel and Conmy, Arthur},
    journal={arXiv preprint arXiv:2602.10371},
    year={2026}
}
```

## Setup

We recommend using [uv](https://docs.astral.sh/uv/) to work with this code base. After cloning, you can setup the package as follows:

```bash
# Install base dependencies
uv sync

# Include development tools (pytest, ruff, Jupyter)
uv sync --extra dev

# Include safety-tooling (for response caching with openrouter)
uv sync --extra safety
```

The diffing and evaluation pipelines uses OpenRouter. Set your API key:
```bash
export OPENROUTER_API_KEY=your-key-here
```

## Running the pipelines

### LLM diffing pipeline

```bash
# Run the full pipeline (generate -> diff -> embed -> cluster -> aggregate)
uv run python scripts/run_diffing.py \
  --model_name_a google/gemini-2.5-flash-lite \
  --model_name_b google/gemini-2.5-flash-lite-preview-09-2025 \
  --comparator_model_name google/gemini-2.5-flash \
  --prompts wild_chat
```

### Evaluation pipeline

```bash
# From scratch — judge hypotheses then evaluate
uv run python scripts/run_eval.py \
  --cluster_path path/to/clusters.jsonl \
  --model_a_responses path/to/model_a/responses.jsonl \
  --model_b_responses path/to/model_b/responses.jsonl \
  --model_a_test_responses path/to/model_a/test_responses.jsonl \
  --model_b_test_responses path/to/model_b/test_responses.jsonl \
  --output_dir output/eval_results

# From pre-computed judge results
uv run python scripts/run_eval.py \
  --train_judge_results path/to/train_judging_results.json \
  --test_judge_results path/to/test_judging_results.json \
  --output_dir output/eval_results
```

### SAE diffing pipeline

To run the SAE-based diffing experiments, we used the [interp_embed](https://github.com/nickjiang2378/interp-embed) package from [Jiang et al.](https://arxiv.org/abs/2512.10092). We used their default configuration of Llama 3.3 70B as the reader model with the Goodfire SAE and the corresponding labels to create the datasets (see [this example](https://github.com/nickjiang2378/interp-embed/blob/main/README.md#quickstart) for details). In addition to the model responses, we also included the prompts when computing the SAE embeddings, but restricted the max pooling to the response tokens.

After generating the datasets (from the train responses only), we ran their pipeline using:
```bash
python paper/diffing/generate_sae_hypotheses.py \
    --dataset1 path/to/model_a/dataset.pkl \
    --dataset2 path/to/model_b/dataset.pkl \
    --max-feature-diffs 1000 \
    --num-hypotheses 40 \
    --both
```

## Development

```bash
uv sync --extra dev
uv run ruff format .         # Auto-format
uv run ruff check --fix .    # Lint with auto-fix
uv run pytest tests/ -v      # Run tests
```

### safetytooling dependency

`safetytooling` is installed from GitHub. It pins exact versions of `transformers` and
`datasets` that conflict with ours, so `pyproject.toml` has `override-dependencies` to
resolve this. The safetytooling import in `model_cached.py` is conditional
(`HAS_SAFETYTOOLING` flag), so `CachedModelWrapper` works without it — you just lose
file-based response caching.


### CUDA

uv installs torch from PyPI which includes CUDA wheels. If the CUDA version doesn't
match your GPU drivers, add a PyTorch index override to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```