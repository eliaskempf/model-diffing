import asyncio
import json
import os

from model_diffing.model_cached import CachedModelWrapper
from model_diffing.parsing import parse_llm_json, resolve_model_labels
from model_diffing.prompts import load_prompts

_prompts = load_prompts("pipeline/aggregation")
system_prompt = _prompts["system_prompt"]
system_prompt_clean = _prompts["system_prompt_clean"]


def llm_aggregation(
    difference_results: str,
    model_name: str,
    open_router_api_key: str | None = None,
) -> None:
    assert os.path.isfile(difference_results)

    with open(difference_results, encoding="utf-8") as f:
        baseline_results = [json.loads(line) for line in f if line.strip()]

    model_a, model_b = None, None
    for results in baseline_results:
        for res in results["result"]:
            if "Model A" in res["model"]:
                model_a = res["model_name"]
            elif "Model B" in res["model"]:
                model_b = res["model_name"]
            if model_a is not None and model_b is not None:
                break
        if model_a is not None and model_b is not None:
            break
    assert model_a is not None and model_b is not None, "Could not determine model names."

    descriptions = []
    models = []
    model_names = []
    for results in baseline_results:
        if "error" in results["result"] and "raw" in results["result"]:
            continue
        for res in results["result"]:
            descriptions.append(res["property_description"])
            models.append(res["model"])
            model_names.append(res["model_name"])
            descriptions.append(res["not_property_description"])
            models.append("Model B" if res["model"] == "Model A" else "Model A")
            model_names.append(model_b if res["model"] == "Model A" else model_a)

    llm_aggregator = CachedModelWrapper(
        model_name=model_name,
        api_key=open_router_api_key,
    )

    user_prompts = []
    for _model, difference in zip(models, descriptions):
        user_prompts.append(difference)
    user_prompt = "\n\n".join(user_prompts)

    response = asyncio.run(
        llm_aggregator.generate_async(
            [[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]],
            max_new_tokens=64000,
            enable_thinking=False,
            seed=42,
        )
    )[0]

    try:
        res = parse_llm_json(response)
    except Exception:
        response2 = asyncio.run(
            llm_aggregator.generate_async(
                [[{"role": "system", "content": system_prompt_clean}, {"role": "user", "content": response}]],
                max_new_tokens=64000,
                enable_thinking=False,
                seed=42,
            )
        )[0]
        res = parse_llm_json(response2)

    resolve_model_labels(res, model_a, model_b)
    with open(difference_results.replace(".jsonl", "_llm_aggregated.jsonl"), "w", encoding="utf-8") as f:
        for result in res:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of LLM to verify responses from")
    parser.add_argument(
        "--open_router_api_key", type=str, default=None, help="API key for OpenRouter if using OpenRouter mode"
    )
    parser.add_argument("--baseline_results", type=str, required=True, help="Path to jsonl containing baseline results")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of samples to verify")
    args = parser.parse_args()

    llm_aggregation(
        difference_results=args.baseline_results,
        model_name=args.model_name,
        open_router_api_key=args.open_router_api_key,
    )
