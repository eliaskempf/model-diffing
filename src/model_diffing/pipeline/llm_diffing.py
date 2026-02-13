import asyncio
import json
import os
import random

from tqdm import tqdm

from model_diffing.model_cached import CachedModelWrapper
from model_diffing.parsing import parse_llm_json, resolve_model_labels
from model_diffing.prompts import load_prompts
from model_diffing.utils import ResponseDict, batch_iterable

_prompts = load_prompts("pipeline/diffing")
system_prompt_diffing = _prompts["system_prompt_diffing"]
system_prompt_fixing = _prompts["system_prompt_fixing"]
user_template_diffing = _prompts["user_template_diffing"]


def compare_llm_responses(
    comparator_model_name: str,
    open_router_api_key: str,
    responses_a: str,
    responses_b: str,
    limit_samples: int,
    batch_size: int,
    seed: int = 42,
    regenerate: bool = False,
    model_a: str | None = None,
    model_b: str | None = None,
) -> str:
    if not os.path.exists(responses_a):
        raise FileNotFoundError(f"Responses file {responses_a} does not exist")
    if not os.path.exists(responses_b):
        raise FileNotFoundError(f"Responses file {responses_b} does not exist")

    part1 = os.path.basename(responses_a).replace("_responses.jsonl", "")
    part2 = os.path.basename(responses_b).replace("_responses.jsonl", "")
    output_file = os.path.join(os.path.dirname(responses_a), f"{part1}__{part2}_diffs.jsonl")
    if os.path.exists(output_file) and not regenerate:
        print(f"Diffs file {output_file} already exists and regenerate is False. Skipping diffing.")
        return output_file

    llm_baseline = CachedModelWrapper(
        model_name=comparator_model_name,
        api_key=open_router_api_key,
    )

    response_dict1 = ResponseDict.from_jsonl(responses_a)
    if model_a is None:  # backward compatibility
        model_a = responses_a.split("/")[-3]
    response_dict2 = ResponseDict.from_jsonl(responses_b)
    if model_b is None:
        model_b = responses_b.split("/")[-3]

    hashes = [set(dataset.keys()) for dataset in [response_dict1, response_dict2]]
    shared_hashes = sorted(list(set.intersection(*hashes)))
    print(len(shared_hashes), "shared samples found across datasets.")
    random.seed(seed)
    if limit_samples < len(shared_hashes):
        shared_hashes = random.sample(shared_hashes, limit_samples)

    prompts = []
    for hash in shared_hashes:
        if len(response_dict1[hash]["conversation"]) != 1:
            raise ValueError(
                f"Expected single-turn conversations, got {len(response_dict1[hash]['conversation'])} turns"
            )
        text1 = response_dict1[hash]["response"]
        text2 = response_dict2[hash]["response"]
        user_prompt = user_template_diffing.format(
            prompt=response_dict1[hash]["conversation"][0]["content"],
            model_a_response=text1,
            model_b_response=text2,
        )
        prompts.append((hash, user_prompt))

    results_raw, results_dict = {}, {}
    with (
        open(output_file.replace(".jsonl", "_raw.jsonl"), "a", encoding="utf-8") as f_raw,
        open(output_file, "a", encoding="utf-8") as f_clean,
    ):
        for batch in tqdm(
            batch_iterable(prompts[:limit_samples], batch_size),
            desc="Processing batches",
            total=(limit_samples + batch_size - 1) // batch_size,
        ):
            batch_hashes, batch_prompts = zip(*batch)

            responses = asyncio.run(
                llm_baseline.generate_async(
                    [
                        [
                            {"role": "system", "content": system_prompt_diffing},
                            {"role": "user", "content": batch_prompt},
                        ]
                        for batch_prompt in batch_prompts
                    ],
                    max_new_tokens=4096,
                    enable_thinking=False,
                    seed=42,
                )
            )

            for hash, response in zip(batch_hashes, responses):
                results_raw[hash] = response
                try:
                    results_dict[hash] = parse_llm_json(response)
                    valid = True
                except Exception:
                    patience = 5
                    while patience > 0:
                        response_fixed = asyncio.run(
                            llm_baseline.generate_async(
                                [
                                    [
                                        {"role": "system", "content": system_prompt_fixing},
                                        {"role": "user", "content": response},
                                    ]
                                ],
                                max_new_tokens=4096,
                                enable_thinking=False,
                                seed=42,
                            )
                        )[0]
                        try:
                            results_dict[hash] = parse_llm_json(response_fixed)
                            valid = True
                            break
                        except Exception:
                            patience -= 1

                    if patience == 0:
                        results_dict[hash] = {"error": "Failed to parse", "raw": response}
                        valid = False

                if valid:
                    resolve_model_labels(results_dict[hash], model_a, model_b)

                f_raw.write(json.dumps({"hash": hash, "result": results_raw[hash]}) + "\n")
                f_raw.flush()

                f_clean.write(json.dumps({"hash": hash, "result": results_dict[hash]}) + "\n")
                f_clean.flush()

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator_model_name", type=str, required=True, help="Name of LLM to act as comparator")
    parser.add_argument(
        "--open_router_api_key", type=str, default=None, help="API key for OpenRouter if using OpenRouter mode"
    )
    parser.add_argument("--responses1", type=str, required=True, help="Path to file containing generated responses")
    parser.add_argument("--responses2", type=str, required=True, help="Path to file containing generated responses")
    parser.add_argument("--limit_samples", type=int, default=1000, help="Limit number of samples from dataset")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of samples to verify")
    args = parser.parse_args()

    compare_llm_responses(
        comparator_model_name=args.comparator_model_name,
        open_router_api_key=args.open_router_api_key,
        responses_a=args.responses1,
        responses_b=args.responses2,
        limit_samples=args.limit_samples,
        batch_size=args.batch_size,
    )
