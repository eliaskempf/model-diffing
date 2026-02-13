import asyncio
import json
import random

from model_diffing.model_cached import CachedModelWrapper
from model_diffing.parsing import strip_json_fences
from model_diffing.prompts import load_prompts
from model_diffing.utils import ResponseDict, batch_iterable

_prompts = load_prompts("evaluation/judge_batched")
system_prompt = _prompts["system_prompt"]
prompt_template = _prompts["prompt_template"]


def try_parsing_response(response: str) -> dict:
    cleaned = strip_json_fences(response)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(response)
        return {"error": "JSONDecodeError"}


def judge_hypotheses_batched(
    model_name: str,
    cluster_path: str,
    model_a_responses: str,
    model_b_responses: str,
    output_file: str = "judging_results.json",
    open_router_api_key: str | None = None,
    min_cluster_specificity: float | None = None,
    limit_samples: int | None = None,
    seed: int = 42,
    hypotheses_per_prompt: int = 10,
) -> str:
    """Judge hypotheses in batches per prompt. Returns output file path.

    Args:
        hypotheses_per_prompt: Number of hypotheses per API call (1-10). Higher = cheaper, lower = more accurate.
        seed: Random seed for sample shuffling.
        limit_samples: Maximum number of prompts to judge. If None, judges all shared responses.
    """
    if not (1 <= hypotheses_per_prompt <= 10):
        raise ValueError(f"hypotheses_per_prompt must be between 1 and 10, got {hypotheses_per_prompt}")
    verifier = CachedModelWrapper(
        model_name=model_name,
        api_key=open_router_api_key,
    )

    clusters = []
    with open(cluster_path, encoding="utf-8") as f:
        for line in f:
            cluster = json.loads(line)

            total_pct = cluster["model_a_percentage"] + cluster["model_b_percentage"]
            if min_cluster_specificity is not None and abs(total_pct - 1.0) >= 0.01:
                raise ValueError(f"Cluster percentages don't sum to 1.0: {total_pct:.4f}")
            if (
                min_cluster_specificity is not None
                and cluster["model_a_percentage"] < min_cluster_specificity
                and (1 - cluster["model_a_percentage"]) < min_cluster_specificity
            ):
                continue

            clusters.append(cluster)

    responses_a = ResponseDict.from_jsonl(model_a_responses)
    responses_b = ResponseDict.from_jsonl(model_b_responses)

    all_hashes = [h for h in responses_a if h in responses_b]

    random.seed(seed)
    random.shuffle(all_hashes)
    if limit_samples is not None:
        all_hashes = all_hashes[:limit_samples]

    eval_prompts_per_hash = []
    responses_flipped = []
    for hash in all_hashes:
        eval_prompts = []
        flipped = []

        for cluster_batch in batch_iterable(clusters, hypotheses_per_prompt):
            prompt = responses_a[hash]["conversation"][0]["content"]
            response1 = responses_a[hash]["response"]
            response2 = responses_b[hash]["response"]

            hypotheses = []
            for cluster in cluster_batch:
                hypothesis = cluster["hypothesis"]
                if hypothesis.startswith("Hypothesis: "):
                    hypothesis = hypothesis[len("Hypothesis: ") :]
                hypotheses.append(hypothesis)

            hypotheses = "\n".join([f"H{i + 1}: {hypo}" for i, hypo in enumerate(hypotheses)])

            if random.random() < 0.5:
                response1, response2 = response2, response1
                flipped.append(True)
            else:
                flipped.append(False)

            eval_prompts.append(
                prompt_template.format(
                    hypotheses=hypotheses,
                    prompt=prompt,
                    response1=response1,
                    response2=response2,
                )
            )

        eval_prompts_per_hash.append(eval_prompts)
        responses_flipped.append(flipped)

    flattened_prompts = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ]
        for hash_prompts in eval_prompts_per_hash
        for p in hash_prompts
    ]

    verification_responses = asyncio.run(
        verifier.generate_async(
            flattened_prompts,
            max_new_tokens=1024,
            enable_thinking=False,
            seed=42,
            show_progress=True,
        )
    )

    flattened_responses = [try_parsing_response(r) for r in verification_responses]

    eval_responses_per_hash = []
    idx = 0
    for i in range(len(all_hashes)):
        num_batched_clusters = len(eval_prompts_per_hash[i])
        eval_responses = flattened_responses[idx : idx + num_batched_clusters]
        idx += num_batched_clusters

        # unflip answers
        for j, resp in enumerate(eval_responses):
            if responses_flipped[i][j]:
                for key in resp:
                    if resp[key] == 1:
                        resp[key] = 2
                    elif resp[key] == 2:
                        resp[key] = 1

        if len(eval_responses) != len(eval_prompts_per_hash[i]):
            raise RuntimeError(f"Response count mismatch: {len(eval_responses)=} {len(eval_prompts_per_hash[i])=}")
        eval_responses_per_hash.append(eval_responses)

    # aggregate per hash results into final results per cluster
    results_per_cluster = [
        {
            "cluster_id": c["cluster_id"],
            "cluster_hypothesis": c["hypothesis"],
            "cluster_model_a_percentage": c["model_a_percentage"],
            "cluster_model_b_percentage": c["model_b_percentage"],
            "num_model_a": 0,
            "num_model_b": 0,
            "num_na": 0,
            "num_err": 0,
            "num_missing": 0,
            "total": 0,
        }
        for c in clusters
    ]

    # Precompute batch sizes (same for every hash)
    cluster_batch_sizes = [len(b) for b in batch_iterable(clusters, hypotheses_per_prompt)]

    if len(eval_responses_per_hash) != len(eval_prompts_per_hash):
        raise RuntimeError(
            f"Hash count mismatch: {len(eval_responses_per_hash)} responses vs {len(eval_prompts_per_hash)} prompts"
        )
    for eval_prompts, eval_responses in zip(eval_prompts_per_hash, eval_responses_per_hash):
        cluster_idx = 0
        for batch_idx, (p, r) in enumerate(zip(eval_prompts, eval_responses)):
            batch_size = cluster_batch_sizes[batch_idx]

            # bad json response
            if "error" in r:
                for _ in range(batch_size):
                    if cluster_idx >= len(clusters):
                        break
                    results_per_cluster[cluster_idx]["num_err"] += 1
                    results_per_cluster[cluster_idx]["total"] += 1
                    cluster_idx += 1
                continue

            for i in range(batch_size):
                if cluster_idx >= len(clusters):
                    break

                key = f"H{i + 1}"
                if key not in r:
                    results_per_cluster[cluster_idx]["num_missing"] += 1
                    results_per_cluster[cluster_idx]["total"] += 1
                    cluster_idx += 1
                    continue

                # sanity check
                lines = p.split("\n")
                matches = [line for line in lines if line.startswith(f"{key}: ")]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected exactly 1 hypothesis line for {key} in prompt, got {len(matches)}")
                cluster_hypothesis = results_per_cluster[cluster_idx]["cluster_hypothesis"]
                if cluster_hypothesis.startswith("Hypothesis: "):
                    cluster_hypothesis = cluster_hypothesis[len("Hypothesis: ") :]
                if cluster_hypothesis not in matches[0]:
                    raise ValueError(f"Hypothesis mismatch: {cluster_hypothesis!r} not found in {matches[0]!r}")

                answer = r[key]
                if answer == 1:
                    results_per_cluster[cluster_idx]["num_model_a"] += 1
                elif answer == 2:
                    results_per_cluster[cluster_idx]["num_model_b"] += 1
                elif answer == "N/A":
                    results_per_cluster[cluster_idx]["num_na"] += 1
                else:
                    results_per_cluster[cluster_idx]["num_err"] += 1
                results_per_cluster[cluster_idx]["total"] += 1
                cluster_idx += 1

                if cluster_idx >= len(clusters):
                    break

    for c in results_per_cluster:
        if c["total"] != len(all_hashes):
            raise RuntimeError(
                f"Cluster {c['cluster_id']} has {c['total']} total judgments, expected {len(all_hashes)}"
            )

    with open(output_file, "w", encoding="utf-8") as f:
        final_results = {
            "judge_model": model_name,
            "cluster_path": cluster_path,
            "min_cluster_specificity": min_cluster_specificity,
            "judging_results": {
                r["cluster_id"]: {
                    "cluster_hypothesis": r["cluster_hypothesis"],
                    "cluster_model_a_percentage": r["cluster_model_a_percentage"],
                    "cluster_model_b_percentage": r["cluster_model_b_percentage"],
                    "pct_model_a": r["num_model_a"] / r["total"] if r["total"] > 0 else 0.0,
                    "pct_model_b": r["num_model_b"] / r["total"] if r["total"] > 0 else 0.0,
                    "pct_na": r["num_na"] / r["total"] if r["total"] > 0 else 0.0,
                    "pct_err": r["num_err"] / r["total"] if r["total"] > 0 else 0.0,
                    "pct_missing": r["num_missing"] / r["total"] if r["total"] > 0 else 0.0,
                    "total": r["total"],
                }
                for r in results_per_cluster
            },
        }
        json.dump(final_results, f, indent=4)

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="google/gemini-2.5-flash", help="Name of LLM to verify responses from"
    )
    parser.add_argument(
        "--open_router_api_key", type=str, default=None, help="API key for OpenRouter if using OpenRouter mode"
    )
    parser.add_argument(
        "--cluster_path", type=str, required=True, help="Path to file containing clusters with hypotheses"
    )
    parser.add_argument("--model_a_responses", type=str, required=True, help="Path to model A responses JSONL file")
    parser.add_argument("--model_b_responses", type=str, required=True, help="Path to model B responses JSONL file")
    parser.add_argument(
        "--min_cluster_specificity", type=float, default=None, help="Minimum cluster specificity to include"
    )
    parser.add_argument("--output_file", type=str, default="judging_results.json", help="File to save judging results")
    parser.add_argument(
        "--limit_samples", type=int, default=None, help="Max samples to judge (default: use all shared responses)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample shuffling")
    parser.add_argument(
        "--hypotheses_per_prompt",
        type=int,
        default=10,
        help="Hypotheses per API call (1-10). 10=cheapest, 1=most accurate",
    )
    args = parser.parse_args()

    judge_hypotheses_batched(
        model_name=args.model_name,
        cluster_path=args.cluster_path,
        model_a_responses=args.model_a_responses,
        model_b_responses=args.model_b_responses,
        output_file=args.output_file,
        open_router_api_key=args.open_router_api_key,
        min_cluster_specificity=args.min_cluster_specificity,
        limit_samples=args.limit_samples,
        seed=args.seed,
        hypotheses_per_prompt=args.hypotheses_per_prompt,
    )
