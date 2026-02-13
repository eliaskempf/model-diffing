import argparse
import asyncio
import json

import tqdm

from model_diffing.autoraters import DEFAULT_JUDGES
from model_diffing.model_cached import CachedModelWrapper
from model_diffing.parsing import parse_llm_json
from model_diffing.prompts import load_prompts
from model_diffing.utils import batch_iterable

_prompts = load_prompts("autoraters/abstraction")
SYSTEM_PROMPT = _prompts["system_prompt"]
USER_PROMPT_TEMPLATE = _prompts["user_prompt_template"]


def try_parsing_response(response: str) -> dict:
    try:
        return parse_llm_json(response)
    except json.JSONDecodeError:
        print(response)
        return {"error": "JSONDecodeError"}


def aggregate_responses(judge_responses: list[dict]) -> dict:
    """Aggregate multiple judge responses by averaging scores."""
    valid_responses = [r for r in judge_responses if "error" not in r]

    if not valid_responses:
        return {"error": "All judges failed to produce valid responses"}

    # Average the main score
    avg_score = sum(r["score"] for r in valid_responses) / len(valid_responses)

    # Average the signal scores
    signal_keys = ["scope", "generality", "conceptual_depth"]
    avg_signals = {}
    for key in signal_keys:
        values = [r["signals"][key] for r in valid_responses if "signals" in r and key in r["signals"]]
        if values:
            avg_signals[key] = sum(values) / len(values)

    return {
        "score": avg_score,
        "score_rounded": round(avg_score),
        "signals": avg_signals,
        "num_valid_judges": len(valid_responses),
        "num_total_judges": len(judge_responses),
    }


def run_judge(
    judge_name: str,
    prompts: list[list[dict]],
    api_key: str,
    batch_size: int,
    seed: int | None = None,
) -> list[dict]:
    """Run a single judge on all prompts and return responses."""
    autorater = CachedModelWrapper(
        model_name=judge_name,
        api_key=api_key,
    )

    responses = []
    approx_num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch in tqdm.tqdm(
        batch_iterable(prompts, batch_size),
        total=approx_num_batches,
        desc=f"Rating with {judge_name}",
    ):
        batch_responses = asyncio.run(
            autorater.generate_async(
                batch,
                max_new_tokens=1024,
                enable_thinking=False,
                seed=seed,
            )
        )
        for response in batch_responses:
            responses.append(try_parsing_response(response))

    return responses


def rate_hypotheses(
    hypotheses_file: str,
    open_router_api_key: str | None = None,
    judges: list[str] | None = None,
    batch_size: int = 4,
    seed: int = 42,
    output_file: str | None = None,
) -> list[dict]:
    """Rate hypotheses for abstraction level using multiple LLM judges.

    Args:
        hypotheses_file: Path to JSON with judging_results containing cluster_hypothesis per cluster.
        open_router_api_key: OpenRouter API key (defaults to env var).
        judges: List of judge model names (defaults to DEFAULT_JUDGES).
        batch_size: Batch size for API calls.
        seed: Random seed.
        output_file: Where to write results. None = derive from hypotheses_file.

    Returns:
        List of dicts with cluster_id, hypothesis, aggregated scores, individual judge responses.
    """
    if judges is None:
        judges = DEFAULT_JUDGES
    print(f"Using {len(judges)} judges: {judges}")

    if not hypotheses_file.endswith(".json"):
        raise ValueError(f"hypotheses_file must end with .json, got: {hypotheses_file}")

    with open(hypotheses_file, encoding="utf-8") as f:
        hypotheses_data = json.load(f)

    # Preserve cluster IDs for later mapping, filtering out invalid entries
    all_cluster_ids = list(hypotheses_data["judging_results"].keys())
    all_hypotheses = [hypotheses_data["judging_results"][cid]["cluster_hypothesis"] for cid in all_cluster_ids]

    # Filter out invalid hypotheses while keeping cluster_ids aligned
    paired = [(cid, h) for cid, h in zip(all_cluster_ids, all_hypotheses) if h != "ext"]
    if not paired:
        print("No valid hypotheses found.")
        return []
    cluster_ids, hypotheses = zip(*paired)

    def adjust_hypothesis_format(hypothesis: str) -> str:
        if hypothesis.startswith("Hypothesis: <MODEL>"):
            return hypothesis[len("Hypothesis: ") :]
        elif hypothesis.startswith("Hypothesis: The model") or hypothesis.startswith("Hypothesis: MODEL"):
            return hypothesis.replace("Hypothesis: The model", "<MODEL>", 1)
        elif hypothesis.startswith("This response"):
            raise ValueError("Unexpected hypothesis format starting with 'This response'")
        else:
            raise ValueError(f"Unexpected hypothesis format: {hypothesis}")

    hypotheses = [adjust_hypothesis_format(hypothesis) for hypothesis in hypotheses]

    prompts = [USER_PROMPT_TEMPLATE.format(candidate_hypothesis=hypothesis) for hypothesis in hypotheses]
    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    # Collect responses from all judges
    all_judge_responses: dict[str, list[dict]] = {}
    for judge_name in judges:
        print(f"\n--- Running judge: {judge_name} ---")
        all_judge_responses[judge_name] = run_judge(
            judge_name=judge_name,
            prompts=prompts,
            api_key=open_router_api_key,
            batch_size=batch_size,
            seed=seed,
        )

    # Build output with individual and aggregated results
    output_data = []
    for idx, (cid, hypothesis) in enumerate(zip(cluster_ids, hypotheses)):
        individual_responses = {judge_name: all_judge_responses[judge_name][idx] for judge_name in judges}
        aggregated = aggregate_responses(list(individual_responses.values()))

        output_data.append(
            {
                "cluster_id": cid,
                "hypothesis": hypothesis,
                "aggregated": aggregated,
                "individual_judges": individual_responses,
            }
        )

    if output_file is None:
        output_file = hypotheses_file.replace(".json", "_abstraction_responses.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nWrote abstraction level responses to {output_file}")

    return output_data


def main(args: argparse.Namespace):
    rate_hypotheses(
        hypotheses_file=args.hypotheses_file,
        open_router_api_key=args.open_router_api_key,
        judges=args.judges,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges",
        type=str,
        nargs="+",
        default=None,
        help=f"List of judge model names. Defaults to: {DEFAULT_JUDGES}",
    )
    parser.add_argument("--open_router_api_key", type=str, default=None, help="API key for OpenRouter")
    parser.add_argument(
        "--hypotheses_file", type=str, required=True, help="Path to JSON file containing list of hypotheses to rate"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generating autorater responses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)
