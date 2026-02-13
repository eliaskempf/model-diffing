"""Evaluation pipeline runner.

Runs hypothesis judging, autoraters, and evaluation in sequence. Supports two modes:

1. From scratch: provide cluster_path + train/test response files to run judging first
2. Pre-computed: provide train/test judge result files to skip judging

Usage:
    # From scratch — judge hypotheses, run autoraters, then evaluate
    uv run python scripts/run_eval.py \
        --cluster_path path/to/clusters.jsonl \
        --model_a_responses path/to/model_a/train_responses.jsonl \
        --model_b_responses path/to/model_b/train_responses.jsonl \
        --model_a_test_responses path/to/model_a/test_responses.jsonl \
        --model_b_test_responses path/to/model_b/test_responses.jsonl \
        --output_dir output/eval_results

    # From pre-computed judge results
    uv run python scripts/run_eval.py \
        --train_judge_results path/to/train_judging_results.json \
        --test_judge_results path/to/test_judging_results.json \
        --output_dir output/eval_results

    # Skip autoraters (faster, no LLM calls for rating)
    uv run python scripts/run_eval.py \
        --train_judge_results path/to/train.json \
        --test_judge_results path/to/test.json \
        --skip_autoraters \
        --output_dir output/eval_results
"""

import os
import time

from model_diffing.evaluation.evaluate_hypotheses import evaluate_all
from model_diffing.evaluation.judge_batched import judge_hypotheses_batched


def _run_autoraters(
    train_results_file: str,
    output_dir: str,
    force: bool = False,
    judges: list[str] | None = None,
    batch_size: int = 4,
    api_key: str | None = None,
    interestingness_file: str | None = None,
    abstraction_file: str | None = None,
) -> tuple[str | None, str | None]:
    """Run autoraters on train results, returning paths to autorater output files.

    Returns (interestingness_file, abstraction_file) — either the pre-computed paths
    or the paths to newly generated / cached-on-disk results.
    """
    from model_diffing.autoraters.autorater_abstraction import rate_hypotheses as rate_abstraction
    from model_diffing.autoraters.autorater_interestingness import rate_hypotheses as rate_interestingness

    # Interestingness
    if interestingness_file:
        print(f"  Using pre-computed interestingness scores: {interestingness_file}")
        int_file = interestingness_file
    else:
        int_file = os.path.join(output_dir, "interestingness_scores.json")
        if os.path.exists(int_file) and not force:
            print(f"  Interestingness scores already exist: {int_file} (use --force_autoraters to regenerate)")
        else:
            print("  Running interestingness autorater...")
            rate_interestingness(
                hypotheses_file=train_results_file,
                open_router_api_key=api_key,
                judges=judges,
                batch_size=batch_size,
                output_file=int_file,
            )

    # Abstraction
    if abstraction_file:
        print(f"  Using pre-computed abstraction scores: {abstraction_file}")
        abs_file = abstraction_file
    else:
        abs_file = os.path.join(output_dir, "abstraction_scores.json")
        if os.path.exists(abs_file) and not force:
            print(f"  Abstraction scores already exist: {abs_file} (use --force_autoraters to regenerate)")
        else:
            print("  Running abstraction autorater...")
            rate_abstraction(
                hypotheses_file=train_results_file,
                open_router_api_key=api_key,
                judges=judges,
                batch_size=batch_size,
                output_file=abs_file,
            )

    return int_file, abs_file


def main(args):
    start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Judge hypotheses (or use pre-computed results)
    has_precomputed = args.train_judge_results or args.test_judge_results
    has_from_scratch = args.cluster_path or args.model_a_responses or args.model_b_responses

    if has_precomputed and has_from_scratch:
        raise ValueError("Cannot mix pre-computed judge results with cluster/response args. Use one mode or the other.")
    if bool(args.train_judge_results) != bool(args.test_judge_results):
        raise ValueError("Provide both --train_judge_results and --test_judge_results, or neither.")

    if args.train_judge_results and args.test_judge_results:
        print("Using pre-computed judge results.")
        train_results_file = args.train_judge_results
        test_results_file = args.test_judge_results
    elif args.cluster_path:
        if not args.model_a_responses or not args.model_b_responses:
            raise ValueError("From-scratch judging requires --model_a_responses and --model_b_responses")
        if not args.model_a_test_responses or not args.model_b_test_responses:
            raise ValueError("From-scratch judging requires --model_a_test_responses and --model_b_test_responses")

        print(f"Running hypothesis judging ({args.hypotheses_per_prompt} hypotheses/prompt)...")

        train_results_file = os.path.join(args.output_dir, "judging_results_train.json")
        judge_hypotheses_batched(
            model_name=args.judge_model_name,
            cluster_path=args.cluster_path,
            model_a_responses=args.model_a_responses,
            model_b_responses=args.model_b_responses,
            output_file=train_results_file,
            open_router_api_key=args.open_router_api_key,
            min_cluster_specificity=args.min_cluster_specificity,
            limit_samples=args.limit_samples,
            seed=42,
            hypotheses_per_prompt=args.hypotheses_per_prompt,
        )

        test_results_file = os.path.join(args.output_dir, "judging_results_test.json")
        judge_hypotheses_batched(
            model_name=args.judge_model_name,
            cluster_path=args.cluster_path,
            model_a_responses=args.model_a_test_responses,
            model_b_responses=args.model_b_test_responses,
            output_file=test_results_file,
            open_router_api_key=args.open_router_api_key,
            min_cluster_specificity=args.min_cluster_specificity,
            limit_samples=args.limit_samples,
            seed=42,
            hypotheses_per_prompt=args.hypotheses_per_prompt,
        )
        print(f"  Train results: {train_results_file}")
        print(f"  Test results: {test_results_file}")
    else:
        raise ValueError(
            "Provide either --train_judge_results + --test_judge_results, or --cluster_path + response files."
        )

    # Step 2: Run autoraters (unless skipped)
    interestingness_file = None
    abstraction_file = None

    if not args.skip_autoraters:
        print("\nRunning autoraters...")
        interestingness_file, abstraction_file = _run_autoraters(
            train_results_file=train_results_file,
            output_dir=args.output_dir,
            force=args.force_autoraters,
            judges=args.autorater_judges,
            batch_size=args.autorater_batch_size,
            api_key=args.open_router_api_key,
            interestingness_file=args.interestingness_file,
            abstraction_file=args.abstraction_file,
        )
    else:
        print("\nSkipping autoraters.")

    # Step 3: Evaluate hypotheses
    print("\nEvaluating hypotheses...")
    output_file = os.path.join(args.output_dir, "hypothesis_evaluation_results.json")
    experiment = {
        "name": args.experiment_name,
        "method": args.method,
        "key": f"{args.experiment_name}_{args.method}",
        "train_file": train_results_file,
        "test_file": test_results_file,
    }
    if interestingness_file:
        experiment["interestingness_file"] = interestingness_file
    if abstraction_file:
        experiment["abstraction_file"] = abstraction_file

    evaluate_all(
        experiments=[experiment],
        alpha=args.alpha,
        output_file=output_file,
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Results in {args.output_dir}/")


if __name__ == "__main__":
    import argparse

    from model_diffing.autoraters import DEFAULT_JUDGES

    parser = argparse.ArgumentParser(description="Run the evaluation pipeline")

    # Input mode 1: from scratch
    parser.add_argument("--cluster_path", type=str, help="Path to clusters JSONL file (from diffing pipeline)")
    parser.add_argument("--model_a_responses", type=str, help="Path to model A train responses JSONL file")
    parser.add_argument("--model_b_responses", type=str, help="Path to model B train responses JSONL file")
    parser.add_argument("--model_a_test_responses", type=str, help="Path to model A test responses JSONL file")
    parser.add_argument("--model_b_test_responses", type=str, help="Path to model B test responses JSONL file")

    # Input mode 2: pre-computed
    parser.add_argument("--train_judge_results", type=str, help="Path to pre-computed train judging results JSON")
    parser.add_argument("--test_judge_results", type=str, help="Path to pre-computed test judging results JSON")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/eval_results", help="Directory for output files")

    # Judge config
    parser.add_argument(
        "--judge_model_name", type=str, default="google/gemini-2.5-flash", help="Model to use for judging"
    )
    parser.add_argument("--open_router_api_key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--min_cluster_specificity", type=float, default=0.65, help="Min cluster specificity filter")
    parser.add_argument(
        "--hypotheses_per_prompt",
        type=int,
        default=10,
        choices=range(1, 11),
        metavar="N",
        help="Hypotheses per API call (1-10). 10=cheapest, 1=most accurate (default: 10)",
    )
    parser.add_argument(
        "--limit_samples", type=int, default=None, help="Max samples for judging (default: use all shared responses)"
    )

    # Autorater config
    parser.add_argument("--skip_autoraters", action="store_true", help="Skip interestingness/abstraction autoraters")
    parser.add_argument(
        "--force_autoraters", action="store_true", help="Re-run autoraters even if output files exist on disk"
    )
    parser.add_argument(
        "--autorater_judges",
        type=str,
        nargs="+",
        default=None,
        help=f"Judge models for autoraters (default: {DEFAULT_JUDGES})",
    )
    parser.add_argument(
        "--autorater_batch_size", type=int, default=4, help="Batch size for autorater API calls (default: 4)"
    )
    parser.add_argument(
        "--interestingness_file", type=str, default=None, help="Pre-computed interestingness scores JSON"
    )
    parser.add_argument("--abstraction_file", type=str, default=None, help="Pre-computed abstraction scores JSON")

    # Evaluation config
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Name for this experiment")
    parser.add_argument("--method", type=str, default="llm", choices=["llm", "sae"], help="Method used for hypotheses")
    parser.add_argument("--alpha", type=float, default=0.01, help="Min margin for judging significance")

    args = parser.parse_args()
    main(args)
