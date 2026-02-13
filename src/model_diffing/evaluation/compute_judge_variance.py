"""
Compute variance statistics for autorater outputs.

For each hypothesis, computes the variance in score across judge models.
Then outputs mean, median, and max variance across all hypotheses.
Processes interestingness and abstraction results separately, grouped by method (SAE/LLM).
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def compute_variance(scores: list[float]) -> float:
    """Compute variance of scores. Returns 0 if fewer than 2 scores."""
    if len(scores) < 2:
        return 0.0
    return statistics.variance(scores)


def extract_scores_per_hypothesis(data: list[dict]) -> list[list[float]]:
    """Extract per-judge scores for each hypothesis."""
    scores_per_hypothesis = []
    for entry in data:
        individual_judges = entry.get("individual_judges", {})
        scores = []
        for _judge_name, judge_response in individual_judges.items():
            if "error" not in judge_response and "score" in judge_response:
                scores.append(judge_response["score"])
        scores_per_hypothesis.append(scores)
    return scores_per_hypothesis


def compute_variance_stats(scores_per_hypothesis: list[list[float]]) -> dict:
    """Compute variance for each hypothesis, then aggregate across hypotheses."""
    variances = []
    for scores in scores_per_hypothesis:
        if len(scores) >= 2:
            variances.append(compute_variance(scores))

    assert len(set(len(scores) for scores in scores_per_hypothesis)) == 1, (
        "Inconsistent number of scores per hypothesis"
    )
    print(f"Each hypothesis has {len(scores_per_hypothesis[0])} judge scores.")
    if not variances:
        return {
            "num_hypotheses": len(scores_per_hypothesis),
            "num_hypotheses_with_variance": 0,
            "mean_variance": None,
            "median_variance": None,
            "max_variance": None,
        }

    return {
        "num_hypotheses": len(scores_per_hypothesis),
        "num_hypotheses_with_variance": len(variances),
        "mean_variance": statistics.mean(variances),
        "median_variance": statistics.median(variances),
        "max_variance": max(variances),
    }


def find_autorater_files(directory: Path) -> dict[str, dict[str, Path]]:
    """
    Find autorater output files in directory, grouped by method.

    Returns dict mapping method -> {"interestingness": path, "abstraction": path}
    """
    files_by_method = defaultdict(dict)

    for path in directory.glob("*_autorater_responses.json"):
        # Extract method from filename (everything before _autorater_responses.json)
        method = path.name.replace("_autorater_responses.json", "")
        files_by_method[method]["interestingness"] = path

    for path in directory.glob("*_abstraction_responses.json"):
        # Extract method from filename (everything before _abstraction_responses.json)
        method = path.name.replace("_abstraction_responses.json", "")
        files_by_method[method]["abstraction"] = path

    return dict(files_by_method)


def process_file(file_path: Path) -> dict:
    """Load file and compute variance stats."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    scores_per_hypothesis = extract_scores_per_hypothesis(data)
    return compute_variance_stats(scores_per_hypothesis)


def main(args: argparse.Namespace):
    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files_by_method = find_autorater_files(directory)

    if not files_by_method:
        print(f"No autorater output files found in {directory}")
        print("Expected files matching *_autorater_responses.json or *_abstraction_responses.json")
        return

    print(f"Found {len(files_by_method)} method(s): {list(files_by_method.keys())}\n")

    results = {}
    for method, files in sorted(files_by_method.items()):
        print(f"=== Method: {method} ===")
        results[method] = {}

        for rating_type in ["interestingness", "abstraction"]:
            if rating_type not in files:
                print(f"  {rating_type}: not found")
                continue

            file_path = files[rating_type]
            stats = process_file(file_path)
            results[method][rating_type] = stats

            print(f"  {rating_type}:")
            print(f"    File: {file_path.name}")
            print(
                f"    Hypotheses: {stats['num_hypotheses']} total, {stats['num_hypotheses_with_variance']} with 2+ judges"
            )
            if stats["mean_variance"] is not None:
                print(f"    Mean variance:   {stats['mean_variance']:.4f}")
                print(f"    Median variance: {stats['median_variance']:.4f}")
                print(f"    Max variance:    {stats['max_variance']:.4f}")
            else:
                print("    No variance data (all hypotheses have <2 valid judge responses)")
        print()

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute variance statistics for autorater outputs")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing autorater output files (*_autorater_responses.json, *_abstraction_responses.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )

    args = parser.parse_args()
    main(args)
