#!/usr/bin/env python3
"""
Aggregate sharded output files from generate_gender_rollouts.py.

This script discovers shard files, validates completeness and integrity,
and merges them into single output files.

Usage:
    python aggregate_shards.py <directory> --model_name <model> [options]

Examples:
    # Basic merge (auto-detect shards):
    python aggregate_shards.py output/gender-rollouts --model_name llama-2-7b

    # With full validation against original CSV:
    python aggregate_shards.py output/gender-rollouts --model_name llama-2-7b \
        --prompts_csv data.csv --min_score 4.0 --strict
"""

import argparse
import json
import re
import sys
from pathlib import Path

from model_diffing.data import load_prompts_from_csv


def discover_shards(directory: Path, model_name: str) -> dict[int, dict]:
    """Discover all shard files for a given model.

    Returns:
        Dict mapping shard_id to dict with file paths:
        {
            0: {"rollouts": Path, "judge_results": Path, "stats": Path, "num_shards": 10},
            1: {...},
            ...
        }
    """
    model_name_safe = model_name.replace("/", "_")

    # Pattern to match shard files: {model}_shard{id}of{total}_{type}.json
    pattern = re.compile(rf"^{re.escape(model_name_safe)}_shard(\d+)of(\d+)_(rollouts|judge_results|stats)\.json$")

    shards = {}
    for file_path in directory.iterdir():
        match = pattern.match(file_path.name)
        if match:
            shard_id = int(match.group(1))
            num_shards = int(match.group(2))
            file_type = match.group(3)

            if shard_id not in shards:
                shards[shard_id] = {"num_shards": num_shards}
            shards[shard_id][file_type] = file_path

    return shards


def validate_shards(
    shards: dict[int, dict],
    expected_num_shards: int | None = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate shard completeness and consistency.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if not shards:
        return False, ["No shard files found"]

    # Determine expected number of shards
    num_shards_values = set(s["num_shards"] for s in shards.values())
    if len(num_shards_values) > 1:
        issues.append(f"Inconsistent num_shards across files: {num_shards_values}")
        return False, issues

    detected_num_shards = num_shards_values.pop()
    if expected_num_shards is not None and detected_num_shards != expected_num_shards:
        issues.append(f"Expected {expected_num_shards} shards but files indicate {detected_num_shards}")

    num_shards = expected_num_shards or detected_num_shards

    # Check for missing shards
    expected_ids = set(range(num_shards))
    found_ids = set(shards.keys())
    missing_ids = expected_ids - found_ids

    if missing_ids:
        msg = f"Missing shards: {sorted(missing_ids)}"
        if strict:
            issues.append(msg)
        else:
            print(f"WARNING: {msg}")

    # Check each shard has all required files
    for shard_id, shard_files in shards.items():
        for file_type in ["rollouts", "judge_results", "stats"]:
            if file_type not in shard_files:
                issues.append(f"Shard {shard_id} missing {file_type} file")

    is_valid = len(issues) == 0 or (not strict and all("Missing shards" in i for i in issues))
    return is_valid, issues


def load_shard_files(shards: dict[int, dict]) -> tuple[dict, dict, dict]:
    """Load all shard files.

    Returns:
        Tuple of (rollouts_by_shard, judge_results_by_shard, stats_by_shard)
    """
    rollouts_by_shard = {}
    judge_results_by_shard = {}
    stats_by_shard = {}

    for shard_id, shard_files in sorted(shards.items()):
        if "rollouts" in shard_files:
            with open(shard_files["rollouts"], encoding="utf-8") as f:
                rollouts_by_shard[shard_id] = json.load(f)

        if "judge_results" in shard_files:
            with open(shard_files["judge_results"], encoding="utf-8") as f:
                judge_results_by_shard[shard_id] = json.load(f)

        if "stats" in shard_files:
            with open(shard_files["stats"], encoding="utf-8") as f:
                stats_by_shard[shard_id] = json.load(f)

    return rollouts_by_shard, judge_results_by_shard, stats_by_shard


def check_duplicates(data_by_shard: dict[int, dict], data_type: str) -> list[str]:
    """Check for duplicate prompt IDs across shards.

    Returns list of error messages.
    """
    errors = []
    seen_ids = {}  # prompt_id -> shard_id where first seen

    for shard_id, data in sorted(data_by_shard.items()):
        # Handle different data structures
        if data_type == "rollouts":
            prompt_ids = data.get("data", {}).keys()
        else:  # judge_results or per_prompt in stats
            prompt_ids = data.keys() if isinstance(data, dict) else []

        for prompt_id in prompt_ids:
            if prompt_id in seen_ids:
                errors.append(f"Duplicate prompt ID '{prompt_id}' in shards {seen_ids[prompt_id]} and {shard_id}")
            else:
                seen_ids[prompt_id] = shard_id

    return errors


def merge_rollouts(rollouts_by_shard: dict[int, dict]) -> dict:
    """Merge rollout data from all shards."""
    merged = {
        "metadata": {},
        "data": {},
    }

    for _shard_id, rollouts in sorted(rollouts_by_shard.items()):
        # Merge metadata (should be identical, take from first shard)
        if not merged["metadata"] and "metadata" in rollouts:
            merged["metadata"] = rollouts["metadata"].copy()

        # Merge data
        for prompt_id, prompt_data in rollouts.get("data", {}).items():
            merged["data"][prompt_id] = prompt_data

    return merged


def merge_judge_results(judge_results_by_shard: dict[int, dict]) -> dict:
    """Merge judge results from all shards."""
    merged = {}

    for _shard_id, judge_results in sorted(judge_results_by_shard.items()):
        for prompt_id, results in judge_results.items():
            merged[prompt_id] = results

    return merged


def recompute_statistics(merged_judge_results: dict, merged_rollouts: dict, stats_by_shard: dict) -> dict:
    """Recompute aggregate statistics from merged data.

    Uses per_prompt stats from shards and recomputes aggregate.
    """
    # Get metadata from first shard
    metadata = {}
    for shard_stats in stats_by_shard.values():
        if "metadata" in shard_stats:
            metadata = shard_stats["metadata"].copy()
            break

    # Merge per_prompt stats from all shards
    per_prompt = {}
    for _shard_id, shard_stats in sorted(stats_by_shard.items()):
        for prompt_id, pstats in shard_stats.get("per_prompt", {}).items():
            per_prompt[prompt_id] = pstats

    # Recompute aggregate from per_prompt data
    all_scores = []
    all_english_scores = []

    for _prompt_id, pstats in per_prompt.items():
        n_valid = pstats.get("n_valid", 0)
        sum_assumptions = pstats.get("sum_assumptions", 0)
        mean_rate = pstats.get("mean_assumption_rate")

        if n_valid > 0 and mean_rate is not None:
            # Reconstruct individual scores for aggregate calculation
            all_scores.extend([1] * sum_assumptions)
            all_scores.extend([0] * (n_valid - sum_assumptions))

        n_english_valid = pstats.get("n_english_valid", 0)
        english_rate = pstats.get("english_rate")
        if n_english_valid > 0 and english_rate is not None:
            english_count = round(english_rate * n_english_valid)
            all_english_scores.extend([1] * english_count)
            all_english_scores.extend([0] * (n_english_valid - english_count))

    n_total_valid = len(all_scores)
    n_total_english_valid = len(all_english_scores)

    aggregate = {
        "total_rollouts": sum(pstats.get("n_rollouts", 0) for pstats in per_prompt.values()),
        "total_valid": n_total_valid,
        "total_english_valid": n_total_english_valid,
    }

    if n_total_valid > 0:
        aggregate["overall_assumption_rate"] = sum(all_scores) / n_total_valid
        aggregate["total_assumptions"] = sum(all_scores)
    else:
        aggregate["overall_assumption_rate"] = None
        aggregate["total_assumptions"] = 0

    if n_total_english_valid > 0:
        aggregate["overall_english_rate"] = sum(all_english_scores) / n_total_english_valid
        aggregate["total_english"] = sum(all_english_scores)
    else:
        aggregate["overall_english_rate"] = None
        aggregate["total_english"] = 0

    # Add merge info to metadata
    metadata["merged_from_shards"] = len(stats_by_shard)

    return {
        "metadata": metadata,
        "per_prompt": per_prompt,
        "aggregate": aggregate,
    }


def validate_against_csv(
    merged_prompt_ids: set[str],
    csv_path: str,
    min_score: float,
    prompt_column: int,
    id_column: int,
    score_column: int,
) -> list[str]:
    """Validate that all expected prompts from CSV are present.

    Returns list of error messages.
    """
    expected_prompts = load_prompts_from_csv(
        csv_path,
        prompt_column=prompt_column,
        id_column=id_column,
        score_column=score_column,
        min_score=min_score,
    )
    expected_ids = set(p["id"] for p in expected_prompts)

    errors = []

    missing = expected_ids - merged_prompt_ids
    if missing:
        errors.append(f"Missing {len(missing)} prompts from CSV: {sorted(list(missing))[:10]}...")

    extra = merged_prompt_ids - expected_ids
    if extra:
        errors.append(f"Found {len(extra)} unexpected prompts not in CSV: {sorted(list(extra))[:10]}...")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Aggregate sharded output files from generate_gender_rollouts.py")
    parser.add_argument("directory", type=str, help="Directory containing shard files")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to find shards for")
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Expected number of shards (auto-detected if not provided)"
    )
    parser.add_argument(
        "--prompts_csv", type=str, default=None, help="Original CSV file for coverage validation (optional)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Score filter used during generation (required if --prompts_csv specified)",
    )
    parser.add_argument("--prompt_column", type=int, default=2, help="0-indexed column for prompts (default: 2)")
    parser.add_argument("--id_column", type=int, default=0, help="0-indexed column for IDs (default: 0)")
    parser.add_argument("--score_column", type=int, default=1, help="0-indexed column for scores (default: 1)")
    parser.add_argument("--strict", action="store_true", help="Fail on missing shards (default: warn and continue)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged files (default: same as input directory)",
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else directory

    # Discover shards
    print(f"Discovering shards for model '{args.model_name}' in {directory}...")
    shards = discover_shards(directory, args.model_name)

    if not shards:
        print(f"Error: No shard files found for model '{args.model_name}'")
        return 1

    print(f"Found {len(shards)} shard(s): {sorted(shards.keys())}")

    # Validate shards
    print("\nValidating shards...")
    is_valid, issues = validate_shards(shards, args.num_shards, args.strict)

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        if not is_valid:
            return 1

    # Load shard files
    print("\nLoading shard files...")
    rollouts_by_shard, judge_results_by_shard, stats_by_shard = load_shard_files(shards)

    # Check for duplicates
    print("Checking for duplicate prompt IDs...")
    duplicate_errors = []
    if rollouts_by_shard:
        duplicate_errors.extend(check_duplicates(rollouts_by_shard, "rollouts"))
    if judge_results_by_shard:
        duplicate_errors.extend(check_duplicates(judge_results_by_shard, "judge_results"))

    if duplicate_errors:
        for error in duplicate_errors:
            print(f"  ERROR: {error}")

    # Merge data
    print("\nMerging data...")
    merged_rollouts = merge_rollouts(rollouts_by_shard) if rollouts_by_shard else {}
    merged_judge_results = merge_judge_results(judge_results_by_shard) if judge_results_by_shard else {}
    merged_stats = recompute_statistics(merged_judge_results, merged_rollouts, stats_by_shard) if stats_by_shard else {}

    # Validate against CSV if provided
    if args.prompts_csv:
        print(f"\nValidating against CSV: {args.prompts_csv}")
        merged_prompt_ids = set(merged_rollouts.get("data", {}).keys())
        csv_errors = validate_against_csv(
            merged_prompt_ids,
            args.prompts_csv,
            args.min_score,
            args.prompt_column,
            args.id_column,
            args.score_column,
        )
        if csv_errors:
            for error in csv_errors:
                print(f"  ERROR: {error}")
            if args.strict:
                return 1

    # Save merged files
    model_name_safe = args.model_name.replace("/", "_")

    if merged_rollouts:
        rollouts_path = output_dir / f"{model_name_safe}_rollouts.json"
        with open(rollouts_path, "w", encoding="utf-8") as f:
            json.dump(merged_rollouts, f, indent=2, ensure_ascii=False)
        print(f"\nSaved merged rollouts: {rollouts_path}")
        print(f"  Total prompts: {len(merged_rollouts.get('data', {}))}")

    if merged_judge_results:
        judge_path = output_dir / f"{model_name_safe}_judge_results.json"
        with open(judge_path, "w", encoding="utf-8") as f:
            json.dump(merged_judge_results, f, indent=2, ensure_ascii=False)
        print(f"Saved merged judge results: {judge_path}")

    if merged_stats:
        stats_path = output_dir / f"{model_name_safe}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(merged_stats, f, indent=2, ensure_ascii=False)
        print(f"Saved merged stats: {stats_path}")

        # Print summary
        agg = merged_stats.get("aggregate", {})
        print("\n" + "=" * 50)
        print("MERGED SUMMARY")
        print("=" * 50)
        print(f"Total prompts: {len(merged_stats.get('per_prompt', {}))}")
        print(f"Total rollouts: {agg.get('total_rollouts', 'N/A')}")
        print(f"Valid judgments: {agg.get('total_valid', 'N/A')}")
        if agg.get("overall_assumption_rate") is not None:
            print(f"Overall assumption rate: {agg['overall_assumption_rate']:.2%}")
        print(f"Merged from {merged_stats.get('metadata', {}).get('merged_from_shards', 'N/A')} shards")

    print("\nAggregation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
