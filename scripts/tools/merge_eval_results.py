"""Merge individual experiment evaluation JSONs into a single multi-experiment file.

Plotting scripts (e.g. plot_hypothesis_results.py) expect a JSON file keyed by all
experiment names (e.g. qwen_em_llm, gemini_sae, etc.). run_eval.py produces one file
per experiment. This script merges them.

Usage:
    uv run python scripts/tools/merge_eval_results.py \
        --input output/qwen_em_llm/hypothesis_evaluation_results.json \
        --input output/qwen_em_sae/hypothesis_evaluation_results.json \
        --input output/gemini_llm/hypothesis_evaluation_results.json \
        --input output/gemini_sae/hypothesis_evaluation_results.json \
        --input output/gemma_gender_llm/hypothesis_evaluation_results.json \
        --input output/gemma_gender_sae/hypothesis_evaluation_results.json \
        --output output/all_hypothesis_evaluation_results.json
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge experiment evaluation results into a single JSON file")
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        required=True,
        dest="input_files",
        help="Path to an experiment evaluation JSON (can be specified multiple times)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged JSON",
    )
    args = parser.parse_args()

    merged = {}
    for filepath in args.input_files:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for key, value in data.items():
            if key in merged:
                print(f"ERROR: Duplicate experiment key '{key}' found in {filepath}", file=sys.stderr)
                print("  Already loaded from a previous input file.", file=sys.stderr)
                raise SystemExit(1)
            merged[key] = value
            print(f"  Loaded {key} from {filepath} ({value.get('num_hypotheses', '?')} hypotheses)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged {len(merged)} experiments into {args.output}")


if __name__ == "__main__":
    main()
