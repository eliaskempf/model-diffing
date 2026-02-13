"""Compare content types (tables, LaTeX, code blocks, lists) between two response JSONL files."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_diffing.content_detection import (
    contains_latex,
    contains_table,
)


def load_responses(path: str) -> list[str]:
    responses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for key in item:
                responses.append(item[key]["response"])
    return responses


def compute_stats(responses: list[str]) -> dict[str, float]:
    n = len(responses)
    return {
        "Table": 100 * sum(1 for r in responses if contains_table(r)) / n,
        "LaTeX": 100 * sum(1 for r in responses if contains_latex(r)) / n,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare content types between two response files")
    parser.add_argument("responses_a", type=str, help="Path to first JSONL responses file")
    parser.add_argument("responses_b", type=str, help="Path to second JSONL responses file")
    parser.add_argument("--label_a", type=str, default="Model A", help="Label for first model")
    parser.add_argument("--label_b", type=str, default="Model B", help="Label for second model")
    parser.add_argument("--output", type=str, default="content_type_comparison.png", help="Output plot filename")
    args = parser.parse_args()

    stats_a = compute_stats(load_responses(args.responses_a))
    stats_b = compute_stats(load_responses(args.responses_b))

    for name, val in stats_a.items():
        print(f"{name} in {args.label_a}: {val:.1f}%")
    for name, val in stats_b.items():
        print(f"{name} in {args.label_b}: {val:.1f}%")

    df = pd.DataFrame(
        [{"Model": args.label_a, "type": t, "value": v} for t, v in stats_a.items()]
        + [{"Model": args.label_b, "type": t, "value": v} for t, v in stats_b.items()]
    )

    plt.figure(figsize=(7.5, 4))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df, x="type", y="value", hue="Model")
    plt.ylabel("Presence in Responses (%)\n")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
