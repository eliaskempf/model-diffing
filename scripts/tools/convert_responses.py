"""
Convert JSONL response files to CSV format for easier LLM parsing.

Each row contains a prompt-response pair.
"""

import argparse
import csv
import json
from pathlib import Path


def convert_to_csv(input_file: str, output_file: str) -> int:
    """
    Convert a JSONL response file to CSV format.

    Args:
        input_file: Path to the input JSONL file.
        output_file: Path to the output CSV file.

    Returns:
        Number of rows written.
    """
    rows_written = 0

    with open(input_file, encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(["prompt", "response"])

        for line in f_in:
            entry = json.loads(line)
            # Each line is {hash: {conversation: [...], response: "..."}}
            data = next(iter(entry.values()))

            # Extract prompt (concatenate all user messages)
            conversation = data["conversation"]
            prompt = " ".join(turn["content"] for turn in conversation if turn["role"] == "user")

            response = data["response"]

            writer.writerow([prompt, response])
            rows_written += 1

    return rows_written


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL response files to CSV format.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL response file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: input file with .csv extension)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".csv")

    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")

    rows = convert_to_csv(str(input_path), str(output_path))

    print(f"Wrote {rows} prompt-response pairs to {output_path}")


if __name__ == "__main__":
    main()
