"""Compare two evaluation result JSONs and highlight differences.

Usage:
    uv run python scripts/tools/compare_eval_results.py \
        output/eval_recompute/all_hypothesis_evaluation_results.json \
        output/old_results/hypothesis_evaluation_results.json
"""

import argparse
import json

IGNORED_KEYS = {"train_file", "test_file"}


def compare_values(path: str, a, b, rtol: float = 1e-6) -> list[str]:
    """Recursively compare two JSON values, returning a list of difference descriptions."""
    diffs = []

    if type(a) is not type(b):
        diffs.append(f"  {path}: type mismatch: {type(a).__name__} vs {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        all_keys = sorted((set(a.keys()) | set(b.keys())) - IGNORED_KEYS)
        for key in all_keys:
            child_path = f"{path}.{key}" if path else key
            if key not in a:
                diffs.append(f"  {child_path}: only in B")
            elif key not in b:
                diffs.append(f"  {child_path}: only in A")
            else:
                diffs.extend(compare_values(child_path, a[key], b[key], rtol))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"  {path}: list length {len(a)} vs {len(b)}")
        for i in range(min(len(a), len(b))):
            diffs.extend(compare_values(f"{path}[{i}]", a[i], b[i], rtol))
    elif isinstance(a, float):
        if a != b and (a == 0 or abs(a - b) / max(abs(a), abs(b)) > rtol):
            diffs.append(f"  {path}: {a} vs {b} (delta={a - b:+.8f})")
    elif a != b:
        a_str = str(a) if len(str(a)) <= 80 else str(a)[:77] + "..."
        b_str = str(b) if len(str(b)) <= 80 else str(b)[:77] + "..."
        diffs.append(f"  {path}: {a_str!r} vs {b_str!r}")

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation result JSONs")
    parser.add_argument("file_a", help="First evaluation results JSON (A)")
    parser.add_argument("file_b", help="Second evaluation results JSON (B)")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for float comparison")
    args = parser.parse_args()

    with open(args.file_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.file_b, encoding="utf-8") as f:
        b = json.load(f)

    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))
    common = sorted(set(a.keys()) & set(b.keys()))

    if only_a:
        print(f"Experiments only in A: {only_a}")
    if only_b:
        print(f"Experiments only in B: {only_b}")

    any_diff = bool(only_a or only_b)

    total_diffs = 0
    for key in common:
        diffs = compare_values(key, a[key], b[key], rtol=args.rtol)
        if diffs:
            any_diff = True
            total_diffs += len(diffs)
            print(f"\n{key}: {len(diffs)} difference(s)")
            for d in diffs:
                print(d)
        else:
            print(f"{key}: identical")

    if not any_diff:
        print("\nResults are identical.")
    else:
        print(f"\nResults differ. {total_diffs} total difference(s).")


if __name__ == "__main__":
    main()
