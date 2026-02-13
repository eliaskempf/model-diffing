#!/usr/bin/env python3
"""
Analyze stats files to find top prompts by assumptions.

Usage:
    python analyze_top_assumptions.py <directory_path>

Discovers all *_stats.json files in the given directory and outputs
markdown reports with the top prompts by assumption rate, including
rollouts and judge rationales.
"""

import argparse
import json
import re
from pathlib import Path


def load_related_files(stats_path: Path) -> tuple[dict, dict, dict] | None:
    """Load stats, rollouts, and judge_results files for a given stats file.

    Returns:
        Tuple of (stats, rollouts, judge_results) or None if files missing.
    """
    # Derive related file paths from stats file name
    # e.g., model_name_stats.json -> model_name_rollouts.json, model_name_judge_results.json
    base_name = stats_path.name.replace("_stats.json", "")
    rollouts_path = stats_path.parent / f"{base_name}_rollouts.json"
    judge_results_path = stats_path.parent / f"{base_name}_judge_results.json"

    if not rollouts_path.exists():
        print(f"  Warning: Rollouts file not found: {rollouts_path}")
        return None
    if not judge_results_path.exists():
        print(f"  Warning: Judge results file not found: {judge_results_path}")
        return None

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    with open(rollouts_path, encoding="utf-8") as f:
        rollouts = json.load(f)
    with open(judge_results_path, encoding="utf-8") as f:
        judge_results = json.load(f)

    return stats, rollouts, judge_results


def get_english_only_prompts(stats: dict) -> list[tuple[str, dict]]:
    """Filter to prompts that are English-only and have valid assumption rates.

    Returns prompts sorted by assumption rate (descending).
    """
    per_prompt = stats.get("per_prompt", {})

    english_prompts = []
    for prompt_id, pstats in per_prompt.items():
        # Filter: must be English (english_rate == 1.0 or close to it)
        # and have a valid assumption rate
        english_rate = pstats.get("english_rate")
        assumption_rate = pstats.get("mean_assumption_rate")

        if english_rate is not None and english_rate >= 0.9 and assumption_rate is not None:
            english_prompts.append((prompt_id, pstats))

    # Sort by assumption rate descending
    english_prompts.sort(key=lambda x: x[1].get("mean_assumption_rate", 0), reverse=True)

    return english_prompts


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def find_evidence_matches(rollout: str, key_evidence: str) -> list[tuple[int, int]]:
    """Find all matching spans of key evidence in the rollout.

    Tries multiple strategies:
    1. Exact match of the full evidence
    2. Quoted portions within the evidence (e.g., 'text in quotes')
    3. Significant phrases (4+ words) from the evidence

    Returns list of (start, end) tuples for matches found.
    """
    if not key_evidence or not key_evidence.strip():
        return []

    rollout_lower = rollout.lower()
    matches = []

    # Strategy 1: Try full evidence (cleaned)
    key_evidence_clean = key_evidence.strip().strip("\"'")
    idx = rollout_lower.find(key_evidence_clean.lower())
    if idx != -1:
        return [(idx, idx + len(key_evidence_clean))]

    # Strategy 2: Extract quoted portions from the evidence
    # Match both "double quotes" and 'single quotes'
    quoted_matches = re.findall(r'["\']([^"\']{4,})["\']', key_evidence)
    for quoted in quoted_matches:
        idx = rollout_lower.find(quoted.lower())
        if idx != -1:
            matches.append((idx, idx + len(quoted)))

    if matches:
        return matches

    # Strategy 3: Find significant phrases (4+ consecutive words)
    # Split evidence into words and try sliding windows
    words = re.findall(r"\b\w+\b", key_evidence)
    for window_size in range(min(6, len(words)), 3, -1):  # Try 6-word, then 5, then 4
        for i in range(len(words) - window_size + 1):
            # Use regex for flexible whitespace matching
            pattern = r"\b" + r"\s+".join(re.escape(w) for w in words[i : i + window_size]) + r"\b"
            match = re.search(pattern, rollout, re.IGNORECASE)
            if match:
                matches.append((match.start(), match.end()))

        if matches:
            # Return matches from longest phrase found
            return matches

    return []


def highlight_evidence_in_rollout(rollout: str, key_evidence: str, score: str) -> str:
    """Highlight key evidence within the rollout text using <mark> tags.

    Args:
        rollout: The rollout text
        key_evidence: The key evidence to highlight
        score: The judge score ("1" = assumption made, "0" = no assumption)

    Returns HTML-escaped rollout with key evidence highlighted.
    """
    if not key_evidence or not key_evidence.strip():
        return escape_html(rollout)

    # Different colors based on score
    if score in ("1", 1):
        # Red/pink for assumptions made
        highlight_color = "#ffcdd2"  # Light red
    elif score in ("0", 0):
        # Green for no assumptions
        highlight_color = "#c8e6c9"  # Light green
    else:
        # Yellow for N/A
        highlight_color = "#ffeb3b"

    matches = find_evidence_matches(rollout, key_evidence)

    if not matches:
        # No matches found - just escape and return
        return escape_html(rollout)

    # Sort matches by start position and merge overlapping
    matches.sort()
    merged = []
    for start, end in matches:
        if merged and start <= merged[-1][1]:
            # Overlapping - extend previous
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build result with highlights
    result_parts = []
    prev_end = 0
    for start, end in merged:
        # Add text before this match
        if start > prev_end:
            result_parts.append(escape_html(rollout[prev_end:start]))
        # Add highlighted match
        evidence_text = escape_html(rollout[start:end])
        result_parts.append(f'<mark style="background-color: {highlight_color}; padding: 2px;">{evidence_text}</mark>')
        prev_end = end

    # Add remaining text after last match
    if prev_end < len(rollout):
        result_parts.append(escape_html(rollout[prev_end:]))

    return "".join(result_parts)


def generate_markdown_report(
    stats_path: Path,
    stats: dict,
    rollouts: dict,
    judge_results: dict,
    top_n: int = 5,
    rollouts_per_prompt: int = 3,
) -> str:
    """Generate a markdown report for top prompts by assumption rate."""

    lines = []
    base_name = stats_path.name.replace("_stats.json", "")

    # Header
    lines.append(f"# Gender Assumption Analysis: {base_name}")
    lines.append("")

    # Aggregate stats
    agg = stats.get("aggregate", {})
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Total rollouts evaluated:** {agg.get('total_rollouts', 'N/A')}")
    lines.append(f"- **Valid gender judgments:** {agg.get('total_valid', 'N/A')}")
    if agg.get("overall_assumption_rate") is not None:
        lines.append(f"- **Overall assumption rate:** {agg['overall_assumption_rate']:.1%}")
    lines.append(f"- **Total assumptions detected:** {agg.get('total_assumptions', 'N/A')}")
    lines.append("")

    # Get English-only prompts sorted by assumption rate
    english_prompts = get_english_only_prompts(stats)

    if not english_prompts:
        lines.append("*No English-only prompts with valid assumption rates found.*")
        return "\n".join(lines)

    lines.append(f"## Top {top_n} English Prompts by Assumption Rate")
    lines.append("")
    lines.append("---")
    lines.append("")

    rollouts_data = rollouts.get("data", {})

    for rank, (prompt_id, pstats) in enumerate(english_prompts[:top_n], 1):
        prompt_text = pstats.get("prompt", "")
        assumption_rate = pstats.get("mean_assumption_rate", 0)
        sum_assumptions = pstats.get("sum_assumptions", 0)
        n_valid = pstats.get("n_valid", 0)

        # Prompt header with anchor for easy navigation
        lines.append(f"### {rank}. Assumption Rate: {assumption_rate:.0%} ({sum_assumptions}/{n_valid})")
        lines.append("")
        lines.append(f"**Prompt ID:** `{prompt_id}`")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(f"> {prompt_text}")
        lines.append("")

        # Get rollouts and judge results for this prompt
        prompt_rollouts = rollouts_data.get(prompt_id, {}).get("rollouts", [])
        prompt_judge_results = judge_results.get(prompt_id, [])

        if not prompt_rollouts:
            lines.append("*No rollouts available for this prompt.*")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        lines.append(
            f"#### Rollouts (showing {min(rollouts_per_prompt, len(prompt_rollouts))} of {len(prompt_rollouts)})"
        )
        lines.append("")

        # Show rollouts with their judge results
        shown_count = 0
        for _idx, (rollout, judge_result) in enumerate(zip(prompt_rollouts, prompt_judge_results)):
            if shown_count >= rollouts_per_prompt:
                break

            score = judge_result.get("score", "N/A")
            rationale = judge_result.get("rationale", "No rationale provided")
            key_evidence = judge_result.get("key_evidence", "")
            is_english = judge_result.get("is_english", "N/A")

            # Skip non-English rollouts in the display
            if is_english not in (1, "1"):
                continue

            shown_count += 1

            # Score badge
            if score in ("1", 1):
                score_badge = "🔴 **Assumption Made**"
            elif score in ("0", 0):
                score_badge = "🟢 **No Assumption**"
            else:
                score_badge = "⚪ **N/A**"

            # Collapsible rollout section
            lines.append("<details>")
            lines.append(f"<summary><strong>Rollout {shown_count}</strong> — {score_badge}</summary>")
            lines.append("")
            lines.append("**Response:**")
            lines.append("")
            highlighted_rollout = highlight_evidence_in_rollout(rollout, key_evidence, score)
            lines.append(
                f'<pre style="white-space: pre-wrap; word-wrap: break-word; background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">{highlighted_rollout}</pre>'
            )
            lines.append("")
            lines.append(f"**Judge Rationale:** {rationale}")
            lines.append("")
            if key_evidence:
                lines.append(f"**Key Evidence:** {escape_html(key_evidence)}")
                lines.append("")
            lines.append("</details>")
            lines.append("")

        if shown_count == 0:
            lines.append("*No English rollouts available for this prompt.*")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Table of contents at the top (prepend)
    toc_lines = [
        "## Table of Contents",
        "",
    ]
    for rank, (_prompt_id, pstats) in enumerate(english_prompts[:top_n], 1):
        rate = pstats.get("mean_assumption_rate", 0)
        prompt_preview = pstats.get("prompt", "")[:50].replace("[", "\\[").replace("]", "\\]")
        toc_lines.append(f"{rank}. [{rate:.0%}] {prompt_preview}...")
    toc_lines.append("")

    # Insert TOC after header
    header_end = lines.index("## Summary Statistics")
    lines = lines[:header_end] + toc_lines + lines[header_end:]

    return "\n".join(lines)


def analyze_stats_file(stats_path: Path, top_n: int = 5, rollouts_per_prompt: int = 3) -> Path | None:
    """Analyze a single stats file and generate markdown report.

    Returns:
        Path to generated markdown file, or None if failed.
    """
    result = load_related_files(stats_path)
    if result is None:
        return None

    stats, rollouts, judge_results = result

    markdown = generate_markdown_report(
        stats_path=stats_path,
        stats=stats,
        rollouts=rollouts,
        judge_results=judge_results,
        top_n=top_n,
        rollouts_per_prompt=rollouts_per_prompt,
    )

    # Write markdown file
    output_path = stats_path.with_name(stats_path.name.replace("_stats.json", "_analysis.md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate markdown reports for top prompts by assumptions")
    parser.add_argument("directory", type=str, help="Directory to search for *_stats.json files")
    parser.add_argument("-n", "--top", type=int, default=5, help="Number of top prompts to include (default: 5)")
    parser.add_argument(
        "-r", "--rollouts", type=int, default=20, help="Number of rollouts to show per prompt (default: 3)"
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return 1

    # Find all *_stats.json files
    stats_files = list(directory.glob("*_stats.json"))

    if not stats_files:
        print(f"No *_stats.json files found in '{directory}'")
        return 1

    print(f"Found {len(stats_files)} stats file(s) in '{directory}'")
    print("=" * 60)

    for stats_file in sorted(stats_files):
        print(f"\nProcessing: {stats_file.name}")
        output_path = analyze_stats_file(
            stats_file,
            top_n=args.top,
            rollouts_per_prompt=args.rollouts,
        )
        if output_path:
            print(f"  -> Generated: {output_path.name}")

    return 0


if __name__ == "__main__":
    exit(main())
