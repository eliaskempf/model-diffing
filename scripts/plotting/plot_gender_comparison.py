"""
Gender Assumption Rate Comparison Visualization

This script creates a grouped bar chart comparing gender assumption rates
between a base model and a model organism (finetuned model) across:
- All prompts
- English prompts only
- Non-English prompts only

Requires judge_results.json files from generate_gender_rollouts.py

Usage:
    python plot_gender_comparison.py --results_dir output/gender-rollouts/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from tueplots.bundles import iclr2024


def find_model_files(results_dir: Path) -> tuple[dict, dict]:
    """Find and categorize model result files in the directory.

    Expects files named like:
    - {model_name}_judge_results.json
    - {model_name}_rollouts.json

    Returns:
        Tuple of (base_model_info, model_organism_info) where each is a dict with:
        - 'name': display name of the model
        - 'judge_results': path to judge_results.json
        - 'rollouts': path to rollouts.json (optional)
    """
    judge_files = list(results_dir.glob("*_judge_results.json"))

    if len(judge_files) < 2:
        raise ValueError(
            f"Expected at least 2 *_judge_results.json files in {results_dir}, "
            f"found {len(judge_files)}: {[f.name for f in judge_files]}"
        )

    if len(judge_files) > 2:
        print(f"Warning: Found {len(judge_files)} judge_results files, using first 2")
        judge_files = judge_files[:2]

    # Extract model names from filenames
    models = []
    for f in judge_files:
        # Remove _judge_results.json suffix
        model_name = f.name.replace("_judge_results.json", "")
        # Convert underscores back to slashes for HuggingFace-style names
        display_name = model_name.replace("_", "/", 1) if "_" in model_name else model_name

        rollouts_file = results_dir / f"{model_name}_rollouts.json"

        models.append(
            {
                "name": display_name,
                "raw_name": model_name,
                "judge_results": f,
                "rollouts": rollouts_file if rollouts_file.exists() else None,
            }
        )

    # Identify base model vs model organism
    # Heuristic: the model organism typically has a longer name or contains
    # terms like "finetuned", "user-female", etc.
    # The base model is typically the shorter/simpler name

    def is_likely_finetuned(name: str) -> bool:
        indicators = ["finetuned", "ft", "user-", "lora", "adapter", "trained", "sft", "rlhf", "dpo", "ppo"]
        name_lower = name.lower()
        return any(ind in name_lower for ind in indicators)

    # Sort: base model first (not finetuned), model organism second
    if is_likely_finetuned(models[0]["name"]) and not is_likely_finetuned(models[1]["name"]):
        base_model, model_organism = models[1], models[0]
    elif is_likely_finetuned(models[1]["name"]) and not is_likely_finetuned(models[0]["name"]):
        base_model, model_organism = models[0], models[1]
    else:
        # Fall back to shorter name = base model
        if len(models[0]["name"]) <= len(models[1]["name"]):
            base_model, model_organism = models[0], models[1]
        else:
            base_model, model_organism = models[1], models[0]

    return base_model, model_organism


def load_judge_results(file_path: str) -> dict:
    """Load judge results from JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def compute_prompt_level_stats(judge_results: dict) -> dict:
    """Compute per-prompt assumption rates and language info.

    Returns:
        Dict with prompt_id -> {assumption_rate, is_english, n_valid}
    """
    prompt_stats = {}

    for prompt_id, results in judge_results.items():
        assumption_scores = []
        english_votes = []

        for r in results:
            # Gender assumption score
            score = r.get("score", "N/A")
            if score in ("0", "1", 0, 1):
                assumption_scores.append(int(score))

            # Language detection
            is_english = r.get("is_english", "N/A")
            if is_english in ("0", "1", 0, 1):
                english_votes.append(int(is_english))

        if len(assumption_scores) > 0:
            # Determine if prompt is English based on majority vote
            if len(english_votes) > 0:
                is_english = sum(english_votes) / len(english_votes) > 0.5
            else:
                is_english = None  # Unknown

            prompt_stats[prompt_id] = {
                "assumption_rate": sum(assumption_scores) / len(assumption_scores),
                "is_english": is_english,
                "n_valid": len(assumption_scores),
            }

    return prompt_stats


def _filter_rates_by_subset(prompt_stats: dict, subset: str = "all") -> tuple[list[float], int, float]:
    """Filter prompt assumption rates by language subset.

    Returns:
        Tuple of (rates, n_prompts, percentage_of_total)
    """
    if subset == "all":
        rates = [p["assumption_rate"] for p in prompt_stats.values()]
    elif subset == "english":
        rates = [p["assumption_rate"] for p in prompt_stats.values() if p["is_english"] is True]
    elif subset == "non_english":
        rates = [p["assumption_rate"] for p in prompt_stats.values() if p["is_english"] is False]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    n_total = len(prompt_stats)
    percentage = (len(rates) / n_total * 100) if n_total > 0 else 0
    return rates, len(rates), percentage


def compute_subset_statistics(prompt_stats: dict, subset: str = "all") -> dict:
    """Compute statistics for a subset of prompts using the mean + t-CI.

    Returns:
        Dict with central, ci_lower, ci_upper, n_prompts, percentage
    """
    rates, n_prompts, percentage = _filter_rates_by_subset(prompt_stats, subset)

    if n_prompts == 0:
        return {
            "central": None,
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_prompts": 0,
            "percentage": 0,
        }

    mean = np.mean(rates)

    # 95% confidence interval
    if n_prompts > 1:
        sem = stats.sem(rates)
        ci = stats.t.interval(0.95, n_prompts - 1, loc=mean, scale=sem)
        ci_lower, ci_upper = ci
    else:
        ci_lower, ci_upper = mean, mean

    return {
        "central": mean,
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_prompts": n_prompts,
        "percentage": percentage,
    }


def _bootstrap_ci(
    rates: list[float], stat_fn, n_bootstrap: int = 10000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for an arbitrary statistic."""
    rng = np.random.default_rng(seed=42)
    arr = np.array(rates)
    boot_stats = np.array([stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)])
    alpha = (1 - confidence) / 2
    return float(np.percentile(boot_stats, 100 * alpha)), float(np.percentile(boot_stats, 100 * (1 - alpha)))


def compute_subset_statistics_quantile(
    prompt_stats: dict,
    subset: str = "all",
    quantile: float = 0.5,
    n_bootstrap: int = 10000,
) -> dict:
    """Compute statistics for a subset using a quantile + bootstrap CI.

    Args:
        quantile: Which quantile to use as the central estimate (0.5 = median).
        n_bootstrap: Number of bootstrap resamples for the CI.

    Returns:
        Dict with central, ci_lower, ci_upper, n_prompts, percentage
    """
    rates, n_prompts, percentage = _filter_rates_by_subset(prompt_stats, subset)

    if n_prompts == 0:
        return {
            "central": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_prompts": 0,
            "percentage": 0,
        }

    central = float(np.quantile(rates, quantile))

    if n_prompts > 1:

        def stat_fn(x):
            return np.quantile(x, quantile)

        ci_lower, ci_upper = _bootstrap_ci(rates, stat_fn, n_bootstrap=n_bootstrap)
    else:
        ci_lower, ci_upper = central, central

    return {
        "central": central,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_prompts": n_prompts,
        "percentage": percentage,
    }


def _compute_bar_data(
    base_stats: dict,
    organism_stats: dict,
    aggregation: str = "mean",
    quantile: float = 0.5,
) -> tuple[dict, dict, str]:
    """Compute per-subset statistics for both models.

    Args:
        aggregation: "mean" or "quantile"
        quantile: Quantile value when aggregation="quantile" (0.5 = median).

    Returns:
        (base_data, organism_data, ylabel_prefix) dicts keyed by subset name.
    """
    subsets = ["all", "english", "non_english"]

    if aggregation == "mean":
        base_data = {s: compute_subset_statistics(base_stats, s) for s in subsets}
        organism_data = {s: compute_subset_statistics(organism_stats, s) for s in subsets}
        ylabel_prefix = "Mean"
    else:
        base_data = {s: compute_subset_statistics_quantile(base_stats, s, quantile=quantile) for s in subsets}
        organism_data = {s: compute_subset_statistics_quantile(organism_stats, s, quantile=quantile) for s in subsets}
        if quantile == 0.5:
            ylabel_prefix = "Median"
        else:
            ylabel_prefix = f"Q{quantile:.0%}"

    return base_data, organism_data, ylabel_prefix


def create_comparison_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
    aggregation: str = "mean",
    quantile: float = 0.5,
):
    """Create grouped bar chart comparing assumption rates.

    Args:
        aggregation: "mean" or "quantile".
        quantile: Quantile to use when aggregation="quantile" (0.5 = median).
    """
    plt.rcParams.update(iclr2024())

    # Load and process results
    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    # Compute statistics for each subset
    subsets = ["all", "english", "non_english"]
    subset_labels = ["All Prompts", "English", "Non-English"]

    base_data, organism_data, ylabel_prefix = _compute_bar_data(
        base_stats,
        organism_stats,
        aggregation=aggregation,
        quantile=quantile,
    )

    # Prepare plot data
    x = np.arange(len(subsets))
    width = 0.35

    base_centrals = [base_data[s]["central"] or 0 for s in subsets]
    organism_centrals = [organism_data[s]["central"] or 0 for s in subsets]

    # Error bars (asymmetric for CI)
    def _asymmetric_errors(data):
        return [
            [(data[s]["central"] - data[s]["ci_lower"]) if data[s]["central"] is not None else 0 for s in subsets],
            [(data[s]["ci_upper"] - data[s]["central"]) if data[s]["central"] is not None else 0 for s in subsets],
        ]

    base_errors = _asymmetric_errors(base_data)
    organism_errors = _asymmetric_errors(organism_data)

    # Create figure
    _fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # Create bars
    ax.bar(
        x - width / 2,
        base_centrals,
        width,
        label=f"{base_model_name}",
        color="tab:blue",
        yerr=base_errors,
        capsize=5,
    )

    ax.bar(
        x + width / 2,
        organism_centrals,
        width,
        label=f"{model_organism_name}",
        color="tab:orange",
        yerr=organism_errors,
        capsize=5,
    )

    # Labels and formatting
    ylabel_prefix = ylabel_prefix if ylabel_prefix != "mean" else ""
    ax.set_ylabel(f"{ylabel_prefix} Gender Assumption Rate (\\%)")
    ax.set_xticks(x)

    # Create x-axis labels with percentages
    x_labels = []
    for i, subset in enumerate(subsets):
        base_pct = base_data[subset]["percentage"]
        organism_pct = organism_data[subset]["percentage"]
        avg_pct = (base_pct + organism_pct) / 2
        x_labels.append(f"{subset_labels[i]}\n({avg_pct:.0f}\\%)")

    ax.set_xticklabels(x_labels)

    # Legend with sample counts
    base_n = base_data["all"]["n_prompts"]
    organism_n = organism_data["all"]["n_prompts"]
    ax.legend(
        [f"{base_model_name} (n={base_n})", f"{model_organism_name} (n={organism_n})"],
        loc="upper right",
    )

    # Set y-axis to percentage
    all_vals = base_centrals + organism_centrals
    ax.set_ylim(0, min(1.0, max(all_vals) * 1.3 + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS ({ylabel_prefix})")
    print("=" * 60)

    for subset, label in zip(subsets, subset_labels):
        print(f"\n{label}:")
        for name, data in [(base_model_name, base_data), (model_organism_name, organism_data)]:
            print(f"  {name}:")
            if data[subset]["central"] is not None:
                print(f"    {ylabel_prefix}: {data[subset]['central']:.2%}")
                print(f"    95% CI: [{data[subset]['ci_lower']:.2%}, {data[subset]['ci_upper']:.2%}]")
                print(f"    N prompts: {data[subset]['n_prompts']} ({data[subset]['percentage']:.1f}%)")
            else:
                print("    No data")


def create_violin_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
):
    """Create violin plots comparing distributions of per-prompt assumption rates."""
    plt.rcParams.update(iclr2024())

    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    subsets = ["all", "english", "non_english"]
    subset_labels = ["All Prompts", "English", "Non-English"]

    # Build a long-form list for manual violin plotting
    # Structure: one entry per (subset_index, model, rate)
    violin_data: dict[str, dict[str, list[float]]] = {}
    for subset in subsets:
        base_rates, _, _ = _filter_rates_by_subset(base_stats, subset)
        organism_rates, _, _ = _filter_rates_by_subset(organism_stats, subset)
        violin_data[subset] = {"base": base_rates, "organism": organism_rates}

    _fig, ax = plt.subplots(figsize=(3.5, 2.4))

    positions_base = []
    positions_organism = []
    data_base = []
    data_organism = []
    x_ticks = []
    x_tick_labels = []

    width = 0.4
    for i, (subset, label) in enumerate(zip(subsets, subset_labels)):
        base_rates = violin_data[subset]["base"]
        organism_rates = violin_data[subset]["organism"]

        pos_b = i - width / 2
        pos_o = i + width / 2

        if len(base_rates) > 1:
            positions_base.append(pos_b)
            data_base.append(base_rates)
        if len(organism_rates) > 1:
            positions_organism.append(pos_o)
            data_organism.append(organism_rates)

        base_n = len(base_rates)
        organism_n = len(organism_rates)
        base_pct = (base_n / len(base_stats) * 100) if len(base_stats) > 0 else 0
        organism_pct = (organism_n / len(organism_stats) * 100) if len(organism_stats) > 0 else 0
        avg_pct = (base_pct + organism_pct) / 2

        x_ticks.append(i)
        x_tick_labels.append(f"{label}\n({avg_pct:.0f}\\%)")

    def _style_violins(parts, color):
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        for key in ("cmins", "cmaxes", "cbars", "cmedians"):
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(1.2)

    if data_base:
        vp_base = ax.violinplot(
            data_base,
            positions=positions_base,
            widths=width * 0.85,
            showmedians=True,
            showextrema=True,
        )
        _style_violins(vp_base, "tab:blue")

    if data_organism:
        vp_organism = ax.violinplot(
            data_organism,
            positions=positions_organism,
            widths=width * 0.85,
            showmedians=True,
            showextrema=True,
        )
        _style_violins(vp_organism, "tab:orange")

    # Manual legend (violinplot doesn't produce legend handles)
    from matplotlib.patches import Patch

    base_all_n = len(violin_data["all"]["base"])
    organism_all_n = len(violin_data["all"]["organism"])
    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.4, label=f"{base_model_name} (n={base_all_n})"),
            Patch(facecolor="tab:orange", alpha=0.4, label=f"{model_organism_name} (n={organism_all_n})"),
        ],
        loc="upper right",
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel("Gender Assumption Rate (\\%)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved violin plot to {output_path}")


def create_box_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
):
    """Create grouped box plots comparing distributions of per-prompt assumption rates."""
    plt.rcParams.update(iclr2024())

    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    subsets = ["all", "english", "non_english"]
    subset_labels = ["All Prompts", "English", "Non-English"]

    _fig, ax = plt.subplots(figsize=(3.5, 2.4))

    width = 0.35
    x_ticks = []
    x_tick_labels = []

    for i, (subset, label) in enumerate(zip(subsets, subset_labels)):
        base_rates, base_n, _ = _filter_rates_by_subset(base_stats, subset)
        organism_rates, organism_n, _ = _filter_rates_by_subset(organism_stats, subset)

        pos_b = i - width / 2
        pos_o = i + width / 2

        if base_n > 0:
            bp_base = ax.boxplot(
                base_rates,
                positions=[pos_b],
                widths=width * 0.75,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="tab:blue"),  # linewidth=1.5),
                flierprops=dict(marker=".", markersize=3, markerfacecolor="tab:blue", markeredgecolor="tab:blue"),
            )
            for patch in bp_base["boxes"]:
                patch.set_facecolor("tab:blue")
                patch.set_alpha(0.5)
            for whisker in bp_base["whiskers"]:
                whisker.set_color("tab:blue")
            for cap in bp_base["caps"]:
                cap.set_color("tab:blue")

        if organism_n > 0:
            bp_org = ax.boxplot(
                organism_rates,
                positions=[pos_o],
                widths=width * 0.75,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="tab:orange"),  # , linewidth=1.5),
                flierprops=dict(marker=".", markersize=3, markerfacecolor="tab:orange", markeredgecolor="tab:orange"),
            )
            for patch in bp_org["boxes"]:
                patch.set_facecolor("tab:orange")
                patch.set_alpha(0.5)
            for whisker in bp_org["whiskers"]:
                whisker.set_color("tab:orange")
            for cap in bp_org["caps"]:
                cap.set_color("tab:orange")

        base_pct = (base_n / len(base_stats) * 100) if len(base_stats) > 0 else 0
        organism_pct = (organism_n / len(organism_stats) * 100) if len(organism_stats) > 0 else 0
        avg_pct = (base_pct + organism_pct) / 2

        x_ticks.append(i)
        x_tick_labels.append(f"{label}\n({avg_pct:.0f}\\%)")

    # Manual legend
    from matplotlib.patches import Patch

    base_all_n = len([p for p in base_stats.values()])
    organism_all_n = len([p for p in organism_stats.values()])
    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=f"{base_model_name} (n={base_all_n})"),
            Patch(facecolor="tab:orange", alpha=0.5, label=f"{model_organism_name} (n={organism_all_n})"),
        ],
        loc="upper right",
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel("Gender Assumption Rate (\\%)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved box plot to {output_path}")


def create_paired_scatter_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
):
    """Create paired scatter plot: base rate (x) vs organism rate (y) per prompt.

    Points above the diagonal indicate higher assumption rate in the organism model.
    Color encodes language (English vs non-English).
    """
    plt.rcParams.update(iclr2024())

    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    # Only include prompts present in both models
    shared_prompts = set(base_stats.keys()) & set(organism_stats.keys())

    base_rates_en, org_rates_en = [], []
    base_rates_non, org_rates_non = [], []
    base_rates_unk, org_rates_unk = [], []

    for pid in shared_prompts:
        br = base_stats[pid]["assumption_rate"]
        orr = organism_stats[pid]["assumption_rate"]
        is_en = base_stats[pid]["is_english"]

        if is_en is True:
            base_rates_en.append(br)
            org_rates_en.append(orr)
        elif is_en is False:
            base_rates_non.append(br)
            org_rates_non.append(orr)
        else:
            base_rates_unk.append(br)
            org_rates_unk.append(orr)

    _fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, lw=0.8, zorder=1)

    marker_kw = dict(s=12, alpha=0.5, edgecolors="none", zorder=2)
    if base_rates_en:
        ax.scatter(
            base_rates_en, org_rates_en, color="tab:blue", label=f"English (n={len(base_rates_en)})", **marker_kw
        )
    if base_rates_non:
        ax.scatter(
            base_rates_non,
            org_rates_non,
            color="tab:orange",
            label=f"Non-English (n={len(base_rates_non)})",
            **marker_kw,
        )
    if base_rates_unk:
        ax.scatter(
            base_rates_unk, org_rates_unk, color="tab:grey", label=f"Unknown (n={len(base_rates_unk)})", **marker_kw
        )

    ax.set_xlabel("Gender Assumption Rate (\\texttt{Base}, \\%)")
    ax.set_ylabel("Gender Assumption Rate (\\texttt{FT}, \\%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", markerscale=2)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved paired scatter plot to {output_path} ({len(shared_prompts)} shared prompts)")


def create_ecdf_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
):
    """Create ECDF plots comparing distributions of per-prompt assumption rates."""
    plt.rcParams.update(iclr2024())

    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    subsets = ["all", "english", "non_english"]
    subset_labels = ["All Prompts", "English", "Non-English"]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)

    for ax, subset, label in zip(axes, subsets, subset_labels):
        base_rates, base_n, _ = _filter_rates_by_subset(base_stats, subset)
        organism_rates, organism_n, _ = _filter_rates_by_subset(organism_stats, subset)

        if base_n > 0:
            sorted_b = np.sort(base_rates)
            ecdf_b = np.arange(1, base_n + 1) / base_n
            ax.step(
                sorted_b, ecdf_b, where="post", color="tab:blue", label=f"{base_model_name} (n={base_n})", linewidth=1.2
            )

        if organism_n > 0:
            sorted_o = np.sort(organism_rates)
            ecdf_o = np.arange(1, organism_n + 1) / organism_n
            ax.step(
                sorted_o,
                ecdf_o,
                where="post",
                color="tab:orange",
                label=f"{model_organism_name} (n={organism_n})",
                linewidth=1.2,
            )

        ax.set_title(label)
        ax.set_xlabel(r"Gender Assumption Rate (\%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    axes[0].set_ylabel("Cumulative Proportion")

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)

    for ax in axes:
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved ECDF plot to {output_path}")


def create_ccdf_plot(
    base_model_results: str,
    model_organism_results: str,
    base_model_name: str,
    model_organism_name: str,
    output_path: str,
):
    """Create complementary CDF (survival function) plots.

    Y-axis shows the proportion of prompts with assumption rate >= x.
    Read as: "y% of prompts have an assumption rate of x% or more."
    """
    plt.rcParams.update(iclr2024())

    base_judge = load_judge_results(base_model_results)
    organism_judge = load_judge_results(model_organism_results)

    base_stats = compute_prompt_level_stats(base_judge)
    organism_stats = compute_prompt_level_stats(organism_judge)

    subsets = ["all", "english", "non_english"]
    subset_labels = ["All Prompts", "English", "Non-English"]

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

    for ax, subset, label in zip(axes, subsets, subset_labels):
        base_rates, base_n, _ = _filter_rates_by_subset(base_stats, subset)
        organism_rates, organism_n, _ = _filter_rates_by_subset(organism_stats, subset)

        if base_n > 0:
            sorted_b = np.sort(base_rates)
            ccdf_b = np.arange(base_n, 0, -1) / base_n
            ax.step(
                sorted_b, ccdf_b, where="post", color="tab:blue", label=f"{base_model_name} (n={base_n})", linewidth=1.2
            )

        if organism_n > 0:
            sorted_o = np.sort(organism_rates)
            ccdf_o = np.arange(organism_n, 0, -1) / organism_n
            ax.step(
                sorted_o,
                ccdf_o,
                where="post",
                color="tab:orange",
                label=f"{model_organism_name} (n={organism_n})",
                linewidth=1.2,
            )

        ax.set_title(label)
        ax.set_xlabel("Gender Assumption Rate (\\%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    axes[0].set_ylabel("Prop.\\ of prompts with rate $\\geq x$")

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)

    for ax in axes:
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    sns.despine()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved CCDF plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare gender assumption rates between two models")

    parser.add_argument(
        "--results_dir",
        type=str,
        default="output/gender-rollouts",
        help="Directory containing *_judge_results.json files (default: output/gender-rollouts)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: {results_dir}/gender_comparison.png)",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Override display name for the base model (auto-detected if not specified)",
    )
    parser.add_argument(
        "--model_organism_name",
        type=str,
        default=None,
        help="Override display name for the model organism (auto-detected if not specified)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Auto-discover model files
    base_model, model_organism = find_model_files(results_dir)

    print(f"Found base model: {base_model['name']}")
    print(f"Found model organism: {model_organism['name']}")

    # Use provided names or auto-detected ones
    base_name = args.base_model_name or base_model["name"]
    organism_name = args.model_organism_name or model_organism["name"]

    base_results = str(base_model["judge_results"])
    organism_results = str(model_organism["judge_results"])

    # Default output stem
    output_stem = args.output or str(results_dir / "gender_comparison.png")
    stem = output_stem.replace(".png", "")

    common_kwargs = dict(
        base_model_results=base_results,
        model_organism_results=organism_results,
        base_model_name=base_name,
        model_organism_name=organism_name,
    )

    # --- Mean plot (original) ---
    create_comparison_plot(**common_kwargs, output_path=f"{stem}.png", aggregation="mean")

    # --- Median plot (bootstrap CI) ---
    create_comparison_plot(**common_kwargs, output_path=f"{stem}_median.png", aggregation="quantile", quantile=0.5)

    # --- 25th and 75th percentile plots ---
    create_comparison_plot(**common_kwargs, output_path=f"{stem}_q25.png", aggregation="quantile", quantile=0.25)
    create_comparison_plot(**common_kwargs, output_path=f"{stem}_q75.png", aggregation="quantile", quantile=0.75)

    # --- Violin distribution plots ---
    create_violin_plot(**common_kwargs, output_path=f"{stem}_violin.png")

    # --- Box plots ---
    create_box_plot(**common_kwargs, output_path=f"{stem}_box.png")

    # --- Paired scatter plot ---
    create_paired_scatter_plot(**common_kwargs, output_path=f"{stem}_scatter.png")

    # --- ECDF plots ---
    create_ecdf_plot(**common_kwargs, output_path=f"{stem}_ecdf.png")

    # --- Complementary CDF (survival) plots ---
    create_ccdf_plot(**common_kwargs, output_path=f"{stem}_ccdf.png")


if __name__ == "__main__":
    main()
