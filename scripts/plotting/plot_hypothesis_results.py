"""
Plot hypothesis evaluation results.

This script reads the JSON output from evaluate_hypotheses.py and generates various
visualizations for analysis.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tueplots.bundles import iclr2024

from model_diffing.hypothesis_config import DATASETS

ICLR_WIDTH = iclr2024()["figure.figsize"][0]
ICLR_HEIGHT = iclr2024()["figure.figsize"][1]

# =============================================================================
# Data Preparation Utilities
# =============================================================================


def prepare_metrics_dataframe(
    all_results: dict,
    accepted_only: bool = True,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare a long-format DataFrame with metrics for each hypothesis.

    This is the primary data preparation function for violin and other distribution plots.

    Args:
        all_results: Results dictionary from evaluate_experiment
        accepted_only: If True, only include accepted hypotheses
        metrics: List of metric names to include. Defaults to ["test_accuracy", "test_frequency"]

    Returns:
        DataFrame with columns: dataset, method, metric, value, hypothesis_id
    """
    if metrics is None:
        metrics = ["test_accuracy", "test_frequency"]

    rows = []
    for dataset_key, dataset_title, llm_key, sae_key in DATASETS:
        for method, key in [("LLM", llm_key), ("SAE", sae_key)]:
            if key not in all_results:
                continue

            res = all_results[key]
            hypotheses = res["hypotheses"]
            filtered_hypos = [h for h in hypotheses if h["accepted"]] if accepted_only else hypotheses

            for h in filtered_hypos:
                for metric in metrics:
                    value = h.get(metric)
                    if value is not None:
                        # Scale metrics appropriately
                        if metric in ["test_accuracy", "test_frequency", "train_accuracy", "train_frequency"]:
                            value = value * 100  # Convert to percentage
                        elif metric in ["interestingness_score", "abstraction_score"]:
                            value = value * 20  # Scale 1-5 to 20-100%
                        rows.append(
                            {
                                "dataset": dataset_title,
                                "dataset_key": dataset_key,
                                "method": method,
                                "metric": metric,
                                "value": value,
                                "hypothesis_id": h["hypothesis_id"],
                            }
                        )

    return pd.DataFrame(rows)


def prepare_aggregated_metrics(
    all_results: dict,
    accepted_only: bool = True,
    include_autorater: bool = False,
    error_type: str | None = None,
) -> pd.DataFrame:
    """
    Prepare aggregated metrics (means and optionally errors) for each dataset-method combination.

    This is used for radar plots and summary tables.

    Args:
        all_results: Results dictionary from evaluate_experiment
        accepted_only: If True, only include accepted hypotheses
        include_autorater: If True, include interestingness and abstraction scores
        error_type: If provided ("ci95" or "std"), include error columns for each metric

    Returns:
        DataFrame with columns: dataset, method, acceptance_rate, test_accuracy, test_frequency,
                               [interestingness, abstraction if include_autorater]
                               [*_error columns if error_type is provided]
    """
    rows = []
    for dataset_key, dataset_title, llm_key, sae_key in DATASETS:
        for method, key in [("LLM", llm_key), ("SAE", sae_key)]:
            if key not in all_results:
                continue

            res = all_results[key]
            hypotheses = res["hypotheses"]
            filtered_hypos = [h for h in hypotheses if h["accepted"]] if accepted_only else hypotheses

            if not filtered_hypos:
                continue

            # Extract raw values for computing errors
            acc_values = [h["test_accuracy"] * 100 for h in filtered_hypos]
            freq_values = [h["test_frequency"] * 100 for h in filtered_hypos]

            row = {
                "dataset": dataset_title,
                "dataset_key": dataset_key,
                "method": method,
                "acceptance_rate": res["acceptance_rate"] * 100,
                "test_accuracy": np.mean(acc_values),
                "test_frequency": np.mean(freq_values),
            }

            # Add error columns if requested
            if error_type:
                row["acceptance_rate_error"] = 0  # No error for acceptance rate (single value)
                row["test_accuracy_error"] = compute_error_bar(acc_values, error_type)
                row["test_frequency_error"] = compute_error_bar(freq_values, error_type)

            if include_autorater:
                int_values = [
                    h["interestingness_score"] for h in filtered_hypos if h.get("interestingness_score") is not None
                ]
                abs_values = [h["abstraction_score"] for h in filtered_hypos if h.get("abstraction_score") is not None]
                row["interestingness"] = np.mean(int_values) if int_values else None
                row["abstraction"] = np.mean(abs_values) if abs_values else None

                if error_type:
                    row["interestingness_error"] = compute_error_bar(int_values, error_type) if int_values else 0
                    row["abstraction_error"] = compute_error_bar(abs_values, error_type) if abs_values else 0

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Statistical Utilities
# =============================================================================


def compute_error_bar(values: list[float], error_type: str = "ci95") -> float:
    """
    Compute error bar value.

    Args:
        values: List of values to compute error for
        error_type: Either "ci95" (95% CI) or "std" (standard deviation)

    Returns:
        Error bar half-width
    """
    if len(values) < 2:
        return 0.0
    n = len(values)
    std = np.std(values, ddof=1)
    if error_type == "std":
        return std
    else:  # ci95 using t-distribution (more accurate for small samples)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        return t_crit * std / np.sqrt(n)


# =============================================================================
# KDE Plot Configuration and Generic Function
# =============================================================================

# KDE plot configuration (easily tunable parameters)
KDE_CONFIG = {
    "figsize": (10, 3),  # Smaller figure size for publication (makes fonts larger in comparison)
    "colors": {"LLM": "tab:blue", "SAE": "tab:orange"},
    "linewidth": 2,
    "fill_alpha": 0.3,
    "mean_line_style": "--",
    "mean_line_alpha": 0.8,
    "stats_fontsize": 7,
    "axis_label_fontsize": 12,
    "title_fontsize": 14,
    "legend_fontsize": 8,
    "legend_loc": "upper left",
    "grid_alpha": 0.3,
    "dpi": 300,
}


def plot_metric_kde(
    all_results: dict,
    output_path: str,
    metric_key: str,
    xlabel: str,
    xlim: tuple[float, float],
    scale_factor: float = 1.0,
    accepted_only: bool = True,
    config: dict | None = None,
) -> None:
    """
    Create a 1x3 KDE plot showing smoothed distribution of a metric.
    LLM and SAE are overlaid on the same subplot for direct comparison.

    Args:
        all_results: Results dictionary from evaluate_experiment
        output_path: Path to save the plot (saves both PNG and PDF)
        metric_key: Key in hypothesis dict (e.g., "test_frequency", "interestingness_score")
        xlabel: X-axis label (use r-string for LaTeX escaping, e.g., r"Frequency (\\%)")
        xlim: Tuple of (min, max) for x-axis limits
        scale_factor: Factor to multiply raw values by (e.g., 100 for percentages)
        accepted_only: If True, only use accepted hypotheses
        config: Optional dict to override KDE_CONFIG settings
    """
    cfg = {**KDE_CONFIG, **(config or {})}

    _fig, axes = plt.subplots(1, 3, figsize=cfg["figsize"])

    for col, (dataset_title, llm_key, sae_key) in enumerate([(d[1], d[2], d[3]) for d in DATASETS]):
        ax = axes[col]

        stats_text_parts = []

        for method, key in [("LLM", llm_key), ("SAE", sae_key)]:
            if key not in all_results:
                continue

            res = all_results[key]
            hypotheses = res["hypotheses"]
            filtered_hypos = [h for h in hypotheses if h["accepted"]] if accepted_only else hypotheses

            # Filter for hypotheses that have this metric
            values = [h[metric_key] * scale_factor for h in filtered_hypos if h.get(metric_key) is not None]

            if len(values) > 1:
                color = cfg["colors"][method]

                # Plot KDE
                sns.kdeplot(
                    values,
                    ax=ax,
                    color=color,
                    linewidth=cfg["linewidth"],
                    label=method,
                    fill=True,
                    alpha=cfg["fill_alpha"],
                )

                # Add mean line
                mean_val = np.mean(values)
                ax.axvline(
                    mean_val,
                    color=color,
                    linestyle=cfg["mean_line_style"],
                    linewidth=cfg["linewidth"],
                    alpha=cfg["mean_line_alpha"],
                )

                # Collect stats (escape % for LaTeX)
                n_hypos = len(values)
                std_val = np.std(values)
                if scale_factor == 100:
                    stats_text_parts.append(
                        f"{method}: n={n_hypos}, $\\mu$={mean_val:.1f}\\%, $\\sigma$={std_val:.1f}\\%"
                    )
                else:
                    stats_text_parts.append(f"{method}: n={n_hypos}, $\\mu$={mean_val:.2f}, $\\sigma$={std_val:.2f}")

        # Add combined stats text
        if stats_text_parts:
            ax.text(
                0.95,
                0.95,
                "\n".join(stats_text_parts),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=cfg["stats_fontsize"],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.tick_params(axis="x", labelsize=cfg["axis_label_fontsize"] - 2)
        ax.tick_params(axis="y", labelsize=cfg["axis_label_fontsize"] - 2)
        ax.set_title(dataset_title, fontsize=cfg["title_fontsize"], fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=cfg["axis_label_fontsize"])
        ax.set_ylabel("Density", fontsize=cfg["axis_label_fontsize"])
        ax.set_xlim(xlim)
        ax.legend(loc=cfg["legend_loc"], fontsize=cfg["legend_fontsize"])
        ax.grid(True, alpha=cfg["grid_alpha"], axis="y")
        sns.despine(ax=ax)

    plt.tight_layout()

    # Save as PDF (primary format for publication)
    pdf_path = output_path.replace(".png", ".pdf") if output_path.endswith(".png") else output_path + ".pdf"
    plt.savefig(pdf_path, dpi=cfg["dpi"], bbox_inches="tight")

    # Also save PNG for quick viewing
    png_path = output_path if output_path.endswith(".png") else output_path + ".png"
    plt.savefig(png_path, dpi=cfg["dpi"], bbox_inches="tight")

    plt.close()
    print(f"KDE plot saved to {pdf_path} and {png_path}")


def plot_frequency_kde(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    config: dict | None = None,
) -> None:
    """Create KDE plot for test frequency distribution."""
    plot_metric_kde(
        all_results=all_results,
        output_path=output_path,
        metric_key="test_frequency",
        xlabel=r"Test Frequency (\%)",
        xlim=(0, 100),
        scale_factor=100,
        accepted_only=accepted_only,
        config=config,
    )


def plot_accuracy_kde(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    config: dict | None = None,
) -> None:
    """Create KDE plot for test accuracy distribution."""
    plot_metric_kde(
        all_results=all_results,
        output_path=output_path,
        metric_key="test_accuracy",
        xlabel=r"Test Accuracy (\%)",
        xlim=(0, 100),
        scale_factor=100,
        accepted_only=accepted_only,
        config=config,
    )


def plot_interestingness_kde(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    config: dict | None = None,
) -> None:
    """Create KDE plot for interestingness score distribution."""
    plot_metric_kde(
        all_results=all_results,
        output_path=output_path,
        metric_key="interestingness_score",
        xlabel="Interestingness Score",
        xlim=(1, 5),
        scale_factor=1,
        accepted_only=accepted_only,
        config=config,
    )


def plot_abstraction_kde(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    config: dict | None = None,
) -> None:
    """Create KDE plot for abstraction level score distribution."""
    plot_metric_kde(
        all_results=all_results,
        output_path=output_path,
        metric_key="abstraction_score",
        xlabel="Abstraction Level Score",
        xlim=(1, 5),
        scale_factor=1,
        accepted_only=accepted_only,
        config=config,
    )


def plot_aggregated_metrics_extended(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    error_bar_type: str | None = None,
) -> None:
    """
    Create a bar plot with extended metrics including autorater scores.
    Metrics: Acceptance Rate, Test Accuracy, Test Frequency, Interestingness, Abstraction

    Args:
        all_results: Results dictionary from evaluate_experiment
        output_path: Path to save the plot
        accepted_only: If True, only use accepted hypotheses for metrics
        error_bar_type: Type of error bars to show ("ci95" or "std"), or None for no error bars
    """
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extended metrics: Acceptance Rate, Accuracy, Frequency, Interestingness, Abstraction
    metric_labels = ["Accept.\nRate", "Test\nAccuracy", "Test\nFreq.", "Interest.", "Abstr."]
    x = np.arange(len(metric_labels))
    width = 0.35
    show_error_bars = error_bar_type is not None

    for ax, (_dataset_key, dataset_title, llm_key, sae_key) in zip(axes, DATASETS):
        # Get metrics for LLM and SAE (scaled appropriately)
        llm_metrics = [0, 0, 0, 0, 0]
        sae_metrics = [0, 0, 0, 0, 0]
        llm_errors = [0, 0, 0, 0, 0]
        sae_errors = [0, 0, 0, 0, 0]

        for key, metrics, errors in [(llm_key, llm_metrics, llm_errors), (sae_key, sae_metrics, sae_errors)]:
            if key not in all_results:
                continue

            res = all_results[key]
            hypotheses = res["hypotheses"]
            filtered_hypos = [h for h in hypotheses if h["accepted"]] if accepted_only else hypotheses

            if filtered_hypos:
                acc_values = [h["test_accuracy"] for h in filtered_hypos]
                freq_values = [h["test_frequency"] for h in filtered_hypos]

                # Base metrics (scaled to 0-100%)
                metrics[0] = res["acceptance_rate"] * 100
                metrics[1] = np.mean(acc_values) * 100
                metrics[2] = np.mean(freq_values) * 100

                if show_error_bars:
                    errors[1] = compute_error_bar(acc_values, error_bar_type) * 100
                    errors[2] = compute_error_bar(freq_values, error_bar_type) * 100

                # Interestingness scores (scale 1-5 -> 0-100% as 20-100%)
                int_values = [
                    h["interestingness_score"] for h in filtered_hypos if h.get("interestingness_score") is not None
                ]
                if int_values:
                    metrics[3] = np.mean(int_values) * 20  # 1-5 scale -> 20-100%
                    if show_error_bars:
                        errors[3] = compute_error_bar(int_values, error_bar_type) * 20

                # Abstraction scores (scale 1-5 -> 0-100% as 20-100%)
                abs_values = [h["abstraction_score"] for h in filtered_hypos if h.get("abstraction_score") is not None]
                if abs_values:
                    metrics[4] = np.mean(abs_values) * 20  # 1-5 scale -> 20-100%
                    if show_error_bars:
                        errors[4] = compute_error_bar(abs_values, error_bar_type) * 20

        # Draw bars with optional error bars
        yerr1 = llm_errors if show_error_bars else None
        yerr2 = sae_errors if show_error_bars else None
        bars1 = ax.bar(
            x - width / 2,
            llm_metrics,
            width,
            label="LLM",
            color="tab:blue",
            yerr=yerr1,
            capsize=3 if show_error_bars else 0,
        )
        bars2 = ax.bar(
            x + width / 2,
            sae_metrics,
            width,
            label="SAE",
            color="tab:orange",
            yerr=yerr2,
            capsize=3 if show_error_bars else 0,
        )

        # Add value labels on bars
        def add_labels(bars, errors=None):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                err = errors[i] if errors else 0
                # Use different format for autorater scores (show as x/5)
                if i >= 3:  # Interestingness and Abstraction
                    label = f"{height / 20:.1f}"  # Convert back to 1-5 scale
                else:
                    label = f"{height:.1f}%"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height + err),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        add_labels(bars1, llm_errors if show_error_bars else None)
        add_labels(bars2, sae_errors if show_error_bars else None)

        ax.set_ylabel("Percent (%) / Score×20")
        ax.set_title(dataset_title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        sns.despine(ax=ax)

    hypo_label = "accepted hypotheses only" if accepted_only else "all hypotheses"
    plt.suptitle(f"Aggregated Metrics incl. Autorater Scores ({hypo_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Extended aggregated plot saved to {output_path}")


def plot_aggregated_metrics_with_ablation(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    ablate_type: str | None = None,
    ablate_threshold: int | None = None,
    error_bar_type: str | None = None,
) -> None:
    """
    Create a bar plot comparing aggregated metrics across experiments with optional ablation.

    Args:
        all_results: Results dictionary from evaluate_experiment
        output_path: Path to save the plot
        accepted_only: If True, only use accepted hypotheses for metrics
        ablate_type: Either "interestingness" or "abstraction" (or None for no ablation)
        ablate_threshold: Score threshold - exclude hypotheses with score >= this value
        error_bar_type: Type of error bars to show ("ci95" or "std"), or None for no error bars
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metric_labels = ["Acceptance\nRate", "Test\nAccuracy", "Test\nFrequency"]
    x = np.arange(len(metric_labels))
    width = 0.35
    show_error_bars = error_bar_type is not None

    # Determine if we're doing ablation
    is_ablation = ablate_type is not None and ablate_threshold is not None

    for ax, (_dataset_key, dataset_title, llm_key, sae_key) in zip(axes, DATASETS):
        # Compute metrics for LLM and SAE
        llm_metrics_original = [0, 0, 0]
        sae_metrics_original = [0, 0, 0]
        llm_metrics_ablated = [0, 0, 0]
        sae_metrics_ablated = [0, 0, 0]
        llm_errors_original = [0, 0, 0]
        sae_errors_original = [0, 0, 0]
        llm_errors_ablated = [0, 0, 0]
        sae_errors_ablated = [0, 0, 0]

        for key, metrics_original, metrics_ablated, errors_original, errors_ablated in [
            (llm_key, llm_metrics_original, llm_metrics_ablated, llm_errors_original, llm_errors_ablated),
            (sae_key, sae_metrics_original, sae_metrics_ablated, sae_errors_original, sae_errors_ablated),
        ]:
            if key not in all_results:
                continue

            res = all_results[key]
            hypotheses = res["hypotheses"]

            # Filter for accepted if requested
            if accepted_only:
                filtered_hypos = [h for h in hypotheses if h["accepted"]]
            else:
                filtered_hypos = hypotheses

            # Original metrics (no ablation)
            if filtered_hypos:
                num_accepted = sum(1 for h in hypotheses if h["accepted"])
                acc_values = [h["test_accuracy"] for h in filtered_hypos]
                freq_values = [h["test_frequency"] for h in filtered_hypos]
                metrics_original[0] = num_accepted / len(hypotheses) if hypotheses else 0
                metrics_original[1] = np.mean(acc_values)
                metrics_original[2] = np.mean(freq_values)
                if show_error_bars:
                    errors_original[1] = compute_error_bar(acc_values, error_bar_type)
                    errors_original[2] = compute_error_bar(freq_values, error_bar_type)

            # Ablated metrics
            if is_ablation:
                score_key = "interestingness_score" if ablate_type == "interestingness" else "abstraction_score"
                # Filter hypotheses with score < threshold
                ablated_hypos = [
                    h for h in filtered_hypos if h.get(score_key) is not None and h[score_key] < ablate_threshold
                ]

                if ablated_hypos:
                    # For acceptance rate, count accepted among ablated hypotheses from the full set
                    all_ablated = [
                        h for h in hypotheses if h.get(score_key) is not None and h[score_key] < ablate_threshold
                    ]
                    num_accepted_ablated = sum(1 for h in all_ablated if h["accepted"])
                    acc_values_abl = [h["test_accuracy"] for h in ablated_hypos]
                    freq_values_abl = [h["test_frequency"] for h in ablated_hypos]
                    metrics_ablated[0] = num_accepted_ablated / len(all_ablated) if all_ablated else 0
                    metrics_ablated[1] = np.mean(acc_values_abl)
                    metrics_ablated[2] = np.mean(freq_values_abl)
                    if show_error_bars:
                        errors_ablated[1] = compute_error_bar(acc_values_abl, error_bar_type)
                        errors_ablated[2] = compute_error_bar(freq_values_abl, error_bar_type)
            else:
                metrics_ablated[:] = metrics_original[:]
                errors_ablated[:] = errors_original[:]

        # Scale all metrics and errors to 0-100%
        llm_metrics_original = [v * 100 for v in llm_metrics_original]
        sae_metrics_original = [v * 100 for v in sae_metrics_original]
        llm_metrics_ablated = [v * 100 for v in llm_metrics_ablated]
        sae_metrics_ablated = [v * 100 for v in sae_metrics_ablated]
        llm_errors_original = [v * 100 for v in llm_errors_original]
        sae_errors_original = [v * 100 for v in sae_errors_original]
        llm_errors_ablated = [v * 100 for v in llm_errors_ablated]
        sae_errors_ablated = [v * 100 for v in sae_errors_ablated]

        if is_ablation:
            # Draw faded bars for original values (reference)
            ax.bar(x - width / 2, llm_metrics_original, width, label="LLM (all)", color="tab:blue", alpha=0.3)
            ax.bar(x + width / 2, sae_metrics_original, width, label="SAE (all)", color="tab:orange", alpha=0.3)

            # Draw solid bars for ablated values with optional error bars
            yerr1 = llm_errors_ablated if show_error_bars else None
            yerr2 = sae_errors_ablated if show_error_bars else None
            bars1_abl = ax.bar(
                x - width / 2,
                llm_metrics_ablated,
                width,
                label=f"LLM ({ablate_type}<{ablate_threshold})",
                color="tab:blue",
                alpha=1.0,
                yerr=yerr1,
                capsize=3 if show_error_bars else 0,
            )
            bars2_abl = ax.bar(
                x + width / 2,
                sae_metrics_ablated,
                width,
                label=f"SAE ({ablate_type}<{ablate_threshold})",
                color="tab:orange",
                alpha=1.0,
                yerr=yerr2,
                capsize=3 if show_error_bars else 0,
            )

            # Add value labels on ablated bars
            def add_labels(bars, values, errors=None):
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    err = errors[i] if errors else 0
                    ax.annotate(
                        f"{val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            add_labels(bars1_abl, llm_metrics_ablated, llm_errors_ablated if show_error_bars else None)
            add_labels(bars2_abl, sae_metrics_ablated, sae_errors_ablated if show_error_bars else None)
        else:
            # No ablation - just draw regular bars with optional error bars
            yerr1 = llm_errors_original if show_error_bars else None
            yerr2 = sae_errors_original if show_error_bars else None
            bars1 = ax.bar(
                x - width / 2,
                llm_metrics_original,
                width,
                label="LLM",
                color="tab:blue",
                yerr=yerr1,
                capsize=3 if show_error_bars else 0,
            )
            bars2 = ax.bar(
                x + width / 2,
                sae_metrics_original,
                width,
                label="SAE",
                color="tab:orange",
                yerr=yerr2,
                capsize=3 if show_error_bars else 0,
            )

            def add_labels(bars, errors=None):
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    err = errors[i] if errors else 0
                    ax.annotate(
                        f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            add_labels(bars1, llm_errors_original if show_error_bars else None)
            add_labels(bars2, sae_errors_original if show_error_bars else None)

        ax.set_ylabel("Percent (%)")
        title = dataset_title
        if is_ablation:
            title += f"\n(ablate {ablate_type} < {ablate_threshold})"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        sns.despine(ax=ax)

    # Add overall title
    hypo_label = "accepted hypotheses only" if accepted_only else "all hypotheses"
    if is_ablation:
        fig.suptitle(
            f"Aggregated Metrics ({hypo_label}, ablation: {ablate_type} < {ablate_threshold})",
            fontsize=14,
            fontweight="bold",
        )
        plt.subplots_adjust(top=0.88)
    else:
        fig.suptitle(f"Aggregated Metrics ({hypo_label})", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Aggregated plot saved to {output_path}")


# =============================================================================
# Radar Plot
# =============================================================================

# Radar plot configuration (can be modified for tuning)
RADAR_CONFIG = {
    "figsize": (1.6 * ICLR_WIDTH, 1.6 * ICLR_HEIGHT),
    # "figsize": (10, 4),
    "scale": 1,  # Scaling factor for figsize and font sizes (e.g., 1.5 = 150%)
    "colors": {"LLM": "tab:blue", "SAE": "tab:orange"},
    "fill_alpha": 0.25,
    "line_width": 1.5,
    "marker_size": 4,
    "grid_style": "circular",  # "circular" or "polygon" (spider web look)
    "grid_color": "gray",
    "grid_alpha": 0.3,
    "grid_linewidth": 0.5,  # Line width for polygon grid
    "label_fontsize": 10,
    "value_fontsize": 8,
    "title_fontsize": 12,
    "legend_fontsize": 9,
    # Error band settings
    "error_band_alpha": 0.3,  # Alpha for error band fill (higher = more visible)
    "error_band_edge": True,  # Draw dashed lines at error band boundaries
    "error_band_edge_width": 1.0,  # Line width for error band edges
    "error_band_edge_style": "--",  # Line style for edges: "--", ":", "-.", etc.
}


def _plot_single_radar(
    ax,
    df_dataset: pd.DataFrame,
    metrics: list[str],
    metric_labels: list[str],
    angles: list[float],
    error_type: str | None,
    cfg: dict,
    show_legend: bool = True,
    title: str | None = None,
) -> None:
    """
    Plot a single radar chart on the given polar axis.

    Args:
        ax: Matplotlib polar axis to plot on
        df_dataset: DataFrame with data for this dataset (filtered to one dataset)
        metrics: List of metric column names
        metric_labels: List of display labels for metrics
        angles: List of angles for radar chart (should include closing angle)
        error_type: Type of error bands ("ci95" or "std"), or None for no bands
        cfg: Configuration dict (from RADAR_CONFIG)
        show_legend: Whether to show the legend
        title: Optional title for the plot
    """
    # Get scaling factor
    scale = cfg.get("scale", 1.0)

    # Scaled font sizes
    label_fontsize = cfg.get("label_fontsize", 10) * scale
    value_fontsize = cfg.get("value_fontsize", 8) * scale
    title_fontsize = cfg.get("title_fontsize", 12) * scale
    legend_fontsize = cfg.get("legend_fontsize", 9) * scale
    line_width = cfg.get("line_width", 2) * scale
    marker_size = cfg.get("marker_size", 6) * scale

    if df_dataset.empty:
        if title:
            ax.set_title(f"{title}\n(no data)", fontsize=title_fontsize, fontweight="bold")
        return

    # Plot each method
    for method in ["LLM", "SAE"]:
        row = df_dataset[df_dataset["method"] == method]
        if row.empty:
            continue

        # Extract mean values
        values = [row[m].values[0] if m in row.columns and pd.notna(row[m].values[0]) else 0 for m in metrics]
        values += values[:1]  # Complete the loop

        color = cfg["colors"][method]

        # Draw confidence band if error_type is specified
        if error_type:
            errors = [
                row[f"{m}_error"].values[0]
                if f"{m}_error" in row.columns and pd.notna(row[f"{m}_error"].values[0])
                else 0
                for m in metrics
            ]
            errors += errors[:1]  # Complete the loop

            # Compute upper and lower bounds (clipped to valid range)
            upper = [min(100, v + e) for v, e in zip(values, errors)]
            lower = [max(0, v - e) for v, e in zip(values, errors)]

            # Draw the confidence band as a filled region between upper and lower
            ax.fill_between(angles, lower, upper, alpha=cfg["error_band_alpha"], color=color)

            # Optionally draw dashed lines at the error band boundaries
            if cfg.get("error_band_edge", True):
                ax.plot(
                    angles,
                    upper,
                    linestyle=cfg["error_band_edge_style"],
                    linewidth=cfg["error_band_edge_width"] * scale,
                    color=color,
                    alpha=0.7,
                )
                ax.plot(
                    angles,
                    lower,
                    linestyle=cfg["error_band_edge_style"],
                    linewidth=cfg["error_band_edge_width"] * scale,
                    color=color,
                    alpha=0.7,
                )

        # Plot the mean radar shape
        ax.plot(angles, values, "o-", linewidth=line_width, color=color, label=method, markersize=marker_size)
        ax.fill(angles, values, alpha=cfg["fill_alpha"], color=color)

    # Configure radar chart appearance
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=label_fontsize)
    ax.tick_params(axis="x", pad=7)

    # Set radial limits
    ax.set_ylim(0, 100)
    radial_ticks = [20, 40, 60, 80, 100]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=value_fontsize)

    # Grid styling - circular or polygon
    grid_style = cfg.get("grid_style", "circular")
    grid_color = cfg["grid_color"]
    grid_alpha = cfg["grid_alpha"]
    grid_linewidth = cfg.get("grid_linewidth", 0.5) * scale

    if grid_style == "polygon":
        # Turn off default circular grid
        ax.grid(False)
        ax.spines["polar"].set_visible(False)

        # Draw polygon grid lines at each radial level
        for r in radial_ticks:
            polygon_angles = angles  # Already includes closing angle
            polygon_radii = [r] * len(polygon_angles)
            ax.plot(polygon_angles, polygon_radii, color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha)

        # Draw radial spokes from center to each axis
        for angle in angles[:-1]:
            ax.plot([angle, angle], [0, 100], color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha)
    else:
        # Default circular grid
        ax.grid(True, color=grid_color, alpha=grid_alpha, linewidth=grid_linewidth)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20 * scale)

    # Add error bar legend entry (gray dashed line) if error bands are shown
    if error_type:
        error_label = r"95\% CI" if "95" in error_type else error_type.upper()
        ax.plot([], [], color="gray", linestyle="--", linewidth=1.5 * scale, label=error_label)

    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=legend_fontsize)


def plot_aggregated_metrics_radar(
    all_results: dict,
    output_path: str,
    accepted_only: bool = True,
    include_autorater: bool = False,
    error_type: str | None = None,
    config: dict | None = None,
) -> None:
    """
    Create a radar/spider plot comparing aggregated metrics across experiments.

    Three subplots (Gemini, Qwen EM, Gemma Gender), with LLM and SAE shown
    as overlapping radar shapes. Optionally includes shaded confidence bands.

    Args:
        all_results: Results dictionary from evaluate_experiment
        output_path: Path to save the plot
        accepted_only: If True, only use accepted hypotheses
        include_autorater: If True, include interestingness and abstraction on the radar
        error_type: Type of error bands ("ci95" or "std"), or None for no bands
        config: Optional dict to override RADAR_CONFIG settings
    """
    # Merge config with defaults
    cfg = {**RADAR_CONFIG, **(config or {})}

    # Prepare aggregated data (with errors if requested)
    df = prepare_aggregated_metrics(
        all_results,
        accepted_only=accepted_only,
        include_autorater=include_autorater,
        error_type=error_type,
    )

    if df.empty:
        print("No data available for radar plot")
        return

    # Define metrics to plot
    if include_autorater:
        metrics = ["acceptance_rate", "test_accuracy", "test_frequency", "interestingness", "abstraction"]
        metric_labels = ["Accep-\ntance\nrate", "\nAccuracy", "Frequen-\ncy", "Interest-\ningness", "Abstrac-\ntion"]
        # Scale interestingness/abstraction from 1-5 to 0-100 for consistent display
        df["interestingness"] = df["interestingness"].apply(lambda x: x * 20 if pd.notna(x) else 0)
        df["abstraction"] = df["abstraction"].apply(lambda x: x * 20 if pd.notna(x) else 0)
        if error_type:
            df["interestingness_error"] = df["interestingness_error"].apply(lambda x: x * 20 if pd.notna(x) else 0)
            df["abstraction_error"] = df["abstraction_error"].apply(lambda x: x * 20 if pd.notna(x) else 0)
    else:
        metrics = ["acceptance_rate", "test_accuracy", "test_frequency"]
        metric_labels = ["AR", "Acc", "Freq"]

    num_metrics = len(metrics)

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Get scaling factor
    scale = cfg.get("scale", 1.0)
    scaled_figsize = (cfg["figsize"][0] * scale, cfg["figsize"][1] * scale)

    # -------------------------------------------------------------------------
    # 1. Combined 1x3 radar plot (original)
    # -------------------------------------------------------------------------
    _fig, axes = plt.subplots(1, 3, figsize=scaled_figsize, subplot_kw=dict(polar=True))

    for ax, (_, dataset_title, _, _) in zip(axes, DATASETS):
        df_dataset = df[df["dataset"] == dataset_title]
        _plot_single_radar(
            ax, df_dataset, metrics, metric_labels, angles, error_type, cfg, show_legend=True, title=dataset_title
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Radar plot saved to {output_path}")

    # -------------------------------------------------------------------------
    # 2. Individual radar plots per dataset (PDF, 1/3 size, no title)
    # -------------------------------------------------------------------------
    individual_figsize = (scaled_figsize[0] / 3, scaled_figsize[1] / 3)

    for dataset_key, dataset_title, _llm_key, _sae_key in DATASETS:
        df_dataset = df[df["dataset"] == dataset_title]
        if df_dataset.empty:
            continue

        fig_ind, ax_ind = plt.subplots(1, 1, figsize=individual_figsize, subplot_kw=dict(polar=True))
        _plot_single_radar(
            ax_ind,
            df_dataset,
            metrics,
            metric_labels,
            angles,
            error_type,
            cfg,
            show_legend=False,
            title=None,  # No title or legend for individual plots
        )

        individual_path = output_path.replace(".png", f"_{dataset_key}.pdf")
        plt.savefig(individual_path, format="pdf", bbox_inches="tight")
        plt.close(fig_ind)
        print(f"Individual radar plot saved to {individual_path}")

    # -------------------------------------------------------------------------
    # 3. Average radar plot (mean across all datasets)
    # -------------------------------------------------------------------------
    avg_rows = []
    for method in ["LLM", "SAE"]:
        method_data = df[df["method"] == method]
        if method_data.empty:
            continue

        avg_row = {
            "dataset": "Average",
            "method": method,
        }
        for metric in metrics:
            if metric in method_data.columns:
                avg_row[metric] = method_data[metric].mean()
            if error_type and f"{metric}_error" in method_data.columns:
                # Average the errors across datasets
                avg_row[f"{metric}_error"] = method_data[f"{metric}_error"].mean()
        avg_rows.append(avg_row)

    if avg_rows:
        df_avg = pd.DataFrame(avg_rows)

        fig_avg, ax_avg = plt.subplots(1, 1, figsize=scaled_figsize, subplot_kw=dict(polar=True))
        _plot_single_radar(
            ax_avg,
            df_avg,
            metrics,
            metric_labels,
            angles,
            error_type,
            cfg,
            show_legend=False,
            title="Average",  # No legend, use standalone PDF
        )

        avg_path = output_path.replace(".png", "_average.png")
        plt.savefig(avg_path, dpi=150, bbox_inches="tight")
        plt.close(fig_avg)
        print(f"Average radar plot saved to {avg_path}")

        # Also save average as PDF without title
        fig_avg_pdf, ax_avg_pdf = plt.subplots(1, 1, figsize=individual_figsize, subplot_kw=dict(polar=True))
        _plot_single_radar(
            ax_avg_pdf, df_avg, metrics, metric_labels, angles, error_type, cfg, show_legend=False, title=None
        )
        avg_pdf_path = output_path.replace(".png", "_average.pdf")
        plt.savefig(avg_pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig_avg_pdf)
        print(f"Average radar plot (PDF) saved to {avg_pdf_path}")

    # -------------------------------------------------------------------------
    # 4. Standalone legend PDF (vertical layout)
    # -------------------------------------------------------------------------
    line_width = cfg.get("line_width", 2) * scale
    marker_size = cfg.get("marker_size", 6) * scale
    legend_fontsize = cfg.get("legend_fontsize", 9) * scale

    fig_legend, ax_legend = plt.subplots(figsize=(2 * scale, 1.5 * scale))
    ax_legend.axis("off")

    # Create dummy lines for legend entries
    legend_handles = []
    for method in ["LLM", "SAE"]:
        (line,) = ax_legend.plot(
            [], [], "o-", linewidth=line_width, color=cfg["colors"][method], markersize=marker_size, label=method
        )
        legend_handles.append(line)

    # Add error bar entry if applicable
    if error_type:
        error_label = r"95\% CI" if "95" in error_type else error_type.upper()
        (error_line,) = ax_legend.plot([], [], color="gray", linestyle="--", linewidth=1.5 * scale, label=error_label)
        legend_handles.append(error_line)

    # Create vertical legend
    ax_legend.legend(
        handles=legend_handles, loc="center", frameon=True, fontsize=legend_fontsize, ncol=1
    )  # ncol=1 for vertical layout

    legend_path = output_path.replace(".png", "_legend.pdf")
    plt.savefig(legend_path, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)
    print(f"Standalone legend (vertical) saved to {legend_path}")

    # -------------------------------------------------------------------------
    # 5. Standalone legend PDF (horizontal layout)
    # -------------------------------------------------------------------------
    fig_legend_h, ax_legend_h = plt.subplots(figsize=(3.5 * scale, 0.6 * scale))
    ax_legend_h.axis("off")

    # Recreate dummy lines for horizontal legend
    legend_handles_h = []
    for method in ["LLM", "SAE"]:
        (line,) = ax_legend_h.plot(
            [], [], "o-", linewidth=line_width, color=cfg["colors"][method], markersize=marker_size, label=method
        )
        legend_handles_h.append(line)

    if error_type:
        error_label = r"95\% CI" if "95" in error_type else error_type.upper()
        (error_line,) = ax_legend_h.plot([], [], color="gray", linestyle="--", linewidth=1.5 * scale, label=error_label)
        legend_handles_h.append(error_line)

    # Create horizontal legend
    ax_legend_h.legend(
        handles=legend_handles_h, loc="center", frameon=True, fontsize=legend_fontsize, ncol=len(legend_handles_h)
    )

    legend_h_path = output_path.replace(".png", "_legend_horizontal.pdf")
    plt.savefig(legend_h_path, format="pdf", bbox_inches="tight")
    plt.close(fig_legend_h)
    print(f"Standalone legend (horizontal) saved to {legend_h_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args):
    plt.rcParams.update(iclr2024())

    # Load results
    with open(args.input_file, encoding="utf-8") as f:
        all_results = json.load(f)

    print(f"Loaded results from {args.input_file}")

    # Determine output paths
    base_path = args.input_file.replace(".json", "")
    accepted_only = not args.use_all_hypotheses
    hypo_suffix = "_accepted" if accepted_only else "_all"
    include_autorater = not args.no_autorater_scores

    # Generate aggregated metrics bar plot
    agg_plot_path = f"{base_path}_aggregated{hypo_suffix}.png"
    plot_aggregated_metrics_with_ablation(
        all_results,
        agg_plot_path,
        accepted_only=accepted_only,
        error_bar_type=args.error_bars,
    )

    # Generate frequency KDE plot
    freq_kde_path = f"{base_path}_frequency_kde{hypo_suffix}.png"
    plot_frequency_kde(all_results, freq_kde_path, accepted_only=accepted_only)

    # Generate accuracy KDE plot
    acc_kde_path = f"{base_path}_accuracy_kde{hypo_suffix}.png"
    plot_accuracy_kde(all_results, acc_kde_path, accepted_only=accepted_only)

    if include_autorater:
        # Generate interestingness KDE plot
        int_kde_path = f"{base_path}_interestingness_kde{hypo_suffix}.png"
        plot_interestingness_kde(all_results, int_kde_path, accepted_only=accepted_only)

        # Generate abstraction level KDE plot
        abs_kde_path = f"{base_path}_abstraction_kde{hypo_suffix}.png"
        plot_abstraction_kde(all_results, abs_kde_path, accepted_only=accepted_only)

    # Generate radar plot
    radar_path = f"{base_path}_radar{hypo_suffix}.png"
    plot_aggregated_metrics_radar(
        all_results,
        radar_path,
        accepted_only=accepted_only,
        include_autorater=include_autorater,
        error_type=args.error_bars,
    )

    # Generate extended plot with autorater scores
    if include_autorater:
        extended_plot_path = f"{base_path}_aggregated_extended{hypo_suffix}.png"
        plot_aggregated_metrics_extended(
            all_results,
            extended_plot_path,
            accepted_only=accepted_only,
            error_bar_type=args.error_bars,
        )

    # Generate ablation plots if requested
    if args.ablate_interestingness is not None:
        ablation_path = (
            f"{base_path}_aggregated_ablate_interestingness_lt{args.ablate_interestingness}{hypo_suffix}.png"
        )
        plot_aggregated_metrics_with_ablation(
            all_results,
            ablation_path,
            accepted_only=accepted_only,
            ablate_type="interestingness",
            ablate_threshold=args.ablate_interestingness,
            error_bar_type=args.error_bars,
        )

    if args.ablate_abstraction is not None:
        ablation_path = f"{base_path}_aggregated_ablate_abstraction_lt{args.ablate_abstraction}{hypo_suffix}.png"
        plot_aggregated_metrics_with_ablation(
            all_results,
            ablation_path,
            accepted_only=accepted_only,
            ablate_type="abstraction",
            ablate_threshold=args.ablate_abstraction,
            error_bar_type=args.error_bars,
        )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from hypothesis evaluation results")
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/hypothesis_evaluation_results.json",
        help="Input JSON file with evaluation results",
    )
    parser.add_argument(
        "--use_all_hypotheses",
        action="store_true",
        help="Use all hypotheses for metrics instead of only accepted ones (default: accepted only)",
    )
    parser.add_argument(
        "--error_bars",
        type=str,
        choices=["ci95", "std"],
        default="ci95",
        help="Show error bars on accuracy and frequency metrics: 'ci95' for 95%% CI, 'std' for standard deviation (default: ci95)",
    )
    parser.add_argument(
        "--no_autorater_scores",
        action="store_true",
        help="Exclude interestingness and abstraction scores from plots (default: included)",
    )
    parser.add_argument(
        "--ablate_interestingness",
        type=int,
        choices=[2, 3, 4, 5],
        default=None,
        help="Create ablation plot excluding hypotheses with interestingness score >= this value (2-5)",
    )
    parser.add_argument(
        "--ablate_abstraction",
        type=int,
        choices=[2, 3, 4, 5],
        default=None,
        help="Create ablation plot excluding hypotheses with abstraction score >= this value (2-5)",
    )
    args = parser.parse_args()

    main(args)
