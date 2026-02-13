"""
Evaluate hypotheses by comparing train and test judging results.

A hypothesis is accepted if the cluster specificity direction matches the judging direction
with a margin of at least alpha (symmetric).

This module produces a JSON file with evaluation results including autorater scores,
which can be used by plot_hypothesis_results.py for visualization.
"""

import json
from dataclasses import asdict, dataclass, field

from model_diffing.hypothesis_config import ExperimentConfig


@dataclass
class HypothesisResult:
    """Results for a single hypothesis."""

    hypothesis_id: str
    hypothesis_text: str
    # Cluster specificity (from training data)
    cluster_model_a_pct: float
    cluster_model_b_pct: float
    # Train judging results
    train_judge_model_a_pct: float
    train_judge_model_b_pct: float
    train_judge_na_pct: float
    train_total: int
    # Test judging results
    test_judge_model_a_pct: float
    test_judge_model_b_pct: float
    test_judge_na_pct: float
    test_total: int
    # Acceptance
    accepted: bool
    predicted_direction: str  # "A" or "B" based on cluster specificity
    train_judge_direction: str  # "A", "B", or "NA" based on judging
    test_judge_direction: str  # "A", "B", or "NA" based on judging
    # Metrics
    train_accuracy: float  # judge_pct_for_predicted / (judge_pct_a + judge_pct_b)
    test_accuracy: float
    train_frequency: float  # 1 - pct_na (how often hypothesis applies)
    test_frequency: float
    # Autorater scores (None if not available)
    interestingness_score: float | None = None
    abstraction_score: float | None = None
    # Additional metrics can be added here
    extra_metrics: dict = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Results for a single experiment (e.g., gemini_llm, gemini_sae)."""

    experiment_name: str
    method: str  # "llm" or "sae"
    alpha: float
    train_file: str
    test_file: str
    model_a: str | None = None  # Name of model A from cluster file
    model_b: str | None = None  # Name of model B from cluster file
    hypotheses: list[HypothesisResult] = field(default_factory=list)

    @property
    def num_accepted(self) -> int:
        return sum(1 for h in self.hypotheses if h.accepted)

    @property
    def num_total(self) -> int:
        return len(self.hypotheses)

    @property
    def acceptance_rate(self) -> float:
        if self.num_total == 0:
            return 0.0
        return self.num_accepted / self.num_total


def load_judging_results(filepath: str) -> tuple[dict[str, dict], str | None, str | None]:
    """Load judging results from JSON file.

    Returns:
        Tuple of (judging_results dict, model_a, model_b)
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("judging_results", {}), data.get("model_a"), data.get("model_b")


def get_direction(model_a_pct: float, model_b_pct: float, alpha: float = 0.0) -> str:
    """
    Determine direction based on percentages.
    Returns "A" if model_a > model_b + alpha, "B" if model_b > model_a + alpha, else "NA".
    """
    if model_a_pct > model_b_pct + alpha:
        return "A"
    elif model_b_pct > model_a_pct + alpha:
        return "B"
    else:
        return "NA"


def compute_accuracy(predicted_direction: str, judge_model_a_pct: float, judge_model_b_pct: float) -> float:
    """
    Compute accuracy: judge_pct_for_predicted_model / (judge_pct_a + judge_pct_b).
    Returns 0.0 if predicted_direction is NA or denominator is 0.
    """
    if predicted_direction == "NA":
        return 0.0
    denom = judge_model_a_pct + judge_model_b_pct
    if denom == 0:
        return 0.0
    if predicted_direction == "A":
        return judge_model_a_pct / denom
    else:  # "B"
        return judge_model_b_pct / denom


def compute_frequency(judge_na_pct: float) -> float:
    """
    Compute frequency: how often the hypothesis applies (1 - pct_na).
    """
    return 1.0 - judge_na_pct


def load_autorater_scores(
    train_file: str,
    score_type: str,
    autorater_file: str | None = None,
) -> dict[str, float]:
    """
    Load autorater scores from a file.

    When autorater_file is provided, loads directly from that path. Otherwise,
    auto-detects by replacing .json with the appropriate autorater suffix
    (matching what the autorater scripts write).

    Args:
        train_file: Path to the training judging results file (used for auto-detection)
        score_type: Either "interestingness" or "abstraction"
        autorater_file: Explicit path to autorater results. Overrides auto-detection.

    Returns:
        Dict mapping hypothesis_id (cluster_id) to score
    """
    if score_type not in ("interestingness", "abstraction"):
        raise ValueError(f"Unknown score_type: {score_type}")

    if autorater_file is None:
        # Auto-detect: autoraters write <base>_autorater_responses.json / <base>_abstraction_responses.json
        clean = train_file.replace("_fixed", "")
        if not clean.endswith(".json"):
            return {}
        base = clean[: -len(".json")]
        suffix = "_autorater_responses.json" if score_type == "interestingness" else "_abstraction_responses.json"
        autorater_file = base + suffix

    try:
        with open(autorater_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    # Build mapping from cluster_id to score
    scores = {}
    for item in data:
        cluster_id = item.get("cluster_id")
        aggregated = item.get("aggregated", {})
        score = aggregated.get("score")
        if cluster_id is not None and score is not None:
            scores[cluster_id] = score

    return scores


def evaluate_hypothesis(
    hypothesis_id: str,
    train_result: dict,
    test_result: dict,
    alpha: float,
    interestingness_score: float | None = None,
    abstraction_score: float | None = None,
) -> HypothesisResult:
    """Evaluate a single hypothesis."""
    # Cluster specificity direction (no alpha here - just raw direction)
    cluster_a = train_result["cluster_model_a_percentage"]
    cluster_b = train_result["cluster_model_b_percentage"]
    predicted_direction = get_direction(cluster_a, cluster_b, alpha=0.0)

    # Train judging direction (with alpha)
    train_a = train_result["pct_model_a"]
    train_b = train_result["pct_model_b"]
    train_na = train_result.get("pct_na", 0.0)
    train_judge_direction = get_direction(train_a, train_b, alpha=alpha)

    # Test judging direction (with alpha)
    test_a = test_result["pct_model_a"]
    test_b = test_result["pct_model_b"]
    test_na = test_result.get("pct_na", 0.0)
    test_judge_direction = get_direction(test_a, test_b, alpha=alpha)

    # Acceptance: predicted direction matches train judging direction
    # (test data is only used for metrics, not for filtering)
    if predicted_direction == "NA":
        raise ValueError("Predicted direction cannot be NA for acceptance check")
    accepted = predicted_direction == train_judge_direction

    # Compute accuracy and frequency metrics
    train_accuracy = compute_accuracy(predicted_direction, train_a, train_b)
    test_accuracy = compute_accuracy(predicted_direction, test_a, test_b)
    train_frequency = compute_frequency(train_na)
    test_frequency = compute_frequency(test_na)

    return HypothesisResult(
        hypothesis_id=hypothesis_id,
        hypothesis_text=train_result["cluster_hypothesis"],
        cluster_model_a_pct=cluster_a,
        cluster_model_b_pct=cluster_b,
        train_judge_model_a_pct=train_a,
        train_judge_model_b_pct=train_b,
        train_judge_na_pct=train_na,
        train_total=train_result.get("total", 0),
        test_judge_model_a_pct=test_a,
        test_judge_model_b_pct=test_b,
        test_judge_na_pct=test_na,
        test_total=test_result.get("total", 0),
        accepted=accepted,
        predicted_direction=predicted_direction,
        train_judge_direction=train_judge_direction,
        test_judge_direction=test_judge_direction,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        train_frequency=train_frequency,
        test_frequency=test_frequency,
        interestingness_score=interestingness_score,
        abstraction_score=abstraction_score,
    )


def evaluate_experiment(
    experiment_name: str,
    method: str,
    train_file: str,
    test_file: str,
    alpha: float,
    interestingness_scores: dict[str, float] | None = None,
    abstraction_scores: dict[str, float] | None = None,
) -> ExperimentResults:
    """Evaluate all hypotheses for an experiment."""
    train_results, model_a, model_b = load_judging_results(train_file)
    test_results, _, _ = load_judging_results(test_file)

    results = ExperimentResults(
        experiment_name=experiment_name,
        method=method,
        alpha=alpha,
        train_file=train_file,
        test_file=test_file,
        model_a=model_a,
        model_b=model_b,
    )

    # Check for mismatches between train and test hypothesis sets
    train_ids = set(train_results.keys())
    test_ids = set(test_results.keys())
    common_ids = train_ids & test_ids

    if train_ids != test_ids:
        only_in_train = train_ids - test_ids
        only_in_test = test_ids - train_ids
        print("  WARNING: Train/test hypothesis sets don't match!")
        if only_in_train:
            print(
                f"    Only in train ({len(only_in_train)}): {sorted(only_in_train, key=lambda x: int(x) if x.isdigit() else x)[:10]}{'...' if len(only_in_train) > 10 else ''}"
            )
        if only_in_test:
            print(
                f"    Only in test ({len(only_in_test)}): {sorted(only_in_test, key=lambda x: int(x) if x.isdigit() else x)[:10]}{'...' if len(only_in_test) > 10 else ''}"
            )
        print(f"    Using {len(common_ids)} common hypotheses")

    for hypothesis_id in sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x):
        # Get autorater scores for this hypothesis
        int_score = interestingness_scores.get(hypothesis_id) if interestingness_scores else None
        abs_score = abstraction_scores.get(hypothesis_id) if abstraction_scores else None

        hypothesis_result = evaluate_hypothesis(
            hypothesis_id=hypothesis_id,
            train_result=train_results[hypothesis_id],
            test_result=test_results[hypothesis_id],
            alpha=alpha,
            interestingness_score=int_score,
            abstraction_score=abs_score,
        )
        results.hypotheses.append(hypothesis_result)

    return results


def evaluate_all(
    experiments: list[ExperimentConfig],
    alpha: float = 0.01,
    output_file: str | None = None,
) -> dict:
    """Evaluate all experiments. Returns results dict."""
    all_results = {}

    for exp in experiments:
        key = exp["key"]
        print(f"Evaluating {key}...")

        # Load autorater scores (prefer explicit file paths, fall back to auto-detection)
        print("  Loading autorater scores...")
        interestingness_scores = load_autorater_scores(
            exp["train_file"], "interestingness", autorater_file=exp.get("interestingness_file")
        )
        abstraction_scores = load_autorater_scores(
            exp["train_file"], "abstraction", autorater_file=exp.get("abstraction_file")
        )

        if interestingness_scores:
            print(f"    Loaded {len(interestingness_scores)} interestingness scores")
        if abstraction_scores:
            print(f"    Loaded {len(abstraction_scores)} abstraction scores")

        results = evaluate_experiment(
            experiment_name=exp["name"],
            method=exp["method"],
            train_file=exp["train_file"],
            test_file=exp["test_file"],
            alpha=alpha,
            interestingness_scores=interestingness_scores,
            abstraction_scores=abstraction_scores,
        )

        all_results[key] = {
            "experiment_name": results.experiment_name,
            "method": results.method,
            "alpha": results.alpha,
            "train_file": results.train_file,
            "test_file": results.test_file,
            "model_a": results.model_a,
            "model_b": results.model_b,
            "num_hypotheses": results.num_total,
            "num_accepted": results.num_accepted,
            "acceptance_rate": results.acceptance_rate,
            "hypotheses": [asdict(h) for h in results.hypotheses],
        }

        print(f"  Total hypotheses: {results.num_total}")
        print(f"  Accepted: {results.num_accepted} ({results.acceptance_rate:.1%})")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<20} {'Hypotheses':>12} {'Accepted':>10} {'Rate':>10}")
    print("-" * 60)
    for key, res in all_results.items():
        print(f"{key:<20} {res['num_hypotheses']:>12} {res['num_accepted']:>10} {res['acceptance_rate']:>9.1%}")

    return all_results


if __name__ == "__main__":
    import argparse

    from model_diffing.hypothesis_config import EXPERIMENTS

    parser = argparse.ArgumentParser(description="Evaluate hypotheses by comparing train and test judging results")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Minimum margin for judging direction to be considered significant (default: 0.01)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="hypothesis_evaluation_results.json",
        help="Output file path for results",
    )
    args = parser.parse_args()

    evaluate_all(
        experiments=EXPERIMENTS,
        alpha=args.alpha,
        output_file=args.output_file,
    )
