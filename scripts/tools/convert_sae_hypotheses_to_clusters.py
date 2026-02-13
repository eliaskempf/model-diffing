#!/usr/bin/env python3
"""
Convert hypothesis JSON files from the SAE format to the clusters JSONL format.

Source format (JSON):
- query, differences, significant_features, dataset1_path, dataset2_paths,
  target_model, other_models, features_csv_path, precomputed_features_path, output_file
- Each difference has: dataset, description, feature_ids, examples, percentage_difference, confidence

Target format (JSONL):
- cluster_id, model_a_percentage, model_b_percentage, model_a, model_b,
  hypothesis, hashes, difference_index, sentences
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

# System prompt for LLM-based hypothesis reformatting
REFORMAT_SYSTEM_PROMPT = """\
You are a technical writing assistant that reformats behavioral hypotheses about language models.

Your task is to convert hypotheses from a "response-centric" format to a "model-centric, comparative" format.

INPUT FORMAT (response-centric):
- Starts with "This response..."
- Describes what the response does/contains
- Example: "This response provides advice that is counter-intuitive, potentially harmful, or goes against common best practices."

OUTPUT FORMAT (model-centric, comparative):
- Starts with "<MODEL>" (literal placeholder)
- Describes what the model does MORE of or TENDS to do compared to another model
- Uses comparative language where appropriate (e.g., "provides more", "uses more", "tends to", "is more likely to")
- Should be concise and clear
- Examples:
  - "<MODEL> provides more detailed explanations, including theoretical background and step-by-step breakdowns."
  - "<MODEL> uses more tables to present structured information."
  - "<MODEL> tends to provide counter-intuitive or potentially harmful advice."
  - "<MODEL> is more likely to use markdown formatting like headers and bullet points."

GUIDELINES:
1. Always start with "<MODEL>" (this exact string, not a model name)
2. Convert passive/response-centric language to active/model-centric comparative language
3. Use comparative words like "more", "tends to", "is more likely to", "prefers" where natural
4. Preserve the core meaning - don't add or remove key behavioral characteristics
5. Make it more concise if the original is verbose, but keep important nuances
6. If the original hypothesis doesn't make sense as a comparative model behavior, output it with <MODEL> prefix but keep it descriptive
7. Output ONLY the reformatted hypothesis, nothing else - no explanations, no quotes"""

REFORMAT_USER_PROMPT_TEMPLATE = """\
Reformat this hypothesis to the model-centric format:

{description}"""


def extract_model_name(path: str) -> str:
    """Extract model name from a path like '../workspace/llm-responses/google_gemini-2.5-flash-lite/...'"""
    # Match the pattern after llm-responses/
    match = re.search(r"llm-responses/([^/]+)/", path)
    if match:
        # Convert underscore-separated to slash-separated (e.g., google_gemini -> google/gemini)
        name = match.group(1)
        # Replace first underscore with slash for org/model format
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
        return name
    return path


def format_hypothesis_simple(description: str) -> str:
    """Simple rule-based hypothesis formatting (no LLM)."""
    if not description:
        return "Hypothesis: <MODEL> "
    # Just lowercase first letter and prepend
    return f"Hypothesis: <MODEL> {description[0].lower() + description[1:]}"


async def reformat_hypotheses_with_llm(
    descriptions: list[str],
    model_name: str = "google/gemini-2.5-flash",
) -> list[str]:
    """
    Reformat hypotheses using an LLM via CachedModelWrapper.

    Args:
        descriptions: List of original hypothesis descriptions
        model_name: OpenRouter model to use for reformatting

    Returns:
        List of reformatted hypotheses (with "Hypothesis: " prefix)
    """
    # Import here to avoid dependency when not using LLM
    from model_diffing.model_cached import CachedModelWrapper

    model = CachedModelWrapper(model_name)

    # Build conversations for each description
    conversations = []
    for desc in descriptions:
        conversations.append(
            [
                {"role": "system", "content": REFORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": REFORMAT_USER_PROMPT_TEMPLATE.format(description=desc)},
            ]
        )

    # Generate reformatted hypotheses
    print(f"Reformatting {len(descriptions)} hypotheses using {model_name}...")
    responses = await model.generate_async(
        conversations=conversations,
        max_new_tokens=256,
        temperature=0.0,
    )

    # Post-process responses
    reformatted = []
    for i, response in enumerate(responses):
        response = response.strip()
        # Ensure it starts with <MODEL>
        if not response.startswith("<MODEL>"):
            # Try to fix common issues
            if response.lower().startswith("the model"):
                response = "<MODEL>" + response[9:]
            else:
                # Fall back to simple formatting
                response = (
                    f"<MODEL> {descriptions[i][0].lower() + descriptions[i][1:]}" if descriptions[i] else "<MODEL> "
                )
        reformatted.append(f"Hypothesis: {response}")

    print(f"LLM reformatting complete. Cost: ${model.running_cost:.4f}")
    return reformatted


def convert_hypothesis_to_cluster(
    difference: dict, cluster_id: int, model_a: str, model_b: str, reformatted_hypothesis: str | None = None
) -> dict:
    """Convert a single difference entry to cluster format."""
    # Determine which model this difference applies to based on 'dataset' field
    # 'target' means it applies to model_a (the target model)
    # 'other' means it applies to model_b (the other model)
    # The percentage_difference sign encodes direction:
    #   - positive for 'target' entries (model_a has feature more)
    #   - negative for 'other' entries (model_b has feature more)
    # We take absolute value since direction is already encoded by 'dataset'
    is_target = difference.get("dataset", "").lower() == "target"

    percentage = abs(difference.get("percentage_difference", 0.0))

    # Set percentages: if difference is for target, model_a gets the percentage
    if is_target:
        model_a_percentage = percentage
        model_b_percentage = 0.0
    else:
        model_a_percentage = 0.0
        model_b_percentage = percentage

    # Use reformatted hypothesis if provided, otherwise use simple formatting
    if reformatted_hypothesis is not None:
        hypothesis = reformatted_hypothesis
    else:
        description = difference.get("description", "")
        hypothesis = format_hypothesis_simple(description)

    return {
        "cluster_id": cluster_id,
        "model_a_percentage": model_a_percentage,
        "model_b_percentage": model_b_percentage,
        "model_a": model_a,
        "model_b": model_b,
        "hypothesis": hypothesis,
        "hashes": [],
        "difference_index": [],
        "sentences": [],
    }


async def convert_file_async(
    input_path: Path,
    output_path: Path | None = None,
    use_llm: bool = False,
    llm_model: str = "google/gemini-2.5-flash",
) -> Path:
    """Convert a hypothesis JSON file to clusters JSONL format."""
    with open(input_path) as f:
        data = json.load(f)

    # Extract model names from paths in the file
    target_model_path = data.get("target_model", "") or data.get("dataset1_path", "")
    other_models_paths = data.get("other_models", []) or data.get("dataset2_paths", [])

    model_a = extract_model_name(target_model_path)
    model_b = extract_model_name(other_models_paths[0]) if other_models_paths else "unknown"

    differences = data.get("differences", [])

    # Optionally reformat hypotheses using LLM
    reformatted_hypotheses = None
    if use_llm:
        descriptions = [d.get("description", "") for d in differences]
        reformatted_hypotheses = await reformat_hypotheses_with_llm(descriptions, llm_model)

    # Convert each difference
    clusters = []
    for idx, diff in enumerate(differences):
        reformatted = reformatted_hypotheses[idx] if reformatted_hypotheses else None
        cluster = convert_hypothesis_to_cluster(diff, idx, model_a, model_b, reformatted)
        clusters.append(cluster)

    # Determine output path
    if output_path is None:
        output_path = input_path.with_suffix(".jsonl").with_name(
            input_path.stem.replace("hypotheses_", "") + "_clusters.jsonl"
        )

    # Write JSONL output
    with open(output_path, "w") as f:
        for cluster in clusters:
            f.write(json.dumps(cluster) + "\n")

    print(f"Converted {len(clusters)} hypotheses from {input_path}")
    print(f"Output written to: {output_path}")
    print(f"Models: {model_a} vs {model_b}")

    return output_path


def convert_file(input_path: Path, output_path: Path | None = None) -> Path:
    """Synchronous wrapper for convert_file_async (without LLM)."""
    return asyncio.run(convert_file_async(input_path, output_path, use_llm=False))


def main():
    parser = argparse.ArgumentParser(description="Convert hypothesis JSON files to clusters JSONL format")
    parser.add_argument("input_file", type=Path, help="Input hypothesis JSON file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output JSONL file (default: derived from input filename)"
    )
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to reformat hypotheses to match target format")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="google/gemini-2.5-flash",
        help="OpenRouter model to use for reformatting (default: google/gemini-2.5-flash)",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    asyncio.run(
        convert_file_async(
            args.input_file,
            args.output,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
        )
    )
    return 0


if __name__ == "__main__":
    exit(main())
