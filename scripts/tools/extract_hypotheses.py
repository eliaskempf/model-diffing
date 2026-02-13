"""Extract hypotheses from evaluation results into separate text files."""

import json

INPUT_FILE = "output/hypothesis_evaluation_results.json"
OUTPUT_DIR = "output"

EXPERIMENTS = ["qwen_em", "gemma_gender", "gemini"]
METHODS = ["llm", "sae"]

with open(INPUT_FILE) as f:
    data = json.load(f)

for exp in EXPERIMENTS:
    for method in METHODS:
        key = f"{exp}_{method}"
        prefix = method.upper()
        exp_data = data[key]
        model_a = exp_data["model_a"]
        model_b = exp_data["model_b"]

        hypotheses = []
        for h in exp_data["hypotheses"]:
            if not h["accepted"]:
                continue
            text = h["hypothesis_text"]
            direction = h["predicted_direction"]
            if direction == "A":
                model, other = model_a, model_b
            else:
                model, other = model_b, model_a
            text = text.replace("<MODEL>", model).replace("<OTHER MODEL>", other)
            hypotheses.append(f"{prefix}: {text}")

        output_path = f"{OUTPUT_DIR}/hypotheses_{key}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(hypotheses))
        print(f"Wrote {len(hypotheses)} hypotheses to {output_path}")
