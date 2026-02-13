#!/usr/bin/env bash
# Recompute evaluation metrics for all experiments using pre-existing judging + autorater files.
# Copies judge + autorater files into each experiment's eval dir with standardized names,
# then runs evaluation. Autorater disk caching sees the copied files and skips API calls.
# Merges individual results into a single JSON for plotting scripts.
set -euo pipefail

OUT=output
EVAL_OUT="$OUT/eval_recompute"
ALPHA=0.01

# Copy source files into eval dir with standardized names, then run evaluation.
# Args: dir_name experiment_name method train_file test_file interestingness_file abstraction_file
run_experiment() {
    local dir_name="$1"
    local experiment_name="$2"
    local method="$3"
    local train_file="$4"
    local test_file="$5"
    local interestingness_file="$6"
    local abstraction_file="$7"

    local dir="$EVAL_OUT/$dir_name"
    mkdir -p "$dir"

    echo "=== ${dir_name} ==="

    # Copy judge + autorater files with standardized names
    cp "$train_file"            "$dir/judging_results_train.json"
    cp "$test_file"             "$dir/judging_results_test.json"
    cp "$interestingness_file"  "$dir/interestingness_scores.json"
    cp "$abstraction_file"      "$dir/abstraction_scores.json"

    uv run python scripts/run_eval.py \
        --train_judge_results "$dir/judging_results_train.json" \
        --test_judge_results  "$dir/judging_results_test.json" \
        --experiment_name "$experiment_name" --method "$method" --alpha "$ALPHA" \
        --output_dir "$dir"
}

# --- gemini (flash lite) ---
GEMINI="$OUT/gemini_flash_lite"
GEMINI_LLM_BASE="$GEMINI/google_gemini-2.5-flash-lite__google_gemini-2.5-flash-lite-preview-09-2025_clustering"

run_experiment gemini_sae gemini sae \
    "$GEMINI/gemini_sae_clustering_judging_results_fixed.json" \
    "$GEMINI/gemini_sae_clustering_judging_test_results_fixed.json" \
    "$GEMINI/gemini_sae_clustering_judging_results_autorater_responses.json" \
    "$GEMINI/gemini_sae_clustering_judging_results_abstraction_responses.json"

run_experiment gemini_llm gemini llm \
    "${GEMINI_LLM_BASE}_judging_results_full.json" \
    "${GEMINI_LLM_BASE}_judging_test_results.json" \
    "${GEMINI_LLM_BASE}_judging_results_full_autorater_responses.json" \
    "${GEMINI_LLM_BASE}_judging_results_full_abstraction_responses.json"

# --- gemma_gender ---
GEMMA="$OUT/gemma_gender"
GEMMA_LLM_BASE="$GEMMA/google_gemma-2-9b-it__bcywinski_gemma-2-9b-it-user-female_clustering"

run_experiment gemma_gender_sae gemma_gender sae \
    "$GEMMA/gender_sae_clustering_judging_results_fixed.json" \
    "$GEMMA/gender_sae_clustering_judging_test_results_fixed.json" \
    "$GEMMA/gender_sae_clustering_judging_results_autorater_responses.json" \
    "$GEMMA/gender_sae_clustering_judging_results_abstraction_responses.json"

run_experiment gemma_gender_llm gemma_gender llm \
    "${GEMMA_LLM_BASE}_judging_results_full.json" \
    "${GEMMA_LLM_BASE}_judging_test_results.json" \
    "${GEMMA_LLM_BASE}_judging_results_full_autorater_responses.json" \
    "${GEMMA_LLM_BASE}_judging_results_full_abstraction_responses.json"

# --- qwen_em ---
QWEN="$OUT/qwen_em"
QWEN_LLM_BASE="$QWEN/Qwen_Qwen2.5-7B-Instruct__ModelOrganismsForEM_Qwen2.5-7B-Instruct_risky-financial-advice_clustering"

run_experiment qwen_em_sae qwen_em sae \
    "$QWEN/misalignment_sae_clustering_judging_results_fixed.json" \
    "$QWEN/misalignment_sae_clustering_judging_test_results_fixed.json" \
    "$QWEN/misalignment_sae_clustering_judging_results_autorater_responses.json" \
    "$QWEN/misalignment_sae_clustering_judging_results_abstraction_responses.json"

run_experiment qwen_em_llm qwen_em llm \
    "${QWEN_LLM_BASE}_judging_results_full.json" \
    "${QWEN_LLM_BASE}_judging_test_results.json" \
    "${QWEN_LLM_BASE}_judging_results_full_autorater_responses.json" \
    "${QWEN_LLM_BASE}_judging_results_full_abstraction_responses.json"

# --- Merge all into single JSON for plotting ---
echo ""
echo "=== Merging results ==="
uv run python scripts/tools/merge_eval_results.py \
    --input "$EVAL_OUT/gemini_sae/hypothesis_evaluation_results.json" \
    --input "$EVAL_OUT/gemini_llm/hypothesis_evaluation_results.json" \
    --input "$EVAL_OUT/gemma_gender_sae/hypothesis_evaluation_results.json" \
    --input "$EVAL_OUT/gemma_gender_llm/hypothesis_evaluation_results.json" \
    --input "$EVAL_OUT/qwen_em_sae/hypothesis_evaluation_results.json" \
    --input "$EVAL_OUT/qwen_em_llm/hypothesis_evaluation_results.json" \
    --output "$EVAL_OUT/all_hypothesis_evaluation_results.json"

echo ""
echo "Done! Merged results: $EVAL_OUT/all_hypothesis_evaluation_results.json"
