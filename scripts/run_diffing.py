import time
from functools import partial

import torch

from model_diffing.pipeline.cluster_sentence_embeddings import cluster_sentence_embeddings
from model_diffing.pipeline.compute_sentence_embeddings import compute_sentence_embeddings
from model_diffing.pipeline.generate import generate_responses
from model_diffing.pipeline.llm_aggregation import llm_aggregation
from model_diffing.pipeline.llm_diffing import compare_llm_responses


def main(args):
    start = time.time()

    if bool(args.responses_file_a) != bool(args.responses_file_b):
        raise ValueError("Provide both --responses_file_a and --responses_file_b, or neither.")
    if args.responses_file_a and args.test_limit_samples > 0:
        raise ValueError("--test_limit_samples is incompatible with --responses_file_a/b.")

    if args.responses_file_a and args.responses_file_b:
        responses_file_a = args.responses_file_a
        responses_file_b = args.responses_file_b
    else:
        _generate_train = partial(
            generate_responses,
            base_model=args.base_model,
            open_router_api_key=args.open_router_api_key,
            prompts=args.prompts,
            limit_samples=args.limit_samples,
            test_shift=0,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
            regenerate=args.regenerate_responses,
        )
        responses_file_a = _generate_train(
            model_name=args.model_name_a,
            model_revision=args.model_revision_a,
        )
        responses_file_b = _generate_train(
            model_name=args.model_name_b,
            model_revision=args.model_revision_b,
        )

    if args.test_limit_samples > 0:
        _generate_test = partial(
            generate_responses,
            base_model=args.base_model,
            open_router_api_key=args.open_router_api_key,
            prompts=args.prompts,
            limit_samples=args.test_limit_samples,
            test_shift=args.test_shift,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
            regenerate=args.regenerate_responses,
        )
        test_responses_file_a = _generate_test(
            model_name=args.model_name_a,
            model_revision=args.model_revision_a,
        )
        test_responses_file_b = _generate_test(
            model_name=args.model_name_b,
            model_revision=args.model_revision_b,
        )
        print(f"Test responses: {test_responses_file_a}, {test_responses_file_b}")

    # then run LLM to compare responses
    responses_diff_file = compare_llm_responses(
        comparator_model_name=args.comparator_model_name,
        open_router_api_key=args.open_router_api_key,
        responses_a=responses_file_a,
        responses_b=responses_file_b,
        model_a=args.model_name_a,
        model_b=args.model_name_b,
        limit_samples=args.limit_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        regenerate=args.regenerate_diffs,
    )

    if args.llm_aggregation:
        llm_aggregation(
            difference_results=responses_diff_file,
            model_name=args.comparator_model_name,
            open_router_api_key=args.open_router_api_key,
        )
    else:
        # compute sentence embeddings for the differences
        sentence_embedding_files = compute_sentence_embeddings(
            difference_results=responses_diff_file,
            sentence_embedder=args.sentence_embedder,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            regenerate=args.regenerate_embeddings,
        )

        # cluster the sentence embeddings and generate hypotheses
        cluster_sentence_embeddings(
            model_name=args.comparator_model_name,
            input_files=sentence_embedding_files,
            open_router_api_key=args.open_router_api_key,
        )

    end = time.time()
    print(f"Total time taken: {end - start} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_a", type=str, required=True, help="Name of LLM to generate responses from")
    parser.add_argument(
        "--model_revision_a", type=str, default=None, help="Revision of the model to generate responses from"
    )
    parser.add_argument("--model_name_b", type=str, required=True, help="Name of LLM to generate responses from")
    parser.add_argument(
        "--model_revision_b", type=str, default=None, help="Revision of the model to generate responses from"
    )
    parser.add_argument(
        "--comparator_model_name", type=str, default="google/gemini-2.5-flash", help="Name of LLM to act as comparator"
    )
    parser.add_argument(
        "--sentence_embedder",
        type=str,
        default="nvidia/llama-embed-nemotron-8b",
        help="Pre-trained model name for sentence embeddings",
    )
    parser.add_argument(
        "--open_router_api_key", type=str, default=None, help="API key for OpenRouter if using OpenRouter mode"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="wild_chat",
        help="Prompt dataset to use",
        choices=["test_prompts", "wild_chat", "mixture"],
    )
    parser.add_argument("--limit_samples", type=int, default=1000, help="Limit number of samples from dataset")
    parser.add_argument(
        "--output_dir", type=str, default="output/llm-responses", help="Directory to save generated responses"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument(
        "--base_model", type=str, default=None, help="Base model name if using OpenRouter with PEFT weights"
    )
    parser.add_argument(
        "--test_shift", type=int, default=10000, help="WildChat offset for test prompts (default: 10000)"
    )
    parser.add_argument(
        "--test_limit_samples", type=int, default=500, help="Number of test samples to generate (0 to skip)"
    )
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode during generation")
    parser.add_argument("--llm_aggregation", action="store_true", help="Enable LLM aggregation")
    parser.add_argument(
        "--regenerate_responses", action="store_true", help="Whether to regenerate responses even if they already exist"
    )
    parser.add_argument(
        "--regenerate_diffs", action="store_true", help="Whether to regenerate diffs even if they already exist"
    )
    parser.add_argument(
        "--regenerate_embeddings",
        action="store_true",
        help="Whether to regenerate sentence embeddings even if they already exist",
    )
    parser.add_argument(
        "--responses_file_a",
        type=str,
        default=None,
        help="Path to pre-generated responses for model A (skips generation)",
    )
    parser.add_argument(
        "--responses_file_b",
        type=str,
        default=None,
        help="Path to pre-generated responses for model B (skips generation)",
    )
    args = parser.parse_args()

    main(args)
