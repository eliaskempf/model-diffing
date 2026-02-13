import asyncio
import os

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_diffing.data import MixtureDataset, WildChat
from model_diffing.model import ModelWrapper
from model_diffing.utils import ResponseDict

test_prompts = [
    [{"role": "user", "content": "Once upon a time in a land far, far away,"}],
    [{"role": "user", "content": "In the future, humanity has discovered the secrets of"}],
    [{"role": "user", "content": "The quick brown fox jumps over"}],
    [{"role": "user", "content": "Artificial intelligence is transforming the world by"}],
    [{"role": "user", "content": "The mysteries of the universe can be unraveled through"}],
]


class DictListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_responses(
    model_name: str,
    model_revision: str,
    base_model: str,
    open_router_api_key: str,
    prompts: str,
    limit_samples: int,
    test_shift: int,
    output_dir: str,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    enable_thinking: bool,
    regenerate: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    model_rev_name = model_name.replace("/", "_")
    if model_revision is not None:
        model_rev_name += f"_{model_revision}"
    suffix = "_test" if test_shift != 0 else ""
    response_file = os.path.join(output_dir, f"{model_rev_name}_{prompts}{suffix}_responses.jsonl")

    if os.path.exists(response_file) and not regenerate:
        print(f"Responses already exist in {output_dir}, loading from disk.")
        response_dict = ResponseDict.from_jsonl(response_file)
    else:
        if os.path.exists(response_file):
            print(f"Responses file {response_file} exists but regenerate is set to True. Regenerating responses.")
            os.remove(response_file)
        response_dict = ResponseDict()

    model = ModelWrapper(
        model_name=model_name, model_revision=model_revision, base_model=base_model, api_key=open_router_api_key
    )
    open_router = model.open_router

    if test_shift != 0 and prompts != "wild_chat":
        raise ValueError("test_shift is only supported for wild_chat prompts.")
    if prompts == "test_prompts":
        data = DictListDataset(test_prompts)
    elif prompts == "wild_chat":
        data = WildChat(limit_samples=limit_samples + test_shift, max_user_turns=1)
        if test_shift > 0:
            data.dataset = data.dataset.select(range(test_shift, limit_samples + test_shift))
            if len(data) != limit_samples:
                raise ValueError(f"Expected {limit_samples} samples after shift, got {len(data)}")
            print(f"Shifted wild chat dataset by {test_shift} samples.")
    elif prompts == "mixture":
        data = MixtureDataset(
            samples_per_dataset={
                "wild_chat": 3320,
                "deception_bench": 180,
                "aegis_safety": 1000,
                "gsm8k": 500,
            },
            shuffle=False,
            seed=seed,
        )
    else:
        raise NotImplementedError("Only 'test_prompts' is implemented for prompts input.")

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    with open(response_file, "a", encoding="utf-8") as f_response_file:
        for batch in tqdm(loader, desc="Generating responses", total=len(loader)):
            # only compute for those not already in response_dict
            filtered_batch = [conversation for conversation in batch if conversation not in response_dict]

            if open_router:  # exclude empty strings for open router
                filtered_batch = [
                    conversation
                    for conversation in filtered_batch
                    if any(message["content"].strip() != "" for message in conversation)
                ]

            if len(filtered_batch) == 0:
                continue

            if open_router:
                responses = asyncio.run(
                    model.generate_async(
                        filtered_batch,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        seed=seed,
                    )
                )
            else:
                responses = model.generate(
                    filtered_batch,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    cache_implementation="dynamic",
                    seed=seed,
                )

            for conversation, response in zip(filtered_batch, responses):
                if not response:
                    continue
                response_dict.insert_and_save(conversation, response, f_response_file)
            f_response_file.flush()

    return response_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of LLM to generate responses from")
    parser.add_argument(
        "--model_revision", type=str, default=None, help="Revision of the model to generate responses from"
    )
    parser.add_argument(
        "--base_model", type=str, default=None, help="Base model name if using OpenRouter with PEFT weights"
    )
    parser.add_argument(
        "--open_router_api_key", type=str, default=None, help="API key for OpenRouter if using OpenRouter mode"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="test_prompts",
        help="Path to file containing prompts or 'test_prompts' for default prompts",
        choices=["test_prompts", "wild_chat", "mixture"],
    )
    parser.add_argument("--limit_samples", type=int, default=1200, help="Limit number of samples from dataset")
    parser.add_argument("--test_shift", type=int, default=0, help="Shift index for test prompts")
    parser.add_argument(
        "--output_dir", type=str, default="output/llm-responses", help="Directory to save generated responses"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode during generation")
    parser.add_argument(
        "--regenerate", action="store_true", help="Whether to regenerate responses even if they already exist"
    )

    args = parser.parse_args()

    generate_responses(
        model_name=args.model_name,
        model_revision=args.model_revision,
        base_model=args.base_model,
        open_router_api_key=args.open_router_api_key,
        prompts=args.prompts,
        limit_samples=args.limit_samples,
        test_shift=args.test_shift,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
        regenerate=args.regenerate,
    )
