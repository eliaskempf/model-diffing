import csv
import random

from datasets import load_dataset
from torch.utils.data import IterableDataset


class WildChat(IterableDataset):
    def __init__(self, limit_samples: int | None = None, max_user_turns: int | None = None, english_only: bool = False):
        split = "train" if limit_samples is None else f"train[:{2 * limit_samples}]"
        dataset = load_dataset("allenai/WildChat", split=split)
        dataset = dataset.filter(lambda s: (not english_only) or s["language"] == "English")
        self.dataset = dataset.select(range(limit_samples)) if limit_samples is not None else dataset
        self.max_user_turns = max_user_turns
        assert max_user_turns is None or max_user_turns > 0, "max_user_turns must be positive or None"

    def get_sample(self, index: int):
        return self.dataset[index]

    def __iter__(self):
        for sample in self.dataset:
            conversation = sample["conversation"]
            assert [turn["role"] for turn in conversation] == ["user", "assistant"] * (len(conversation) // 2), (
                "Unexpected turn roles"
            )
            assert len(conversation) % 2 == 0

            if self.max_user_turns is not None:
                conversation = conversation[: 2 * self.max_user_turns]
            conversation = conversation[:-1]
            assert conversation[-1]["role"] == "user", "Last turn must be from user"

            conversation = [{"role": turn["role"], "content": turn["content"]} for turn in conversation]
            yield conversation

    def __len__(self):
        return len(self.dataset)


class DeceptionBench(IterableDataset):
    def __init__(self, concat_system_prompt: bool = False):
        self.dataset = load_dataset("PKU-Alignment/DeceptionBench", split="test")
        self.concat_system_prompt = concat_system_prompt

    def get_sample(self, index: int):
        return self.dataset[index]

    def __iter__(self):
        for sample in self.dataset:
            content = sample["outer_prompt"]
            if self.concat_system_prompt:
                content = sample["system_prompt"] + "\n" + content
            yield [{"role": "user", "content": content}]

    def __len__(self):
        return len(self.dataset)


class AegisAISafety(IterableDataset):
    def __init__(self, split: str = "test", limit_samples: int | None = None):
        split = split if limit_samples is None else f"{split}[:{limit_samples}]"
        self.dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split)

    def get_sample(self, index: int):
        return self.dataset[index]

    def __iter__(self):
        for sample in self.dataset:
            yield [{"role": "user", "content": sample["prompt"]}]

    def __len__(self):
        return len(self.dataset)


class GSM8K(IterableDataset):
    def __init__(self, split: str = "test", config: str = "main", limit_samples: int | None = None):
        split = split if limit_samples is None else f"{split}[:{limit_samples}]"
        assert config in ["main", "socratic"], "config must be 'main' or 'socratic'"
        self.dataset = load_dataset("gsm8k", config, split=split)

    def get_sample(self, index: int):
        return self.dataset[index]

    def __iter__(self):
        for sample in self.dataset:
            yield [{"role": "user", "content": sample["question"]}]

    def __len__(self):
        return len(self.dataset)


class MixtureDataset(IterableDataset):
    def __init__(
        self,
        samples_per_dataset: dict[str, int],
        shuffle: bool = False,
        seed: int = 42,
        wild_chat_kwargs: dict | None = None,
        deception_bench_kwargs: dict | None = None,
        aegis_safety_kwargs: dict | None = None,
        gsm8k_kwargs: dict | None = None,
    ):
        if gsm8k_kwargs is None:
            gsm8k_kwargs = {"split": "test", "config": "main"}
        if aegis_safety_kwargs is None:
            aegis_safety_kwargs = {"split": "test"}
        if deception_bench_kwargs is None:
            deception_bench_kwargs = {"concat_system_prompt": False}
        if wild_chat_kwargs is None:
            wild_chat_kwargs = {"english_only": True, "max_user_turns": 1}
        assert all(key in samples_per_dataset for key in ["wild_chat", "deception_bench", "aegis_safety", "gsm8k"]), (
            "Expected exactly the keys: 'wild_chat', 'deception_bench', 'aegis_safety', 'gsm8k'"
        )
        assert samples_per_dataset["deception_bench"] == 180, "DeceptionBench must have exactly 180 samples"

        wild_chat_kwargs["limit_samples"] = samples_per_dataset["wild_chat"]
        aegis_safety_kwargs["limit_samples"] = samples_per_dataset["aegis_safety"]
        gsm8k_kwargs["limit_samples"] = samples_per_dataset["gsm8k"]

        self.datasets = {
            "wild_chat": WildChat(**wild_chat_kwargs),
            "deception_bench": DeceptionBench(**deception_bench_kwargs),
            "aegis_safety": AegisAISafety(**aegis_safety_kwargs),
            "gsm8k": GSM8K(**gsm8k_kwargs),
        }

        self.dataset_samples = {name: list(self.datasets[name]) for name in self.datasets}
        assert all(len(self.dataset_samples[name]) == samples_per_dataset[name] for name in self.datasets), (
            "Mismatch in number of samples per dataset"
        )

        self.samples = []
        self.original_datasets = []
        for name in self.datasets:
            self.samples.extend(self.dataset_samples[name])
            self.original_datasets.extend([name] * len(self.dataset_samples[name]))

        if shuffle:
            random.seed(seed)
            combined = list(zip(self.samples, self.original_datasets))
            random.shuffle(combined)
            self.samples, self.original_datasets = zip(*combined)
            self.samples = list(self.samples)
            self.original_datasets = list(self.original_datasets)

    def get_sample(self, index: int):
        return self.samples[index]

    def __iter__(self):
        for sample in self.samples:
            assert isinstance(sample, list), "Each sample should be a list of turns"
            assert all("role" in turn and "content" in turn for turn in sample), (
                "Each turn should have 'role' and 'content'"
            )
            assert len(sample) == 1, "Each sample should contain exactly one user turn"
            yield sample

    def __len__(self):
        return len(self.samples)


def load_prompts_from_csv(
    csv_path: str,
    prompt_column: int = 2,
    id_column: int = 0,
    score_column: int = 1,
    limit: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """Load prompts from a CSV file.

    Args:
        csv_path: Path to CSV file
        prompt_column: 0-indexed column containing prompts (default: 2 = third column)
        id_column: 0-indexed column containing IDs (default: 0 = first column)
        score_column: 0-indexed column containing relevance scores (default: 1 = second column)
        limit: Maximum number of prompts to load
        min_score: Minimum score threshold (only include prompts with score >= min_score)

    Returns:
        List of dicts with 'id', 'prompt', and 'score' keys
    """
    prompts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header

        for row in reader:
            if len(row) > max(prompt_column, id_column, score_column):
                try:
                    score = float(row[score_column])
                except ValueError:
                    score = None

                if min_score is not None and (score is None or score < min_score):
                    continue

                prompts.append(
                    {
                        "id": row[id_column],
                        "prompt": row[prompt_column],
                        "score": score,
                    }
                )

            if limit and len(prompts) >= limit:
                break

    return prompts


if __name__ == "__main__":
    dataset = MixtureDataset(
        samples_per_dataset={
            "wild_chat": 1250,
            "deception_bench": 180,
            "aegis_safety": 320,
            "gsm8k": 250,
        },
        shuffle=False,
    )

    from model_diffing.utils import ResponseDict

    hashes = []
    for sample in dataset:
        hashes.append(ResponseDict._hash_key(sample))

    print(f"Total samples: {len(dataset)}")
    print(f"Hashes computed: {len(hashes)}")
    print(f"Unique samples: {len(set(hashes))}")
