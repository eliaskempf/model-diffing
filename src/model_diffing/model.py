import asyncio
import os

import httpx
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def check_if_open_router(model_name: str) -> bool:
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": "Bearer <token>"}
    response = requests.get(url, headers=headers)
    all_ids = [d["id"] for d in response.json()["data"]]
    return model_name in all_ids


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        base_model: str | None = None,
        force_hf: bool = False,
        force_openrouter: bool = False,
        openrouter_num_threads: int | None = None,
        api_key: str | None = None,
    ):
        if force_openrouter:
            self.open_router = True
        elif force_hf:
            self.open_router = False
        else:
            self.open_router = check_if_open_router(model_name) and not force_hf

        if not self.open_router:
            torch.set_float32_matmul_precision("high")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not getattr(self.tokenizer, "chat_template", None):
                # Define and set fallback template
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ message['content'] }}\n"
                    "{% elif message['role'] == 'user' %}"
                    "User: {{ message['content'] }}\n"
                    "{% elif message['role'] == 'assistant' %}"
                    "Assistant: {{ message['content'] }}\n"
                    "{% endif %}"
                    "{% endfor %}"
                    "Assistant:"
                )
            if model_revision is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, revision=model_revision, torch_dtype=torch.bfloat16, device_map="auto"
                )
            elif base_model is not None:
                # load model with PEFT weights
                from peft import PeftModel

                torch._dynamo.reset()
                torch._dynamo.config.recompile_limit = 32
                torch._dynamo.config.fail_on_recompile_limit_hit = True
                self.model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto"),
                    model_name,
                )
                self.model = self.model.merge_and_unload()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="auto"
                )
        else:
            self.model_name = model_name
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter mode but no API key provided")

        # Optional semaphore for concurrency control in generate_async
        self._semaphore = asyncio.Semaphore(openrouter_num_threads) if openrouter_num_threads else None

    @torch.no_grad()
    def generate(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> list[str]:
        if seed is not None:
            # safer version since hf just seeds everything it might use
            set_seed(seed)

        texts = []
        for conversation in conversations:
            texts.append(
                self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,  # Switches between thinking and non-thinking modes. Default is False
                )
            )
        max_ctx = int(getattr(self.model.config, "max_position_embeddings", 4096))
        max_prompt = max_ctx - max_new_tokens

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=max_prompt,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        responses = []
        for b in range(generated_ids.shape[0]):
            output_ids = generated_ids[b][len(model_inputs.input_ids[b]) :].tolist()
            responses.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))

        return responses

    async def _openrouter_request(
        self,
        conversation: list[dict],
        client: httpx.AsyncClient,
        max_new_tokens: int,
        enable_thinking: bool,
        seed: int | None,
        **kwargs,
    ) -> str:
        """Make a single OpenRouter API request."""
        allowed_extra = ("temperature", "top_p", "presence_penalty", "frequency_penalty")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": conversation,
            "max_tokens": max_new_tokens,
        }
        if enable_thinking:
            payload["thinking"] = {"enabled": True}
        if seed is not None:
            payload["seed"] = seed

        for k in allowed_extra:
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, 120),
        )

        if res.status_code == 403 and "Content violates usage guidelines" in res.text:
            return ""
        res.raise_for_status()
        data = res.json()
        try:
            reply = data["choices"][0]["message"]["content"]
        except KeyError:
            assert "error" in data, f"Unexpected response format: {data}"
            print(f"API Error: {data['error']}")
            reply = ""
        return reply

    async def generate_async(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int,
        enable_thinking: bool = False,
        seed: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> list[str]:
        async def _generate_one(conversation: list[dict], client: httpx.AsyncClient) -> str:
            if self._semaphore:
                async with self._semaphore:
                    return await self._openrouter_request(
                        conversation, client, max_new_tokens, enable_thinking, seed, **kwargs
                    )
            return await self._openrouter_request(conversation, client, max_new_tokens, enable_thinking, seed, **kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [_generate_one(conv, client) for conv in conversations]
            if show_progress:
                from tqdm.asyncio import tqdm_asyncio

                responses = await tqdm_asyncio.gather(*tasks, desc="Generating")
            else:
                responses = await asyncio.gather(*tasks)
        return list(responses)
