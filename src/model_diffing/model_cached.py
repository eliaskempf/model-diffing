"""
Cached version of ModelWrapper that uses safety-tooling's InferenceAPI for caching.

This module provides CachedModelWrapper, a drop-in replacement for ModelWrapper
that caches API calls to OpenRouter, preventing redundant API requests and saving costs.

Only supports OpenRouter models (not local inference).

If safetytooling is not installed, CachedModelWrapper falls back to ModelWrapper's
httpx-based OpenRouter implementation with semaphore concurrency control but no caching.
"""

from __future__ import annotations

import asyncio
import os
import warnings
from pathlib import Path

from model_diffing.model import ModelWrapper

try:
    from openai import BadRequestError
    from safetytooling.apis.inference.api import InferenceAPI
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    HAS_SAFETYTOOLING = True
except ImportError:
    HAS_SAFETYTOOLING = False


class CachedModelWrapper(ModelWrapper):
    """
    A cached version of ModelWrapper that uses safety-tooling's InferenceAPI for caching.

    This class is designed as a drop-in replacement for ModelWrapper when using OpenRouter
    models. It uses force_openrouter=True to skip the OpenRouter API check and sets up
    semaphore-based concurrency control via the parent class.

    When safetytooling is installed, API responses are automatically cached to disk via
    InferenceAPI. When safetytooling is not installed, falls back to the parent's httpx-based
    implementation with semaphore concurrency control (no caching).

    Only supports OpenRouter models - local inference is not supported.

    Args:
        model_name: The model identifier (must be available on OpenRouter)
        api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
        cache_dir: Directory for caching API responses. Defaults to ST_CACHE_DIR env var,
                   then falls back to safety-tooling's default (.cache in repo root).
                   Pass None to disable caching.
        no_cache: If True, disable caching entirely (default: False)
        openrouter_num_threads: Number of concurrent OpenRouter requests (default: 80)
        **inference_api_kwargs: Additional kwargs passed to InferenceAPI constructor

    Example:
        >>> model = CachedModelWrapper("meta-llama/llama-3.1-70b-instruct")
        >>> responses = await model.generate_async(
        ...     conversations=[[{"role": "user", "content": "Hello!"}]], max_new_tokens=100
        ... )
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        cache_dir: Path | str | None = "default",
        no_cache: bool = False,
        openrouter_num_threads: int = 80,
        **inference_api_kwargs,
    ):
        # Initialize parent with force_openrouter (skips check_if_open_router call)
        # and semaphore concurrency control
        super().__init__(
            model_name,
            force_openrouter=True,
            openrouter_num_threads=openrouter_num_threads,
            api_key=api_key,
        )

        if HAS_SAFETYTOOLING:
            # Resolve cache_dir: constructor param > ST_CACHE_DIR env var > safety-tooling default
            if cache_dir == "default":
                env_cache_dir = os.environ.get("ST_CACHE_DIR")
                if env_cache_dir:
                    cache_dir = Path(env_cache_dir)
                else:
                    # Let InferenceAPI use its default (which checks CACHE_DIR env var)
                    cache_dir = "default"
            elif cache_dir is not None:
                cache_dir = Path(cache_dir)

            # Ensure the API key is available in the environment for InferenceAPI
            # (InferenceAPI reads from env var when initializing OpenRouterChatModel)
            if api_key is not None:
                os.environ["OPENROUTER_API_KEY"] = api_key

            # Initialize the InferenceAPI with caching
            # Note: max_mem_usage_mb=None disables in-memory cache eviction to avoid
            # race conditions when multiple async tasks access the cache concurrently.
            self._inference_api = InferenceAPI(
                cache_dir=cache_dir,
                no_cache=no_cache,
                openrouter_num_threads=openrouter_num_threads,
                max_mem_usage_mb=None,
                **inference_api_kwargs,
            )
        else:
            self._inference_api = None
            warnings.warn(
                "safetytooling not installed; CachedModelWrapper will work without caching. "
                "Install with: uv sync --extra safety",
                stacklevel=2,
            )

    @property
    def running_cost(self) -> float:
        """Returns the cumulative cost of API calls made through this wrapper."""
        if self._inference_api is not None:
            return self._inference_api.running_cost
        return 0.0

    def reset_cost(self):
        """Resets the running cost counter to zero."""
        if self._inference_api is not None:
            self._inference_api.reset_cost()

    def _conversation_to_prompt(self, conversation: list[dict]) -> Prompt:
        """Convert a conversation (list of message dicts) to a safety-tooling Prompt."""
        messages = []
        for msg in conversation:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")

            # Map role strings to MessageRole enum
            if role_str == "system":
                role = MessageRole.system
            elif role_str == "assistant":
                role = MessageRole.assistant
            elif role_str == "user":
                role = MessageRole.user
            else:
                # Default to user for unknown roles
                role = MessageRole.user

            messages.append(ChatMessage(role=role, content=content))

        return Prompt(messages=messages)

    async def generate_async(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int,
        enable_thinking: bool = False,
        seed: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Generate responses for multiple conversations using OpenRouter.

        When safetytooling is installed, uses InferenceAPI for automatic caching.
        Otherwise, delegates to the parent's httpx-based implementation with semaphore
        concurrency control.

        Args:
            conversations: List of conversations, where each conversation is a list of
                          message dicts with "role" and "content" keys.
            max_new_tokens: Maximum number of tokens to generate.
            enable_thinking: Whether to enable thinking mode (for supported models).
            seed: Random seed for reproducibility.
            show_progress: Whether to show a tqdm progress bar (default: False).
            **kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            List of response strings, one per conversation.
        """
        if self._inference_api is None:
            # No safetytooling — delegate to parent (which has semaphore + httpx)
            return await super().generate_async(
                conversations, max_new_tokens, enable_thinking, seed, show_progress, **kwargs
            )

        # Filter kwargs to only include supported parameters
        allowed_params = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}

        # Build base parameters for the API call
        api_kwargs = {
            "max_tokens": max_new_tokens,
            **filtered_kwargs,
        }

        if seed is not None:
            api_kwargs["seed"] = seed

        # Make concurrent API calls for all conversations
        async def generate_one(conversation: list[dict]) -> str:
            prompt = self._conversation_to_prompt(conversation)
            async with self._semaphore:
                try:
                    responses = await self._inference_api(
                        model_id=self.model_name,
                        prompt=prompt,
                        n=1,
                        force_provider="openrouter",
                        **api_kwargs,
                    )
                except (BadRequestError, RuntimeError) as e:
                    print(f"OpenRouter API request failed for model {self.model_name} with prompt {prompt}: {e!s}")
                    return ""
            if responses and len(responses) > 0:
                return responses[0].completion
            return ""

        # Run all conversations concurrently
        tasks = [generate_one(conv) for conv in conversations]

        if show_progress:
            from tqdm.asyncio import tqdm_asyncio

            results = await tqdm_asyncio.gather(*tasks, desc="Generating")
        else:
            results = await asyncio.gather(*tasks)

        return list(results)

    def generate(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Synchronous generation is not supported for CachedModelWrapper.

        CachedModelWrapper only supports OpenRouter (API-based) inference, which
        is inherently async. Use generate_async instead.

        Raises:
            NotImplementedError: Always, as sync generation is not supported.
        """
        raise NotImplementedError(
            "CachedModelWrapper only supports async generation via generate_async(). "
            "Use 'await model.generate_async(...)' or run with asyncio.run()."
        )
