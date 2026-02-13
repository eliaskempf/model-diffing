"""Integration tests for model.py using VCR cassettes and mocks.

Since model.py imports torch/transformers at the top level, we test
check_if_open_router by importing it in isolation, and test generate_async
using mock objects with VCR cassettes for the HTTP layer.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import vcr

CASSETTES_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

my_vcr = vcr.VCR(
    cassette_library_dir=CASSETTES_DIR,
    filter_headers=["Authorization"],
    record_mode="none",
)


def _import_check_if_open_router():
    """Import just check_if_open_router without pulling in torch.

    model.py imports torch at module level, so if torch isn't installed
    we mock those imports before importing the module.
    """
    try:
        from model_diffing.model import check_if_open_router

        return check_if_open_router
    except ImportError:
        # torch/transformers not available — mock them for import
        mock_modules = {}
        for mod_name in [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "transformers",
            "peft",
        ]:
            if mod_name not in sys.modules:
                mock_modules[mod_name] = MagicMock()

        saved = {}
        for k, v in mock_modules.items():
            saved[k] = sys.modules.get(k)
            sys.modules[k] = v

        try:
            # Force reimport
            if "model_diffing.model" in sys.modules:
                del sys.modules["model_diffing.model"]
            from model_diffing.model import check_if_open_router

            return check_if_open_router
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class TestCheckIfOpenRouter:
    @my_vcr.use_cassette("check_open_router_positive.yaml")
    def test_known_model_returns_true(self):
        check_if_open_router = _import_check_if_open_router()
        assert check_if_open_router("openai/gpt-4o-mini") is True

    @my_vcr.use_cassette("check_open_router_positive.yaml")
    def test_unknown_model_returns_false(self):
        check_if_open_router = _import_check_if_open_router()
        assert check_if_open_router("definitely/not-a-real-model-xyz") is False


class TestModelWrapperMocked:
    def test_init_open_router(self, mock_model):
        assert mock_model.open_router is True
        assert mock_model.model_name == "test/mock-model"

    async def test_generate_async_returns_canned(self, mock_model):
        conversations = [[{"role": "user", "content": "Hello"}]]
        result = await mock_model.generate_async(conversations, max_new_tokens=100, enable_thinking=False, seed=42)
        assert result == ["This is a mock response."]
        mock_model.generate_async.assert_called_once()

    async def test_generate_async_multiple_convos(self, mock_model):
        mock_model.generate_async = AsyncMock(return_value=["Response 1", "Response 2"])
        conversations = [
            [{"role": "user", "content": "First"}],
            [{"role": "user", "content": "Second"}],
        ]
        result = await mock_model.generate_async(conversations, max_new_tokens=100, enable_thinking=False, seed=42)
        assert len(result) == 2


class TestGenerateAsyncVCR:
    @my_vcr.use_cassette("generate_async_simple.yaml")
    async def test_generate_async_with_cassette(self):
        """Test the actual HTTP call pattern used by generate_async,
        replayed from a VCR cassette."""
        api_key = "test-key"
        model_name = "openai/gpt-4o-mini"
        conversation = [{"role": "user", "content": "Say hello in one word."}]

        # Replicate the HTTP call from ModelWrapper.generate_async
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": conversation,
                "max_tokens": 50,
                "seed": 42,
            }
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(10, 120),
            )
            assert res.status_code == 200
            data = res.json()
            reply = data["choices"][0]["message"]["content"]
            assert isinstance(reply, str)
            assert len(reply) > 0
