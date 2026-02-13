import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock heavy optional deps so test collection succeeds without them.
# Modules that need real torch/openai should skip via pytest.importorskip().
for _mod in [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "peft",
    "openai",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CASSETTES_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture
def vcr_config():
    return {
        "cassette_library_dir": str(CASSETTES_DIR),
        "filter_headers": ["Authorization"],
        "record_mode": "none",
    }


@pytest.fixture
def mock_model():
    """Create a mock ModelWrapper that doesn't need torch/transformers.
    Skips the check_if_open_router HTTP call and returns canned responses."""
    model = MagicMock()
    model.open_router = True
    model.model_name = "test/mock-model"
    model.api_key = "test-key-123"
    model.generate_async = AsyncMock(return_value=["This is a mock response."])
    return model


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a tmp_path-based workspace for file I/O tests."""
    return tmp_path


@pytest.fixture
def sample_responses_path():
    return FIXTURES_DIR / "sample_responses.jsonl"


@pytest.fixture
def sample_diffs_path():
    return FIXTURES_DIR / "sample_diffs.jsonl"
