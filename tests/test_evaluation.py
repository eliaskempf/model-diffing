"""Tests for evaluation module parsing functions."""

from model_diffing.evaluation.judge_batched import try_parsing_response as judge_batched_try_parsing


class TestJudgeBatchedTryParsingResponse:
    def test_valid_json(self):
        result = judge_batched_try_parsing('{"H1": 1, "H2": 2, "H3": "N/A"}')
        assert result == {"H1": 1, "H2": 2, "H3": "N/A"}

    def test_fenced_json(self):
        result = judge_batched_try_parsing('```json\n{"H1": 1}\n```')
        assert result == {"H1": 1}

    def test_invalid_json_returns_error(self):
        result = judge_batched_try_parsing("not json at all")
        assert "error" in result
