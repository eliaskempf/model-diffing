import json

import pytest

from model_diffing.parsing import parse_llm_json, resolve_model_labels, strip_json_fences


class TestStripJsonFences:
    def test_with_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_json_fences(text) == '{"key": "value"}'

    def test_without_fences(self):
        text = '{"key": "value"}'
        assert strip_json_fences(text) == '{"key": "value"}'

    def test_with_whitespace(self):
        text = "  \n```json\n[1, 2, 3]\n```\n  "
        assert strip_json_fences(text) == "[1, 2, 3]"

    def test_only_opening_fence(self):
        text = '```json\n{"key": "value"}'
        assert strip_json_fences(text) == '{"key": "value"}'

    def test_empty_string(self):
        assert strip_json_fences("") == ""

    def test_multiline_json(self):
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = strip_json_fences(text)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}


class TestParseLlmJson:
    def test_fenced_object(self):
        text = '```json\n{"answer": 42}\n```'
        assert parse_llm_json(text) == {"answer": 42}

    def test_fenced_array(self):
        text = "```json\n[1, 2, 3]\n```"
        assert parse_llm_json(text) == [1, 2, 3]

    def test_unfenced_json(self):
        text = '{"model": "Model A"}'
        assert parse_llm_json(text) == {"model": "Model A"}

    def test_invalid_json_raises(self):
        text = "```json\n{invalid json}\n```"
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json(text)

    def test_nested_structure(self):
        data = {"results": [{"model": "A", "score": 0.9}]}
        text = f"```json\n{json.dumps(data)}\n```"
        assert parse_llm_json(text) == data

    def test_plain_text_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("This is not JSON at all")


class TestResolveModelLabels:
    def test_model_a_and_b(self):
        results = [
            {"model": "Model A", "description": "foo"},
            {"model": "Model B", "description": "bar"},
        ]
        resolved = resolve_model_labels(results, "gpt-4o", "claude-3")
        assert resolved[0]["model_name"] == "gpt-4o"
        assert resolved[1]["model_name"] == "claude-3"

    def test_short_labels(self):
        results = [
            {"model": "A", "description": "foo"},
            {"model": "B", "description": "bar"},
        ]
        resolved = resolve_model_labels(results, "gpt-4o", "claude-3")
        assert resolved[0]["model_name"] == "gpt-4o"
        assert resolved[1]["model_name"] == "claude-3"

    def test_unclear_label(self):
        results = [{"model": "Unknown", "description": "foo"}]
        resolved = resolve_model_labels(results, "gpt-4o", "claude-3")
        assert resolved[0]["model_name"] == "unclear"

    def test_empty_list(self):
        assert resolve_model_labels([], "a", "b") == []

    def test_mutates_in_place(self):
        results = [{"model": "Model A"}]
        returned = resolve_model_labels(results, "x", "y")
        assert returned is results
        assert results[0]["model_name"] == "x"

    def test_mixed_labels(self):
        results = [
            {"model": "Model A"},
            {"model": "B"},
            {"model": "Model B"},
            {"model": "A"},
            {"model": "something else"},
        ]
        resolve_model_labels(results, "alpha", "beta")
        assert [r["model_name"] for r in results] == [
            "alpha",
            "beta",
            "beta",
            "alpha",
            "unclear",
        ]
