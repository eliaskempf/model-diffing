import json

import pandas as pd

from model_diffing.utils import ResponseDict, batch_iterable


class TestBatchIterable:
    def test_even_batches(self):
        result = list(batch_iterable(range(6), 3))
        assert result == [[0, 1, 2], [3, 4, 5]]

    def test_remainder_batch(self):
        result = list(batch_iterable(range(5), 3))
        assert result == [[0, 1, 2], [3, 4]]

    def test_single_batch(self):
        result = list(batch_iterable(range(2), 10))
        assert result == [[0, 1]]

    def test_empty(self):
        result = list(batch_iterable([], 3))
        assert result == []


class TestResponseDict:
    def test_insert_and_get(self):
        rd = ResponseDict()
        convo = [{"role": "user", "content": "Hello"}]
        rd[convo] = "Hi there"
        assert rd[convo]["response"] == "Hi there"
        assert rd[convo]["conversation"] == convo

    def test_hash_key_deterministic(self):
        convo = [{"role": "user", "content": "test"}]
        h1 = ResponseDict._hash_key(convo)
        h2 = ResponseDict._hash_key(convo)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # sha256 hex

    def test_hash_key_different_for_different_convos(self):
        c1 = [{"role": "user", "content": "hello"}]
        c2 = [{"role": "user", "content": "world"}]
        assert ResponseDict._hash_key(c1) != ResponseDict._hash_key(c2)

    def test_contains(self):
        rd = ResponseDict()
        convo = [{"role": "user", "content": "test"}]
        assert convo not in rd
        rd[convo] = "response"
        assert convo in rd

    def test_contains_by_hash(self):
        rd = ResponseDict()
        convo = [{"role": "user", "content": "test"}]
        rd[convo] = "response"
        h = ResponseDict._hash_key(convo)
        assert h in rd

    def test_delete(self):
        rd = ResponseDict()
        convo = [{"role": "user", "content": "delete me"}]
        rd[convo] = "gone"
        assert convo in rd
        del rd[convo]
        assert convo not in rd

    def test_duplicate_insert_warns(self):
        import warnings

        rd = ResponseDict()
        convo = [{"role": "user", "content": "dup"}]
        f = open("/dev/null", "w")
        rd.insert_and_save(convo, "first", f)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rd.insert_and_save(convo, "second", f)
            assert len(w) == 1
            assert "already exists" in str(w[0].message)
        f.close()

    def test_jsonl_roundtrip(self, tmp_workspace):
        rd = ResponseDict()
        convos = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "Tell me a joke"}],
            [{"role": "user", "content": "What is Python?"}],
        ]
        for convo in convos:
            rd[convo] = f"Response to: {convo[0]['content']}"

        path = str(tmp_workspace / "test_roundtrip.jsonl")
        rd.save_jsonl(path)

        rd2 = ResponseDict.from_jsonl(path)
        assert len(rd2) == 3
        for convo in convos:
            assert rd[convo] == rd2[convo]

    def test_from_jsonl_fixture(self, sample_responses_path):
        rd = ResponseDict.from_jsonl(str(sample_responses_path))
        assert len(rd) == 3
        convo = [{"role": "user", "content": "What is 2+2?"}]
        assert rd[convo]["response"] == "2+2 equals 4."

    def test_to_df(self):
        rd = ResponseDict()
        convo = [{"role": "user", "content": "test"}]
        rd[convo] = "response"
        df = rd.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "hash" in df.columns
        assert "response" in df.columns
        assert "conversation_with_response" in df.columns
        # conversation_with_response should include the assistant turn
        cwr = df.iloc[0]["conversation_with_response"]
        assert cwr[-1]["role"] == "assistant"
        assert cwr[-1]["content"] == "response"

    def test_insert_and_save_writes_to_file(self, tmp_workspace):
        rd = ResponseDict()
        path = tmp_workspace / "incremental.jsonl"
        convo = [{"role": "user", "content": "save me"}]
        with open(path, "a") as f:
            rd.insert_and_save(convo, "saved response", f)

        with open(path) as f:
            line = f.readline()
            entry = json.loads(line)
            h = ResponseDict._hash_key(convo)
            assert h in entry
            assert entry[h]["response"] == "saved response"
