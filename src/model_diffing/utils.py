import hashlib
import json
import warnings
from itertools import islice

import pandas as pd


def batch_iterable(iterable, batch_size):
    """Yield successive batches of given size from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


class ResponseDict(dict):
    def __setitem__(self, conversation: list[dict], response: str):
        assert isinstance(response, str), "Response must be a string"
        value = {
            "conversation": conversation,
            "response": response,
        }
        super().__setitem__(self._hash_key(conversation), value)

    def __getitem__(self, conversation: list[dict] | str):
        if isinstance(conversation, str):
            return super().__getitem__(conversation)
        return super().__getitem__(self._hash_key(conversation))

    def __delitem__(self, conversation: list[dict] | str):
        if isinstance(conversation, str):
            super().__delitem__(conversation)
            return
        super().__delitem__(self._hash_key(conversation))

    def __contains__(self, conversation: list[dict] | str):
        if isinstance(conversation, str):
            return super().__contains__(conversation)
        return super().__contains__(self._hash_key(conversation))

    def insert_and_save(self, conversation: list[dict], response: str, file_handle):
        if conversation in self:
            if self[conversation] != {"conversation": conversation, "response": response}:
                warnings.warn(
                    "Conversation already exists with a different response and won't be overwritten.", stacklevel=2
                )
            return

        self[conversation] = response
        item = {self._hash_key(conversation): self[conversation]}
        file_handle.write(json.dumps(item) + "\n")

    def save_jsonl(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            for hash, item in self.items():
                entry = {hash: item}
                f.write(json.dumps(entry) + "\n")

    def to_df(self) -> pd.DataFrame:
        records = []
        for hash, item in self.items():
            convo_with_resp = item["conversation"] + [{"role": "assistant", "content": item["response"]}]
            record = {"hash": hash, **item, "conversation_with_response": convo_with_resp}
            records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _hash_key(conversation: list[dict]) -> str:
        json_str = json.dumps(conversation, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @classmethod
    def from_jsonl(cls, file_path: str):
        instance = cls()
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                assert len(entry) == 1, "Each line must contain exactly one entry"
                hash = next(iter(entry))
                item = entry[hash]
                conversation = item["conversation"]
                response = item["response"]
                assert cls._hash_key(conversation) == hash, "Hash mismatch in loaded data"
                instance[conversation] = response
        return instance
