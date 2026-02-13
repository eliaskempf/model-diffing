from model_diffing.hypothesis_config import (
    EXPERIMENTS,
    get_experiment_by_key,
    get_experiments_by_method,
    get_experiments_by_name,
)


class TestGetExperimentByKey:
    def test_existing_key(self):
        exp = get_experiment_by_key("gemini_llm")
        assert exp is not None
        assert exp["name"] == "gemini"
        assert exp["method"] == "llm"

    def test_nonexistent_key(self):
        assert get_experiment_by_key("nonexistent") is None

    def test_all_keys_unique(self):
        keys = [exp["key"] for exp in EXPERIMENTS]
        assert len(keys) == len(set(keys))

    def test_sae_key(self):
        exp = get_experiment_by_key("qwen_em_sae")
        assert exp is not None
        assert exp["method"] == "sae"


class TestGetExperimentsByName:
    def test_gemini(self):
        exps = get_experiments_by_name("gemini")
        assert len(exps) == 2
        methods = {e["method"] for e in exps}
        assert methods == {"llm", "sae"}

    def test_qwen(self):
        exps = get_experiments_by_name("qwen_em")
        assert len(exps) == 2

    def test_nonexistent(self):
        assert get_experiments_by_name("nonexistent") == []


class TestGetExperimentsByMethod:
    def test_llm(self):
        exps = get_experiments_by_method("llm")
        assert len(exps) == 3
        assert all(e["method"] == "llm" for e in exps)

    def test_sae(self):
        exps = get_experiments_by_method("sae")
        assert len(exps) == 3
        assert all(e["method"] == "sae" for e in exps)
