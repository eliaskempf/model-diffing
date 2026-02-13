# Default judges - 3 diverse, capable models for robust evaluation
DEFAULT_JUDGES = [
    "anthropic/claude-opus-4.5",
    "openai/gpt-5.2",
    "x-ai/grok-4",
]


def __getattr__(name):
    if name == "rate_interestingness":
        from model_diffing.autoraters.autorater_interestingness import rate_hypotheses

        return rate_hypotheses
    if name == "rate_abstraction":
        from model_diffing.autoraters.autorater_abstraction import rate_hypotheses

        return rate_hypotheses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
