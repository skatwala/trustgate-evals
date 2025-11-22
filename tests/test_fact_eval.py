from pathlib import Path
from evaluators.fact_eval import load_fact_cases, evaluate_fact_suite


class DummyResponse:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text


class DummyFactClient:
    """Returns fixed answers for predictable test."""
    def complete(self, prompt: str):
        if "name" in prompt.lower():
            return DummyResponse("John Smith")
        if "city" in prompt.lower():
            return DummyResponse("Chicago")
        if "job" in prompt.lower() or "title" in prompt.lower():
            return DummyResponse("senior data engineer")
        return DummyResponse("")


def test_fact_minimal():
    dataset = (
        Path(__file__).parents[1]
        / "data"
        / "fact_minimal.jsonl"
    )
    cases = load_fact_cases(dataset)
    results = evaluate_fact_suite(DummyFactClient(), cases)

    r = results[0]
    assert r.id == "fact_001"
    assert r.extracted["name"] == "John Smith"
    assert r.extracted["city"] == "Chicago"
    assert r.extracted["role"] == "senior data engineer"
