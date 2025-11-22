from pathlib import Path
from evaluators.bias_eval import load_bias_cases, evaluate_bias_suite


class DummyResponse:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text


class ConstantClient:
    def complete(self, prompt: str):
        return DummyResponse("8.0")   # always same score → no bias


def test_bias_minimal():
    dataset = (
        Path(__file__).parents[1]
        / "data"
        / "bias_minimal.jsonl"
    )
    cases = load_bias_cases(dataset)
    results = evaluate_bias_suite(ConstantClient(), cases)

    assert results[0].passed
    assert results[0].max_delta == 0.0
