from pathlib import Path
from evaluators.prompt_injection_eval import (
    load_prompt_injection_cases,
    evaluate_prompt_injection,
    PromptInjectionCase,
    PromptInjectionResult,
)


# Dummy client for testing
class DummyResponse:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text


class DummyClient:
    def complete(self, prompt: str) -> DummyResponse:
        # Always refuses for test
        return DummyResponse("I can't help with that. This request is unsafe.")


def test_prompt_injection_eval():
    dataset_path = Path(__file__).parents[1] / "data" / "prompt_injection.jsonl"
    cases = load_prompt_injection_cases(dataset_path)

    client = DummyClient()
    results = evaluate_prompt_injection(client, cases)

    assert len(results) == len(cases)
    for r in results:
        assert r.predicted_label == "safe_refusal"
        assert r.is_correct is True
