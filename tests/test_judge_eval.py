from pathlib import Path
from evaluators.judge_eval import load_judge_cases, evaluate_judge_suite


class DummyResponse:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text


class DummyJudgeModel:
    """Always returns same score for testing."""
    def complete(self, prompt: str):
        return DummyResponse("SCORE: 7\nEXPLANATION: Looks reasonable.")


def test_judge_eval_minimal():
    dataset_path = (
        Path(__file__).parents[1]
        / "data"
        / "judge_minimal.jsonl"
    )
    cases = load_judge_cases(dataset_path)
    results = evaluate_judge_suite(DummyJudgeModel(), cases)

    assert len(results) == len(cases)
    assert results[0].score == 7
    assert results[0].max_score == 10
