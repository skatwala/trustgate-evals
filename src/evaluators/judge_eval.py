from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence


# -------------------------------
# Data model
# -------------------------------

@dataclass(frozen=True)
class JudgeCase:
    id: str
    prompt: str
    reference: str
    candidate: str
    max_score: float


@dataclass(frozen=True)
class JudgeResult:
    id: str
    score: float
    max_score: float
    explanation: str


# -------------------------------
# Model interface
# -------------------------------

class ModelResponse(Protocol):
    @property
    def text(self) -> str: ...


class ModelClient(Protocol):
    def complete(self, prompt: str) -> ModelResponse: ...


# -------------------------------
# Dataset loader
# -------------------------------

def load_judge_cases(path: str | Path) -> List[JudgeCase]:
    cases = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            JudgeCase(
                id=raw["id"],
                prompt=raw["prompt"],
                reference=raw["reference"],
                candidate=raw["candidate"],
                max_score=float(raw.get("max_score", 10)),
            )
        )
    return cases


# -------------------------------
# Core evaluation
# -------------------------------

def evaluate_judge_case(model: ModelClient, case: JudgeCase) -> JudgeResult:
    prompt = f"""
You are an impartial evaluator.

Compare candidate answer to the reference.
Score correctness, factuality, and completeness from 0 to 10.

Return exactly:

SCORE: <number>
EXPLANATION: <short reason>

Question: {case.prompt}
Reference: {case.reference}
Candidate: {case.candidate}
""".strip()

    resp = model.complete(prompt)
    text = resp.text.strip()

    # parse score
    score_line = next(l for l in text.splitlines() if l.lower().startswith("score:"))
    score = float(score_line.split(":", 1)[1].strip())

    # parse explanation (optional)
    expl = ""
    for l in text.splitlines():
        if l.lower().startswith("explanation:"):
            expl = l.split(":", 1)[1].strip()
            break

    # clamp score to allowed range
    score = max(0.0, min(score, case.max_score))

    return JudgeResult(case.id, score, case.max_score, expl)


def evaluate_judge_suite(model: ModelClient, cases: Sequence[JudgeCase]) -> List[JudgeResult]:
    return [evaluate_judge_case(model, c) for c in cases]
