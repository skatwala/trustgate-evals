from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence


# -------------------------------
# Data models
# -------------------------------

@dataclass(frozen=True)
class FactCase:
    id: str
    passage: str
    fields: Dict[str, str]    # field_name -> question


@dataclass(frozen=True)
class FactResult:
    id: str
    extracted: Dict[str, str]  # field_name -> model answer


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

def load_fact_cases(path: str | Path) -> List[FactCase]:
    cases: List[FactCase] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            FactCase(
                id=raw["id"],
                passage=raw["passage"],
                fields=raw["fields"],
            )
        )
    return cases


# -------------------------------
# Core evaluation logic
# -------------------------------

def build_prompt(passage: str, question: str) -> str:
    return f"""
Extract a fact from the passage below.

Return ONLY the extracted text with no explanation.

Passage:
{passage}

Question: {question}
""".strip()


def evaluate_fact_case(model: ModelClient, case: FactCase) -> FactResult:
    out: Dict[str, str] = {}

    for field, question in case.fields.items():
        prompt = build_prompt(case.passage, question)
        resp = model.complete(prompt)
        out[field] = resp.text.strip()

    return FactResult(id=case.id, extracted=out)


def evaluate_fact_suite(model: ModelClient, cases: Sequence[FactCase]) -> List[FactResult]:
    return [evaluate_fact_case(model, c) for c in cases]
