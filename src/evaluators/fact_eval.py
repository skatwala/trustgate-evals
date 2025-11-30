from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence

from .common import DummyModelClient
from .eval_writer import EvaluationRecord, append_evaluations


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    fact_cases = load_fact_cases(args.data)
    client: ModelClient = DummyModelClient()  # avoid shadowing "model"

    fact_results = evaluate_fact_suite(client, fact_cases)

    # ---- NEW: append to unified evaluations.json ----
    eval_records: list[EvaluationRecord] = []
    for case, res in zip(fact_cases, fact_results):
        num_fields = len(case.fields)

        eval_records.append(
            EvaluationRecord(
                eval_type="fact",
                name=case.id,
                dataset=str(args.data),
                metrics={
                    "num_fields": float(num_fields),
                },
                thresholds=None,          # no thresholds yet
                passed=True,              # structural check only for now
                num_examples=1,
                tags=["fact_eval"],
                notes=f"extracted_fields={list(res.extracted.keys())}",
            )
        )

    append_evaluations(eval_records)
