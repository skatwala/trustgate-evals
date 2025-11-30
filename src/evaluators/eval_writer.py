from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EVALS_PATH = Path(__file__).resolve().parents[2] / "results" / "evaluations.json"


@dataclass(frozen=True)
class EvaluationRecord:
    eval_type: str
    name: str
    dataset: str
    metrics: Dict[str, float]
    thresholds: Optional[Dict[str, float]] = None
    passed: Optional[bool] = None
    num_examples: Optional[int] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "run_id": datetime.now(timezone.utc).isoformat(),
            "project": "TrustGate Evals",
            "evaluations": [],
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_evaluations(records: List[EvaluationRecord]) -> None:
    data = _load_json(EVALS_PATH)
    existing = data.get("evaluations", [])
    for rec in records:
        existing.append(asdict(rec))
    data["evaluations"] = existing
    _save_json(EVALS_PATH, data)
