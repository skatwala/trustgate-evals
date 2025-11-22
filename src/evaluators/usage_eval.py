from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


# -----------------------------
# Pricing
# -----------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing in dollars."""
    input_token_cost: float   # $ per 1 token
    output_token_cost: float  # $ per 1 token


# -----------------------------
# LLM response protocol
# Your real client response should expose these.
# -----------------------------

class ModelResponse(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def model(self) -> str: ...


# -----------------------------
# Usage record (auditable unit)
# -----------------------------

@dataclass(frozen=True)
class LLMUsageRecord:
    trace_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    timestamp_ms: int
    meta: Dict[str, Any]


def calculate_cost(input_tokens: int, output_tokens: int, pricing: ModelPricing) -> float:
    return (
        input_tokens * pricing.input_token_cost
        + output_tokens * pricing.output_token_cost
    )


def build_usage_record(
    *,
    trace_id: str,
    resp: ModelResponse,
    pricing: ModelPricing,
    latency_ms: int,
    meta: Optional[Dict[str, Any]] = None,
) -> LLMUsageRecord:
    meta = meta or {}
    cost = calculate_cost(resp.input_tokens, resp.output_tokens, pricing)

    return LLMUsageRecord(
        trace_id=trace_id,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost,
        timestamp_ms=int(time.time() * 1000),
        meta=meta,
    )


# -----------------------------
# Append-only audit log (JSONL)
# -----------------------------

def append_audit_log(record: LLMUsageRecord, log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_audit_log(log_path: str | Path) -> List[LLMUsageRecord]:
    path = Path(log_path)
    if not path.exists():
        return []

    records: List[LLMUsageRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(
            LLMUsageRecord(
                trace_id=raw["trace_id"],
                model=raw["model"],
                input_tokens=int(raw["input_tokens"]),
                output_tokens=int(raw["output_tokens"]),
                latency_ms=int(raw["latency_ms"]),
                cost_usd=float(raw["cost_usd"]),
                timestamp_ms=int(raw["timestamp_ms"]),
                meta=dict(raw.get("meta", {})),
            )
        )
    return records
