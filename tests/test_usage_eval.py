from pathlib import Path
from evaluators.usage_eval import (
    ModelPricing,
    build_usage_record,
    append_audit_log,
    read_audit_log,
)


class DummyResp:
    def __init__(self, model: str, input_tokens: int, output_tokens: int, text: str = "ok"):
        self._model = model
        self._in = input_tokens
        self._out = output_tokens
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @property
    def input_tokens(self) -> int:
        return self._in

    @property
    def output_tokens(self) -> int:
        return self._out

    @property
    def model(self) -> str:
        return self._model


def test_usage_record_and_audit_log(tmp_path: Path):
    pricing = ModelPricing(input_token_cost=1e-6, output_token_cost=2e-6)
    resp = DummyResp("gpt-test", 1000, 500)

    record = build_usage_record(
        trace_id="trace-123",
        resp=resp,
        pricing=pricing,
        latency_ms=120,
        meta={"route": "/transcribe", "country": "IN"},
    )

    log_file = tmp_path / "audit.jsonl"
    append_audit_log(record, log_file)

    records = read_audit_log(log_file)
    assert len(records) == 1
    r = records[0]

    assert r.trace_id == "trace-123"
    assert r.model == "gpt-test"
    assert r.input_tokens == 1000
    assert r.output_tokens == 500
    assert r.latency_ms == 120
    assert abs(r.cost_usd - (1000 * 1e-6 + 500 * 2e-6)) < 1e-9
    assert r.meta["route"] == "/transcribe"
