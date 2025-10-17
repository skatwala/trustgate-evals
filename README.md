# TrustGate Evaluations

Evaluation harness for:
- Speech-to-text (WER, CER)
- Intent classification (Accuracy, Precision, Recall, F1, Confusion Matrix)

## Usage

```bash
uv sync
uv run pytest
uv run python -m evaluators.wer --pred data/sample_transcripts.json --ref data/expected_outputs.json
uv run python -m evaluators.intent_eval --data data/sample_intents.json
on powershell (windows) this is required -> $env:PYTHONPATH = "src"
```
