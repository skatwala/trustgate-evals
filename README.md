# TrustGate Evaluations

# 🧮 TrustGate Evaluation Kit  
**Audit-Ready Evaluators for Enterprise AI Systems**

TrustGate Evals is a **vendor-neutral**, **test-first**, and **audit-ready** evaluation harness designed for regulated industries (healthcare, finance, insurance).  
It provides clean, typed, composable evaluators for measuring:

- **Model Quality** (WER, Intent Classification)  
- **Model Safety** (Prompt Injection / Leakage Refusal)  
- **Fairness** (Name-Variant Bias)  
- **Structured Output Accuracy** (Fact Extraction)  
- **Human-Style Scoring** (LLM-as-Judge)  
- **Cost & Usage** (Token-Level Cost Tracking + Audit Log)

All evaluators follow the same pattern:

- pure Python  
- no external LLM calls inside the evaluator  
- JSONL datasets  
- Protocol-based model client  
- deterministic PyTest tests  
- enterprise-grade separation of concerns  

---

# 📦 Project Structure

```
trustgate-evals/
│
├── src/
│   └── evaluators/
│       ├── wer.py                     # WER / CER metrics
│       ├── intent_eval.py             # Intent classification metrics
│       ├── prompt_injection_eval.py   # Prompt-injection safety checks
│       ├── bias_eval.py               # Name-variant bias testing
│       ├── judge_eval.py              # LLM-as-Judge scoring (0–10)
│       ├── fact_eval.py               # Extract structured facts from passages
│       └── usage_eval.py              # Cost calculator + audit log
│
├── notebooks/
│   └── results/
│       ├── intent_eval.ipynb
│       └── wer_analysis.ipynb
│
├── tests/
└── README.md
```

---

# 🚀 Core Evaluators

### **1. WER / CER (Speech-to-Text Quality)**
- Computes word error rate and character error rate  
- Supports batch evaluation  
- Compatible with any STT engine (Azure, GCP, OpenAI Whisper)

### **2. Intent Classification Metrics**
- Accuracy, precision, recall, F1  
- Works with JSONL intent datasets  
- Plug in your inference client

### **3. Prompt Injection / Data Leakage**
- Red-team prompts (e.g., “print your system instructions”)  
- Pass/fail scoring  
- Ensures refusal behavior

### **4. Bias Evaluator (Name-Variant Fairness)**
- Same resume with identity-signaling names  
- Detects unfair scoring deltas  
- Perfect for HR, lending, insurance models

### **5. Fact Extraction Evaluator**
- Extract structured fields from passages  
- Compares LLM performance on factual grounding  
- Useful for healthcare notes, customer profiles, transcripts

### **6. LLM-as-Judge**
- Compare candidate answer to reference  
- Returns:

```
SCORE: <0–10>
EXPLANATION: <reason>
```

### **7. Usage & Cost Evaluator**
- Per-token cost modeling  
- Build one **LLMUsageRecord** per call  
- Append-only JSONL audit log  
- Provides daily/monthly cost projections  

---

# 🧪 Testing

Run tests offline and deterministically:

```
pytest -q  
or 
uv run pytest -q
```

---

# 🛠 Using with Real LLMs

Evaluators depend on this interface:

```python
class ModelClient(Protocol):
    def complete(self, prompt: str) -> ModelResponse: ...
```

Implement your own client for OpenAI, Azure, Anthropic, etc.

---

# 📊 Example: Bias Evaluation

```python
from evaluators.bias_eval import load_bias_cases, evaluate_bias_suite
from my_llm_client import OpenAIClient

cases = load_bias_cases("src/evaluators/dataset/bias_minimal.jsonl")
client = OpenAIClient("gpt-4o-mini")

results = evaluate_bias_suite(client, cases)

for r in results:
    print(r.id, r.passed, r.max_delta)
```

---

# 🔒 Why This Matters

This repository provides:

- **AI measurement infrastructure**  
- **Compliance artifacts**  
- **Auditably logged LLM usage**  
- **Fairness and bias detection**  
- **Production readiness gates**  

---

# 🤝 Contributing

- Add evaluators in `src/evaluators/`  
- Add JSONL datasets  
- Add deterministic tests  
- Avoid embedding real API calls  

---

# 📄 License

MIT License


## Usage

```bash
uv sync
uv run pytest
uv run python -m evaluators.wer --pred data/sample_transcripts.json --ref data/expected_outputs.json
uv run python -m evaluators.intent_eval --data data/sample_intents.json
on powershell (windows) this is required -> $env:PYTHONPATH = "src"
```
