from evaluators.intent_eval import evaluate_intents

def test_intent_eval():
    results = evaluate_intents("data/sample_intents.json")
    assert "report" in results
    assert "confusion_matrix" in results
    assert "labels" in results
