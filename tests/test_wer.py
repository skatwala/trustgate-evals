from evaluators.wer import evaluate_stt

def test_stt_eval():
    results = evaluate_stt("data/sample_transcripts.json", "data/expected_outputs.json")
    assert "avg_wer" in results
    assert "avg_cer" in results
    assert isinstance(results["wer_scores"], list)
    assert isinstance(results["cer_scores"], list)
