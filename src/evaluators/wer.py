"""
Simple Word Error Rate (WER) and Character Error Rate (CER) evaluator.
Usage:
    python -m evaluators.wer
    python -m evaluators.wer --pred data/pred.txt --ref data/ref.txt
"""

import argparse, json, os
from pathlib import Path

from .eval_writer import EvaluationRecord, append_evaluations
from jiwer import wer

# Optional: some JiWER versions removed compute_measures
try:
    from jiwer import compute_measures
    _HAS_COMPUTE_MEASURES = True
except ImportError:
    _HAS_COMPUTE_MEASURES = False


def load_file(path: str):
    """Load JSON or text file and return a list of strings."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
        if not text:
            return []
        if text.startswith("{"):
            try:
                data = json.loads(text)
                for key in ("expected", "predicted", "refs", "preds", "transcripts"):
                    if key in data:
                        return data[key]
            except json.JSONDecodeError:
                pass
        return [line.strip() for line in text.splitlines() if line.strip()]


def char_error_rate(refs, preds):
    """Compute average CER and per-sample CERs."""
    import Levenshtein
    total_chars, total_distance = 0, 0
    cer_scores = []
    for r, p in zip(refs, preds):
        if not r:
            cer_scores.append(0.0)
            continue
        dist = Levenshtein.distance(r, p)
        cer = dist / len(r)
        cer_scores.append(cer)
        total_chars += len(r)
        total_distance += dist
    avg_cer = total_distance / total_chars if total_chars > 0 else 0.0
    return avg_cer, cer_scores


def evaluate_stt(pred_file: str, ref_file: str):
    refs = load_file(ref_file)
    preds = load_file(pred_file)

    if len(refs) != len(preds):
        print(f"⚠️ line count mismatch: refs={len(refs)} preds={len(preds)}")

    # Per-sample WERs
    wer_scores = [wer([r], [p]) for r, p in zip(refs, preds)]
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0

    # Per-sample CERs
    avg_cer, cer_scores = char_error_rate(refs, preds)

    results = {
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "wer_scores": wer_scores,
        "cer_scores": cer_scores,
    }

    # Optionally include detailed measures
    if _HAS_COMPUTE_MEASURES:
        measures = compute_measures(refs, preds)
        results.update({
            "substitutions": measures.get("substitutions", 0),
            "deletions": measures.get("deletions", 0),
            "insertions": measures.get("insertions", 0),
        })

    print(f"\n✅ WER: {avg_wer:.3f}")
    print(f"✅ CER: {avg_cer:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="data/pred.txt")
    parser.add_argument("--ref", default="data/ref.txt")
    parser.add_argument("--no-write-json", action="store_true")
    args = parser.parse_args()

    if not Path(args.pred).exists():
        print(f"Error: Prediction file '{args.pred}' not found.")
        exit(1)
    if not Path(args.ref).exists():
        print(f"Error: Reference file '{args.ref}' not found.")
        exit(1)

    results = evaluate_stt(args.pred, args.ref)

    # -------- NEW: write to evaluations.json --------
    if not args.no_write_json:
        record = EvaluationRecord(
            eval_type="wer",
            name="stt_wer",
            dataset=f"ref={args.ref},pred={args.pred}",
            metrics={
                "avg_wer": float(results["avg_wer"]),
                "avg_cer": float(results["avg_cer"]),
            },
            num_examples=len(results["wer_scores"]),
            tags=["stt"],
            notes="JiWER-based WER/CER",
        )

        append_evaluations([record])

