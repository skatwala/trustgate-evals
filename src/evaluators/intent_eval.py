import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .eval_writer import EvaluationRecord, append_evaluations

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_intents(data_file: str, results_dir: str = "results") -> dict:
    with open(data_file) as f:
        data = json.load(f)

    y_true = data["labels"]
    y_pred = data["predictions"]

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true + y_pred)))

    os.makedirs(results_dir, exist_ok=True)

    # Save JSON
    with open(os.path.join(results_dir, "intent_eval.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save CSV (per-class metrics)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(results_dir, "intent_eval.csv"))

    return {"report": report, "confusion_matrix": cm, "labels": sorted(set(y_true + y_pred))}

def plot_confusion_matrix(cm, labels, results_dir: str = "results"):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.show()

def plot_metrics(report: dict, results_dir: str = "results"):
    # Drop avg rows, keep only classes
    metrics_df = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

    plt.figure(figsize=(8, 5))
    metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar")
    plt.title("Precision, Recall, F1 per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_metrics.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    results = evaluate_intents(args.data)

    print("Classification Report:")
    for label, metrics in results["report"].items():
        print(label, metrics)
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    plot_confusion_matrix(results["confusion_matrix"], results["labels"])
    plot_metrics(results["report"])

    # -------- NEW: write to evaluations.json --------
    report = results["report"]
    # Count examples: exclude summary rows
    num_examples = sum(
        report[label]["support"]
        for label in report
        if label not in ("accuracy", "macro avg", "weighted avg")
    )

    record = EvaluationRecord(
        eval_type="intent",
        name="intent_classification",
        dataset=args.data,
        metrics={
            "accuracy": float(report.get("accuracy", 0.0)),
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        },
        num_examples=num_examples,
        tags=["offline_eval"],
        notes="sklearn classification report",
    )

    append_evaluations([record])

