from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from glioma_hc.config import SETTINGS


def compute_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "sensitivity": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(
            (np.sum((y_true_arr == 0) & (y_pred == 0)) / max(np.sum(y_true_arr == 0), 1))
        ),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true_arr)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        metrics["auc"] = 0.0

    return metrics


def write_summary(metrics: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    model_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for model evaluation.") from exc
    from glioma_hc.inference import _load_keras_model
    from glioma_hc.training.data_loader import build_image_dataset
    from glioma_hc.training.model_zoo import MODEL_SPECS

    model_file = Path(model_path).resolve()
    test_dir = Path(data_dir).resolve() / "test"
    if not model_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Held-out test split not found: {test_dir}")

    dataset = build_image_dataset(
        test_dir,
        image_size=(MODEL_SPECS[model_name]["input_size"],) * 2,
        model_name=model_name,
        shuffle=False,
    )
    model = _load_keras_model(model_file)
    true_labels = []
    glioma_probabilities = []
    for images, labels in dataset:
        probabilities = np.asarray(model.predict(images, verbose=0), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(SETTINGS.label_order):
            raise ValueError(f"Model returned invalid prediction shape: {probabilities.shape}")
        true_labels.extend(labels.numpy().astype(int).tolist())
        glioma_probabilities.extend(probabilities[:, SETTINGS.binary_index].tolist())

    metrics = compute_metrics(true_labels, glioma_probabilities)
    manifest_path = Path(data_dir) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    report = {
        "model": model_file.name,
        "model_path": str(model_file),
        "dataset": manifest.get("dataset", "unspecified"),
        "test_split": str(test_dir),
        "test_samples": len(true_labels),
        "label_order": SETTINGS.label_order,
        "class_mapping": manifest.get("class_mapping"),
        "negative_class_note": manifest.get("negative_class_note"),
        "modality_note": manifest.get("modality_note"),
        "metrics": metrics,
    }
    write_summary(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for a smoke or real classification run.")
    parser.add_argument("--model", type=Path, help="Saved Keras checkpoint to evaluate on a held-out test split")
    parser.add_argument("--data-dir", type=Path, help="Processed dataset directory containing test/")
    parser.add_argument("--model-name", default="inception_v3", help="Backbone name used for preprocessing")
    parser.add_argument("--true", nargs="*", help="Ground-truth labels in [0, 1] format")
    parser.add_argument("--prob", nargs="*", help="Predicted positive probabilities")
    parser.add_argument("--output", type=str, default="reports/results.json", help="Where to save the report")
    args = parser.parse_args()

    if args.model is not None:
        if args.data_dir is None:
            parser.error("--data-dir is required when --model is provided")
        report = evaluate_model(args.model, args.data_dir, args.model_name, args.output)
    else:
        if args.true is None or args.prob is None:
            parser.error("provide --model and --data-dir for real evaluation, or both --true and --prob for explicit metric inputs")
        y_true = [int(value) for value in args.true]
        y_prob = [float(value) for value in args.prob]
        report = {"metrics": compute_metrics(y_true, y_prob)}
        write_summary(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
