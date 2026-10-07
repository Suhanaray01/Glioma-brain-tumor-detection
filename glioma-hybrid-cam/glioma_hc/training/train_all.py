from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.make_smoke_data import make_smoke_dataset
from glioma_hc.training.data_loader import build_image_dataset
from glioma_hc.training.model_zoo import MODEL_SPECS, build_model, list_supported_models

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - runtime dependency is optional
    tf = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GliomaScan-HC models")
    parser.add_argument("--models", nargs="*", default=["inception_v3"], help="Model names to train")
    parser.add_argument("--preset", choices=["paper", "fast"], default="paper")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-dir", type=Path, help="Processed dataset directory containing train/ and val/ class folders")
    parser.add_argument("--model-dir", type=Path, help="Directory for trained model checkpoints")
    parser.add_argument("--fine-tune", action="store_true", help="Train the pretrained backbone as well as the classification head")
    parser.add_argument("--epochs", type=int, help="Override the epoch count (default: 10 for attention_cnn, 2 otherwise)")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--no-balance", action="store_false", dest="balance")
    parser.set_defaults(balance=True)
    return parser.parse_args()


def _ensure_directory(project_root: Path) -> Path:
    project_root = Path(project_root)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "reports").mkdir(parents=True, exist_ok=True)
    return project_root


def _write_summary(summary: dict[str, Any], project_root: Path) -> dict[str, Any]:
    _ensure_directory(project_root)
    summary_path = project_root / "reports" / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_training(
    model_names: list[str] | tuple[str, ...] | None = None,
    preset: str = "paper",
    quick: bool = False,
    project_root: str | Path | None = None,
    balance: bool = True,
    data_dir_override: str | Path | None = None,
    model_dir_override: str | Path | None = None,
    fine_tune: bool = False,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    use_configured_model_dir = project_root is None
    project_root = Path(project_root) if project_root is not None else SETTINGS.project_root
    project_root = _ensure_directory(project_root)

    requested = list(model_names or ["inception_v3"])
    supported = list_supported_models()
    for name in requested:
        if name not in supported:
            raise ValueError(f"Unsupported model {name}. Choose from {supported}")

    if quick:
        data_dir = project_root / "data" / "smoke" / "processed"
    elif data_dir_override is not None:
        data_dir = Path(data_dir_override)
    else:
        data_dir = project_root / "data" / "processed"
    if quick:
        make_smoke_dataset(data_dir)
    if model_dir_override is not None:
        model_dir = Path(model_dir_override)
    elif use_configured_model_dir:
        model_dir = SETTINGS.models_dir
    else:
        model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "skipped",
        "preset": preset,
        "quick": quick,
        "balance": balance,
        "fine_tune": fine_tune,
        "training_strategy": (
            "custom_attention_cnn_from_scratch"
            if requested == ["attention_cnn"]
            else "mixed_model_specific"
            if "attention_cnn" in requested
            else "fine_tuning"
            if fine_tune
            else "frozen_backbone_feature_extraction"
        ),
        "project_root": str(project_root),
        "training_data_dir": str(data_dir),
        "models": [],
        "note": "No real training was executed because TensorFlow is unavailable in this environment.",
    }
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["dataset"] = manifest.get("dataset")
        summary["dataset_manifest"] = str(manifest_path)
        summary["class_mapping"] = manifest.get("class_mapping")
        summary["negative_class_note"] = manifest.get("negative_class_note")
        summary["modality_note"] = manifest.get("modality_note")

    raw_dir = project_root / "data" / "raw"
    if not quick and data_dir_override is None and (not raw_dir.exists() or not any(path.is_file() for path in raw_dir.rglob("*"))):
        summary["note"] = "Training skipped: add real source data under data/raw before training."
        return _write_summary(summary, project_root)

    if tf is None:
        return _write_summary(summary, project_root)

    if epochs_override is not None and epochs_override < 1:
        raise ValueError("epochs_override must be at least 1")
    epochs = 1 if quick else epochs_override or (10 if "attention_cnn" in requested else 2)
    batch_size = 8 if quick else 16
    training_records: list[dict[str, Any]] = []

    for name in requested:
        input_size = MODEL_SPECS[name]["input_size"]
        backbone_frozen = name != "attention_cnn" and not fine_tune
        model = build_model(
            name,
            input_shape=(input_size, input_size, 3),
            num_classes=2,
            freeze_backbone=backbone_frozen,
        )
        model_path = model_dir / f"{name}_{preset}.keras"
        train_ds = build_image_dataset(
            data_dir / "train",
            image_size=(input_size, input_size),
            batch_size=batch_size,
            model_name=name,
        )
        val_ds = build_image_dataset(
            data_dir / "val",
            image_size=(input_size, input_size),
            batch_size=batch_size,
            model_name=name,
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
        model.save(model_path)

        last_epoch = history.history
        record = {
            "name": name,
            "status": "trained",
            "model_path": str(model_path),
            "epochs": epochs,
            "backbone_frozen": backbone_frozen,
            "training_strategy": (
                "custom_attention_cnn_from_scratch"
                if name == "attention_cnn"
                else "fine_tuning" if fine_tune else "frozen_backbone_feature_extraction"
            ),
            "final_train_accuracy": float(last_epoch["accuracy"][-1]),
            "final_val_accuracy": float(last_epoch["val_accuracy"][-1]),
        }
        training_records.append(record)

    summary.update({
        "status": "ok",
        "models": training_records,
        "note": (
            "Training completed using generated synthetic smoke data; these metrics are only for pipeline validation."
            if quick
            else f"Training completed using {data_dir}; metrics describe this local dataset and are not paper or clinical results."
        ),
    })
    return _write_summary(summary, project_root)


def main() -> None:
    args = parse_args()
    summary = run_training(
        model_names=args.models,
        preset=args.preset,
        quick=args.quick,
        project_root=Path(__file__).resolve().parents[2],
        balance=args.balance,
        data_dir_override=args.data_dir,
        model_dir_override=args.model_dir,
        fine_tune=args.fine_tune,
        epochs_override=args.epochs,
    )

    if summary["status"] == "ok":
        print(json.dumps({"status": "ok", "models": summary["models"]}, indent=2))
    else:
        print(json.dumps({"status": "skipped", "note": summary["note"]}, indent=2))


if __name__ == "__main__":
    main()
