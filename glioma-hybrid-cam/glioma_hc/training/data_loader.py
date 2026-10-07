from __future__ import annotations

from pathlib import Path
from typing import Any

from glioma_hc.config import SETTINGS

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - optional for smoke-only environments
    tf = None  # type: ignore[assignment]


def build_image_dataset(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
    validation_split: float | None = None,
    subset: str | None = None,
    seed: int = 42,
    model_name: str | None = None,
    shuffle: bool = True,
):
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required for dataset loading and model training.")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {data_path}")

    if validation_split is not None:
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(data_path),
            labels="inferred",
            class_names=SETTINGS.label_order,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
        )
    else:
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(data_path),
            labels="inferred",
            class_names=SETTINGS.label_order,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
        )

    if model_name is not None:
        dataset = dataset.map(
            lambda images, labels: (preprocess_model_input(images, model_name), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset


def preprocess_model_input(images: Any, model_name: str) -> Any:
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required to preprocess model inputs.")
    name = model_name.lower().replace("-", "_")
    if name in {"attention_cnn", "multiscale_cbam_cnn"}:
        return tf.cast(images, tf.float32) / 255.0
    application_name = "resnet" if name in {"resnet50", "resnet101"} else name
    applications = {
        "vgg16": tf.keras.applications.vgg16,
        "vgg19": tf.keras.applications.vgg19,
        "resnet": tf.keras.applications.resnet,
        "densenet201": tf.keras.applications.densenet,
        "xception": tf.keras.applications.xception,
        "inception_v3": tf.keras.applications.inception_v3,
        "inception_resnet_v2": tf.keras.applications.inception_resnet_v2,
    }
    try:
        preprocess_input = applications[application_name].preprocess_input
    except KeyError as exc:
        raise ValueError(f"Unsupported model preprocessing: {model_name}") from exc
    return preprocess_input(tf.cast(images, tf.float32))
