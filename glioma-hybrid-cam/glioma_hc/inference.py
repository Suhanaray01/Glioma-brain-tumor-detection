from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from glioma_hc.config import SETTINGS
from glioma_hc.explainability.gradcam import GradCAMUnavailable, make_gradcam_overlay
from glioma_hc.explainability.hybrid_cam import HybridCAMUnavailable, make_hybrid_cam
from glioma_hc.training.data_loader import preprocess_model_input
from glioma_hc.training.model_zoo import ChannelPool2D, MODEL_SPECS


GRADCAM_LAYERS = {
    "inception_v3": "mixed10",
    "attention_cnn": "target_conv_activation",
    "multiscale_cbam_cnn": "target_conv_activation",
}


class InferenceUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


def _load_keras_model(path: Path) -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise InferenceUnavailable("TensorFlow is unavailable in the server environment.") from exc
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "ChannelPool2D": ChannelPool2D,
            "GliomaScanHC>ChannelPool2D": ChannelPool2D,
        },
    )


def _load_eligible_models() -> list[tuple[str, Any]]:
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    if not summary_path.is_file():
        raise InferenceUnavailable("No completed non-smoke training run was recorded.")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceUnavailable("The training summary is unavailable or invalid.") from exc

    if summary.get("status") != "ok" or summary.get("quick") is not False:
        raise InferenceUnavailable("Predictions require a completed non-smoke training run.")

    models_root = SETTINGS.models_dir.resolve()
    loaded_models = []
    for record in summary.get("models", []):
        model_path = Path(record.get("model_path", "")).resolve()
        if record.get("status") != "trained" or model_path.suffix != ".keras":
            continue
        if not model_path.is_relative_to(models_root) or not model_path.is_file():
            continue
        try:
            model = _load_keras_model(model_path)
        except Exception as exc:
            raise InferenceUnavailable(f"Could not load trained model {record.get('name', 'unknown')}.") from exc
        if not isinstance(model.input_shape, tuple) or len(model.input_shape) != 4:
            raise InferenceUnavailable("A trained model has an unsupported input shape.")
        if not isinstance(model.output_shape, tuple) or model.output_shape[-1] != len(SETTINGS.label_order):
            raise InferenceUnavailable("A trained model has an unsupported output shape.")
        loaded_models.append((str(record.get("name", model.name)), model))

    if not loaded_models:
        raise InferenceUnavailable("No usable trained model checkpoints were found.")
    return loaded_models


def predict_images(
    image_files: list[tuple[str, bytes]],
    aggregation: str = "max",
    explain: str = "gradcam",
    sequence: str = "UNSPECIFIED",
) -> dict[str, Any]:
    if not image_files:
        raise InvalidImage("Upload at least one raster image.")
    if aggregation not in {"max", "mean"}:
        raise InvalidImage("Aggregation must be 'max' or 'mean'.")
    if explain not in {"gradcam", "hybrid_cam", "none"}:
        raise InvalidImage("Explain must be 'gradcam', 'hybrid_cam', or 'none'.")

    images = []
    for filename, content in image_files:
        try:
            with Image.open(BytesIO(content)) as image:
                images.append((filename, image.convert("RGB")))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImage(f"{filename} is not a supported raster image.") from exc

    models = _load_eligible_models()
    per_model_scores: list[np.ndarray] = []
    model_results = []
    first_model_input = None
    first_model_name = ""
    first_model_images = []
    for model_index, (name, model) in enumerate(models):
        _, height, width, channels = model.input_shape
        if not all(isinstance(size, int) and size > 0 for size in (height, width)) or channels != 3:
            raise InferenceUnavailable(f"Model {name} has an unsupported input shape.")
        resized_images = [
            np.asarray(
                image.resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            for _, image in images
        ]
        batch = np.stack([
            np.asarray(
                resized,
                dtype=np.float32,
            )
            for resized in resized_images
        ])
        batch = preprocess_model_input(batch, name)
        if model_index == 0:
            first_model_input = batch
            first_model_name = name.lower().replace("-", "_")
            first_model_images = resized_images

        started = time.perf_counter()
        scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if scores.shape != (len(images), len(SETTINGS.label_order)):
            raise InferenceUnavailable(f"Model {name} returned an unsupported prediction shape.")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise InferenceUnavailable(f"Model {name} returned invalid probabilities.")
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
            raise InferenceUnavailable(f"Model {name} outputs are not normalized probabilities.")

        per_model_scores.append(scores)
        mean_scores = scores.mean(axis=0)
        model_results.append({
            "name": name,
            "probs": mean_scores.tolist(),
            "label": SETTINGS.label_order[int(np.argmax(mean_scores))],
            "inference_ms": round(elapsed_ms, 3),
        })

    ensemble_scores = np.mean(per_model_scores, axis=0)
    glioma_slice_scores = ensemble_scores[:, SETTINGS.binary_index]
    glioma_probability = float(
        glioma_slice_scores.max() if aggregation == "max" else glioma_slice_scores.mean()
    )
    label = SETTINGS.label_order[int(glioma_probability >= 0.5)]
    gradcam_overlay = None
    hybrid_cam_overlay = None
    attribution_overlap = None
    gradcam_status = "disabled" if explain == "none" else "unavailable"
    hybrid_cam_status = "disabled" if explain != "hybrid_cam" else "unavailable"
    gradcam_reason = "Explainability was not requested." if explain == "none" else ""
    hybrid_cam_reason = "Hybrid-CAM was not requested." if explain != "hybrid_cam" else ""
    gradcam_slice_index = None
    if explain in {"gradcam", "hybrid_cam"}:
        gradcam_slice_index = int(np.argmax(glioma_slice_scores))
        cam_layer = GRADCAM_LAYERS.get(first_model_name)
        if cam_layer is None:
            gradcam_reason = f"Grad-CAM is not configured for {first_model_name}."
            hybrid_cam_reason = gradcam_reason
        else:
            try:
                gradcam_overlay = make_gradcam_overlay(
                    models[0][1],
                    first_model_input[gradcam_slice_index:gradcam_slice_index + 1],
                    first_model_images[gradcam_slice_index],
                    layer_name=cam_layer,
                    class_index=SETTINGS.binary_index,
                )
                gradcam_status = "generated"
            except GradCAMUnavailable as exc:
                gradcam_reason = str(exc)
                hybrid_cam_reason = gradcam_reason

        if explain == "hybrid_cam" and gradcam_status == "generated":
            def lime_predict(batch_images: np.ndarray) -> np.ndarray:
                model_batch = preprocess_model_input(np.asarray(batch_images, dtype=np.float32), first_model_name)
                return np.asarray(models[0][1].predict(model_batch, verbose=0), dtype=np.float32)

            try:
                hybrid_result = make_hybrid_cam(
                    models[0][1],
                    first_model_input[gradcam_slice_index:gradcam_slice_index + 1],
                    first_model_images[gradcam_slice_index],
                    lime_predict,
                    SETTINGS.binary_index,
                    layer_name=cam_layer,
                )
                hybrid_cam_overlay = hybrid_result["overlay"]
                attribution_overlap = hybrid_result["attribution_overlap"]
                hybrid_cam_status = "generated"
                hybrid_cam_reason = ""
            except HybridCAMUnavailable as exc:
                hybrid_cam_reason = str(exc)

    return {
        "id": str(uuid4()),
        "label": label,
        "glioma_probability": glioma_probability,
        "confidence": max(glioma_probability, 1.0 - glioma_probability),
        "threshold": 0.5,
        "aggregation": aggregation,
        "ensemble": {
            "soft": {
                "label": label,
                "probs": [1.0 - glioma_probability, glioma_probability],
            },
        },
        "models": model_results,
        "slices": [
            {
                "index": index,
                "sequence": sequence.upper(),
                "used_for_classification": True,
                "label": SETTINGS.label_order[int(score >= 0.5)],
                "glioma_probability": float(score),
            }
            for index, score in enumerate(glioma_slice_scores)
        ],
        "preprocessing": {
            "method": "RGB conversion, bilinear resize, then the selected Keras backbone's preprocess_input",
        },
        "explain": {
            "requested": explain,
            "status": hybrid_cam_status if explain == "hybrid_cam" else gradcam_status,
            "reason": hybrid_cam_reason if explain == "hybrid_cam" else gradcam_reason,
            "gradcam_overlay": gradcam_overlay,
            "hybrid_cam_overlay": hybrid_cam_overlay,
            "attribution_overlap": attribution_overlap,
            "slice_index": gradcam_slice_index,
            "model": first_model_name or None,
        },
        "timing_ms": round(sum(item["inference_ms"] for item in model_results), 3),
        "warnings": (
            (["MRI sequence was not specified by the dataset."] if sequence.upper() == "UNSPECIFIED" else [])
            + ([f"{explain} unavailable: {hybrid_cam_reason if explain == 'hybrid_cam' else gradcam_reason}"] if explain in {"gradcam", "hybrid_cam"} and (hybrid_cam_status if explain == "hybrid_cam" else gradcam_status) != "generated" else [])
        ),
        "disclaimer": "Research and educational use only; not a clinical diagnostic device.",
    }