from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
from lime import lime_image

from glioma_hc.explainability.gradcam import encode_overlay, make_gradcam_heatmap


class HybridCAMUnavailable(RuntimeError):
    pass


def fuse_heatmaps(
    gradcam_heatmap: np.ndarray,
    lime_mask: np.ndarray,
    gradcam_weight: float = 0.5,
) -> tuple[np.ndarray, float]:
    if not 0 <= gradcam_weight <= 1:
        raise ValueError("gradcam_weight must be between 0 and 1")

    gradcam = np.asarray(gradcam_heatmap, dtype=np.float32)
    lime = np.asarray(lime_mask, dtype=np.float32)
    if gradcam.ndim != 2 or lime.ndim != 2:
        raise ValueError("Grad-CAM and LIME inputs must be 2D heatmaps")

    gradcam = cv2.resize(lime_safe_normalize(gradcam), (lime.shape[1], lime.shape[0]))
    lime = lime_safe_normalize(lime)
    combined = (gradcam_weight * gradcam) + ((1 - gradcam_weight) * lime)
    combined = lime_safe_normalize(combined)

    gradcam_support = gradcam >= np.quantile(gradcam, 0.8)
    lime_support = lime > 0
    union = np.logical_or(gradcam_support, lime_support).sum()
    intersection = np.logical_and(gradcam_support, lime_support).sum()
    overlap = float(intersection / union) if union else 0.0
    return combined, overlap


def lime_safe_normalize(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0)
    maximum = float(heatmap.max()) if heatmap.size else 0.0
    return heatmap / maximum if maximum > 0 else np.zeros_like(heatmap)


def make_hybrid_cam(
    model: Any,
    model_input: Any,
    original_rgb: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    class_index: int,
    layer_name: str = "mixed10",
    num_samples: int = 96,
    segmentation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    try:
        gradcam_heatmap = make_gradcam_heatmap(model, model_input, layer_name, class_index)
        explainer = lime_image.LimeImageExplainer(random_state=42)
        explanation = explainer.explain_instance(
            np.asarray(original_rgb, dtype=np.uint8),
            classifier_fn=predict_fn,
            labels=(class_index,),
            top_labels=None,
            hide_color=0,
            num_samples=num_samples,
            batch_size=16,
            segmentation_fn=segmentation_fn,
        )
        _, lime_mask = explanation.get_image_and_mask(
            class_index,
            positive_only=True,
            num_features=8,
            hide_rest=False,
        )
        combined, overlap = fuse_heatmaps(gradcam_heatmap, lime_mask)
        overlay = encode_overlay(original_rgb, combined, alpha=0.46)
    except Exception as exc:
        raise HybridCAMUnavailable(f"Could not generate the LIME/Grad-CAM fusion: {exc}") from exc
    return {
        "overlay": overlay,
        "attribution_overlap": overlap,
        "fusion": "equal-weight normalized Grad-CAM and positive LIME superpixel mask",
        "lime_samples": num_samples,
    }