"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
For Brain Tumor Localization and Explainability
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
import os


def get_last_conv_layer(model):
    """
    Find the best convolutional layer for Grad-CAM.
    
    Args:
        model: Keras model
        
    Returns:
        str: Name of the last convolutional layer
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape'):
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if shape is not None and len(shape) == 4:
                return layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return model.layers[-2].name


def make_gradcam_heatmap(img_array, model, layer_name):
    """
    Generate Grad-CAM heatmap.
    
    Args:
        img_array: Preprocessed image array
        model: Keras model
        layer_name: Name of target layer
        
    Returns:
        numpy array: Heatmap
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        layer_output, predictions = grad_model(img_array)
        pred_score = predictions[0][0]

    grads = tape.gradient(pred_score, layer_output)

    # Handle different output shapes
    if len(layer_output.shape) == 4:  # Conv2D: (batch, h, w, channels)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    elif len(layer_output.shape) == 3:  # Transformer: (batch, patches, features)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        layer_output = layer_output[0]
        heatmap = tf.reduce_sum(layer_output * pooled_grads, axis=-1)
        grid_size = int(np.sqrt(len(heatmap)))
        heatmap = tf.reshape(heatmap, (grid_size, grid_size))
    else:
        heatmap = tf.ones((14, 14))

    # Normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def create_tumor_overlay(img_path, heatmap, is_tumor, threshold=0.5):
    """
    Create tumor localization overlay on original MRI image.
    
    Args:
        img_path: Path to original image
        heatmap: Grad-CAM heatmap
        is_tumor: Boolean indicating if tumor is present
        threshold: Threshold for creating mask
        
    Returns:
        original: Original image
        overlay: Image with tumor overlay
    """
    # Load original
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w = original.shape[:2]

    # Ensure heatmap is 2D
    heatmap = np.array(heatmap, dtype=np.float32)
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap)

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # If NO TUMOR, return original image without overlay
    if not is_tumor:
        return original, original

    # For TUMOR cases: apply moderate smoothing
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)

    # Normalize to full range
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    heatmap_min = heatmap_resized.min()
    heatmap_max = heatmap_resized.max()
    if heatmap_max > heatmap_min:
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)

    # Use adaptive thresholding
    threshold_value = np.percentile(heatmap_resized, 80)
    threshold_value = max(threshold_value, threshold)

    # Create binary mask
    mask = (heatmap_resized >= threshold_value).astype(np.uint8)

    # Morphological operations to clean mask
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.dilate(mask, kernel_small, iterations=1)

    # Keep only significant regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.01  # At least 1% of image
    mask_cleaned = np.zeros_like(mask)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if valid_contours:
        cv2.drawContours(mask_cleaned, valid_contours, -1, 1, -1)
        mask_cleaned = cv2.GaussianBlur(mask_cleaned.astype(np.float32), (15, 15), 0)
        mask_cleaned = (mask_cleaned > 0.5).astype(np.uint8)

    # If no significant regions, use less strict threshold
    if mask_cleaned.sum() == 0:
        threshold_value = np.percentile(heatmap_resized, 70)
        mask = (heatmap_resized >= threshold_value).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area * 0.5]
        if valid_contours:
            cv2.drawContours(mask_cleaned, valid_contours, -1, 1, -1)

    if mask_cleaned.sum() == 0:
        return original, original

    # Create colored heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), 
                                       cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Apply overlay with smooth blending
    overlay = original.copy().astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_cleaned.astype(np.float32), (11, 11), 0)
    mask_3ch = np.stack([mask_smooth] * 3, axis=-1)

    alpha = 0.6
    overlay = (original * (1 - alpha * mask_3ch) + 
              heatmap_colored * alpha * mask_3ch).astype(np.uint8)

    return original, overlay


def visualize_gradcam(model, img_path, layer_name=None, threshold=0.5):
    """
    Complete Grad-CAM visualization pipeline.
    
    Args:
        model: Trained Keras model
        img_path: Path to image file
        layer_name: Target layer (auto-detected if None)
        threshold: Threshold for mask creation
    """
    from tensorflow.keras.preprocessing import image
    
    # Auto-detect layer if not specified
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        print(f"🎯 Using layer: {layer_name}")

    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    pred = model.predict(x, verbose=0)[0][0]
    is_tumor = pred >= 0.5
    label = 1 if is_tumor else 0
    confidence = pred if is_tumor else (1 - pred)

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(x, model, layer_name)
    original, overlay = create_tumor_overlay(img_path, heatmap, is_tumor, threshold)

    # Visualize
    fig = plt.figure(figsize=(8, 4))

    # Left panel: Original MRI
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(original)
    ax1.set_title(f'MRI Image (Label={label})', fontsize=12)
    ax1.axis('off')

    # Right panel: Tumor Mask Overlay
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title('Tumor Mask Overlay', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Prediction: {'Tumor' if is_tumor else 'No Tumor'}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Raw Score: {pred:.4f}")

    return original, overlay, heatmap


def batch_visualize_gradcam(model, test_dir, num_samples=5, layer_name=None):
    """
    Generate Grad-CAM visualizations for multiple images.
    
    Args:
        model: Trained Keras model
        test_dir: Directory containing test images
        num_samples: Number of samples to visualize
        layer_name: Target layer (auto-detected if None)
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        print(f"🎯 Using layer: {layer_name}\n")

    # Collect image paths
    test_images = []
    for cls in os.listdir(test_dir):
        cls_path = os.path.join(test_dir, cls)
        if os.path.isdir(cls_path):
            for img_name in sorted(os.listdir(cls_path))[:num_samples//2 + 1]:
                test_images.append(os.path.join(cls_path, img_name))
    
    test_images = test_images[:num_samples]

    # Process each image
    for idx, img_path in enumerate(test_images):
        print(f"{'='*70}")
        print(f"Processing {idx+1}/{len(test_images)}: {os.path.basename(img_path)}")
        print(f"{'='*70}")
        visualize_gradcam(model, img_path, layer_name)
        print()

    print("✅ Grad-CAM visualization complete")
