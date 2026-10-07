"""
End-to-End Pipeline Testing
Binary → Multiclass Classification Pipeline
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import preprocess_single_image
from model_architecture import load_model_with_custom_objects


class PipelineConfig:
    """Pipeline configuration"""
    MODEL_DIR = "/content/drive/MyDrive/brain_tumor_models"
    BINARY_TEST_DIR = "/content/drive/MyDrive/brain_tumor_split/binary/test"
    MULTI_TEST_DIR = "/content/drive/MyDrive/brain_tumor_split/multiclass/test"
    IMG_SIZE = 224


def load_models(config=None):
    """
    Load both binary and multiclass models.
    
    Args:
        config: Configuration object
        
    Returns:
        binary_model, multiclass_model
    """
    if config is None:
        config = PipelineConfig()
    
    print("📦 Loading models...")
    
    binary_model = load_model_with_custom_objects(
        f"{config.MODEL_DIR}/binary_best.keras"
    )
    
    multiclass_model = load_model_with_custom_objects(
        f"{config.MODEL_DIR}/multiclass_best.keras"
    )
    
    print("✅ Models loaded successfully\n")
    
    return binary_model, multiclass_model


def test_binary_model(binary_model, test_dir, img_size=224):
    """
    Test binary classification model.
    
    Args:
        binary_model: Loaded binary model
        test_dir: Binary test directory
        img_size: Image size
        
    Returns:
        dict: Test results
    """
    print("📊 TESTING BINARY MODEL\n")
    
    binary_class_map = {
        "no_tumor": 0,
        "tumor": 1
    }
    
    imgs = []
    labels = []
    
    for cls, label in binary_class_map.items():
        cls_path = os.path.join(test_dir, cls)
        for img in os.listdir(cls_path):
            imgs.append(os.path.join(cls_path, img))
            labels.append(label)
    
    labels = np.array(labels)
    
    # Predictions
    probs = []
    preds = []
    
    for img_path in imgs:
        x = preprocess_single_image(img_path, img_size)
        prob = binary_model.predict(x, verbose=0)[0][0]
        probs.append(prob)
        preds.append(1 if prob >= 0.5 else 0)
    
    probs = np.array(probs)
    preds = np.array(preds)
    
    # Results
    print(classification_report(
        labels,
        preds,
        target_names=["No Tumor", "Tumor"]
    ))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    
    from sklearn.metrics import roc_auc_score
    print(f"ROC-AUC: {roc_auc_score(labels, probs):.4f}\n")
    
    return {
        'labels': labels,
        'predictions': preds,
        'probabilities': probs
    }


def test_multiclass_model(multiclass_model, test_dir, img_size=224):
    """
    Test multiclass classification model.
    
    Args:
        multiclass_model: Loaded multiclass model
        test_dir: Multiclass test directory
        img_size: Image size
        
    Returns:
        dict: Test results
    """
    print("📊 TESTING MULTICLASS MODEL\n")
    
    multi_class_map = {
        "Glioma": 0,
        "Meningioma": 1,
        "Normal": 2,
        "Pituitary": 3
    }
    
    imgs = []
    labels = []
    
    for cls, label in multi_class_map.items():
        cls_path = os.path.join(test_dir, cls)
        for img in os.listdir(cls_path):
            imgs.append(os.path.join(cls_path, img))
            labels.append(label)
    
    labels = np.array(labels)
    
    # Predictions
    preds = []
    
    for img_path in imgs:
        x = preprocess_single_image(img_path, img_size)
        probs = multiclass_model.predict(x, verbose=0)[0]
        preds.append(np.argmax(probs))
    
    preds = np.array(preds)
    
    # Results
    print(classification_report(
        labels,
        preds,
        target_names=list(multi_class_map.keys())
    ))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    print()
    
    return {
        'labels': labels,
        'predictions': preds,
        'images': imgs
    }


def test_end_to_end_pipeline(binary_model, multiclass_model, test_dir, img_size=224):
    """
    Test complete pipeline: Binary → Multiclass.
    
    Args:
        binary_model: Loaded binary model
        multiclass_model: Loaded multiclass model
        test_dir: Multiclass test directory
        img_size: Image size
        
    Returns:
        dict: Pipeline test results
    """
    print("📊 TESTING END-TO-END PIPELINE (Binary → Multiclass)\n")
    
    multi_class_map = {
        "Glioma": 0,
        "Meningioma": 1,
        "Normal": 2,
        "Pituitary": 3
    }
    
    imgs = []
    labels = []
    
    for cls, label in multi_class_map.items():
        cls_path = os.path.join(test_dir, cls)
        for img in os.listdir(cls_path):
            imgs.append(os.path.join(cls_path, img))
            labels.append(label)
    
    labels = np.array(labels)
    
    # Pipeline predictions
    pipeline_preds = []
    
    for img_path in imgs:
        x = preprocess_single_image(img_path, img_size)
        
        # Stage 1: Binary classification
        bin_prob = binary_model.predict(x, verbose=0)[0][0]
        
        if bin_prob < 0.5:
            # Classified as No Tumor → map to Normal
            pipeline_preds.append(2)  # Normal
        else:
            # Classified as Tumor → use multiclass model
            probs = multiclass_model.predict(x, verbose=0)[0]
            pipeline_preds.append(np.argmax(probs))
    
    pipeline_preds = np.array(pipeline_preds)
    
    # Results
    print(classification_report(
        labels,
        pipeline_preds,
        target_names=list(multi_class_map.keys())
    ))
    print("Confusion Matrix:\n", confusion_matrix(labels, pipeline_preds))
    print()
    
    return {
        'labels': labels,
        'predictions': pipeline_preds,
        'images': imgs
    }


def main():
    """Main testing pipeline"""
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE TESTING")
    print("="*70 + "\n")
    
    config = PipelineConfig()
    
    # Load models
    binary_model, multiclass_model = load_models(config)
    
    # Test Binary Model
    print("="*70)
    binary_results = test_binary_model(
        binary_model, 
        config.BINARY_TEST_DIR, 
        config.IMG_SIZE
    )
    
    # Test Multiclass Model
    print("="*70)
    multi_results = test_multiclass_model(
        multiclass_model, 
        config.MULTI_TEST_DIR, 
        config.IMG_SIZE
    )
    
    # Test End-to-End Pipeline
    print("="*70)
    pipeline_results = test_end_to_end_pipeline(
        binary_model,
        multiclass_model,
        config.MULTI_TEST_DIR,
        config.IMG_SIZE
    )
    
    print("="*70)
    print("✅ ALL TESTING COMPLETED")
    print("="*70 + "\n")
    
    return binary_results, multi_results, pipeline_results


if __name__ == "__main__":
    main()
