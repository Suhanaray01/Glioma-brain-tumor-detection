"""
Comprehensive Model Evaluation with Medical AI Metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, recall_score, precision_score,
    f1_score, roc_auc_score
)


def plot_confusion_matrix(y_true, y_pred, classes, model_type, save_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        model_type: 'binary' or 'multiclass'
        save_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_type.upper()}', 
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{model_type}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_type, save_dir, class_names=None):
    """
    Plot ROC curve for binary or multiclass classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_type: 'binary' or 'multiclass'
        save_dir: Directory to save plot
        class_names: List of class names (for multiclass)
    """
    plt.figure(figsize=(10, 8))

    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random')

    else:
        # Multiclass classification
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(y_pred_proba.shape[1]))

        for i in range(y_pred_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            label = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_type.upper()}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{model_type}_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curve saved: {save_path}")
    plt.show()


def comprehensive_evaluation(model, test_gen, model_type, save_dir):
    """
    Complete medical AI evaluation with all metrics.
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator
        model_type: 'binary' or 'multiclass'
        save_dir: Directory to save results
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*70}")
    print(f"📊 COMPREHENSIVE EVALUATION - {model_type.upper()}")
    print(f"{'='*70}\n")

    # Get predictions
    print("🔄 Generating predictions...")
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes

    # Get class names
    class_names = list(test_gen.class_indices.keys())

    # Binary vs Multiclass
    is_binary = len(class_names) == 2

    if is_binary:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba_flat = y_pred_proba.flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_proba_flat = y_pred_proba

    # Basic metrics
    print(f"\n{'='*70}")
    print("📈 PERFORMANCE METRICS")
    print(f"{'='*70}\n")

    test_results = model.evaluate(test_gen, verbose=0)
    print(f"Test Loss:     {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC:      {test_results[2]:.4f}\n")

    # Medical AI metrics
    if is_binary:
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{'='*70}")
        print("🏥 MEDICAL AI METRICS (FDA Requirements)")
        print(f"{'='*70}\n")
        print(f"Sensitivity (Recall):  {sensitivity:.4f}  "
              f"{'✅' if sensitivity >= 0.95 else '⚠️'}  (Target: ≥0.95)")
        print(f"Specificity:           {specificity:.4f}  "
              f"{'✅' if specificity >= 0.90 else '⚠️'}  (Target: ≥0.90)")
        print(f"Precision (PPV):       {precision:.4f}")
        print(f"F1-Score:              {f1:.4f}\n")

        # Calculate NPV
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = precision

        print(f"True Positives:  {tp}")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}\n")
        print(f"PPV (Precision): {ppv:.4f}")
        print(f"NPV:             {npv:.4f}\n")

    else:
        # Multiclass metrics
        print(f"{'='*70}")
        print("🏥 PER-CLASS METRICS")
        print(f"{'='*70}\n")

        for i, class_name in enumerate(class_names):
            mask = (y_true == i)
            if mask.sum() > 0:
                sens = recall_score(y_true == i, y_pred == i)
                prec = precision_score(y_true == i, y_pred == i, zero_division=0)
                print(f"{class_name:20s} - Sensitivity: {sens:.4f}, "
                      f"Precision: {prec:.4f}")
        print()

    # Classification report
    print(f"{'='*70}")
    print("📋 DETAILED CLASSIFICATION REPORT")
    print(f"{'='*70}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Save report to file
    report_path = os.path.join(save_dir, f'{model_type}_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"EVALUATION REPORT - {model_type.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Loss:     {test_results[0]:.4f}\n")
        f.write(f"Test Accuracy: {test_results[1]:.4f}\n")
        f.write(f"Test AUC:      {test_results[2]:.4f}\n\n")

        if is_binary:
            f.write("MEDICAL AI METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Precision:   {precision:.4f}\n")
            f.write(f"F1-Score:    {f1:.4f}\n")
            f.write(f"PPV:         {ppv:.4f}\n")
            f.write(f"NPV:         {npv:.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*70 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, 
                                     digits=4))

    print(f"✅ Report saved: {report_path}\n")

    # Visualizations
    plot_confusion_matrix(y_true, y_pred, class_names, model_type, save_dir)
    plot_roc_curve(y_true, y_pred_proba_flat, model_type, save_dir, class_names)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': test_results
    }
