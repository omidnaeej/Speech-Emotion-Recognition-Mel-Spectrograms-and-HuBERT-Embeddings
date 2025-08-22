import torch

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, outdir: str = None):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred)
    
    if outdir:
        plot_confusion_matrix(cm, outdir, labels=["NEU", "HAP", "SAD", "ANG"])
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray):
    # Set zero_division=0 to suppress UndefinedMetricWarning
    report = classification_report(y_true, y_pred, target_names=["NEU", "HAP", "SAD", "ANG"], output_dict=True, zero_division=0)
    return report

def plot_confusion_matrix(cm: np.ndarray, outdir: str, labels: list):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cm.png"))
    plt.close()