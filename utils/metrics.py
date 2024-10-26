"""Metrics for evaluating the performance of a classification model.
"""

from typing import Tuple

import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
import torch
from scipy.special import expit  # Import sigmoid function
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def calculate_metrics_from_logits1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    # Convert logits to probabilities using sigmoid
    y_probs = expit(y_pred) #
    # Convert probabilities to binary labels using a threshold of 0.5
    y_pred_bin = (y_probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_true[:,1], y_pred_bin[:,1])
    return accuracy

def calculate_metrics_from_logits(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    # Convert logits to probabilities using sigmoid
    y_probs = expit(y_pred)
    # Convert probabilities to binary labels using a threshold of 0.5
    y_pred_bin = (y_probs >= 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true[:, 1], y_pred_bin[:, 1])
    print("confusion matrix:", cm)
    accuracy = accuracy_score(y_true[:, 1], y_pred_bin[:, 1])
    auc = roc_auc_score(y_true, y_probs)
    precision = precision_score(y_true[:, 1], y_pred_bin[:, 1])
    recall = recall_score(y_true[:, 1], y_pred_bin[:, 1])
    f1 = f1_score(y_true[:, 1], y_pred_bin[:, 1])
    return accuracy, auc, precision, recall, f1





