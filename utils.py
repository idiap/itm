# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0


import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def classification_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F1 score, accuracy, and F1 score for each class

    Parameters:
    y_true (numpy array): ground truth labels
    y_pred (numpy array): predicted labels

    Returns:
    dict: A dictionary containing precision, recall, F1 score, accuracy, and F1 score for each class
    """
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate F1 score
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Create a dictionary with the results
    metrics = {
        "Overview": {
            "Accuracy": accuracy*100,
            "F1 Score": macro_f1*100
        },
        "Class 0": {
            "Precision": precision[0]*100,
            "Recall": recall[0]*100,
            "F1 Score": f1[0]*100
        },
        "Class 1": {
            "Precision": precision[1]*100,
            "Recall": recall[1]*100,
            "F1 Score": f1[1]*100
        }
    }

    return metrics