#!/usr/bin/env python3
"""Create confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    m, classes = labels.shape
    result = np.zeros((classes, classes))

    for i in range(m):
        sub_label = labels[i]
        sub_logits = logits[i]

        x, *_ = np.where(sub_label == 1)
        y, *_ = np.where(sub_logits == 1)
        result[x[0]][y[0]] += 1

    return result
