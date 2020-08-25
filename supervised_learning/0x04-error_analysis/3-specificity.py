#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class
    in a confusion matrix
    """
    classes, _ = confusion.shape
    result = np.zeros(classes)

    total = np.sum(confusion)

    for cl in range(classes):
        true_positive = confusion[cl][cl]
        false_positive = np.sum(confusion[cl]) - true_positive
        false_negative = np.sum(confusion[:, cl]) - true_positive

        sub_total = total - false_positive - false_negative - true_positive

        result[cl] = np.divide(
            sub_total,
            sub_total + false_negative
        )

    return result
