#!/usr/bin/env python3
"""F1 Score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    sens_v = sensitivity(confusion)
    prec_v = precision(confusion)

    f1 = np.divide(2, np.power(sens_v, -1) + np.power(prec_v, -1))
    return f1
