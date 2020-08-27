
"""
The custom metric `gini` coefficient for this project -> 2 * auc - 1 (when there's no same scores for different label)
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/overview/evaluation
"""

import numpy as np
from evan_utils.metric import binary_auc
from typing import List, Union


def _gini(y_true: Union[np.ndarray, List], y_scores: Union[np.ndarray, List]):
    n = len(y_true)
    if len(y_scores) != n:
        raise ValueError("The pred score must have same length with true labels.")

    y_true, y_score = map(np.asarray, [y_true, y_scores])
    order = np.lexsort((np.arange(n), -1 * y_score))  # sorted by score desc and index asc
    y_true = y_true[order]
    total = y_true.sum()
    gini_sum = y_true.cumsum().sum() / total
    gini_sum -= (n + 1) / 2.
    return gini_sum / n


def gini_normalized(y_true: np.ndarray, y_scores: np.ndarray):
    return _gini(y_true, y_scores) / _gini(y_true, y_true)


def gini(y_true: np.ndarray, y_scores: np.ndarray):
    return 2 * binary_auc(y_true, y_scores) - 1
