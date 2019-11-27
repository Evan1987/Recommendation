
import numpy as np
from numba import njit
from typing import Any, Union, List


@njit
def recall(gt_item: Any, pred_items: Union[List[Any], np.ndarray]) -> float:
    for i in range(len(pred_items)):
        if gt_item == pred_items[i]:
            return 1.0
    return 0.0


@njit
def mrr(gt_item: Any, pred_items: Union[List[Any], np.ndarray]) -> float:
    """Mean Reciprocal Rank"""
    for i in range(len(pred_items)):
        if gt_item == pred_items[i]:
            return 1.0 / (i + 1)
    return 0


@njit
def normal_dcg(gt_item: Any, pred_items: Union[List[Any], np.ndarray]) -> float:
    """normalized discounted cumulative gain"""
    for i in range(len(pred_items)):
        if gt_item == pred_items[i]:
            return 1.0 / np.log2(i + 2)  # The expected normalizer is 1.0 / log2(0 + 2) which equals 1
    return 0

