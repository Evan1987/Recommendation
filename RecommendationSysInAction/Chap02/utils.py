
"""Some utils functions for recommendation."""

from typing import Dict, Any


def pearson(x: Dict[Any, int], y: Dict[Any, int]) -> float:
    sum_xy, sum_x, sum_y, sum_x2, sum_y2, n = 0, 0, 0, 0, 0, 0
    for key, vx in x.items():
        if key not in y:
            continue
        n += 1
        vy = y[key]
        sum_x += vx
        sum_y += vy
        sum_x2 += vx ** 2
        sum_y2 += vy ** 2
        sum_xy += vx * vy
    if n == 0:
        return 0.

    denominator = ((sum_x2 - sum_x ** 2 / n) * (sum_y2 - sum_y ** 2 / n)) ** 0.5
    if denominator == 0:
        return 0
    return (sum_xy - (sum_x * sum_y) / n) / denominator


if __name__ == '__main__':
    import os
    import json
    file = os.path.join(os.path.dirname(__file__), "data/train.json")
    with open(file, "r") as fp:
        train = json.load(fp)
    print(pearson(train["436670"], train["171580"]))
