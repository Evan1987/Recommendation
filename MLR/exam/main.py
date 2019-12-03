
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import Random
from collections import namedtuple
from MLR.model_utils import MLR
from MLR.data_utils import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import List
from constant import PROJECT_HOME


PACKAGE_HOME = os.path.join(PROJECT_HOME, "MLR/exam")
Point = namedtuple("Point", ["x", "y", "label"])


def generate_diamond_data(n: int, seed: int = 0) -> List[Point]:
    rng = Random(seed)
    points = []
    for i in range(n):
        x = rng.random()
        y = rng.random()
        if (x + y) <= 0.5 or (x + y) >= 1.5 or (x - y) >= 0.5 or (y - x) >= 0.5:
            label = 0
        else:
            label = 1
        points.append(Point(x, y, label))
    return points


def points_plot(x: np.ndarray, y: np.ndarray, colors: np.ndarray, pic_save_file: str):
    fig, axe = plt.subplots(figsize=(10, 10))
    axe.scatter(x=x, y=y, c=colors, marker="o")
    fig.savefig(pic_save_file, dpi=50)
    fig.clear()


if __name__ == '__main__':
    samples = generate_diamond_data(5000, 0)
    samples = pd.DataFrame(samples)
    points_plot(samples["x"].values, samples["y"].values, samples["label"].values,
                os.path.join(PACKAGE_HOME, "diamonds.png"))
    train, test = train_test_split(samples, test_size=0.2)

    lr = LogisticRegression()
    lr.fit(train[["x", "y"]].values, train["label"].values.reshape(-1, 1))
    y_pred_lr = lr.predict(test[["x", "y"]].values).reshape(-1,)
    points_plot(test["x"].values, test["y"].values, y_pred_lr, os.path.join(PACKAGE_HOME, "lr.png"))

    mlr = MLR(d=2, m=4, learning_rate=0.02)
    train_seq = DataGenerator(train, batch_size=64)
    mlr.train(train_seq, epochs=1000)
    y_pred_mlr = mlr.predict(test[["x", "y"]].values)
    y_pred_mlr = (y_pred_mlr >= 0.5).astype(int).reshape(-1, )

    points_plot(test["x"].values, test["y"].values, y_pred_mlr, os.path.join(PACKAGE_HOME, "mlr.png"))
