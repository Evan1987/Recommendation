"""
Model: Factorization Machine
Dataset: MovieLens  # user -> [1, 943], item -> [1, 1682]
Framework: Tensorflow
"""

import os
import numpy as np
import pandas as pd
from constant import PROJECT_HOME


data_path = os.path.join(PROJECT_HOME, "FM_FFM/data")
columns = ["user", "item", "rating", "ts"]


class FMDataLoader(object):
    def __init__(self):
        self.max_user = None
        self.n_features = None

    def fit(self, X: pd.DataFrame):
        self.max_user = X["user"].max()
        self.n_features = self.max_user + X["item"].max()

    def transform(self, X: pd.DataFrame):






if __name__ == '__main__':
    train = pd.read_csv(os.path.join(data_path, "ua.base"), sep="\t", names=columns)
    test = pd.read_csv(os.path.join(data_path, "ua.test"), sep="\t", names=columns)
    max_user, max_item = train["user"].max(), train["item"].max()


