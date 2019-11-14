
"""
Some toolkit for transforming raw data
"""


import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from random import Random
from tensorflow.keras.utils import Sequence
from constant import PROJECT_HOME
from typing import Tuple, Dict, Any


package_home = os.path.join(PROJECT_HOME, "FM_FFM")
data_path = os.path.join(package_home, "data")
columns = ["user", "item", "rating", "ts"]
FEATURES = ["user", "item"]
LABEL = "rating"
DATA_ALIAS = {
        "train": pd.read_csv(os.path.join(data_path, "ua.base"), sep="\t", names=columns),
        "test": pd.read_csv(os.path.join(data_path, "ua.test"), sep="\t", names=columns)
}

TEST_Y_TRUE = DATA_ALIAS["test"][LABEL].values
MAX_USER = DATA_ALIAS["train"]["user"].max()
MAX_ITEM = DATA_ALIAS["train"]["item"].max()
N = MAX_USER + MAX_ITEM + 1


class InputKeys:
    USER = "user"
    ITEM = "item"
    LABEL = "label"
    BIAS = "bias_const"


# 944 & 1683
FEATURE_MAX_NUM = {InputKeys.USER: MAX_USER + 1, InputKeys.ITEM: MAX_ITEM + 1}


class FMDataTransformer(object):
    def __init__(self):
        self.max_user = None
        self.max_item = None
        self.n = None

    def fit(self, data: pd.DataFrame):
        self.max_user = data["user"].max()
        self.max_item = data["item"].max()
        self.n = self.max_user + self.max_item + 1
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[sp.csr_matrix, np.ndarray]:
        y = data[LABEL].values
        mat = sp.dok_matrix((len(data), self.n))
        for i, (user, item) in enumerate(zip(data["user"].values, data["item"].values)):
            if user > self.max_user or item > self.max_item:
                continue
            mat[i, user] = 1
            mat[i, item + self.max_user] = 1
        X = mat.tocsr()
        return X, y

    def fit_transform(self, data: pd.DataFrame) -> Tuple[sp.csr_matrix, np.ndarray]:
        if self.n is None:
            return self.fit(data).transform(data)
        return self.transform(data)


class FFMDataTransformer(object):
    def __init__(self, feature_field_mapping: Dict[Any, int] = None):
        if feature_field_mapping is None:
            feature_field_mapping = {}
        self.feature_field_mapping = feature_field_mapping
        self.max_user = None
        self.max_item = None
        self.n = None

    def fit(self, data: pd.DataFrame):
        self.max_user = data["user"].max()
        self.max_item = data["item"].max()
        self.n = self.max_user + self.max_item + 1

        last_index = max(self.feature_field_mapping.values()) + 1 if self.feature_field_mapping else 0
        for col in FEATURES:
            if col not in self.feature_field_mapping:
                self.feature_field_mapping[col] = last_index
                last_index += 1
        return self

    def transform_row(self, row: pd.Series) -> str:
        label = row[LABEL]
        ffm_row = [str(label)]
        for col in FEATURES:
            field_index = self.feature_field_mapping[col]
            if col == "user":
                col_index = row[col]
            elif col == "item":
                col_index = row[col] + self.max_user
            else:
                continue
            feature = f"{field_index}:{col_index}:1"
            ffm_row.append(feature)
        return " ".join(ffm_row)

    def transform(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series({idx: self.transform_row(row) for idx, row in data.iterrows()})

    def fit_transform(self, data: pd.DataFrame) -> pd.Series:
        if self.n is None:
            return self.fit(data).transform(data)
        return self.transform(data)


class TFFMDataSet(Sequence):
    def __init__(self, data: pd.DataFrame, batch_size: int, seed: int = 0):
        self.entries = data[FEATURES + [LABEL]].values
        self.batch_size = batch_size
        self._max_buckets = len(self.entries) // self.batch_size
        self._indexes = np.arange(self._max_buckets)
        self._rng = Random(seed)

    def __len__(self):
        return self._max_buckets

    def __getitem__(self, index) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        batch_index = self._indexes[index]
        batch_data = self.entries[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        users = batch_data[:, 0].astype(np.int32)
        items = batch_data[:, 1].astype(np.int32)
        inputs = {
            InputKeys.USER: users,
            InputKeys.ITEM: items,
            InputKeys.BIAS: np.ones(shape=[len(users), 1], dtype=np.float32),
        }
        labels = batch_data[:, -1].astype(np.float32)
        return inputs, labels

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
