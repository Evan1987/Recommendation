
"""
The base treatment with dataset
"""
import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from _utils.constant import PUBLIC_SOURCE_PATH


SEED = 2017
DATA_PATH = os.path.join(PUBLIC_SOURCE_PATH, "data/Kaggle/Porto Seguro’s Safe Driver Prediction")
_TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
_TEST_FILE = os.path.join(DATA_PATH, "test.csv")
SUB_PATH = os.path.join(DATA_PATH, "output")

CATEGORICAL_COLS = [
    'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    'ps_car_10_cat', 'ps_car_11_cat',
]

NUMERIC_COLS = [
    # binary
    "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
    "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
    "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
    "ps_ind_17_bin", "ps_ind_18_bin",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
    # numeric
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",

    # feature engineering
    "missing_feat", "ps_car_13_x_ps_reg_03",
]

IGNORE_COLS = [
    "id",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]

LABEL_COL = "target"
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def load_data():
    train_data = pd.read_csv(_TRAIN_FILE)
    test_data = pd.read_csv(_TEST_FILE)
    features = train_data.columns.difference(["id", "target"])
    for df in [train_data, test_data]:
        df["missing_feat"] = np.sum(df[features] == -1, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]

    # replace -1 value
    replacing = {}
    for col in CATEGORICAL_COLS:
        miss_replacing = train_data[col].max() + 1
        cond = train_data[col] == -1
        if np.any(cond):
            replacing[col] = miss_replacing
            train_data.loc[cond, col] = miss_replacing
            test_data.loc[test_data[col] == -1, col] = miss_replacing

    return train_data[FEATURES + [LABEL_COL]], test_data[["id"] + FEATURES], replacing


TRAIN, TEST, FEAT_REPLACE = load_data()
FEAT_SIZE = {col: TRAIN[col].max() + 1 for col in CATEGORICAL_COLS}  # Use value as embedding index


class DataGenerator(Sequence):
    def __init__(self, data: pd.DataFrame, batch_size: int):
        self.entries = data
        self.batch_size = batch_size
        self._indexes = np.arange(len(self.entries))
        self._rng = random.Random(SEED)

    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, index):
        indexes = self._indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data: pd.DataFrame = self.entries.iloc[indexes]
        x = {feature: batch_data[feature].values.reshape(-1, 1) for feature in FEATURES}
        y = batch_data[LABEL_COL].values
        return x, y

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
