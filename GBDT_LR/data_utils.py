
"""
The base treatment with dataset
"""

import os
import pandas as pd
from evan_utils.constant import PUBLIC_SOURCE_PATH


SEED = 2017
DATA_PATH = os.path.join(PUBLIC_SOURCE_PATH, "data/Kaggle/Porto Seguro’s Safe Driver Prediction")
_TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
_TEST_FILE = os.path.join(DATA_PATH, "test.csv")


FEATURES = ["ps_reg_01", "ps_reg_02", "ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]
LABEL_COL = "target"


def load_data():
    cols = ["id"] + FEATURES
    train_data = pd.read_csv(_TRAIN_FILE)[cols + [LABEL_COL]]
    test_data = pd.read_csv(_TEST_FILE)[cols]
    return train_data, test_data

