
import numpy as np
import pandas as pd
from random import Random
from tensorflow.keras.utils import Sequence
from .data.data_prepare import DATASET_FILE, CATE_MAPPING_FILE, PACKAGE_HOME
from typing import Dict


class InputKeys:
    USER = "user"
    ITEM = "item"
    CATE = "cate"
    HISTORY_ITEM = "history"
    HISTORY_CATE = "history_cate"
    HISTORY_LENGTH = "history_length"
    LABEL = "label"


PACKAGE_HOME = PACKAGE_HOME
dataset = pd.read_pickle(DATASET_FILE)
cate_mapping = pd.read_csv(CATE_MAPPING_FILE, sep="\t").set_index("asin")["categories"].to_dict()


class DataGenerator(Sequence):
    def __init__(self, entries: pd.DataFrame, item_cate_mapping: Dict[int, int], batch_size: int, seed: int):
        self.entries = entries
        self.entries["history_length"] = self.entries["histories"].apply(len)
        self.item_cate_mapping = item_cate_mapping
        self.batch_size = batch_size
        self.seed = seed
        self._rng = Random(self.seed)
        self._map_func = np.vectorize(self.item_cate_mapping.get)
        self._indexes = np.arange(len(self.entries))

    def __len__(self):
        return len(self.entries) // self.batch_size

    @staticmethod
    def pad(x: np.ndarray):
        max_len = max(len(row) for row in x)
        res = np.zeros((len(x), max_len))
        for i, row in enumerate(x):
            res[i, :len(row)] = row
        return res

    def __getitem__(self, index: int):
        batch_indexes = self._indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.entries.iloc[batch_indexes]
        items = batch_data["asin"].values
        item_cate = batch_data["asin"].map(self.item_cate_mapping).values
        history_items = self.pad(batch_data["histories"].values).astype(int)
        res = {
            InputKeys.USER: batch_data["reviewerID"].values,
            InputKeys.ITEM: items,
            InputKeys.CATE: item_cate,
            InputKeys.HISTORY_ITEM: history_items,
            InputKeys.HISTORY_CATE: self._map_func(history_items),
            InputKeys.HISTORY_LENGTH: batch_data["history_length"].values,
            InputKeys.LABEL: batch_data["label"].values.reshape(-1, 1),
        }
        return res

    @property
    def user_count(self):
        return self.entries["reviewerID"].nunique()

    @property
    def item_count(self):
        return len(self.item_cate_mapping)

    @property
    def cate_count(self):
        return len(set(self.item_cate_mapping.values()))

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
