
import numpy as np
import pandas as pd
from random import Random
from tensorflow.keras.utils import Sequence


class InputKeys:
    USER = "user"
    ITEM = "item"
    HISTORIES = "histories"
    HISTORY_LENGTH = "history_length"


class DataGenerator(Sequence):
    def __init__(self, entries: pd.DataFrame, batch_size: int, seed: int):
        self.entries = entries
        self.batch_size = batch_size
        self.seed = seed
        self._rng = Random(self.seed)
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
        batch_data = self.entries[batch_indexes]
        res = {
            InputKeys.USER: batch_data["reviewerID"].values.reshape(-1, 1),
            InputKeys.ITEM: batch_data["asin"].values.reshape(-1, 1),
            InputKeys.HISTORIES: self.pad(batch_data["histories"].values).astype(int),
            InputKeys.HISTORY_LENGTH: batch_data["histories"].apply(len).values
        }
        return res

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
