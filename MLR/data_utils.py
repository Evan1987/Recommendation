
import numpy as np
import pandas as pd
from random import Random
from evan_utils.dataset import load_adult
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence


def _generate_data():
    _data = load_adult()
    categorical_columns = _data.categorical_values.keys()

    # For simple usage, just drop the null
    train = _data.train_data.dropna(axis=0)
    test = _data.test_data.dropna(axis=0)

    # Concat data and one-hot
    train["is_train"] = 1
    test["is_train"] = 0
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = pd.get_dummies(all_data, columns=categorical_columns)

    train = all_data.query("is_train == 1").drop("is_train", axis=1)
    test = all_data.query("is_train == 0").drop("is_train", axis=1)

    # Normalize the values
    ss = StandardScaler()
    for col in _data.continuous_columns:
        train[col] = ss.fit_transform(train[col].values.reshape(-1, 1))
        test[col] = ss.transform(test[col].values.reshape(-1, 1))

    return train, test


train_data, test_data = _generate_data()


class DataGenerator(Sequence):
    def __init__(self, entries: pd.DataFrame, batch_size: int, seed: int = None):
        self.entries = entries
        self.batch_size = batch_size
        self.seed = seed
        self._rng = Random(self.seed) if seed is not None else None
        self._indexes = np.arange(len(self.entries))

    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, index: int):
        indexes = self._indexes[self.batch_size * index: self.batch_size * (index + 1)]
        batch_data = self.entries.iloc[indexes]
        x = batch_data.drop("label", axis=1).values.astype(np.float32)
        y = batch_data["label"].values.astype(np.int32).reshape(-1, 1)
        return x, y

    def on_epoch_end(self):
        if self._rng is not None:
            self._rng.shuffle(self._indexes)
