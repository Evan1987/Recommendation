
import numpy as np
import pandas as pd
from random import Random
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from evan_utils.dataset import load_movielens

tqdm.pandas()
SEED = 2017
_NEGATIVE_SAMPLING_NUM = 4   # The num to generate negative samples for per positive entries
DATASET = load_movielens()


def load_data():
    np.random.seed(SEED)
    ratings = DATASET.ratings.sort_values(by=["user", "ts"])[["user", "item"]]
    all_items = set(ratings["item"])
    user_positive = ratings.groupby("user").agg({"item": lambda s: set(s)})["item"].to_dict()

    is_train = ratings.groupby("user")["item"] \
        .transform(lambda x: [True] * (len(x) - 1) + [False])  # Take the last as test

    train_data, test_data = ratings[is_train], ratings[~is_train]

    def add_negative_samples(df: pd.DataFrame):
        """Add neg samples for target user"""
        user = df["user"].iloc[0]
        negative_total = list(all_items - user_positive[user])
        n_pos = len(df)
        n_neg = n_pos * _NEGATIVE_SAMPLING_NUM
        neg_items = np.random.choice(negative_total, n_neg, replace=True)
        items = np.r_[df["item"], neg_items]
        labels = [1] * n_pos + [0] * n_neg
        return pd.DataFrame({"user": user, "item": items, "label": labels})

    train_data = train_data.groupby("user", as_index=False).progress_apply(add_negative_samples)
    test_data = test_data.groupby("user", as_index=False).progress_apply(add_negative_samples)
    return train_data, test_data


class InputKeys:
    USER = "user"
    ITEM = "item"


LABEL = "label"

FEATURE_MAX_INFOS = {
    InputKeys.USER: DATASET.max_user,
    InputKeys.ITEM: DATASET.max_item,
}


class DataGenerator(Sequence):
    def __init__(self, entries: pd.DataFrame, batch_size: int, seed: int = None, is_train: bool = True):
        self.entries = entries
        self.batch_size = batch_size
        self.seed = seed if seed is not None else SEED
        self.is_train = is_train
        self._rng = Random(self.seed)
        self._indexes = np.arange(len(entries))

    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, index: int):
        indexes = self._indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.entries.iloc[indexes]
        x = {key: batch_data[key].values.reshape(-1, 1) for key in [InputKeys.USER, InputKeys.ITEM]}
        y = batch_data[LABEL].values
        return x, y

    def on_epoch_end(self):
        if self.is_train:
            self._rng.shuffle(self._indexes)

