
import numpy as np
import pandas as pd
from random import Random
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from _utils.dataset import load_movielens

tqdm.pandas()
SEED = 2017
TEST_SIZE = 0.2
_NEGATIVE_SAMPLING_NUM = 4   # The num to generate negative samples for per positive entries
DATASET = load_movielens()


def load_data():
    rng = Random(SEED)
    ratings = DATASET.ratings[["user", "item"]]
    all_items = set(ratings["item"])
    user_positive = ratings.groupby("user").agg({"item": lambda s: set(s)})["item"].to_dict()

    def add_negative_samples(df: pd.DataFrame):
        user = df["user"].iloc[0]
        negative_total = all_items - user_positive[user]
        df["neg_items"] = [rng.sample(negative_total, _NEGATIVE_SAMPLING_NUM) for _ in range(len(df))]
        return df

    ratings = ratings.groupby("user", as_index=False).progress_apply(add_negative_samples)
    is_train = ratings.groupby("user")["item"]\
        .transform(lambda x: [rng.random() > TEST_SIZE for _ in range(len(x))])
    return ratings[is_train], ratings[~is_train]


class DataGenerator(Sequence):
    def __init__(self, entries: pd.DataFrame, batch_size: int, seed: int):
        self.entries = entries
        self.batch_size = batch_size
        self.seed = seed
        self._rng = Random(self.seed)
        self._indexes = np.arange(len(entries))

    def __getitem__(self, item: int):
        batch_data =
