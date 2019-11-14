
import pandas as pd
import numpy as np
import scipy.sparse as sp
from random import Random
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from _utils.dataset import load_movielens


SEED = 0
TEST_SIZE = 0.2
tqdm.pandas()
_loader = load_movielens()
_users, _movies, _ratings = _loader.users, _loader.movies, _loader.ratings


def _make_data():
    rng = Random(SEED)
    genres = {genre: index for index, genre in enumerate(_loader.genres)}
    _movies["genres"] = _movies["genres"]\
        .progress_apply(lambda x: [genres[genre] for genre in x.split("|") if genre in genres])

    data = _ratings.join(_users.set_index("user"), on="user")\
        .join(_movies.set_index("item"), on="item")[["user", "item", "genres", "occupation", "rating"]]

    is_train = data.groupby("user")["item"].transform(lambda x: [rng.random() > TEST_SIZE for _ in range(len(x))])
    return data[is_train], data[~is_train]


_train, _test = _make_data()
DATA_ALAS = {"train": _train, "test": _test}
LABEL = "rating"
FEATURES = ["user", "item", "genres", "occupation"]


class InputKeys:
    USER = "user"
    ITEM = "item"
    GENRES = "genres"
    OCCUPATION = "occupation"


FEATURE_MAX = {
    "user": _users["user"].max(),               # 6040
    "item": _movies["item"].max(),              # 3952
    "genres": len(_loader.genres) - 1,          # 17
    "occupation": _users["occupation"].max(),   # 20
}


class DataGenerator(Sequence):
    def __init__(self, data: pd.DataFrame, batch_size: int, seed: int):
        self.entries = data
        self.batch_size = batch_size
        self._indexes = np.arange(len(self.entries))
        self._rng = Random(seed)
        self._max_genre = FEATURE_MAX[InputKeys.GENRES]

    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, index):
        indexes = self._indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data: pd.DataFrame = self.entries.iloc[indexes]
        inputs = {
            InputKeys.USER: batch_data[InputKeys.USER].values.reshape(-1, 1).astype(np.int32),
            InputKeys.ITEM: batch_data[InputKeys.ITEM].values.reshape(-1, 1).astype(np.int32),
            InputKeys.OCCUPATION: batch_data[InputKeys.OCCUPATION].values.astype(np.int32),
            InputKeys.GENRES: self._convert_to_csr(batch_data[InputKeys.GENRES].values, self._max_genre + 1)
        }
        labels = batch_data[LABEL].values.astype(np.float32)
        return inputs, labels

    @staticmethod
    def _convert_to_csr(values: np.ndarray, n_cols: int):
        row_indexes = []
        col_indexes = []
        for i, cols in enumerate(values):
            row_indexes.extend([i] * len(cols))
            col_indexes.extend(cols)
        data = np.ones(len(row_indexes))
        return sp.csr_matrix((data, (row_indexes, col_indexes)), shape=[len(values), n_cols])

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
