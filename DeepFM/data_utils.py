
import pandas as pd
import numpy as np
from random import Random
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from evan_utils.dataset import load_movielens


SEED = 0
TEST_SIZE = 0.2
tqdm.pandas()
_loader = load_movielens()
_users, _movies, _ratings = _loader.users, _loader.movies, _loader.ratings
_genres = {genre: index + 1 for index, genre in enumerate(_loader.genres)}  # keep `0` as padding


def _make_data():
    """Make target train and test data."""
    rng = Random(SEED)

    _movies["genres"] = _movies["genres"]\
        .progress_apply(lambda x: [_genres[genre] for genre in x.split("|") if genre in _genres])

    _users["occupation"] += 1  # to avoid 0 padding mistake

    data = _ratings.join(_users.set_index("user"), on="user")\
        .join(_movies.set_index("item"), on="item")[["user", "item", "genres", "occupation", "rating"]]

    is_train = data.groupby("user")["item"].transform(lambda x: [rng.random() > TEST_SIZE for _ in range(len(x))])
    return data[is_train], data[~is_train]


_train, _test = _make_data()
DATA_ALIAS = {"train": _train, "test": _test}


class InputKeys:
    USER = "user"
    ITEM = "item"
    GENRES = "genres"
    OCCUPATION = "occupation"

    @classmethod
    def features(cls):
        return ["user", "item", "genres", "occupation"]


LABEL = "rating"
FEATURES = InputKeys.features()
TEST_Y_TRUE = _test[LABEL].values

_FEATURE_INFO = namedtuple("FEATURE_INFO", ["length", "max_id"])  # input_length, max_id
_MAX_GENRE_NUM = 3  # The max num of genres to considered, if less than it, will be padded 0s ta post.

FEATURE_INFOS = {
    InputKeys.USER: _FEATURE_INFO(1, _users["user"].max()),                            # 1, 6040
    InputKeys.ITEM: _FEATURE_INFO(1, _movies["item"].max()),                           # 1, 3952
    InputKeys.GENRES: _FEATURE_INFO(_MAX_GENRE_NUM, max(_genres.values())),            # 3, 18
    InputKeys.OCCUPATION: _FEATURE_INFO(1, _users["occupation"].max()),                # 1, 20
}


def pad_genres(genres: np.ndarray):
    return pad_sequences(genres, maxlen=_MAX_GENRE_NUM, padding="post")


class DataGenerator(Sequence):
    def __init__(self, data: pd.DataFrame, batch_size: int, seed: int = SEED):
        self.entries = data
        self.batch_size = batch_size
        self._indexes = np.arange(len(self.entries))
        self._rng = Random(seed)

    def __len__(self):
        return len(self.entries) // self.batch_size

    def __getitem__(self, index):
        indexes = self._indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data: pd.DataFrame = self.entries.iloc[indexes]
        inputs = {
            InputKeys.USER: batch_data[InputKeys.USER].values.reshape(-1, 1).astype(np.int32),
            InputKeys.ITEM: batch_data[InputKeys.ITEM].values.reshape(-1, 1).astype(np.int32),
            InputKeys.OCCUPATION: batch_data[InputKeys.OCCUPATION].values.reshape(-1, 1).astype(np.int32),
            InputKeys.GENRES: pad_genres(batch_data[InputKeys.GENRES]).astype(np.int32)
        }
        labels = batch_data[LABEL].values.astype(np.float32)
        return inputs, labels

    # @staticmethod
    # def _convert_to_csr(values: np.ndarray, n_cols: int):
    #     row_indexes = []
    #     col_indexes = []
    #     for i, cols in enumerate(values):
    #         row_indexes.extend([i] * len(cols))
    #         col_indexes.extend(cols)
    #     data = np.ones(len(row_indexes))
    #     return sp.csr_matrix((data, (row_indexes, col_indexes)), shape=[len(values), n_cols])

    def on_epoch_end(self):
        self._rng.shuffle(self._indexes)
