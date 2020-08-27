
import numpy as np
import pandas as pd
from evan_utils.dataset import load_movielens
from typing import Tuple, Dict


DATA = load_movielens("100k")
ALL_MOVIES = DATA.movies["item"].values


def _generate_train_label(total_length: int, test_size: float):
    r = np.ones(total_length)
    test_length = int(total_length * test_size)
    r[test_length:] = 0
    return r


def make_data(pos_threshold: float = 3.0, test_size: float = 0.2, seed: int = 1234) ->\
        Tuple[pd.DataFrame, pd.DataFrame, Dict[int, np.ndarray]]:
    """Generate train/test data based on MovieLens 100K
    :param pos_threshold: The threshold to judge as positive or negative,
                            assuming positive when score bigger than it.
    :param test_size: The test size.
    :param seed: The random seed
    :return: A tuple of useful data.
             1) train_data: the training dataset, with columns: [`user`, `pos_item`, `neg_item`]
             2) test_data: the testing dataset, with columns: [`user`, `pos_item`, `neg_item`]
             3) neg_collections: the total negative samples for each user.
    """
    ratings = DATA.ratings

    data = ratings[ratings["rating"] >= pos_threshold].copy()\
        .sort_values(by=["user", "ts"], ascending=True)\
        .rename(columns={"item": "pos_item"})

    pos_collections: Dict[int, np.ndarray] = data.groupby("user").agg({"pos_item": "unique"})["pos_item"].to_dict()
    neg_collections: Dict[int, np.ndarray] = {user: np.setdiff1d(ALL_MOVIES, pos_items, assume_unique=True)
                                              for user, pos_items in pos_collections.items()}

    rng = np.random.RandomState(seed)

    # Randomly select movies from the non-pos ones for each user
    data["neg_item"] = data.groupby("user", as_index=False)["user"]\
        .transform(lambda s: rng.choice(neg_collections[s.iloc[0]], len(s)))

    # The last `test_size` ratio entries will be treated as test
    data["is_train"] = data.groupby("user")["ts"].transform(lambda s: _generate_train_label(len(s), test_size))
    columns = ["user", "pos_item", "neg_item"]

    train_data = data.query("is_train == 1")[columns].copy()
    test_data = data.query("is_train == 0")[columns].copy()

    return train_data, test_data, neg_collections
