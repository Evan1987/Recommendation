"""A User-based CF Recommendation on MovieLens data"""

import os
import random
import math
import json
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Set, Iterable, Tuple


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
MOVIE_LENS_SRC = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\ml-1m"


class UserCFRecommend(object):
    seed = 0
    test_size = 0.1

    def __init__(self, rating_data: pd.DataFrame):
        self.data = rating_data
        self.user_index_mapping = self.generate_index_mapping(rating_data["user"].unique())
        self.item_index_mapping = self.generate_index_mapping(rating_data["item"].unique())
        self.n_users = len(self.user_index_mapping)
        self.n_items = len(self.item_index_mapping)
        self.rng = random.Random(self.seed)
        self.train_data, self.test_data = self.split(self.data)

    @classmethod
    def from_file(cls, data_file: str):
        if not os.path.exists(data_file):
            raise IOError(f"File not found `{data_file}`.")
        data = pd.read_csv(data_file, sep="::", names=["user", "item", "score", "timestamp"], header=None)
        return cls(data[["user", "item", "score"]])

    @staticmethod
    def generate_index_mapping(seq: Iterable[Any]) -> Dict[Any, int]:
        return {k: i for i, k in enumerate(seq)}

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Group by user to split entries."""
        is_train = data.groupby("user")["item"]\
            .transform(lambda s: [self.rng.random() > self.test_size for _ in range(len(s))])
        return data[is_train], data[~is_train]

    @staticmethod
    def _cal_user_similarity(a: Set[Any], b: Set[Any]) -> float:
        """Cosine similarity between two set, each element indicates a dimension"""
        return len(a & b) / ((len(a) * len(b)) ** 0.5)

    def cal_user_similarity(self):
        similarity = np.zeros(shape=(self.n_users, self.n_users))

        user_summary = self.train_data.groupby("user").agg({"item": lambda s: set(list(s))})
