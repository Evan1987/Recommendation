"""A Item-based CF Recommendation on MovieLens data"""

import os
import random
import json
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from evan_utils.ucollections import PriorityQueue
from evan_utils.context import timer
from typing import Dict, Set, Tuple


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
MOVIE_LENS_SRC = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\ml-1m"


class ItemCFRecommend(object):
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
    similarity_file = os.path.join(tmp_path, "item_similarity.json")

    def __init__(self, rating_data: pd.DataFrame):
        self.train_data, self.test_data = self.split(rating_data, seed=0, test_size=0.1)
        self.item_users, self.user_rated_summary, self.train_data = self.generate_data(self.train_data)
        self.item_sims: Dict[str, Dict[str, float]] = self.cal_item_similarity()  # User rated items summary

    @classmethod
    def from_file(cls, data_file: str):
        if not os.path.exists(data_file):
            raise IOError(f"File not found `{data_file}`.")
        LOGGER.info(f"Reading file from {data_file}...")
        data = pd.read_csv(
            data_file, sep="::", names=["user", "item", "score", "timestamp"], header=None, engine="python")

        data["user"] = data["user"].astype(str)
        data["item"] = data["item"].astype(str)
        return cls(data[["user", "item", "score"]])

    @staticmethod
    def split(data: pd.DataFrame, seed: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Group by user to split entries."""
        LOGGER.info(f"Split data into train and test. Test size `{test_size}`")
        rng = random.Random(seed)
        is_train = data.groupby("user")["item"].transform(lambda s: [rng.random() > test_size for _ in range(len(s))])
        return data[is_train], data[~is_train]

    @staticmethod
    def generate_data(data: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
        """Generate different type of data from given original rating data.
        :returns: 1) The total rated user for each item.
                  2) The total set of items for each user.
                  3) The dict-transformed data from original DataFrame.
        """
        LOGGER.info("Extract data information...")
        # Summary of item, the num of users who rated on it.
        item_users: Dict[str, Set[str]] = data.groupby("item").agg({"user": lambda s: set(list(s))})["user"].to_dict()

        # Summary of all rated item set for each user
        user_rated_summary: Dict[str, Set[str]] = data.groupby("user")\
            .agg({"item": lambda s: set(list(s))})["item"].to_dict()

        # Transform data from pd.DataFrame to Dict of Dict
        def generate_record(d: pd.DataFrame):
            d["record"] = d.apply(lambda row: dict(zip(row["item"], row["score"])), axis=1)
            return d[["record"]]

        manifest_data = data.groupby("user").agg({"item": tuple, "score": tuple}).pipe(generate_record)
        manifest_data = manifest_data["record"].to_dict()
        return item_users, user_rated_summary, manifest_data

    @staticmethod
    def _cal_item_similarity(a: Set[str], b: Set[str]) -> float:
        """Cosine similarity between two set, each element indicates a dimension"""
        return len(a & b) / ((len(a) * len(b)) ** 0.5)

    # 3698 items
    # To cal similarity will cost 1 minute, and 330MB disk space.
    def cal_item_similarity(self) -> Dict[str, Dict[str, float]]:
        if os.path.exists(self.similarity_file):
            LOGGER.info("Loading similarity from disk...")
            with open(self.similarity_file, "r") as f:
                similarity = json.load(f)
            return similarity

        LOGGER.info("Generate new similarity files...")
        similarity: Dict[str, Dict[str, float]] = defaultdict(dict)
        features = list(self.item_users.items())
        n = len(features)
        for i in tqdm(range(n)):
            u, u_sets = features[i]
            for j in range(i + 1, n):
                v, v_sets = features[j]
                similarity[u][v] = similarity[v][u] = self._cal_item_similarity(u_sets, v_sets)

        os.makedirs(self.tmp_path, exist_ok=True)
        with open(self.similarity_file, "w") as f:
            json.dump(similarity, f)
        return similarity

    def recommend(self, user: str, k: int = 8, n_items: int = 40):
        """
        Recommend for target user, only consider Top`k` neighbour items for each rated item and output `n_items` items.
        """

        entries = self.train_data[user]
        recall_items = {}

        for rated_item, score in entries.items():
            neighbours = sorted(self.item_sims[rated_item].items(), key=lambda pair: pair[1], reverse=True)[:k]
            for item_, sim in neighbours:
                if item_ in entries:
                    continue
                recall_items.setdefault(item_, 0)
                recall_items[item_] += score * sim

        r = PriorityQueue(maxsize=n_items)
        for item, score in recall_items.items():
            r.push(score, item)

        return r.items()

    def evaluation(self, k: int = 8, n_items: int = 10) -> Tuple[float, float]:
        """Compute precision and recall"""
        test_user_rated_items: Dict[str, Set[str]] = \
            self.test_data.groupby("user").agg({"item": lambda s: set(list(s))})["item"].to_dict()

        hit = 0
        test_num = 0
        pred_num = 0
        for test_user, test_items in tqdm(test_user_rated_items.items()):
            pred_items = self.recommend(test_user, k=k, n_items=n_items)
            test_num += len(test_items)
            pred_num += len(pred_items)
            for _, pred_item in pred_items:
                if pred_item in test_items:
                    hit += 1

        recall_ = hit / test_num
        precision_ = hit / pred_num
        return precision_, recall_


if __name__ == '__main__':
    rec = ItemCFRecommend.from_file(os.path.join(MOVIE_LENS_SRC, "ratings.dat"))
    with timer("Recommend"):  # 0.1356s
        result = rec.recommend("1", k=8, n_items=40)
    print(result)

    precision, recall = rec.evaluation(k=8, n_items=10)
    print(f"Precision: {precision}, Recall: {recall}.")  # Precision: 0.188542, Recall: 0.11229.
