"""A Latent Semantic Analysis Recommendation on MovieLens data"""

import os
import random
import math
import dill as pickle
import heapq
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from _utils.ucollections import PriorityQueue
from _utils.context import timer
from typing import Dict, Set, Tuple, Optional, List


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
MOVIE_LENS_SRC = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\ml-1m"


def read_file(data_file: str):
    if not os.path.exists(data_file):
        raise IOError(f"File not found `{data_file}`.")
    LOGGER.info(f"Reading file from {data_file}...")
    data = pd.read_csv(data_file, sep="::", names=["user", "item", "score", "timestamp"], header=None, engine="python")

    data["user"] = data["user"].astype(str)
    data["item"] = data["item"].astype(str)
    return data[["user", "item", "score"]]


def split(data: pd.DataFrame, seed: int = 0, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group by user to split entries."""
    LOGGER.info(f"Split data into train and test. Test size `{test_size}`")
    rng = random.Random(seed)
    is_train = data.groupby("user")["item"].transform(lambda s: [rng.random() > test_size for _ in range(len(s))])
    return data[is_train], data[~is_train]


def evaluate(obj, test_data: pd.DataFrame, n_recall_items: int):
    if not hasattr(obj, "recommend"):
        raise TypeError(f"{obj.__class__} don't have `recommend` method.")
    test_user_rated_summary = test_data.groupby("user").agg({"item": lambda s: set(list(s))})["item"].to_dict()
    hit = 0
    test_num = 0
    pred_num = 0
    for user, test_items in tqdm(test_user_rated_summary.items(), desc="Evaluate"):
        pred_items = obj.recommend(user, n_items=n_recall_items)
        test_num += len(test_items)
        pred_num += len(pred_items)
        for _, pred_item in pred_items:
            if pred_item in test_items:
                hit += 1
    precision = hit / pred_num
    recall = hit / test_num
    return precision, recall


class LFMRecommend(object):
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
    model_file = os.path.join(tmp_path, "model.pkl")

    def __init__(self, k: int = 5, neg_ratio: float = 1.0, lr: float = 0.02, lam: float = 0.01, decay: float = 0.9):
        """
        :param k: The num of latent features.
        :param neg_ratio: The ratio for sampling neg samples, #neg_samples / #pos_samples
        :param lr: The learning rate.
        :param lam: The L2 factor.
        :param decay: The learning rate decay factor for each epoch.
        """
        self.k = k
        self.neg_ratio = neg_ratio
        self.lr = lr
        self.lam = lam
        self.decay = decay
        self.user_recall_items: Dict[str, Set[str]] = {}  # For each user, get the set of non-rated items.
        self.P = None  # The latent user matrix
        self.Q = None  # The latent item matrix

    @staticmethod
    def _get_neg_samples_for_user(weighted_neg_items: List[Tuple[str, float]], n_neg: int) -> List[str]:
        """Get neg samples for single user
        :param weighted_neg_items: The List of neg samples with weight.
        :param n_neg: The num of samples to choose.
        :return: neg samples.
        """
        if len(weighted_neg_items) <= n_neg:  # If the total is smaller than requirement, then return entire set.
            neg_items = weighted_neg_items
        else:
            neg_items = heapq.nlargest(n_neg, weighted_neg_items, key=lambda t: t[1])  # Get the n_neg largest neg items
        return [item for item, _ in neg_items]

    @staticmethod
    def _init_model(n_users: int, n_items: int, class_count: int,
                    user_names: Optional[List] = None,
                    item_names: Optional[List] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate initial random matrix for P and Q
        :param n_users: The num of users.
        :param n_items: The num of items.
        :param class_count: The num of latent features.
        :param user_names: The users' names for index of P.
        :param item_names: The items' names for column of Q.
        :return: Tuple of two DataFrames.
                 1) The users' latent DataFrame, with users' names as column and range(class_count) as index.
                 2) The items' latent DataFrame, with items' names as column and range(class_count) as index.
        :raise ValueError: When given user_names but len(user_names) != n_users,
                            or given item_names but len(item_names) != n_items
        """
        if not user_names:
            user_names = np.arange(n_users)
        if not item_names:
            item_names = np.arange(n_items)
        if len(user_names) != n_users:
            raise ValueError("The num of users and user_names should be equal.")
        if len(item_names) != n_items:
            raise ValueError("The num of items and item_names should be equal.")

        features = np.arange(class_count)
        P = np.random.randn(class_count, n_users)
        Q = np.random.randn(class_count, n_items)

        P = pd.DataFrame(data=P, index=features, columns=user_names)
        Q = pd.DataFrame(data=Q, index=features, columns=item_names)
        return P, Q

    def predict(self, user: str, item: str):
        p = self.P[user]
        q = self.Q[item]
        return np.dot(p, q)

    def fit(self, X: pd.DataFrame, epochs: int = 5):
        """Train LFM model
        :param X: The original rating data for training.
        :param epochs: The training epochs for SGD.
        """
        LOGGER.info("Generating users' recall set.")
        # For each user, summary the total set of items ever rated. It will be treated as pos samples.
        user_rated_items: Dict[str, Set[str]] =\
            X.groupby("user").agg({"item": lambda s: set(list(s))})["item"].to_dict()

        # For each item, the times of rating. It will be treated as weights for negative sampling.
        item_weights: Dict[str, float] = X["item"].value_counts(sort=False).to_dict()

        # For each user, generate the non-rated items e.g. the recall set for recommendation.
        self.user_recall_items: Dict[str, Set[str]] =\
            {user: item_weights.keys() - rated_items
             for user, rated_items in tqdm(user_rated_items.items(), desc="Generating recall sets")}

        # For each user, get positive and negative samples for train.
        LOGGER.info("Generating training samples.")
        user_samples: Dict[str, List] = {}
        for user, pos_samples in tqdm(user_rated_items.items(), desc="Generating train samples"):
            weighted_neg_items: List[Tuple[str, float]] = [(item, item_weights[item]) for item in self.user_recall_items[user]]
            n_neg = int(self.neg_ratio * len(pos_samples))
            neg_samples = self._get_neg_samples_for_user(weighted_neg_items, n_neg)
            user_samples[user] = [(item, 1) for item in pos_samples] + [(item, 0) for item in neg_samples]

        self.P, self.Q = self._init_model(n_users=len(user_samples), n_items=len(item_weights),
                                          class_count=self.k, user_names=list(user_samples.keys()),
                                          item_names=list(item_weights.keys()))

        LOGGER.info(f"Start training. Learning-rate: {self.lr}, decay: {self.decay}, lambda: {self.lam}, epochs: {epochs}")
        lr = self.lr
        for epoch in range(epochs):
            LOGGER.info(f"Running epoch: {epoch + 1}, learning-rate: {lr}")
            for user, samples in tqdm(user_samples.items(), desc=f"Loop user on epoch {epoch + 1}"):
                random.shuffle(samples)
                for item, score in samples:
                    err = score - self.predict(user, item)
                    self.P[user] += lr * (err * self.Q[item] - self.lam * self.P[user])
                    self.Q[item] += lr * (err * self.P[user] - self.lam * self.Q[item])
            lr *= self.decay
        self.save()

    def recommend(self, user: str, n_items: int = 10):
        result = PriorityQueue(maxsize=n_items)
        for item in self.user_recall_items[user]:
            score = self.predict(user, item)
            score = 1 / (1 + math.exp(-score))
            result.push(score, item)
        return result.queue()

    def save(self):
        LOGGER.info(f"Save model to `{self.model_file}`")
        with open(self.model_file, "wb") as f:
            pickle.dump((self.P, self.Q), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        LOGGER.info(f"Load model from `{self.model_file}`")
        with open(self.model_file, "rb") as f:
            self.P, self.Q = pickle.load(f)


if __name__ == '__main__':
    data = read_file(os.path.join(MOVIE_LENS_SRC, "ratings.dat"))
    train, test = split(data, seed=0, test_size=0.1)
    lfm = LFMRecommend()
    lfm.fit(train)

    n_recall = 10
    with timer("Recommend"):
        print(lfm.recommend("6027", n_recall))

    precision, recall = evaluate(lfm, test, n_recall_items=n_recall)
    print(f"Precision: {precision}, Recall: {recall}.")  # Precision: 0.188542, Recall: 0.11229.
