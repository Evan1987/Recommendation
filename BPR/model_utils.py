
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from typing import Union, Optional, Set, Dict


class BayesianPersonalizedRanking(object):

    def __init__(self, learning_rate: float = 0.01, k: int = 15, reg: float = 0.01, random_state: int = 0):
        self.learning_rate = learning_rate
        self.k = k
        self.reg = reg
        self.random_state = random_state
        self._rng = np.random.RandomState(self.random_state)
        self.w_users: Optional[np.ndarray] = None
        self.w_items: Optional[np.ndarray] = None

    def fit(self, train_data: Union[pd.DataFrame, np.ndarray], batch_size: int = 1000, epochs: int = 10):

        if isinstance(train_data, pd.DataFrame):
            train_data: np.ndarray = train_data.values

        if train_data.shape[1] < 3:
            raise ValueError("The input data is expected to be of 3 features(user, item_i, item_j)")
        elif train_data.shape[1] > 3:
            warnings.warn("The given input data has > 3 features, will use the first 3 as (user, item_i, item_j)")
            train_data = train_data[:, :3]

        # To make sure user/item's id matches same corresponding row index
        n_users = 1 + np.max(train_data[:, 0])
        n_items = 1 + np.max(train_data[:, 1:])

        # initialize parameters
        self.w_users = self._rng.normal(size=(n_users, self.k))
        self.w_items = self._rng.normal(size=(n_items, self.k))

        for _ in tqdm(range(epochs), desc="Training"):
            self._rng.shuffle(train_data)
            for i in range(len(train_data) // batch_size):
                batch_data = train_data[i * batch_size: (i + 1) * batch_size]
                users, pos_items, neg_items = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
                self._update(users, pos_items, neg_items)

    def _update(self, u: np.ndarray, i: np.ndarray, j: np.ndarray):
        """Update weights by given users, item_i(pos_items), item_j(neg_items)"""
        user_u = self.w_users[u, :]                         # [batch, k]
        item_i = self.w_items[i, :]                         # [batch, k]
        item_j = self.w_items[j, :]                         # [batch, k]

        r_uij = np.sum(user_u * (item_i - item_j), axis=1, keepdims=True)   # [batch, 1]
        sigmoid = expit(-r_uij)                              # [batch, 1]

        grad_u = -sigmoid * (item_i - item_j) + self.reg * user_u            # [batch, k]
        grad_i = -sigmoid * user_u + self.reg * item_i                       # [batch, k]
        grad_j = -sigmoid * (-user_u) + self.reg * item_j                    # [batch, k]

        for u_, i_, j_, grad_u_, grad_i_, grad_j_ in zip(u, i, j, grad_u, grad_i, grad_j):
            self.w_users[u_] -= self.learning_rate * grad_u_
            self.w_items[i_] -= self.learning_rate * grad_i_
            self.w_items[j_] -= self.learning_rate * grad_j_

    def _predict_user(self, user: int):
        return self.w_users[user, :].dot(self.w_items.T)

    def recommend(self, user: int, topN: int, recall_items: Set[int]):
        """Recommend for specified user with top-N items in given recalled items"""
        scores = self._predict_user(user).reshape(-1)
        ids = np.argsort(-scores)
        result = []
        for rec in ids:
            if rec in recall_items:
                result.append(rec)
                if len(result) == topN:
                    return result
        return result

    def evaluate(self, test_data: pd.DataFrame, neg_collections: Dict[int, np.ndarray]):
        g = test_data.groupby("user")
        acc_collection = []
        auc_collection = []
        for user, df in tqdm(g, total=len(g), desc="Testing"):
            pos_items = df["pos_item"].values
            neg_items = df["neg_item"].values
            total_neg_items = neg_collections[user]

            total_scores = self._predict_user(user)
            pos_scores = total_scores[pos_items]
            neg_scores = total_scores[neg_items]
            total_neg_scores = total_scores[total_neg_items]

            acc = np.sum(pos_scores > neg_scores) / len(df)
            auc = roc_auc_score([1] * len(pos_items) + [0] * len(total_neg_items),
                                np.r_[pos_scores, total_neg_scores])
            acc_collection.append(acc)
            auc_collection.append(auc)
        return acc_collection, auc_collection
