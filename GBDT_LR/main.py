
import os
import numpy as np
import lightgbm as lgb
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from GBDT_LR.data_utils import load_data, LABEL_COL, SEED, FEATURES
from constant import PROJECT_HOME


package_home = os.path.join(PROJECT_HOME, "GBDT_LR")
MODEL_DIR = os.path.join(package_home, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_LEAVES = 64
NUM_BOOSTERS = 100
lgb_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": {"binary_logloss"},
    "num_leaves": NUM_LEAVES,
    "num_trees": NUM_BOOSTERS,
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 1
}

lr = LogisticRegression(penalty="l2", C=0.05, verbose=1)


def transform(leaf_indexes: np.ndarray, num_leaves: int) -> sp.csr_matrix:
    """One-hot each column of leaf-indexes"""
    n, m = leaf_indexes.shape
    shape = (n, m * num_leaves)
    data = np.ones(shape=(n * m,), dtype=np.float32)
    rows = np.repeat(np.arange(n), m)
    cols = (np.arange(m) * num_leaves + leaf_indexes).reshape(-1)
    return sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)


class CombinedModel(object):
    def __init__(self, lgb_model: lgb.Booster, lr_model: LogisticRegression):
        self.lgb_model = lgb_model
        self.lr_model = lr_model
        self.num_leaves = lgb_model.params["num_leaves"]

    def predict_proba(self, X):
        leaf_indexes = self.lgb_model.predict(X, pred_leaf=True)
        leaf_indexes = transform(leaf_indexes, num_leaves=self.num_leaves)
        return self.lr_model.predict_proba(leaf_indexes)


if __name__ == '__main__':
    train_data, test_data = load_data()
    X_train, X_val, y_train, y_val =\
        train_test_split(train_data[FEATURES], train_data[LABEL_COL], test_size=0.2, random_state=SEED)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    gbm = lgb.train(params=lgb_params, train_set=lgb_train, num_boost_round=NUM_BOOSTERS, valid_sets=lgb_val)
    gbm.save_model(os.path.join(MODEL_DIR, "model.txt"))

    y_leaf_indexes = gbm.predict(X_train, pred_leaf=True)    # [len(X_train), NUM_BOOSTERS]
    sparse_leaf_indexes = transform(y_leaf_indexes, NUM_LEAVES)
    lr.fit(sparse_leaf_indexes, y_train)

    model = CombinedModel(gbm, lr)
    y_score_train = model.predict_proba(X_train)
    y_score_val = model.predict_proba(X_val)

    y_pred_train = np.argmax(y_score_train, axis=1)
    y_pred_val = np.argmax(y_score_val, axis=1)

    print(f"Train metrics Acc: {accuracy_score(y_train, y_pred_train)}, Auc: {roc_auc_score(y_train, y_score_train[:, 1])}")
    print(f"Eval metrics Acc: {accuracy_score(y_val, y_pred_val)}, Auc: {roc_auc_score(y_val, y_score_val[:, 1])}")
