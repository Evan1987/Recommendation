"""Chap 8.6 based on dataset Telco-Customer-Churn"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, accuracy_score
from RecommendationSysInAction.utils.data import TelecomChurn
from typing import Dict


class GBDT_LR(BaseEstimator, ClassifierMixin):
    def __init__(self, gbdt_params: Dict, lr_params: Dict):
        self.gbdt = GradientBoostingClassifier(**gbdt_params)
        self.lr = LogisticRegression(**lr_params)
        self.enc = OneHotEncoder()

    def fit(self, X, y=None):
        self.gbdt.fit(X, y)
        indices = self.gbdt.apply(X).reshape(-1, self.gbdt.n_estimators)
        self.enc.fit(indices)
        self.lr.fit(self.enc.transform(indices), y)
        return self

    def predict_proba(self, X):
        indices = self.gbdt.apply(X).reshape(-1, self.gbdt.n_estimators)
        return self.lr.predict_proba(self.enc.transform(indices))

    def predict(self, X):
        scores = self.predict_proba(X)
        return self.lr.classes_[np.argmax(scores, axis=1).reshape(-1)]

    def get_params(self, deep=True):
        return {"gbdt": self.gbdt.get_params(), "lr": self.lr.get_params()}


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
gbdt_params = {
    "learning_rate": 0.1, "n_estimators": 100, "max_depth": 7
}

lr_params = {
    "penalty": "l2", "tol": 1e-4, "fit_intercept": True
}

gbdt = GradientBoostingClassifier(**gbdt_params)
lr = LogisticRegression(**lr_params)
MODEL_ALIAS = {
    "gbdt": gbdt,
    "lr": lr,
    "gbdt+lr": GBDT_LR(gbdt_params, lr_params)
}


class Demo(object):
    path = os.path.dirname(__file__)

    feature_mapping = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "MultipleLines": {"Yes": 1, "No": 0, "No phone service": 2},
        "InternetService": {"DSL": 1, "Fiber optic": 2, "No": 0},
        "OnlineSecurity": {"Yes": 1, "No": 0, "No internet service": 2},
        "OnlineBackup": {"Yes": 1, "No": 0, "No internet service": 2},
        "DeviceProtection": {"Yes": 1, "No": 0, "No internet service": 2},
        "TechSupport": {"Yes": 1, "No": 0, "No internet service": 2},
        "StreamingTV": {"Yes": 1, "No": 0, "No internet service": 2},
        "StreamingMovies": {"Yes": 1, "No": 0, "No internet service": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3,
        },
        "Churn": {"Yes": 1, "No": 0},
    }

    def __init__(self, data: pd.DataFrame, model: str = "gbdt+lr"):
        self.data = self.feature_transform(data)
        print(self.data.info(null_counts=True))
        self.train_data, self.test_data = self.split_data(self.data)
        self.feature_names = [col for col in self.data.columns if col not in ["customerID", "Churn"]]
        self.label_name = "Churn"
        self.model = MODEL_ALIAS[model]

    @classmethod
    def feature_transform(cls, data):
        LOGGER.info("Transform features.")
        for col in data.columns:
            data[col] = data[col].apply(lambda x: "0.0" if x == " " or x is None else x)
            if col not in cls.feature_mapping:
                continue
            data[col] = data[col].map(cls.feature_mapping[col])

        file = os.path.join(cls.path, "new_churn.csv")
        LOGGER.info(f"Saving new file to `{file}`")
        data.to_csv(file, index=False)
        return pd.read_csv(file, engine="python")

    @staticmethod
    def split_data(data: pd.DataFrame):
        return train_test_split(data, test_size=0.1, random_state=40)

    def train(self):
        LOGGER.info("Start training.")
        x_train = self.train_data[self.feature_names]
        y_train = self.train_data[self.label_name]
        self.model.fit(x_train, y_train)

    def evaluate(self):
        LOGGER.info("Start evaluating.")
        x_test = self.test_data[self.feature_names]
        y_test = self.test_data[self.label_name]
        y_score = self.model.predict_proba(x_test)[:, 1]
        y_pred = self.model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        LOGGER.info(f"The model's evaluation result: ACC: {acc}, AUC: {auc}.")


if __name__ == '__main__':
    dat = TelecomChurn.load_data()
    demo = Demo(dat, model="gbdt+lr")
    demo.train()

    # gbdt: ACC: 0.7660, AUC: 0.8220.
    # lr: ACC: 0.7816, AUC: 0.8269.
    # gbdt+lr: ACC: 0.7433, AUC: 0.7910.
    demo.evaluate()
