"""案例：利用 Naive Bayes 进行异常用户检测"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Tuple, List

feature_names = ["register_days", "active_days", "order_num", "click_num"]
train_data = np.array(
    [[320,  204,  198,  265],
     [253,  53,   15,   2243],
     [53,   32,   5,    325],
     [63,   50,   42,   98],
     [1302, 523,  202,  5430],
     [32,   22,   5,    143],
     [105,  85,   70,   322],
     [872,  730,  840,  2762],
     [16,   15,   13,   52],
     [92,   70,   21,   693]]
)

train_labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])


class NaiveBayesian:

    def __init__(self, alpha: float):
        """
        Initialize NB classifier.
        :param alpha: Smooth parameter for computing P(y_k) and P(x_i|y_k)(when `x_i` is discrete)
        """
        self.alpha = alpha
        self.n_features = None
        self.labels = None
        self.label_prob: Dict[Any, float] = {}  # to hold P(y_k)

        # to hold x_i's mean and std for each y_k
        # when fitted, the value will be like: {label_0: [(mu_0, std_0), (mu_1, std_1), ...], label_1: [...]}
        self.label_feature_summary: Dict[Any, List[Tuple]] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError("The num between features and labels doesn't match.")
        if np.ndim(features) != 2:
            raise ValueError("The features expect to be a 2-D array.")
        if np.ndim(labels) != 1:
            raise ValueError("The labels expect to be a 1-D array.")

        n_samples, self.n_features = features.shape
        self.labels, counts = np.unique(labels, return_counts=True)
        n_labels = len(self.labels)

        # Computing prior P(y_k)
        for label, count in zip(self.labels, counts):
            self.label_prob[label] = (count + self.alpha) / (n_samples + n_labels * self.alpha)

        # Computing each feature distribution e.g. mean and std for `y_k`
        for label in self.labels:
            data = features[labels == label, :]
            for i in range(self.n_features):
                mu, std = np.mean(data[:, i]), np.std(data[:, i])
                self.label_feature_summary.setdefault(label, []).append((mu, std))

    def _prob(self, feature: np.ndarray, label: Any) -> float:
        """Cal joint prob for single sample and given label"""
        label_p = self.label_prob[label]  # P(y_k)
        label_feature_summary = self.label_feature_summary[label]

        log_prob = np.log(label_p)
        for x, (mu, std) in zip(feature, label_feature_summary):
            log_prob += norm.logpdf(x, loc=mu, scale=std)
        return log_prob

    def predict(self, data: np.ndarray):
        if np.ndim(data) != 2:
            raise ValueError("The features expect to be a 2-D array.")
        if data.shape[1] != self.n_features:
            raise ValueError(f"The shape expects to be (?, {self.n_features}), but got ({len(data)}, {data.shape[1]})")

        result: List[List] = []
        for feature in data:
            temp: List[Tuple[Any, float]] = []
            for label in self.labels:
                joint_prob = self._prob(feature, label)
                temp.append((label, joint_prob))
            result.append(temp)
        result = [sorted(scores, key=lambda k: k[1], reverse=True) for scores in result]

        labels = [scores[0][0] for scores in result]
        return labels, result


if __name__ == '__main__':
    test_data = np.array([[134, 84, 235, 349]])
    clf = NaiveBayesian(alpha=1.0)
    clf.fit(train_data, train_labels)
    labels, results = clf.predict(test_data)
    print("Labels: ", labels)
    print("Scores: ", results)
