"""推荐程序"""

import os
import json
try:
    from .utils import pearson
except ImportError as e:
    print("Importing error for `pearson`", e)
    from RecommendationSysInAction.Chap02.utils import pearson  # possibly be used for directly test
from evan_utils.ucollections import PriorityQueue
from collections import defaultdict
from typing import Dict, Any, List


class FirstRec(object):
    """User-based Collaborate Filtering"""
    def __init__(self, features: Dict[str, Dict[str, int]], k: int, n: int):
        """Initialize recommend instance
        :param features: The known features about users.
        :param k: The num of neighbours to be considered.
        :param n: The num of recommend items for one user.
        """
        self.features = features
        self.k = k
        self.n = n

    @classmethod
    def from_json_file(cls, file: str, k: int, n: int):
        with open(file, "r") as fp:
            features = json.load(fp)
        return cls(features, k, n)

    def recommend(self, target_user: str) -> List:
        """Recommend `self.n` items for `target_user`"""

        if target_user not in self.features:
            return []

        target_entries = self.features[target_user]

        # Select TopK neighbourhood's entries
        neighbour_users = PriorityQueue(maxsize=self.k)
        for user, entries in self.features.items():
            if user == target_user:
                continue
            corr = pearson(entries, target_entries)
            neighbour_users.push(corr, entries)  # different from source code, push entries not users.

        movies = defaultdict(float)
        for corr, entries in neighbour_users.queue():
            for movie, rate in entries.items():
                movies[movie] += corr * rate  # corr as the weight of user

        # sort movies
        result = sorted(movies.items(), key=lambda k: k[1], reverse=True)

        return result[: self.n]


if __name__ == '__main__':
    from evan_utils.context import timer
    json_file = os.path.join(os.path.dirname(__file__), "data/train.json")
    rec = FirstRec.from_json_file(json_file, k=15, n=20)
    with timer(name="Recommend Test"):  # ~30 ms
        print(rec.recommend("436670"))

