"""Content-based recommendation, Compare the sim between item's profile and user's profile."""

import os
import json
import pandas as pd
import logging
from scipy.spatial.distance import cosine
from _utils.collections import PriorityQueue
from _utils.context import timer
from typing import Dict, List, Set


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
MOVIE_LENS_SRC = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\ml-1m"


class ContentBasedRec(object):
    def __init__(self, item_profile: Dict[int, List[int]], user_profile: Dict[int, List[int]], k: int):
        self.item_profile = item_profile
        self.user_profile = user_profile
        self.k = k
        LOGGER.info("Read base rating data and analysis.")
        self.ratings = pd.read_csv(os.path.join(MOVIE_LENS_SRC, "ratings.dat"), sep="::", engine="python",
                                   names=["UserID", "MovieID", "Rating", "Timestamp"])
        self.total_items = set(self.item_profile.keys())
        self.user_non_rating_items: Dict[int, Set[int]] = self.get_non_rating_items(self.ratings, self.total_items)

    @classmethod
    def from_json_file(cls, src: str, k: int):
        if not os.path.exists(src):
            raise IOError(f"`{src}` not exists.")

        item_profile_file = os.path.join(src, "item_profile.json")
        user_profile_file = os.path.join(src, "user_profile.json")
        with open(item_profile_file, "r") as f1, open(user_profile_file, "r") as f2:
            item_profile = json.load(f1)
            user_profile = json.load(f2)

        item_profile = {int(key): v for key, v in item_profile.items()}
        user_profile = {int(key): v for key, v in user_profile.items()}
        return cls(item_profile, user_profile, k)

    @staticmethod
    def get_non_rating_items(rating: pd.DataFrame, total: Set[int]) -> Dict[int, Set[int]]:
        """Get not rated items for each user."""
        rated_summary = rating.groupby("UserID").agg({"MovieID": lambda s: set(s)})
        rated_summary = dict(rated_summary["MovieID"])
        return {user: total.difference(rated) for user, rated in rated_summary.items()}

    def recommend(self, user: int) -> List:
        """Recommend item which has not been rated by user and has biggest similarity with user's favor."""
        LOGGER.info(f"Give recommendation for {user}.")
        user_vec = self.user_profile[user]
        result = PriorityQueue(self.k)
        non_rating_items = self.user_non_rating_items[user]
        LOGGER.info(f"Recommend from {len(non_rating_items)} / {len(self.total_items)} non-rated items for `{user}`")
        for movie in non_rating_items:  # Not recommend rated movies
            item_vec = self.item_profile[movie]
            sim = 1 - cosine(item_vec, user_vec)
            result.push(sim, movie)
        return sorted(result.queue(), key=lambda k: k[0], reverse=True)


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    rec = ContentBasedRec.from_json_file(os.path.join(path, "data"), k=10)
    with timer("CBRecommend"):
        print(rec.recommend(1))


