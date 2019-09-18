""" 5.4.2 Construct a simple Item-based CF Recommendation"""

from RecommendationSysInAction.utils.decorator import profiler
from RecommendationSysInAction.Chap05.utils import generate_score_data
from _utils.context import timer
from typing import Dict, Set, List


class ItemCF(object):

    def __init__(self, user_scores: Dict[str, Dict[str, float]]):
        self.user_scores = user_scores
        self.user_non_score_items: Dict[str, List[str]] = self.get_user_non_score_items(user_scores)
        self.items_sim = self.cal_item_similarity()

    @staticmethod
    def get_user_non_score_items(user_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Get the non-rated items of each user, it is also the base recall set for each user."""
        return {user: [item for item, rating in item_ratings.items() if rating <= 0]
                for user, item_ratings in user_scores.items()}

    @staticmethod
    def _cal_user_similarity(a: Set[str], b: Set[str]) -> float:
        """Cosine similarity between two set, each element indicates a dimension"""
        return len(a & b) / ((len(a) * len(b)) ** 0.5)

    @profiler("item similarity", 1)
    def cal_item_similarity(self) -> Dict[str, Dict[str, float]]:
        similarity: Dict[str, Dict[str, float]] = {}
        item_users: Dict[str, Set[str]] = {}  # The inverted table, e.g. for each item, summary the users who rated.

        for user, items in self.user_scores.items():
            for item, score in items.items():
                if score <= 0:
                    continue
                item_users.setdefault(item, set())
                item_users[item].add(user)

        for u, u_items in item_users.items():
            similarity.setdefault(u, {})
            for v, v_items in item_users.items():
                if u == v:
                    continue
                sim = self._cal_user_similarity(u_items, v_items)
                similarity[u][v] = sim

        return similarity

    def user_item_score(self, user: str, item: str) -> float:
        """
        Calculate the recommend score between specified user and item.
        For example, the score computation between user `C` and item `a` is as follows:
            Score(C, a) = sum([Sim(a, x) * Score(C, x) for x in items-rated-by-C])
        """
        score = 0.0
        item_sim = self.items_sim[item]

        for item_, sim in item_sim.items():
            score += sim * self.user_scores[user][item_]
        return score

    def recommend(self, user: str) -> Dict[str, float]:
        """Just give the scores on non-rated items of `user`, not sorted or top-k"""
        return {item: self.user_item_score(user, item) for item in self.user_non_score_items[user]}


if __name__ == '__main__':
    Data = generate_score_data(100, 1000, 0.2, 0)
    # Data = Data

    # user similarity cost: 3.86 sec
    # recommend cost: 0.0429 sec
    ub = ItemCF(Data)
    with timer(name="Item-based CF"):
        print(ub.recommend("C"))
