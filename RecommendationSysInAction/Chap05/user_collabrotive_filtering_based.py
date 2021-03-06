""" 5.4.1 Construct a simple User-based CF Recommendation"""

import math
from RecommendationSysInAction.utils.decorator import profiler
from RecommendationSysInAction.Chap05.utils import generate_score_data
from evan_utils.context import timer
from typing import Dict, Set, List


class UserCF(object):
    """
    The recommend score(uj) between user `u` and item `j` is computed as follows:
        score(uj) = sum{score(xj) * sim(u, x) for x in users besides `u`}
    The sim(u, x) is computed as follows:
        sim(u, x) = Jaccard(Whether rated item vector of u, the one of x)
    """

    def __init__(self, user_scores: Dict[str, Dict[str, float]]):
        # Delete zero-scored items
        self.user_scores = user_scores
        self.user_non_score_items: Dict[str, List[str]] = self.get_user_non_score_items(user_scores)
        self.users_sim = self.cal_user_similarity()

    @staticmethod
    def get_user_non_score_items(user_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Get the non-rated items of each user, it is also the base recall set for each user."""
        return {user: [item for item, rating in item_ratings.items() if rating <= 0]
                for user, item_ratings in user_scores.items()}

    @staticmethod
    def _cal_user_similarity(a: Set[str], b: Set[str]) -> float:
        """Cosine similarity between two set, each element indicates a dimension"""
        return len(a & b) / ((len(a) * len(b)) ** 0.5)

    @profiler("user similarity", 1)
    def cal_user_similarity(self) -> Dict[str, Dict[str, float]]:
        """
        Cal similarity by cosine, but vector's value is not the rating but the whether rated,e.g. 1 or 0.
        For example:
        a = {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0} => vec_a = [1, 1, 0, 1, 0]
        b = {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5} => vec_b = [1, 0, 1, 0, 1]
        sim(a, b) = (a * b) / |a| * |b| = 1 / sqrt(6)
        # todo time-complexity O(#user * #user.
        # todo usually, most users' similarity is zero since there's no intersection between them.
        """
        similarity: Dict[str, Dict[str, float]] = {}
        user_items: Dict[str, Set[str]] = {user: set([item for item, rating in item_ratings.items() if rating > 0])
                                           for user, item_ratings in self.user_scores.items()}
        for u, u_items in user_items.items():
            similarity.setdefault(u, {})
            for v, v_items in user_items.items():
                if u == v:
                    continue
                sim = self._cal_user_similarity(u_items, v_items)
                similarity[u][v] = sim
        return similarity

    @profiler("user similarity better", 1)
    def cal_user_similarity_better(self) -> Dict[str, Dict[str, float]]:
        """Use item-inverted table to construct users' sim matrix without building user's vector at first."""
        similarity: Dict[str, Dict[str, float]] = {}  # the result to return.
        item_users: Dict[str, List[str]] = {}  # The inverted table, e.g. for each item, summary the users who rated.
        N: Dict[str, int] = {}  # Count the num of items rated by each user.
        C: Dict[str, Dict[str, int]] = {}  # Count the num of intersect items between each pair of user.

        # Step1: Generate inverted table
        for user, items in self.user_scores.items():
            for item, score in items.items():
                item_users.setdefault(item, [])
                if score > 0:
                    item_users[item].append(user)

        # Step2: Summary to N and C
        for item, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if v == u:
                        continue
                    C[u][v] += 1

        # Step3: Compute similarity.
        for u, related_users in C.items():  # loop on each user
            similarity.setdefault(u, {})
            for v, cuv in related_users.items():
                if v == u:
                    continue
                similarity[u][v] = cuv / ((N[u] * N[v]) ** 0.5)

        return similarity

    @profiler("user similarity best", 1)
    def cal_user_similarity_best(self) -> Dict[str, Dict[str, float]]:
        """Based on `cal_user_similarity_better`, but add punishment on hot items while computing similarity.
        """
        similarity: Dict[str, Dict[str, float]] = {}  # the result to return.
        item_users: Dict[str, List[str]] = {}  # The inverted table, e.g. for each item, summary the users who rated.
        N: Dict[str, int] = {}  # Count the num of items rated by each user.
        C: Dict[str, Dict[str, int]] = {}  # Count the num of intersect items between each pair of user.

        # Step1: Generate inverted table
        for user, items in self.user_scores.items():
            for item, score in items.items():
                item_users.setdefault(item, [])
                if score > 0:
                    item_users[item].append(user)

        # Step2: Summary to N and C
        for item, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if v == u:
                        continue
                    C[u][v] += 1 / math.log(1 + len(users))  # Hot item punishment, the only difference from `better`.

        # Step3: Compute similarity.
        for u, related_users in C.items():  # loop on each user
            similarity.setdefault(u, {})
            for v, cuv in related_users.items():
                if v == u:
                    continue
                similarity[u][v] = cuv / ((N[u] * N[v]) ** 0.5)

        return similarity

    def user_item_score(self, user: str, item: str) -> float:
        """
        Calculate the recommend score between specified user and item.
        For example, the score computation between user `C` and item `a` is as follows:
        Score(C, a) = sum([Score(U, a) * Sim(U, C) for U in Users-besides-C])
        """
        score = 0.0
        user_sim = self.users_sim[user]
        for user_, sim in user_sim.items():
            score += sim * self.user_scores[user_][item]
        return score

    def recommend(self, user: str) -> Dict[str, float]:
        """Just give the scores on non-rated items of `user`, not sorted or top-k"""
        return {item: self.user_item_score(user, item) for item in self.user_non_score_items[user]}


if __name__ == '__main__':
    Data = generate_score_data(100, 1000, 0.2, 0)

    # user similarity cost: 0.536 sec
    # recommend cost: 0.015 sec
    ub = UserCF(Data)
    with timer(name="User-based CF"):
        print(ub.recommend("C"))
