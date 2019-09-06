""" Construct a simple User-based CF Recommendation"""
import time
from _utils.context import timer
from typing import Dict, Set, List, Callable


Data = {
    "A": {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0},
    "B": {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5},
    "C": {"a": 0.0, "b": 3.5, "c": 0.0, "d": 0.0, "e": 3.0},
    "D": {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 3.0},
}


def profiler(name: str, repeats: int):
    def wrapper(func: Callable):
        def inner_wrapper(*args, **kwargs):
            res = None
            tic = time.time()
            for _ in range(repeats):
                res = func(*args, **kwargs)
            toc = time.time()
            print(f"{name} cost: {(toc - tic) / repeats} sec")
            return res
        return inner_wrapper
    return wrapper


class UserCF(object):
    def __init__(self, user_scores: Dict[str, Dict[str, float]]):
        # Delete zero-scored items
        self.user_scores = user_scores
        self.user_non_score_items: Dict[str, List[str]] = self.get_user_non_score_items(user_scores)
        self.item_users: Dict[str, Set[str]] = self.get_item_users(user_scores)
        self.users_sim = self.cal_user_similarity()

    @staticmethod
    def get_user_non_score_items(user_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        return {user: [item for item, rating in item_ratings.items() if rating <= 0]
                for user, item_ratings in user_scores.items()}

    @staticmethod
    def get_item_users(user_scores: Dict[str, Dict[str, float]]) -> Dict[str, Set[str]]:
        pass

    @staticmethod
    def _cal_user_similarity(a: Set[str], b: Set[str]) -> float:
        return len(a & b) / ((len(a) * len(b)) ** 0.5)

    @profiler("user similarity", 10)
    def cal_user_similarity(self) -> Dict[str, Dict[str, float]]:
        """
        Cal similarity by cosine, but vector's value is not the rating but the whether rated,e.g. 1 or 0.
        For example:
        a = {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0} => vec_a = [1, 1, 0, 1, 0]
        b = {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5} => vec_b = [1, 0, 1, 0, 1]
        sim(a, b) = (a * b) / |a| * |b| = 1 / sqrt(6)
        #todo time-complexity O(#user * #user.
        #todo usually, most users' similarity is zero since there's no intersection between them.
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

    def cal_user_similarity2(self) -> Dict[str, Dict[str, float]]:
        pass

    def user_item_score(self, user: str, item: str) -> float:
        """
        Calculate the recommend score between specified user and item.
        For example, the score computation between user `C` and item `a` is as follows:
        W{Ca} = Sim{CA} * Score{Aa} + Sim{CB} * Score{Ba} + Sim{CD} * Score{Da}
              = Sim{CA} * 3 + Sim{CB} * 4 + Sim{CD} * 0
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
    ub = UserCF(Data)
    with timer(name="User-based CF"):
        print(ub.recommend("C"))
