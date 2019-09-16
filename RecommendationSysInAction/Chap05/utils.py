
import time
import random
from typing import Dict, Callable


Data = {
    "A": {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0},
    "B": {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5},
    "C": {"a": 0.0, "b": 3.5, "c": 0.0, "d": 0.0, "e": 3.0},
    "D": {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 3.0},
}


def generate_score_data(n_users: int, n_items: int, sparse_ratio: float, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Generate a random user-item score data. The score is between [0, 5] and integer times of 0.5.
    :param n_users: Num of users.
    :param n_items: Num of items.
    :param sparse_ratio: The ratio of zero in score matrix, between [0, 1]. `0` means there is no zero scored item.
        `1` means the score data is empty.
    :param seed: The random seed.
    :raises: ValueError: When sparse_ratio not between [0, 1].
    :return: A score data like `Data` above.
    """
    if not 0 <= sparse_ratio <= 1:
        raise ValueError(f"The sparse_ratio expects to be between [0, 1], got `{sparse_ratio}`.")

    rng = random.Random(seed)
    return {chr(65 + x): {chr(97 + k): rng.randint(0, 10) * 0.5 if rng.random() > sparse_ratio else 0
                          for k in range(n_items)}
            for x in range(n_users)}


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
