
from random import Random
from typing import List


class Bandit(object):
    def __init__(self, identifier: int, p: float, seed: int, pos_reward: float = 1.0, neg_reward: float = 0.0):
        """Initialize a bandit with parameters, default reward is 1
        :param identifier: The identifier for this bandit, unique
        :param p: The probability to win
        :param seed: The random seed for this bandit
        :param pos_reward: The reward if win
        :param neg_reward: The reward if lose
        """
        if not 0 < p < 1:
            raise ValueError("The `p` must be strictly in (0, 1).")
        if neg_reward >= pos_reward:
            raise ValueError("The neg_reward must be smaller than pos_reward.")

        self.identifier = identifier
        self.p = p
        self.seed = seed
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self._rng = Random(self.seed)

    def play(self):
        """Play the bandit, check the reward"""
        if self._rng.random() <= self.p:
            return True, self.pos_reward
        return False, self.neg_reward

    def __hash__(self):
        return self.identifier

    def __eq__(self, other: "Bandit"):
        return self.identifier == other.identifier


def generate_multi_armed_bandits(num: int, seed: int) -> List[Bandit]:
    """
    A quick method to generate a bunch of bandits.
    For each bandit, the `p` will be randomly set, and obey the uniform distribution in (0, 0.5)
    and the seed will be its index in the list times by 10.
    :param num: The num of bandits to generate
    :param seed: The meta seed to set p for each bandit, not the bandit's seed
    :return: A list of bandits.
    """
    rng = Random(seed)
    return [Bandit(identifier=i, p=rng.uniform(0, 0.5), seed=10 * i) for i in range(num)]
