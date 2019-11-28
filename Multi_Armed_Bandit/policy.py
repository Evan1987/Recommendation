
import abc
import math
import numpy as np
from random import Random
from .bandit import Bandit
from typing import Dict, Iterable, Callable, Any


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, bandits: Iterable[Bandit]) -> float:
        pass


class _Entry(object):
    def __init__(self, n: int, expect_score: float):
        self.n = n
        self.expect_score = expect_score

    def update(self, score: float):
        self.n += 1
        self.expect_score += ((score - self.expect_score) / self.n)

    @classmethod
    def empty(cls):
        return cls(0, 0.0)


class EGreedy(Policy):

    def __init__(self, epsilon: float, seed: int):
        self._epsilon = epsilon
        self.seed = seed
        self._rng = Random(seed)
        self.logs: Dict[Bandit, _Entry] = {}

    @property
    def epsilon(self):
        return self._epsilon

    def choice(self, bandits: Iterable[Bandit]):
        if not self.logs or self._rng.random() < self.epsilon:
            return self._rng.choice(bandits)
        return max(self.logs, key=lambda k: self.logs[k].expect_score)

    def step(self, bandits: Iterable[Bandit]) -> float:
        bandit = self.choice(bandits)
        _, score = bandit.play()                                          # get the score reward
        self.logs.setdefault(bandit, _Entry.empty()).update(score)     # update the logs
        return score


class DecreasingEGreedy(EGreedy):
    def __init__(self, seed: int):
        super(DecreasingEGreedy, self).__init__(0., seed)
        self.seen = set()

        # add a decorator to avoid overriding
        self.choice = self.track_choice(super(DecreasingEGreedy, self).choice)

    def track_choice(self, func: Callable):
        def wrap(*args, **kwargs):
            x = func(*args, **kwargs)
            self.seen.add(x)
            return x
        return wrap

    @property
    def epsilon(self):
        return 1 / math.log(len(self.seen) + 1e-3)


class UCB1(Policy):
    def __init__(self):
        self.logs: Dict[Bandit, _Entry] = {}
        self.n = 0

    def _upper_bound(self, x: _Entry):
        return x.expect_score + math.sqrt(2 * math.log(self.n) / x.n)

    def choice(self):
        return max(self.logs, key=lambda k: self._upper_bound(self.logs[k]))

    def step(self, bandits: Iterable[Bandit]) -> float:
        # Initialize
        for bandit in bandits:
            if bandit not in self.logs:
                _, score = bandit.play()
                self.n += 1
                self.logs[bandit] = _Entry(1, score)

        bandit = self.choice()
        _, score = bandit.play()
        self.logs[bandit].update(score)
        self.n += 1
        return score


class LinUCB(Policy):
    def __init__(self, alpha: float, rp: float, rf: float, d: int):
        """
        Initialize a Lin-UCB policy which is a contextual bandit algorithm
        :param alpha: The hyper-parameter
        :param rp: The positive reward score
        :param rf: The negative reward score
        :param d: The length of contextual features
        """
        self.alpha = alpha
        self.rp = rp
        self.rf = rf
        self.d = d
        self.Aa: Dict[Any, np.ndarray] = {}  # collection of matrix(d * d) to compute disjoint part for each item
        self.AaI: Dict[Any, np.ndarray] = {}  # store the inverse of Aa matrix
        self.ba: Dict[Any, np.ndarray] = {}  # collection of vectors(d * 1) to compute disjoint part
        self.theta = {}
        self.a_max = 0
        self.x = None
        self.xT = None

    def initialize(self, bandits: Iterable[Bandit]):
        for bandit in bandits:
            self.Aa[bandit] = np.identity(self.d, dtype=np.float32)
            self.ba[bandit] = np.zeros((self.d, 1), dtype=np.float32)
            self.AaI[bandit] = np.identity(self.d, dtype=np.float32)
            self.theta[bandit] = np.zeros((self.d, 1), dtype=np.float32)

    def update(self, reward: float):
        pass

    def step(self, bandits: Iterable[Bandit]) -> float:
        pass


class ThompsonSampling(Policy):

    class Distribution(object):
        def __init__(self, alpha: int, beta: int):
            self.alpha = alpha
            self.beta = beta
            self.pos_gain = 0
            self.neg_gain = 0

        def rand(self):
            return np.random.beta(self.alpha + self.pos_gain, self.beta + self.neg_gain)

        def add_pos(self, gain: int):
            self.pos_gain += gain

        def add_neg(self, gain: int):
            self.neg_gain += gain

    def __init__(self, alpha: int, beta: int):
        """
        Initialize a ThompsonSampling policy
        :param alpha: The prior parameter for Beta-distribution, the pos part
        :param beta: The prior parameter for Beta-distribution, the neg part
        """
        self.alpha = alpha
        self.beta = beta
        self.logs: Dict[Bandit, ThompsonSampling.Distribution] = {}

    def choice(self):
        return max(self.logs, key=lambda k: self.logs[k].rand())

    def step(self, bandits: Iterable[Bandit]) -> float:
        # Initialize
        for bandit in bandits:
            if bandit not in self.logs:
                self.logs[bandit] = ThompsonSampling.Distribution(self.alpha, self.beta)

        bandit = self.choice()
        is_win, score = bandit.play()
        if is_win:
            self.logs[bandit].add_pos(1)
        else:
            self.logs[bandit].add_neg(1)
        return score



