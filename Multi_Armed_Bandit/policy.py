
import abc
import math
import numpy as np
import warnings
import scipy.stats as ss
from random import Random
from .bandit import Bandit
from typing import Dict, Iterable, Callable, Any


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, bandits: Iterable[Bandit], **kwargs) -> float:
        raise NotImplementedError


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

    def step(self, bandits: Iterable[Bandit], **kwargs) -> float:
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

    def step(self, bandits: Iterable[Bandit], **kwargs) -> float:
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
    def __init__(self, d: int, alpha: float = None, confidence: float = None):
        """
        Initialize a Lin-UCB policy which is a contextual bandit algorithm
        :param d: The length of contextual features
        :param alpha: The hyper-parameter for setting confidence degree
        :param confidence: The hyper-parameter for setting confidence degree
                at least one of `alpha` and `confidence` should be set.
        """
        if alpha is None and confidence is None:
            raise ValueError("`alpha` or `confidence` should be set.")
        if alpha is None and confidence is not None:
            self.confidence = confidence
            self.alpha = ss.norm.ppf(0.5 + confidence / 2)
        else:
            if alpha is not None and confidence is not None:
                warnings.warn("both `alpha` and `confidence` have been set, only use `alpha` next.", category=Warning)
            self.alpha = alpha
            self.confidence = 2 * ss.norm.cdf(alpha) - 1

        self.d = d
        self.theta: Dict[Bandit, np.ndarray] = {}           # The theta for each bandit, value shape: d * 1
        self.A: Dict[Bandit, np.ndarray] = {}               # The A for each bandit, value shape: d * d
        self.A_inv: Dict[Bandit, np.ndarray] = {}           # The inv of A for each bandit, value shape: d * d
        self.b: Dict[Bandit, np.ndarray] = {}               # The b for each bandit, value shape: d * 1

    def choice_score(self, theta: np.ndarray, A_inv: np.ndarray, feature: np.ndarray):
        expect_score = np.dot(theta, feature)
        delta = np.sqrt(feature.T @ A_inv @ feature)
        return expect_score + self.alpha * delta

    def step(self, bandits: Iterable[Bandit], features: Dict[Bandit, np.ndarray]) -> float:
        for bandit in bandits:
            if bandit not in self.theta:
                self.A[bandit] = np.identity(self.d, dtype=np.float32)
                self.b[bandit] = np.zeros((self.d, 1), dtype=np.float32)
                self.A_inv[bandit] = np.identity(self.d, dtype=np.float32)
                self.theta[bandit] = np.zeros((self.d, 1), dtype=np.float32)

        bandit = max(self.theta,
                     key=lambda k: self.choice_score(self.theta[k], self.A_inv[k], features[k]))
        r = bandit.play()
        self.A[bandit] += features[bandit] @ features[bandit].T
        self.b[bandit] += features[bandit] * r
        self.A_inv[bandit] = np.linalg.inv(self.A[bandit])
        self.theta[bandit] = self.A_inv[bandit] @ self.b[bandit]
        return r


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

    def step(self, bandits: Iterable[Bandit], **kwargs) -> float:
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
