"""Base components."""

import abc
from ..utils.data import HotelMess
from typing import Dict, List, Any, Tuple

DataSource = {"base": HotelMess.load_data().set_index("name")}


class Context(object):
    """Define the query context feature info."""
    def __init__(self, **kwargs):
        self.id = kwargs["query_id"]
        self.user = kwargs.get("user_id")
        self.time = kwargs["query_time"]
        self.location = kwargs["query_location"]

    @classmethod
    def from_session(cls, session_id: int = None, user: int = None, time: int = None, location: str = None):
        return cls(query_id=session_id, user_id=user, query_time=time, query_location=location)


class Item(object):
    def __init__(self, key: str, features: Dict[str, Any] = None):
        self.poi = key
        self.features = features if features else {}
        self._feature_names = features.keys() if features else set()

    def __getitem__(self, item):
        return self.features[item]

    @property
    def feature_names(self):
        return self._feature_names

    def drop_features(self, features: List[str]):
        for feature in features:
            if feature in self.features:
                del self.features[feature]
                self._feature_names.remove(feature)

    def add_feature(self, feature_name: str, value: Any):
        self.features[feature_name] = value
        self._feature_names.add(feature_name)


def base_recall_items() -> List[Item]:
    return [Item(key=x) for x in DataSource["base"].index]


class RecallStrategy(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def required_features(self) -> List[str]:
        pass

    @abc.abstractmethod
    def recall(self, item: Item) -> bool:
        raise NotImplementedError


class ScoreStrategy(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def required_features(self) -> List[str]:
        pass

    @abc.abstractmethod
    def score(self, item: Item) -> float:
        raise NotImplementedError


class SortStrategy(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def required_features(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def score_strategy(self) -> ScoreStrategy:
        pass

    def score(self, items: List[Item]) -> List[Tuple[Item, float]]:
        return [(item, self.score_strategy.score(item)) for item in items]

    def sort(self, items: List[Item]):
        items = self.score(items)
        return sorted(items, key=lambda x: -x[1])

