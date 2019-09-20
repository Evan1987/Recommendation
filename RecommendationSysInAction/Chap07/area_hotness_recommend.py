""" 7.4 A simple recommendation base on area and hotness."""

from RecommendationSysInAction.Chap07.base import Context, SortStrategy, RecallStrategy, ScoreStrategy, Item, base_recall_items
from RecommendationSysInAction.Chap07.features import features, get_feature
from _utils.context import timer
from typing import Dict, List


COMBINE_WEIGHTS = {
    "comment_num": 2., "score": 1., "decoration_time_duration": 0.5, "open_time_duration": 0.5, "lowest_price": 1.5}


class AreaRecallStrategy(RecallStrategy):
    """Recall based on same area."""
    def __init__(self):
        self.area = None

    @property
    def required_features(self):
        return ["location"]

    def set_area(self, area: str):
        self.area = area

    def recall(self, item: Item) -> bool:
        return not self.area or item["location"] == self.area


class CombineScoreStrategy(ScoreStrategy):
    def __init__(self, weights: Dict[str, float]):
        for feature_name in weights.keys():
            if feature_name not in features:
                raise ValueError(f"feature `{feature_name}` not defined.")
        self.weights = weights

    @property
    def required_features(self) -> List[str]:
        return list(self.weights.keys())

    def score(self, item: Item) -> float:
        return sum([item[feature_name] * weight for feature_name, weight in self.weights.items()])


class CombineSortStrategy(SortStrategy):
    """Sort based on multi features."""
    def __init__(self, scorer: ScoreStrategy):
        self._scorer = scorer

    @property
    def required_features(self) -> List[str]:
        return self._scorer.required_features

    @property
    def score_strategy(self) -> ScoreStrategy:
        return self._scorer


def assembly_features(items: List[Item], required_features: List[str], ctx: Context, **kwargs) -> None:
    for feature_name in required_features:
        for item in items:
            if feature_name not in item.feature_names:
                item.add_feature(feature_name, get_feature(feature_name, item.poi, ctx=ctx, **kwargs))


class AreaHotnessRecommend(object):
    _base_recall_items = base_recall_items()

    def __init__(self, recall_strategy: AreaRecallStrategy, sort_strategy: CombineSortStrategy):
        self.recall_strategy = recall_strategy
        self.sort_strategy = sort_strategy
        self.items = self._base_recall_items

    def recommend(self, ctx: Context):
        self.recall_strategy.set_area(ctx.location)
        # recall stage
        assembly_features(self.items, self.recall_strategy.required_features, ctx)
        self.items = [item for item in self.items if self.recall_strategy.recall(item)]

        # sort stage
        assembly_features(self.items, self.sort_strategy.required_features, ctx)
        items = self.sort_strategy.sort(self.items)
        return [(item.poi, score) for item, score in items]


if __name__ == '__main__':
    area_recall_strategy = AreaRecallStrategy()
    combine_score_strategy = CombineScoreStrategy(COMBINE_WEIGHTS)
    combine_sort_strategy = CombineSortStrategy(combine_score_strategy)
    rec = AreaHotnessRecommend(recall_strategy=area_recall_strategy, sort_strategy=combine_sort_strategy)

    with timer("Recommend"):
        context = Context.from_session(session_id=100001, user=1, time=2018, location="丰台区")
        print(rec.recommend(context))


