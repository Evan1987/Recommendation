"""The module to give recommend at different conditions."""

import abc
from collections import namedtuple
from .news.models import News, NewsSim, NewsHotness, NewsTag
from typing import List


RecallEntry = namedtuple("RecallEntry", ["news", "score"])


class BaseRecommend(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def k(self) -> int:
        pass

    @abc.abstractmethod
    def recommend(self, **kwargs) -> List[RecallEntry]:
        pass


class SimRecommend(BaseRecommend):
    """Recommend based on news's sim neighbour."""

    def __init__(self, k: int):
        self._k = k

    @property
    def k(self):
        return self._k

    def recommend(self, news: News) -> List[RecallEntry]:
        recall_newses = NewsSim.objects.filter(news_id_left=news.news_id).order_by("-sim")[:self._k]
        result = []
        for recall_news in recall_newses:
            score = recall_news.sim
            news = News.objects.get(news_id=recall_news.news_id_right)
            result.append(RecallEntry(news=news, score=score))
        return result


class MostSeenRecommend(BaseRecommend):
    def __init__(self, k: int):
        self._k = k

    @property
    def k(self):
        return self._k

    def recommend(self, news: News) -> List[RecallEntry]:
        recall_newses = News.objects.filter(cate=news.cate).order_by("-view_num")[:self._k]
        result = []
        for recall_news in recall_newses:
            score = recall_news.sim
            news = News.objects.get(news_id=recall_news.news_id)[0]
            result.append(RecallEntry(news=news, score=score))
        return result






