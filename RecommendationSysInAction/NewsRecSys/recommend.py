"""The module to give recommend at different conditions."""

import abc
from collections import namedtuple
from .news.models import News, NewsSim, NewsHotness, NewsTag, NewsClick
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
            news = News.objects.get(news_id=recall_news.news_id)
            result.append(RecallEntry(news=news, score=score))
        return result


class HotNewsRecommend(BaseRecommend):
    """Recommend based on hotness of news."""
    def __init__(self, start: int, end: int):
        """Recommend news based on hotness and given slice.
        :param start: The start of slice.
        :param end: The end of slice.
        """
        if end <= start:
            raise ValueError("`end` should be greater than `start`.")
        self.start = start
        self.end = end

    @property
    def k(self):
        return self.end - self.start

    def recommend(self) -> List[RecallEntry]:
        hot_newses = NewsHotness.objects.order_by("-hotness").values("news_id", "hotness")[self.start: self.end]
        ids = [one.news_id for one in hot_newses]
        scores = [one.hotness for one in hot_newses]
        newses = News.objects.filter(news_id__in=ids)
        return [RecallEntry(news=news, score=score) for news, score in zip(newses, scores)]


class UserBasedNewsRecommend(BaseRecommend):
    def __init__(self, k: int = 20):
        self.is_new = False
        self._k = k

    @property
    def k(self):
        return self._k

    def recommend(self, username: str) -> List[RecallEntry]:
        # If the user is new, recommend 20:40 of hot list
        if not NewsClick.objects.filter(user=username).exists():
            hot_news_rec = HotNewsRecommend(20, 40)
            return hot_news_rec.recommend()

        latest_news = NewsClick.objects.filter(user=username).order_by("-click_dt")[:self._k // 2]

        # find sim news for each clicked news
        per_rec_num = self._k // len(latest_news) + 1 if len(latest_news) < (self.k // 2) else 2
        sim_rec = SimRecommend(k=per_rec_num)

        res = []
        for one in latest_news:
            res.extend(sim_rec.recommend(one))
        return res[: self._k]
