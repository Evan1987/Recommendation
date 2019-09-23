import time
from django.http import JsonResponse
from django.http import request as _request
from .models import News, NewsClick
from ..recommend import SimRecommend, MostSeenRecommend


sim_rec = SimRecommend(5)
most_seen_rec = MostSeenRecommend(5)


def one(request: _request.HttpRequest):
    """Get the news"""
    news_id = request.GET.get("news_id")
    if "username" not in request.session.keys():
        return JsonResponse({"code": 0})
    user = request.session["username"]
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Add an entry in click table.
    NewsClick.objects.create(news_id=news_id, user=user, dt=now).save()
    # The main info of query news
    news: News = News.objects.get(news_id=news_id)

    # The recommend news
    # The sim rec based on this click
    recommend_news = sim_rec.recommend(news)
    # If sim rec not succeed, do most-seen rec
    recommend_news = recommend_news if recommend_news else most_seen_rec.recommend(news)

    recommend = [{
        "news_id": rec_news.news_id,
        "title": rec_news.title,
        "dt": rec_news.dt,
        "cate_id": rec_news.cate_id,
        "score": score
    } for rec_news, score in recommend_news]

    result = {
        "code": 2,
        "news_id": news_id,
        "title": news.title,
        "dt": news.dt,
        "content": news.content,
        "view_num": news.view_num,
        "comment_num": news.comment_num,
        "cate": news.cate.cate_name,
        "recommend": recommend
    }

    return JsonResponse(result)







