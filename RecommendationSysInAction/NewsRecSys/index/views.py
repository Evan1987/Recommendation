
from django.http import JsonResponse
from django.http import request as _request
from django.views.decorators.csrf import csrf_exempt
from ..news.models import News, Cate
from ..NewsRecSys.settings import ALLOW_USERS, ALLOW_TAGS
from ..recommend import UserBasedNewsRecommend, HotNewsRecommend


@csrf_exempt
def login(request: _request.HttpRequest):
    if request.method == "GET":
        return JsonResponse({"users": ALLOW_USERS, "tags": ALLOW_TAGS})
    if request.method == "POST":
        username = request.POST.get("username")
        request.session["username"] = username
        tags = request.POST.get("tags")
        return JsonResponse({"username": username, "tags": tags, "baseclick": 0, "code": 1})


def home(request: _request.HttpRequest):
    """Main page response"""
    _cate = int(request.GET.get("cate"))
    if "username" not in request.session.keys():
        return JsonResponse({"code": 0})

    username = request.session["username"]

    if _cate == 1:  # 为你推荐
        rec_news = UserBasedNewsRecommend(20).recommend(username)
        total = 0
    elif _cate == 2:  # 热门榜
        rec_news = HotNewsRecommend(0, 20).recommend()
        total = 0
    else:  # 正常翻页
        _page_id = int(request.GET.get("pageid"))
        news = News.objects.filter(cate_id=_cate).order_by("-dt")
        rec_news = news[_page_id * 10: (_page_id + 1) * 10]
        total = len(news)

    result = {
        "code": 2,
        "total": total,
        "cate_id": _cate,
        "cate_name": Cate.objects.get(cate_id=_cate),
        "rec_news": rec_news}
    return JsonResponse(result)


def switch_user(request: _request.HttpRequest):
    if "username" in request.session.keys():
        username = request.session["username"]
        del request.session["username"]
        print(f"User: {username} switched. Del its value in session.")
    return JsonResponse({"code": 1})

