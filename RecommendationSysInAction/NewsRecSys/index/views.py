
from django.http import JsonResponse
from django.http import request as _request
from django.views.decorators.csrf import csrf_exempt
from ..news.models import News, Cate, NewsHotness, NewsTag, NewsClick, NewsSim
from ..NewsRecSys.settings import ALLOW_USERS, ALLOW_TAGS


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
    _cate = request.GET.get("cate")
    if "username" not in request.session.keys():
        return JsonResponse({"code": 0})

    # total pages
    total = 0

