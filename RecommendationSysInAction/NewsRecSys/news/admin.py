from django.contrib import admin
from .models import Cate, News, NewsHotness, NewsSim, NewsTag, NewsClick


@admin.register(Cate)
class AdminCate(admin.ModelAdmin):
    list_display = ("cate_id", "cate_name",)
    search_fields = ("cate_id", "cate_name",)
    list_filter = ("cate_name",)


@admin.register(News)
class AdminNews(admin.ModelAdmin):
    list_display = ("news_id", "cate_id", "dt", "view_num", "comment_num", "title",)
    search_fields = ("title", "dt", "cate_id",)
    list_filter = ("dt", "cate_id",)
    ordering = ("-dt",)


@admin.register(NewsHotness)
class AdminNewsHotness(admin.ModelAdmin):
    list_display = ("news_id", "cate_id", "hotness",)
    search_fields = ("news_id", "cate_id", "hotness",)


@admin.register(NewsTag)
class AdminNewsTag(admin.ModelAdmin):
    list_display = ("news_id", "tag",)
    search_fields = ("news_id", "tag",)


@admin.register(NewsSim)
class AdminNewsSim(admin.ModelAdmin):
    list_display = ("news_id_left", "news_id_right", "sim",)
    search_fields = ("news_id_left", "news_id_right", "sim",)


@admin.register(NewsClick)
class AdminNewsClick(admin.ModelAdmin):
    list_display = ("news_id", "user", "click_dt",)
    search_fields = ("news_id", "user", "click_dt",)
    list_filter = ("user",)



