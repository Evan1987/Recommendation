""" Features' definition """

from .base import DataSource, Context
from typing import Callable, Dict


features: Dict[str, Callable] = {}


def register(name: str):
    def wrap(func):
        features[name] = func
        return func
    return wrap


def get_feature(feature: str, poi: str = None, ctx: Context = None, **kwargs):
    return features[feature](poi, ctx, **kwargs)


@register("comment_num")
def get_comments(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "comment_num"]


@register("location")
def get_location(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "addr"]


@register("score")
def get_score(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "score"]


@register("decoration_time")
def get_decoration_time(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "decoration_time"]


@register("open_time")
def get_open_time(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "open_time"]


@register("decoration_time_duration")
def get_decoration_duration(poi: str = None, ctx: Context = None, **kwargs):
    decoration_time = get_feature("decoration_time", poi, ctx, **kwargs)
    return decoration_time - ctx.time


@register("open_time_duration")
def get_open_time_duration(poi: str = None, ctx: Context = None, **kwargs):
    open_time = get_feature("open_time", poi, ctx, **kwargs)
    return ctx.time - open_time


@register("lowest_price")
def get_lowest_price(poi: str = None, ctx: Context = None, **kwargs):
    return DataSource["base"].at[poi, "lowest_price"]


