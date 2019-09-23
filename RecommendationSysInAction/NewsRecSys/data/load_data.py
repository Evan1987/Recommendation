"""The method to load all local news."""

import os
import pandas as pd

_PATH = os.path.join(os.path.dirname(__file__), "news")
COLUMNS = ["id", "cate_id", "dt", "view_num", "comment_num", "title", "content"]


def load_data(file: str):
    pass



