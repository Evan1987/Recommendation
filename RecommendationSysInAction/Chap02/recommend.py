"""推荐程序"""

import os
import json
from RecommendationSysInAction.Chap02.utils import pearson
from _utils.collections import PriorityQueue
from typing import Dict, Any


class FirsrRec(object):
    def __init__(self, features: Dict[str, Dict[str, int]], k: int):
        self.features = features
        self.k = k

    def recommend(self, user_id: str):
        neighbour_users = PriorityQueue(maxsize=self.k)


