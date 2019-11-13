"""Evaluation on Rec"""

import os
import json
import tqdm
from RecommendationSysInAction.Chap02.recommend import FirstRec
from _utils.context import timer
from typing import Dict


data_path = os.path.join(os.path.dirname(__file__), "data")
train_file = os.path.join(data_path, "train.json")
test_file = os.path.join(data_path, "test.json")
k = 15
n = 20


class Evaluation(object):
    def __init__(self, eval_results: Dict[str, Dict[str, int]]):
        self.eval_results = eval_results

    @classmethod
    def from_json_file(cls, file: str):
        with open(file, "r") as fp:
            eval_results = json.load(fp)
        return cls(eval_results)

    def evaluate(self, recommendation: FirstRec):
        """
        Evaluate on recommendation object.
        :param recommendation: The recommend object, can return items for query user.
        :return: recall, precision
        """
        print("Start evaluation.")
        recalls, precisions = [], []
        for user, eval_entries in tqdm.tqdm(self.eval_results.items()):
            hit = 0
            rec_results = recommendation.recommend(user)
            if not rec_results:
                print(f"No recommendation for {user}")
                continue  # possibly because `user` not in `train_file`
            for movie, _ in rec_results:
                if movie in eval_entries:
                    hit += 1
            recalls.append(hit / len(eval_entries))
            precisions.append(hit / len(rec_results))
        return sum(recalls) / len(recalls), sum(precisions) / len(precisions)


if __name__ == '__main__':
    rec = FirstRec.from_json_file(train_file, k=k, n=n)
    evaluation = Evaluation.from_json_file(test_file)
    with timer(name="Evaluation on `FirstRec`"):
        recall, precision = evaluation.evaluate(rec)
    print(f"Recall: {recall}, Precision: {precision}")

