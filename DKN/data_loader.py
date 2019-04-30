
import os
import numpy as np
import pandas as pd
import argparse
from typing import Dict
from DKN.constant import PATH


class TreatingData(object):
    def __init__(self, train_file: str, test_file: str, max_click_history: int=30):
        """

        :param train_file:
        :param test_file:
        :param max_click_history: 从用户点击历史采样数量
        """
        self.train, self.test = [self.read(file) for file in [train_file, test_file]]
        self.max_click_history = max_click_history
        self.uid2words = {}  # uid -> words_matrix [max_click_history, max_title_len]
        self.uid2entities = {}  # # uid -> entities_matrix [max_click_history, max_title_len]

    @staticmethod
    def read(file):
        df = pd.read_table(file, sep="\t", header=None, names=["user_id", "news_words", "news_entities", "label"])
        df["news_words"] = df["news_words"].apply(lambda x: [int(word_index) for word_index in x.split(",")])
        df["news_entities"] = df["news_entities"].apply(lambda x: [int(entity_index) for entity_index in x.split(",")])
        return df

    def _agg_click_info_by_user(self, df_user):
        """
        汇总用户历史点击信息，并抽样记录
        :param df_user: 单一用户的全部历史点击信息
        :return: unit
        """
        uid = df_user["user_id"].iloc[0]
        choices = np.random.choice(range(len(df_user)), size=self.max_click_history, replace=True)  # 有放回的采样
        words = np.array(df_user["news_words"].tolist())[choices]
        entities = np.array(df_user["news_entities"].tolist())[choices]
        self.uid2words[uid] = words
        self.uid2entities[uid] = entities

    def transform(self):
        self.train.groupby("user_id").apply(self._agg_click_info_by_user)  # 从训练数据中抽样抽取用户历史点击信息









if __name__ == '__main__':
    path = os.path.join(PATH, "data")
    train_file = os.path.join(path, "train.txt")
    test_file = os.path.join(path, "test.txt")
