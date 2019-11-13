"""
读取并处理原始数据，增加点击历史特征
"""
import numpy as np
import pandas as pd
from DKN.model import BATCH
from typing import Dict


class TreatingData(object):
    def __init__(self, train_file: str, test_file: str, max_click_history: int=30):
        """
        处理原始数据集，并为数据增加点击历史特征
        :param train_file:
        :param test_file:
        :param max_click_history: 从用户点击历史采样数量
        """
        self.train, self.test = [self.read(file) for file in [train_file, test_file]]
        self.max_click_history = max_click_history
        self.uid2words: Dict[int, np.ndarray] = {}  # uid -> words_matrix: [max_click_history, max_title_len]
        self.uid2entities: Dict[int, np.ndarray] = {}  # uid -> entities_matrix: [max_click_history, max_title_len]

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
        self.train.query("label == 1").groupby("user_id").apply(self._agg_click_info_by_user)  # 从训练数据中抽样抽取用户历史点击信息

        # 为训练集和测试集都加上用户的历史点击信息特征: clicked_words, clicked_entities
        self.train["clicked_words"] = self.train["user_id"].map(self.uid2words)
        self.train["clicked_entities"] = self.train["user_id"].map(self.uid2entities)

        self.test["clicked_words"] = self.test["user_id"].map(self.uid2words)
        self.test["clicked_entities"] = self.test["user_id"].map(self.uid2entities)


def transform_to_batch(batch_df: pd.DataFrame):
    """针对某一分片生成 model.BATCH类型"""
    clicked_words = np.array(batch_df["clicked_words"].tolist())  # [batch_size, max_click_history, max_title_length]
    clicked_entities = np.array(batch_df["clicked_entities"].tolist())  # [batch_size, max_click_history, max_title_length]
    news_words = np.array(batch_df["news_words"].tolist())  # [batch_size, max_title_len]
    news_entities = np.array(batch_df["news_entities"].tolist())  # [batch_size, max_title_len]
    labels = batch_df["label"].values  # [batch_size,]
    return BATCH(clicked_words, clicked_entities, news_words, news_entities, labels)


def get_batch(df: pd.DataFrame, batch_size: int):
    """根据输入数据生成训练用的 batch"""
    size = len(df)
    for i in range(0, size, batch_size):
        yield transform_to_batch(df[i: i + batch_size])



