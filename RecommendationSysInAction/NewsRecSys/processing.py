"""The method to load all local news."""

import os
import pandas as pd
import jieba
import time
import itertools
import logging
from tqdm import tqdm
from jieba.analyse import extract_tags
from RecommendationSysInAction.NewsRecSys.NewsRecSys.settings import DB_NAME
from evan_utils.conn import get_mysql_conn
from evan_utils.nlp.tools import stopwords
from typing import Dict


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
tqdm.pandas()
_PATH = os.path.join(os.path.dirname(__file__), "data")
COLUMNS = ["news_id", "cate_id", "dt", "view_num", "comment_num", "title", "content"]
conn = get_mysql_conn("local", db=DB_NAME)


class DataProcessing(object):
    _keywords_type_choices = ["tf-idf", "title-cut"]
    cate_mapping: Dict[int, str] = {1: "为你推荐", 2: "热度榜"}

    def __init__(self, path: str, keywords_type: str = "tf-idf"):
        if not os.path.exists(path):
            raise IOError(f"Unknown path `{path}`.")

        keywords_type = keywords_type.lower()
        if keywords_type not in self._keywords_type_choices:
            raise ValueError(f"Unknown keyword generating type `{keywords_type}`,"
                             f" possible choices are {self._keywords_type_choices}.")

        self.path = path
        self.data = self.read_data_batch(path)
        self.news_keywords = self.generate_keywords(self.data, keywords_type)
        self.news_hotness = self.generate_hotness(self.data)
        self.news_sim = self.generate_similarity(self.news_keywords)

        self.write_mapping = {
            "News": self.data,
            "Cate": pd.DataFrame(self.cate_mapping.items(), columns=["cate_id", "cate_name"]),
            "News_Hotness": self.news_hotness,
            "News_Sim": self.news_sim,
            "News_Tag": self.flatten_keywords_data(self.news_keywords),
        }

    @staticmethod
    def read_data(file: str) -> pd.DataFrame:
        df = pd.read_excel(file, header=0, names=COLUMNS)
        # df["dt"] = pd.to_datetime(df["dt"])
        return df

    def read_data_batch(self, dirname: str) -> pd.DataFrame:
        """Read multi excel data in `dirname`, and log the category mapping."""
        LOGGER.info(f"Reading data from source {dirname}.")
        cate_key_bias = max(self.cate_mapping.keys())  # indicate the adding bias for newly inserted key number.

        data_list = []
        for file in os.listdir(dirname):
            if not (file.endswith("xlsx") or file.endswith("xls")):
                continue
            data = self.read_data(os.path.join(dirname, file))

            # Identify the cate_id and cate_name
            cate_name = data["cate_id"].unique()
            if len(cate_name) > 1:
                raise ValueError(f"The category in file {file} is not unique.")
            cate_name = cate_name[0]
            cate_id = int(file.split("-")[0]) + cate_key_bias
            self.cate_mapping[cate_id] = cate_name

            data["cate_id"] = cate_id  # replace category with its id.
            data_list.append(data)

        return pd.concat(data_list, axis=0, ignore_index=True)

    @staticmethod
    def generate_keywords(data: pd.DataFrame, generate_type: str) -> pd.DataFrame:
        """Generate keywords from title or content
        :param data: Original data.
        :param generate_type: If `tf-idf`, extract keywords from content by `jieba.analysis`.
                              If `title-cut`, extract keywords from title by `jieba.cut`.
        :return: A DataFrame with two columns ["news_id", "keywords"].
        """
        LOGGER.info("Generating the keywords of news.")

        news_keywords_data = data[["news_id", "title", "content"]].copy()
        if generate_type == "tf-idf":
            news_keywords_data["keywords"] = news_keywords_data["content"].progress_apply(
                    lambda content: extract_tags(content, topK=10, allowPOS=('ns', 'n', 'vn', 'v')))
        elif generate_type == "title-cut":
            stop_words = stopwords()
            news_keywords_data["keywords"] = news_keywords_data["title"].progress_apply(
                lambda title: [word for word in jieba.cut(title) if not (word in stop_words or word.strip() == "")])
        else:
            raise ValueError(f"Unknown keyword generating type `{generate_type}`.")

        return news_keywords_data[["news_id", "keywords"]]

    @staticmethod
    def flatten_keywords_data(keywords_data: pd.DataFrame) -> pd.DataFrame:
        data = keywords_data.copy()
        data["keywords"] = data["keywords"].apply(list)
        data = data.explode("keywords").rename(columns={"keywords": "tag"})
        return data

    @staticmethod
    def generate_hotness(data: pd.DataFrame) -> pd.DataFrame:
        """Generate hotness for each news"""
        LOGGER.info("Generating the hotness for news.")
        hotness_data = data.copy()
        now_time = pd.to_datetime(time.time() - time.timezone, unit="s")
        hotness_data["diff_days"] = (now_time - pd.to_datetime(hotness_data["dt"])).apply(lambda x: x.days - 292)
        hotness_data["hotness"] = hotness_data.progress_apply(
            lambda row: row["view_num"] * 0.4 + row["comment_num"] * 0.5 - row["diff_days"] * 0.1, axis=1)
        return hotness_data[["news_id", "cate_id", "hotness"]]

    @staticmethod
    def generate_similarity(keywords_data: pd.DataFrame) -> pd.DataFrame:
        """Generate similarity between each pair of news based on jaccard index of tags"""
        LOGGER.info("Generating the similarity between each pair of news.")
        data = keywords_data.copy()
        data["keywords"] = data["keywords"].apply(set)
        news_keywords = data.set_index("news_id")["keywords"].to_dict().items()

        pair_data = []
        for (left_news, left_keywords), (right_news, right_keywords)\
                in tqdm(itertools.combinations(news_keywords, 2),
                        desc="Generate sims", total=len(data) * (len(data) - 1) / 2):
            intersection = left_keywords & right_keywords
            if not intersection:
                continue
            sim = len(intersection) / (len(left_keywords) + len(right_keywords) - len(intersection))
            pair_data.append((left_news, right_news, sim))
            pair_data.append((right_news, left_news, sim))
        return pd.DataFrame(pair_data, columns=["news_id_left", "news_id_right", "sim"])

    @staticmethod
    def write_to_mysql(conn, data: pd.DataFrame, table: str):
        data.to_sql(table, conn, if_exists="append", index=False, chunksize=5000)

    @staticmethod
    def write_to_fs(data: pd.DataFrame, file: str):
        data.to_csv(file, sep="\t", index=False)

    def write(self):
        for model_name, data in self.write_mapping.items():
            LOGGER.info(f"Saving `{model_name}` data to disk.")
            self.write_to_fs(data, os.path.join(self.path, f"tomysql/{model_name}.txt"))

        # First insert `Cate` because other table has foreign key with it.
        model_name = "Cate"
        data = self.write_mapping[model_name]
        LOGGER.info(f"Saving `{model_name}` data to mysql.")
        self.write_to_mysql(conn, data, table=model_name.lower())

        for model_name, data in self.write_mapping.items():
            if model_name == "Cate":
                continue
            LOGGER.info(f"Saving `{model_name}` data to mysql.")
            self.write_to_mysql(conn, data, table=model_name.lower())


if __name__ == '__main__':
    obj = DataProcessing(_PATH)
    obj.write()
