"""
读取并处理原始数据，对文本进行 embedding，输出用于推荐训练的可用数据
"""

import re
import os
import gensim
import numpy as np
from collections import defaultdict
from typing import List, Optional
from utils import timer

MAX_TITLE_LENGTH = 10
PATTERN1 = re.compile('[^A-Za-z]')  # 非字母
PATTERN2 = re.compile('[ ]{2,}')  # >=两个空格

class DataPreprocessor(object):
    def __init__(self, input_files: str, min_word_count: int=2, min_entity_count: int=1, word_embedding_dim: int=50):
        self.input_files = input_files
        self.word2freq = defaultdict(int)  # word: count
        self.entity2freq = defaultdict(int)  # entity_id: count
        self.word2index = {}  # word: word_index(int)
        self.entity2index = {}  # entity_id(kg里的id, int): entity_index(索引id, int)
        self.corpus: List[List[str]] = []

        self.min_word_count = min_word_count
        self.min_entity_count = min_entity_count
        self.K = word_embedding_dim

    def file_parsing(self):
        """
        读取输入数据，并作文本解析、词频统计、实体统计。
        并根据阈值去掉低频词与低频实体，并生成词索引与实体索引。
        """
        with timer("File Parsing", verbose=True):
            print("**** Starting Parsing Input Files! ****")
            for file in self.input_files:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            user, title, label, entities = line.split("\t")

                            content: List[str] = title.split()
                            # collect corpus
                            self.corpus.append(content)

                            # collect word in title to summarise count
                            for s in content:
                                self.word2freq[s] += 1

                            # collect entity in entities to summarise count
                            for pair in entities.split(";"):
                                ent_id, ent_name = pair.split(":")
                                self.entity2freq[int(ent_id)] += 1

            # 生成word2index
            index = 1  # 起始从 1 开始， 0 for dummy
            for word, freq in self.word2freq.items():
                if freq >= self.min_word_count:
                    self.word2index[word] = index
                    index += 1

            # 生成entity2index
            index = 1
            for ent_id, freq in self.entity2freq.items():
                if freq >= self.min_entity_count:
                    self.entity2index[int(ent_id)] = index
                    index += 1

            print("Succeed in Parsing Input Files!")
            print("Words num: %d.\tEntity num: %d" % (len(self.word2index), len(self.entity2index)))

    def encoding_title(self, title: str, entities: str) -> (str, str):
        """
        根据当前的 entities，寻找 title与entities的单词关联，并返回 title的 word_encoding序列和 entity_encoding序列
        :param title: 标题
        :param entities: 当前标题对应的 entities信息, id_1:entity_1;id_2:entity_2...
        :return: word_encoding, entity_encoding
        """
        # 对该entity进行拆分，每个单词获得该entity的index_id
        local_mapping = {}  # ent_w1: ent_index_id, ent_w2: ent_index_id
        for pair in entities.split(";"):
            ent_id, ent_name = pair.split(":")
            ent_name = PATTERN1.sub(" ", ent_name)  # 去掉非字母部分
            ent_name = PATTERN2.sub(" ", ent_name).lower()  # 去掉2个及以上的连续空格

            for w in ent_name.split():
                entity_index = self.entity2index[int(ent_id)]
                local_mapping[w] = entity_index

        word_encoding = ["0"] * MAX_TITLE_LENGTH  # 结果数据容器
        entity_encoding = ["0"] * MAX_TITLE_LENGTH  # 结果数据容器
        curr = 0
        for word in title.split():
            if word in self.word2index:
                word_encoding[curr] = str(self.word2index[word])  # 获取该词的word_index_id
                if word in local_mapping:
                    entity_encoding[curr] = str(local_mapping[word])  # 获取该词的entity_index_id
                curr += 1
            if curr == MAX_TITLE_LENGTH:
                break

        return ",".join(word_encoding), ",".join(entity_encoding)

    def transform(self, input_file, output_file):
        """
        对 input_file的标题进行 index处理，生成 word_index_encoding 和 entity_index_encoding
        """
        with timer("Transform", True):
            print("**** Starting Transform %s ****" % input_file)
            with open(input_file, "r", encoding="utf-8") as fr, open(output_file, "w", encoding="utf-8") as fw:
                for line in fr:
                    line = line.strip()
                    if line:
                        user, title, label, entities = line.split("\t")
                        word_encoding, entity_encoding = self.encoding_title(title, entities)  # 从entity中获取特征
                        content = "\t".join([user, word_encoding, entity_encoding, label])
                        fw.write(content + "\n")
            print("Transformation Done!")

    def save_info(self, root_path: Optional[str]=None):
        """
        将 entity2index，word_embedding写入硬盘
        :param root_path: 根目录
        """
        if not root_path:
            root_path = os.getcwd()

        # 写入 entity2index
        with open(os.path.join(root_path, "kg/entity2index.txt"), "w", encoding="utf-8") as f:
            for ent_id, ent_index in self.entity2index.items():
                f.write("%d\t%d\n" % (ent_id, ent_index))

        # 写入wordembedding














