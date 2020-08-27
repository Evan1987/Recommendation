"""
读取并处理原始数据，对文本进行 embedding，输出用于推荐训练的可用数据
"""

import re
import os
import gensim
import numpy as np
from collections import defaultdict
from typing import List, Optional
from evan_utils.context import timer
from DKN.constant import PATH

MAX_TITLE_LENGTH = 10
PATTERN1 = re.compile('[^A-Za-z]')  # 非字母
PATTERN2 = re.compile('[ ]{2,}')  # >=两个空格


class DataPreprocessor(object):

    def __init__(self, *, input_files: List[str], output_path: Optional[str]=None,
                 min_word_count: int=2, min_entity_count: int=1, word_embedding_dim: int=50):
        """
        原始数据处理类
        :param input_files: 输入文件地址
        :param output_path: 缺省的输出地址
        :param min_word_count: 筛除低频词的阈值
        :param min_entity_count: 筛出低频实体的阈值
        :param word_embedding_dim: word2vec的embedding维数
        """
        self.output_path = output_path if output_path else os.getcwd()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
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

    def _encoding_title(self, title: str, entities: str) -> (str, str):
        """
        根据当前的 entities，寻找 title与entities的单词关联，并返回 title的 word_encoding序列和 entity_encoding序列
        :param title: 标题 (w1, w2, ..., wN)
        :param entities: 当前标题对应的 entities信息, id_1:entity_1;id_2:entity_2...
        :return: word_encoding(w1's word_index, w2's word_index,..., wN's word_index),
                 entity_encoding(w1's entity_index, w2's entity_index,..., wN's entity_index)
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

    def transform(self, input_file: str, output_file: str):
        """
        对 input_file的标题进行 index处理，生成 word_index_encoding 和 entity_index_encoding
        并输出至output_file
        :param input_file: 输入数据地址
        :param output_file: 输出数据地址
        """
        with timer("Transform", True):
            print("**** Starting Transform %s ****" % input_file)
            with open(input_file, "r", encoding="utf-8") as fr, open(output_file, "w", encoding="utf-8") as fw:
                for line in fr:
                    line = line.strip()
                    if line:
                        user, title, label, entities = line.split("\t")
                        word_encoding, entity_encoding = self._encoding_title(title, entities)  # 从entity中获取特征
                        content = "\t".join([user, word_encoding, entity_encoding, label])
                        fw.write(content + "\n")
            print("Transformation Done!")

    def get_w2v_model(self):
        """generate or load word2vec model"""

        w2v_file = "word_embeddings_" + str(self.K) + ".model"
        output_path = os.path.join(self.output_path, w2v_file)
        if os.path.exists(output_path):
            print("**** Model already exists. Loading model now ... ****")
            self.w2v_model = gensim.models.word2vec.Word2Vec.load(output_path)
        else:
            print("**** Training word2vec model now ****")
            self.w2v_model = gensim.models.word2vec.Word2Vec(sentences=self.corpus, size=self.K, min_count=1, workers=8)
            print("**** Succeed in training. Saving model... ****")
            self.w2v_model.save(output_path)

    def save_info(self):
        """
        将 entity2index，word_embedding向量写入硬盘
        """
        # 写入 entity2index
        print("**** Saving entity2index map ****")
        with open(os.path.join(PATH, "kg/entity2index.txt"), "w", encoding="utf-8") as f:
            for ent_id, ent_index in self.entity2index.items():
                f.write("%d\t%d\n" % (ent_id, ent_index))

        # 写入wordembedding 以numpy文件写入，行标对应word2index的index
        print("**** Saving word2vec embeddings ****")
        file = "word_embeddings_" + str(self.K) + ".vec"
        embeddings = np.zeros(shape=[len(self.word2index) + 1, self.K])  # 第一行留给标号0的word(dummy word)
        for word, index in self.word2index.items():
            if word in self.w2v_model.wv.vocab:
                vec = self.w2v_model[word]
                embeddings[index] = vec
        np.save(os.path.join(self.output_path, file), embeddings)


if __name__ == '__main__':
    output_path = os.path.join(PATH, "data")
    input_files = [os.path.join(PATH, "raw_data/raw_train.txt"), os.path.join(PATH, "raw_data/raw_test.txt")]
    output_files = [os.path.join(output_path, "train.txt"), os.path.join(output_path, "test.txt")]
    preprocessor = DataPreprocessor(input_files=input_files, output_path=output_path,
                                    min_word_count=2, min_entity_count=1, word_embedding_dim=50)

    preprocessor.file_parsing()
    for input_file, output_file in zip(input_files, output_files):
        preprocessor.transform(input_file, output_file)

    preprocessor.get_w2v_model()
    preprocessor.save_info()
