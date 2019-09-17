"""6.3 A demo for keywords extraction by tf-idf"""

import os
import jieba
import math
import jieba.analyse
from collections import Counter, defaultdict
from _utils.nlp.tools import punctuations
from typing import List, Dict, Tuple


PUNCTUATIONS = punctuations
PATH = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\phone-title"


def read_data(file: str) -> Dict[str, str]:
    if not os.path.exists(file):
        raise IOError(f"{file} not found.")
    res: Dict[str, str] = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id, title = line.split("\t")
            res[id] = title
    return res


class TFIDF(object):
    def __init__(self, stop_words: List[str]):
        self.stop_words = set(stop_words)
        self.word_idf: Dict[str, float] = {}  # Contain idf score for each word

    @classmethod
    def from_file(cls, stop_words_file: str):
        if not os.path.exists(stop_words_file):
            raise IOError(f"{stop_words_file} not found.")

        words = []
        with open(stop_words_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words.append(line)
        return cls(words)

    @staticmethod
    def get_term_frequency(words: List[str]) -> Dict[str, int]:
        return dict(Counter(words))

    def process_content(self, content: str) -> List[str]:
        words = []
        for word in jieba.cut(content.replace(" ", "")):
            if word in self.stop_words or word in PUNCTUATIONS:
                continue
            words.append(word)
        return words

    def fit(self, data: Dict[str, str]):
        """Get the idf score for each word."""
        total_docs = len(data)
        word_in_docs: Dict[str, int] = defaultdict(int)
        for id_, title in data.items():
            words = self.process_content(title)
            for word in set(words):
                word_in_docs[word] += 1

        for word, n_doc in word_in_docs.items():
            self.word_idf[word] = math.log(total_docs / (1 + n_doc))

    def transform(self, content: str) -> List[Tuple[str, float]]:
        """Output the tf-idf score for each word"""
        words = self.process_content(content)
        word_count = self.get_term_frequency(words)
        res = [(word, count / len(words) * self.word_idf[word]) for word, count in word_count.items()]
        return sorted(res, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    tfidf = TFIDF.from_file(os.path.join(PATH, "stop_words.txt"))
    data = read_data(os.path.join(PATH, "id_title.txt"))
    tfidf.fit(data)

    content = data["5594"]
    print(tfidf.transform(content))
    print(jieba.analyse.extract_tags(content, topK=10, withWeight=True))
