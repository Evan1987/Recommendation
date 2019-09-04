"""随机选取1000个用户，并根据这些用户生成训练集和测试集"""

import os
import glob
import tqdm
import json
import random
import logging
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set


Entry = namedtuple("Entry", ["user", "movie", "rate"])


class DataGenerator(object):
    def __init__(self, file_path: str, user_num: int = 1000, test_ratio: float = 0.2, seed: int = 30):
        """
        Initialize a data generator instance.
        :param file_path: The source file dir where mv_****.txt placed.
        :param user_num: How many users to select from total set.
        :param test_ratio: The ratio of test part to generate.
        :param seed: The random seed.
        """
        if not os.path.isdir(file_path):
            raise IOError(f"`{file_path}` is not valid, expect to be a dir path to hold source files.")
        if not (0 < test_ratio < 1):
            raise ValueError("Test ratio must between 0 and 1.")

        self.test_ratio = test_ratio
        self.rng = random.Random(seed)
        self.sample_users = set(self.rng.sample(self.get_total_users(file_path), user_num))
        self.file_path = file_path

    @staticmethod
    def get_total_users(file_path: str) -> Set[int]:
        """Get total users from multi files in `file_path`"""
        users = set()
        for file in tqdm.tqdm(glob.glob(os.path.join(file_path, "*.txt"))):
            with open(file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.endswith(":"):
                        continue
                    user, _, _ = line.split(",")
                    users.add(int(user))
        return users

    def _sample_data(self, file: str) -> List[Entry]:
        res = []
        with open(file, "r") as f:
            movie = int(next(f).strip().strip(":"))
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user, rate, _ = line.split(",")
                user, rate = int(user), int(rate)
                if user in self.sample_users:
                    res.append(Entry(user=user, movie=movie, rate=rate))
        return res

    def generate_data(self):
        files = glob.glob(os.path.join(self.file_path, "*.txt"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            sampled_data: List[List[Entry]] = list(tqdm.tqdm(pool.map(self._sample_data, files), total=len(files)))

        train_data, test_data = {}, {}
        for entries in sampled_data:
            for entry in entries:
                user, movie, rate = entry.user, entry.movie, entry.rate
                if self.rng.uniform(0, 1) < self.test_ratio:
                    test_data.setdefault(user, {})[movie] = rate
                else:
                    train_data.setdefault(user, {})[movie] = rate
        return train_data, test_data


def write(data: Dict, dst: str):
    with open(dst, "w") as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger("logger")
    read_path = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\netflix\training_set"
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Read source: `{read_path}`.")
    logger.info(f"Output dir: `{output_dir}`")

    logger.info("Initialize data generator...")
    data_generator = DataGenerator(read_path, user_num=1000, test_ratio=0.2, seed=30)

    logger.info("Get the sample data.")
    train, test = data_generator.generate_data()

    logger.info(f"Train set users: {len(train)}, total items: {sum([len(v) for _, v in train.items()])}")
    logger.info(f"Test set users: {len(test)}, total items: {sum([len(v) for _, v in test.items()])}")

    logger.info("Write into disk...")
    write(train, os.path.join(output_dir, "train.json"))
    write(test, os.path.join(output_dir, "test.json"))

    logger.info("Write files successfully.")
