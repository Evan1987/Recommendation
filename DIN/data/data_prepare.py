
import os
import pandas as pd
from random import Random
from tqdm import tqdm
from sklearn.utils import shuffle
from constant import PROJECT_HOME
from evan_utils.dataset import load_amazon
from typing import Dict


tqdm.pandas()
SEED = 1234
PACKAGE_HOME = os.path.join(PROJECT_HOME, "DIN")
DATA_PATH = os.path.join(PACKAGE_HOME, "data")
DATASET_FILE = os.path.join(DATA_PATH, "dataset.pkl")
CATE_MAPPING_FILE = os.path.join(DATA_PATH, "cate_mapping.txt")


class DataProcessor(object):
    meta_file = os.path.join(DATA_PATH, "meta.txt")
    reviews_file = os.path.join(DATA_PATH, "reviews.txt")

    def __init__(self):
        self.meta: pd.DataFrame = pd.read_csv(self.meta_file, sep="\t")
        self.reviews: pd.DataFrame = pd.read_csv(self.reviews_file, sep="\t")\
            .sort_values(["reviewerID", "unixReviewTime"])
        self._asin_set = set(self.reviews["asin"])
        self.meta = self.meta[self.meta["asin"].isin(self._asin_set)]
        self.mapping: Dict[str, Dict] = {}
        self.build_mapping(self.meta, "asin")
        self.build_mapping(self.meta, "categories")
        self.build_mapping(self.reviews, "reviewerID")
        self.reviews["asin"] = self.reviews["asin"].map(self.mapping["asin"])
        self._rng = Random(SEED)

    def build_mapping(self, df: pd.DataFrame, column: str) -> None:
        x = pd.Categorical(df[column], ordered=True)
        mapping = {cat: i for i, cat in enumerate(x.categories)}
        df[column] = x.codes
        self.mapping[column] = mapping

    def generate_data(self) -> pd.DataFrame:

        def group_treating(df: pd.DataFrame):
            user = df["reviewerID"].iloc[0]
            pos_items = df["asin"]
            pos_set = set(pos_items)
            # Sample the negative samples from total set
            # neg_items = self._rng.choices(list(total_asin_set - set(pos_items)), k=len(df))  # to slow
            neg_items = []
            for _ in range(len(df)):
                while True:
                    item = self._rng.randint(0, self.item_count)
                    if item not in pos_set:
                        neg_items.append(item)
                        break

            # The accumulated review history
            histories = [] + df["asin"].apply(lambda x: [x]).cumsum().tolist()
            is_train = [1] * (len(df) - 1) + [0]  # Only the last entry is the test

            pos_entries = pd.DataFrame({"histories": histories, "asin": pos_items, "label": 1, "is_train": is_train})
            neg_entries = pd.DataFrame({"histories": histories, "asin": neg_items, "label": 0, "is_train": is_train})

            total_entries = pd.concat([pos_entries, neg_entries], axis=0, ignore_index=True)
            total_entries["reviewerID"] = user

            return total_entries

        data = self.reviews.groupby("reviewerID", as_index=False).progress_apply(group_treating)
        return shuffle(data, random_state=SEED).reset_index(drop=True)

    @property
    def user_count(self):
        return len(self.mapping["reviewerID"])

    @property
    def item_count(self):
        return len(self.mapping["asin"])

    @property
    def cate_count(self):
        return len(self.mapping["categories"])


if __name__ == '__main__':

    # Save the raw data for next usage
    electronics_data = load_amazon("Electronics")
    if not os.path.exists(DataProcessor.reviews_file):
        reviews = electronics_data.reviews[["reviewerID", "asin", "unixReviewTime"]]
        reviews.to_csv(DataProcessor.reviews_file, sep="\t", index=False)

    if not os.path.exists(DataProcessor.meta_file):
        meta = electronics_data.meta[["asin", "categories"]]
        meta["categories"] = meta["categories"].apply(lambda x: x[-1][-1])
        meta.to_csv(DataProcessor.meta_file, sep="\t", index=False)

    # Generate train-test file
    data_processor = DataProcessor()
    total_data = data_processor.generate_data()
    total_data.to_pickle(DATASET_FILE)
    data_processor.meta.to_csv(CATE_MAPPING_FILE, sep="\t", index=False)
