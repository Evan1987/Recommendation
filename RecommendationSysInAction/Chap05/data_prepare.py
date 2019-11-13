"""Profile generation from `MovieLens` source data"""

import os
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm
from itertools import chain
from typing import Dict, List, Any


tqdm.pandas()
logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
MOVIE_LENS_SRC = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\ml-1m"


class ProfileGenerator(object):
    def __init__(self, src: str):
        if not os.path.exists(src):
            raise IOError(f"`{src}` not exists.")
        self.src = src
        self.movies_file = os.path.join(self.src, "movies.dat")
        self.users_file = os.path.join(self.src, "users.dat")
        self.ratings_file = os.path.join(self.src, "ratings.dat")

        LOGGER.info("Loading source data...")
        LOGGER.info("Reading Movies.")
        self.movies = pd.read_csv(self.movies_file, sep="::", engine="python", names=["MovieID", "Title", "Genres"])
        self.movies["Genres"] = self.movies["Genres"].str.split("|")
        self.genres: List[str] = sorted(list(set(list(chain.from_iterable(self.movies["Genres"])))))  # 18 tags
        self.genre_index: Dict[str, int] = {genre: i for i, genre in enumerate(self.genres)}
        self.movies["Genre_indexes"] = self.movies["Genres"] \
            .progress_apply(lambda genres: self._index_encoding(genres, self.genre_index))

        LOGGER.info("Reading Users and Ratings.")
        self.users = pd.read_csv(self.users_file, sep="::", engine="python",
                                 names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        self.ratings = pd.read_csv(self.ratings_file, sep="::", engine="python",
                                   names=["UserID", "MovieID", "Rating", "Timestamp"])
        LOGGER.info("Loading successfully.")

    @staticmethod
    def _index_encoding(target: List[Any], indexing: Dict[Any, int]) -> List[int]:
        """Index list of tags to multi-hot int labels"""
        res = [0] * len(indexing)
        for tag in target:
            res[indexing[tag]] = 1
        return res

    @staticmethod
    def _process_profile_by_ratings(df: pd.DataFrame) -> List[float]:
        """Generate profile by his rating history
        :param df: Rating history of single user. Should include columns: `weighted_Genre_indexes`
        :return: User favor scores on each genre.
        """
        all_ratings = np.vstack(df["weighted_Genre_indexes"])
        mask = np.ma.masked_equal(all_ratings, 0)  # Use masked-array to avoid zero-division or nan
        mean = mask.mean(axis=0)
        return mean.filled(0).tolist()

    @staticmethod
    def write(data: Dict, dst: str):
        with open(dst, "w") as fp:
            json.dump(data, fp)

    def generate_item_profile(self, output_file: str):
        LOGGER.info("Generating item profile...")
        profiles = dict(zip(self.movies["MovieID"], self.movies["Genre_indexes"]))
        LOGGER.info(f"Saving item profile into `{output_file}`")
        self.write(profiles, output_file)

    def generate_user_profile(self, output_file: str):
        LOGGER.info("Generating user profile...")
        rating = pd.merge(left=self.ratings, right=self.movies[["MovieID", "Genre_indexes"]], on="MovieID")
        rating["user_mean_rating"] = rating.groupby("UserID")["Rating"].transform("mean")  # Average rating of user
        rating["normed_Rating"] = rating["Rating"] - rating["user_mean_rating"]  # normed rating by minus avg
        rating["Genre_indexes"] = rating["Genre_indexes"].apply(np.asarray)
        rating["weighted_Genre_indexes"] = rating["normed_Rating"] * rating["Genre_indexes"]  # weight on genre indexes
        profile = rating[["UserID", "weighted_Genre_indexes"]]\
            .groupby("UserID")\
            .progress_apply(self._process_profile_by_ratings)

        profile = dict(profile)
        LOGGER.info(f"Saving user profile into `{output_file}`")
        self.write(profile, output_file)


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(path, exist_ok=True)
    item_profile = os.path.join(path, "item_profile.json")
    user_profile = os.path.join(path, "user_profile.json")

    g = ProfileGenerator(MOVIE_LENS_SRC)
    g.generate_item_profile(item_profile)
    g.generate_user_profile(user_profile)
